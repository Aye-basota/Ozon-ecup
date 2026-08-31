"""EXP-057: target-free production-state reweighting for the historical UNC slot.

The registered CPU experiment is intentionally limited to fold 2025-10-16::

    python src/production_state_reweight.py

The command is resumable at the weight and model artifacts.  A deterministic
analysis replay never trains and never reads test/public paths::

    python src/production_state_reweight.py --analysis-only
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ARTIFACTS, LGB_PARAMS, ROOT, SEED
from src.features import feature_names, panel_users, to_np
from src.merge_oof import auc_positive
from src.tabular_backbone_refresh import (EXPECTED_H, historical_baseline, saved_features,
                                           seed_recipe, sha256_array, sha256_file)
from src.train import Setup, _XY, xy
from src.validation import calibrate, rmsle_z


EXP_NUM = 57
EXP_ID = "PRODUCTION-STATE-REWEIGHT"
PREFIX = "STATE_REWEIGHT_EXP057"
PILOT_FOLD = dt.date(2025, 10, 16)
RUN_DIR = ARTIFACTS / PREFIX
RESULTS = ROOT / "research" / "strategies" / "results" / PREFIX
WEIGHT_FILE = RUN_DIR / f"{PREFIX}_weights.npz"
PRED_FILE = RUN_DIR / f"{PREFIX}_UNC_ARMS_V1016.npz"
UNIFORM_COMPONENT = "S1-E02"
UNIFORM_WEIGHT = 0.20
ROUNDS = 600
ODDS_CLIP = (0.25, 4.0)
DOMAIN_PROB_EPS = 1e-6
DOMAIN_PARAMS = {
    "objective": "binary", "metric": "auc", "learning_rate": 0.03,
    "num_leaves": 31, "min_data_in_leaf": 2000,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
    "lambda_l2": 20.0, "max_bin": 63, "force_row_wise": True,
    "verbose": -1, "seed": SEED,
}
DOMAIN_ROUNDS = 200
BASE_COMPONENTS = ("S1-E03a", "S1-E02", "S1-DIST", "ETX-AVG3", "SEQ-AVG3")
BASE_WEIGHTS = (0.10, 0.20, 0.25, 0.225, 0.225)
EXPECTED_LATE = EXPECTED_H["fold_cal"][-1]
FORBIDDEN_FEATURE_TOKENS = ("user_id", "cutoff", "weekday", "cdow", "avail", "depth",
                            "target", "prediction", "pred_")

_WINDOW_METRICS = {
    "days_present", "days_buy", "gmv", "orders", "searches", "carts",
    "aov", "gmv_per_day", "cart2ord", "srch2cart", "buyday_rate",
}
_RECENCY = {"rec_any", "rec_search", "rec_cart", "rec_buy", "rec_cat"}
_TREND_PREFIXES = ("trend_gmv_", "trend_pres_", "trend_srch_",
                   "dlog_gmv_", "dlog_buyd_")


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(jsonable(value), ensure_ascii=False, indent=2, sort_keys=True)


def save_text_once(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite different artifact: {path}")
    else:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    return sha256_file(path)


def save_json_once(path: Path, value: Any) -> str:
    return save_text_once(path, canonical_json(value) + "\n")


def save_csv_once(path: Path, rows: list[dict[str, Any]]) -> str:
    if not rows:
        raise ValueError(f"no rows for {path}")
    fields: list[str] = []
    for row in rows:
        fields.extend(k for k in row if k not in fields)
    from io import StringIO
    buf = StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: (json.dumps(jsonable(v), ensure_ascii=False, sort_keys=True)
                            if isinstance(v, (dict, list, tuple)) else v)
                         for k, v in row.items()})
    return save_text_once(path, buf.getvalue())


def save_npz_once(path: Path, **arrays: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = np.load(path, allow_pickle=False)
        if set(old.files) != set(arrays):
            raise FileExistsError(f"artifact schema drift: {path}")
        for name, value in arrays.items():
            if not np.array_equal(old[name], value, equal_nan=True):
                raise FileExistsError(f"artifact content drift: {path}:{name}")
    else:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("wb") as fh:
            np.savez_compressed(fh, **arrays)
        tmp.replace(path)
    return sha256_file(path)


def splitmix64(user_id: np.ndarray) -> np.ndarray:
    x = np.asarray(user_id, dtype=np.uint64)
    z = x + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def unc_setup() -> Setup:
    return Setup(L=None, min_history=90, step=7, panel_blocks=3, train_blocks=1,
                 model="direct", rounds=ROUNDS, params={"seed": SEED},
                 cutoffs="all", vals=[PILOT_FOLD], norm_long=False)


def select_domain_features(all_features: list[str]) -> list[str]:
    """Fixed semantic subset; never admits identifiers or depth/cutoff proxies."""
    selected: list[str] = []
    for name in all_features:
        low = name.lower()
        if any(token in low for token in FORBIDDEN_FEATURE_TOKENS):
            continue
        m = re.fullmatch(r"w(7|14|30|60|90|180)_(.+)", name)
        if m and m.group(2) in _WINDOW_METRICS:
            selected.append(name)
            continue
        if name in _RECENCY:
            selected.append(name)
            continue
        if name.startswith(_TREND_PREFIXES):
            nums = [int(v) for v in re.findall(r"\d+", name)]
            if nums and max(nums) <= 180:
                selected.append(name)
    if not selected:
        raise AssertionError("empty domain feature subset")
    if any(name.startswith("w365_") for name in selected):
        raise AssertionError("w365 feature entered domain model")
    if any(any(token in name.lower() for token in FORBIDDEN_FEATURE_TOKENS)
           for name in selected):
        raise AssertionError("forbidden domain feature")
    return selected


def row_keys(cutoff_code: np.ndarray, user_id: np.ndarray) -> np.ndarray:
    return np.char.add(np.asarray(cutoff_code).astype("U3"),
                       np.char.add("|", np.asarray(user_id, dtype=np.int64).astype("U20")))


def _fold_artifact(name: str) -> dict[str, np.ndarray]:
    path = ARTIFACTS / f"oof_{name}.npz"
    d = np.load(path, allow_pickle=False)
    mask = np.asarray(d["cutoff"], dtype="U10") == PILOT_FOLD.isoformat()
    order = np.argsort(d["user_id"][mask], kind="stable")
    return {k: d[k][mask][order] for k in ("user_id", "cutoff", "y", "z")}


def exact_latest_baseline() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    parts = [_fold_artifact(name) for name in BASE_COMPONENTS]
    uid, y = parts[0]["user_id"].astype(np.int64), parts[0]["y"].astype(np.float64)
    for part in parts[1:]:
        if not np.array_equal(part["user_id"], uid) or not np.array_equal(part["y"], y):
            raise AssertionError("historical component alignment failed")
    z = np.average(np.vstack([p["z"].astype(np.float64) for p in parts]),
                   axis=0, weights=BASE_WEIGHTS)
    offset, score = calibrate(y, z)
    if abs(score - EXPECTED_LATE) > 5e-10:
        raise AssertionError(f"STRONGEST replay drift: {score} != {EXPECTED_LATE}")
    return uid, y, z, {"offset": offset, "rmsle_cal": score,
                       "prediction_sha256": sha256_array(z)}


def phase0_audit() -> dict[str, Any]:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    _, _, _, all_report = historical_baseline()
    uid, y, z_base, late_report = exact_latest_baseline()
    old = _fold_artifact(UNIFORM_COMPONENT)
    if not np.array_equal(old["user_id"], uid) or not np.array_equal(old["y"], y):
        raise AssertionError("UNC/base fold alignment failed")
    tbr_npz = ARTIFACTS / "TBR_EXP046" / "TBR_EXP046_UNC_S42_V1016.npz"
    tbr_model = ARTIFACTS / "TBR_EXP046" / "TBR_EXP046_UNC_S42_V1016.txt"
    fresh = np.load(tbr_npz, allow_pickle=False)
    if not np.array_equal(old["z"], fresh["z_r600"]):
        raise AssertionError("existing fresh historical UNC replay is not bitwise")
    s = unc_setup()
    Xv, _ = xy(PILOT_FOLD, s)
    feats = saved_features("UNC")
    if feature_names(Xv) != feats:
        raise AssertionError("current UNC feature order differs from historical recipe")
    import lightgbm as lgb
    booster = lgb.Booster(model_file=str(tbr_model))
    z_current = np.maximum(booster.predict(to_np(Xv, feats), num_iteration=ROUNDS), 0.0).astype(
        np.float32)
    if not np.array_equal(old["z"], z_current):
        raise AssertionError("current fold feature replay is not bitwise")
    _XY.clear()
    manifest = {
        "experiment": EXP_ID, "number": EXP_NUM, "prefix": PREFIX,
        "base_head": None,
        "phase0": {
            "status": "PASS_BITWISE", "fold": PILOT_FOLD.isoformat(),
            "unc_prediction_sha256": sha256_array(old["z"]),
            "fresh_prediction_sha256": sha256_array(z_current),
            "tbr_npz": str(tbr_npz.resolve()), "tbr_npz_sha256": sha256_file(tbr_npz),
            "tbr_model": str(tbr_model.resolve()), "tbr_model_sha256": sha256_file(tbr_model),
            "n_validation": len(uid), "n_features": len(feats),
            "feature_order_sha256": sha256_array(np.asarray(feats, dtype="U")),
        },
        "strongest_current": {
            "formula": dict(zip(BASE_COMPONENTS, BASE_WEIGHTS)),
            "fold": PILOT_FOLD.isoformat(), **late_report,
            "all_fold_scores": all_report["fold_cal"], "wcv": all_report["wcv"],
            "expected_fold_score": EXPECTED_LATE,
        },
        "guardrails": {
            "gpu_used": False, "test_or_public_paths_read": False,
            "features_only_via_build_features_cache": True,
            "validation_utility_modified": False, "config_modified": False,
        },
    }
    manifest["base_head"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()
    save_json_once(RESULTS / "phase0_audit.json", manifest)
    del Xv, booster, z_current
    gc.collect()
    return manifest


def _source_semantic_matrix(s: Setup, cuts: list[dt.date], feats: list[str]) -> dict[str, np.ndarray]:
    sizes = [panel_users(T, s.train_blocks).height for T in cuts]
    X = np.empty((sum(sizes), len(feats)), np.float32)
    uid = np.empty(sum(sizes), np.int64)
    cutoff_code = np.empty(sum(sizes), np.uint8)
    rec_buy = np.empty(sum(sizes), np.float32)
    w180_buy = np.empty(sum(sizes), np.float32)
    i = 0
    for code, (T, n) in enumerate(zip(cuts, sizes)):
        Xb, _ = xy(T, s, with_target=False, blocks=s.train_blocks)
        if Xb.height != n or not np.all(np.diff(Xb["user_id"].to_numpy()) > 0):
            raise AssertionError(f"source row/order drift at {T}")
        X[i:i + n] = to_np(Xb, feats)
        uid[i:i + n] = Xb["user_id"].to_numpy().astype(np.int64)
        cutoff_code[i:i + n] = code
        rec_buy[i:i + n] = Xb["rec_buy"].to_numpy().astype(np.float32)
        w180_buy[i:i + n] = Xb["w180_days_buy"].to_numpy().astype(np.float32)
        i += n
    return {"X": X, "user_id": uid, "cutoff_code": cutoff_code,
            "rec_buy": rec_buy, "w180_days_buy": w180_buy,
            "cutoffs": np.asarray([v.isoformat() for v in cuts], dtype="U10")}


def assemble_unc_memory_safe(cuts: list[dt.date], s: Setup,
                             feats: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Historical row order with one feature block live at a time.

    ``src.train.assemble`` keeps a small dataframe cache and ``to_np`` performs
    an unconditional dtype copy.  This equivalent UNC-only path clears the
    cache after every cutoff and requests one C-order float32 array directly,
    which is important when another registered experiment shares the machine.
    """
    sizes = [panel_users(T, s.train_blocks).height for T in cuts]
    X = np.empty((sum(sizes), len(feats)), np.float32)
    y = np.empty(sum(sizes), np.float64)
    i = 0
    for T, n in zip(cuts, sizes):
        Xb, yb = xy(T, s, blocks=s.train_blocks)
        A = Xb.select(feats).to_numpy(order="c")
        if A.dtype != np.float32 or A.shape != (n, len(feats)):
            raise AssertionError(f"memory-safe matrix drift at {T}: {A.shape}/{A.dtype}")
        X[i:i + n] = A
        y[i:i + n] = yb
        i += n
        del A, Xb, yb
        _XY.clear()
        gc.collect()
    return X, y


def crossfit_side_masks(source_uid: np.ndarray, target_uid: np.ndarray, donor_side: int
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ss = (splitmix64(source_uid) & np.uint64(1)).astype(np.int8)
    ts = (splitmix64(target_uid) & np.uint64(1)).astype(np.int8)
    donor_s, donor_t = ss == donor_side, ts == donor_side
    recipient_s, recipient_t = ~donor_s, ~donor_t
    if np.any(donor_s & recipient_s) or np.any(donor_t & recipient_t):
        raise AssertionError("domain donor/recipient overlap")
    return donor_s, donor_t, recipient_s, recipient_t


def domain_crossfit(source_X: np.ndarray, source_uid: np.ndarray,
                    target_X: np.ndarray, target_uid: np.ndarray,
                    trainer: Callable[..., tuple[Any, np.ndarray, np.ndarray]] | None = None
                    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], list[Any]]:
    """Two-sided user cross-fit.  A model never scores its own user side."""
    if trainer is None:
        trainer = _fit_domain_side
    source_p = np.empty(len(source_uid), np.float32)
    target_p = np.empty(len(target_uid), np.float32)
    audits, models = [], []
    for donor_side in (0, 1):
        ds, dtgt, rs, rtgt = crossfit_side_masks(source_uid, target_uid, donor_side)
        model, ps, pt = trainer(source_X[ds], target_X[dtgt], source_X[rs], target_X[rtgt],
                                donor_side)
        source_p[rs], target_p[rtgt] = ps, pt
        donor_users = np.unique(np.concatenate([source_uid[ds], target_uid[dtgt]]))
        recipient_users = np.unique(np.concatenate([source_uid[rs], target_uid[rtgt]]))
        overlap = np.intersect1d(donor_users, recipient_users, assume_unique=True)
        if len(overlap):
            raise AssertionError("outer user side entered its own domain fit")
        audits.append({
            "donor_side": donor_side, "recipient_side": 1 - donor_side,
            "n_source_fit": int(ds.sum()), "n_target_fit": int(dtgt.sum()),
            "n_source_score": int(rs.sum()), "n_target_score": int(rtgt.sum()),
            "user_overlap": 0,
        })
        models.append(model)
    if not np.isfinite(source_p).all() or not np.isfinite(target_p).all():
        raise AssertionError("non-finite domain probabilities")
    return source_p, target_p, audits, models


def _fit_domain_side(source_fit: np.ndarray, target_fit: np.ndarray,
                     source_score: np.ndarray, target_score: np.ndarray,
                     donor_side: int) -> tuple[Any, np.ndarray, np.ndarray]:
    import lightgbm as lgb
    ns, nt = len(source_fit), len(target_fit)
    if not ns or not nt:
        raise AssertionError("empty domain class/side")
    Xfit = np.concatenate([source_fit, target_fit], axis=0)
    label = np.concatenate([np.zeros(ns, np.int8), np.ones(nt, np.int8)])
    # Equal total class mass makes p/(1-p) the density ratio up to one constant.
    class_weight = np.concatenate([np.ones(ns, np.float32),
                                   np.full(nt, ns / nt, np.float32)])
    params = dict(LGB_PARAMS)
    params.update(DOMAIN_PARAMS)
    ds = lgb.Dataset(Xfit, label, weight=class_weight, params=params).construct()
    del Xfit, label, class_weight
    gc.collect()
    model = lgb.train(params, ds, num_boost_round=DOMAIN_ROUNDS)
    ps = model.predict(source_score).astype(np.float32)
    pt = model.predict(target_score).astype(np.float32)
    del ds
    return model, ps, pt


def normalize_state_weights(raw_odds: np.ndarray, cutoff_code: np.ndarray,
                            user_id: np.ndarray) -> tuple[np.ndarray, dict[str, Any],
                                                         np.ndarray, np.ndarray]:
    w = np.asarray(raw_odds, np.float64).copy()
    code = np.asarray(cutoff_code)
    for c in np.unique(code):
        m = code == c
        w[m] /= w[m].mean()
    users, inv = np.unique(np.asarray(user_id, np.int64), return_inverse=True)
    total_pre = np.bincount(inv, weights=w, minlength=len(users))
    median_pre = float(np.median(total_pre))
    cap = 2.0 * median_pre
    scales = np.minimum(1.0, cap / np.maximum(total_pre, 1e-12))
    w *= scales[inv]
    w /= w.mean()
    total_post = np.bincount(inv, weights=w, minlength=len(users))
    median_post = float(np.median(total_post))
    if float(total_post.max()) > 2.0 * median_post + 1e-8:
        raise AssertionError("final user contribution cap failed")
    audit = {
        "mean": float(w.mean()), "min": float(w.min()), "max": float(w.max()),
        "ess": float(w.sum() ** 2 / np.square(w).sum()),
        "ess_fraction": float(w.sum() ** 2 / np.square(w).sum() / len(w)),
        "median_user_total_before_cap": median_pre, "cap_before_global_norm": cap,
        "fraction_users_capped": float(np.mean(scales < 1.0)),
        "median_user_total_final": median_post, "max_user_total_final": float(total_post.max()),
        "max_to_median_user_total_final": float(total_post.max() / median_post),
    }
    return w.astype(np.float32), audit, users, total_post


def recency_bin(value: np.ndarray) -> np.ndarray:
    x = np.asarray(value, float)
    return np.where(np.isfinite(x) & (x <= 14), 0,
                    np.where(np.isfinite(x) & (x <= 60), 1, 2)).astype(np.uint8)


def buyday_bin(value: np.ndarray) -> np.ndarray:
    x = np.asarray(value, float)
    return np.where(np.isfinite(x) & (x <= 1), 0,
                    np.where(np.isfinite(x) & (x <= 15), 1, 2)).astype(np.uint8)


def shuffle_weights(real: np.ndarray, cutoff_code: np.ndarray, rec_buy: np.ndarray,
                    w180_buy: np.ndarray, seed: int = SEED
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    rbin, bbin = recency_bin(rec_buy), buyday_bin(w180_buy)
    strata = (np.asarray(cutoff_code, np.int32) * 9
              + rbin.astype(np.int32) * 3 + bbin.astype(np.int32))
    rng = np.random.default_rng(seed)
    out = np.empty_like(real)
    permutation = np.empty(len(real), np.int64)
    rows = []
    for key in np.unique(strata):
        idx = np.flatnonzero(strata == key)
        perm = rng.permutation(idx)
        out[idx] = real[perm]
        permutation[idx] = perm
        exact = np.array_equal(np.sort(real[idx]), np.sort(out[idx]))
        if not exact:
            raise AssertionError(f"shuffle multiset failed in stratum {key}")
        rows.append({
            "stratum": int(key), "cutoff_code": int(key // 9),
            "rec_buy_bin": int((key % 9) // 3), "w180_days_buy_bin": int(key % 3),
            "n": len(idx), "row_fraction": len(idx) / len(real),
            "weight_fraction": float(real[idx].astype(np.float64).sum()
                                     / real.astype(np.float64).sum()),
            "mean_weight": float(real[idx].mean()), "multiset_exact": exact,
            "multiset_sha256": sha256_array(np.sort(real[idx])),
        })
    if not np.array_equal(np.sort(real), np.sort(out)):
        raise AssertionError("global shuffled multiset failed")
    return out, permutation, strata, rows


def _nan_smd(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        sm, tm = np.nanmean(source, axis=0), np.nanmean(target, axis=0)
        sv, tv = np.nanvar(source, axis=0), np.nanvar(target, axis=0)
        den = np.sqrt((sv + tv) / 2.0)
        out = np.divide(sm - tm, den, out=np.zeros_like(sm), where=den > 1e-12)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def target_day_audit(cuts: list[dt.date]) -> dict[str, Any]:
    windows = [[T + dt.timedelta(days=k) for k in range(1, 31)] for T in cuts]
    all_days = [d for window in windows for d in window]
    unique, counts = np.unique(np.asarray(all_days, dtype="datetime64[D]"), return_counts=True)
    exposure_ess_days = float(counts.sum() ** 2 / np.square(counts).sum())
    adjacent = [len(set(a) & set(b)) / 30.0 for a, b in zip(windows[:-1], windows[1:])]
    return {
        "n_target_windows": len(windows), "target_days_with_multiplicity": len(all_days),
        "unique_target_days": len(unique), "mean_reuse_per_unique_day": float(counts.mean()),
        "kish_effective_target_days": exposure_ess_days,
        "effective_independent_30d_windows_from_unique_days": len(unique) / 30.0,
        "effective_independent_30d_windows_kish": exposure_ess_days / 30.0,
        "adjacent_overlap_fraction_mean": float(np.mean(adjacent)),
        "adjacent_overlap_fraction_min": float(np.min(adjacent)),
        "adjacent_overlap_fraction_max": float(np.max(adjacent)),
    }


def build_weights() -> dict[str, Any]:
    if WEIGHT_FILE.exists():
        manifest_path = RESULTS / "weights_manifest.json"
        if not manifest_path.exists():
            raise RuntimeError("weights artifact exists without manifest")
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    t0 = time.time()
    s = unc_setup()
    all_feats = saved_features("UNC")
    semantic = select_domain_features(all_feats)
    cuts = s.train_cutoffs(PILOT_FOLD)
    if len(cuts) != 24 or cuts[-1] + dt.timedelta(days=30) > PILOT_FOLD:
        raise AssertionError("historical clean train grid drift")
    source = _source_semantic_matrix(s, cuts, semantic)
    Xv, _ = xy(PILOT_FOLD, s, with_target=False, blocks=s.panel_blocks)
    target_X = to_np(Xv, semantic)
    target_uid = Xv["user_id"].to_numpy().astype(np.int64)
    p_source, p_target, side_audit, domain_models = domain_crossfit(
        source["X"], source["user_id"], target_X, target_uid)
    from sklearn.metrics import roc_auc_score
    domain_y = np.concatenate([np.zeros(len(p_source), np.int8),
                               np.ones(len(p_target), np.int8)])
    domain_p = np.concatenate([p_source, p_target])
    auc = float(roc_auc_score(domain_y, domain_p))
    p_clip = np.clip(p_source.astype(np.float64), DOMAIN_PROB_EPS, 1.0 - DOMAIN_PROB_EPS)
    raw_odds = np.clip(p_clip / (1.0 - p_clip), *ODDS_CLIP)
    real, weight_audit, users, user_total = normalize_state_weights(
        raw_odds, source["cutoff_code"], source["user_id"])
    shuffled, permutation, strata, strata_rows = shuffle_weights(
        real, source["cutoff_code"], source["rec_buy"], source["w180_days_buy"])
    if not np.array_equal(np.sort(real), np.sort(shuffled)):
        raise AssertionError("B/C total weight multiset mismatch")
    total = float(real.astype(np.float64).sum())
    tiny_dominates = any(row["row_fraction"] < 0.01 and row["weight_fraction"] > 0.05
                         for row in strata_rows)
    cutoff_rows = []
    for code, cut in enumerate(cuts):
        m = source["cutoff_code"] == code
        cutoff_rows.append({
            "cutoff": cut.isoformat(), "code": code, "n": int(m.sum()),
            "weight_sum": float(real[m].astype(np.float64).sum()),
            "weight_mean": float(real[m].mean()), "weight_max": float(real[m].max()),
            "ess_fraction": float(real[m].astype(np.float64).sum() ** 2
                                  / np.square(real[m].astype(np.float64)).sum() / m.sum()),
        })
    user_rows = [{"user_id": int(u), "rows": int(n), "weight_total": float(w)}
                 for u, n, w in zip(users, np.bincount(np.unique(source["user_id"],
                                                                  return_inverse=True)[1]),
                                    user_total)]
    counts = np.asarray([row["rows"] for row in user_rows])
    repeated_rows = [{"rows_per_user": int(v), "users": int((counts == v).sum()),
                      "fraction_users": float(np.mean(counts == v))}
                     for v in np.unique(counts)]
    distance_rows = []
    for code, cut in enumerate(cuts):
        smd = np.abs(_nan_smd(source["X"][source["cutoff_code"] == code], target_X))
        distance_rows.append({
            "cutoff": cut.isoformat(), "days_to_validation": (PILOT_FOLD - cut).days,
            "mean_abs_smd": float(np.mean(smd)), "rms_smd": float(np.sqrt(np.mean(smd ** 2))),
            "p90_abs_smd": float(np.quantile(smd, 0.9)), "max_abs_smd": float(np.max(smd)),
        })
    model_rows = []
    for side, model in enumerate(domain_models):
        model_path = RUN_DIR / f"{PREFIX}_DOMAIN_SIDE{side}.txt"
        model_hash = save_text_once(model_path, model.model_to_string(num_iteration=DOMAIN_ROUNDS))
        model_rows.append({"donor_side": side, "path": str(model_path.resolve()),
                           "sha256": model_hash})
    weights_hash = save_npz_once(
        WEIGHT_FILE, user_id=source["user_id"], cutoff_code=source["cutoff_code"],
        rec_buy=source["rec_buy"], w180_days_buy=source["w180_days_buy"],
        domain_probability=p_source, state_matched=real, shuffled=shuffled,
        user_side=(splitmix64(source["user_id"]) & np.uint64(1)).astype(np.uint8),
        stratum=strata.astype(np.int16))
    save_csv_once(RESULTS / "cutoff_weight_totals.csv", cutoff_rows)
    save_csv_once(RESULTS / "user_weight_totals.csv", user_rows)
    save_csv_once(RESULTS / "repeated_rows_per_user.csv", repeated_rows)
    save_csv_once(RESULTS / "shuffle_strata.csv", strata_rows)
    save_csv_once(RESULTS / "train_state_distance.csv", distance_rows)
    audit = {
        "status": "PASS", "feature_names": semantic, "n_features": len(semantic),
        "feature_order_sha256": sha256_array(np.asarray(semantic, dtype="U")),
        "forbidden_features_present": [], "target_used_by_domain_model": False,
        "source_rows": len(p_source), "target_rows": len(p_target),
        "source_cutoffs": [v.isoformat() for v in cuts], "adversarial_auc": auc,
        "domain_params": {**DOMAIN_PARAMS, "num_threads": LGB_PARAMS["num_threads"]},
        "domain_rounds": DOMAIN_ROUNDS, "class_prior_policy": "equal total source/target mass",
        "crossfit": side_audit, "models": model_rows,
        "odds_clip": list(ODDS_CLIP), "probability_numeric_clip": DOMAIN_PROB_EPS,
        "weights": weight_audit, "weights_artifact": str(WEIGHT_FILE.resolve()),
        "weights_sha256": weights_hash,
        "matched_shuffled_multiset_exact": True,
        "matched_sorted_sha256": sha256_array(np.sort(real)),
        "shuffled_sorted_sha256": sha256_array(np.sort(shuffled)),
        "permutation_sha256": sha256_array(permutation),
        "total_weight_matched": total,
        "total_weight_shuffled": float(shuffled.astype(np.float64).sum()),
        "tiny_weight_stratum_dominates": tiny_dominates,
        "target_day_audit": target_day_audit(cuts),
        "repeated_rows_summary": {
            "users": len(users), "min": int(counts.min()), "median": float(np.median(counts)),
            "mean": float(counts.mean()), "max": int(counts.max()),
        },
        "runtime_s": time.time() - t0, "gpu_used": False,
        "test_or_public_paths_read": False,
    }
    save_json_once(RESULTS / "weights_manifest.json", audit)
    del source, target_X, Xv, domain_models, domain_y, domain_p
    _XY.clear()
    gc.collect()
    return audit


def _arm_path(name: str) -> Path:
    return RUN_DIR / f"{PREFIX}_UNC_{name}_V1016.txt"


def _arm_pred_path(name: str) -> Path:
    return RUN_DIR / f"{PREFIX}_UNC_{name}_V1016.npz"


def _train_arm(name: str, box: list[np.ndarray | None], ytr: np.ndarray,
               weight: np.ndarray, s: Setup, feats: list[str]
               ) -> tuple[np.ndarray, dict[str, Any]]:
    import lightgbm as lgb
    path = _arm_path(name)
    manifest_path = path.with_suffix(".json")
    if path.exists() or manifest_path.exists():
        raise RuntimeError(f"partial arm artifact without completed prediction: {name}")
    params = seed_recipe(SEED)
    Xtr = box[0]
    if Xtr is None:
        raise AssertionError("training matrix already released")
    n_train, n_features = len(ytr), Xtr.shape[1]
    ds = lgb.Dataset(Xtr, np.log1p(ytr), weight=weight, params=params).construct()
    box[0] = None
    del Xtr
    _XY.clear()
    gc.collect()
    model = lgb.train(params, ds, num_boost_round=ROUNDS)
    Xv, _ = xy(PILOT_FOLD, s)
    if feature_names(Xv) != feats:
        raise AssertionError("validation feature order drift")
    if not np.array_equal(Xv["user_id"].to_numpy(), _fold_artifact(UNIFORM_COMPONENT)["user_id"]):
        raise AssertionError("validation row order differs from historical UNC")
    Av = to_np(Xv, feats)
    z = np.maximum(model.predict(Av, num_iteration=ROUNDS), 0.0).astype(np.float32)
    model_text = model.model_to_string(num_iteration=ROUNDS)
    replay = lgb.Booster(model_str=model_text)
    z_replay = np.maximum(replay.predict(Av, num_iteration=ROUNDS), 0.0).astype(np.float32)
    if not np.array_equal(z, z_replay):
        raise AssertionError(f"serialized model replay failed: {name}")
    model_hash = save_text_once(path, model_text)
    meta = {
        "arm": name, "params": params, "rounds": ROUNDS, "seed": SEED,
        "n_train": n_train, "n_features": n_features, "weight_sum": float(weight.sum()),
        "weight_mean": float(weight.mean()), "weight_sha256": sha256_array(weight),
        "prediction_sha256": sha256_array(z), "model_sha256": model_hash,
        "rows_and_matrix": "same materialized float32 Xtr for SHUFFLED and STATE_MATCH",
        "early_stopping": False, "gpu_used": False,
    }
    save_json_once(manifest_path, meta)
    del model, replay, z_replay, ds, Av, Xv
    _XY.clear()
    gc.collect()
    return z, meta


def train_one_arm(name: str) -> dict[str, Any]:
    if name not in ("SHUFFLED", "STATE_MATCH"):
        raise ValueError(name)
    completed = _arm_pred_path(name)
    result_manifest = RESULTS / f"arm_{name.lower()}_manifest.json"
    if completed.exists():
        if not result_manifest.exists():
            raise RuntimeError(f"completed arm lacks result manifest: {name}")
        return json.loads(result_manifest.read_text(encoding="utf-8"))
    weights = np.load(WEIGHT_FILE, allow_pickle=False)
    weight_key = "shuffled" if name == "SHUFFLED" else "state_matched"
    s = unc_setup()
    feats = saved_features("UNC")
    cuts = s.train_cutoffs(PILOT_FOLD)
    Xtr, ytr = assemble_unc_memory_safe(cuts, s, feats)
    if Xtr.dtype != np.float32 or Xtr.shape != (len(weights["user_id"]), len(feats)):
        raise AssertionError("training matrix shape/dtype drift")
    target_hash = sha256_array(np.asarray(ytr))
    tbr_meta = json.loads((ARTIFACTS / "TBR_EXP046" /
                           "TBR_EXP046_UNC_S42_V1016.json").read_text(encoding="utf-8"))
    if target_hash != tbr_meta["train_target_sha256"]:
        raise AssertionError("training target/order differs from exact historical replay")
    old = _fold_artifact(UNIFORM_COMPONENT)
    _XY.clear()
    gc.collect()
    box: list[np.ndarray | None] = [Xtr]
    del Xtr
    z, meta = _train_arm(name, box, ytr, weights[weight_key].astype(np.float32), s, feats)
    if box[0] is not None:
        raise AssertionError("training matrix was not released after Dataset construction")
    pred_path = _arm_pred_path(name)
    pred_hash = save_npz_once(pred_path, user_id=old["user_id"].astype(np.int64),
                              y=old["y"].astype(np.float32), z=z)
    meta = {**meta, "prediction_artifact": str(pred_path.resolve()),
            "prediction_artifact_sha256": pred_hash,
            "train_target_sha256": target_hash,
            "row_keys_sha256": sha256_array(row_keys(weights["cutoff_code"],
                                                        weights["user_id"])),
            "feature_order_sha256": sha256_array(np.asarray(feats, dtype="U"))}
    save_json_once(result_manifest, meta)
    del ytr
    _XY.clear()
    gc.collect()
    return meta


def _launch_arm(name: str) -> None:
    pred_path = _arm_pred_path(name)
    if pred_path.exists():
        return
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = str(SEED)
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--train-one-arm", name],
                   cwd=ROOT, env=env, check=True)


def train_unc_arms() -> dict[str, Any]:
    if PRED_FILE.exists():
        manifest_path = RESULTS / "arms_manifest.json"
        if not manifest_path.exists():
            raise RuntimeError("prediction artifact exists without arm manifest")
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    t0 = time.time()
    weights = np.load(WEIGHT_FILE, allow_pickle=False)
    if not np.array_equal(np.sort(weights["state_matched"]), np.sort(weights["shuffled"])):
        raise AssertionError("weight multiset drift before arm training")
    for name in ("SHUFFLED", "STATE_MATCH"):
        _launch_arm(name)
    shuf = np.load(_arm_pred_path("SHUFFLED"), allow_pickle=False)
    match = np.load(_arm_pred_path("STATE_MATCH"), allow_pickle=False)
    old = _fold_artifact(UNIFORM_COMPONENT)
    for part in (shuf, match):
        if not np.array_equal(part["user_id"], old["user_id"]) or not np.array_equal(
                part["y"], old["y"]):
            raise AssertionError("arm validation alignment drift")
    z_uniform = old["z"].astype(np.float32)
    z_shuffled, z_matched = shuf["z"], match["z"]
    shuf_meta = json.loads((RESULTS / "arm_shuffled_manifest.json").read_text("utf-8"))
    match_meta = json.loads((RESULTS / "arm_state_match_manifest.json").read_text("utf-8"))
    params_equal = shuf_meta["params"] == match_meta["params"]
    same_rows_configs = (params_equal and shuf_meta["rounds"] == match_meta["rounds"]
                         and shuf_meta["seed"] == match_meta["seed"]
                         and shuf_meta["n_train"] == match_meta["n_train"]
                         and shuf_meta["n_features"] == match_meta["n_features"])
    if not same_rows_configs:
        raise AssertionError("B/C rows or configs differ")
    pred_hash = save_npz_once(PRED_FILE, user_id=old["user_id"].astype(np.int64),
                              y=old["y"].astype(np.float32),
                              uniform=z_uniform, shuffled=z_shuffled, matched=z_matched)
    manifest = {
        "status": "COMPLETE", "fold": PILOT_FOLD.isoformat(),
        "prediction_artifact": str(PRED_FILE.resolve()), "prediction_sha256": pred_hash,
        "uniform": {
            "source": str((ARTIFACTS / f"oof_{UNIFORM_COMPONENT}.npz").resolve()),
            "prediction_sha256": sha256_array(z_uniform), "bitwise_replay": True,
        },
        "shuffled": shuf_meta, "state_matched": match_meta,
        "same_rows_order_matrix_params_seed_rounds_threads": same_rows_configs,
        "same_materialized_Xtr": True,
        "weight_multiset_exact": True,
        "train_target_sha256": shuf_meta["train_target_sha256"],
        "row_keys_sha256": sha256_array(row_keys(weights["cutoff_code"], weights["user_id"])),
        "feature_order_sha256": shuf_meta["feature_order_sha256"],
        "runtime_s": time.time() - t0, "gpu_used": False,
        "test_or_public_paths_read": False,
    }
    save_json_once(RESULTS / "arms_manifest.json", manifest)
    return manifest


def finite_corr(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.ptp(x[m]) == 0 or np.ptp(y[m]) == 0:
        return float("nan")
    return float(np.corrcoef(x[m], y[m])[0, 1])


def predictor_metrics(y: np.ndarray, z: np.ndarray) -> dict[str, Any]:
    off, score = calibrate(y, z)
    cal = np.asarray(z, float) + off
    pos, zero = y > 0, y == 0
    return {
        "offset": off, "rmsle_cal": score, "auc_positive": auc_positive(y, z),
        "rmsle_positive_fixed_final_cal": rmsle_z(y[pos], cal[pos]),
        "rmsle_zero_fixed_final_cal": rmsle_z(y[zero], cal[zero]),
        "mean_z_raw": float(np.mean(z)), "prediction_sha256": sha256_array(np.asarray(z)),
        "calibration_after_final_assembly": True,
    }


def _slice_masks(rec_buy: np.ndarray, w180: np.ndarray) -> dict[str, np.ndarray]:
    rec, buy = np.asarray(rec_buy, float), np.asarray(w180, float)
    return {
        "all": np.ones(len(rec), bool),
        "rec_buy_15_60": np.isfinite(rec) & (rec >= 15) & (rec <= 60),
        "w180_days_buy_2_15": np.isfinite(buy) & (buy >= 2) & (buy <= 15),
        "history_poor_w180_0_1": ~np.isfinite(buy) | (buy <= 1),
        "frequent_w180_16plus": np.isfinite(buy) & (buy >= 16),
    }


def _fixed_cal_score(y: np.ndarray, z: np.ndarray, offset: float, mask: np.ndarray) -> float:
    return rmsle_z(y[mask], np.asarray(z, float)[mask] + offset)


def _analysis_core() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    pred = np.load(PRED_FILE, allow_pickle=False)
    weights = np.load(WEIGHT_FILE, allow_pickle=False)
    uid, y = pred["user_id"].astype(np.int64), pred["y"].astype(np.float64)
    base_uid, base_y, z_base, base_replay = exact_latest_baseline()
    if not np.array_equal(uid, base_uid) or not np.array_equal(y, base_y):
        raise AssertionError("arm/base alignment drift")
    z_unc = pred["uniform"].astype(np.float64)
    z_shuf = pred["shuffled"].astype(np.float64)
    z_match = pred["matched"].astype(np.float64)
    z_shuf_slot = z_base + UNIFORM_WEIGHT * (z_shuf - z_unc)
    z_match_slot = z_base + UNIFORM_WEIGHT * (z_match - z_unc)
    predictors = {
        "UNC_UNIFORM": z_unc, "UNC_SHUFFLED": z_shuf, "UNC_STATE_MATCH": z_match,
        "STRONGEST_CURRENT": z_base, "SHUFFLED_SLOT": z_shuf_slot,
        "MATCHED_SLOT": z_match_slot,
    }
    metrics = {name: predictor_metrics(y, z) for name, z in predictors.items()}
    metrics["STRONGEST_CURRENT"]["exact_replay"] = base_replay
    contrasts = {}
    for name, candidate, reference in (
        ("UNC_MATCHED_MINUS_UNIFORM", "UNC_STATE_MATCH", "UNC_UNIFORM"),
        ("UNC_SHUFFLED_MINUS_UNIFORM", "UNC_SHUFFLED", "UNC_UNIFORM"),
        ("UNC_MATCHED_MINUS_SHUFFLED", "UNC_STATE_MATCH", "UNC_SHUFFLED"),
        ("MATCHED_SLOT_MINUS_SHUFFLED_SLOT", "MATCHED_SLOT", "SHUFFLED_SLOT"),
        ("MATCHED_SLOT_MINUS_STRONGEST", "MATCHED_SLOT", "STRONGEST_CURRENT"),
        ("SHUFFLED_SLOT_MINUS_STRONGEST", "SHUFFLED_SLOT", "STRONGEST_CURRENT"),
    ):
        dz = predictors[candidate] - predictors[reference]
        residual = np.log1p(y) - (predictors[reference] + metrics[reference]["offset"])
        contrasts[name] = {
            "delta_rmsle_cal": metrics[candidate]["rmsle_cal"] - metrics[reference]["rmsle_cal"],
            "delta_auc_positive": metrics[candidate]["auc_positive"] - metrics[reference]["auc_positive"],
            "var_delta_z": float(np.var(dz)), "mean_delta_z": float(np.mean(dz)),
            "max_abs_delta_z": float(np.max(np.abs(dz))),
            "residual_alignment": finite_corr(dz, residual),
            "positive_alignment": bool(finite_corr(dz, residual) > 0),
        }
    Xv, _ = xy(PILOT_FOLD, unc_setup(), with_target=False)
    if not np.array_equal(Xv["user_id"].to_numpy(), uid):
        raise AssertionError("slice feature alignment drift")
    rec_buy = Xv["rec_buy"].to_numpy()
    w180 = Xv["w180_days_buy"].to_numpy()
    side = (splitmix64(uid) & np.uint64(1)).astype(np.uint8)
    masks = _slice_masks(rec_buy, w180)
    masks.update({"user_half_A": side == 0, "user_half_B": side == 1})
    segment_rows = []
    for name, mask in masks.items():
        if mask.sum() < 2:
            continue
        row: dict[str, Any] = {"segment": name, "n": int(mask.sum())}
        for predictor in ("STRONGEST_CURRENT", "SHUFFLED_SLOT", "MATCHED_SLOT"):
            row[f"{predictor}_fixed_final_cal"] = _fixed_cal_score(
                y, predictors[predictor], metrics[predictor]["offset"], mask)
        row["matched_minus_shuffled"] = (row["MATCHED_SLOT_fixed_final_cal"]
                                           - row["SHUFFLED_SLOT_fixed_final_cal"])
        row["matched_minus_strongest"] = (row["MATCHED_SLOT_fixed_final_cal"]
                                            - row["STRONGEST_CURRENT_fixed_final_cal"])
        segment_rows.append(row)
    half = {row["segment"]: row for row in segment_rows if row["segment"].startswith("user_half")}
    both_halves = (all(half[h]["matched_minus_shuffled"] < 0 for h in half)
                   and all(half[h]["matched_minus_strongest"] < 0 for h in half)
                   and len(half) == 2)
    weights_manifest = json.loads((RESULTS / "weights_manifest.json").read_text(encoding="utf-8"))
    causal = contrasts["MATCHED_SLOT_MINUS_SHUFFLED_SLOT"]
    prod = contrasts["MATCHED_SLOT_MINUS_STRONGEST"]
    ess_ok = weights_manifest["weights"]["ess_fraction"] >= 0.35
    strata_ok = not weights_manifest["tiny_weight_stratum_dominates"]
    shape_ok = causal["delta_rmsle_cal"] < 0 and causal["positive_alignment"]
    if (causal["delta_rmsle_cal"] <= -0.0007 and prod["delta_rmsle_cal"] <= -0.0007
            and both_halves and shape_ok and ess_ok and strata_ok):
        verdict = "STRONG_PASS"
    elif (causal["delta_rmsle_cal"] <= -0.0005 and prod["delta_rmsle_cal"] < 0
          and both_halves and causal["positive_alignment"] and ess_ok and strata_ok):
        verdict = "PASS_TO_FULL_FOLDS"
    elif -0.0005 < causal["delta_rmsle_cal"] <= -0.0003 and ess_ok and strata_ok:
        verdict = "BORDERLINE_STOP"
    else:
        verdict = "REJECT"
    summary = {
        "experiment": EXP_ID, "number": EXP_NUM, "prefix": PREFIX,
        "fold": PILOT_FOLD.isoformat(), "predictor_metrics": metrics,
        "contrasts": contrasts, "both_user_halves_improve": both_halves,
        "ten_sixteen_improvement_not_just_level": shape_ok,
        "weight_support": {
            "ess_fraction": weights_manifest["weights"]["ess_fraction"],
            "max_weight": weights_manifest["weights"]["max"],
            "tiny_weight_stratum_dominates": weights_manifest["tiny_weight_stratum_dominates"],
        },
        "decision": verdict,
        "full_folds_run": False, "cap_dist_weighting_run": False,
        "test_inference_run": False, "submission_created": False, "gpu_used": False,
        "calibration_after_final_ensemble": True,
    }
    prediction_rows = [{"predictor": name, **metric} for name, metric in metrics.items()]
    del Xv
    _XY.clear()
    return summary, segment_rows, prediction_rows


def analyze() -> dict[str, Any]:
    summary, segment_rows, prediction_rows = _analysis_core()
    core = {
        "summary": summary, "segments": segment_rows,
        "predictions_sha256": sha256_file(PRED_FILE),
        "weights_sha256": sha256_file(WEIGHT_FILE),
    }
    replay_hash = hashlib.sha256(canonical_json(core).encode("utf-8")).hexdigest()
    summary["analysis_replay_sha256"] = replay_hash
    save_csv_once(RESULTS / "prediction_metrics.csv", prediction_rows)
    save_csv_once(RESULTS / "segment_metrics.csv", segment_rows)
    contrast_rows = [{"contrast": name, **value} for name, value in summary["contrasts"].items()]
    save_csv_once(RESULTS / "contrasts.csv", contrast_rows)
    save_json_once(RESULTS / "summary.json", summary)
    save_json_once(RESULTS / "analysis_replay.json", {
        "status": "PASS", "sha256": replay_hash,
        "prediction_artifact_sha256": sha256_file(PRED_FILE),
        "weights_artifact_sha256": sha256_file(WEIGHT_FILE),
        "analysis_only": True, "test_or_public_paths_read": False,
    })
    return summary


def run() -> dict[str, Any]:
    phase0_audit()
    build_weights()
    print("weights built", flush=True)
    train_unc_arms()
    print("UNC arms complete", flush=True)
    summary = analyze()
    print("analysis complete", flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-only", action="store_true")
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--weights-only", action="store_true")
    ap.add_argument("--train-one-arm", choices=("SHUFFLED", "STATE_MATCH"))
    args = ap.parse_args()
    if args.train_one_arm:
        train_one_arm(args.train_one_arm)
        return
    if args.analysis_only:
        analyze()
        print("analysis complete", flush=True)
        return
    phase0_audit()
    if args.audit_only:
        print("audit complete", flush=True)
        return
    build_weights()
    print("weights built", flush=True)
    if args.weights_only:
        return
    train_unc_arms()
    print("UNC arms complete", flush=True)
    analyze()
    print("analysis complete", flush=True)


if __name__ == "__main__":
    main()
