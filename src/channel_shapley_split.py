"""EXP-052: audit-gated Search/Catalog Shapley decomposition.

The default command performs the exact STRONGEST_CURRENT reconstruction, the
raw-data/channel audit, and the residual pre-flight.  The one-fold four-head
pilot is reached only when every registered pre-flight gate passes::

    python src/channel_shapley_split.py

Completed artifacts can be checked without touching raw data or training::

    python src/channel_shapley_split.py --analysis-only

No test inference, submission path, public-LB input, feature addition, tuning,
or full-fold training exists in this runner.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.blend import aligned
from src.config import (ARTIFACTS, FOLD_WEIGHTS_S1, LGB_PARAMS, ROOT, SEED,
                        VAL_FOLDS_S1)
from src.data import load
from src.features import feature_names, features_cached, panel_users, target, to_np
from src.report import evaluate
from src.tabular_backbone_refresh import sha256_array, sha256_file
from src.train import Setup, _XY, assemble, xy
from src.validation import calibrate, rmsle_z


EXP_NUM = 52
EXP_ID = "CHANNEL-SHAPLEY-SPLIT"
PREFIX = "CHANNEL_SHAPLEY_EXP052"
RUN_DIR = ARTIFACTS / PREFIX
RESULTS = ROOT / "research" / "strategies" / "results" / "CHANNEL_SHAPLEY_SPLIT"
FOLDS = tuple(VAL_FOLDS_S1)
FOLD_LABELS = tuple(v.isoformat() for v in FOLDS)
FOLD_WEIGHT = dict(zip(FOLD_LABELS, map(float, FOLD_WEIGHTS_S1)))
BASE_COMPONENTS = ("S1-E03a", "S1-E02", "S1-DIST", "ETX-AVG3", "SEQ-AVG3")
BASE_WEIGHTS = (0.10, 0.20, 0.25, 0.225, 0.225)
EXPECTED_BASE = (1.766883357, 1.760509577, 1.748629224, 1.741278566)
EXPECTED_WCV = 1.747509863
FLOAT_ATOL = 1e-6
FLOAT_RTOL = 1e-5
SHRINKAGE = 20_000.0
ALPHAS = (0.0, 0.25, 0.50, 1.00)
PILOT_FOLD = dt.date(2025, 10, 16)
PILOT_ROUNDS = 300
FUTURE_CLEAN_END = dt.date(2025, 11, 15)
REGIMES = ("neither", "search_only", "catalog_only", "both")
DOMINANT = ("no_purchase", "catalog_heavy", "mixed", "search_heavy")


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
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(jsonable(value), ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def save_json_once(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(jsonable(value), ensure_ascii=False, indent=2, sort_keys=True,
                      allow_nan=False) + "\n"
    if path.exists():
        assert path.read_text(encoding="utf-8") == text, f"refusing to overwrite changed {path}"
    else:
        path.write_text(text, encoding="utf-8")
    return sha256_file(path)


def save_csv_once(path: Path, rows: list[dict[str, Any]]) -> str:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        fields.extend(k for k in row if k not in fields)
    normalized = []
    for row in rows:
        normalized.append({k: (json.dumps(jsonable(v), ensure_ascii=False,
                                             sort_keys=True, allow_nan=False)
                                      if isinstance(v, (dict, list, tuple)) else
                                      ("" if isinstance(v, float) and not math.isfinite(v) else v))
                           for k, v in row.items()})
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as fh:
            old = list(csv.DictReader(fh))
        new = [{k: str(v) for k, v in row.items()} for row in normalized]
        assert old == new, f"refusing to overwrite changed {path}"
    else:
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(normalized)
    return sha256_file(path)


def save_npz_once(path: Path, **arrays: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = np.load(path, allow_pickle=False)
        assert set(old.files) == set(arrays), f"schema drift: {path}"
        for name, value in arrays.items():
            assert np.array_equal(old[name], value, equal_nan=True), f"content drift: {path}:{name}"
    else:
        np.savez_compressed(path, **arrays)
    return sha256_file(path)


def save_text_once(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_text(encoding="utf-8") == text, f"refusing to overwrite {path}"
    else:
        path.write_text(text, encoding="utf-8")
    return sha256_file(path)


def row_keys(cutoff: np.ndarray, user_id: np.ndarray) -> np.ndarray:
    return np.char.add(np.char.add(np.asarray(cutoff, dtype="U10"), "|"),
                       np.asarray(user_id, dtype=np.int64).astype("U20"))


def finite_corr(x: np.ndarray, y: np.ndarray, method: str = "pearson",
                weight: np.ndarray | None = None) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    if weight is not None:
        weight = np.asarray(weight, float)
        keep &= np.isfinite(weight) & (weight > 0)
    x, y = x[keep], y[keep]
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    if method == "spearman":
        x, y = rankdata(x, method="average"), rankdata(y, method="average")
    if weight is None:
        return float(np.corrcoef(x, y)[0, 1])
    w = weight[keep]
    w = w / w.sum()
    xm, ym = float(np.dot(w, x)), float(np.dot(w, y))
    xv, yv = float(np.dot(w, (x - xm) ** 2)), float(np.dot(w, (y - ym) ** 2))
    if xv <= 0 or yv <= 0:
        return float("nan")
    return float(np.dot(w, (x - xm) * (y - ym)) / math.sqrt(xv * yv))


def wcv_row_weights(cutoff: np.ndarray) -> np.ndarray:
    cut = np.asarray(cutoff, dtype="U10")
    out = np.zeros(len(cut), np.float64)
    for fold, fw in FOLD_WEIGHT.items():
        m = cut == fold
        assert m.any()
        out[m] = fw / m.sum()
    return out


def exact_baseline() -> tuple[pd.DataFrame, dict[str, Any]]:
    Z, y, cut = aligned(list(BASE_COMPONENTS))
    z = np.average(Z.astype(np.float64), axis=0, weights=BASE_WEIGHTS)
    report = evaluate(y, z, cut)
    assert np.allclose(report["fold_cal"], EXPECTED_BASE, rtol=0, atol=5e-10)
    assert abs(report["wcv"] - EXPECTED_WCV) <= 5e-10

    source = []
    canonical_uid = None
    canonical_keys = None
    for name in BASE_COMPONENTS:
        path = ARTIFACTS / f"oof_{name}.npz"
        d = np.load(path, allow_pickle=False)
        key = row_keys(d["cutoff"], d["user_id"])
        order = np.argsort(key, kind="stable")
        key = key[order]
        assert len(key) == len(np.unique(key))
        if canonical_keys is None:
            canonical_keys = key
            canonical_uid = d["user_id"][order].astype(np.int64)
            assert np.allclose(d["y"][order], y, rtol=FLOAT_RTOL, atol=FLOAT_ATOL)
        else:
            assert np.array_equal(key, canonical_keys)
            assert np.allclose(d["y"][order], y, rtol=FLOAT_RTOL, atol=FLOAT_ATOL)
        source.append({
            "component": name,
            "path": str(path.resolve()),
            "file_sha256": sha256_file(path),
            "prediction_sha256": sha256_array(d["z"]),
            "row_keys_sha256": sha256_array(key),
            "target_sha256": sha256_array(d["y"][order]),
            "n": len(key),
            "dtype": str(d["z"].dtype),
        })
    assert canonical_uid is not None and canonical_keys is not None
    frame = pd.DataFrame({"cutoff": np.asarray(cut, dtype="U10"),
                          "user_id": canonical_uid, "y": np.asarray(y, np.float64),
                          "z_base_raw": z})
    frame["z_base_cal"] = frame["z_base_raw"]
    offsets = {}
    for fold in FOLD_LABELS:
        m = frame["cutoff"].to_numpy() == fold
        off, score = calibrate(frame.loc[m, "y"], frame.loc[m, "z_base_raw"])
        offsets[fold] = {"offset": off, "score": score, "n": int(m.sum())}
        frame.loc[m, "z_base_cal"] += off
    frame["residual"] = np.log1p(frame["y"].to_numpy()) - frame["z_base_cal"].to_numpy()
    manifest = {
        "experiment": EXP_ID,
        "prefix": PREFIX,
        "formula": dict(zip(BASE_COMPONENTS, BASE_WEIGHTS)),
        "weight_sum": sum(BASE_WEIGHTS),
        "components": source,
        "folds": list(FOLD_LABELS),
        "fold_sizes": report["fold_sizes"],
        "fold_scores_calibrated": report["fold_cal"],
        "wcv": report["wcv"],
        "fold_calibration": offsets,
        "row_keys_sha256": sha256_array(canonical_keys),
        "target_sha256": sha256_array(np.asarray(y)),
        "prediction_sha256": sha256_array(z),
        "log_space_assembly": True,
        "status": "PASS_EXACT",
        "tolerance": 5e-10,
        "test_prediction_or_submission_paths_accessed": False,
    }
    return frame, manifest


def raw_daily_audit(df: pl.DataFrame) -> dict[str, Any]:
    discrepancy = (pl.col("gmv_search") + pl.col("gmv_cat") - pl.col("gmv")).abs()
    out = df.select([
        pl.len().alias("n"), pl.col("user_id").n_unique().alias("n_users"),
        pl.col("event_date").min().alias("date_min"), pl.col("event_date").max().alias("date_max"),
        (pl.col("gmv") < 0).sum().alias("negative_gmv"),
        (pl.col("gmv_search") < 0).sum().alias("negative_search"),
        (pl.col("gmv_cat") < 0).sum().alias("negative_catalog"),
        (discrepancy > FLOAT_ATOL).sum().alias("identity_failures"),
        discrepancy.max().alias("max_abs_discrepancy"),
        discrepancy.mean().alias("mean_abs_discrepancy"),
    ]).row(0, named=True)
    n_unique_days = int(df.select(pl.struct(["user_id", "event_date"]).n_unique()).item())
    dup = int(out["n"]) - n_unique_days
    out["duplicate_user_days"] = dup
    out["source_path"] = str((ROOT / "data" / "raw" / "train.parquet").resolve())
    out["source_sha256"] = sha256_file(ROOT / "data" / "raw" / "train.parquet")
    out["tolerance"] = {"atol": FLOAT_ATOL, "rtol": 0.0}
    out["status"] = "PASS" if not any(int(out[k]) for k in
        ("negative_gmv", "negative_search", "negative_catalog", "identity_failures",
         "duplicate_user_days")) else "FAIL"
    assert out["status"] == "PASS"
    return out


def _left_join_numpy(users: np.ndarray, aggregate: pl.DataFrame,
                     columns: list[str], fill_zero: bool = True) -> dict[str, np.ndarray]:
    left = pl.DataFrame({"user_id": users, "__order": np.arange(len(users), dtype=np.int64)})
    got = left.join(aggregate, on="user_id", how="left").sort("__order")
    if fill_zero:
        got = got.with_columns([pl.col(c).fill_null(0.0) for c in columns])
    return {c: got[c].to_numpy() for c in columns}


def channel_target(df: pl.DataFrame, cutoff: dt.date, users: np.ndarray,
                   horizon: int = 30) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    start, end = cutoff + dt.timedelta(days=1), cutoff + dt.timedelta(days=horizon)
    src = df.filter((pl.col("event_date") >= start) & (pl.col("event_date") <= end))
    agg = src.group_by("user_id").agg([
        pl.col("gmv_search").sum().alias("S"), pl.col("gmv_cat").sum().alias("C"),
        pl.col("gmv").sum().alias("Y")])
    out = _left_join_numpy(np.asarray(users, dtype=np.int64), agg, ["S", "C", "Y"])
    for k in out:
        out[k] = np.asarray(out[k], np.float64)
    disc = np.abs(out["S"] + out["C"] - out["Y"])
    assert np.all(out["S"] >= 0) and np.all(out["C"] >= 0) and np.all(out["Y"] >= 0)
    assert float(disc.max(initial=0.0)) <= FLOAT_ATOL
    meta = {"cutoff": cutoff.isoformat(), "window_start": start.isoformat(),
            "window_end": end.isoformat(), "source_min": start.isoformat(),
            "source_max": end.isoformat(), "n_users": len(users), "n_source_rows": src.height,
            "max_abs_channel_total_discrepancy": float(disc.max(initial=0.0)),
            "mean_abs_channel_total_discrepancy": float(disc.mean()),
            "no_dates_at_or_before_cutoff": True, "no_dates_after_horizon": True}
    return out, meta


def pilot_setup() -> Setup:
    return Setup(L=None, min_history=90, step=7, panel_blocks=3, train_blocks=1,
                 model="direct", rounds=PILOT_ROUNDS, params={"seed": SEED},
                 cutoffs="all", vals=[PILOT_FOLD], norm_long=True)


def target_audit(df: pl.DataFrame, frame: pd.DataFrame) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    s = pilot_setup()
    train_cutoffs = s.train_cutoffs(PILOT_FOLD)
    assert len(train_cutoffs) == 24
    train_parts: dict[str, list[np.ndarray]] = {k: [] for k in ("user_id", "cutoff_code", "S", "C", "Y")}
    rows: list[dict[str, Any]] = []
    for i, cutoff in enumerate(train_cutoffs):
        users_df = panel_users(cutoff, 1)
        users = users_df["user_id"].to_numpy().astype(np.int64)
        ch, meta = channel_target(df, cutoff, users)
        current = target(cutoff, users_df)["y"].to_numpy().astype(np.float64)
        assert np.allclose(ch["Y"], current, rtol=FLOAT_RTOL, atol=FLOAT_ATOL)
        assert np.allclose(np.log1p(ch["Y"]), np.log1p(current),
                           rtol=FLOAT_RTOL, atol=FLOAT_ATOL)
        assert cutoff + dt.timedelta(days=30) <= PILOT_FOLD
        train_parts["user_id"].append(users)
        train_parts["cutoff_code"].append(np.full(len(users), i, np.uint8))
        for name in ("S", "C", "Y"):
            train_parts[name].append(ch[name])
        rows.append({**meta, "role": "train", "panel_blocks": 1,
                     "current_target_match": True,
                     "validation_boundary": PILOT_FOLD.isoformat(),
                     "no_rows_after_validation_boundary": meta["window_end"] <= PILOT_FOLD.isoformat(),
                     "row_keys_sha256": sha256_array(row_keys(
                         np.full(len(users), cutoff.isoformat(), dtype="U10"), users)),
                     "channel_target_sha256": sha256_array(np.column_stack(
                         [ch["S"], ch["C"], ch["Y"]]))})
    train = {k: np.concatenate(v) for k, v in train_parts.items()}
    assert len(np.unique(row_keys(np.asarray([train_cutoffs[i].isoformat()
                                              for i in train["cutoff_code"]], dtype="U10"),
                                  train["user_id"]))) == len(train["user_id"])

    val_S, val_C, val_Y = np.empty(len(frame)), np.empty(len(frame)), np.empty(len(frame))
    for cutoff in FOLDS:
        mask = frame["cutoff"].to_numpy() == cutoff.isoformat()
        users = frame.loc[mask, "user_id"].to_numpy(np.int64)
        panel = panel_users(cutoff, 3)["user_id"].to_numpy().astype(np.int64)
        assert np.array_equal(np.sort(users), panel)
        ch, meta = channel_target(df, cutoff, users)
        val_S[mask], val_C[mask], val_Y[mask] = ch["S"], ch["C"], ch["Y"]
        assert np.allclose(ch["Y"], frame.loc[mask, "y"], rtol=FLOAT_RTOL, atol=FLOAT_ATOL)
        assert np.allclose(np.log1p(ch["Y"]), np.log1p(frame.loc[mask, "y"]),
                           rtol=FLOAT_RTOL, atol=FLOAT_ATOL)
        rows.append({**meta, "role": "validation", "panel_blocks": 3,
                     "current_target_match": True,
                     "clean_global_boundary": FUTURE_CLEAN_END.isoformat(),
                     "no_rows_after_validation_boundary": meta["window_end"] <= FUTURE_CLEAN_END.isoformat(),
                     "row_keys_sha256": sha256_array(row_keys(
                         np.full(len(users), cutoff.isoformat(), dtype="U10"), users)),
                     "channel_target_sha256": sha256_array(np.column_stack(
                         [ch["S"], ch["C"], ch["Y"]]))})
    frame["S"], frame["C"], frame["Y"] = val_S, val_C, val_Y
    audit = {
        "status": "PASS",
        "project_float_tolerance": {"atol": FLOAT_ATOL, "rtol": FLOAT_RTOL},
        "pilot_train_recipe": s.as_dict(),
        "train_cutoffs": [v.isoformat() for v in train_cutoffs],
        "train_fold_membership": {v.isoformat(): [t.isoformat() for t in s.train_cutoffs(v)]
                                  for v in FOLDS},
        "n_unique_train_rows": len(train["user_id"]),
        "n_validation_rows": len(frame),
        "rows": rows,
        "train_row_keys_sha256": sha256_array(row_keys(
            np.asarray([train_cutoffs[i].isoformat() for i in train["cutoff_code"]], dtype="U10"),
            train["user_id"])),
        "train_targets_sha256": sha256_array(np.column_stack([train["S"], train["C"], train["Y"]])),
        "validation_targets_sha256": sha256_array(np.column_stack([val_S, val_C, val_Y])),
        "full_parquet_used": True,
        "train_preview_used": False,
    }
    return audit, train


def contributions(S: np.ndarray, C: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    S, C = np.asarray(S, float), np.asarray(C, float)
    if np.any(S < 0) or np.any(C < 0):
        raise ValueError("channel GMV must be nonnegative")
    z, zs, zc = np.log1p(S + C), np.log1p(S), np.log1p(C)
    phi_s = 0.5 * zs + 0.5 * (z - zc)
    phi_c = 0.5 * zc + 0.5 * (z - zs)
    assert np.all(phi_s >= -1e-12) and np.all(phi_c >= -1e-12)
    assert np.allclose(phi_s + phi_c, z, rtol=0, atol=1e-12)
    u = np.divide(phi_s, z, out=np.full_like(z, 0.5), where=z > 0)
    return phi_s, phi_c, z, u


def simple_regime(S: np.ndarray, C: np.ndarray) -> np.ndarray:
    sp, cp = np.asarray(S) > 0, np.asarray(C) > 0
    return np.select([~sp & ~cp, sp & ~cp, ~sp & cp, sp & cp],
                     REGIMES, default="neither")


def dominant_regime(search: np.ndarray, total: np.ndarray) -> np.ndarray:
    search, total = np.asarray(search, float), np.asarray(total, float)
    share = np.divide(search, total, out=np.full_like(total, 0.5), where=total > 0)
    return np.select([total <= 0, share <= 0.30, share >= 0.70],
                     ["no_purchase", "catalog_heavy", "search_heavy"], default="mixed")


def history_for_fold(df: pl.DataFrame, cutoff: dt.date, users: np.ndarray) -> dict[str, np.ndarray]:
    hist = df.filter(pl.col("event_date") <= cutoff)
    age = (pl.lit(cutoff) - pl.col("event_date")).dt.total_days().cast(pl.Int32)
    hist = hist.with_columns(__age=age)
    expr: list[pl.Expr] = []
    for w in (30, 90, 180):
        m = pl.col("__age") < w
        expr += [
            pl.when(m).then(pl.col("gmv_search")).otherwise(0.0).sum().alias(f"hist_search_gmv_{w}"),
            pl.when(m).then(pl.col("gmv")).otherwise(0.0).sum().alias(f"hist_total_gmv_{w}"),
            pl.when(m).then(pl.col("search_to_ord")).otherwise(0).sum().alias(f"hist_search_ord_{w}"),
            pl.when(m).then(pl.col("to_ord")).otherwise(0).sum().alias(f"hist_total_ord_{w}"),
            pl.when(m).then(pl.col("search_to_cart")).otherwise(0).sum().alias(f"hist_search_cart_{w}"),
            pl.when(m).then(pl.col("to_cart")).otherwise(0).sum().alias(f"hist_total_cart_{w}"),
        ]
    expr += [
        pl.when(pl.col("gmv_search") > 0).then(pl.col("__age")).min().alias("rec_search_purchase"),
        pl.when(pl.col("gmv_cat") > 0).then(pl.col("__age")).min().alias("rec_catalog_purchase"),
        pl.when(pl.col("gmv") > 0).then(pl.col("__age")).min().alias("rec_buy_diag"),
        ((pl.col("__age") < 180) & (pl.col("gmv") > 0)).sum().alias("w180_days_buy_diag"),
    ]
    agg = hist.group_by("user_id").agg(expr)
    names = [e.meta.output_name() for e in expr]
    out = _left_join_numpy(users, agg, names, fill_zero=False)
    for name in names:
        if not name.startswith("rec_"):
            out[name] = np.nan_to_num(np.asarray(out[name], float), nan=0.0)
    horizon_no_history = (cutoff - dt.date(2025, 1, 1)).days + 1
    for c in ("rec_search_purchase", "rec_catalog_purchase", "rec_buy_diag"):
        a = np.asarray(out[c], float)
        a[~np.isfinite(a)] = horizon_no_history
        out[c] = a
    for w in (30, 90, 180):
        for kind in ("gmv", "ord", "cart"):
            num = np.asarray(out[f"hist_search_{kind}_{w}"], float)
            den = np.asarray(out[f"hist_total_{kind}_{w}"], float)
            out[f"hist_{kind}_share_{w}"] = np.divide(
                num, den, out=np.full_like(den, 0.5), where=den > 0)
            out[f"hist_{kind}_support_{w}"] = den
    out["share_trend_30_180"] = out["hist_gmv_share_30"] - out["hist_gmv_share_180"]
    out["recency_difference"] = out["rec_search_purchase"] - out["rec_catalog_purchase"]
    for w in (30, 90, 180):
        out[f"dominant_{w}"] = dominant_regime(out[f"hist_search_gmv_{w}"],
                                                 out[f"hist_total_gmv_{w}"])
    out["switch_30_90"] = (out["dominant_30"] != out["dominant_90"]).astype(np.int8)
    out["switch_90_180"] = (out["dominant_90"] != out["dominant_180"]).astype(np.int8)
    return out


def add_history(df: pl.DataFrame, frame: pd.DataFrame) -> None:
    columns: set[str] = set()
    chunks: dict[str, np.ndarray] = {}
    for cutoff in FOLDS:
        mask = frame["cutoff"].to_numpy() == cutoff.isoformat()
        users = frame.loc[mask, "user_id"].to_numpy(np.int64)
        hist = history_for_fold(df, cutoff, users)
        if not columns:
            columns = set(hist)
            for name, value in hist.items():
                dtype = object if np.asarray(value).dtype.kind in "OU" else np.asarray(value).dtype
                chunks[name] = np.empty(len(frame), dtype=dtype)
        assert set(hist) == columns
        for name, value in hist.items():
            chunks[name][mask] = value

        cached = features_cached(cutoff, L=None, norm_long=True)
        left = pl.DataFrame({"user_id": users, "__order": np.arange(len(users))})
        f = (left.join(cached.select(["user_id", "rec_buy", "w180_days_buy"]),
                       on="user_id", how="left").sort("__order"))
        cached_rec = f["rec_buy"].cast(pl.Float64).fill_null(
            (cutoff - dt.date(2025, 1, 1)).days + 1).to_numpy()
        assert np.allclose(cached_rec, hist["rec_buy_diag"],
                           rtol=0, atol=FLOAT_ATOL)
        assert np.allclose(f["w180_days_buy"].to_numpy(), hist["w180_days_buy_diag"],
                           rtol=0, atol=FLOAT_ATOL)
    for name, value in chunks.items():
        frame[name] = value
    frame["rec_buy"] = frame["rec_buy_diag"].astype(float)
    frame["w180_days_buy"] = frame["w180_days_buy_diag"].astype(float)


def target_identity(frame: pd.DataFrame) -> dict[str, Any]:
    ps, pc, z, u = contributions(frame["S"], frame["C"])
    frame["phi_s"], frame["phi_c"], frame["z_true"], frame["u_future"] = ps, pc, z, u
    frame["future_regime"] = simple_regime(frame["S"], frame["C"])
    frame["future_dominant"] = dominant_regime(frame["S"], frame["Y"])
    frame["future_search_share"] = np.divide(frame["S"], frame["Y"],
        out=np.full(len(frame), 0.5, dtype=float), where=frame["Y"].to_numpy() > 0)
    frame["historical_dominant"] = frame["dominant_90"]
    frame["channel_stayer"] = (frame["historical_dominant"] == frame["future_dominant"])
    frame["channel_switcher"] = (~frame["channel_stayer"]
                                 & (frame["historical_dominant"] != "no_purchase")
                                 & (frame["future_dominant"] != "no_purchase"))
    return {
        "status": "PASS",
        "n": len(frame),
        "max_abs_phi_identity": float(np.max(np.abs(ps + pc - z))),
        "min_phi_s": float(ps.min()), "min_phi_c": float(pc.min()),
        "zero_rows": int((z == 0).sum()),
        "zero_rows_nonzero_contribution": int(((z == 0) & ((ps != 0) | (pc != 0))).sum()),
        "symmetry_max_abs": float(np.max(np.abs(ps - contributions(frame["C"], frame["S"])[1]))),
        "targets_sha256": sha256_array(np.column_stack([ps, pc, z, u])),
    }


def _scope_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    rec = frame["rec_buy"].to_numpy(float)
    days = frame["w180_days_buy"].to_numpy(float)
    y = frame["Y"].to_numpy(float)
    return {
        "all": np.ones(len(frame), bool),
        "rec_buy_15_60": (rec >= 15) & (rec <= 60),
        "w180_days_buy_2_15": (days >= 2) & (days <= 15),
        "intersection": (rec >= 15) & (rec <= 60) & (days >= 2) & (days <= 15),
        "w180_days_buy_0_1": days <= 1,
        "w180_days_buy_ge16": days >= 16,
        "actual_total_y_0": y == 0,
        "gmv_low_0_100": (y > 0) & (y <= 100),
        "gmv_medium_100_1000": (y > 100) & (y <= 1000),
        "gmv_high_gt1000": y > 1000,
    }


def _channel_stats(frame: pd.DataFrame, mask: np.ndarray, scope: str, fold: str,
                   weight: np.ndarray | None = None) -> dict[str, Any]:
    s = frame.loc[mask, "S"].to_numpy(float)
    c = frame.loc[mask, "C"].to_numpy(float)
    y = frame.loc[mask, "Y"].to_numpy(float)
    if weight is None:
        weight = np.ones(len(s), float)
    else:
        weight = np.asarray(weight)[mask]
    weight = weight / weight.sum() if weight.sum() else weight
    regimes = frame.loc[mask, "future_regime"].to_numpy()
    ls, lc = np.log1p(s), np.log1p(c)
    wy = float(np.dot(weight, y))
    out: dict[str, Any] = {"fold": fold, "scope": scope, "n": len(s),
                           "gmv_share_search": float(np.dot(weight, s) / wy) if wy else 0.5,
                           "gmv_share_catalog": float(np.dot(weight, c) / wy) if wy else 0.5,
                           "zero_rate_s": float(np.dot(weight, s == 0)),
                           "zero_rate_c": float(np.dot(weight, c == 0)),
                           "zero_rate_total": float(np.dot(weight, y == 0))}
    for regime in REGIMES:
        out[f"regime_{regime}"] = float(np.dot(weight, regimes == regime))
    for subset, submask in (("all", np.ones(len(s), bool)),
                            ("total_positive", y > 0), ("both_positive", (s > 0) & (c > 0))):
        w = weight[submask]
        out[f"pearson_logsc_{subset}"] = finite_corr(ls[submask], lc[submask], weight=w)
        out[f"spearman_logsc_{subset}"] = finite_corr(ls[submask], lc[submask], "spearman", w)
    past = frame.loc[mask, "hist_gmv_share_90"].to_numpy(float)
    future = frame.loc[mask, "u_future"].to_numpy(float)
    out["past90_future_u_pearson"] = finite_corr(past, future, weight=weight)
    out["past90_future_u_spearman"] = finite_corr(past, future, "spearman", weight)
    out["switch_rate"] = float(np.dot(weight, frame.loc[mask, "channel_switcher"].to_numpy(bool)))
    return out


def structural_tables(frame: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]],
                                                     list[dict[str, Any]]]:
    stats: list[dict[str, Any]] = []
    drift: list[dict[str, Any]] = []
    scopes = _scope_masks(frame)
    cut = frame["cutoff"].to_numpy()
    for fold in FOLD_LABELS:
        fm = cut == fold
        stats.append(_channel_stats(frame, fm, "all", fold))
        for scope, sm in scopes.items():
            if scope != "all" and np.any(fm & sm):
                drift.append(_channel_stats(frame, fm & sm, scope, fold))
    stats.append(_channel_stats(frame, np.ones(len(frame), bool), "all", "wcv_weighted",
                                wcv_row_weights(cut)))
    for scope, sm in scopes.items():
        if scope != "all" and sm.any():
            drift.append(_channel_stats(frame, sm, scope, "wcv_weighted",
                                        wcv_row_weights(cut)))

    confusion: list[dict[str, Any]] = []
    for fold in (*FOLD_LABELS, "wcv_weighted"):
        fm = np.ones(len(frame), bool) if fold == "wcv_weighted" else cut == fold
        weight = wcv_row_weights(cut)[fm] if fold == "wcv_weighted" else np.ones(fm.sum())
        past = frame.loc[fm, "historical_dominant"].to_numpy()
        future = frame.loc[fm, "future_dominant"].to_numpy()
        for a in DOMINANT:
            den = float(weight[past == a].sum())
            for b in DOMINANT:
                n = float(weight[(past == a) & (future == b)].sum())
                confusion.append({"fold": fold, "past": a, "future": b,
                                  "weighted_count": n,
                                  "row_rate_within_past": n / den if den else float("nan")})
    return stats, drift, confusion


def _dominant_code(value: np.ndarray) -> np.ndarray:
    mapping = {"no_purchase": -1.0, "catalog_heavy": 0.0, "mixed": 1.0,
               "search_heavy": 2.0}
    return np.asarray([mapping[str(v)] for v in value], float)


def diagnostic_variables(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    out = {}
    for kind in ("gmv", "ord", "cart"):
        for w in (30, 90, 180):
            out[f"search_{kind}_share_{w}"] = frame[f"hist_{kind}_share_{w}"].to_numpy(float)
    out.update({
        "share_trend_30_minus_180": frame["share_trend_30_180"].to_numpy(float),
        "recency_last_search_purchase": frame["rec_search_purchase"].to_numpy(float),
        "recency_last_catalog_purchase": frame["rec_catalog_purchase"].to_numpy(float),
        "recency_difference": frame["recency_difference"].to_numpy(float),
        "dominant_historical_channel": _dominant_code(frame["dominant_90"].to_numpy()),
        "historical_switch_30_90": frame["switch_30_90"].to_numpy(float),
        "historical_switch_90_180": frame["switch_90_180"].to_numpy(float),
    })
    return out


def fixed_bin(variable: str, value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, float)
    if "share" in variable and "trend" not in variable:
        return np.searchsorted([0.05, 0.25, 0.50, 0.75, 0.95], value, side="right")
    if "trend" in variable:
        return np.searchsorted([-0.50, -0.25, -0.05, 0.05, 0.25, 0.50], value, side="right")
    if variable.startswith("recency_last"):
        return np.searchsorted([7, 15, 30, 60, 90, 180], value, side="right")
    if variable == "recency_difference":
        return np.searchsorted([-60, -30, -7, 7, 30, 60], value, side="right")
    return value.astype(int)


def residual_tables(frame: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]],
                                                   list[dict[str, Any]]]:
    variables = diagnostic_variables(frame)
    residual = frame["residual"].to_numpy(float)
    cut = frame["cutoff"].to_numpy()
    corr_rows, bin_rows, segment_rows = [], [], []
    scopes = _scope_masks(frame)
    for name, value in variables.items():
        signs = []
        for fold in FOLD_LABELS:
            m = cut == fold
            p = finite_corr(value[m], residual[m])
            s = finite_corr(value[m], residual[m], "spearman")
            signs.append(int(np.sign(s)) if np.isfinite(s) else 0)
            corr_rows.append({"variable": name, "fold": fold, "n": int(m.sum()),
                              "pearson": p, "spearman": s})
        weight = wcv_row_weights(cut)
        corr_rows.append({"variable": name, "fold": "wcv_weighted", "n": len(frame),
                          "pearson": finite_corr(value, residual, weight=weight),
                          "spearman": finite_corr(value, residual, "spearman", weight),
                          "positive_fold_signs": int(sum(v > 0 for v in signs)),
                          "negative_fold_signs": int(sum(v < 0 for v in signs))})
        bins = fixed_bin(name, value)
        for fold in (*FOLD_LABELS, "wcv_weighted"):
            fm = np.ones(len(frame), bool) if fold == "wcv_weighted" else cut == fold
            w = wcv_row_weights(cut) if fold == "wcv_weighted" else np.ones(len(frame))
            for b in np.unique(bins[fm]):
                m = fm & (bins == b)
                bin_rows.append({"variable": name, "fold": fold, "bin": int(b),
                                 "n": int(m.sum()),
                                 "residual_mean": float(np.average(residual[m], weights=w[m])),
                                 "residual_std": float(np.sqrt(np.average(
                                     (residual[m] - np.average(residual[m], weights=w[m])) ** 2,
                                     weights=w[m])))})
        for scope, sm in scopes.items():
            if sm.sum() < 3:
                continue
            segment_rows.append({"variable": name, "segment": scope, "n": int(sm.sum()),
                                 "pearson": finite_corr(value[sm], residual[sm]),
                                 "spearman": finite_corr(value[sm], residual[sm], "spearman"),
                                 "residual_mean": float(residual[sm].mean())})
    return corr_rows, bin_rows, segment_rows


def stable_deciles(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(value, float)
    order = np.argsort(value, kind="stable")
    dec = np.empty(len(value), np.int8)
    dec[order] = np.minimum((np.arange(len(value), dtype=np.int64) * 10) // len(value), 9)
    edges = np.array([value[order[min((k * len(value)) // 10, len(value) - 1)]]
                      for k in range(1, 10)], float)
    return dec, edges


def oracle_lofo(frame: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    cut = frame["cutoff"].to_numpy()
    base = frame["z_base_cal"].to_numpy(float)
    residual = frame["residual"].to_numpy(float)
    regime_code = pd.Categorical(frame["future_regime"], categories=REGIMES).codes
    u_bin = np.searchsorted([0.05, 0.25, 0.50, 0.75, 0.95],
                            frame["u_future"].to_numpy(float), side="right").astype(np.int8)
    pred_dec = np.empty(len(frame), np.int8)
    decile_edges = {}
    for fold in FOLD_LABELS:
        m = cut == fold
        pred_dec[m], edges = stable_deciles(base[m])
        decile_edges[fold] = edges.tolist()
    cell = regime_code.astype(np.int32) * 60 + u_bin.astype(np.int32) * 10 + pred_dec
    correction = np.zeros(len(frame), float)
    mappings: list[dict[str, Any]] = []
    for outer in FOLD_LABELS:
        donor, recipient = cut != outer, cut == outer
        assert not np.any(donor & recipient)
        for code in np.unique(cell[donor]):
            dm = donor & (cell == code)
            n = int(dm.sum())
            mean = float(residual[dm].mean())
            corr = mean * n / (n + SHRINKAGE)
            correction[recipient & (cell == code)] = corr
            mappings.append({"outer_fold": outer, "cell": int(code), "n_donor": n,
                             "residual_mean": mean, "shrinkage": n / (n + SHRINKAGE),
                             "correction": corr})
    candidate = base + correction
    base_scores, candidate_scores, centered_scores, offsets = [], [], [], []
    fold_rows = []
    for fold in FOLD_LABELS:
        m = cut == fold
        bo, bs = calibrate(frame.loc[m, "Y"], base[m])
        co, cs = calibrate(frame.loc[m, "Y"], candidate[m])
        centered = base[m] + correction[m] - correction[m].mean()
        _, csc = calibrate(frame.loc[m, "Y"], centered)
        base_scores.append(bs); candidate_scores.append(cs); centered_scores.append(csc); offsets.append(co)
        fold_rows.append({"fold": fold, "n": int(m.sum()), "baseline": bs, "oracle": cs,
                          "delta": cs - bs, "centered_shape_only": csc,
                          "shape_delta": csc - bs, "candidate_calibration_offset": co,
                          "correction_mean": float(correction[m].mean()),
                          "correction_var": float(correction[m].var()),
                          "correction_max_abs": float(np.abs(correction[m]).max())})
    w = np.asarray(FOLD_WEIGHTS_S1, float); w /= w.sum()
    delta = float(w @ (np.asarray(candidate_scores) - np.asarray(base_scores)))
    shape_delta = float(w @ (np.asarray(centered_scores) - np.asarray(base_scores)))
    improved = int((np.asarray(candidate_scores) < np.asarray(base_scores)).sum())
    summary = {
        "method": "honest LOFO joint cell: future regime x fixed-u-bin x STRONGEST decile",
        "shrinkage": "n/(n+20000)", "decile_method": "stable rank deciles within fold",
        "decile_edges": decile_edges, "folds": fold_rows,
        "baseline_wcv": float(w @ np.asarray(base_scores)),
        "oracle_wcv": float(w @ np.asarray(candidate_scores)),
        "delta_wcv": delta, "shape_only_delta_wcv": shape_delta,
        "improved_folds": improved,
        "improves_2025_10_16": bool(candidate_scores[-1] < base_scores[-1]),
        "not_only_level_shift": bool(np.var(correction) > 0 and shape_delta < -1e-12),
        "correction_sha256": sha256_array(correction),
    }
    return summary, mappings, correction


def predictability(frame: pd.DataFrame) -> dict[str, Any]:
    cut = frame["cutoff"].to_numpy()
    past = frame["hist_gmv_share_90"].to_numpy(float)
    future = frame["u_future"].to_numpy(float)
    per_fold = []
    for fold in FOLD_LABELS:
        m = cut == fold
        pos = frame["z_true"].to_numpy(float)[m] > 0
        per_fold.append({
            "fold": fold,
            "n": int(m.sum()),
            "spearman_all": finite_corr(past[m], future[m], "spearman"),
            "pearson_all": finite_corr(past[m], future[m]),
            "spearman_future_positive": finite_corr(past[m][pos], future[m][pos], "spearman"),
            "pearson_future_positive": finite_corr(past[m][pos], future[m][pos]),
        })
    w = np.asarray(FOLD_WEIGHTS_S1, float); w /= w.sum()
    sp = np.asarray([r["spearman_all"] for r in per_fold], float)
    corr_gate = bool(float(w @ sp) >= 0.15 and int((sp > 0).sum()) >= 3)
    result = {"past90_future_contribution_share": per_fold,
              "weighted_spearman": float(w @ sp),
              "positive_folds": int((sp > 0).sum()),
              "correlation_gate_pass": corr_gate,
              "classifier": {"status": "NOT_NEEDED_CORRELATION_GATE_PASS"}}
    if corr_gate:
        result["predictability_gate_pass"] = True
        return result

    import lightgbm as lgb
    variables = diagnostic_variables(frame)
    names = list(variables)
    X = np.column_stack([np.nan_to_num(variables[n], nan=0.5, posinf=999.0, neginf=-999.0)
                         for n in names]).astype(np.float32)
    y = pd.Categorical(frame["future_regime"], categories=REGIMES).codes.astype(np.int32)
    params = {"objective": "multiclass", "num_class": 4, "metric": "multi_logloss",
              "learning_rate": 0.1, "num_leaves": 15, "min_data_in_leaf": 500,
              "feature_fraction": 1.0, "bagging_fraction": 1.0, "lambda_l2": 5.0,
              "max_bin": 63, "force_row_wise": True, "verbose": -1,
              "seed": SEED, "num_threads": int(LGB_PARAMS["num_threads"])}
    rows = []
    for fold in FOLD_LABELS:
        tr, va = cut != fold, cut == fold
        model = lgb.train(params, lgb.Dataset(X[tr], y[tr]), num_boost_round=50)
        proba = model.predict(X[va])
        auc = float(roc_auc_score(y[va], proba, average="macro", multi_class="ovr",
                                  labels=np.arange(4)))
        rows.append({"fold": fold, "macro_ovr_auc": auc, "n": int(va.sum())})
    auc = np.asarray([r["macro_ovr_auc"] for r in rows])
    classifier_gate = bool(int((auc >= 0.60).sum()) >= 3)
    result["classifier"] = {"status": "RUN", "features": names, "params": params,
                            "rounds": 50, "folds": rows,
                            "passing_folds": int((auc >= 0.60).sum()),
                            "gate_pass": classifier_gate}
    result["predictability_gate_pass"] = classifier_gate
    return result


def stratified_shuffle(z: np.ndarray, u: np.ndarray, cutoff_code: np.ndarray,
                       seed: int = SEED, validation_mask: np.ndarray | None = None
                       ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    z, u = np.asarray(z, float), np.asarray(u, float)
    cutoff_code = np.asarray(cutoff_code)
    if validation_mask is not None and np.any(validation_mask):
        raise AssertionError("validation rows are forbidden in shuffle edges/permutations")
    rng = np.random.default_rng(seed)
    shuffled = np.empty_like(u)
    permutation = np.empty(len(u), np.int64)
    decile = np.empty(len(u), np.int8)
    strata = []
    for code in np.unique(cutoff_code):
        cm = cutoff_code == code
        local_dec, edges = stable_deciles(z[cm])
        decile[cm] = local_dec
        idx_code = np.flatnonzero(cm)
        for d in range(10):
            idx = idx_code[local_dec == d]
            if not len(idx):
                continue
            perm = idx[rng.permutation(len(idx))]
            permutation[idx] = perm
            shuffled[idx] = u[perm]
            assert np.array_equal(np.sort(shuffled[idx]), np.sort(u[idx]))
            strata.append({"cutoff_code": int(code), "decile": d, "n": len(idx),
                           "z_edge_low": float(edges[d - 1]) if d > 0 else None,
                           "z_edge_high": float(edges[d]) if d < 9 else None,
                           "u_sha256_before": sha256_array(np.sort(u[idx])),
                           "u_sha256_after": sha256_array(np.sort(shuffled[idx]))})
    assert np.array_equal(np.sort(permutation), np.arange(len(u)))
    return shuffled, permutation, {"seed": seed, "strata": strata,
                                   "permutation_sha256": sha256_array(permutation),
                                   "u_shuffled_sha256": sha256_array(shuffled)}


def splitmix64(user_id: np.ndarray) -> np.ndarray:
    x = np.asarray(user_id, dtype=np.uint64)
    z = x + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def choose_alpha(y: np.ndarray, base: np.ndarray, direction: np.ndarray,
                 selector: np.ndarray, recipient: np.ndarray,
                 alphas: tuple[float, ...] = ALPHAS) -> tuple[float, list[dict[str, float]]]:
    selector, recipient = np.asarray(selector, bool), np.asarray(recipient, bool)
    assert not np.any(selector & recipient)
    assert np.all(selector | recipient)
    curve = []
    for alpha in alphas:
        _, score = calibrate(y[selector], base[selector] + alpha * direction[selector])
        curve.append({"alpha": float(alpha), "selector_score": score})
    best = min(v["selector_score"] for v in curve)
    selected = min(v["alpha"] for v in curve if v["selector_score"] <= best + 1e-5)
    return float(selected), curve


def raw_log_blend(base: np.ndarray, direction: np.ndarray, alpha: float) -> np.ndarray:
    """Assembly only.  Calibration is deliberately not available here."""
    return np.asarray(base, float) + float(alpha) * np.asarray(direction, float)


def pilot_train(frame: pd.DataFrame, train: dict[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    import lightgbm as lgb
    s = pilot_setup()
    feats = (ARTIFACTS / "feats_S1-E10.txt").read_text(encoding="utf-8").splitlines()
    assert len(feats) == 227
    Xv, yv = xy(PILOT_FOLD, s)
    assert feature_names(Xv) == feats
    val_mask = frame["cutoff"].to_numpy() == PILOT_FOLD.isoformat()
    assert np.array_equal(Xv["user_id"].to_numpy(), np.sort(frame.loc[val_mask, "user_id"]))
    cuts = s.train_cutoffs(PILOT_FOLD)
    Xtr, ytr, _ = assemble(cuts, s, feats, PILOT_FOLD)
    assert Xtr.dtype == np.float32 and Xtr.shape[1] == 227
    assert np.allclose(ytr, train["Y"], rtol=FLOAT_RTOL, atol=FLOAT_ATOL)
    ps, pc, z, u = contributions(train["S"], train["C"])
    ush, perm, shuffle_manifest = stratified_shuffle(z, u, train["cutoff_code"], SEED,
                                                     np.zeros(len(z), bool))
    ps_sh, pc_sh = ush * z, (1.0 - ush) * z
    assert np.allclose(ps_sh + pc_sh, z, rtol=0, atol=1e-12)
    targets = {"REAL_SEARCH": ps, "REAL_CATALOG": pc,
               "SHUF_SEARCH": ps_sh, "SHUF_CATALOG": pc_sh}
    seeds = {"REAL_SEARCH": SEED, "REAL_CATALOG": SEED + 1,
             "SHUF_SEARCH": SEED, "SHUF_CATALOG": SEED + 1}
    Av = to_np(Xv, feats)
    predictions = {}
    configs = {}
    model_files = {}
    for name in ("REAL_SEARCH", "REAL_CATALOG", "SHUF_SEARCH", "SHUF_CATALOG"):
        params = dict(LGB_PARAMS); params["seed"] = seeds[name]
        ds = lgb.Dataset(Xtr, targets[name], params=params).construct()
        model = lgb.train(params, ds, num_boost_round=PILOT_ROUNDS)
        pred = model.predict(Av).astype(np.float32)
        predictions[name] = pred
        model_path = RUN_DIR / f"{PREFIX}_{name}_V1016.txt"
        text = model.model_to_string(num_iteration=PILOT_ROUNDS)
        model_files[name] = {"path": str(model_path.resolve()),
                             "sha256": save_text_once(model_path, text)}
        configs[name] = {"target": name, "seed": seeds[name], "params": params,
                         "rounds": PILOT_ROUNDS, "features": feats,
                         "feature_order_sha256": sha256_array(np.asarray(feats, dtype="U")),
                         "n_train": len(ytr), "n_validation": len(yv),
                         "train_rows_sha256": sha256_array(np.column_stack(
                             [train["cutoff_code"].astype(np.int64), train["user_id"]])),
                         "matrix_identity": "one materialized float32 Xtr shared by all four trajectories",
                         "early_stopping": False, "thread_policy": int(params["num_threads"])}
        del model, ds
        gc.collect()
    assert configs["REAL_SEARCH"]["params"] == configs["SHUF_SEARCH"]["params"]
    assert configs["REAL_CATALOG"]["params"] == configs["SHUF_CATALOG"]["params"]
    z_real = predictions["REAL_SEARCH"].astype(float) + predictions["REAL_CATALOG"].astype(float)
    z_shuf = predictions["SHUF_SEARCH"].astype(float) + predictions["SHUF_CATALOG"].astype(float)
    out = {**predictions, "z_real": z_real.astype(np.float32),
           "z_shuf": z_shuf.astype(np.float32), "d": (z_real - z_shuf).astype(np.float32),
           "user_id": Xv["user_id"].to_numpy().astype(np.int64),
           "y": np.asarray(yv, np.float32)}
    manifest = {"status": "COMPLETE", "fold": PILOT_FOLD.isoformat(),
                "features": "exact S1-E10", "n_features": len(feats),
                "rounds": PILOT_ROUNDS, "configs": configs, "models": model_files,
                "shuffle": shuffle_manifest, "permutation_sha256": sha256_array(perm),
                "real_shuf_matrix_identity": True, "direct_total_head": False,
                "target_hashes": {k: sha256_array(v) for k, v in targets.items()}}
    del Xtr, Av, Xv, ytr
    _XY.clear(); gc.collect()
    return manifest, out


def _rmsle_with_offset(y: np.ndarray, z: np.ndarray, offset: float) -> float:
    return rmsle_z(y, np.asarray(z, float) + offset)


def pilot_analysis(frame: pd.DataFrame, pred: dict[str, np.ndarray]) -> tuple[dict[str, Any],
                                                                            list[dict[str, Any]],
                                                                            list[dict[str, Any]]]:
    mask = frame["cutoff"].to_numpy() == PILOT_FOLD.isoformat()
    fold = frame.loc[mask].copy().sort_values("user_id")
    assert np.array_equal(fold["user_id"].to_numpy(np.int64), pred["user_id"])
    y, base = pred["y"].astype(float), fold["z_base_raw"].to_numpy(float)
    zr, zs, d = pred["z_real"].astype(float), pred["z_shuf"].astype(float), pred["d"].astype(float)
    bo, bs = calibrate(y, base); ro, rs = calibrate(y, zr); so, ss = calibrate(y, zs)
    standalone = {"baseline": {"offset": bo, "rmsle": bs},
                  "z_real": {"offset": ro, "rmsle": rs},
                  "z_shuf": {"offset": so, "rmsle": ss},
                  "real_minus_shuf": rs - ss,
                  "auc_real": float(roc_auc_score(y > 0, zr)),
                  "auc_shuf": float(roc_auc_score(y > 0, zs)),
                  "positive_rmsle_real": _rmsle_with_offset(y[y > 0], zr[y > 0], ro),
                  "positive_rmsle_shuf": _rmsle_with_offset(y[y > 0], zs[y > 0], so),
                  "zero_rmsle_real": _rmsle_with_offset(y[y == 0], zr[y == 0], ro),
                  "zero_rmsle_shuf": _rmsle_with_offset(y[y == 0], zs[y == 0], so)}
    half = (splitmix64(pred["user_id"]) & np.uint64(1)).astype(np.int8)
    A, B = half == 0, half == 1
    assert np.all(A ^ B) and not np.any(A & B)
    alpha_ab, curve_ab = choose_alpha(y, base, d, A, B)
    alpha_ba, curve_ba = choose_alpha(y, base, d, B, A)
    assembled = base.copy()
    assembled[B] = raw_log_blend(base[B], d[B], alpha_ab)
    assembled[A] = raw_log_blend(base[A], d[A], alpha_ba)
    final_offset, final_score = calibrate(y, assembled)
    fixed_curve = []
    for alpha in ALPHAS:
        off, score = calibrate(y, raw_log_blend(base, d, alpha))
        fixed_curve.append({"alpha": alpha, "offset": off, "rmsle": score, "delta": score - bs})
    half_rows = []
    residual = np.log1p(y) - (base + bo)
    for name, hm, selected in (("recipient_A", A, alpha_ba), ("recipient_B", B, alpha_ab)):
        base_half = rmsle_z(y[hm], base[hm] + bo)
        cand_half = rmsle_z(y[hm], assembled[hm] + final_offset)
        half_rows.append({"half": name, "n": int(hm.sum()), "selected_alpha": selected,
                          "baseline_fixed_fold_cal": base_half,
                          "candidate_fixed_fold_cal": cand_half,
                          "delta": cand_half - base_half,
                          "corr_d_residual": finite_corr(d[hm], residual[hm]),
                          "spearman_d_residual": finite_corr(d[hm], residual[hm], "spearman")})
    diagnostics = {"two_sided_rmsle": final_score, "two_sided_delta": final_score - bs,
                   "final_fold_offset": final_offset, "alpha_A_to_B": alpha_ab,
                   "alpha_B_to_A": alpha_ba, "selector_curve_A": curve_ab,
                   "selector_curve_B": curve_ba, "fixed_curves": fixed_curve,
                   "halves": half_rows, "var_d": float(np.var(d)),
                   "max_abs_d": float(np.max(np.abs(d))), "mean_d": float(np.mean(d)),
                   "pearson_real_strongest": finite_corr(zr, base),
                   "spearman_real_strongest": finite_corr(zr, base, "spearman"),
                   "corr_residuals_real_strongest": finite_corr(np.log1p(y) - zr,
                                                                 np.log1p(y) - base),
                   "halves_sha256": sha256_array(half), "raw_log_space_blend": True,
                   "calibration_after_final_assembly": True,
                   "gain_not_level_shift": bool(final_score < bs and np.var(d) > 0)}

    fold = fold.reset_index(drop=True)
    fold["candidate"] = assembled
    fold["half"] = half
    segment_rows = []
    seg = _scope_masks(fold)
    seg.update({"historical_search_heavy": fold["historical_dominant"].to_numpy() == "search_heavy",
                "historical_mixed": fold["historical_dominant"].to_numpy() == "mixed",
                "historical_catalog_heavy": fold["historical_dominant"].to_numpy() == "catalog_heavy",
                "future_search_only": fold["future_regime"].to_numpy() == "search_only",
                "future_catalog_only": fold["future_regime"].to_numpy() == "catalog_only",
                "future_both": fold["future_regime"].to_numpy() == "both",
                "future_neither": fold["future_regime"].to_numpy() == "neither",
                "channel_stayers": fold["channel_stayer"].to_numpy(bool),
                "channel_switchers": fold["channel_switcher"].to_numpy(bool),
                "actual_total_y_positive": y > 0})
    for name, sm in seg.items():
        if sm.sum() < 2:
            continue
        fixed_b = rmsle_z(y[sm], base[sm] + bo)
        fixed_c = rmsle_z(y[sm], assembled[sm] + final_offset)
        _, shape_b = calibrate(y[sm], base[sm]); _, shape_c = calibrate(y[sm], assembled[sm])
        segment_rows.append({"segment": name, "n": int(sm.sum()),
                             "fixed_fold_baseline": fixed_b, "fixed_fold_candidate": fixed_c,
                             "fixed_fold_delta": fixed_c - fixed_b,
                             "shape_only_baseline": shape_b, "shape_only_candidate": shape_c,
                             "shape_only_delta": shape_c - shape_b})
    real_better = rs < ss - 1e-5
    halves_better = all(v["delta"] < 0 for v in half_rows)
    corr_positive = all(v["corr_d_residual"] > 0 for v in half_rows)
    alpha_positive = alpha_ab > 0 and alpha_ba > 0
    delta = final_score - bs
    if delta <= -0.001 and halves_better and alpha_positive and real_better and corr_positive \
            and diagnostics["gain_not_level_shift"]:
        verdict = "STRONG_PASS"
    elif delta <= -0.0007 and halves_better and alpha_positive and real_better:
        verdict = "PASS_TO_FULL_FOLDS"
    elif (delta > -0.0003 or not alpha_positive or not halves_better or not real_better
          or not corr_positive or not diagnostics["gain_not_level_shift"]):
        verdict = "REJECT"
    else:
        verdict = "STOP_WEAK_SIGNAL"
    summary = {"standalone": standalone, "residual": diagnostics, "verdict": verdict,
               "promote_to_full_folds": verdict in ("STRONG_PASS", "PASS_TO_FULL_FOLDS"),
               "full_folds_run": False}
    return summary, half_rows, segment_rows


def validation_arrays(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    numeric = [c for c in frame.columns if frame[c].dtype != object and c != "user_id"]
    arrays = {"user_id": frame["user_id"].to_numpy(np.int64),
              "cutoff": frame["cutoff"].to_numpy(dtype="U10")}
    for c in numeric:
        arrays[c] = frame[c].to_numpy()
    for c in ("future_regime", "future_dominant", "historical_dominant"):
        arrays[c] = frame[c].to_numpy(dtype="U32")
    arrays["channel_stayer"] = frame["channel_stayer"].to_numpy(bool)
    arrays["channel_switcher"] = frame["channel_switcher"].to_numpy(bool)
    return arrays


def load_validation_frame(path: Path) -> pd.DataFrame:
    d = np.load(path, allow_pickle=False)
    return pd.DataFrame({k: d[k] for k in d.files})


def analysis_replay(frame: pd.DataFrame, expected: str | None = None) -> str:
    oracle, _, correction = oracle_lofo(frame)
    payload = {"frame_row_keys": sha256_array(row_keys(frame["cutoff"].to_numpy(),
                                                       frame["user_id"].to_numpy())),
               "frame_targets": sha256_array(np.column_stack([frame["S"], frame["C"], frame["Y"]])),
               "base_scores": exact_baseline()[1]["fold_scores_calibrated"],
               "oracle_delta": oracle["delta_wcv"],
               "oracle_fold_delta": [v["delta"] for v in oracle["folds"]],
               "correction_sha256": sha256_array(correction)}
    digest = canonical_sha256(payload)
    if expected is not None:
        assert digest == expected, f"analysis replay drift: {digest} != {expected}"
    return digest


def run_analysis_only() -> None:
    summary_path = RESULTS / f"{PREFIX}_summary.json"
    frame_path = RUN_DIR / f"{PREFIX}_validation_frame.npz"
    assert summary_path.exists() and frame_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    frame = load_validation_frame(frame_path)
    digest = analysis_replay(frame, summary["analysis_replay_sha256"])
    print(f"analysis replay PASS {digest}", flush=True)


def run() -> dict[str, Any]:
    started = time.time()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame, baseline = exact_baseline()
    save_json_once(RESULTS / f"{PREFIX}_baseline_manifest.json", baseline)
    raw = load()
    daily = raw_daily_audit(raw)
    target_report, train = target_audit(raw, frame)
    add_history(raw, frame)
    identity = target_identity(frame)
    save_json_once(RESULTS / f"{PREFIX}_channel_data_audit.json",
                   {"daily": daily, "targets": target_report})
    save_json_once(RESULTS / f"{PREFIX}_target_identity_audit.json", identity)
    save_npz_once(RUN_DIR / f"{PREFIX}_train_targets.npz", **train)
    frame_hash = save_npz_once(RUN_DIR / f"{PREFIX}_validation_frame.npz",
                               **validation_arrays(frame))

    stats, drift, confusion = structural_tables(frame)
    residual_corr, residual_bins, residual_segments = residual_tables(frame)
    oracle, oracle_maps, correction = oracle_lofo(frame)
    pred = predictability(frame)
    save_csv_once(RESULTS / f"{PREFIX}_fold_channel_statistics.csv", stats)
    save_csv_once(RESULTS / f"{PREFIX}_drift_segments.csv", drift)
    save_csv_once(RESULTS / f"{PREFIX}_regime_confusion.csv", confusion)
    save_csv_once(RESULTS / f"{PREFIX}_residual_correlations.csv", residual_corr)
    save_csv_once(RESULTS / f"{PREFIX}_residual_bins.csv", residual_bins)
    save_csv_once(RESULTS / f"{PREFIX}_residual_segments.csv", residual_segments)
    save_csv_once(RESULTS / f"{PREFIX}_oracle_lofo_cells.csv", oracle_maps)
    save_json_once(RESULTS / f"{PREFIX}_oracle_lofo.json", oracle)
    save_json_once(RESULTS / f"{PREFIX}_predictability.json", pred)

    oracle_gate = (oracle["delta_wcv"] <= -0.0015 and oracle["improved_folds"] >= 3
                   and oracle["improves_2025_10_16"] and oracle["not_only_level_shift"])
    preflight_pass = bool(oracle_gate and pred["predictability_gate_pass"])
    preflight = {"verdict": "GO" if preflight_pass else "NO_GO_PREFLIGHT",
                 "oracle_gate_pass": oracle_gate,
                 "predictability_gate_pass": pred["predictability_gate_pass"],
                 "requirements": {"oracle_delta_max": -0.0015, "oracle_folds_min": 3,
                                  "oracle_must_improve_2025_10_16": True,
                                  "not_level_shift": True,
                                  "weighted_spearman_min": 0.15,
                                  "positive_spearman_folds_min": 3,
                                  "classifier_macro_ovr_auc_min": 0.60},
                 "oracle": oracle, "predictability": pred}
    save_json_once(RESULTS / f"{PREFIX}_preflight_verdict.json", preflight)

    pilot = {"status": "NOT_RUN_PREFLIGHT_GATE_FAILED"}
    pilot_result = None
    if preflight_pass:
        pilot_manifest, pilot_pred = pilot_train(frame, train)
        save_json_once(RESULTS / f"{PREFIX}_pilot_model_manifest.json", pilot_manifest)
        save_npz_once(RUN_DIR / f"{PREFIX}_pilot_validation_predictions.npz", **pilot_pred)
        pilot_result, halves, segments = pilot_analysis(frame, pilot_pred)
        save_json_once(RESULTS / f"{PREFIX}_standalone_and_residual.json", pilot_result)
        save_csv_once(RESULTS / f"{PREFIX}_two_sided_halves.csv", halves)
        save_csv_once(RESULTS / f"{PREFIX}_pilot_segments.csv", segments)
        pilot = pilot_result

    verdict = "NO_GO_PREFLIGHT" if not preflight_pass else pilot["verdict"]
    summary_core = {"experiment": EXP_ID, "experiment_number": EXP_NUM,
                    "prefix": PREFIX, "baseline": baseline,
                    "daily_audit": daily, "target_audit_status": target_report["status"],
                    "target_identity": identity, "oracle": oracle,
                    "predictability": pred, "preflight": preflight,
                    "pilot": pilot, "verdict": verdict,
                    "promote_to_full_folds": bool(pilot_result and
                        pilot_result["verdict"] in ("STRONG_PASS", "PASS_TO_FULL_FOLDS")),
                    "full_folds_run": False, "test_inference_run": False,
                    "submission_created": False, "public_lb_accessed": False,
                    "test_paths_accessed": False, "frame_artifact_sha256": frame_hash,
                    "runtime_seconds": round(time.time() - started, 3)}
    replay = analysis_replay(frame)
    summary_core["analysis_replay_sha256"] = replay
    summary_core["canonical_result_sha256"] = canonical_sha256(
        {k: v for k, v in summary_core.items() if k != "runtime_seconds"})
    save_json_once(RESULTS / f"{PREFIX}_summary.json", summary_core)
    return summary_core


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-only", action="store_true")
    args = ap.parse_args()
    if args.analysis_only:
        run_analysis_only()
    else:
        summary = run()
        print(summary["verdict"], flush=True)


if __name__ == "__main__":
    main()
