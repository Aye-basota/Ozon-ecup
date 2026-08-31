"""EXP-053: artifact-only residual signal discovery for STRONGEST_CURRENT.

The runner never reads test/submission/LB artifacts and never trains a base
model.  It aligns saved OOF predictions on (cutoff, user_id), joins the existing
227 cutoff-safe S1-E10 feature columns, runs fixed small CPU LightGBM probes,
and writes a fully reproducible diagnostic bundle.

Run:
    python src/residual_signal_discovery.py
    python src/residual_signal_discovery.py --analysis-only
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import rankdata
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.btyd_day_bgnbd import user_group
from src.config import FOLD_WEIGHTS_S1, SEED
from src.validation import calibrate


PREFIX = "RESDISC_053"
EXPERIMENT_ID = 53
FOLDS = ("2025-09-04", "2025-09-18", "2025-10-02", "2025-10-16")
FOLD_WEIGHTS = np.asarray(FOLD_WEIGHTS_S1, dtype=float)
EXPECTED_FOLD_SCORES = np.asarray(
    [1.766883357, 1.760509577, 1.748629224, 1.741278566], dtype=float
)
EXPECTED_WCV = 1.747509863

ARTIFACTS = ROOT / "artifacts"
OUT_ARTIFACTS = ARTIFACTS / PREFIX
RESULTS = ROOT / "research" / "strategies" / "results" / "RESIDUAL_SIGNAL_DISCOVERY"
DIAGNOSTIC_FRAME = ROOT / "research" / "rmsle_diagnostics" / "fold_predictions.parquet"

CORE_COMPONENTS = {
    "cap": ARTIFACTS / "oof_S1-E03a.npz",
    "unc": ARTIFACTS / "oof_S1-E02.npz",
    "dist": ARTIFACTS / "oof_S1-DIST.npz",
    "etx": ARTIFACTS / "oof_ETX-AVG3.npz",
    "seq": ARTIFACTS / "oof_SEQ-AVG3.npz",
}
OPTIONAL_COMPONENTS = {
    "etx_s42": ARTIFACTS / "oof_ETX-01-S42.npz",
    "etx_s43": ARTIFACTS / "oof_ETX-01-S43.npz",
    "etx_s44": ARTIFACTS / "oof_ETX-01-S44.npz",
    "seq_s42": ARTIFACTS / "oof_SEQ-01-S42.npz",
    "seq_s43": ARTIFACTS / "oof_SEQ-01-S43.npz",
    "seq_s44": ARTIFACTS / "oof_SEQ-01-S44.npz",
    "d3a_avg3": ARTIFACTS / "oof_SEQ-D3A-AVG3.npz",
    "ridge": ARTIFACTS / "oof_RIDGE15.npz",
}

PROBE_PARAMS = {
    "num_leaves": 31,
    "min_data_in_leaf": 2000,
    "learning_rate": 0.03,
    "num_boost_round": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 20.0,
    "max_bin": 63,
    "force_row_wise": True,
    "seed": int(SEED),
    "early_stopping": False,
}
SCALES = np.asarray([0.0, 0.25, 0.50, 1.0], dtype=float)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True),
                    encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_csv(path, index=False, lineterminator="\n")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_sha256(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for value in arrays:
        a = np.ascontiguousarray(value)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.view(np.uint8))
    return h.hexdigest()


def canonical_order(uid: np.ndarray, cutoff: np.ndarray) -> np.ndarray:
    return np.lexsort((np.asarray(uid, dtype=np.int64), np.asarray(cutoff, dtype="U10")))


def assert_unique_keys(uid: np.ndarray, cutoff: np.ndarray) -> None:
    order = canonical_order(uid, cutoff)
    u = np.asarray(uid, dtype=np.int64)[order]
    c = np.asarray(cutoff, dtype="U10")[order]
    duplicate = (u[1:] == u[:-1]) & (c[1:] == c[:-1])
    if bool(duplicate.any()):
        raise AssertionError(f"duplicate (cutoff,user_id) rows: {int(duplicate.sum())}")


def _load_core() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    missing = [str(path) for path in CORE_COMPONENTS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing required raw OOF artifacts: {missing}")

    arrays: dict[str, np.ndarray] = {}
    entries: list[dict[str, Any]] = []
    base_uid = base_cutoff = base_y = None
    for name, path in CORE_COMPONENTS.items():
        data = np.load(path, allow_pickle=False)
        order = canonical_order(data["user_id"], data["cutoff"])
        uid = np.asarray(data["user_id"], dtype=np.int64)[order]
        cutoff = np.asarray(data["cutoff"], dtype="U10")[order]
        y = np.asarray(data["y"], dtype=float)[order]
        z = np.asarray(data["z"], dtype=float)[order]
        assert_unique_keys(uid, cutoff)
        if not np.isfinite(z).all() or not np.isfinite(y).all():
            raise AssertionError(f"{name}: non-finite target/prediction")
        if base_uid is None:
            base_uid, base_cutoff, base_y = uid, cutoff, y
        else:
            if not np.array_equal(uid, base_uid) or not np.array_equal(cutoff, base_cutoff):
                raise AssertionError(f"{name}: row keys do not match CAP")
            if not np.allclose(y, base_y, rtol=0.0, atol=1e-6):
                raise AssertionError(f"{name}: targets do not match CAP")
        arrays[name] = z
        entries.append({
            "name": name,
            "path": path.resolve(),
            "file_sha256": file_sha256(path),
            "prediction_sha256": array_sha256(z),
            "target_sha256": array_sha256(y),
            "row_key_sha256": array_sha256(cutoff, uid),
            "n": len(z),
            "prediction_dtype": str(data["z"].dtype),
        })

    assert base_uid is not None and base_cutoff is not None and base_y is not None
    arrays.update(user_id=base_uid, cutoff=base_cutoff, y=base_y)
    counts = [int(np.sum(base_cutoff == fold)) for fold in FOLDS]
    if counts != [188518, 191025, 193694, 197379]:
        raise AssertionError(f"unexpected fold sizes: {counts}")
    if sorted(np.unique(base_cutoff).tolist()) != list(FOLDS):
        raise AssertionError("unexpected fold set")
    manifest = {
        "core_components": entries,
        "n": len(base_uid),
        "folds": list(FOLDS),
        "fold_sizes": counts,
        "row_key_sha256": array_sha256(base_cutoff, base_uid),
        "target_sha256": array_sha256(base_y),
        "duplicates": 0,
        "missing_rows": 0,
        "finite_predictions": True,
        "alignment_key": ["cutoff", "user_id"],
    }
    return arrays, manifest


def _load_aligned_npz(path: Path, uid: np.ndarray, cutoff: np.ndarray, y: np.ndarray,
                      uid_key: str = "user_id", cutoff_key: str = "cutoff",
                      y_key: str = "y", z_key: str = "z") -> tuple[np.ndarray, dict[str, Any]]:
    data = np.load(path, allow_pickle=False)
    order = canonical_order(data[uid_key], data[cutoff_key])
    u = np.asarray(data[uid_key], dtype=np.int64)[order]
    c = np.asarray(data[cutoff_key], dtype="U10")[order]
    yy = np.asarray(data[y_key], dtype=float)[order]
    z = np.asarray(data[z_key], dtype=float)[order]
    if not np.array_equal(u, uid) or not np.array_equal(c, cutoff):
        raise AssertionError(f"{path.name}: row alignment failed")
    if not np.allclose(yy, y, rtol=0.0, atol=1e-6):
        raise AssertionError(f"{path.name}: target alignment failed")
    if not np.isfinite(z).all():
        raise AssertionError(f"{path.name}: non-finite prediction")
    return z, {
        "path": path.resolve(), "file_sha256": file_sha256(path),
        "prediction_sha256": array_sha256(z), "n": len(z),
    }


def _load_optional(frame: dict[str, np.ndarray], manifest: dict[str, Any]) -> None:
    entries = []
    for name, path in OPTIONAL_COMPONENTS.items():
        if not path.exists():
            continue
        z, entry = _load_aligned_npz(path, frame["user_id"], frame["cutoff"], frame["y"])
        frame[name] = z
        entry["name"] = name
        entries.append(entry)
    manifest["optional_components"] = entries

    fresh_path = ARTIFACTS / "oof_FRESH_CONTRAST_MOE.npz"
    if fresh_path.exists():
        data = np.load(fresh_path, allow_pickle=False)
        order = canonical_order(data["uid"], data["cutoff"])
        u = np.asarray(data["uid"], dtype=np.int64)[order]
        c = np.asarray(data["cutoff"], dtype="U10")[order]
        yy = np.asarray(data["y"], dtype=float)[order]
        if not (np.array_equal(u, frame["user_id"]) and np.array_equal(c, frame["cutoff"])):
            raise AssertionError("FRESH honest outer row alignment failed")
        if not np.allclose(yy, frame["y"], atol=1e-6, rtol=0.0):
            raise AssertionError("FRESH honest outer target alignment failed")
        frame["fresh_processed"] = np.asarray(data["fresh_processed_nested"], float)[order]
        if not np.allclose(np.asarray(data["z_base"], float)[order], frame["z_strong_raw"],
                           atol=5e-7, rtol=0.0):
            raise AssertionError("FRESH base is not exact STRONGEST_CURRENT")
        entries.append({"name": "fresh_processed_honest_outer", "path": fresh_path.resolve(),
                        "file_sha256": file_sha256(fresh_path),
                        "prediction_sha256": array_sha256(frame["fresh_processed"]),
                        "n": len(u)})

    btyd_path = ARTIFACTS / "BTYD_STABLE_EXP051" / "oof_raw.npz"
    if btyd_path.exists():
        data = np.load(btyd_path, allow_pickle=False)
        order = canonical_order(data["user_id"], data["cutoff"])
        u = np.asarray(data["user_id"], np.int64)[order]
        c = np.asarray(data["cutoff"], dtype="U10")[order]
        yy = np.asarray(data["y"], float)[order]
        if not (np.array_equal(u, frame["user_id"]) and np.array_equal(c, frame["cutoff"])):
            raise AssertionError("BTYD row alignment failed")
        if not np.allclose(yy, frame["y"], atol=1e-6, rtol=0.0):
            raise AssertionError("BTYD target alignment failed")
        frame["btyd"] = np.asarray(data["z_btyd"], float)[order]
        if not np.allclose(np.asarray(data["z_strongest"], float)[order], frame["z_strong_raw"],
                           atol=5e-7, rtol=0.0):
            raise AssertionError("BTYD base is not exact STRONGEST_CURRENT")
        entries.append({"name": "btyd_stable_oof", "path": btyd_path.resolve(),
                        "file_sha256": file_sha256(btyd_path),
                        "prediction_sha256": array_sha256(frame["btyd"]), "n": len(u)})
    manifest["optional_components"] = entries


def fold_calibrated(y: np.ndarray, z: np.ndarray, fold_index: np.ndarray,
                    scope: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project fold calibration: each fold receives its own optimal log shift."""
    y = np.asarray(y, float)
    z = np.asarray(z, float)
    fi = np.asarray(fold_index, np.int8)
    use = np.ones(len(y), bool) if scope is None else np.asarray(scope, bool)
    z_cal = np.full(len(y), np.nan, float)
    scores = np.full(4, np.nan, float)
    offsets = np.full(4, np.nan, float)
    for fold in range(4):
        mask = use & (fi == fold)
        if not mask.any():
            continue
        delta, score = calibrate(y[mask], z[mask])
        offsets[fold] = delta
        scores[fold] = score
        z_cal[mask] = np.maximum(z[mask] + delta, 0.0)
    return z_cal, scores, offsets


def weighted_fold_score(scores: np.ndarray, folds: Iterable[int] | None = None) -> float:
    idx = np.arange(4) if folds is None else np.asarray(list(folds), int)
    values = np.asarray(scores, float)[idx]
    weights = FOLD_WEIGHTS[idx]
    return float(np.average(values, weights=weights))


def reconstruct_strongest(cap: np.ndarray, unc: np.ndarray, dist: np.ndarray,
                          etx: np.ndarray, seq: np.ndarray) -> np.ndarray:
    return 0.10 * cap + 0.20 * unc + 0.25 * dist + 0.225 * etx + 0.225 * seq


def full_slot_weights(kind: str) -> dict[str, float]:
    if kind == "etx":
        return {"cap": 0.10, "unc": 0.20, "dist": 0.25, "etx": 0.45, "seq": 0.0}
    if kind == "seq":
        return {"cap": 0.10, "unc": 0.20, "dist": 0.25, "etx": 0.0, "seq": 0.45}
    if kind == "fixed":
        return {"cap": 0.10, "unc": 0.20, "dist": 0.25, "etx": 0.225, "seq": 0.225}
    raise ValueError(kind)


def gate_weight(probability: np.ndarray) -> np.ndarray:
    return 0.25 + 0.50 * np.asarray(probability, float)


def gate_assembly(frame: dict[str, np.ndarray], probability: np.ndarray) -> np.ndarray:
    w = gate_weight(probability)
    return (0.10 * frame["cap"] + 0.20 * frame["unc"] + 0.25 * frame["dist"]
            + 0.45 * (w * frame["etx"] + (1.0 - w) * frame["seq"]))


def _audit_baseline(frame: dict[str, np.ndarray], manifest: dict[str, Any]) -> None:
    fi = np.asarray([FOLDS.index(value) for value in frame["cutoff"]], np.int8)
    frame["fold_index"] = fi
    z = reconstruct_strongest(frame["cap"], frame["unc"], frame["dist"],
                              frame["etx"], frame["seq"])
    frame["z_strong_raw"] = z
    z_cal, scores, offsets = fold_calibrated(frame["y"], z, fi)
    wcv = weighted_fold_score(scores)
    if np.max(np.abs(scores - EXPECTED_FOLD_SCORES)) > 5e-7 or abs(wcv - EXPECTED_WCV) > 5e-7:
        raise AssertionError(f"STRONGEST reconstruction mismatch: folds={scores}, wCV={wcv}")
    frame["z_strong_cal"] = z_cal
    frame["r_strong"] = np.log1p(frame["y"]) - z_cal
    frame["loss_strong"] = np.square(frame["r_strong"])
    manifest["calibration_audit"] = {
        "fold_scores": scores, "fold_offsets": offsets, "wcv": wcv,
        "expected_fold_scores": EXPECTED_FOLD_SCORES, "expected_wcv": EXPECTED_WCV,
        "max_abs_fold_mismatch": float(np.max(np.abs(scores - EXPECTED_FOLD_SCORES))),
        "status": "PASS_EXACT",
        "strongest_prediction_sha256": array_sha256(z),
    }


def assert_no_future_feature_columns(names: Iterable[str]) -> None:
    forbidden = ("future", "target", "label", "lead_", "y30", "next_")
    bad = [name for name in names if any(token in name.lower() for token in forbidden)]
    if bad:
        raise AssertionError(f"future/target-like feature columns: {bad}")
    if "user_id" in set(names):
        raise AssertionError("user_id must not be a model feature")


def _load_state_features(frame: dict[str, np.ndarray], manifest: dict[str, Any]) -> tuple[pl.DataFrame, list[str]]:
    parts: list[pl.DataFrame] = []
    paths = []
    names: list[str] | None = None
    for fold in FOLDS:
        tag = fold.replace("-", "")
        path = ROOT / "data" / "processed" / f"feat_{tag}_LnormNone.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing existing S1-E10 feature cache: {path}")
        source = pl.read_parquet(path).with_columns(pl.lit(1, dtype=pl.Int8).alias("__found"))
        fold_uid = frame["user_id"][frame["cutoff"] == fold]
        joined = (pl.DataFrame({"user_id": fold_uid})
                  .join(source, on="user_id", how="left", validate="1:1")
                  .sort("user_id"))
        if joined.height != len(fold_uid) or joined["__found"].null_count():
            raise AssertionError(f"feature coverage failed on {fold}")
        current = [c for c in joined.columns if c not in {"user_id", "__found"}]
        if names is None:
            names = current
        elif current != names:
            raise AssertionError(f"feature order changed on {fold}")
        if not np.array_equal(joined["user_id"].to_numpy(), fold_uid):
            raise AssertionError(f"feature row order failed on {fold}")
        parts.append(joined.select(current))
        paths.append({"cutoff": fold, "path": path.resolve(), "file_sha256": file_sha256(path),
                      "rows_source": source.height, "rows_aligned": joined.height})
    assert names is not None
    if len(names) != 227:
        raise AssertionError(f"expected 227 S1-E10 features, found {len(names)}")
    assert_no_future_feature_columns(names)
    manifest["feature_caches"] = paths
    manifest["feature_manifest"] = {
        "n_features": len(names), "feature_names": names,
        "feature_order_sha256": array_sha256(np.asarray(names, dtype="U")),
        "generation_semantics": "existing feat_<cutoff>_LnormNone caches from build_features(cutoff)",
        "cutoff_safe": True, "user_id_is_feature": False,
    }
    return pl.concat(parts, how="vertical"), names


def _load_pact(frame: dict[str, np.ndarray], manifest: dict[str, Any]) -> None:
    p0 = np.empty(len(frame["y"]), float)
    entries = []
    for fold in FOLDS:
        path = ARTIFACTS / f"PACT_dist_{fold}.npz"
        data = np.load(path, allow_pickle=False)
        order = np.argsort(np.asarray(data["user_id"], dtype=np.int64), kind="mergesort")
        mask = frame["cutoff"] == fold
        uid = np.asarray(data["user_id"], dtype=np.int64)[order]
        if not np.array_equal(uid, frame["user_id"][mask]):
            raise AssertionError(f"PACT user alignment failed on {fold}")
        if not np.allclose(np.asarray(data["y"], float)[order], frame["y"][mask], atol=1e-6):
            raise AssertionError(f"PACT target alignment failed on {fold}")
        if not np.array_equal(np.asarray(data["p_act"], float),
                              1.0 - np.asarray(data["p0"], float)):
            raise AssertionError(f"PACT p_act identity failed on {fold}")
        if np.max(np.abs(np.asarray(data["z_ref"], float)[order] - frame["dist"][mask])) > 1e-6:
            raise AssertionError(f"PACT DIST identity failed on {fold}")
        p0[mask] = np.asarray(data["p0"], float)[order]
        entries.append({"cutoff": fold, "path": path.resolve(), "file_sha256": file_sha256(path),
                        "n": int(mask.sum())})
    if not np.all((p0 >= 0.0) & (p0 <= 1.0)):
        raise AssertionError("PACT p0 outside [0,1]")
    frame["p0"] = p0
    frame["p_act"] = 1.0 - p0
    manifest["pact_artifacts"] = entries


def _load_zero2d(frame: dict[str, np.ndarray], manifest: dict[str, Any]) -> None:
    """Rebuild the registered exact honest outer ZERO2D prediction artifact-only."""
    from src.zero2d_shrink import load_frame as zero_load_frame, run_nested

    source, audit = zero_load_frame()
    if not (np.array_equal(source["uid"], frame["user_id"])
            and np.array_equal(source["cutoff"], frame["cutoff"])):
        raise AssertionError("ZERO2D source alignment failed")
    result = run_nested(source, "ZERO2D")
    if abs(float(result["delta_wcv"]) - (-0.000024756)) > 2e-7:
        raise AssertionError("ZERO2D honest reconstruction mismatch")
    frame["zero2d"] = np.asarray(result["z_honest"], float)
    manifest["zero2d_reconstruction"] = {
        "source": "src.zero2d_shrink.load_frame + run_nested(method=ZERO2D)",
        "delta_wcv": result["delta_wcv"], "improved_folds": result["improved_folds"],
        "prediction_sha256": array_sha256(frame["zero2d"]),
        "base_wcv": audit["base_wcv"], "status": "PASS_EXACT",
    }


def build_disagreement_features(frame: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    components = np.column_stack([frame[k] for k in ("cap", "unc", "dist", "etx", "seq")])
    names = ["CAP", "UNC", "DIST", "ETX", "SEQ"]
    values: list[np.ndarray] = [frame[k] for k in ("cap", "unc", "dist", "etx", "seq")]
    out_names = ["pred_cap", "pred_unc", "pred_dist", "pred_etx", "pred_seq"]
    z_tab = (0.10 * frame["cap"] + 0.20 * frame["unc"] + 0.25 * frame["dist"]) / 0.55
    z_neural = 0.50 * frame["etx"] + 0.50 * frame["seq"]
    derived = {
        "pred_strongest": frame["z_strong_raw"],
        "etx_minus_seq": frame["etx"] - frame["seq"],
        "etx_minus_dist": frame["etx"] - frame["dist"],
        "seq_minus_dist": frame["seq"] - frame["dist"],
        "neural_mean_minus_tabular_mean": z_neural - z_tab,
        "component_prediction_mean": components.mean(axis=1),
        "component_prediction_std": components.std(axis=1),
        "component_prediction_range": np.ptp(components, axis=1),
        "component_prediction_median": np.median(components, axis=1),
        "component_prediction_min": components.min(axis=1),
        "component_prediction_max": components.max(axis=1),
        "component_max_minus_min": components.max(axis=1) - components.min(axis=1),
        "dist_p0": frame["p0"],
        "dist_p_act": frame["p_act"],
    }
    for key, value in derived.items():
        out_names.append(key)
        values.append(value)
    for i in range(5):
        for j in range(i + 1, 5):
            out_names.append(f"abs_{names[i].lower()}_minus_{names[j].lower()}")
            values.append(np.abs(components[:, i] - components[:, j]))
    ranks = np.argsort(np.argsort(components, axis=1), axis=1).astype(np.float32)
    for i, name in enumerate(names):
        out_names.append(f"component_rank_{name.lower()}")
        values.append(ranks[:, i])
    matrix = np.column_stack(values).astype(np.float32)
    if not np.isfinite(matrix).all():
        raise AssertionError("non-finite disagreement feature")
    return matrix, out_names


def add_analysis_objects(frame: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    fi, y = frame["fold_index"], frame["y"]
    frame["z_tab_raw"] = (0.10 * frame["cap"] + 0.20 * frame["unc"]
                          + 0.25 * frame["dist"]) / 0.55
    frame["z_neural_raw"] = 0.50 * frame["etx"] + 0.50 * frame["seq"]
    frame["z_full_etx"] = (0.10 * frame["cap"] + 0.20 * frame["unc"]
                           + 0.25 * frame["dist"] + 0.45 * frame["etx"])
    frame["z_full_seq"] = (0.10 * frame["cap"] + 0.20 * frame["unc"]
                           + 0.25 * frame["dist"] + 0.45 * frame["seq"])
    variants = ["z_strong_raw", "z_tab_raw", "z_neural_raw", "z_full_etx", "z_full_seq"]
    rows = []
    for name in variants:
        z_cal, scores, offsets = fold_calibrated(y, frame[name], fi)
        frame[name + "_cal"] = z_cal
        frame[name + "_loss"] = np.square(np.log1p(y) - z_cal)
        for fold, score, offset in zip(FOLDS, scores, offsets):
            rows.append({"variant": name, "fold": fold, "rmsle": score, "offset": offset})
        rows.append({"variant": name, "fold": "wCV", "rmsle": weighted_fold_score(scores),
                     "offset": np.nan})
    frame["adv_etx_vs_seq"] = frame["z_full_seq_loss"] - frame["z_full_etx_loss"]
    frame["winner_etx"] = (frame["adv_etx_vs_seq"] > 0.0).astype(np.int8)
    frame["abs_adv_etx_vs_seq"] = np.abs(frame["adv_etx_vs_seq"])
    frame["adv_neural_vs_tabular"] = frame["z_tab_raw_loss"] - frame["z_neural_raw_loss"]
    frame["winner_neural"] = (frame["adv_neural_vs_tabular"] > 0.0).astype(np.int8)
    return rows


def fixed_bins(values: np.ndarray, n_bins: int = 10) -> np.ndarray:
    values = np.asarray(values, float)
    out = np.full(len(values), -1, np.int16)
    idx = np.flatnonzero(np.isfinite(values))
    if not len(idx):
        return out
    order = idx[np.argsort(values[idx], kind="mergesort")]
    out[order] = np.minimum((np.arange(len(order)) * n_bins) // len(order), n_bins - 1)
    return out


def fixed_rec_bin(rec: np.ndarray) -> np.ndarray:
    return np.digitize(np.nan_to_num(np.asarray(rec, float), nan=1e9), [14.5, 60.5, 180.5]).astype(np.int8)


def fixed_buy_bin(days: np.ndarray) -> np.ndarray:
    return np.digitize(np.nan_to_num(np.asarray(days, float), nan=0.0), [1.5, 15.5]).astype(np.int8)


def build_strata(frame: dict[str, np.ndarray], state: pl.DataFrame) -> np.ndarray:
    decile = np.empty(len(frame["y"]), np.int16)
    for fold in range(4):
        mask = frame["fold_index"] == fold
        decile[mask] = fixed_bins(frame["z_strong_raw"][mask], 10)
    rec = state["rec_buy"].to_numpy().astype(float)
    buy = state["w180_days_buy"].to_numpy().astype(float)
    return (((frame["fold_index"].astype(np.int32) * 10 + decile) * 4
             + fixed_rec_bin(rec)) * 3 + fixed_buy_bin(buy)).astype(np.int32)


def permutation_within_strata(strata: np.ndarray, mask: np.ndarray,
                              seed: int = int(SEED)) -> np.ndarray:
    rows = np.flatnonzero(mask)
    result = np.arange(len(strata), dtype=np.int64)
    rng = np.random.default_rng(seed)
    for value in np.unique(strata[rows]):
        group_rows = rows[strata[rows] == value]
        result[group_rows] = rng.permutation(group_rows)
    return result


def shuffle_preserves_strata(values: np.ndarray, shuffled: np.ndarray,
                             strata: np.ndarray, mask: np.ndarray) -> bool:
    rows = np.flatnonzero(mask)
    for value in np.unique(strata[rows]):
        group = rows[strata[rows] == value]
        if not np.array_equal(np.sort(np.asarray(values)[group]),
                              np.sort(np.asarray(shuffled)[group])):
            return False
    return True


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3 or np.std(a[mask]) == 0 or np.std(b[mask]) == 0:
        return float("nan")
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3 or np.nanstd(a[mask]) == 0 or np.nanstd(b[mask]) == 0:
        return float("nan")
    return _safe_corr(rankdata(a[mask], method="average"), rankdata(b[mask], method="average"))


def residual_map(feature_matrix: np.ndarray, feature_names: list[str],
                 residual: np.ndarray, loss: np.ndarray, fold_index: np.ndarray
                 ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary, bins_rows = [], []
    total_loss = float(np.sum(loss))
    for j, name in enumerate(feature_names):
        x = np.asarray(feature_matrix[:, j], float)
        pearson_r = _safe_corr(x, residual)
        spearman_r = _spearman(x, residual)
        pearson_loss = _safe_corr(x, loss)
        fold_corr = [_safe_corr(x[fold_index == f], residual[fold_index == f]) for f in range(4)]
        sign = np.sign(pearson_r) if np.isfinite(pearson_r) else 0.0
        sign_consistency = int(sum(np.sign(value) == sign for value in fold_corr if np.isfinite(value)))
        summary.append({
            "feature": name, "pearson_signed_residual": pearson_r,
            "spearman_signed_residual": spearman_r, "pearson_squared_loss": pearson_loss,
            "fold_corr_0904": fold_corr[0], "fold_corr_0918": fold_corr[1],
            "fold_corr_1002": fold_corr[2], "fold_corr_1016": fold_corr[3],
            "fold_sign_consistency": sign_consistency,
            "predicts_signed_residual": bool(np.isfinite(pearson_r) and np.isfinite(spearman_r)
                                             and abs(pearson_r) >= 0.01
                                             and abs(spearman_r) >= 0.01
                                             and sign_consistency == 4),
            "predicts_loss_magnitude_only": bool(np.isfinite(pearson_loss)
                                                  and abs(pearson_loss) >= 0.02
                                                  and (not np.isfinite(pearson_r)
                                                       or abs(pearson_r) < 0.01)),
        })
        bins = fixed_bins(x, 10)
        for b in range(10):
            mask = bins == b
            if not mask.any():
                continue
            bins_rows.append({
                "feature": name, "decile": b + 1, "n": int(mask.sum()),
                "x_min": float(np.nanmin(x[mask])), "x_max": float(np.nanmax(x[mask])),
                "mean_residual": float(np.mean(residual[mask])),
                "mean_squared_loss": float(np.mean(loss[mask])),
                "rmsle_bin": float(np.sqrt(np.mean(loss[mask]))),
                "mse_share": float(np.sum(loss[mask]) / total_loss),
            })
    return summary, bins_rows


def segment_masks(frame: dict[str, np.ndarray], state: pl.DataFrame,
                  disagreement: np.ndarray, disagreement_names: list[str]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    y = frame["y"]
    rec = state["rec_buy"].to_numpy().astype(float)
    buy = state["w180_days_buy"].to_numpy().astype(float)
    never = ~np.isfinite(rec)
    abs_dis = np.abs(frame["etx"] - frame["seq"])
    q_dis = np.quantile(abs_dis, [1 / 3, 2 / 3])
    etx_seq = frame["etx"] - frame["seq"]
    q_sem = np.quantile(etx_seq, [0.1, 0.9])
    neu_tab = frame["z_neural_raw"] - frame["z_tab_raw"]
    q_nt = np.quantile(neu_tab, [0.1, 0.9])
    masks: dict[str, np.ndarray] = {
        "all": np.ones(len(y), bool), "y_zero": y == 0, "y_positive": y > 0,
        "rec_buy_15_60": np.isfinite(rec) & (rec >= 15) & (rec <= 60),
        "w180_days_buy_2_15": (buy >= 2) & (buy <= 15),
        "intersection": np.isfinite(rec) & (rec >= 15) & (rec <= 60) & (buy >= 2) & (buy <= 15),
        "never_buyer": never, "frequent_buyer": buy >= 16,
        "disagreement_low": abs_dis <= q_dis[0],
        "disagreement_medium": (abs_dis > q_dis[0]) & (abs_dis <= q_dis[1]),
        "disagreement_high": abs_dis > q_dis[1],
        "etx_much_greater_seq": etx_seq >= q_sem[1],
        "seq_much_greater_etx": etx_seq <= q_sem[0],
        "neural_much_greater_tabular": neu_tab >= q_nt[1],
        "tabular_much_greater_neural": neu_tab <= q_nt[0],
    }
    pred_decile = np.empty(len(y), np.int16)
    for fold in range(4):
        m = frame["fold_index"] == fold
        pred_decile[m] = fixed_bins(frame["z_strong_raw"][m], 10)
    for value in range(10):
        masks[f"prediction_decile_{value + 1:02d}"] = pred_decile == value
    thresholds = {
        "abs_etx_seq_disagreement_terciles": q_dis,
        "etx_minus_seq_p10_p90": q_sem,
        "neural_minus_tabular_p10_p90": q_nt,
    }
    return masks, thresholds


def segment_residual_table(masks: dict[str, np.ndarray], residual: np.ndarray,
                           loss: np.ndarray, fold_index: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for name, mask in masks.items():
        fold_mean = [float(np.mean(residual[mask & (fold_index == f)]))
                     if (mask & (fold_index == f)).any() else np.nan for f in range(4)]
        rows.append({
            "segment": name, "n": int(mask.sum()), "share": float(mask.mean()),
            "mean_residual": float(np.mean(residual[mask])),
            "rmsle": float(np.sqrt(np.mean(loss[mask]))),
            "mse_share": float(np.sum(loss[mask]) / np.sum(loss)),
            "mean_residual_0904": fold_mean[0], "mean_residual_0918": fold_mean[1],
            "mean_residual_1002": fold_mean[2], "mean_residual_1016": fold_mean[3],
            "fold_mean_sign_consistency": int(max(sum(v > 0 for v in fold_mean if np.isfinite(v)),
                                                  sum(v < 0 for v in fold_mean if np.isfinite(v)))),
        })
    return rows


def _oracle_fold_rows(name: str, baseline_loss: np.ndarray, oracle_loss: np.ndarray,
                      fold_index: np.ndarray) -> tuple[list[dict[str, Any]], float]:
    rows, deltas = [], []
    for fold in range(4):
        mask = fold_index == fold
        base = float(np.sqrt(np.mean(baseline_loss[mask])))
        score = float(np.sqrt(np.mean(oracle_loss[mask])))
        rows.append({"oracle": name, "fold": FOLDS[fold], "baseline": base,
                     "oracle_rmsle": score, "delta": score - base, "gain": base - score})
        deltas.append(score - base)
    delta = float(np.average(deltas, weights=FOLD_WEIGHTS))
    rows.append({"oracle": name, "fold": "wCV", "baseline": EXPECTED_WCV,
                 "oracle_rmsle": EXPECTED_WCV + delta, "delta": delta, "gain": -delta})
    return rows, -delta


def oracle_analysis(frame: dict[str, np.ndarray], masks: dict[str, np.ndarray]) -> dict[str, Any]:
    fi, y = frame["fold_index"], frame["y"]
    loss_fixed = frame["z_strong_raw_loss"]
    loss_etx, loss_seq = frame["z_full_etx_loss"], frame["z_full_seq_loss"]
    semantic_loss = np.minimum(loss_etx, loss_seq)
    semantic_rows, semantic_gain = _oracle_fold_rows("ETX_vs_SEQ", loss_fixed, semantic_loss, fi)
    advantage = loss_seq - loss_etx
    winner_rows = []
    for fold in range(4):
        m = fi == fold
        winner_rows.append({
            "fold": FOLDS[fold], "n": int(m.sum()),
            "share_etx_winner": float(np.mean(advantage[m] > 0)),
            "share_seq_winner": float(np.mean(advantage[m] < 0)),
            "mean_advantage": float(np.mean(advantage[m])),
            "median_advantage": float(np.median(advantage[m])),
            "p10_advantage": float(np.quantile(advantage[m], 0.10)),
            "p90_advantage": float(np.quantile(advantage[m], 0.90)),
        })

    fixed_tab = 0.10 * frame["cap"] + 0.20 * frame["unc"] + 0.25 * frame["dist"]
    grid_losses = []
    grid_rows = []
    for share in [0.0, 0.25, 0.50, 0.75, 1.0]:
        z = fixed_tab + 0.45 * (share * frame["etx"] + (1.0 - share) * frame["seq"])
        z_cal, scores, _ = fold_calibrated(y, z, fi)
        loss = np.square(np.log1p(y) - z_cal)
        grid_losses.append(loss)
        grid_rows.append({"etx_share": share, "wcv": weighted_fold_score(scores),
                          **{f"rmsle_{FOLDS[i]}": scores[i] for i in range(4)}})
    grid_stack = np.vstack(grid_losses)
    best_grid = np.argmin(grid_stack, axis=0)
    grid_oracle_rows, grid_gain = _oracle_fold_rows(
        "ETX_weight_grid", loss_fixed, np.min(grid_stack, axis=0), fi)
    for i, share in enumerate([0.0, 0.25, 0.50, 0.75, 1.0]):
        grid_rows[i]["row_oracle_share"] = float(np.mean(best_grid == i))

    seed_rows, seed_gains = [], {}
    seed_pair_targets = {}
    for family in ("etx", "seq"):
        keys = [f"{family}_s42", f"{family}_s43", f"{family}_s44"]
        if not all(key in frame for key in keys):
            continue
        losses = []
        for key in keys:
            z = fixed_tab + 0.45 * frame[key]
            z_cal, _, _ = fold_calibrated(y, z, fi)
            losses.append(np.square(np.log1p(y) - z_cal))
        avg_key = family
        avg_z = fixed_tab + 0.45 * frame[avg_key]
        avg_cal, _, _ = fold_calibrated(y, avg_z, fi)
        avg_loss = np.square(np.log1p(y) - avg_cal)
        rows, gain = _oracle_fold_rows(f"{family.upper()}_seed_42_43_44", avg_loss,
                                       np.min(np.vstack(losses), axis=0), fi)
        seed_rows.extend(rows)
        seed_gains[family] = gain
        seed_pair_targets[family] = {
            "advantage": losses[1] - losses[0],
            "winner": (losses[1] - losses[0] > 0).astype(np.int8),
        }
    seed_null = max(seed_gains.values()) if seed_gains else float("nan")
    semantic_excess = semantic_gain - seed_null if np.isfinite(seed_null) else float("nan")

    candidates: dict[str, np.ndarray] = {
        "STRONGEST": frame["z_strong_raw"],
        "FULL_ETX": frame["z_full_etx"], "FULL_SEQ": frame["z_full_seq"],
        "DIST_FULL": fixed_tab + 0.45 * frame["dist"],
    }
    if "fresh_processed" in frame:
        candidates["FRESH"] = frame["z_strong_raw"] + frame["fresh_processed"]
    if "btyd" in frame:
        candidates["BTYD05"] = 0.95 * frame["z_strong_raw"] + 0.05 * frame["btyd"]
    if "zero2d" in frame:
        candidates["ZERO2D"] = frame["zero2d"]
    if "ridge" in frame:
        candidates["RIDGE_CONTROL"] = (0.10 * frame["cap"] + 0.125 * frame["unc"]
                                         + 0.175 * frame["dist"] + 0.15 * frame["ridge"]
                                         + 0.225 * frame["etx"] + 0.225 * frame["seq"])
    candidate_losses, best_rows = [], []
    for name, z in candidates.items():
        z_cal, scores, _ = fold_calibrated(y, z, fi)
        candidate_losses.append(np.square(np.log1p(y) - z_cal))
        best_rows.append({"candidate": name, "wcv": weighted_fold_score(scores),
                          **{f"rmsle_{FOLDS[i]}": scores[i] for i in range(4)}})
    stack = np.vstack(candidate_losses)
    selected = np.argmin(stack, axis=0)
    best_loss = np.min(stack, axis=0)
    best_oracle_rows, best_gain = _oracle_fold_rows("best_existing", loss_fixed, best_loss, fi)
    for i, row in enumerate(best_rows):
        row["oracle_selected_share"] = float(np.mean(selected == i))

    oracle_segment_rows = []
    for name, mask in masks.items():
        if not mask.any():
            continue
        for oracle_name, loss in (("ETX_vs_SEQ", semantic_loss), ("best_existing", best_loss)):
            base = float(np.sqrt(np.mean(loss_fixed[mask])))
            score = float(np.sqrt(np.mean(loss[mask])))
            oracle_segment_rows.append({"oracle": oracle_name, "segment": name,
                                        "n": int(mask.sum()), "baseline": base,
                                        "oracle_rmsle": score, "gain": base - score})
    return {
        "semantic_rows": semantic_rows, "winner_rows": winner_rows,
        "weight_grid_rows": grid_rows, "weight_grid_oracle_rows": grid_oracle_rows,
        "seed_rows": seed_rows, "seed_gains": seed_gains,
        "seed_pair_targets": seed_pair_targets,
        "best_existing_rows": best_rows, "best_existing_oracle_rows": best_oracle_rows,
        "segment_rows": oracle_segment_rows,
        "summary": {
            "semantic_gain_wcv": semantic_gain, "weight_grid_gain_wcv": grid_gain,
            "seed_null_gain_wcv": seed_null, "semantic_excess_wcv": semantic_excess,
            "best_existing_gain_wcv": best_gain,
            "share_etx_winner": float(np.mean(advantage > 0)),
            "share_seq_winner": float(np.mean(advantage < 0)),
            "mean_advantage": float(np.mean(advantage)),
            "median_advantage": float(np.median(advantage)),
            "advantage_quantiles": np.quantile(advantage, [0.01, 0.1, 0.5, 0.9, 0.99]),
        },
    }


def _lgb_params(objective: str) -> dict[str, Any]:
    params = {k: v for k, v in PROBE_PARAMS.items()
              if k not in {"num_boost_round", "early_stopping"}}
    params.update(objective=objective, verbosity=-1,
                  num_threads=min(12, os.cpu_count() or 1))
    if objective == "binary":
        params["metric"] = "binary_logloss"
    else:
        params["metric"] = "l1" if objective == "regression_l1" else "l2"
    return params


def fit_probe(X: np.ndarray, y: np.ndarray, mask: np.ndarray, objective: str,
              weights: np.ndarray | None = None) -> lgb.Booster:
    train_x = np.asarray(X[mask], dtype=np.float32)
    train_y = np.asarray(y[mask], dtype=np.float32)
    train_w = None if weights is None else np.asarray(weights[mask], dtype=np.float32)
    dataset = lgb.Dataset(train_x, label=train_y, weight=train_w, free_raw_data=True)
    return lgb.train(_lgb_params(objective), dataset,
                     num_boost_round=int(PROBE_PARAMS["num_boost_round"]))


def classifier_metrics(y_true: np.ndarray, probability: np.ndarray,
                       advantage: np.ndarray, lo: float, hi: float) -> dict[str, float]:
    y_true = np.asarray(y_true, np.int8)
    probability = np.clip(np.asarray(probability, float), 0.0, 1.0)
    weights = np.clip(np.abs(np.asarray(advantage, float)), lo, hi)
    return {
        "auc": float(roc_auc_score(y_true, probability)),
        "weighted_auc": float(roc_auc_score(y_true, probability, sample_weight=weights)),
        "brier": float(brier_score_loss(y_true, probability)),
        "advantage_weighted_accuracy": float(np.average((probability >= 0.5) == y_true,
                                                          weights=weights)),
        "mean_probability": float(np.mean(probability)),
        "winner_rate": float(np.mean(y_true)),
    }


def probability_calibration_rows(probability: np.ndarray, winner: np.ndarray,
                                 direction: str, feature_set: str, control: str) -> list[dict[str, Any]]:
    bins = np.minimum((np.clip(probability, 0, 1) * 10).astype(int), 9)
    rows = []
    for value in range(10):
        mask = bins == value
        if mask.any():
            rows.append({"direction": direction, "feature_set": feature_set,
                         "control": control, "probability_bin": value,
                         "n": int(mask.sum()), "mean_probability": float(np.mean(probability[mask])),
                         "winner_rate": float(np.mean(winner[mask]))})
    return rows


def train_winner_probe(X: np.ndarray, feature_names: list[str], frame: dict[str, np.ndarray],
                       strata: np.ndarray, winner: np.ndarray, advantage: np.ndarray,
                       feature_set: str, shuffled: bool = False) -> tuple[np.ndarray, list[dict[str, Any]],
                                                                          list[dict[str, Any]], list[dict[str, Any]]]:
    fi, side = frame["fold_index"], user_group(frame["user_id"])
    late_prediction = np.full(len(frame["y"]), np.nan, float)
    metric_rows, cal_rows, audit_rows = [], [], []
    for donor_side in (0, 1):
        recipient_side = 1 - donor_side
        train = (fi < 3) & (side == donor_side)
        evaluate = (fi == 3) & (side == recipient_side)
        if np.intersect1d(np.unique(frame["user_id"][train]),
                          np.unique(frame["user_id"][evaluate])).size:
            raise AssertionError("recipient user leaked into winner training labels")
        train_target = np.asarray(winner, np.int8).copy()
        train_adv = np.asarray(advantage, float).copy()
        if shuffled:
            permutation = permutation_within_strata(strata, train)
            train_target[train] = train_target[permutation[train]]
            train_adv[train] = train_adv[permutation[train]]
            if not shuffle_preserves_strata(winner, train_target, strata, train):
                raise AssertionError("winner shuffle changed a stratum distribution")
            if not shuffle_preserves_strata(advantage, train_adv, strata, train):
                raise AssertionError("advantage shuffle changed a stratum distribution")
        lo, hi = np.quantile(np.abs(train_adv[train]), [0.10, 0.99])
        sample_weight = np.zeros(len(train_adv), float)
        sample_weight[train] = np.clip(np.abs(train_adv[train]), lo, hi)
        booster = fit_probe(X, train_target, train, "binary", sample_weight)
        probability = booster.predict(np.asarray(X[evaluate], np.float32))
        late_prediction[evaluate] = probability
        metrics = classifier_metrics(winner[evaluate], probability, advantage[evaluate], lo, hi)
        direction = f"{donor_side}->{recipient_side}"
        metric_rows.append({"target": "semantic_etx_vs_seq", "direction": direction,
                            "feature_set": feature_set, "control": "SHUFFLED" if shuffled else "REAL",
                            "n_train": int(train.sum()), "n_eval": int(evaluate.sum()),
                            "weight_p10": lo, "weight_p99": hi, **metrics})
        cal_rows.extend(probability_calibration_rows(probability, winner[evaluate], direction,
                                                     feature_set, "SHUFFLED" if shuffled else "REAL"))
        audit_rows.append({"direction": direction, "feature_set": feature_set,
                           "control": "SHUFFLED" if shuffled else "REAL",
                           "donor_side": donor_side, "recipient_side": recipient_side,
                           "recipient_user_overlap": 0, "shuffle_strata_preserved": bool(shuffled),
                           "sample_weight_sum": float(sample_weight[train].sum())})
    late = fi == 3
    if np.isnan(late_prediction[late]).any():
        raise AssertionError("winner probe did not cover the full late fold")
    lo, hi = np.quantile(np.abs(advantage[(fi < 3)]), [0.10, 0.99])
    combined = classifier_metrics(winner[late], late_prediction[late], advantage[late], lo, hi)
    metric_rows.append({"target": "semantic_etx_vs_seq", "direction": "combined",
                        "feature_set": feature_set, "control": "SHUFFLED" if shuffled else "REAL",
                        "n_train": int((fi < 3).sum()), "n_eval": int(late.sum()),
                        "weight_p10": lo, "weight_p99": hi, **combined})
    return late_prediction, metric_rows, cal_rows, audit_rows


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target, prediction = np.asarray(target, float), np.asarray(prediction, float)
    weights = np.abs(target)
    return {
        "pearson": _safe_corr(target, prediction), "spearman": _spearman(target, prediction),
        "mae": float(np.mean(np.abs(target - prediction))),
        "rmse": float(np.sqrt(np.mean(np.square(target - prediction)))),
        "sign_accuracy": float(np.mean(np.sign(target) == np.sign(prediction))),
        "advantage_weighted_sign_accuracy": float(np.average(
            np.sign(target) == np.sign(prediction), weights=np.maximum(weights, 1e-12))),
        "prediction_std": float(np.std(prediction)),
    }


def train_advantage_probe(X: np.ndarray, frame: dict[str, np.ndarray], strata: np.ndarray,
                          shuffled: bool = False) -> tuple[np.ndarray, list[dict[str, Any]]]:
    fi, side = frame["fold_index"], user_group(frame["user_id"])
    target = frame["adv_etx_vs_seq"]
    late_prediction = np.full(len(target), np.nan, float)
    rows = []
    for donor_side in (0, 1):
        recipient_side = 1 - donor_side
        train = (fi < 3) & (side == donor_side)
        evaluate = (fi == 3) & (side == recipient_side)
        train_target = target.copy()
        if shuffled:
            permutation = permutation_within_strata(strata, train)
            train_target[train] = train_target[permutation[train]]
            if not shuffle_preserves_strata(target, train_target, strata, train):
                raise AssertionError("advantage shuffle changed a stratum distribution")
        booster = fit_probe(X, train_target, train, "regression_l1")
        prediction = booster.predict(np.asarray(X[evaluate], np.float32))
        late_prediction[evaluate] = prediction
        rows.append({"direction": f"{donor_side}->{recipient_side}",
                     "control": "SHUFFLED" if shuffled else "REAL",
                     "n_train": int(train.sum()), "n_eval": int(evaluate.sum()),
                     **regression_metrics(target[evaluate], prediction)})
    late = fi == 3
    rows.append({"direction": "combined", "control": "SHUFFLED" if shuffled else "REAL",
                 "n_train": int((fi < 3).sum()), "n_eval": int(late.sum()),
                 **regression_metrics(target[late], late_prediction[late])})
    return late_prediction, rows


def select_scale_without_late(y: np.ndarray, z_base: np.ndarray, correction: np.ndarray,
                              fold_index: np.ndarray, donor_mask: np.ndarray,
                              scales: np.ndarray = SCALES) -> tuple[float, list[dict[str, Any]]]:
    curves = []
    for scale in scales:
        scores = []
        for fold in range(3):
            mask = donor_mask & (fold_index == fold)
            _, score = calibrate(y[mask], z_base[mask] + scale * correction[mask])
            scores.append(score)
        value = float(np.average(scores, weights=FOLD_WEIGHTS[:3]))
        curves.append({"scale": float(scale), "selection_wcv_3f": value,
                       "fold_scores": scores})
    best = min(row["selection_wcv_3f"] for row in curves)
    selected = min(row["scale"] for row in curves
                   if row["selection_wcv_3f"] <= best + 1e-5)
    return float(selected), curves


def train_residual_probe(X: np.ndarray, frame: dict[str, np.ndarray], strata: np.ndarray,
                         shuffled: bool = False) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    fi, side = frame["fold_index"], user_group(frame["user_id"])
    residual = frame["r_strong"]
    late_correction = np.full(len(residual), np.nan, float)
    selection_rows, audit_rows = [], []
    for donor_side in (0, 1):
        recipient_side = 1 - donor_side
        donor = (fi < 3) & (side == donor_side)
        recipient = (fi == 3) & (side == recipient_side)
        centered = residual.copy()
        for fold in range(3):
            mask = donor & (fi == fold)
            centered[mask] -= float(np.mean(centered[mask]))
        if shuffled:
            permutation = permutation_within_strata(strata, donor)
            shuffled_target = centered.copy()
            shuffled_target[donor] = centered[permutation[donor]]
            if not shuffle_preserves_strata(centered, shuffled_target, strata, donor):
                raise AssertionError("residual shuffle changed a stratum distribution")
            centered = shuffled_target

        donor_oof = np.full(len(residual), np.nan, float)
        for held in range(3):
            train = donor & (fi != held)
            valid = donor & (fi == held)
            booster = fit_probe(X, centered, train, "regression_l1")
            donor_oof[valid] = booster.predict(np.asarray(X[valid], np.float32))
        if np.isnan(donor_oof[donor]).any():
            raise AssertionError("donor residual OOF is incomplete")
        winsor_lo, winsor_hi = np.quantile(donor_oof[donor], [0.01, 0.99])
        donor_processed = np.zeros(len(residual), float)
        donor_processed[donor] = np.clip(donor_oof[donor], winsor_lo, winsor_hi)
        for fold in range(3):
            mask = donor & (fi == fold)
            donor_processed[mask] -= float(np.mean(donor_processed[mask]))
        selected, curve = select_scale_without_late(
            frame["y"], frame["z_strong_raw"], donor_processed, fi, donor)

        final_model = fit_probe(X, centered, donor, "regression_l1")
        recipient_prediction = final_model.predict(np.asarray(X[recipient], np.float32))
        recipient_prediction = np.clip(recipient_prediction, winsor_lo, winsor_hi)
        recipient_prediction -= float(np.mean(recipient_prediction))
        late_correction[recipient] = selected * recipient_prediction
        direction = f"{donor_side}->{recipient_side}"
        for row in curve:
            selection_rows.append({"direction": direction,
                                   "control": "SHUFFLED" if shuffled else "REAL",
                                   "selected": row["scale"] == selected, **row})
        audit_rows.append({
            "direction": direction, "control": "SHUFFLED" if shuffled else "REAL",
            "n_donor": int(donor.sum()), "n_recipient": int(recipient.sum()),
            "winsor_p01": winsor_lo, "winsor_p99": winsor_hi,
            "selected_scale": selected,
            "recipient_prediction_mean_after_center": float(np.mean(recipient_prediction)),
            "recipient_user_overlap": int(np.intersect1d(
                np.unique(frame["user_id"][donor]), np.unique(frame["user_id"][recipient])).size),
        })
    late = fi == 3
    if np.isnan(late_correction[late]).any():
        raise AssertionError("residual probe did not cover the full late fold")
    return late_correction, selection_rows, audit_rows


def train_seed_winner_controls(X: np.ndarray, frame: dict[str, np.ndarray], strata: np.ndarray,
                               seed_targets: dict[str, dict[str, np.ndarray]]) -> list[dict[str, Any]]:
    rows = []
    for family, target in seed_targets.items():
        _, metric_rows, _, _ = train_winner_probe(
            X, [], frame, strata, target["winner"], target["advantage"],
            feature_set=f"COMBINED_SEED_CONTROL_{family.upper()}", shuffled=False)
        for row in metric_rows:
            row["target"] = f"{family}_seed42_vs_seed43"
        rows.extend(metric_rows)
    return rows


def evaluate_candidate_scope(frame: dict[str, np.ndarray], z_candidate: np.ndarray,
                             correction: np.ndarray, mask: np.ndarray,
                             candidate: str, scope: str) -> dict[str, Any]:
    y = frame["y"][mask]
    z_base = frame["z_strong_raw"][mask]
    z_new = np.asarray(z_candidate, float)[mask]
    d_base, base_score = calibrate(y, z_base)
    d_new, new_score = calibrate(y, z_new)
    base_cal = np.maximum(z_base + d_base, 0.0)
    new_cal = np.maximum(z_new + d_new, 0.0)
    residual = np.log1p(y) - base_cal
    corr = _safe_corr(np.asarray(correction, float)[mask], residual)
    return {
        "candidate": candidate, "scope": scope, "n": int(mask.sum()),
        "base_rmsle": base_score, "candidate_rmsle": new_score,
        "delta_rmsle": new_score - base_score,
        "base_mse": float(np.mean(np.square(np.log1p(y) - base_cal))),
        "candidate_mse": float(np.mean(np.square(np.log1p(y) - new_cal))),
        "delta_mse": float(np.mean(np.square(np.log1p(y) - new_cal))
                           - np.mean(np.square(np.log1p(y) - base_cal))),
        "base_offset": d_base, "candidate_offset": d_new,
        "variance_correction": float(np.var(np.asarray(correction, float)[mask])),
        "mean_correction": float(np.mean(np.asarray(correction, float)[mask])),
        "corr_correction_residual": corr,
    }


def candidate_metrics(frame: dict[str, np.ndarray], candidates: dict[str, tuple[np.ndarray, np.ndarray]]) -> list[dict[str, Any]]:
    fi, side = frame["fold_index"], user_group(frame["user_id"])
    late = fi == 3
    scopes = {"combined": late, "recipient_side_0": late & (side == 0),
              "recipient_side_1": late & (side == 1)}
    rows = []
    for name, (z, correction) in candidates.items():
        for scope, mask in scopes.items():
            rows.append(evaluate_candidate_scope(frame, z, correction, mask, name, scope))
    return rows


def segment_candidate_table(frame: dict[str, np.ndarray], candidates: dict[str, tuple[np.ndarray, np.ndarray]],
                            masks: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    late = frame["fold_index"] == 3
    y = frame["y"]
    d_base, _ = calibrate(y[late], frame["z_strong_raw"][late])
    base_cal = np.maximum(frame["z_strong_raw"] + d_base, 0.0)
    rows = []
    for candidate, (z, _) in candidates.items():
        d_new, _ = calibrate(y[late], z[late])
        z_cal = np.maximum(z + d_new, 0.0)
        for segment, segment_mask in masks.items():
            mask = late & segment_mask
            if not mask.any():
                continue
            base = float(np.sqrt(np.mean(np.square(np.log1p(y[mask]) - base_cal[mask]))))
            score = float(np.sqrt(np.mean(np.square(np.log1p(y[mask]) - z_cal[mask]))))
            rows.append({"candidate": candidate, "segment": segment, "n": int(mask.sum()),
                         "share_late": float(mask.sum() / late.sum()), "base_rmsle": base,
                         "candidate_rmsle": score, "delta_rmsle": score - base})
    return rows


def decision_verdict(metrics: list[dict[str, Any]], segment_rows: list[dict[str, Any]],
                     oracle_summary: dict[str, Any]) -> dict[str, Any]:
    by = {(row["candidate"], row["scope"]): row for row in metrics}
    seg = {(row["candidate"], row["segment"]): row for row in segment_rows}
    pairs = [("GATE_REAL", "GATE_SHUFFLED"), ("RESIDUAL_REAL", "RESIDUAL_SHUFFLED")]
    checks = {}
    strong, moderate, weak = [], [], []
    for real, shuffled in pairs:
        combined = by[(real, "combined")]
        halves = [by[(real, "recipient_side_0")], by[(real, "recipient_side_1")]]
        control = by[(shuffled, "combined")]
        delta = combined["delta_rmsle"]
        control_gap = delta - control["delta_rmsle"]
        both_halves = all(row["delta_rmsle"] < 0 for row in halves)
        both_corr = all(row["corr_correction_residual"] > 0 for row in halves)
        broad = (seg.get((real, "y_zero"), {"delta_rmsle": 1})["delta_rmsle"] < 0
                 and seg.get((real, "y_positive"), {"delta_rmsle": 1})["delta_rmsle"] < 0)
        current = {
            "late_delta": delta, "both_halves_improve": both_halves,
            "real_minus_shuffled": control_gap, "positive_corr_both_halves": both_corr,
            "broad_zero_and_positive_gain": broad,
            "semantic_oracle_excess": oracle_summary["semantic_excess_wcv"],
        }
        checks[real] = current
        if (delta <= -0.0010 and both_halves and control_gap <= -0.0007 and both_corr
                and broad and oracle_summary["semantic_excess_wcv"] >= 0.003):
            strong.append(real)
        if (-0.0010 < delta <= -0.0007 and both_halves and control_gap <= -0.0007):
            moderate.append(real)
        if (-0.0007 < delta <= -0.0003 and both_halves and control_gap < -0.0002):
            weak.append(real)
    if strong:
        verdict = "STRONG"
    elif moderate:
        verdict = "MODERATE"
    elif weak and oracle_summary["semantic_excess_wcv"] >= 0.001:
        verdict = "WEAK"
    else:
        verdict = "NONE"
    late_pass = bool(strong or moderate)
    return {"verdict": verdict, "candidate_checks": checks, "late_fold_pass": late_pass,
            "promote_to_full_artifact_lofo": "YES" if verdict == "STRONG" else "NO"}


def add_past_residual(frame: dict[str, np.ndarray]) -> np.ndarray:
    """Only 09-04 can be safe at 10-16; donor folds have no supported past residual."""
    out = np.full(len(frame["y"]), np.nan, np.float32)
    donor_fold = FOLDS[0]
    donor_end = dt.date.fromisoformat(donor_fold) + dt.timedelta(days=30)
    recipient_fold = FOLDS[3]
    if donor_end > dt.date.fromisoformat(recipient_fold):
        raise AssertionError("past residual target end is after recipient cutoff")
    donor = frame["cutoff"] == donor_fold
    recipient = frame["cutoff"] == recipient_fold
    mapping = dict(zip(frame["user_id"][donor].tolist(), frame["r_strong"][donor].tolist()))
    out[recipient] = np.asarray([mapping.get(int(uid), np.nan) for uid in frame["user_id"][recipient]],
                               np.float32)
    return out


def save_aligned_frame(frame: dict[str, np.ndarray], state: pl.DataFrame,
                       disagreement: np.ndarray, disagreement_names: list[str]) -> None:
    columns = {
        "cutoff": frame["cutoff"], "user_id": frame["user_id"], "fold": frame["fold_index"],
        "y_true": frame["y"].astype(np.float32),
        "z_cap": frame["cap"].astype(np.float32), "z_unc": frame["unc"].astype(np.float32),
        "z_dist": frame["dist"].astype(np.float32), "z_etx_avg3": frame["etx"].astype(np.float32),
        "z_seq_avg3": frame["seq"].astype(np.float32),
        "z_strong_raw": frame["z_strong_raw"].astype(np.float32),
        "z_strong_calibrated": frame["z_strong_cal"].astype(np.float32),
        "r_strong": frame["r_strong"].astype(np.float32),
        "loss_strong": frame["loss_strong"].astype(np.float32),
        "adv_etx_vs_seq": frame["adv_etx_vs_seq"].astype(np.float32),
        "adv_neural_vs_tabular": frame["adv_neural_vs_tabular"].astype(np.float32),
        "winner_etx": frame["winner_etx"], "p0_dist": frame["p0"].astype(np.float32),
        "p_act_dist": frame["p_act"].astype(np.float32),
        "past_residual_0904": frame["past_residual_0904"],
    }
    for name in OPTIONAL_COMPONENTS:
        if name in frame:
            columns[f"z_{name}"] = frame[name].astype(np.float32)
    for name in ("z_full_etx", "z_full_seq", "z_tab_raw", "z_neural_raw"):
        columns[name] = frame[name].astype(np.float32)
    base = pl.DataFrame(columns)
    dis = pl.DataFrame(disagreement, schema=disagreement_names, orient="row")
    base.hstack(state).hstack(dis).write_parquet(
        OUT_ARTIFACTS / "aligned_oof.parquet", compression="zstd", compression_level=3)


def train_all_probes(frame: dict[str, np.ndarray], state: pl.DataFrame,
                     state_names: list[str], disagreement: np.ndarray,
                     disagreement_names: list[str], strata: np.ndarray,
                     oracle: dict[str, Any]) -> dict[str, Any]:
    state_x = state.to_numpy().astype(np.float32)
    state_x[~np.isfinite(state_x)] = np.nan
    combined_x = np.column_stack([state_x, disagreement]).astype(np.float32)
    feature_sets = {
        "DISAGREEMENT_ONLY": (disagreement, disagreement_names),
        "STATE_ONLY": (state_x, state_names),
        "COMBINED": (combined_x, state_names + disagreement_names),
    }
    winner_rows, winner_cal, split_audit = [], [], []
    primary_real = primary_shuf = None
    for name, (X, names) in feature_sets.items():
        assert_no_future_feature_columns(names)
        pred, metrics, cal, audit = train_winner_probe(
            X, names, frame, strata, frame["winner_etx"], frame["adv_etx_vs_seq"],
            feature_set=name, shuffled=False)
        winner_rows.extend(metrics); winner_cal.extend(cal); split_audit.extend(audit)
        if name == "COMBINED":
            primary_real = pred
    primary_shuf, metrics, cal, audit = train_winner_probe(
        combined_x, state_names + disagreement_names, frame, strata,
        frame["winner_etx"], frame["adv_etx_vs_seq"], "COMBINED", shuffled=True)
    winner_rows.extend(metrics); winner_cal.extend(cal); split_audit.extend(audit)
    assert primary_real is not None and primary_shuf is not None

    adv_real, advantage_rows_real = train_advantage_probe(combined_x, frame, strata, False)
    adv_shuf, advantage_rows_shuf = train_advantage_probe(combined_x, frame, strata, True)
    residual_real, residual_scale_real, residual_audit_real = train_residual_probe(
        combined_x, frame, strata, False)
    residual_shuf, residual_scale_shuf, residual_audit_shuf = train_residual_probe(
        combined_x, frame, strata, True)
    seed_rows = train_seed_winner_controls(combined_x, frame, strata,
                                           oracle["seed_pair_targets"])

    late = frame["fold_index"] == 3
    probability_real = np.where(late, primary_real, 0.5)
    probability_shuf = np.where(late, primary_shuf, 0.5)
    z_gate_real = gate_assembly(frame, probability_real)
    z_gate_shuf = gate_assembly(frame, probability_shuf)
    z_residual_real = frame["z_strong_raw"] + np.nan_to_num(residual_real, nan=0.0)
    z_residual_shuf = frame["z_strong_raw"] + np.nan_to_num(residual_shuf, nan=0.0)
    candidates = {
        "GATE_REAL": (z_gate_real, z_gate_real - frame["z_strong_raw"]),
        "GATE_SHUFFLED": (z_gate_shuf, z_gate_shuf - frame["z_strong_raw"]),
        "RESIDUAL_REAL": (z_residual_real, np.nan_to_num(residual_real, nan=0.0)),
        "RESIDUAL_SHUFFLED": (z_residual_shuf, np.nan_to_num(residual_shuf, nan=0.0)),
    }
    return {
        "feature_sets": {name: names for name, (_, names) in feature_sets.items()},
        "winner_rows": winner_rows, "winner_calibration_rows": winner_cal,
        "split_audit_rows": split_audit, "advantage_rows": advantage_rows_real + advantage_rows_shuf,
        "residual_scale_rows": residual_scale_real + residual_scale_shuf,
        "residual_audit_rows": residual_audit_real + residual_audit_shuf,
        "seed_winner_rows": seed_rows, "candidates": candidates,
        "late_arrays": {
            "p_etx_real": primary_real[late], "p_etx_shuffled": primary_shuf[late],
            "advantage_prediction_real": adv_real[late],
            "advantage_prediction_shuffled": adv_shuf[late],
            "residual_correction_real": residual_real[late],
            "residual_correction_shuffled": residual_shuf[late],
            "z_gate_real": z_gate_real[late], "z_gate_shuffled": z_gate_shuf[late],
            "z_residual_real": z_residual_real[late], "z_residual_shuffled": z_residual_shuf[late],
        },
    }


def _probe_output_path() -> Path:
    return OUT_ARTIFACTS / "probe_outputs.npz"


def save_probe_outputs(frame: dict[str, np.ndarray], probes: dict[str, Any]) -> None:
    late = frame["fold_index"] == 3
    arrays = {
        "late_user_id": frame["user_id"][late], "late_side": user_group(frame["user_id"])[late],
        "late_y": frame["y"][late].astype(np.float32),
        "late_z_strong": frame["z_strong_raw"][late].astype(np.float32),
        "late_r_strong": frame["r_strong"][late].astype(np.float32),
    }
    arrays.update({key: np.asarray(value, np.float32) for key, value in probes["late_arrays"].items()})
    np.savez_compressed(_probe_output_path(), **arrays)


def load_cached_probes(frame: dict[str, np.ndarray], metadata: dict[str, Any]) -> dict[str, Any]:
    data = np.load(_probe_output_path(), allow_pickle=False)
    late = frame["fold_index"] == 3
    if not np.array_equal(data["late_user_id"], frame["user_id"][late]):
        raise AssertionError("cached probe row alignment failed")
    arrays = {key: np.asarray(data[key], float) for key in data.files
              if key not in {"late_user_id", "late_side", "late_y", "late_z_strong", "late_r_strong"}}
    full = {}
    for key, value in arrays.items():
        full_value = np.full(len(frame["y"]), np.nan, float)
        full_value[late] = value
        full[key] = full_value
    candidates = {
        "GATE_REAL": (full["z_gate_real"], full["z_gate_real"] - frame["z_strong_raw"]),
        "GATE_SHUFFLED": (full["z_gate_shuffled"], full["z_gate_shuffled"] - frame["z_strong_raw"]),
        "RESIDUAL_REAL": (full["z_residual_real"], full["residual_correction_real"]),
        "RESIDUAL_SHUFFLED": (full["z_residual_shuffled"], full["residual_correction_shuffled"]),
    }
    result = dict(metadata)
    result["candidates"] = candidates
    result["late_arrays"] = arrays
    return result


def save_late_predictions(frame: dict[str, np.ndarray], probes: dict[str, Any]) -> None:
    late = frame["fold_index"] == 3
    data = np.load(_probe_output_path(), allow_pickle=False)
    pl.DataFrame({key: data[key] for key in data.files}).write_parquet(
        OUT_ARTIFACTS / "real_shuffled_predictions.parquet", compression="zstd")


def canonical_output_hashes() -> dict[str, str]:
    files = [
        "calibration_audit.csv", "residual_map.csv", "residual_map_bins.csv",
        "segment_residual_map.csv", "disagreement_summary.csv",
        "oracle_etx_seq.csv", "oracle_weight_grid.csv",
        "seed_null_oracle.csv", "best_existing_oracle.csv", "winner_probe_metrics.csv",
        "advantage_probe_metrics.csv", "recipient_half_metrics.csv",
        "segment_candidate_metrics.csv", "late_fold_candidates.csv", "summary.json",
    ]
    return {name: file_sha256(RESULTS / name) for name in files}


def analyze_and_write(frame: dict[str, np.ndarray], state: pl.DataFrame, state_names: list[str],
                      disagreement: np.ndarray, disagreement_names: list[str],
                      calibration_rows: list[dict[str, Any]], oracle: dict[str, Any],
                      masks: dict[str, np.ndarray], thresholds: dict[str, Any],
                      probes: dict[str, Any], input_manifest: dict[str, Any]) -> dict[str, Any]:
    feature_matrix = np.column_stack([state.to_numpy(), disagreement]).astype(np.float32)
    feature_matrix[~np.isfinite(feature_matrix)] = np.nan
    feature_names = state_names + disagreement_names
    map_rows, map_bins = residual_map(feature_matrix, feature_names, frame["r_strong"],
                                      frame["loss_strong"], frame["fold_index"])
    segment_residual = segment_residual_table(masks, frame["r_strong"], frame["loss_strong"],
                                              frame["fold_index"])
    disagreement_rows = []
    for index, name in enumerate(disagreement_names):
        values = np.asarray(disagreement[:, index], float)
        finite = values[np.isfinite(values)]
        quantiles = np.quantile(finite, [0.01, 0.10, 0.50, 0.90, 0.99])
        disagreement_rows.append({
            "feature": name, "n": len(values), "n_finite": len(finite),
            "mean": float(np.mean(finite)), "std": float(np.std(finite)),
            "p01": quantiles[0], "p10": quantiles[1], "p50": quantiles[2],
            "p90": quantiles[3], "p99": quantiles[4],
        })
    metric_rows = candidate_metrics(frame, probes["candidates"])
    segment_candidates = segment_candidate_table(frame, probes["candidates"], masks)
    decision = decision_verdict(metric_rows, segment_candidates, oracle["summary"])

    write_csv(RESULTS / "calibration_audit.csv", calibration_rows)
    write_csv(RESULTS / "residual_map.csv", map_rows)
    write_csv(RESULTS / "residual_map_bins.csv", map_bins)
    write_csv(RESULTS / "segment_residual_map.csv", segment_residual)
    write_csv(RESULTS / "disagreement_summary.csv", disagreement_rows)
    write_csv(RESULTS / "oracle_etx_seq.csv", oracle["semantic_rows"] + oracle["winner_rows"]
              + oracle["weight_grid_oracle_rows"])
    write_csv(RESULTS / "oracle_weight_grid.csv", oracle["weight_grid_rows"])
    write_csv(RESULTS / "seed_null_oracle.csv", oracle["seed_rows"])
    write_csv(RESULTS / "best_existing_oracle.csv", oracle["best_existing_rows"]
              + oracle["best_existing_oracle_rows"])
    write_csv(RESULTS / "oracle_segments.csv", oracle["segment_rows"])
    write_csv(RESULTS / "winner_probe_metrics.csv", probes["winner_rows"])
    write_csv(RESULTS / "winner_probe_calibration.csv", probes["winner_calibration_rows"])
    write_csv(RESULTS / "advantage_probe_metrics.csv", probes["advantage_rows"])
    write_csv(RESULTS / "seed_winner_probe_metrics.csv", probes["seed_winner_rows"])
    write_csv(RESULTS / "recipient_split_audit.csv", probes["split_audit_rows"]
              + probes["residual_audit_rows"])
    write_csv(RESULTS / "residual_scale_selection.csv", probes["residual_scale_rows"])
    write_csv(RESULTS / "recipient_half_metrics.csv", metric_rows)
    write_csv(RESULTS / "late_fold_candidates.csv",
              [row for row in metric_rows if row["scope"] == "combined"])
    write_csv(RESULTS / "segment_candidate_metrics.csv", segment_candidates)
    write_json(RESULTS / "segment_thresholds.json", thresholds)
    write_json(RESULTS / "probe_config.json", {
        "params": PROBE_PARAMS, "feature_sets": probes["feature_sets"],
        "winner_target": "1[loss_full_etx < loss_full_seq]",
        "winner_weight": "clip(abs(advantage), donor p10, donor p99)",
        "advantage_objective": "regression_l1", "residual_scales": SCALES,
        "shuffle_strata": "fold x STRONGEST prediction decile x rec_buy fixed bin x w180_days_buy fixed bin",
    })
    write_json(RESULTS / "input_manifest.json", input_manifest)
    write_json(RESULTS / "full_lofo_status.json", {
        "performed": False,
        "reason": "late-fold gate not passed" if not decision["late_fold_pass"]
        else "methodological stop: earlier-fold temporal LOFO would require future labels",
    })

    combined = {(row["candidate"]): row for row in metric_rows if row["scope"] == "combined"}
    winner_combined = {(row["feature_set"], row["control"]): row for row in probes["winner_rows"]
                       if row["direction"] == "combined" and row["target"] == "semantic_etx_vs_seq"}
    adv_combined = {row["control"]: row for row in probes["advantage_rows"]
                    if row["direction"] == "combined"}
    confirmed = [row for row in map_rows if row["predicts_signed_residual"]]
    confirmed = sorted(confirmed, key=lambda row: abs(row["pearson_signed_residual"]), reverse=True)[:20]
    summary = {
        "prefix": PREFIX, "experiment_id": EXPERIMENT_ID,
        "residual_signal_verdict": decision["verdict"],
        "decision": decision,
        "oracle_headroom": oracle["summary"],
        "cutoff_safe_predictability": {
            "winner_auc": winner_combined.get(("COMBINED", "REAL"), {}).get("auc"),
            "winner_weighted_auc": winner_combined.get(("COMBINED", "REAL"), {}).get("weighted_auc"),
            "winner_shuffle_auc": winner_combined.get(("COMBINED", "SHUFFLED"), {}).get("auc"),
            "advantage_correlation": adv_combined.get("REAL", {}).get("pearson"),
            "advantage_shuffle_correlation": adv_combined.get("SHUFFLED", {}).get("pearson"),
        },
        "late_fold_realized_gain": {
            "gate": combined["GATE_REAL"], "gate_shuffled": combined["GATE_SHUFFLED"],
            "residual_probe": combined["RESIDUAL_REAL"],
            "residual_shuffled": combined["RESIDUAL_SHUFFLED"],
        },
        "what_predicts_signed_residual": confirmed,
        "does_ensemble_design_leave_ge_0_001": (
            "YES" if decision["verdict"] == "STRONG" else
            "MAYBE" if decision["verdict"] == "MODERATE" else "NO"),
        "does_this_explain_0_005_gap": "NO",
        "next_representation_bet": (None if decision["verdict"] in {"STRONG", "MODERATE"}
                                     else "BURST-STATE REPRESENTATION: activity episodes + explicit inactivity gaps + regime transitions"),
        "past_residual": {
            "available_source_fold": "2025-09-04", "source_target_end": "2025-10-04",
            "eligible_recipient_fold": "2025-10-16", "included_in_probe": False,
            "reason": "no cutoff-safe donor-fold support",
        },
        "no_test_submission_leaderboard_paths_touched": True,
    }
    write_json(RESULTS / "summary.json", summary)
    save_late_predictions(frame, probes)
    return summary


def _validate_no_forbidden_paths(manifest: dict[str, Any]) -> None:
    text = json.dumps(_jsonable(manifest), ensure_ascii=False).lower()
    forbidden = ("\\submissions\\", "/submissions/", "ztest_", "test_predictions.parquet",
                 "public_lb", "sample_submit")
    bad = [value for value in forbidden if value in text]
    if bad:
        raise AssertionError(f"forbidden test/submission/LB input paths: {bad}")


def build_everything() -> tuple[dict[str, np.ndarray], pl.DataFrame, list[str], np.ndarray,
                                list[str], list[dict[str, Any]], dict[str, Any],
                                dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    frame, manifest = _load_core()
    _audit_baseline(frame, manifest)
    _load_optional(frame, manifest)
    _load_pact(frame, manifest)
    _load_zero2d(frame, manifest)
    state, state_names = _load_state_features(frame, manifest)

    diagnostic = pl.read_parquet(DIAGNOSTIC_FRAME, columns=["cutoff", "user_id", "y"])
    diagnostic = diagnostic.sort(["cutoff", "user_id"])
    if not (np.array_equal(diagnostic["user_id"].to_numpy(), frame["user_id"])
            and np.array_equal(diagnostic["cutoff"].to_numpy(), frame["cutoff"])):
        raise AssertionError("rmsle diagnostics frame row alignment failed")
    if not np.allclose(diagnostic["y"].to_numpy(), frame["y"], atol=1e-6):
        raise AssertionError("rmsle diagnostics frame target alignment failed")
    manifest["rmsle_diagnostics_frame"] = {
        "path": DIAGNOSTIC_FRAME.resolve(), "file_sha256": file_sha256(DIAGNOSTIC_FRAME),
        "n": diagnostic.height, "status": "PASS_ALIGNED",
    }

    calibration_rows = add_analysis_objects(frame)
    disagreement, disagreement_names = build_disagreement_features(frame)
    frame["past_residual_0904"] = add_past_residual(frame)
    masks, thresholds = segment_masks(frame, state, disagreement, disagreement_names)
    oracle = oracle_analysis(frame, masks)
    strata = build_strata(frame, state)
    manifest["strata_sha256"] = array_sha256(strata)
    manifest["source_sha256"] = {
        "residual_signal_discovery.py": file_sha256(Path(__file__)),
        "features.py": file_sha256(ROOT / "src" / "features.py"),
        "validation.py": file_sha256(ROOT / "src" / "validation.py"),
        "config.py": file_sha256(ROOT / "src" / "config.py"),
    }
    manifest["forbidden_inputs"] = {
        "test_inference": False, "submission": False, "leaderboard_data_read": False,
        "base_model_training": False,
    }
    _validate_no_forbidden_paths(manifest)
    return (frame, state, state_names, disagreement, disagreement_names,
            calibration_rows, oracle, masks, thresholds, manifest)


def save_probe_metadata(probes: dict[str, Any]) -> None:
    payload = {key: value for key, value in probes.items()
               if key not in {"candidates", "late_arrays"}}
    write_json(OUT_ARTIFACTS / "probe_metadata.json", payload)


def load_probe_metadata() -> dict[str, Any]:
    return json.loads((OUT_ARTIFACTS / "probe_metadata.json").read_text(encoding="utf-8"))


def main(analysis_only: bool = False) -> None:
    started = time.time()
    OUT_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (frame, state, state_names, disagreement, disagreement_names, calibration_rows,
     oracle, masks, thresholds, manifest) = build_everything()

    if analysis_only:
        before_path = RESULTS / "reproducibility.json"
        before = json.loads(before_path.read_text(encoding="utf-8"))["canonical_hashes"]
        probes = load_cached_probes(frame, load_probe_metadata())
    else:
        save_aligned_frame(frame, state, disagreement, disagreement_names)
        strata = build_strata(frame, state)
        probes = train_all_probes(frame, state, state_names, disagreement,
                                  disagreement_names, strata, oracle)
        save_probe_outputs(frame, probes)
        save_probe_metadata(probes)
        # Analyze the serialized float32 artifact on both paths.  Otherwise the
        # training run uses in-memory float64 predictions while --analysis-only
        # uses the persisted float32 arrays, making byte reproducibility fail.
        probes = load_cached_probes(frame, load_probe_metadata())

    summary = analyze_and_write(frame, state, state_names, disagreement, disagreement_names,
                                calibration_rows, oracle, masks, thresholds, probes, manifest)
    hashes = canonical_output_hashes()
    if analysis_only:
        status = "PASS" if hashes == before else "FAIL"
        if status != "PASS":
            raise AssertionError(f"analysis-only hashes changed: before={before}, after={hashes}")
    else:
        status = "BASELINE"
    write_json(RESULTS / "reproducibility.json", {
        "analysis_only_status": status, "canonical_hashes": hashes,
        "probe_outputs_sha256": file_sha256(_probe_output_path()),
        "aligned_oof_sha256": file_sha256(OUT_ARTIFACTS / "aligned_oof.parquet"),
    })
    print(f"{PREFIX}: verdict={summary['residual_signal_verdict']} "
          f"runtime={time.time() - started:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-only", action="store_true")
    args = parser.parse_args()
    main(args.analysis_only)
