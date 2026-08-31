"""Gate-authorized cheap observable pilot for EXP083 Branch A.

Only cumulative short-horizon targets Y1/Y3/Y7/Y14 are learned.  Every head is
trained on historical snapshots whose longest (14-day) label ends seven days
before the validation cutoff.  No Y30 head, no direct residual target, no
same-period user cross-fit, no TEST inference.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

import run_discovery as d


HERE = Path(__file__).resolve().parent
TRAIN_LAGS = (63, 49, 35, 21)
RIDGE_ALPHA = 100.0
HEAD_NAMES = ("p1", "p3", "p7", "p14")
DERIVED_NAMES = [
    "p1", "p3", "p7", "p14",
    "pred_I2_3", "pred_I4_7", "pred_I8_14",
    "pred_share_Y1_Y7", "pred_share_Y3_Y14", "pred_share_Y7_Y14",
    "pred_early_late_intensity_logratio",
]


def dataset(cutoff: str, ids: np.ndarray, uid_all: np.ndarray,
            panel: np.ndarray, gmv: np.ndarray) -> dict[str, Any]:
    X, finite = d.e80obs.load_feature_matrix(cutoff, ids)
    idx = np.searchsorted(uid_all, ids)
    day = int((np.datetime64(cutoff) - d.e80.DATA_START).astype("timedelta64[D]").astype(int))
    future = np.asarray(gmv[idx, day + 1:day + 15], np.float64)
    cum = np.cumsum(future, axis=1)
    raw = np.column_stack([cum[:, 0], cum[:, 2], cum[:, 6], cum[:, 13]])
    return {"cutoff": cutoff, "ids": ids, "X": X, "target": np.log1p(raw),
            "target_raw": raw, "finite_before": finite}


def eligible(cutoff: str) -> np.ndarray:
    return d.e80obs.eligible_ids(cutoff)


def standardize(train: np.ndarray, val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, dtype=np.float64)
    std = train.std(axis=0, dtype=np.float64)
    std[std < 1e-6] = 1.0
    return ((train - mean) / std).astype(np.float32), ((val - mean) / std).astype(np.float32)


def derived(pred_log: np.ndarray) -> np.ndarray:
    pred_log = np.maximum(np.asarray(pred_log, np.float64), 0.0)
    raw = np.expm1(pred_log)
    y1, y3, y7, y14 = (raw[:, i] for i in range(4))
    i23 = np.maximum(y3 - y1, 0.0)
    i47 = np.maximum(y7 - y3, 0.0)
    i814 = np.maximum(y14 - y7, 0.0)
    return np.column_stack([
        pred_log,
        np.log1p(i23), np.log1p(i47), np.log1p(i814),
        d.safe_ratio(y1, y7), d.safe_ratio(y3, y14), d.safe_ratio(y7, y14),
        np.log1p(d.safe_ratio(y7, np.full(len(y7), 7.0)))
        - np.log1p(d.safe_ratio(i814, np.full(len(y7), 7.0))),
    ]).astype(np.float32)


def main() -> None:
    t0 = time.time()
    gate = json.loads((HERE / "multi_horizon_gate.json").read_text(encoding="utf-8"))
    if not gate["models_authorized"]:
        raise RuntimeError("Branch A oracle gate did not authorize heads")
    ctx = d.load_context()
    uid_all = np.load(d.e80.UID_PATH, mmap_mode="r")
    panel = np.load(d.e80.PANEL_PATH, mmap_mode="r")
    gmv = np.load(d.e80.GMV_PATH, mmap_mode="r")
    cache: dict[str, dict[str, Any]] = {}

    def get(cutoff: str, ids: np.ndarray | None = None) -> dict[str, Any]:
        if cutoff in cache:
            return cache[cutoff]
        chosen = eligible(cutoff) if ids is None else ids
        cache[cutoff] = dataset(cutoff, chosen, uid_all, panel, gmv)
        return cache[cutoff]

    prediction_frames = []
    quality_rows = []
    per_fold_features: dict[str, np.ndarray] = {}
    train_audit = []
    for fold in d.FOLDS:
        m = ctx["masks"][fold]
        val = get(fold, ctx["uid"][m])
        train_cutoffs = [d.e80obs.date_minus(fold, lag) for lag in TRAIN_LAGS]
        max_end = max(np.datetime64(x) + np.timedelta64(14, "D") for x in train_cutoffs)
        if max_end > np.datetime64(fold) - np.timedelta64(7, "D"):
            raise AssertionError("short-horizon training embargo violated")
        train_sets = [get(x) for x in train_cutoffs]
        Xtr = np.concatenate([x["X"] for x in train_sets])
        Ytr = np.concatenate([x["target"] for x in train_sets])
        Xtr, Xv = standardize(Xtr, val["X"])
        model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True, solver="lsqr", tol=1e-4, max_iter=1000)
        model.fit(Xtr, Ytr)
        pred = np.maximum(model.predict(Xv), 0.0)
        per_fold_features[fold] = derived(pred)
        for j, name in enumerate(HEAD_NAMES):
            quality_rows.append({
                "cutoff": fold, "head": name,
                "target_corr": d.corr(pred[:, j], val["target"][:, j]),
                "target_RMSE_log": d.rms(pred[:, j] - val["target"][:, j]),
                "prediction_mean_log": float(pred[:, j].mean()),
                "target_mean_log": float(val["target"][:, j].mean()),
            })
        prediction_frames.append(pd.DataFrame({
            "user_id": val["ids"], "cutoff": fold,
            **{HEAD_NAMES[j]: pred[:, j].astype(np.float32) for j in range(4)},
        }))
        train_audit.append({
            "cutoff": fold, "train_cutoffs": train_cutoffs,
            "max_training_target_end": str(max_end), "validation_cutoff": fold,
            "embargo_days": int((np.datetime64(fold) - max_end).astype(int)),
            "train_rows": len(Xtr), "validation_rows": len(Xv),
            "finite_validation_features_before_fill": val["finite_before"],
        })

    individual, summary, latest_q = d.evaluate_observable_block(
        "multi_horizon_predicted_future_state", DERIVED_NAMES, per_fold_features,
        ctx, d.BOOTSTRAP_SEED + 2)
    individual = d.aggregate_individual(individual)
    individual.to_csv(HERE / "multi_horizon_observable_candidate_metrics.csv", index=False)
    pd.DataFrame(quality_rows).to_csv(HERE / "multi_horizon_head_quality.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_parquet(
        HERE / "multi_horizon_head_predictions.parquet", index=False)
    d.write_json(HERE / "multi_horizon_observable_metrics.json", summary)
    audit = {
        "phase": "gate_authorized_short_horizon_Ridge_pilot",
        "model": "multi-output Ridge on 108 cutoff-safe RFM/channel features",
        "ridge_alpha": RIDGE_ALPHA,
        "targets": ["log1p(Y1)", "log1p(Y3)", "log1p(Y7)", "log1p(Y14)"],
        "forbidden_targets": ["Y30", "production residual"],
        "train_lags_days": TRAIN_LAGS,
        "training_audit": train_audit,
        "target_values_used_in_observable_features": False,
        "same_period_cross_sectional_training": False,
        "test_inference": False,
        "submission_created": False,
        "runtime_seconds": time.time() - t0,
    }
    d.write_json(HERE / "multi_horizon_head_audit.json", audit)
    print(json.dumps(d.jsonable({"summary": summary, "audit": audit}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
