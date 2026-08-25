"""Canonical OOF evaluation and baseline comparison helpers."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.metrics import bias_z, calibrate_log_offset, rmsle_z, weighted_cv
from src.validation.folds import fold_weights, validation_cutoffs


def expected_fold_names() -> list[str]:
    return [value.isoformat() for value in validation_cutoffs()]


def evaluate_oof(y_true, z_pred, cutoff) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=float)
    z = np.asarray(z_pred, dtype=float)
    folds = np.asarray(cutoff, dtype="U10")
    if not (len(y) == len(z) == len(folds)):
        raise ValueError("y_true, z_pred and cutoff must have equal length")
    present = sorted(set(folds.tolist()))
    configured = expected_fold_names()
    ordered = [fold for fold in configured if fold in present] + [fold for fold in present if fold not in configured]
    per_fold: list[dict[str, Any]] = []
    for fold in ordered:
        mask = folds == fold
        offset, calibrated = calibrate_log_offset(y[mask], z[mask])
        per_fold.append(
            {
                "cutoff": fold,
                "n": int(mask.sum()),
                "rmsle": rmsle_z(y[mask], z[mask]),
                "bias": bias_z(y[mask], z[mask]),
                "offset": offset,
                "rmsle_cal": calibrated,
                "mean_z": float(z[mask].mean()),
                "mean_log1p_y": float(np.log1p(y[mask]).mean()),
            }
        )
    full_scheme = ordered == configured
    offset, calibrated = calibrate_log_offset(y, z)
    calibrated_folds = [row["rmsle_cal"] for row in per_fold]
    raw_folds = [row["rmsle"] for row in per_fold]
    return {
        "n": int(len(y)),
        "folds": ordered,
        "fold_sizes": [row["n"] for row in per_fold],
        "per_fold": per_fold,
        "fold_scores": raw_folds,
        "fold_cal": calibrated_folds,
        "wCV": weighted_cv(calibrated_folds, fold_weights()) if full_scheme else None,
        "wCV_raw": weighted_cv(raw_folds, fold_weights()) if full_scheme else None,
        "oof_rmsle": rmsle_z(y, z),
        "oof_bias": bias_z(y, z),
        "oof_offset": offset,
        "oof_calibrated": calibrated,
        "mean_z": float(z.mean()),
        "mean_log1p_y": float(np.log1p(y).mean()),
        "partial_validation": not full_scheme,
    }


def assert_unique_row_keys(frame: pd.DataFrame) -> None:
    required = {"cutoff", "user_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing row-key columns: {sorted(missing)}")
    duplicated = frame.duplicated(["cutoff", "user_id"])
    if duplicated.any():
        raise ValueError(f"Duplicate (cutoff, user_id) keys: {int(duplicated.sum())}")


def compare_oof(challenger: pd.DataFrame, baseline: pd.DataFrame) -> dict[str, Any]:
    assert_unique_row_keys(challenger)
    assert_unique_row_keys(baseline)
    joined = challenger.merge(
        baseline[["cutoff", "user_id", "y_true", "z_pred"]],
        on=["cutoff", "user_id"],
        how="inner",
        suffixes=("_challenger", "_baseline"),
        validate="one_to_one",
    )
    if len(joined) != len(challenger) or len(joined) != len(baseline):
        raise ValueError(
            f"OOF row sets differ: challenger={len(challenger)}, baseline={len(baseline)}, aligned={len(joined)}"
        )
    if not np.array_equal(joined["y_true_challenger"].to_numpy(), joined["y_true_baseline"].to_numpy()):
        raise ValueError("OOF targets differ after row alignment")
    challenger_metrics = evaluate_oof(
        joined["y_true_challenger"], joined["z_pred_challenger"], joined["cutoff"]
    )
    baseline_metrics = evaluate_oof(
        joined["y_true_baseline"], joined["z_pred_baseline"], joined["cutoff"]
    )
    challenger_by_fold = {row["cutoff"]: row["rmsle_cal"] for row in challenger_metrics["per_fold"]}
    baseline_by_fold = {row["cutoff"]: row["rmsle_cal"] for row in baseline_metrics["per_fold"]}
    deltas = [challenger_by_fold[fold] - baseline_by_fold[fold] for fold in challenger_metrics["folds"]]
    return {
        "challenger": challenger_metrics,
        "baseline": baseline_metrics,
        "delta_wCV": (
            challenger_metrics["wCV"] - baseline_metrics["wCV"]
            if challenger_metrics["wCV"] is not None and baseline_metrics["wCV"] is not None
            else None
        ),
        "fold_deltas": deltas,
        "folds_positive": int(sum(delta < 0 for delta in deltas)),
    }
