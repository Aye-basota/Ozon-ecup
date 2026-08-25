"""Canonical competition metric and fold-level calibration."""
from __future__ import annotations

import numpy as np


def rmsle(y_true, y_pred) -> float:
    y = np.asarray(y_true, dtype=float)
    pred = np.maximum(np.asarray(y_pred, dtype=float), 0.0)
    return float(np.sqrt(np.mean((np.log1p(y) - np.log1p(pred)) ** 2)))


def rmsle_z(y_true, z_pred) -> float:
    """RMSLE when predictions are already ``z = log1p(prediction)``."""
    y = np.asarray(y_true, dtype=float)
    z = np.maximum(np.asarray(z_pred, dtype=float), 0.0)
    return float(np.sqrt(np.mean((np.log1p(y) - z) ** 2)))


def bias_z(y_true, z_pred) -> float:
    return float(np.log1p(np.asarray(y_true, dtype=float)).mean() - np.asarray(z_pred).mean())


def calibrate_log_offset(y_true, z_pred, iterations: int = 25) -> tuple[float, float]:
    """Return the optimal global log offset and RMSLE after applying it.

    The fixed-point update handles the non-negative prediction boundary exactly.
    Calibration is performed independently inside every validation fold.
    """
    y = np.asarray(y_true, dtype=float)
    z = np.asarray(z_pred, dtype=float)
    ly = np.log1p(y)
    offset = float((ly - z).mean())
    for _ in range(iterations):
        active = z + offset > 0
        if not active.any():
            break
        updated = float((ly[active] - z[active]).mean())
        if abs(updated - offset) < 1e-12:
            offset = updated
            break
        offset = updated
    return offset, rmsle_z(y, z + offset)


def weighted_cv(fold_scores, weights) -> float:
    scores = np.asarray(fold_scores, dtype=float)
    fold_weights = np.asarray(weights, dtype=float)
    if len(scores) != len(fold_weights):
        raise ValueError("wCV requires one configured weight for every fold")
    return float(np.dot(scores, fold_weights) / fold_weights.sum())
