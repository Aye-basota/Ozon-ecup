"""Key-safe alignment and blending of standardized OOF artifacts."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.artifacts import load_oof


def align_oof(sources: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    if not sources:
        raise ValueError("At least one OOF source is required")
    frames = [load_oof(source).sort_values(["cutoff", "user_id"]).reset_index(drop=True) for source in sources]
    keys = frames[0][["cutoff", "user_id"]]
    y_true = frames[0]["y_true"].to_numpy()
    predictions = []
    for source, frame in zip(sources, frames):
        if not frame[["cutoff", "user_id"]].equals(keys):
            raise ValueError(f"OOF row keys differ for {source}")
        if not np.array_equal(frame["y_true"].to_numpy(), y_true):
            raise ValueError(f"OOF targets differ for {source}")
        predictions.append(frame["z_pred"].to_numpy())
    base = keys.copy()
    base["y_true"] = y_true
    return base, np.vstack(predictions)


def log_space_blend(predictions: np.ndarray, weights) -> np.ndarray:
    values = np.asarray(predictions, dtype=float)
    blend_weights = np.asarray(weights, dtype=float)
    if values.ndim != 2 or values.shape[0] != len(blend_weights):
        raise ValueError("Expected prediction matrix shaped (models, rows)")
    if not np.isclose(blend_weights.sum(), 1.0):
        raise ValueError("Blend weights must sum to one")
    return np.average(values, axis=0, weights=blend_weights)
