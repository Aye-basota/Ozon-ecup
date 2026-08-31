"""Shared validation scheme and metric for the project.

This module must stay model-agnostic: no baseline-specific thresholds,
experiment-specific heuristics, or model comparisons.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from src.config import TARGET_DAYS


FOLDS = [
    {"fold": 1, "train_cutoff": "2025-12-15", "val_cutoff": "2026-01-14"},
    {"fold": 2, "train_cutoff": "2025-11-15", "val_cutoff": "2025-12-15"},
]


def get_folds() -> list[dict[str, int | str]]:
    """Return out-of-time folds from PLAN.md."""
    return [dict(fold) for fold in FOLDS]


def metric(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """Competition RMSLE with predictions clipped below zero."""
    y_pred = np.clip(np.asarray(y_pred), 0, None)
    y_true = np.asarray(y_true)
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)))


def assert_same_features(features: Sequence[str], *frames: pd.DataFrame) -> None:
    """Check that all datasets use exactly the same features in the same order."""
    expected = list(features)
    for frame in frames:
        assert list(frame.columns) == expected


def target_window(cutoff: str | pd.Timestamp, target_days: int = TARGET_DAYS) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the target interval [cutoff, cutoff + target_days)."""
    start = pd.Timestamp(cutoff)
    end = start + pd.Timedelta(days=target_days)
    return start, end


def assert_cutoff_gap(
    train_cutoff: str | pd.Timestamp,
    val_cutoff: str | pd.Timestamp,
    target_days: int = TARGET_DAYS,
) -> None:
    """Ensure a train target window does not overlap the validation history."""
    train_start, train_end = target_window(train_cutoff, target_days)
    val_start = pd.Timestamp(val_cutoff)
    assert train_end <= val_start, (
        f"Train cutoff {train_start.date()} target ends at {train_end.date()}, "
        f"but val cutoff is {val_start.date()}"
    )


def _assert_prediction_shape(y_true: pd.Series, y_pred: pd.Series | np.ndarray) -> None:
    assert len(y_pred) == len(y_true), (
        f"Got {len(y_pred)} predictions, expected {len(y_true)}"
    )


def run_validation(
    df: pd.DataFrame,
    all_users: pd.Index,
    make_dataset_fn: Callable[
        [pd.DataFrame, str | pd.Timestamp, pd.Index],
        tuple[pd.DataFrame, pd.Series],
    ],
    features: Sequence[str],
    fit_predict_fn: Callable[
        [pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, dict[str, int | str]],
        pd.Series | np.ndarray,
    ],
    folds: Sequence[dict[str, int | str]] | None = None,
    verbose: bool = True,
) -> float:
    """Run out-of-time validation and return mean RMSLE.

    fit_predict_fn receives X_train, y_train, X_val, y_val, fold_cfg and returns
    validation predictions.
    """
    fold_list = [dict(fold) for fold in (folds if folds is not None else get_folds())]
    scores: list[float] = []

    for fold_cfg in fold_list:
        fold = int(fold_cfg["fold"])
        train_cutoff = str(fold_cfg["train_cutoff"])
        val_cutoff = str(fold_cfg["val_cutoff"])
        assert_cutoff_gap(train_cutoff, val_cutoff)

        if verbose:
            print(f"\nfold={fold} train_cutoff={train_cutoff} val_cutoff={val_cutoff}")

        X_tr, y_tr = make_dataset_fn(df, train_cutoff, all_users)
        X_va, y_va = make_dataset_fn(df, val_cutoff, all_users)
        assert_same_features(features, X_tr, X_va)

        pred_result = fit_predict_fn(X_tr, y_tr, X_va, y_va, fold_cfg)
        _assert_prediction_shape(y_va, pred_result)
        fold_score = metric(y_va, pred_result)
        scores.append(fold_score)

        if verbose:
            print(f"fold={fold} metric={fold_score:.6f}")

    mean_score = float(np.mean(scores))
    if verbose:
        print(f"CV mean metric={mean_score:.6f}")
    return mean_score
