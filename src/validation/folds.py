"""Single source of truth for cutoff dates and fold construction."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from src.settings import competition


@dataclass(frozen=True)
class Fold:
    train_cutoffs: tuple[dt.date, ...]
    validation_cutoff: dt.date


def validation_cutoffs() -> list[dt.date]:
    return [dt.date.fromisoformat(value) for value in competition()["validation"]["folds"]]


def fold_weights() -> list[float]:
    return [float(value) for value in competition()["validation"]["fold_weights"]]


def cutoff_grid(
    min_history_days: int | None = None,
    step_days: int | None = None,
    end: dt.date | None = None,
) -> list[dt.date]:
    cfg = competition()
    validation = cfg["validation"]
    eligibility = cfg["eligibility"]
    start_date = dt.date.fromisoformat(str(cfg["data"]["start_date"]))
    history = int(min_history_days or validation["min_history_days"])
    step = int(step_days or validation["cutoff_step_days"])
    corridor_end = end or dt.date.fromisoformat(str(validation["clean_corridor_end"]))
    panel_history = int(eligibility["panel_blocks"]) * int(eligibility["block_days"])
    earliest = start_date + dt.timedelta(days=max(history, panel_history))
    values: list[dt.date] = []
    current = corridor_end
    while current >= earliest:
        values.append(current)
        current -= dt.timedelta(days=step)
    return sorted(values)


def canonical_folds(
    min_history_days: int | None = None,
    step_days: int | None = None,
    values: list[dt.date] | None = None,
) -> list[Fold]:
    cfg = competition()
    horizon = int(cfg["data"]["target_horizon_days"])
    grid = cutoff_grid(min_history_days, step_days)
    folds: list[Fold] = []
    for validation_cutoff in values or validation_cutoffs():
        train = tuple(
            cutoff
            for cutoff in grid
            if cutoff + dt.timedelta(days=horizon) <= validation_cutoff
        )
        if not train:
            raise ValueError(f"No leakage-safe training cutoffs for {validation_cutoff}")
        folds.append(Fold(train, validation_cutoff))
    return folds


def assert_fold_safety(fold: Fold) -> None:
    horizon = int(competition()["data"]["target_horizon_days"])
    if any(cutoff + dt.timedelta(days=horizon) > fold.validation_cutoff for cutoff in fold.train_cutoffs):
        raise AssertionError(f"Target leakage in fold {fold.validation_cutoff}")
