"""Canonical cutoff-safe feature pipeline.

Every train, validation, and test feature frame is produced by
``build_features(cutoff_date)``. The observable window is always bounded by
``event_date <= cutoff_date``; targets are built separately on ``(T, T+30]``.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl

from src.data import load_events
from src.settings import competition, paths


UNBOUNDED_PREFIXES = ("all_", "lifetime_", "tenure", "first_buy_age", "w365")
WINDOWS_BY_HISTORY = {
    90: [7, 14, 30, 60, 90],
    180: [7, 14, 30, 60, 90, 180],
    270: [7, 14, 30, 60, 90, 180, 270],
    None: [7, 14, 30, 60, 90, 180, 365],
}


def _tag(cutoff_date: dt.date) -> str:
    return cutoff_date.strftime("%Y%m%d")


def _target_days() -> int:
    return int(competition()["data"]["target_horizon_days"])


def panel_users(cutoff_date: dt.date, n_blocks: int | None = None) -> pl.DataFrame:
    """Apply the organizer eligibility rule at a cutoff.

    A user must have at least one observed row in each of the most recent
    non-overlapping 30-day blocks. The default is three blocks.
    """
    cfg = competition()
    blocks = int(n_blocks or cfg["eligibility"]["panel_blocks"])
    block_days = int(cfg["eligibility"]["block_days"])
    cache = paths().processed_root / f"panel_{_tag(cutoff_date)}_b{blocks}.parquet"
    if cache.exists():
        return pl.read_parquet(cache)

    events = load_events()
    users: pl.DataFrame | None = None
    for block_index in range(blocks):
        end = cutoff_date - dt.timedelta(days=block_days * block_index)
        start = end - dt.timedelta(days=block_days - 1)
        observed = (
            events.lazy()
            .filter((pl.col("event_date") >= start) & (pl.col("event_date") <= end))
            .select("user_id")
            .unique()
            .collect()
        )
        users = observed if users is None else users.join(observed, on="user_id", how="inner")
    assert users is not None
    users = users.sort("user_id")
    cache.parent.mkdir(parents=True, exist_ok=True)
    users.write_parquet(cache)
    return users


def target(
    cutoff_date: dt.date,
    users: pl.DataFrame,
    horizon_days: int | None = None,
) -> pl.DataFrame:
    """Total GMV in ``(cutoff_date, cutoff_date + horizon]`` for eligible users."""
    horizon = int(horizon_days or _target_days())
    start = cutoff_date + dt.timedelta(days=1)
    end = cutoff_date + dt.timedelta(days=horizon)
    observed = (
        load_events()
        .lazy()
        .filter(
            (pl.col("event_date") >= start)
            & (pl.col("event_date") <= end)
            & (pl.col("gmv") > 0)
        )
        .group_by("user_id")
        .agg(pl.col("gmv").sum().alias("y_true"))
        .collect()
    )
    return (
        users.select("user_id")
        .join(observed, on="user_id", how="left")
        .with_columns(pl.col("y_true").fill_null(0.0))
        .sort("user_id")
    )


def _aggregate_expressions(windows: list[int]) -> list[pl.Expr]:
    expressions: list[pl.Expr] = []
    for window in windows:
        inside = pl.col("age") < window
        prefix = f"w{window}"
        expressions.extend(
            [
                inside.sum().alias(f"{prefix}_days_present"),
                (inside & (pl.col("searches") > 0)).sum().alias(f"{prefix}_days_search"),
                (inside & (pl.col("cat") > 0)).sum().alias(f"{prefix}_days_cat"),
                (inside & (pl.col("gmv") > 0)).sum().alias(f"{prefix}_days_buy"),
                (inside & (pl.col("to_cart") > 0)).sum().alias(f"{prefix}_days_cart"),
                (
                    inside
                    & (pl.col("searches") == 0)
                    & (pl.col("cat") == 0)
                    & (pl.col("to_cart") == 0)
                    & (pl.col("to_ord") == 0)
                )
                .sum()
                .alias(f"{prefix}_days_presence_only"),
                pl.when(inside).then(pl.col("searches")).otherwise(0).sum().alias(f"{prefix}_searches"),
                pl.when(inside).then(pl.col("to_cart")).otherwise(0).sum().alias(f"{prefix}_carts"),
                pl.when(inside).then(pl.col("to_ord")).otherwise(0).sum().alias(f"{prefix}_orders"),
                pl.when(inside).then(pl.col("gmv")).otherwise(0.0).sum().alias(f"{prefix}_gmv"),
                pl.when(inside).then(pl.col("gmv_cat")).otherwise(0.0).sum().alias(f"{prefix}_gmv_cat"),
                pl.when(inside & (pl.col("gmv") > 0)).then(pl.col("gmv")).max().alias(f"{prefix}_gmv_max"),
                pl.when(inside & (pl.col("gmv") > 0))
                .then(pl.col("gmv").log1p())
                .mean()
                .alias(f"{prefix}_lgmv_mean"),
                pl.when(inside & (pl.col("gmv") > 0))
                .then(pl.col("gmv").log1p())
                .std()
                .alias(f"{prefix}_lgmv_std"),
            ]
        )
    expressions.extend(
        [
            pl.col("age").min().alias("rec_any"),
            pl.when(pl.col("searches") > 0).then(pl.col("age")).min().alias("rec_search"),
            pl.when(pl.col("to_cart") > 0).then(pl.col("age")).min().alias("rec_cart"),
            pl.when(pl.col("gmv") > 0).then(pl.col("age")).min().alias("rec_buy"),
            pl.when(pl.col("cat") > 0).then(pl.col("age")).min().alias("rec_cat"),
            pl.col("age").max().alias("tenure"),
            pl.when(pl.col("gmv") > 0).then(pl.col("age")).max().alias("first_buy_age"),
            pl.len().alias("all_days_present"),
            (pl.col("gmv") > 0).sum().alias("all_days_buy"),
            pl.col("gmv").sum().alias("all_gmv"),
            pl.col("to_ord").sum().alias("all_orders"),
            pl.col("searches").sum().alias("all_searches"),
            pl.col("age").sort().diff().abs().mean().alias("gap_mean"),
            pl.col("age").sort().diff().abs().std().alias("gap_std"),
            pl.col("age").sort().diff().abs().max().alias("gap_max"),
            pl.when(pl.col("gmv") > 0).then(pl.col("age")).sort().diff().abs().mean().alias("buygap_mean"),
            pl.when(pl.col("gmv") > 0).then(pl.col("age")).sort().diff().abs().std().alias("buygap_std"),
            (pl.col("event_date").dt.weekday() >= 6).mean().alias("weekend_share"),
        ]
    )
    return expressions


def _derived_expressions(windows: list[int], normalize_long: bool) -> list[pl.Expr]:
    epsilon = 1e-6
    expressions: list[pl.Expr] = []
    for window in windows:
        prefix = f"w{window}"
        expressions.extend(
            [
                (pl.col(f"{prefix}_gmv") / (pl.col(f"{prefix}_orders") + epsilon)).alias(f"{prefix}_aov"),
                (pl.col(f"{prefix}_gmv") / (pl.col(f"{prefix}_days_present") + epsilon)).alias(f"{prefix}_gmv_per_day"),
                (pl.col(f"{prefix}_orders") / (pl.col(f"{prefix}_carts") + epsilon)).alias(f"{prefix}_cart2ord"),
                (pl.col(f"{prefix}_carts") / (pl.col(f"{prefix}_searches") + epsilon)).alias(f"{prefix}_srch2cart"),
                (pl.col(f"{prefix}_days_buy") / (pl.col(f"{prefix}_days_present") + epsilon)).alias(f"{prefix}_buyday_rate"),
                (pl.col(f"{prefix}_days_present") / float(window)).alias(f"{prefix}_presence_rate"),
                (pl.col(f"{prefix}_days_presence_only") / (pl.col(f"{prefix}_days_present") + epsilon)).alias(f"{prefix}_ponly_share"),
                (pl.col(f"{prefix}_gmv_cat") / (pl.col(f"{prefix}_gmv") + epsilon)).alias(f"{prefix}_cat_gmv_share"),
                (pl.col(f"{prefix}_searches") / (pl.col(f"{prefix}_days_search") + epsilon)).alias(f"{prefix}_srch_per_day"),
                pl.col(f"{prefix}_gmv").log1p().alias(f"{prefix}_lgmv"),
                (pl.col(f"{prefix}_gmv") * 30.0 / float(window)).log1p().alias(f"{prefix}_gmv30eq"),
            ]
        )
    candidate_pairs = [(7, 14), (7, 30), (14, 30), (30, 60), (30, 90), (60, 180), (90, 365), (90, 270)]
    for short, long in (pair for pair in candidate_pairs if pair[0] in windows and pair[1] in windows):
        expressions.extend(
            [
                ((pl.col(f"w{short}_gmv") / short) / (pl.col(f"w{long}_gmv") / long + epsilon)).alias(f"trend_gmv_{short}_{long}"),
                ((pl.col(f"w{short}_days_present") / short) / (pl.col(f"w{long}_days_present") / long + epsilon)).alias(f"trend_pres_{short}_{long}"),
                ((pl.col(f"w{short}_searches") / short) / (pl.col(f"w{long}_searches") / long + epsilon)).alias(f"trend_srch_{short}_{long}"),
                (pl.col(f"w{short}_gmv").log1p() - pl.col(f"w{long}_gmv").log1p()).alias(f"dlog_gmv_{short}_{long}"),
                (pl.col(f"w{short}_days_buy").log1p() - pl.col(f"w{long}_days_buy").log1p()).alias(f"dlog_buyd_{short}_{long}"),
            ]
        )
    if not normalize_long:
        expressions.extend(
            [
                (pl.col("all_gmv") / (pl.col("tenure") + 1)).alias("lifetime_gmv_per_day"),
                (pl.col("all_gmv") / (pl.col("all_orders") + epsilon)).alias("lifetime_aov"),
                (pl.col("all_days_buy") / (pl.col("all_days_present") + epsilon)).alias("lifetime_buyrate"),
                pl.col("all_gmv").log1p().alias("all_lgmv"),
            ]
        )
    expressions.extend(
        [
            (pl.col("rec_buy") / (pl.col("buygap_mean") + epsilon)).alias("rec_over_buygap"),
            (pl.col("rec_any") / (pl.col("gap_mean") + epsilon)).alias("rec_over_gap"),
            (pl.col("gap_std") / (pl.col("gap_mean") + epsilon)).alias("gap_cv"),
            (pl.col("buygap_std") / (pl.col("buygap_mean") + epsilon)).alias("buygap_cv"),
        ]
    )
    return expressions


LONG_SUM_COLUMNS = [
    "w365_days_present",
    "w365_days_search",
    "w365_days_cat",
    "w365_days_buy",
    "w365_days_cart",
    "w365_days_presence_only",
    "w365_searches",
    "w365_carts",
    "w365_orders",
    "w365_gmv",
    "w365_gmv_cat",
]


def _normalize_long(frame: pl.DataFrame, cutoff_date: dt.date) -> pl.DataFrame:
    data_start = dt.date.fromisoformat(str(competition()["data"]["start_date"]))
    available_days = (cutoff_date - data_start).days + 1
    scale = 365.0 / min(available_days, 365)
    frame = frame.with_columns([pl.col(column) * scale for column in LONG_SUM_COLUMNS if column in frame.columns])
    frame = frame.with_columns(
        [
            (pl.col("tenure") / available_days).alias("tenure_frac"),
            (pl.col("first_buy_age") / available_days).alias("first_buy_frac"),
            (pl.col("gap_max") / available_days).alias("gap_max_frac"),
        ]
    )
    return frame.drop(
        [
            column
            for column in frame.columns
            if column.startswith("all_") or column in ("tenure", "first_buy_age", "gap_max")
        ]
    )


def build_features(
    cutoff_date: dt.date,
    history_days: int | None = None,
    normalize_long: bool = True,
) -> pl.DataFrame:
    """Build all canonical features from observations available at ``cutoff_date``."""
    if history_days not in WINDOWS_BY_HISTORY:
        raise ValueError(f"Unsupported history_days={history_days}; use one of {list(WINDOWS_BY_HISTORY)}")
    windows = WINDOWS_BY_HISTORY[history_days]
    history = load_events().lazy().filter(pl.col("event_date") <= cutoff_date)
    if history_days is not None:
        history = history.filter(pl.col("event_date") > cutoff_date - dt.timedelta(days=history_days))
    relative_day = (pl.col("event_date") - pl.lit(cutoff_date)).dt.total_days()
    history = history.with_columns(age=(-relative_day).cast(pl.Int32))
    frame = history.group_by("user_id").agg(_aggregate_expressions(windows)).collect()
    if normalize_long:
        frame = _normalize_long(frame, cutoff_date)
    frame = frame.with_columns(_derived_expressions(windows, normalize_long))
    if history_days is not None:
        frame = frame.drop(
            [
                column
                for column in frame.columns
                if column != "user_id" and column.startswith(UNBOUNDED_PREFIXES)
            ]
        )
    return frame.sort("user_id")


def feature_cache_path(cutoff_date: dt.date, history_days: int | None, normalize_long: bool) -> Path:
    history_tag = "None" if history_days is None else str(history_days)
    normalization_tag = "norm" if normalize_long else "raw"
    return paths().processed_root / f"feat_{_tag(cutoff_date)}_L{normalization_tag}{history_tag}.parquet"


def features_cached(
    cutoff_date: dt.date,
    history_days: int | None = None,
    normalize_long: bool = True,
) -> pl.DataFrame:
    cache = feature_cache_path(cutoff_date, history_days, normalize_long)
    if cache.exists():
        return pl.read_parquet(cache)
    frame = build_features(cutoff_date, history_days, normalize_long)
    cache.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(cache)
    return frame


def make_xy(
    cutoff_date: dt.date,
    history_days: int | None = None,
    panel_blocks: int | None = None,
    with_target: bool = True,
    normalize_long: bool = True,
) -> tuple[pl.DataFrame, np.ndarray | None]:
    features = features_cached(cutoff_date, history_days, normalize_long)
    users = panel_users(cutoff_date, panel_blocks)
    frame = users.join(features, on="user_id", how="left").sort("user_id")
    frame = frame.with_columns(
        [pl.col(column).cast(pl.Float32) for column in frame.columns if column != "user_id"]
    )
    if not with_target:
        return frame, None
    y_true = target(cutoff_date, users)["y_true"].to_numpy()
    if frame.height != len(y_true):
        raise AssertionError("Feature/target row alignment failed")
    return frame, y_true


def feature_names(frame: pl.DataFrame) -> list[str]:
    return [column for column in frame.columns if column != "user_id"]


def to_numpy(frame: pl.DataFrame, columns: list[str]) -> np.ndarray:
    return frame.select(columns).to_numpy().astype(np.float32, copy=False)
