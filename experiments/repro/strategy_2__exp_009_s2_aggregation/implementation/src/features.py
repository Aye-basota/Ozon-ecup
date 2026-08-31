"""Leakage-safe feature builder used by Strategy 2.

The public entry point is deliberately a single function:
``build_features(cutoff_date)``.  Every raw-data expression is evaluated on the
half-open history window ``(cutoff - 180 days, cutoff]``.  Missing calendar days
are not materialised: an absent row and a present all-zero row are different
states in this dataset.

The base 195 columns are the fixed-depth Strategy 1 aggregates requested by
``research/strategy_2.md``.  Strategy 2 only *adds* point-process columns; it
does not redefine any existing feature.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl

# Keep the repository's documented direct invocation working.
if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_PROCESSED, DATA_RAW, TARGET_DAYS

HISTORY_DAYS = 180
WINDOWS = [7, 14, 30, 60, 90, 180]
RAW_FILE = DATA_RAW / "train.parquet"


def _as_date(cutoff_date: str | dt.date) -> dt.date:
    return cutoff_date if isinstance(cutoff_date, dt.date) else dt.date.fromisoformat(cutoff_date)


def _cache_path(cutoff: dt.date) -> Path:
    # This is the established cache name used by the repository's Strategy 1
    # implementation.  Only the base columns are stored in this file.
    return DATA_PROCESSED / f"feat_{cutoff:%Y%m%d}_L{HISTORY_DAYS}.parquet"


def _aggregate_expressions() -> list[pl.Expr]:
    aggs: list[pl.Expr] = []
    for window in WINDOWS:
        mask = pl.col("age") < window
        prefix = f"w{window}"
        aggs += [
            mask.sum().alias(f"{prefix}_days_present"),
            (mask & (pl.col("searches") > 0)).sum().alias(f"{prefix}_days_search"),
            (mask & (pl.col("cat") > 0)).sum().alias(f"{prefix}_days_cat"),
            (mask & (pl.col("gmv") > 0)).sum().alias(f"{prefix}_days_buy"),
            (mask & (pl.col("to_cart") > 0)).sum().alias(f"{prefix}_days_cart"),
            (mask & (pl.col("searches") == 0) & (pl.col("cat") == 0)
             & (pl.col("to_cart") == 0) & (pl.col("to_ord") == 0)).sum()
            .alias(f"{prefix}_days_presence_only"),
            pl.when(mask).then(pl.col("searches")).otherwise(0).sum().alias(f"{prefix}_searches"),
            pl.when(mask).then(pl.col("to_cart")).otherwise(0).sum().alias(f"{prefix}_carts"),
            pl.when(mask).then(pl.col("to_ord")).otherwise(0).sum().alias(f"{prefix}_orders"),
            pl.when(mask).then(pl.col("gmv")).otherwise(0.0).sum().alias(f"{prefix}_gmv"),
            pl.when(mask).then(pl.col("gmv_cat")).otherwise(0.0).sum().alias(f"{prefix}_gmv_cat"),
            pl.when(mask & (pl.col("gmv") > 0)).then(pl.col("gmv")).max()
            .alias(f"{prefix}_gmv_max"),
            pl.when(mask & (pl.col("gmv") > 0)).then(pl.col("gmv").log1p()).mean()
            .alias(f"{prefix}_lgmv_mean"),
            pl.when(mask & (pl.col("gmv") > 0)).then(pl.col("gmv").log1p()).std()
            .alias(f"{prefix}_lgmv_std"),
        ]
    aggs += [
        pl.col("age").min().alias("rec_any"),
        pl.when(pl.col("searches") > 0).then(pl.col("age")).min().alias("rec_search"),
        pl.when(pl.col("to_cart") > 0).then(pl.col("age")).min().alias("rec_cart"),
        pl.when(pl.col("gmv") > 0).then(pl.col("age")).min().alias("rec_buy"),
        pl.when(pl.col("cat") > 0).then(pl.col("age")).min().alias("rec_cat"),
        pl.col("age").sort().diff().abs().mean().alias("gap_mean"),
        pl.col("age").sort().diff().abs().std().alias("gap_std"),
        pl.col("age").sort().diff().abs().max().alias("gap_max"),
        pl.when(pl.col("gmv") > 0).then(pl.col("age")).sort().diff().abs().mean()
        .alias("buygap_mean"),
        pl.when(pl.col("gmv") > 0).then(pl.col("age")).sort().diff().abs().std()
        .alias("buygap_std"),
        (pl.col("event_date").dt.weekday() >= 6).mean().alias("weekend_share"),
    ]
    return aggs


def _derived_expressions() -> list[pl.Expr]:
    eps = 1e-6
    derived: list[pl.Expr] = []
    for window in WINDOWS:
        prefix = f"w{window}"
        derived += [
            (pl.col(f"{prefix}_gmv") / (pl.col(f"{prefix}_orders") + eps)).alias(f"{prefix}_aov"),
            (pl.col(f"{prefix}_gmv") / (pl.col(f"{prefix}_days_present") + eps))
            .alias(f"{prefix}_gmv_per_day"),
            (pl.col(f"{prefix}_orders") / (pl.col(f"{prefix}_carts") + eps))
            .alias(f"{prefix}_cart2ord"),
            (pl.col(f"{prefix}_carts") / (pl.col(f"{prefix}_searches") + eps))
            .alias(f"{prefix}_srch2cart"),
            (pl.col(f"{prefix}_days_buy") / (pl.col(f"{prefix}_days_present") + eps))
            .alias(f"{prefix}_buyday_rate"),
            (pl.col(f"{prefix}_days_present") / float(window)).alias(f"{prefix}_presence_rate"),
            (pl.col(f"{prefix}_days_presence_only") / (pl.col(f"{prefix}_days_present") + eps))
            .alias(f"{prefix}_ponly_share"),
            (pl.col(f"{prefix}_gmv_cat") / (pl.col(f"{prefix}_gmv") + eps))
            .alias(f"{prefix}_cat_gmv_share"),
            (pl.col(f"{prefix}_searches") / (pl.col(f"{prefix}_days_search") + eps))
            .alias(f"{prefix}_srch_per_day"),
            pl.col(f"{prefix}_gmv").log1p().alias(f"{prefix}_lgmv"),
            (pl.col(f"{prefix}_gmv") * TARGET_DAYS / float(window)).log1p()
            .alias(f"{prefix}_gmv30eq"),
        ]
    pairs = [(7, 14), (7, 30), (14, 30), (30, 60), (30, 90), (60, 180)]
    for short, long in pairs:
        derived += [
            ((pl.col(f"w{short}_gmv") / short) / (pl.col(f"w{long}_gmv") / long + eps))
            .alias(f"trend_gmv_{short}_{long}"),
            ((pl.col(f"w{short}_days_present") / short)
             / (pl.col(f"w{long}_days_present") / long + eps))
            .alias(f"trend_pres_{short}_{long}"),
            ((pl.col(f"w{short}_searches") / short)
             / (pl.col(f"w{long}_searches") / long + eps))
            .alias(f"trend_srch_{short}_{long}"),
            (pl.col(f"w{short}_gmv").log1p() - pl.col(f"w{long}_gmv").log1p())
            .alias(f"dlog_gmv_{short}_{long}"),
            (pl.col(f"w{short}_days_buy").log1p() - pl.col(f"w{long}_days_buy").log1p())
            .alias(f"dlog_buyd_{short}_{long}"),
        ]
    derived += [
        (pl.col("rec_buy") / (pl.col("buygap_mean") + eps)).alias("rec_over_buygap"),
        (pl.col("rec_any") / (pl.col("gap_mean") + eps)).alias("rec_over_gap"),
        (pl.col("gap_std") / (pl.col("gap_mean") + eps)).alias("gap_cv"),
        (pl.col("buygap_std") / (pl.col("buygap_mean") + eps)).alias("buygap_cv"),
    ]
    return derived


def _build_base(cutoff: dt.date) -> pl.DataFrame:
    start = cutoff - dt.timedelta(days=HISTORY_DAYS)
    frame = (
        pl.scan_parquet(
            RAW_FILE,
        )
        .select(["user_id", "event_date", "searches", "cat", "to_cart", "to_ord", "gmv", "gmv_cat"])
        .filter((pl.col("event_date") > start) & (pl.col("event_date") <= cutoff))
        .with_columns(
            age=(pl.lit(cutoff) - pl.col("event_date")).dt.total_days().cast(pl.Int32)
        )
        .group_by("user_id")
        .agg(_aggregate_expressions())
        .collect()
        .with_columns(_derived_expressions())
        .sort("user_id")
    )
    return frame


def _add_strategy_2_columns(frame: pl.DataFrame) -> pl.DataFrame:
    """Only append columns; never replace an existing feature."""
    additions = {
        "k_i": pl.col("w180_days_buy").cast(pl.Float32),
        "hazard_proxy": 1.0 / (pl.col("buygap_mean") + 1.0),
        "n_expected_L": pl.col("w180_days_buy") * TARGET_DAYS / float(HISTORY_DAYS),
        "lgmv_mean": pl.col("w180_lgmv_mean"),
        "lgmv_std": pl.col("w180_lgmv_std"),
        "lgmv_n_eff": pl.col("w180_days_buy").cast(pl.Float32),
    }
    new_columns = [expr.alias(name) for name, expr in additions.items() if name not in frame.columns]
    return frame.with_columns(new_columns) if new_columns else frame


def build_features(cutoff_date: str | dt.date) -> pl.DataFrame:
    """Build all Strategy 2 features using rows with ``event_date <= cutoff`` only."""
    cutoff = _as_date(cutoff_date)
    cache = _cache_path(cutoff)
    if cache.exists():
        base = pl.read_parquet(cache)
    else:
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        base = _build_base(cutoff)
        # Store only established base columns so another strategy cannot silently
        # start consuming Strategy 2-specific features.
        base.write_parquet(cache)
    return _add_strategy_2_columns(base).sort("user_id")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build leakage-safe Strategy 2 features")
    parser.add_argument("cutoff", help="YYYY-MM-DD")
    args = parser.parse_args()
    features = build_features(args.cutoff)
    print(f"{args.cutoff}: {features.height:,} users, {features.width - 1} features")
