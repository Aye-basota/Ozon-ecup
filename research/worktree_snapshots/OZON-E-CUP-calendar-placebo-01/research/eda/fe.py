"""Reusable feature builder + panel rule + target, all leakage-safe w.r.t. a cutoff."""
import datetime as dt

import numpy as np
import polars as pl

RAW = r"C:\Users\Admin\Desktop\OZON-E-CUP\data\raw\train.parquet"
END = dt.date(2026, 2, 13)
WINDOWS = [7, 14, 30, 60, 90, 180, 365]

_CACHE = {}


def load():
    if "df" not in _CACHE:
        df = pl.read_parquet(RAW, columns=[
            "user_id", "event_date", "searches", "cat", "search_to_cart", "search_to_ord",
            "cat_to_cart", "cat_to_ord", "to_cart", "to_ord", "gmv", "gmv_search", "gmv_cat"])
        _CACHE["df"] = df
    return _CACHE["df"]


def panel_users(T: dt.date, n_blocks: int = 3):
    """Users satisfying the organiser rule: active in each trailing 30-day block."""
    df = load()
    u = None
    for k in range(n_blocks):
        b = T - dt.timedelta(days=30 * k)
        a = b - dt.timedelta(days=29)
        blk = (df.lazy().filter((pl.col("event_date") >= a) & (pl.col("event_date") <= b))
               .select("user_id").unique().collect())
        u = blk if u is None else u.join(blk, on="user_id", how="inner")
    return u.sort("user_id")


def target(T: dt.date, users: pl.DataFrame, horizon: int = 30):
    df = load()
    a, b = T + dt.timedelta(days=1), T + dt.timedelta(days=horizon)
    y = (df.lazy().filter((pl.col("event_date") >= a) & (pl.col("event_date") <= b) & (pl.col("gmv") > 0))
         .group_by("user_id").agg(pl.col("gmv").sum().alias("y")).collect())
    return (users.join(y, on="user_id", how="left")
            .with_columns(pl.col("y").fill_null(0.0)).sort("user_id"))


def build_features(T: dt.date, users: pl.DataFrame) -> pl.DataFrame:
    """All aggregates use ONLY rows with event_date <= T."""
    df = load().lazy().filter(pl.col("event_date") <= T)
    dnum = (pl.col("event_date") - pl.lit(T)).dt.total_days()   # <= 0
    df = df.with_columns(age=(-dnum).cast(pl.Int32))            # 0 = day T

    aggs = []
    for w in WINDOWS:
        m = pl.col("age") < w
        s = f"w{w}"
        aggs += [
            (m).sum().alias(f"{s}_days_present"),
            (m & (pl.col("searches") > 0)).sum().alias(f"{s}_days_search"),
            (m & (pl.col("cat") > 0)).sum().alias(f"{s}_days_cat"),
            (m & (pl.col("gmv") > 0)).sum().alias(f"{s}_days_buy"),
            (m & (pl.col("to_cart") > 0)).sum().alias(f"{s}_days_cart"),
            (m & (pl.col("searches") == 0) & (pl.col("cat") == 0) & (pl.col("to_cart") == 0)
             & (pl.col("to_ord") == 0)).sum().alias(f"{s}_days_presence_only"),
            pl.when(m).then(pl.col("searches")).otherwise(0).sum().alias(f"{s}_searches"),
            pl.when(m).then(pl.col("to_cart")).otherwise(0).sum().alias(f"{s}_carts"),
            pl.when(m).then(pl.col("to_ord")).otherwise(0).sum().alias(f"{s}_orders"),
            pl.when(m).then(pl.col("gmv")).otherwise(0.0).sum().alias(f"{s}_gmv"),
            pl.when(m).then(pl.col("gmv_cat")).otherwise(0.0).sum().alias(f"{s}_gmv_cat"),
            pl.when(m & (pl.col("gmv") > 0)).then(pl.col("gmv")).max().alias(f"{s}_gmv_max"),
            pl.when(m & (pl.col("gmv") > 0)).then(pl.col("gmv").log1p()).mean().alias(f"{s}_lgmv_mean"),
            pl.when(m & (pl.col("gmv") > 0)).then(pl.col("gmv").log1p()).std().alias(f"{s}_lgmv_std"),
        ]
    # recency
    aggs += [
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
        # gap structure between presence days
        pl.col("age").sort().diff().abs().mean().alias("gap_mean"),
        pl.col("age").sort().diff().abs().std().alias("gap_std"),
        pl.col("age").sort().diff().abs().max().alias("gap_max"),
        # gap between purchase days
        pl.when(pl.col("gmv") > 0).then(pl.col("age")).sort().diff().abs().mean().alias("buygap_mean"),
        pl.when(pl.col("gmv") > 0).then(pl.col("age")).sort().diff().abs().std().alias("buygap_std"),
        # weekday profile of activity
        (pl.col("event_date").dt.weekday() >= 6).mean().alias("weekend_share"),
    ]
    f = df.group_by("user_id").agg(aggs).collect()
    f = users.join(f, on="user_id", how="left").sort("user_id")

    # ---- derived, leakage-safe (functions of the aggregates only) ----
    e = 1e-6
    d = []
    for w in WINDOWS:
        s = f"w{w}"
        d += [
            (pl.col(f"{s}_gmv") / (pl.col(f"{s}_orders") + e)).alias(f"{s}_aov"),
            (pl.col(f"{s}_gmv") / (pl.col(f"{s}_days_present") + e)).alias(f"{s}_gmv_per_day"),
            (pl.col(f"{s}_orders") / (pl.col(f"{s}_carts") + e)).alias(f"{s}_cart2ord"),
            (pl.col(f"{s}_carts") / (pl.col(f"{s}_searches") + e)).alias(f"{s}_srch2cart"),
            (pl.col(f"{s}_days_buy") / (pl.col(f"{s}_days_present") + e)).alias(f"{s}_buyday_rate"),
            (pl.col(f"{s}_days_present") / float(w)).alias(f"{s}_presence_rate"),
            (pl.col(f"{s}_days_presence_only") / (pl.col(f"{s}_days_present") + e)).alias(f"{s}_ponly_share"),
            (pl.col(f"{s}_gmv_cat") / (pl.col(f"{s}_gmv") + e)).alias(f"{s}_cat_gmv_share"),
            (pl.col(f"{s}_searches") / (pl.col(f"{s}_days_search") + e)).alias(f"{s}_srch_per_day"),
            pl.col(f"{s}_gmv").log1p().alias(f"{s}_lgmv"),
            # rate normalised to a 30-day horizon = a direct persistence-style predictor
            (pl.col(f"{s}_gmv") * 30.0 / float(w)).log1p().alias(f"{s}_gmv30eq"),
        ]
    # trends / accelerations
    pairs = [(7, 14), (7, 30), (14, 30), (30, 60), (30, 90), (60, 180), (90, 365)]
    for a, b in pairs:
        d += [
            ((pl.col(f"w{a}_gmv") / float(a)) / (pl.col(f"w{b}_gmv") / float(b) + e)).alias(f"trend_gmv_{a}_{b}"),
            ((pl.col(f"w{a}_days_present") / float(a)) / (pl.col(f"w{b}_days_present") / float(b) + e)
             ).alias(f"trend_pres_{a}_{b}"),
            ((pl.col(f"w{a}_searches") / float(a)) / (pl.col(f"w{b}_searches") / float(b) + e)
             ).alias(f"trend_srch_{a}_{b}"),
            (pl.col(f"w{a}_gmv").log1p() - pl.col(f"w{b}_gmv").log1p()).alias(f"dlog_gmv_{a}_{b}"),
        ]
    d += [
        (pl.col("all_gmv") / (pl.col("tenure") + 1)).alias("lifetime_gmv_per_day"),
        (pl.col("all_gmv") / (pl.col("all_orders") + e)).alias("lifetime_aov"),
        (pl.col("all_days_buy") / (pl.col("all_days_present") + e)).alias("lifetime_buyrate"),
        (pl.col("rec_buy") / (pl.col("buygap_mean") + e)).alias("rec_over_buygap"),
        (pl.col("rec_any") / (pl.col("gap_mean") + e)).alias("rec_over_gap"),
        pl.col("all_gmv").log1p().alias("all_lgmv"),
    ]
    f = f.with_columns(d)
    return f


def rmsle(y, p):
    return float(np.sqrt(np.mean((np.log1p(y) - np.log1p(np.maximum(p, 0.0))) ** 2)))


def make_xy(T, n_blocks=3, horizon=30):
    u = panel_users(T, n_blocks)
    X = build_features(T, u)
    Y = target(T, u, horizon)
    assert X.height == Y.height
    return X, Y
