"""Единый пайплайн фичей: build_features(cutoff, L).

Один и тот же код строит фичи для train, val и test.
Все агрегаты считаются ТОЛЬКО по строкам с `event_date <= T`; при усечении —
по строкам из полуинтервала (T - L, T]. Никакого лукапа по построению.

Ключевые решения (research/strategy_1.md §4):
  * панель на каждом cutoff'е переприменяет правило организатора (3 блока по 30 дней);
  * пропущенные дни НЕ достраиваются нулями — «нет строки» и «строка из нулей»
    это разные состояния (eda §6);
  * при усечении до L из набора выбрасываются признаки с неограниченным окном
    (`all_*`, `lifetime_*`, `tenure`, `first_buy_age`, `w365_*`), потому что
    доступная глубина истории на разных cutoff'ах разная (eda §7.4).

Запуск: python -m src.features --cutoffs grid --L 180   (прогрев кэша фичей)
"""
from __future__ import annotations

import argparse
import datetime as dt
import time

import numpy as np
import polars as pl

from src.config import (DATA_PROCESSED, DATA_START, HISTORY_L, PANEL_BLOCKS, SEED, TARGET_DAYS,
                        WINDOWS_BY_L, cutoff_grid)
from src.data import load

UNBOUNDED = ("all_", "lifetime_", "tenure", "first_buy_age", "w365")


def _tag(T: dt.date) -> str:
    return T.strftime("%Y%m%d")


# --------------------------------------------------------------------------- панель
def panel_users(T: dt.date, n_blocks: int = PANEL_BLOCKS) -> pl.DataFrame:
    """Правило организатора: >=1 активный день в каждом из n_blocks последних 30-дневных блоков.

    Проверено на тесте: panel_users(2026-02-13, 3).height == 250_000 (eda §3.1).
    """
    p = DATA_PROCESSED / f"panel_{_tag(T)}_b{n_blocks}.parquet"
    if p.exists():
        return pl.read_parquet(p)
    df = load()
    u = None
    for k in range(n_blocks):
        b = T - dt.timedelta(days=30 * k)
        a = b - dt.timedelta(days=29)
        blk = (df.lazy().filter((pl.col("event_date") >= a) & (pl.col("event_date") <= b))
               .select("user_id").unique().collect())
        u = blk if u is None else u.join(blk, on="user_id", how="inner")
    u = u.sort("user_id")
    u.write_parquet(p)
    return u


# --------------------------------------------------------------------------- таргет
def target(T: dt.date, users: pl.DataFrame, horizon: int = TARGET_DAYS) -> pl.DataFrame:
    """Суммарный GMV пользователя в окне (T, T+horizon]. Пользователи без покупок -> 0."""
    df = load()
    a, b = T + dt.timedelta(days=1), T + dt.timedelta(days=horizon)
    y = (df.lazy()
         .filter((pl.col("event_date") >= a) & (pl.col("event_date") <= b) & (pl.col("gmv") > 0))
         .group_by("user_id").agg(pl.col("gmv").sum().alias("y")).collect())
    return (users.select("user_id").join(y, on="user_id", how="left")
            .with_columns(pl.col("y").fill_null(0.0)).sort("user_id"))


# --------------------------------------------------------------------------- фичи
_BLOCK_EXISTING = {
    "gmv": "gmv",
    "orders": "orders",
    "days_buy": "days_buy",
    "days_present": "days_present",
    "searches": "searches",
    "to_cart": "carts",
    "gmv_cat": "gmv_cat",
}
_BLOCK_MISSING = ("gmv_search", "search_to_ord", "cat_to_ord")


def _block_raw_exprs() -> list[pl.Expr]:
    """Агрегаты, у которых нет эквивалента среди штатных ``w30/w60/w90``.

    Для остальных семи величин block0 уже равен ``w30_*``, а block1/block2
    ниже считаются разностями накопительных окон. Так мы не создаём точных
    дубликатов существующих колонок.
    """
    out: list[pl.Expr] = []
    for k, (lo, hi) in enumerate(((0, 30), (30, 60), (60, 90))):
        m = (pl.col("age") >= lo) & (pl.col("age") < hi)
        for c in _BLOCK_MISSING:
            out.append(pl.when(m).then(pl.col(c)).otherwise(0).sum().alias(f"block{k}_{c}"))
    return out


def _block_level_exprs() -> list[pl.Expr]:
    """Непересекающиеся блоки 31--60 и 61--90 из накопительных окон."""
    out: list[pl.Expr] = []
    for name, base in _BLOCK_EXISTING.items():
        out += [
            (pl.col(f"w60_{base}") - pl.col(f"w30_{base}")).alias(f"block1_{name}"),
            (pl.col(f"w90_{base}") - pl.col(f"w60_{base}")).alias(f"block2_{name}"),
        ]
    return out


def _block_source(metric: str, k: int) -> pl.Expr:
    if metric in _BLOCK_EXISTING and k == 0:
        return pl.col(f"w30_{_BLOCK_EXISTING[metric]}")
    return pl.col(f"block{k}_{metric}")


def _block_derived_exprs() -> list[pl.Expr]:
    """Selection-aware BLOCK4 признаки динамики трёх непересекающихся блоков."""
    eps = 1e-6
    out: list[pl.Expr] = []
    for name in tuple(_BLOCK_EXISTING) + _BLOCK_MISSING:
        b0, b1, b2 = (_block_source(name, k).cast(pl.Float64) for k in range(3))
        mean = (b0 + b1 + b2) / 3.0
        std = (((b0 - mean) ** 2 + (b1 - mean) ** 2 + (b2 - mean) ** 2) / 3.0).sqrt()
        out += [
            (b0 - b1).alias(f"block_d01_{name}"),
            (b1 - b2).alias(f"block_d12_{name}"),
            (b0 - 2.0 * b1 + b2).alias(f"block_accel_{name}"),
            (b0 / (b1 + eps)).alias(f"block_ratio01_{name}"),
            (b1 / (b2 + eps)).alias(f"block_ratio12_{name}"),
            mean.alias(f"block_mean3_{name}"),
            std.alias(f"block_std3_{name}"),
        ]
    return out


OPEN_FUNNEL_COLUMNS = [
    "of90_search_days", "of90_cart_days", "of90_funnel_days",
    "of90_search_no_order_days", "of90_cart_no_order_days",
    "of90_search_cart_days", "of90_searches", "of90_carts",
    "of90_search_to_cart", "of90_cat_to_cart", "of90_oldest_search_age",
    "of90_oldest_cart_age", "of90_search_span", "of90_cart_span",
]

PLATFORM_DETREND_COLUMNS = [
    f"pd_w{w}_{metric}_rel"
    for w in (30, 90)
    for metric in ("present", "searches", "carts", "orders", "gmv")
]

EVENT_ORDER_COLUMNS = [
    "eo90_transition_count", "eo90_state_change_count", "eo90_repeat_share",
    "eo90_up_count", "eo90_down_count", "eo90_search_to_cartbuy_count",
    "eo90_cart_to_buy_count", "eo90_nobuy_to_buy_count",
    "eo90_buy_to_nobuy_count", "eo90_buy_to_buy_count",
    "eo30_up_count", "eo30_down_count", "eo90_up_gap_mean",
    "eo90_down_gap_mean", "eo_last_transition_code",
]


def _event_order_daily(T: dt.date, source: str = "real") -> pl.DataFrame:
    """Observed daily three-bit funnel states, optionally permuted within user.

    The shuffled arm keeps the exact user-specific state multiset and observed
    dates. Only the assignment of complete state vectors to dates changes.
    """
    if source not in {"real", "shuffled"}:
        raise ValueError("event order source must be real or shuffled")
    dnum = (pl.col("event_date") - pl.lit(T)).dt.total_days()
    daily = (load().lazy()
             .filter((pl.col("event_date") <= T)
                     & (pl.col("event_date") > T - dt.timedelta(days=90)))
             .select(["user_id", "event_date", "searches", "to_cart", "to_ord", "gmv"])
             .with_columns([
                 (-dnum).cast(pl.Int16).alias("_eo_age"),
                 ((pl.col("searches") > 0).cast(pl.Int8)
                  + 2 * (pl.col("to_cart") > 0).cast(pl.Int8)
                  + 4 * ((pl.col("to_ord") > 0) | (pl.col("gmv") > 0)).cast(pl.Int8))
                 .alias("_eo_state"),
             ])
             .select(["user_id", "event_date", "_eo_age", "_eo_state"])
             .collect().sort(["user_id", "event_date"]))
    if source == "shuffled":
        salt = int(SEED) + (T - DATA_START).days
        daily = (daily.with_columns(
            pl.struct(["user_id", "event_date"]).hash(seed=salt).alias("_eo_perm_key"))
            .with_columns(
                pl.col("_eo_state").sort_by("_eo_perm_key").over("user_id").alias("_eo_state"))
            .drop("_eo_perm_key"))
    return daily


def _event_order_frame(T: dt.date, source: str = "real") -> pl.DataFrame:
    daily = _event_order_daily(T, source).with_columns([
        (pl.col("_eo_state") % 2).cast(pl.Int8).alias("_eo_s"),
        ((pl.col("_eo_state") // 2) % 2).cast(pl.Int8).alias("_eo_c"),
        ((pl.col("_eo_state") // 4) % 2).cast(pl.Int8).alias("_eo_b"),
    ]).with_columns(
        pl.when(pl.col("_eo_b") > 0).then(3)
        .when(pl.col("_eo_c") > 0).then(2)
        .when(pl.col("_eo_s") > 0).then(1).otherwise(0).cast(pl.Int8).alias("_eo_stage")
    ).with_columns([
        pl.col("_eo_state").shift(1).over("user_id").alias("_eo_prev_state"),
        pl.col("_eo_s").shift(1).over("user_id").alias("_eo_prev_s"),
        pl.col("_eo_c").shift(1).over("user_id").alias("_eo_prev_c"),
        pl.col("_eo_b").shift(1).over("user_id").alias("_eo_prev_b"),
        pl.col("_eo_stage").shift(1).over("user_id").alias("_eo_prev_stage"),
        pl.col("event_date").diff().dt.total_days().over("user_id").alias("_eo_gap"),
    ])
    valid = pl.col("_eo_prev_state").is_not_null()
    up = valid & (pl.col("_eo_stage") > pl.col("_eo_prev_stage"))
    down = valid & (pl.col("_eo_stage") < pl.col("_eo_prev_stage"))
    search_to_cartbuy = (valid & (pl.col("_eo_prev_s") > 0)
                         & (pl.col("_eo_prev_c") == 0) & (pl.col("_eo_prev_b") == 0)
                         & ((pl.col("_eo_c") > 0) | (pl.col("_eo_b") > 0)))
    cart_to_buy = (valid & (pl.col("_eo_prev_c") > 0) & (pl.col("_eo_prev_b") == 0)
                   & (pl.col("_eo_b") > 0))
    nobuy_to_buy = valid & (pl.col("_eo_prev_b") == 0) & (pl.col("_eo_b") > 0)
    buy_to_nobuy = valid & (pl.col("_eo_prev_b") > 0) & (pl.col("_eo_b") == 0)
    buy_to_buy = valid & (pl.col("_eo_prev_b") > 0) & (pl.col("_eo_b") > 0)
    out = daily.group_by("user_id").agg([
        valid.sum().alias("eo90_transition_count"),
        (valid & (pl.col("_eo_state") != pl.col("_eo_prev_state"))).sum()
        .alias("eo90_state_change_count"),
        up.sum().alias("eo90_up_count"),
        down.sum().alias("eo90_down_count"),
        search_to_cartbuy.sum().alias("eo90_search_to_cartbuy_count"),
        cart_to_buy.sum().alias("eo90_cart_to_buy_count"),
        nobuy_to_buy.sum().alias("eo90_nobuy_to_buy_count"),
        buy_to_nobuy.sum().alias("eo90_buy_to_nobuy_count"),
        buy_to_buy.sum().alias("eo90_buy_to_buy_count"),
        (up & (pl.col("_eo_age") < 30)).sum().alias("eo30_up_count"),
        (down & (pl.col("_eo_age") < 30)).sum().alias("eo30_down_count"),
        pl.when(up).then(pl.col("_eo_gap")).mean().alias("eo90_up_gap_mean"),
        pl.when(down).then(pl.col("_eo_gap")).mean().alias("eo90_down_gap_mean"),
        pl.when(valid).then(pl.col("_eo_prev_state") * 8 + pl.col("_eo_state"))
        .sort_by("event_date").last().alias("eo_last_transition_code"),
    ]).with_columns([
        ((pl.col("eo90_transition_count") - pl.col("eo90_state_change_count"))
         / pl.col("eo90_transition_count").clip(lower_bound=1)).alias("eo90_repeat_share"),
        pl.col("eo90_up_gap_mean").fill_null(0),
        pl.col("eo90_down_gap_mean").fill_null(0),
        pl.col("eo_last_transition_code").fill_null(0),
    ])
    return out.select(["user_id"] + EVENT_ORDER_COLUMNS).sort("user_id")


def _open_funnel_frame(T: dt.date) -> pl.DataFrame:
    """Cutoff-safe unresolved Search/Cart state after the last positive-GMV day.

    Only the last 90 days are accumulated, so train and test have identical
    support despite different total history depth.  The last-buy boundary is
    found from the complete history available at ``T``.  Daily data cannot order
    events inside a day, therefore the boundary day itself is excluded.
    """
    hist = load().lazy().filter(pl.col("event_date") <= T)
    dnum = (pl.col("event_date") - pl.lit(T)).dt.total_days()
    hist = hist.with_columns(age=(-dnum).cast(pl.Int32))
    last_buy = hist.group_by("user_id").agg(
        pl.when(pl.col("gmv") > 0).then(pl.col("age")).min().alias("_of_last_buy_age")
    )
    recent = hist.filter(pl.col("age") < 90).join(last_buy, on="user_id", how="left")
    open_row = pl.col("_of_last_buy_age").is_null() | (pl.col("age") < pl.col("_of_last_buy_age"))
    has_search = pl.col("searches") > 0
    has_cart = pl.col("to_cart") > 0
    no_order = pl.col("to_ord") == 0
    search_open = open_row & has_search
    cart_open = open_row & has_cart
    out = recent.group_by("user_id").agg([
        search_open.sum().alias("of90_search_days"),
        cart_open.sum().alias("of90_cart_days"),
        (open_row & (has_search | has_cart)).sum().alias("of90_funnel_days"),
        (search_open & no_order).sum().alias("of90_search_no_order_days"),
        (cart_open & no_order).sum().alias("of90_cart_no_order_days"),
        (open_row & has_search & has_cart).sum().alias("of90_search_cart_days"),
        pl.when(open_row).then(pl.col("searches")).otherwise(0).sum().alias("of90_searches"),
        pl.when(open_row).then(pl.col("to_cart")).otherwise(0).sum().alias("of90_carts"),
        pl.when(open_row).then(pl.col("search_to_cart")).otherwise(0).sum().alias("of90_search_to_cart"),
        pl.when(open_row).then(pl.col("cat_to_cart")).otherwise(0).sum().alias("of90_cat_to_cart"),
        pl.when(search_open).then(pl.col("age")).max().alias("of90_oldest_search_age"),
        pl.when(cart_open).then(pl.col("age")).max().alias("of90_oldest_cart_age"),
        pl.when(search_open).then(pl.col("age")).min().alias("_of_recent_search_age"),
        pl.when(cart_open).then(pl.col("age")).min().alias("_of_recent_cart_age"),
    ]).collect()
    out = out.with_columns([
        (pl.col("of90_oldest_search_age") - pl.col("_of_recent_search_age"))
        .fill_null(0).alias("of90_search_span"),
        (pl.col("of90_oldest_cart_age") - pl.col("_of_recent_cart_age"))
        .fill_null(0).alias("of90_cart_span"),
        pl.col("of90_oldest_search_age").fill_null(90),
        pl.col("of90_oldest_cart_age").fill_null(90),
    ]).drop(["_of_recent_search_age", "_of_recent_cart_age"])
    return out.select(["user_id"] + OPEN_FUNNEL_COLUMNS).sort("user_id")


def _platform_daily_factors(T: dt.date, source: str = "real") -> pl.DataFrame:
    """Same-day platform intensity using only the fixed 90-day history at ``T``.

    ``shuffled`` jointly permutes the five factor vectors inside fixed 28-day
    calendar blocks. This preserves their marginals and cross-metric dependence
    while breaking alignment with each user's event dates.
    """
    if source not in {"real", "shuffled"}:
        raise ValueError("platform detrend source must be real or shuffled")
    # The daily reference population is the current cutoff's observable panel,
    # never the globally future-selected universe on its own.
    current_panel = panel_users(T).lazy()
    hist = (load().lazy()
            .filter((pl.col("event_date") <= T)
                    & (pl.col("event_date") > T - dt.timedelta(days=90)))
            .join(current_panel, on="user_id", how="inner"))
    daily = (hist.group_by("event_date").agg([
        pl.len().alias("_pd_present"),
        pl.col("searches").sum().alias("_pd_searches"),
        pl.col("to_cart").sum().alias("_pd_carts"),
        pl.col("to_ord").sum().alias("_pd_orders"),
        pl.col("gmv").sum().alias("_pd_gmv"),
    ]).collect().sort("event_date"))
    rate_cols = []
    for metric in ("searches", "carts", "orders", "gmv"):
        name = f"_pd_{metric}_factor"
        rate_cols.append(name)
        daily = daily.with_columns(
            (pl.col(f"_pd_{metric}") / pl.col("_pd_present").clip(lower_bound=1))
            .alias(name))
    rate_cols.insert(0, "_pd_present_factor")
    daily = daily.with_columns(pl.col("_pd_present").cast(pl.Float64).alias(rate_cols[0]))
    daily = daily.with_columns([
        (pl.col(c) / max(float(daily[c].median()), 1e-9)).alias(c) for c in rate_cols
    ])
    if source == "shuffled":
        days = daily["event_date"].to_list()
        block = np.asarray([(d - DATA_START).days // 28 for d in days], dtype=np.int32)
        perm = np.arange(len(days))
        rng = np.random.default_rng(int(SEED) + (T - DATA_START).days)
        for value in np.unique(block):
            idx = np.flatnonzero(block == value)
            if len(idx) > 1:
                perm[idx] = rng.permutation(idx)
        values = daily.select(rate_cols).to_numpy()[perm]
        daily = daily.select("event_date").with_columns([
            pl.Series(name, values[:, j]) for j, name in enumerate(rate_cols)
        ])
    return daily.select(["event_date"] + rate_cols)


def _platform_detrend_frame(T: dt.date, source: str = "real") -> pl.DataFrame:
    hist = (load().lazy()
            .filter((pl.col("event_date") <= T)
                    & (pl.col("event_date") > T - dt.timedelta(days=90))))
    factors = _platform_daily_factors(T, source).lazy()
    dnum = (pl.col("event_date") - pl.lit(T)).dt.total_days()
    joined = hist.join(factors, on="event_date", how="left").with_columns(
        age=(-dnum).cast(pl.Int32))
    aggs: list[pl.Expr] = []
    mapping = {
        "present": pl.lit(1.0), "searches": pl.col("searches").cast(pl.Float64),
        "carts": pl.col("to_cart").cast(pl.Float64),
        "orders": pl.col("to_ord").cast(pl.Float64),
        "gmv": pl.col("gmv").cast(pl.Float64),
    }
    for w in (30, 90):
        inside = pl.col("age") < w
        for metric, value in mapping.items():
            factor = pl.col(f"_pd_{metric}_factor").clip(lower_bound=1e-6)
            aggs.append(pl.when(inside).then(value / factor).otherwise(0.0).sum()
                         .alias(f"pd_w{w}_{metric}_rel"))
    return (joined.group_by("user_id").agg(aggs).collect()
            .select(["user_id"] + PLATFORM_DETREND_COLUMNS).sort("user_id"))


def _agg_exprs(windows: list[int], block_features: bool = False) -> list[pl.Expr]:
    aggs: list[pl.Expr] = []
    for w in windows:
        m = pl.col("age") < w
        s = f"w{w}"
        aggs += [
            m.sum().alias(f"{s}_days_present"),
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
        pl.col("age").sort().diff().abs().mean().alias("gap_mean"),
        pl.col("age").sort().diff().abs().std().alias("gap_std"),
        pl.col("age").sort().diff().abs().max().alias("gap_max"),
        pl.when(pl.col("gmv") > 0).then(pl.col("age")).sort().diff().abs().mean().alias("buygap_mean"),
        pl.when(pl.col("gmv") > 0).then(pl.col("age")).sort().diff().abs().std().alias("buygap_std"),
        (pl.col("event_date").dt.weekday() >= 6).mean().alias("weekend_share"),
    ]
    if block_features:
        aggs += _block_raw_exprs()
    return aggs


def _derived_exprs(windows: list[int], norm_long: bool = False) -> list[pl.Expr]:
    e = 1e-6
    d: list[pl.Expr] = []
    for w in windows:
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
            (pl.col(f"{s}_gmv") * 30.0 / float(w)).log1p().alias(f"{s}_gmv30eq"),
        ]
    pairs = [(a, b) for a, b in [(7, 14), (7, 30), (14, 30), (30, 60), (30, 90), (60, 180),
                                 (90, 365), (90, 270)] if a in windows and b in windows]
    for a, b in pairs:
        d += [
            ((pl.col(f"w{a}_gmv") / float(a)) / (pl.col(f"w{b}_gmv") / float(b) + e)).alias(f"trend_gmv_{a}_{b}"),
            ((pl.col(f"w{a}_days_present") / float(a)) / (pl.col(f"w{b}_days_present") / float(b) + e)
             ).alias(f"trend_pres_{a}_{b}"),
            ((pl.col(f"w{a}_searches") / float(a)) / (pl.col(f"w{b}_searches") / float(b) + e)
             ).alias(f"trend_srch_{a}_{b}"),
            (pl.col(f"w{a}_gmv").log1p() - pl.col(f"w{b}_gmv").log1p()).alias(f"dlog_gmv_{a}_{b}"),
            (pl.col(f"w{a}_days_buy").log1p() - pl.col(f"w{b}_days_buy").log1p()).alias(f"dlog_buyd_{a}_{b}"),
        ]
    d += [] if norm_long else [
        (pl.col("all_gmv") / (pl.col("tenure") + 1)).alias("lifetime_gmv_per_day"),
        (pl.col("all_gmv") / (pl.col("all_orders") + e)).alias("lifetime_aov"),
        (pl.col("all_days_buy") / (pl.col("all_days_present") + e)).alias("lifetime_buyrate"),
        pl.col("all_gmv").log1p().alias("all_lgmv"),
    ]
    d += [
        # персонализованная свежесть: recency, нормированная на личный ритм (eda §7.2)
        (pl.col("rec_buy") / (pl.col("buygap_mean") + e)).alias("rec_over_buygap"),
        (pl.col("rec_any") / (pl.col("gap_mean") + e)).alias("rec_over_gap"),
        (pl.col("gap_std") / (pl.col("gap_mean") + e)).alias("gap_cv"),
        (pl.col("buygap_std") / (pl.col("buygap_mean") + e)).alias("buygap_cv"),
    ]
    return d


# суммы и счётчики за 365 дней: их величина зависит от того, сколько истории
# реально доступно на cutoff'е (92 дня в начале коридора, 409 на тесте)
_LONG_SUMS = ["w365_days_present", "w365_days_search", "w365_days_cat", "w365_days_buy",
              "w365_days_cart", "w365_days_presence_only", "w365_searches", "w365_carts",
              "w365_orders", "w365_gmv", "w365_gmv_cat"]


def normalize_long(f: pl.DataFrame, T: dt.date) -> pl.DataFrame:
    """Приводит признаки с 365-дневным окном к одинаковой глубине истории.

    Мотивация (измерено): `all_*` ПОБИТОВО совпадают с `w365_*` на каждом обучающем
    cutoff'е (доступно 92..289 дней < 365) и расходятся ТОЛЬКО на тесте (409 дней).
    То есть `all_*` и производные `lifetime_*` не несут в обучении никакой информации
    сверх `w365_*`, а на тесте ведут себя иначе — это чистый риск без выгоды,
    поэтому они выбрасываются.

    Сами `w365_*` — это суммы за `min(365, avail)` дней. Умножение на `365/avail`
    делает их сопоставимой ГОДОВОЙ ставкой; `tenure` и `first_buy_age` заменяются
    долями от доступной истории. Сигнал длинного окна при этом сохраняется —
    в отличие от полного усечения до L.
    """
    avail = (T - DATA_START).days + 1
    k = 365.0 / min(avail, 365)
    f = f.with_columns([pl.col(c) * k for c in _LONG_SUMS if c in f.columns])
    f = f.with_columns([
        (pl.col("tenure") / avail).alias("tenure_frac"),
        (pl.col("first_buy_age") / avail).alias("first_buy_frac"),
        (pl.col("gap_max") / avail).alias("gap_max_frac"),
    ])
    return f.drop([c for c in f.columns
                   if c.startswith("all_") or c in ("tenure", "first_buy_age", "gap_max")])


def build_features(T: dt.date, L: int | None = HISTORY_L, norm_long: bool = False,
                   block_features: bool = False,
                   open_funnel_features: bool = False,
                   platform_detrend_source: str | None = None,
                   event_order_source: str | None = None,
                   base_features: pl.DataFrame | None = None) -> pl.DataFrame:
    """Признаки всех пользователей, имеющих хотя бы одну строку в окне.

    L=None — без усечения (окно = вся доступная история <= T).
    Панель НЕ применяется здесь: она накладывается join'ом в make_xy, чтобы один
    кэш фичей обслуживал любое правило панели.
    """
    windows = WINDOWS_BY_L[L]
    if base_features is None:
        df = load().lazy().filter(pl.col("event_date") <= T)
        if L is not None:
            df = df.filter(pl.col("event_date") > T - dt.timedelta(days=L))
        dnum = (pl.col("event_date") - pl.lit(T)).dt.total_days()      # <= 0
        df = df.with_columns(age=(-dnum).cast(pl.Int32))               # 0 = день T
        f = df.group_by("user_id").agg(_agg_exprs(windows, block_features)).collect()
        if norm_long:
            # нормировка ДО производных: отношения и логарифмы должны считаться
            # уже от сопоставимых величин
            f = normalize_long(f, T)
        f = f.with_columns(_derived_exprs(windows, norm_long))
        if L is not None:
            drop = [c for c in f.columns if c != "user_id" and c.startswith(UNBOUNDED)]
            f = f.drop(drop)
    else:
        if not (block_features or open_funnel_features or platform_detrend_source
                or event_order_source):
            raise ValueError("base_features разрешён только для opt-in профиля")
        f = base_features
        # Оптимизированный путь для нового opt-in профиля: штатные признаки уже
        # лежат в совместимом parquet-кэше, поэтому повторно читаются только 90
        # последних дней и три отсутствующих в старом pipeline агрегата.
        if block_features:
            df = (load().lazy()
                  .filter((pl.col("event_date") <= T)
                          & (pl.col("event_date") > T - dt.timedelta(days=90))))
            dnum = (pl.col("event_date") - pl.lit(T)).dt.total_days()
            extra = (df.with_columns(age=(-dnum).cast(pl.Int32))
                     .group_by("user_id").agg(_block_raw_exprs()).collect())
            f = f.join(extra, on="user_id", how="left")
            raw_cols = [f"block{k}_{c}" for k in range(3) for c in _BLOCK_MISSING]
            f = f.with_columns([pl.col(c).fill_null(0) for c in raw_cols])
    if block_features:
        f = f.with_columns(_block_level_exprs())
        f = f.with_columns(_block_derived_exprs())
    if open_funnel_features:
        f = f.join(_open_funnel_frame(T), on="user_id", how="left")
        f = f.with_columns([pl.col(c).fill_null(0) for c in OPEN_FUNNEL_COLUMNS])
    if platform_detrend_source:
        f = f.join(_platform_detrend_frame(T, platform_detrend_source),
                   on="user_id", how="left")
        f = f.with_columns([pl.col(c).fill_null(0) for c in PLATFORM_DETREND_COLUMNS])
    if event_order_source:
        f = f.join(_event_order_frame(T, event_order_source), on="user_id", how="left")
        f = f.with_columns([pl.col(c).fill_null(0) for c in EVENT_ORDER_COLUMNS])
    return f.sort("user_id")


def features_cached(T: dt.date, L: int | None = HISTORY_L,
                    norm_long: bool = False) -> pl.DataFrame:
    p = DATA_PROCESSED / f"feat_{_tag(T)}_L{'norm' if norm_long else ''}{L}.parquet"
    if p.exists():
        return pl.read_parquet(p)
    f = build_features(T, L, norm_long)
    f.write_parquet(p)
    return f


def make_xy(T: dt.date, L: int | None = HISTORY_L, n_blocks: int = PANEL_BLOCKS,
            horizon: int = TARGET_DAYS, with_target: bool = True, norm_long: bool = False,
            ptime: str | None = None, ptime_source: str = "real"):
    """(X, y) на cutoff'е T: панель -> фичи -> таргет. y=None для тестового cutoff'а.

    `ptime` (STRATEGY_08) — подмножество признаков личного времени из `src/ptime.py`
    ('od' | 'full'), приклеиваемое отдельным join'ом. Кэш `feat_*` при этом не
    меняется, поэтому OOF прежних экспериментов остаётся сравнимым напрямую.
    """
    f = features_cached(T, L, norm_long)
    if ptime:
        from src.ptime import SUBSETS, ptime_cached
        pt = ptime_cached(T, L, ptime_source).select(["user_id"] + SUBSETS[ptime])
        f = f.join(pt, on="user_id", how="left")
    # n_blocks=0 — панель не накладывается: берём всех, у кого есть хоть одна строка
    # в окне признаков. Максимум обучающих данных, проверяется экспериментом S1-E05.
    u = f.select("user_id") if n_blocks == 0 else panel_users(T, n_blocks)
    X = u.join(f, on="user_id", how="left").sort("user_id")
    # float32: вдвое меньше памяти, LightGBM всё равно бинует признаки
    X = X.with_columns([pl.col(c).cast(pl.Float32) for c in X.columns if c != "user_id"])
    if not with_target:
        return X, None
    y = target(T, u, horizon)["y"].to_numpy()
    assert X.height == len(y)
    return X, y


def feature_names(X: pl.DataFrame) -> list[str]:
    return [c for c in X.columns if c != "user_id"]


def to_np(X: pl.DataFrame, feats: list[str]) -> np.ndarray:
    return X.select(feats).to_numpy().astype(np.float32)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Прогрев кэша фичей")
    ap.add_argument("--L", type=int, default=HISTORY_L)
    ap.add_argument("--min-history", type=int, default=90)
    ap.add_argument("--norm-long", action="store_true")
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--extra", nargs="*", default=[], help="доп. cutoff-даты YYYY-MM-DD")
    a = ap.parse_args()
    L = None if a.L <= 0 else a.L
    from src.config import CUTOFF_STEP, CUTOFF_TEST
    cuts = cutoff_grid(a.min_history, a.step or CUTOFF_STEP) + [CUTOFF_TEST]
    cuts += [dt.date.fromisoformat(s) for s in a.extra]
    load()
    for T in cuts:
        t = time.time()
        n_before = (DATA_PROCESSED /
                    f"feat_{_tag(T)}_L{'norm' if a.norm_long else ''}{L}.parquet").exists()
        f = features_cached(T, L, a.norm_long)
        print(f"{T}  n={f.height:>7,}  feats={f.width - 1:>3}  "
              f"{'cached' if n_before else f'{time.time() - t:5.1f}s'}", flush=True)
