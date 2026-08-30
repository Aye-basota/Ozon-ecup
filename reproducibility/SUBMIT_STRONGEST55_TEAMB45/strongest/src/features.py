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

from src.config import (DATA_PROCESSED, DATA_START, HISTORY_L, PANEL_BLOCKS, TARGET_DAYS,
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
def _agg_exprs(windows: list[int]) -> list[pl.Expr]:
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


def build_features(T: dt.date, L: int | None = HISTORY_L, norm_long: bool = False) -> pl.DataFrame:
    """Признаки всех пользователей, имеющих хотя бы одну строку в окне.

    L=None — без усечения (окно = вся доступная история <= T).
    Панель НЕ применяется здесь: она накладывается join'ом в make_xy, чтобы один
    кэш фичей обслуживал любое правило панели.
    """
    windows = WINDOWS_BY_L[L]
    df = load().lazy().filter(pl.col("event_date") <= T)
    if L is not None:
        df = df.filter(pl.col("event_date") > T - dt.timedelta(days=L))
    dnum = (pl.col("event_date") - pl.lit(T)).dt.total_days()      # <= 0
    df = df.with_columns(age=(-dnum).cast(pl.Int32))               # 0 = день T
    f = df.group_by("user_id").agg(_agg_exprs(windows)).collect()
    if norm_long:
        # нормировка ДО производных: отношения и логарифмы должны считаться
        # уже от сопоставимых величин
        f = normalize_long(f, T)
    f = f.with_columns(_derived_exprs(windows, norm_long))
    if L is not None:
        drop = [c for c in f.columns if c != "user_id" and c.startswith(UNBOUNDED)]
        f = f.drop(drop)
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
