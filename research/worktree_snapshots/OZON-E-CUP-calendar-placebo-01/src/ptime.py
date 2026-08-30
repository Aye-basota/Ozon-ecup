"""Личное время пользователя как единица измерения (STRATEGY_08).

Все существующие признаки живут в ОДНОМ времени — календарном: окна 7/14/30/60/
90/180/365 одинаковы для 250k человек. Гипотеза стратегии: полезная величина —
не «сколько дней прошло», а «на сколько СВОИХ циклов пользователь просрочил»,
и не «сколько куплено за 30 дней», а «сколько куплено за свои N циклов».

Единица личного времени:

    rho_buy = средний интервал между покупательными днями   } clip(3, 120):
    rho_act = средний интервал между активными днями         } 1-2 наблюдения
                                                               и вырождение

Признаки строятся ОТДЕЛЬНЫМ кэшем (`pt_<T>_L<L>_<variant>.parquet`) и
приклеиваются к основному join'ом в `make_xy`. Это сознательное решение:
кэш `feat_*` остаётся побитово тем же, поэтому сравнение с S1-E10/S1-SEEDAVG3
идёт против СТАРОГО OOF, а не против молча пересобранного (STRATEGY_08
§Implementation plan — единственное место, где стратегия предупреждает об ошибке).

Антилукап: используются только строки `event_date <= T`, ровно как в
`build_features`. Проверка — `python -m src.ptime --selftest`.

Вариант `shuf` — обязательный контроль честности (§Experiment variants C):
профиль личного времени (rho, квантили интервалов, сам список интервалов)
переставляется МЕЖДУ пользователями с сохранением маргинального распределения.
Если `shuf` работает так же, как `real`, выигрыш идёт от новых нелинейностей,
а не от личного времени, и гипотеза не подтверждена.

Запуск: python -m src.ptime --variant real          (прогрев кэша)
"""
from __future__ import annotations

import argparse
import datetime as dt
import time

import numpy as np
import polars as pl

from src.config import DATA_PROCESSED, DATA_START, SEED
from src.data import load

EPS = 1e-6
RHO_LO, RHO_HI = 3.0, 120.0          # клип личной единицы времени
KS = (1, 2, 4)                       # окна в личном времени: k циклов

# --- состав групп признаков -----------------------------------------------------
# Группа 2 стратегии: просрочка относительно СОБСТВЕННОГО распределения интервалов.
# Это вариант A — самая дешёвая и самая прямая проверка (обобщение rec_over_buygap).
OD_COLS = ["pt_od_rank_buy", "pt_od_rank_act", "pt_od_z_buy", "pt_buygap_p50",
           "pt_buygap_p90", "pt_rec_over_p50", "pt_od_dev_cyc",
           "pt_cyc_since_buy", "pt_cyc_since_act"]

PROFILE_COLS = ["pt_rho_buy", "pt_rho_act", "pt_rho_defined"]
CYCLE_COLS = ["pt_cyc30", "pt_cyc180"]
WINDOW_COLS = ([f"pt_w{k}_buyrate" for k in KS] + [f"pt_w{k}_lgmv_cyc" for k in KS]
               + [f"pt_w{k}_presrate" for k in KS])
RHYTHM_COLS = ["pt_gap_recent3", "pt_rhythm_drift", "pt_rec_over_recent3",
               "pt_buyrate_drift", "pt_int_srch", "pt_int_cart", "pt_int_pres"]

ALL_COLS = PROFILE_COLS + OD_COLS + CYCLE_COLS + WINDOW_COLS + RHYTHM_COLS

SUBSETS = {"od": OD_COLS, "full": ALL_COLS}


def _tag(T: dt.date) -> str:
    return T.strftime("%Y%m%d")


def _events(T: dt.date, L: int | None) -> pl.LazyFrame:
    """Тот же срез истории, что и в `build_features`: только `event_date <= T`."""
    df = load().lazy().filter(pl.col("event_date") <= T)
    if L is not None:
        df = df.filter(pl.col("event_date") > T - dt.timedelta(days=L))
    dnum = (pl.col("event_date") - pl.lit(T)).dt.total_days()
    return df.with_columns(age=(-dnum).cast(pl.Int32))


def _gaps(ev: pl.LazyFrame, mask: pl.Expr | None) -> pl.LazyFrame:
    """(user_id, gap) — интервалы между соседними событиями, СВЕЖИЕ первыми.

    `age` растёт в прошлое, поэтому сортировка по возрастанию `age` внутри
    пользователя ставит последнее событие первым, и `diff()` даёт интервалы
    в порядке «самый свежий -> самый старый». Порядок важен: `head(3)` ниже
    читает именно три последних интервала.
    """
    g = ev if mask is None else ev.filter(mask)
    g = g.select("user_id", "age").sort(["user_id", "age"])
    return g.with_columns(gap=pl.col("age").diff().over("user_id")).drop_nulls("gap")


def _gap_profile(g: pl.LazyFrame, pre: str) -> pl.LazyFrame:
    return g.group_by("user_id").agg([
        pl.col("gap").mean().alias(f"{pre}_mean"),
        pl.col("gap").std().alias(f"{pre}_std"),
        pl.col("gap").median().alias(f"{pre}_p50"),
        pl.col("gap").quantile(0.9).alias(f"{pre}_p90"),
        pl.col("gap").head(3).mean().alias(f"{pre}_recent3"),
        pl.len().alias(f"{pre}_n"),
    ])


def _donor_map(users: np.ndarray, shuffle: bool) -> pl.DataFrame:
    """user_id -> чей профиль личного времени ему приписан.

    `real` — свой собственный (тождественная перестановка). `shuf` — чужой,
    выбранный фиксированной перестановкой: маргинальное распределение rho
    и интервалов сохраняется в точности, связь с пользователем разрывается.
    Перестановка детерминирована (`config.SEED`) и зависит от cutoff'а только
    через состав пользователей.
    """
    if not shuffle:
        donor = users
    else:
        donor = users[np.random.default_rng(SEED).permutation(len(users))]
    return pl.DataFrame({"user_id": users, "donor": donor})


def _od_rank(gaps: pl.LazyFrame, dmap: pl.DataFrame, rec: pl.LazyFrame,
             rec_col: str, out: str) -> pl.LazyFrame:
    """Доля интервалов ДОНОРА, которые короче текущей паузы пользователя.

    Это эмпирическая функция распределения личных интервалов, взятая в точке
    `rec`: 0 — пауза короче всего, что бывало, 1.0 — просрочено всё, что бывало.
    Безразмерная и не зависит от масштаба личного времени, в отличие от
    `rec_buy / buygap_mean`.
    """
    j = (dmap.lazy().join(gaps, left_on="donor", right_on="user_id", how="inner")
         .join(rec.select("user_id", rec_col), on="user_id", how="inner"))
    return (j.group_by("user_id")
            .agg((pl.col("gap") < pl.col(rec_col)).mean().alias(out)))


def _window_aggs(ev: pl.LazyFrame, prof: pl.LazyFrame) -> pl.LazyFrame:
    """Окна ПЕРЕМЕННОЙ длины `k * rho_i` — по одному фильтру на пользователя.

    В `features._agg_exprs` маска окна — константа (`age < 30`), здесь граница
    своя у каждого пользователя, поэтому нужен второй проход с приклеенной
    колонкой `rho`. Это и есть «нормировка на личный масштаб времени», которую
    осевые сплиты дерева приближают тысячами разрезов (STRATEGY_08 §Evidence).
    """
    e = ev.join(prof.select("user_id", "pt_rho_buy", "pt_rho_act"), on="user_id", how="left")
    aggs = []
    for k in KS:
        mb = pl.col("age") < float(k) * pl.col("pt_rho_buy")
        ma = pl.col("age") < float(k) * pl.col("pt_rho_act")
        aggs += [
            (mb & (pl.col("gmv") > 0)).sum().alias(f"_b{k}_days_buy"),
            pl.when(mb).then(pl.col("gmv")).otherwise(0.0).sum().alias(f"_b{k}_gmv"),
            ma.sum().alias(f"_a{k}_days_present"),
            pl.when(ma).then(pl.col("searches")).otherwise(0).sum().alias(f"_a{k}_searches"),
            pl.when(ma).then(pl.col("to_cart")).otherwise(0).sum().alias(f"_a{k}_carts"),
        ]
    return e.group_by("user_id").agg(aggs)


def build_ptime(T: dt.date, L: int | None = None, variant: str = "real") -> pl.DataFrame:
    """Признаки личного времени всех пользователей с историей до `T` включительно."""
    assert variant in ("real", "shuf"), variant
    avail = float(L if L is not None else (T - DATA_START).days + 1)
    ev = _events(T, L)

    base = ev.group_by("user_id").agg([
        pl.when(pl.col("gmv") > 0).then(pl.col("age")).min().alias("rec_buy"),
        pl.col("age").min().alias("rec_any"),
        (pl.col("gmv") > 0).sum().alias("n_buy"),
    ])
    gb, ga = _gaps(ev, pl.col("gmv") > 0), _gaps(ev, None)
    prof = (base.join(_gap_profile(gb, "bg"), on="user_id", how="left")
            .join(_gap_profile(ga, "ag"), on="user_id", how="left"))

    users = np.sort(prof.select("user_id").collect()["user_id"].to_numpy())
    dmap = _donor_map(users, variant == "shuf")

    # профиль личного времени берётся у ДОНОРА, текущая пауза rec_* — своя
    own = prof.select("user_id", "rec_buy", "rec_any", "n_buy")
    donated = (dmap.lazy()
               .join(prof.drop("rec_buy", "rec_any", "n_buy"),
                     left_on="donor", right_on="user_id", how="left")
               .drop("donor"))
    p = own.join(donated, on="user_id", how="left").with_columns([
        pl.col("bg_mean").clip(RHO_LO, RHO_HI).alias("pt_rho_buy"),
        pl.col("ag_mean").clip(RHO_LO, RHO_HI).alias("pt_rho_act"),
        (pl.col("bg_n") >= 1).cast(pl.Float32).alias("pt_rho_defined"),
    ])

    p = (p.join(_od_rank(gb, dmap, own, "rec_buy", "pt_od_rank_buy"), on="user_id", how="left")
         .join(_od_rank(ga, dmap, own, "rec_any", "pt_od_rank_act"), on="user_id", how="left"))
    p = p.join(_window_aggs(ev, p), on="user_id", how="left")

    d: list[pl.Expr] = []
    # --- группа 2: просрочка относительно собственного распределения интервалов
    d += [
        ((pl.col("rec_buy") - pl.col("bg_mean")) / (pl.col("bg_std") + 1.0)).alias("pt_od_z_buy"),
        pl.col("bg_p50").alias("pt_buygap_p50"),
        pl.col("bg_p90").alias("pt_buygap_p90"),
        (pl.col("rec_buy") / (pl.col("bg_p50") + EPS)).alias("pt_rec_over_p50"),
        ((pl.col("rec_buy") - pl.col("bg_p50")) / pl.col("pt_rho_buy")).alias("pt_od_dev_cyc"),
        # «сколько своих циклов пользователь молчит» — ожидаемое число пропущенных покупок
        (pl.col("rec_buy") / pl.col("pt_rho_buy")).alias("pt_cyc_since_buy"),
        (pl.col("rec_any") / pl.col("pt_rho_act")).alias("pt_cyc_since_act"),
    ]
    # --- группа 3: стандартные окна, выраженные в личных циклах
    d += [(30.0 / pl.col("pt_rho_buy")).alias("pt_cyc30"),
          (180.0 / pl.col("pt_rho_buy")).alias("pt_cyc180")]
    # --- группа 1: окна в личном времени
    # длина окна обрезана глубиной истории cutoff'а (92 дня в начале коридора,
    # 409 на тесте), поэтому счётчики делятся на ФАКТИЧЕСКОЕ число циклов —
    # иначе признак снова станет функцией cutoff'а, как w365_* до нормировки (E10)
    for k in KS:
        cyc_b = pl.min_horizontal(float(k) * pl.col("pt_rho_buy"), pl.lit(avail)) / pl.col("pt_rho_buy")
        cyc_a = pl.min_horizontal(float(k) * pl.col("pt_rho_act"), pl.lit(avail)) / pl.col("pt_rho_act")
        d += [
            (pl.col(f"_b{k}_days_buy") / (cyc_b + EPS)).alias(f"pt_w{k}_buyrate"),
            (pl.col(f"_b{k}_gmv") / (cyc_b + EPS)).log1p().alias(f"pt_w{k}_lgmv_cyc"),
            (pl.col(f"_a{k}_days_present") / (cyc_a + EPS)).alias(f"pt_w{k}_presrate"),
        ]
    # --- группа 4: сдвиг личного ритма и интенсивность относительно своей базы
    d += [
        pl.col("bg_recent3").alias("pt_gap_recent3"),
        (pl.col("bg_recent3") / pl.col("pt_rho_buy")).alias("pt_rhythm_drift"),
        (pl.col("rec_buy") / (pl.col("bg_recent3") + EPS)).alias("pt_rec_over_recent3"),
    ]
    kf, kl = KS[0], KS[-1]
    d += [
        (pl.col(f"_b{kf}_days_buy") * float(kl) / (pl.col(f"_b{kl}_days_buy") * float(kf) + EPS)
         ).alias("pt_buyrate_drift"),
        (pl.col(f"_a{kf}_searches") * float(kl) / (pl.col(f"_a{kl}_searches") * float(kf) + EPS)
         ).alias("pt_int_srch"),
        (pl.col(f"_a{kf}_carts") * float(kl) / (pl.col(f"_a{kl}_carts") * float(kf) + EPS)
         ).alias("pt_int_cart"),
        (pl.col(f"_a{kf}_days_present") * float(kl)
         / (pl.col(f"_a{kl}_days_present") * float(kf) + EPS)).alias("pt_int_pres"),
    ]
    out = (p.with_columns(d).select(["user_id"] + ALL_COLS)
           .with_columns([pl.col(c).cast(pl.Float32) for c in ALL_COLS])
           .sort("user_id"))
    return out.collect()


def ptime_cached(T: dt.date, L: int | None = None, variant: str = "real") -> pl.DataFrame:
    p = DATA_PROCESSED / f"pt_{_tag(T)}_L{L}_{variant}.parquet"
    if p.exists():
        return pl.read_parquet(p)
    f = build_ptime(T, L, variant)
    f.write_parquet(p)
    return f


def selftest(T: dt.date = dt.date(2025, 9, 16), L: int | None = None) -> None:
    """Лукап-тест в том же виде, что и `src/smoke.py` §3, плюс проверка контроля C."""
    from src import data
    df = data.load()
    full = build_ptime(T, L, "real")
    data._CACHE["df"] = df.filter(pl.col("event_date") <= T)
    trunc = build_ptime(T, L, "real")
    data._CACHE["df"] = df
    assert full.equals(trunc), "build_ptime(T) зависит от строк с event_date > T — ЛУКАП"
    print(f"  [OK ] build_ptime({T}) не зависит от строк с event_date > T "
          f"({full.height:,} строк x {full.width - 1} признаков)")

    shuf = build_ptime(T, L, "shuf")
    same = float((full["pt_rho_buy"] == shuf["pt_rho_buy"]).mean())
    m_full = full["pt_rho_buy"].drop_nulls().mean()
    m_shuf = shuf["pt_rho_buy"].drop_nulls().mean()
    assert same < 0.05, f"перестановка почти ничего не сдвинула: совпало {same:.1%}"
    assert abs(m_full - m_shuf) < 1e-3, "маргинальное распределение rho не сохранилось"
    print(f"  [OK ] контроль C: rho совпала у {same:.2%} пользователей, "
          f"mean(rho) {m_full:.4f} против {m_shuf:.4f}")

    r = full.select([pl.col(c).is_not_null().mean().alias(c) for c in ALL_COLS]).row(0)
    worst = sorted(zip(ALL_COLS, r), key=lambda t: t[1])[:4]
    print("  заполненность (4 худших): "
          + ", ".join(f"{c} {v:.1%}" for c, v in worst))


def main():
    ap = argparse.ArgumentParser(description="Прогрев кэша признаков личного времени")
    ap.add_argument("--L", type=int, default=0, help="глубина истории; 0 = без усечения")
    ap.add_argument("--min-history", type=int, default=90)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--variant", default="real", choices=["real", "shuf"])
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    L = None if a.L <= 0 else a.L
    load()
    if a.selftest:
        selftest(L=L)
        return
    from src.config import CUTOFF_STEP, CUTOFF_TEST, cutoff_grid
    cuts = cutoff_grid(a.min_history, a.step or CUTOFF_STEP) + [CUTOFF_TEST]
    for T in cuts:
        t = time.time()
        cached = (DATA_PROCESSED / f"pt_{_tag(T)}_L{L}_{a.variant}.parquet").exists()
        f = ptime_cached(T, L, a.variant)
        print(f"{T}  n={f.height:>7,}  feats={f.width - 1:>3}  "
              f"{'cached' if cached else f'{time.time() - t:5.1f}s'}", flush=True)


if __name__ == "__main__":
    main()
