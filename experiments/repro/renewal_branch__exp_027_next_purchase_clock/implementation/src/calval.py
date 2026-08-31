"""STRATEGY_04 — интенсивная компонента, обученная на всём календаре.

Пространство имён артефактов — `S04-*`; общие модули проекта не изменяются.

Гипотеза (`research/strategies/STRATEGY_04_intensive_full_calendar.md`):
`E[z|x] = P(y>0|x) * E[z | y>0, x]`, и правило отбора панели (активность в каждом
из трёх блоков `2025-11-16..2026-02-13`) искажает ТОЛЬКО первый множитель. Тогда
вторую голову можно учить и на 120 днях, выброшенных из коридора, — это 13
дополнительных cutoff'ов, ближайших к тесту.

Три источника данных для интенсивной головы:

    CLEAN  2025-04-03..2025-10-16, шаг 7   — как сейчас (правило T+30 <= V)
    EXTRA  2025-10-22..2026-01-14, шаг 7   — 13 cutoff'ов, target-окна <= 2026-02-13
    EARLY  промежутки сетки шага 3 в начале коридора — контроль ОБЪЁМА

**Анти-лукап.** `EXTRA` лежит в будущем относительно любого val-фолда, поэтому
строки оттуда берутся только у пользователей группы B (`hash(user_id) % 2 == 1`),
а метрика считается на группе A. Полнопанельный честный OOF получается
кросс-фиттингом: предсказания группы A даёт модель с `EXTRA` от группы B и
наоборот. `CLEAN` участвует целиком в обеих моделях — это штатный режим
пайплайна (`T + 30 <= V`).

Цель обучения интенсива центрирована по cutoff'у: `z~ = log1p(y) - c(T)`,
`c(T) = mean log1p(y)` по строкам `y>0` этого cutoff'а. Уровень восстанавливается
средним по `CLEAN`: `z^ = p^ * (mu^ + c^)`. Без центрирования модель выучила бы
новогодний уровень `EXTRA`.

Запуск:
    python -m src.calval audit
    python -m src.calval run   --val 2025-10-16 2025-09-04 --seed 42
    python -m src.calval merge --seed 42
    python -m src.calval gap   --val 2025-10-16
    python -m src.calval blend
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import time

import numpy as np
import polars as pl

from src.config import (ARTIFACTS, CORRIDOR_END, DATA_END, LGB_PARAMS, SEED, TARGET_DAYS,
                        VAL_FOLDS_S1, cutoff_grid)
from src.data import load
from src.features import feature_names, make_xy, panel_users, target, to_np
from src.report import evaluate, format_report, save_report
from src.tracking import save_oof
from src.validation import calibrate, rmsle_z, wcv

NS = "S04"
EXTRA_END = dt.date(2026, 1, 14)     # позднейший легальный cutoff: (T, T+30] <= 2026-02-13
ROUNDS = 600                          # как `two_part` в src/models.py (спека стратегии)
SNAPS = (200, 300, 450, 600)          # срезы интенсивной головы: контроль ёмкости
TRAIN_BLOCKS = 1                      # панель обучения 1-блочная (как S1-E10/E02)
VAL_BLOCKS = 3                        # панель валидации — всегда как в тесте
VARIANTS = ("A", "B_A", "B_B", "C")

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


# --------------------------------------------------------------------- cutoff'ы
def clean_cutoffs(V: dt.date | None = None) -> list[dt.date]:
    """Штатный чистый коридор; при заданном V — обучающая выборка фолда (T+30 <= V)."""
    g = cutoff_grid(90, 7)
    if V is None:
        return g
    return [T for T in g if T + dt.timedelta(days=TARGET_DAYS) <= V]


def extra_cutoffs(step: int = 7, end: dt.date = EXTRA_END) -> list[dt.date]:
    """13 cutoff'ов вне коридора, отсчитанных назад от `end`.

    Легальность: у самого позднего target-окно `(T, T+30]` кончается ровно на
    границе данных, поэтому относительно теста лукапа нет по построению.
    """
    out, T = [], end
    while T > CORRIDOR_END:
        out.append(T)
        T -= dt.timedelta(days=step)
    out = sorted(out)
    for T in out:
        assert T > CORRIDOR_END, f"{T} лежит внутри чистого коридора"
        assert T + dt.timedelta(days=TARGET_DAYS) <= DATA_END, (
            f"{T}: target-окно выходит за границу данных {DATA_END}")
    return out


def early_cutoffs() -> list[dt.date]:
    """Промежутки сетки шага 3 в начале коридора: контроль объёма (вариант C).

    Это cutoff'ы шага 3, не попавшие в боевую сетку шага 7. Все они лежат внутри
    коридора и удовлетворяют `T + 30 <= V` для любого фолда S1, то есть добавляют
    ровно объём и ничего больше.
    """
    g7 = set(cutoff_grid(90, 7))
    return [T for T in cutoff_grid(90, 3) if T not in g7]


# --------------------------------------------------------- расщепление по пользователям
def user_group(uid) -> np.ndarray:
    """True = группа B (доноры строк вне коридора). splitmix64, target-free.

    Умножение `user_id` на нечётную константу сохраняет чётность, поэтому
    смешивание обязано быть нелинейным — иначе «хеш» вырождается в `uid % 2`.
    """
    h = np.asarray(uid, dtype=np.uint64).copy()
    with np.errstate(over="ignore"):
        h += np.uint64(0x9E3779B97F4A7C15)
        h ^= h >> np.uint64(30)
        h *= np.uint64(0xBF58476D1CE4E5B9)
        h ^= h >> np.uint64(27)
        h *= np.uint64(0x94D049BB133111EB)
        h ^= h >> np.uint64(31)
    return (h & np.uint64(1)).astype(bool)


# ------------------------------------------------------------------ панель и таргет
_PT: dict = {}


def panel_target(T: dt.date, blocks: int = TRAIN_BLOCKS):
    """(user_id, y) на cutoff'е без чтения признаков — нужно, чтобы заранее знать
    размеры блоков и не держать в памяти лишнюю матрицу."""
    k = (T, blocks)
    if k not in _PT:
        u = panel_users(T, blocks) if blocks else None
        assert u is not None, "blocks=0 в этой стратегии не используется"
        y = target(T, u, TARGET_DAYS)["y"].to_numpy()
        _PT[k] = (u["user_id"].to_numpy(), y)
    return _PT[k]


def release_raw():
    """Сырьё (3.06 ГБ) держать во время бустинга незачем.

    Панели и признаки лежат в parquet-кэше, таргеты — в `_PT`. Освобождение
    снимает пик на треть; без него матрица фолда 10-16 (4.5 ГБ) вместе с сырьём
    и `Dataset` уводит процесс в своп, и LightGBM получает меньше одного ядра.
    """
    from src import data as _data
    if _data._CACHE.pop("df", None) is not None:
        gc.collect()


def xy_frame(T: dt.date, blocks: int) -> pl.DataFrame:
    """Панель x признаки только из кэша, без обращения к сырью.

    Повторяет `features.make_xy` в части признаков; таргет берётся из `_PT`,
    поэтому вызов не требует `train.parquet` в памяти.
    """
    from src.features import features_cached
    f = features_cached(T, None, norm_long=True)
    u = panel_users(T, blocks)
    X = u.join(f, on="user_id", how="left").sort("user_id")
    return X.with_columns([pl.col(c).cast(pl.Float32) for c in X.columns if c != "user_id"])


def warm(cuts: list[dt.date], vals: list[dt.date] = ()):
    load()
    for T in cuts:
        panel_target(T, TRAIN_BLOCKS)
    for V in vals:
        panel_target(V, VAL_BLOCKS)


def row_mask(T: dt.date, positive: bool, group: bool | None):
    """Маска строк cutoff'а: `y>0` и/или принадлежность группе."""
    uid, y = panel_target(T)
    m = (y > 0) if positive else np.ones(len(y), bool)
    if group is not None:
        m &= (user_group(uid) == group)
    return m


# ------------------------------------------------------------------------ сборка
def assemble(specs, feats, kind: str):
    """specs = [(T, blocks, mask)]. kind: 'binary' -> 1[y>0]; 'centered' -> z - c(T)."""
    sizes = [int(m.sum()) for _, _, m in specs]
    n = sum(sizes)
    X = np.empty((n, len(feats)), np.float32)
    v = np.empty(n, np.float64)
    levels: dict[dt.date, float] = {}
    i = 0
    for (T, b, m), k in zip(specs, sizes):
        if k == 0:
            continue
        Xb = xy_frame(T, b)
        uid_ref, yb = panel_target(T, b)
        y_ref = yb
        assert np.array_equal(Xb["user_id"].to_numpy(), uid_ref), f"{T}: порядок строк"
        assert np.array_equal(yb, y_ref), f"{T}: таргет разошёлся"
        miss = [c for c in feats if c not in Xb.columns]
        assert not miss, f"{T}: нет признаков {miss[:5]}"
        A = to_np(Xb, feats) if m.all() else to_np(Xb.filter(pl.Series(m)), feats)
        assert A.shape[0] == k
        X[i:i + k] = A
        yy = yb[m]
        if kind == "binary":
            v[i:i + k] = (yy > 0).astype(np.float64)
        else:
            assert (yy > 0).all(), f"{T}: в интенсивную выборку попали нули"
            z = np.log1p(yy)
            levels[T] = float(z.mean())
            v[i:i + k] = z - levels[T]
        i += k
        del A, Xb, yb
    assert i == n
    return X, v, levels


def fit_head(box: list, v, objective: str, rounds: int, seed: int):
    """Матрица биннится и освобождается ДО бустинга (пик памяти -4.5 ГБ)."""
    import lightgbm as lgb
    p = dict(LGB_PARAMS)
    p["seed"] = seed
    if objective == "binary":
        p.update(objective="binary", metric="binary_logloss")
    ds = lgb.Dataset(box[0], v, params=p).construct()
    box[0] = None
    gc.collect()
    m = lgb.train(p, ds, num_boost_round=rounds)
    del ds
    gc.collect()
    return m


# --------------------------------------------------------------------------- прогон
def intensive_specs(clean: list[dt.date], extra: list[dt.date], group: bool | None):
    """CLEAN целиком + `extra` только от указанной группы; строки только `y>0`."""
    specs = [(T, TRAIN_BLOCKS, row_mask(T, True, None)) for T in clean]
    specs += [(T, TRAIN_BLOCKS, row_mask(T, True, group)) for T in extra]
    return specs


def matched_early(V: dt.date, n_target: int, group: bool | None):
    """Самые ранние промежуточные cutoff'ы, набирающие `n_target` строк `y>0`."""
    out, tot = [], 0
    for T in early_cutoffs():
        if T + dt.timedelta(days=TARGET_DAYS) > V:
            break
        k = int(row_mask(T, True, group).sum())
        out.append(T)
        tot += k
        if tot >= n_target:
            break
    return out, tot


def run_fold(V: dt.date, seed: int, rounds: int, variants: tuple[str, ...], max_clean=None,
             diag: bool = True, p_from_seed: int | None = None):
    cuts_clean = clean_cutoffs(V)
    if max_clean:
        cuts_clean = cuts_clean[-max_clean:]
    cuts_extra = extra_cutoffs()
    log(f"фолд {V}: CLEAN {len(cuts_clean)} ({cuts_clean[0]}..{cuts_clean[-1]}), "
        f"EXTRA {len(cuts_extra)} ({cuts_extra[0]}..{cuts_extra[-1]})")
    for T in cuts_clean:
        assert T + dt.timedelta(days=TARGET_DAYS) <= V, f"CLEAN {T} нарушает T+30<=V"

    # таргеты всех cutoff'ов считаются заранее, чтобы сырьё можно было отпустить
    warm(sorted(set(cutoff_grid(90, 7)) | set(cuts_extra)), [V])
    n_extra_b = sum(int(row_mask(T, True, True).sum()) for T in cuts_extra)
    cuts_early, n_early = matched_early(V, n_extra_b, True)
    release_raw()
    log(f"  таргеты предпосчитаны, сырьё отпущено ({len(_PT)} cutoff'ов в кэше)")

    Xv = xy_frame(V, VAL_BLOCKS)
    uid_v, yv = panel_target(V, VAL_BLOCKS)
    feats = feature_names(Xv)
    grp_v = user_group(uid_v)
    w180 = Xv["w180_days_buy"].to_numpy().astype(np.float32)
    rec_buy = Xv["rec_buy"].to_numpy().astype(np.float32)
    Av = to_np(Xv, feats)
    del Xv
    gc.collect()
    log(f"  валидация: {len(yv):,} строк, группа A {int((~grp_v).sum()):,}, "
        f"признаков {len(feats)}")

    # --- экстенсивная голова: ТОЛЬКО CLEAN, одна на все варианты --------------
    # при усреднении по сидам её переобучать не нужно: она одинакова у A/B/C, и
    # фиксация её сидом 42 делает парное сравнение A-B чище (шум только интенсива)
    if p_from_seed is not None:
        d0 = np.load(ARTIFACTS / f"{NS}_fold_{V.isoformat()}_s{p_from_seed}.npz",
                     allow_pickle=False)
        assert np.array_equal(d0["user_id"], uid_v), "чужой порядок строк в опорном прогоне"
        p_hat = d0["p_hat"].astype(np.float64)
        log(f"  экстенсив взят из прогона seed={p_from_seed}, mean p={p_hat.mean():.4f}")
        return _run_intensive(V, seed, rounds, variants, cuts_clean, cuts_extra,
                              cuts_early, n_extra_b, n_early, feats,
                              Av, yv, uid_v, grp_v, w180, rec_buy, p_hat, diag)

    t = time.time()
    Xb, vb, _ = assemble([(T, TRAIN_BLOCKS, row_mask(T, False, None)) for T in cuts_clean],
                         feats, "binary")
    log(f"  бинарная матрица {Xb.shape[0]:,} x {Xb.shape[1]}, доля y>0 {vb.mean():.4f}")
    box = [Xb]
    del Xb
    clf = fit_head(box, vb, "binary", rounds, seed)
    p_hat = clf.predict(Av)
    log(f"  экстенсив обучен за {time.time() - t:.0f}s, mean p={p_hat.mean():.4f}")
    del vb, box, clf
    gc.collect()
    return _run_intensive(V, seed, rounds, variants, cuts_clean, cuts_extra,
                          cuts_early, n_extra_b, n_early, feats,
                          Av, yv, uid_v, grp_v, w180, rec_buy, p_hat, diag)


def _run_intensive(V, seed, rounds, variants, cuts_clean, cuts_extra, cuts_early,
                   n_extra_b, n_early, feats,
                   Av, yv, uid_v, grp_v, w180, rec_buy, p_hat, diag):
    log(f"  EXTRA(группа B) строк {n_extra_b:,}; EARLY {len(cuts_early)} cutoff'ов "
        f"({cuts_early[0]}..{cuts_early[-1]}) даёт {n_early:,} "
        f"({n_early / max(n_extra_b, 1):.3f} от объёма)")

    plans = {"A":   ([], None),
             "B_A": (cuts_extra, True),
             "B_B": (cuts_extra, False),
             "C":   (cuts_early, True)}

    out = dict(val=V.isoformat(), seed=seed, rounds=rounds, user_id=uid_v, y=yv,
               group_b=grp_v, p_hat=p_hat, w180_days_buy=w180, rec_buy=rec_buy,
               n_extra_b=n_extra_b, n_early=n_early,
               cuts_clean=[c.isoformat() for c in cuts_clean],
               cuts_extra=[c.isoformat() for c in cuts_extra],
               cuts_early=[c.isoformat() for c in cuts_early])

    for name in variants:
        extra, group = plans[name]
        t = time.time()
        Xi, vi, levels = assemble(intensive_specs(cuts_clean, extra, group), feats, "centered")
        c_hat = float(np.mean([levels[T] for T in cuts_clean]))
        n_rows = Xi.shape[0]
        box = [Xi]
        del Xi
        reg = fit_head(box, vi, "regression", rounds, seed)
        snaps = sorted({k for k in SNAPS if k <= rounds} | {rounds})
        mu = {k: reg.predict(Av, num_iteration=k) for k in snaps}
        out[f"mu_{name}"] = mu[rounds].astype(np.float32)
        out[f"c_{name}"] = c_hat
        out[f"n_{name}"] = n_rows
        for k, m in mu.items():
            out[f"z_{name}_r{k}"] = np.maximum(p_hat * (m + c_hat), 0.0).astype(np.float32)
        z = out[f"z_{name}_r{rounds}"]
        a = ~grp_v
        log(f"  {name}: строк {n_rows:,}, c^={c_hat:.4f}, "
            f"RMSLE(A)={rmsle_z(yv[a], z[a]):.5f} cal={calibrate(yv[a], z[a])[1]:.5f} "
            f"| полная панель cal={calibrate(yv, z)[1]:.5f}  [{time.time() - t:.0f}s]")
        if name == "A":
            out["levels_clean"] = json.dumps({k.isoformat(): v for k, v in levels.items()})
            if diag:
                _poison_diag(V, reg, c_hat, feats, out, rounds)
        del reg, box, vi
        gc.collect()

    p = ARTIFACTS / f"{NS}_fold_{V.isoformat()}_s{seed}.npz"
    np.savez_compressed(p, **out)
    log(f"  сохранено: {p.name}")
    return out


def _poison_diag(V: dt.date, reg, c_hat: float, feats, out: dict, rounds: int):
    """Диагностика отравления интенсива (ключевая проверка допущения стратегии).

    Голова, обученная ТОЛЬКО на CLEAN, применяется к строкам `y>0`:
      * `EXTRA`     — будущее относительно V и отравленное правилом панели;
      * `HOLD`      — чистые cutoff'ы, оставшиеся за правилом `T+30<=V`, то есть
                      тоже будущее относительно V, но НЕ отравленные.
    `HOLD` калибрует масштаб: без него ноль остатка не с чем сравнивать.
    """
    hold = [T for T in cutoff_grid(90, 7) if T > V - dt.timedelta(days=TARGET_DAYS)]
    for tag, cuts in (("hold", hold), ("extra", extra_cutoffs())):
        if not cuts:
            continue
        specs = [(T, TRAIN_BLOCKS, row_mask(T, True, True)) for T in cuts]
        X, v, _ = assemble(specs, feats, "centered")
        mu = reg.predict(X, num_iteration=rounds)
        w = np.concatenate([xy_frame(T, TRAIN_BLOCKS)
                            .filter(pl.Series(row_mask(T, True, True)))["w180_days_buy"]
                            .to_numpy() for T in cuts])
        out[f"diag_{tag}_resid"] = (v - mu).astype(np.float32)
        out[f"diag_{tag}_w180"] = w.astype(np.float32)
        out[f"diag_{tag}_cut"] = np.concatenate(
            [np.full(int(row_mask(T, True, True).sum()), T.isoformat()) for T in cuts])
        log(f"  диагностика {tag}: n={len(v):,}, mean(z~ - mu^)={float((v - mu).mean()):+.5f}, "
            f"std={float((v - mu).std()):.4f}")
        del X, v, mu
        gc.collect()


# --------------------------------------------------------------------------- audit
def cmd_audit(a):
    load()
    ce, cl, ea = extra_cutoffs(), clean_cutoffs(), early_cutoffs()
    print(f"CLEAN  {len(cl)}: {cl[0]}..{cl[-1]}")
    print(f"EXTRA  {len(ce)}: {ce[0]}..{ce[-1]}   "
          f"target-окна кончаются {ce[0] + dt.timedelta(days=30)}..{ce[-1] + dt.timedelta(days=30)}")
    print(f"EARLY  {len(ea)}: {ea[0]}..{ea[-1]} (промежутки сетки шага 3)")
    print(f"граница данных {DATA_END}; конец коридора {CORRIDOR_END}\n")

    uid_all = load()["user_id"].unique().sort().to_numpy()
    g = user_group(uid_all)
    print(f"пользователей всего {len(uid_all):,}; группа B {g.mean():.5f} "
          f"(нечётных user_id {float((uid_all % 2 == 1).mean()):.5f} — контроль вырождения)")
    print(f"  corr(group_B, user_id % 2) = "
          f"{np.corrcoef(g.astype(float), (uid_all % 2).astype(float))[0, 1]:+.5f}\n")

    rows = []
    for tag, cuts in (("CLEAN", cl), ("EXTRA", ce)):
        for T in cuts:
            uid, y = panel_target(T)
            pos = y > 0
            z = np.log1p(y[pos])
            gb = user_group(uid)
            rows.append(dict(kind=tag, cutoff=T.isoformat(), n_panel=len(y),
                             p_positive=float(pos.mean()), c_T=float(z.mean()),
                             sd_z=float(z.std()), n_pos=int(pos.sum()),
                             n_pos_b=int((pos & gb).sum()),
                             y_win_end=(T + dt.timedelta(days=30)).isoformat()))
    df = pl.DataFrame(rows)
    out = ARTIFACTS / f"{NS}_audit_cutoffs.csv"
    df.write_csv(out)
    for tag in ("CLEAN", "EXTRA"):
        s = df.filter(pl.col("kind") == tag)
        print(f"{tag}: панель {s['n_panel'].min():,}..{s['n_panel'].max():,}  "
              f"P(y>0) {s['p_positive'].min():.4f}..{s['p_positive'].max():.4f} "
              f"(среднее {s['p_positive'].mean():.4f})  "
              f"c(T) {s['c_T'].min():.4f}..{s['c_T'].max():.4f} "
              f"(среднее {s['c_T'].mean():.4f})")
    print(f"\nтаблица: {out}")
    print(df.filter(pl.col("kind") == "EXTRA").select(
        ["cutoff", "n_panel", "p_positive", "c_T", "n_pos_b"]))

    # прямая проверка отсутствия лукапа в признаках
    df_raw = load()
    for T in [ce[0], ce[-1]]:
        f = make_xy(T, None, TRAIN_BLOCKS, norm_long=True)[0]
        u0 = int(f["user_id"][0])
        ev = df_raw.filter(pl.col("user_id") == u0)["event_date"]
        print(f"проверка {T}: признаки построены по {int((ev <= T).sum())} строкам "
              f"пользователя из {len(ev)}; событий после cutoff'а в окне признаков "
              f"{int((ev > T).sum())} (они не участвуют по построению build_features)")


# --------------------------------------------------------------------------- merge
def load_folds(seed: int, folds=None):
    folds = folds or VAL_FOLDS_S1
    out = []
    for V in folds:
        p = ARTIFACTS / f"{NS}_fold_{V.isoformat()}_s{seed}.npz"
        assert p.exists(), f"нет {p.name} — прогон фолда {V} не сделан"
        out.append(dict(np.load(p, allow_pickle=False)))
    return out


def stack(ds, key):
    return np.concatenate([d[key] for d in ds])


def variant_z(ds, name: str, rounds: int = ROUNDS):
    """z варианта на полной панели. Для B — кросс-фиттинг по половинам."""
    if name != "B":
        return stack(ds, f"z_{name}_r{rounds}")
    out = []
    for d in ds:
        gb = d["group_b"]
        z = np.where(gb, d[f"z_B_B_r{rounds}"], d[f"z_B_A_r{rounds}"])
        out.append(z)
    return np.concatenate(out)


def variant_mu(ds, name: str):
    if name != "B":
        return stack(ds, f"mu_{name}")
    return np.concatenate([np.where(d["group_b"], d["mu_B_B"], d["mu_B_A"]) for d in ds])


def cmd_merge(a):
    ds = load_folds(a.seed)
    uid, y = stack(ds, "user_id"), stack(ds, "y")
    cut = np.concatenate([np.full(len(d["y"]), str(d["val"])) for d in ds])
    grp = stack(ds, "group_b")
    A = ~grp
    print(f"строк OOF {len(y):,}; группа A {int(A.sum()):,} ({A.mean():.4f})\n")

    names = [("A", "S04-A"), ("B", "S04-B"), ("C", "S04-C")]
    reports = {}
    for short, exp in names:
        z = variant_z(ds, short, a.rounds)
        rep = evaluate(y, z, cut)
        rep_a = evaluate(y[A], z[A], cut[A])
        reports[short] = dict(full=rep, groupA=rep_a, z=z)
        save_oof(exp, uid, cut, z, y)
        save_report(exp, rep, extra=dict(description=f"STRATEGY_04 вариант {short}",
                                         model="two_part_centered", seed=a.seed,
                                         rounds=a.rounds, group_a_wcv=rep_a["wcv"]))

    print(f"{'вариант':<8}{'wCV (группа A)':>16}{'wCV (полная)':>15}   пофолдово, группа A")
    for short, _ in names:
        r = reports[short]
        print(f"{short:<8}{r['groupA']['wcv']:>16.5f}{r['full']['wcv']:>15.5f}   "
              + " ".join(f"{v:.5f}" for v in r["groupA"]["fold_cal"]))

    base = reports["A"]
    print("\nдельты к варианту A (главная — B; C отделяет объём от близости к тесту):")
    for short in ("B", "C"):
        r = reports[short]
        d = [x - b for x, b in zip(r["groupA"]["fold_cal"], base["groupA"]["fold_cal"])]
        wins = sum(x < 0 for x in d)
        print(f"  {short}: dwCV(A-группа) {r['groupA']['wcv'] - base['groupA']['wcv']:+.5f}  "
              f"пофолдово " + " ".join(f"{x:+.5f}" for x in d)
              + f"  лучше на {wins}/4"
              + (" (включая 10-16)" if d[-1] < 0 else " (10-16 ХУЖЕ)"))
    print(f"\n  B против C: dwCV(A-группа) "
          f"{reports['B']['groupA']['wcv'] - reports['C']['groupA']['wcv']:+.5f}")

    # ёмкость: тот же замер на срезах интенсивной головы
    print("\nёмкость интенсивной головы (wCV на группе A, срез по раундам):")
    print(f"{'раундов':>8}" + "".join(f"{n:>10}" for n, _ in names))
    for k in SNAPS:
        if k > a.rounds:
            continue
        row = []
        for short, _ in names:
            z = variant_z(ds, short, k)
            row.append(evaluate(y[A], z[A], cut[A])["wcv"])
        print(f"{k:>8}" + "".join(f"{v:>10.5f}" for v in row))

    _dump_fold_csv(ds, reports, y, cut, grp, a)
    return reports


def _dump_fold_csv(ds, reports, y, cut, grp, a):
    rows = []
    for short in ("A", "B", "C"):
        r = reports[short]
        for i, f in enumerate(r["groupA"]["per_fold"]):
            rows.append(dict(variant=short, fold=f["cutoff"], n_groupA=f["n"],
                             rmsle=f["rmsle"], rmsle_cal=f["rmsle_cal"], bias=f["bias"],
                             mean_z=f["mean_z"],
                             rmsle_cal_full=r["full"]["per_fold"][i]["rmsle_cal"]))
    p = ARTIFACTS / f"{NS}_fold_metrics_s{a.seed}.csv"
    pl.DataFrame(rows).write_csv(p)
    print(f"\nпофолдовые метрики: {p}")


# ------------------------------------------------- диагностика отравления интенсива
BUY_BANDS = [(0, 0, "0"), (1, 1, "1"), (2, 3, "2-3"), (4, 7, "4-7"), (8, 15, "8-15"),
             (16, 30, "16-30"), (31, 10_000, "31+")]
REC_BANDS = [(-1, -1, "нет"), (0, 7, "0-7"), (8, 14, "8-14"), (15, 30, "15-30"),
             (31, 60, "31-60"), (61, 90, "61-90"), (91, 180, "91-180"), (181, 10_000, "180+")]


def band(v, bands):
    out = np.full(len(v), "n/a", dtype=object)
    for lo, hi, name in bands:
        out[(v >= lo) & (v <= hi)] = name
    return np.asarray(out, dtype="U8")


def cmd_diag(a):
    """Прямая проверка допущения §2 стратегии: отравлен ли интенсив.

    Голова обучена только на CLEAN. Если правило панели не искажает
    `E[z | y>0]`, средний остаток на `EXTRA` обязан совпадать с остатком на
    таких же будущих, но чистых cutoff'ах (`hold`).
    """
    ds = load_folds(a.seed)
    rows = []
    for d in ds:
        V = str(d["val"])
        for tag in ("hold", "extra"):
            key = f"diag_{tag}_resid"
            if key not in d:
                continue
            r, w = d[key].astype(float), d[f"diag_{tag}_w180"].astype(float)
            b = band(w, BUY_BANDS)
            rows.append(dict(fold=V, source=tag, stratum="ВСЕ", n=len(r),
                             mean_resid=float(r.mean()), sd_resid=float(r.std()),
                             se=float(r.std() / np.sqrt(len(r)))))
            for _, _, name in BUY_BANDS:
                m = b == name
                if m.sum() < 500:
                    continue
                rows.append(dict(fold=V, source=tag, stratum=name, n=int(m.sum()),
                                 mean_resid=float(r[m].mean()), sd_resid=float(r[m].std()),
                                 se=float(r[m].std() / np.sqrt(m.sum()))))
    df = pl.DataFrame(rows)
    p = ARTIFACTS / f"{NS}_poison_diag_s{a.seed}.csv"
    df.write_csv(p)

    piv = (df.pivot(values="mean_resid", index=["fold", "stratum"], on="source")
           .with_columns((pl.col("extra") - pl.col("hold")).alias("extra_minus_hold")))
    print("средний остаток z~ - mu^ головы, обученной ТОЛЬКО на CLEAN")
    print("  hold  = чистые cutoff'ы, отсечённые правилом T+30<=V (будущее, НЕ отравлено)")
    print("  extra = 13 cutoff'ов вне коридора (будущее, отравлено правилом панели)")
    print("  порог закрытия стратегии: систематический сдвиг > 0.03 в лог-пространстве\n")
    with pl.Config(tbl_rows=100, tbl_width_chars=120):
        print(piv)
    tot = df.filter(pl.col("stratum") == "ВСЕ")
    print("\nпо фолдам, все страты вместе:")
    for V in sorted(set(tot["fold"].to_list())):
        s = tot.filter(pl.col("fold") == V)
        h = s.filter(pl.col("source") == "hold")
        e = s.filter(pl.col("source") == "extra")
        if h.height and e.height:
            print(f"  {V}: hold {h['mean_resid'][0]:+.5f} (+-{h['se'][0]:.5f})   "
                  f"extra {e['mean_resid'][0]:+.5f} (+-{e['se'][0]:.5f})   "
                  f"разница {e['mean_resid'][0] - h['mean_resid'][0]:+.5f}")
    print(f"\nтаблица: {p}")


# ------------------------------------------------------------------ сегменты и качество
def cmd_seg(a):
    ds = load_folds(a.seed)
    y = stack(ds, "y")
    cut = np.concatenate([np.full(len(d["y"]), str(d["val"])) for d in ds])
    grp = stack(ds, "group_b")
    A = ~grp
    ly = np.log1p(y)
    pos = y > 0
    p_hat = stack(ds, "p_hat")
    w180 = stack(ds, "w180_days_buy")
    rec = np.nan_to_num(stack(ds, "rec_buy"), nan=-1.0)
    fw = {f: w for f, w in zip([d.isoformat() for d in VAL_FOLDS_S1], [1.0, 2.0, 4.0, 8.0])}
    wrow = np.array([fw[c] for c in cut])

    Z = {n: variant_z(ds, n, a.rounds) for n in ("A", "B", "C")}
    MU = {n: variant_mu(ds, n) for n in ("A", "B", "C")}
    # c^ — свой у каждого фолда (среднее уровня по его CLEAN-выборке)
    c_hat = np.concatenate([np.full(len(d["y"]), float(d["c_A"])) for d in ds])
    for d in ds:      # уровень общий для вариантов: CLEAN-часть у них одна и та же
        assert abs(float(d["c_A"]) - float(d["c_B_A"])) < 1e-9

    def wm(v, m=None):
        m = np.ones(len(v), bool) if m is None else m
        return float(np.average(v[m], weights=wrow[m]))

    print(f"строк {len(y):,}; группа A {int(A.sum()):,}; y>0 {pos.mean():.4f}; "
          f"AUC общей экстенсивной головы {_auc(y, p_hat):.5f}; "
          f"c^ по фолдам {np.unique(np.round(c_hat, 4))}\n")

    print("КАЧЕСТВО УСЛОВНОЙ ВЕЛИЧИНЫ (только строки y>0, группа A) — то, что меняет стратегия")
    print(f"{'вариант':<8}{'RMSE(z+ , mu+c)':>18}{'sd остатка':>13}{'bias':>10}"
          f"{'RMSLE(z^) на y>0':>19}{'corr(mu, z+)':>14}")
    m = pos & A
    for n in ("A", "B", "C"):
        r = ly[m] - (MU[n][m] + c_hat[m])
        rbar = float(np.average(r, weights=wrow[m]))
        print(f"{n:<8}{np.sqrt(np.average(r ** 2, weights=wrow[m])):>18.5f}"
              f"{float(np.sqrt(np.average((r - rbar) ** 2, weights=wrow[m]))):>13.5f}"
              f"{rbar:>+10.5f}"
              f"{np.sqrt(np.average((ly[m] - Z[n][m]) ** 2, weights=wrow[m])):>19.5f}"
              f"{np.corrcoef(MU[n][m], ly[m])[0, 1]:>14.5f}")

    print("\nРАЗБИЕНИЕ ОШИБКИ y=0 / y>0 (группа A, веса фолдов 1:2:4:8)")
    print(f"{'вариант':<8}{'RMSLE y=0':>12}{'RMSLE y>0':>12}{'доля ошибки y=0':>18}"
          f"{'AUC(y>0) по z^':>17}")
    for n in ("A", "B", "C"):
        e = (ly - Z[n]) ** 2
        z0, z1 = A & ~pos, A & pos
        share = float((wrow[z0] * e[z0]).sum() / (wrow[A] * e[A]).sum())
        print(f"{n:<8}{np.sqrt(np.average(e[z0], weights=wrow[z0])):>12.5f}"
              f"{np.sqrt(np.average(e[z1], weights=wrow[z1])):>12.5f}{share:>18.4f}"
              f"{_auc(y[A], Z[n][A]):>17.5f}")

    print("\nПРОБЛЕМНЫЕ СЕГМЕНТЫ (группа A; дельта B-A и C-A по RMSLE, веса 1:2:4:8)")
    rows = []
    bb = band(w180, BUY_BANDS)
    rb = band(rec, REC_BANDS)
    segs = [(f"покупок180={nm} / y=0", A & (bb == nm) & ~pos) for _, _, nm in BUY_BANDS]
    segs += [(f"покупок180={nm} / y>0", A & (bb == nm) & pos) for _, _, nm in BUY_BANDS]
    segs += [(f"rec_buy {nm}", A & (rb == nm)) for _, _, nm in REC_BANDS]
    segs += [("полоса 2-15 покупок", A & np.isin(bb, ["2-3", "4-7", "8-15"])),
             ("rec_buy 15-60", A & np.isin(rb, ["15-30", "31-60"])),
             ("пересечение 2-15 и rec 15-60",
              A & np.isin(bb, ["2-3", "4-7", "8-15"]) & np.isin(rb, ["15-30", "31-60"]))]
    print(f"{'сегмент':<30}{'доля':>7}{'RMSLE A':>10}{'B-A':>10}{'C-A':>10}")
    for name, m in segs:
        if m.sum() < 1000:
            continue
        v = {n: float(np.sqrt(np.average((ly[m] - Z[n][m]) ** 2, weights=wrow[m])))
             for n in ("A", "B", "C")}
        rows.append(dict(segment=name, n=int(m.sum()), share=float(m.sum() / A.sum()),
                         rmsle_A=v["A"], rmsle_B=v["B"], rmsle_C=v["C"],
                         d_BA=v["B"] - v["A"], d_CA=v["C"] - v["A"]))
        print(f"{name:<30}{m.sum() / A.sum():>7.3f}{v['A']:>10.5f}"
              f"{v['B'] - v['A']:>+10.5f}{v['C'] - v['A']:>+10.5f}")
    pl.DataFrame(rows).write_csv(ARTIFACTS / f"{NS}_segments_s{a.seed}.csv")

    print("\nРАЗНООБРАЗИЕ (полная панель)")
    for x, ref in (("B", "A"), ("C", "A"), ("B", "C")):
        rx, rr = ly - Z[x], ly - Z[ref]
        print(f"  {x} против {ref}: Var(dz)={np.var(Z[x] - Z[ref]):.5f}  "
              f"corr предсказаний {np.corrcoef(Z[x], Z[ref])[0, 1]:.5f}  "
              f"corr остатков {np.corrcoef(rx, rr)[0, 1]:.5f}")
    print("  ориентиры exp_018: пол разнообразия Var=0.00712, потолок corr остатков 0.99885")
    print(f"\nсегменты: {ARTIFACTS / f'{NS}_segments_s{a.seed}.csv'}")


def _auc(y, s) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score((y > 0).astype(np.int8), s))


# ------------------------------------------------------------------- тест и сабмит
def cmd_predict(a):
    """z на тесте для S04-A и S04-B: экстенсив на CLEAN, интенсив на CLEAN(+EXTRA).

    На тесте расщепление по пользователям не нужно: `EXTRA` целиком в ПРОШЛОМ
    относительно 2026-02-13 (последнее target-окно кончается ровно на границе
    данных), поэтому строки берутся у всех.

    Экстенсивная голова обучается ОДИН раз и переиспользуется обоими вариантами:
    они обязаны отличаться ровно источником данных интенсива, иначе разность их
    скоров на LB перестанет быть измерением одной величины.
    """
    from src.config import CUTOFF_TEST
    cuts_clean, cuts_extra = clean_cutoffs(), extra_cutoffs()
    warm(cuts_clean + cuts_extra)
    load()
    panel_users(CUTOFF_TEST, VAL_BLOCKS)
    release_raw()
    Xt = xy_frame(CUTOFF_TEST, VAL_BLOCKS)
    feats = feature_names(Xt)
    At = to_np(Xt, feats)
    uid = Xt["user_id"].to_numpy()
    del Xt
    gc.collect()
    log(f"тестовая матрица {At.shape[0]:,} x {At.shape[1]}; CLEAN {len(cuts_clean)}, "
        f"EXTRA {len(cuts_extra)}")

    t = time.time()
    Xb, vb, _ = assemble([(T, TRAIN_BLOCKS, row_mask(T, False, None)) for T in cuts_clean],
                         feats, "binary")
    log(f"  бинарная матрица {Xb.shape[0]:,}, доля y>0 {vb.mean():.4f}")
    box = [Xb]; del Xb
    clf = fit_head(box, vb, "binary", a.rounds, a.seed)
    p_hat = clf.predict(At)
    del clf, box, vb; gc.collect()
    log(f"  экстенсив обучен за {time.time() - t:.0f}s, mean p={p_hat.mean():.4f}")
    np.save(ARTIFACTS / f"{NS}_ptest_s{a.seed}.npy", p_hat)

    plan = [("S04-A", []), ("S04-B", cuts_extra)]
    for name, extra in (plan if a.both else [(a.variant, cuts_extra if a.extra else [])]):
        t = time.time()
        Xi, vi, lv = assemble(intensive_specs(cuts_clean, extra, None), feats, "centered")
        c_hat = float(np.mean([lv[T] for T in cuts_clean]))
        n_rows = Xi.shape[0]
        box = [Xi]; del Xi
        reg = fit_head(box, vi, "regression", a.rounds, a.seed)
        z = np.maximum(p_hat * (reg.predict(At) + c_hat), 0.0)
        np.save(ARTIFACTS / f"ztest_{name}.npy", z)
        np.save(ARTIFACTS / f"uid_{name}.npy", uid)
        del reg, box, vi; gc.collect()
        log(f"  {name}: интенсив {n_rows:,} строк, c^={c_hat:.4f}, "
            f"mean(log1p(pred))={z.mean():.4f}  [{time.time() - t:.0f}s]")


# --------------------------------------------------------------------------- gap
def cmd_gap(a):
    """Ценность свежести для ИНТЕНСИВНОЙ головы: та самая ось, ради которой всё.

    На тесте `EXTRA` — прошлое, и её роль в том, что разрыв «последний обучающий
    cutoff -> прогноз» падает со 120 дней до 30. Локально это меряется прямо:
    фиксируем число cutoff'ов и отодвигаем их от val на G дней.
    """
    V = dt.date.fromisoformat(a.val)
    grid = cutoff_grid(90, 7)
    warm(grid, [V])
    release_raw()
    Xv = xy_frame(V, VAL_BLOCKS)
    _, yv = panel_target(V, VAL_BLOCKS)
    feats = feature_names(Xv)
    Av = to_np(Xv, feats)
    del Xv
    pos = yv > 0
    zp = np.log1p(yv[pos])
    rows = []
    for G in a.gaps:
        cuts = [T for T in grid if T + dt.timedelta(days=G) <= V][-a.k:]
        gap = (V - max(cuts)).days
        t = time.time()
        Xi, vi, lv = assemble([(T, TRAIN_BLOCKS, row_mask(T, True, None)) for T in cuts],
                              feats, "centered")
        c_hat = float(np.mean(list(lv.values())))
        n_i = Xi.shape[0]
        box = [Xi]; del Xi
        reg = fit_head(box, vi, "regression", a.rounds, a.seed)
        mu = reg.predict(Av) + c_hat
        cond = float(np.sqrt(np.mean((zp - mu[pos]) ** 2)))
        del reg, box, vi; gc.collect()

        Xb, vb, _ = assemble([(T, TRAIN_BLOCKS, row_mask(T, False, None)) for T in cuts],
                             feats, "binary")
        box = [Xb]; del Xb
        clf = fit_head(box, vb, "binary", a.rounds, a.seed)
        z = np.maximum(clf.predict(Av) * mu, 0.0)
        full = calibrate(yv, z)[1]
        del clf, box, vb; gc.collect()
        rows.append(dict(gap_requested=G, gap_actual=gap, n_cutoffs=len(cuts),
                         last_cutoff=max(cuts).isoformat(), n_rows_int=n_i, c_hat=c_hat,
                         cond_rmse_pos=cond, rmsle_cal=full))
        log(f"  G={G} (факт {gap}d, последний {max(cuts)}): условный RMSE(y>0)={cond:.5f}, "
            f"полная RMSLE_cal={full:.5f}  [{time.time() - t:.0f}s]")
    df = pl.DataFrame(rows)
    p = ARTIFACTS / f"{NS}_gap_probe_{V.isoformat()}.csv"
    df.write_csv(p)
    print(df)
    print(f"\nсохранено: {p}")


# --------------------------------------------------------------------------- blend
MIX = {"S1-E10": 0.15, "S1-E02": 0.30, "S1-E03a": 0.10, "S1-DIST": 0.45}


def cmd_blend(a):
    from src.blend import aligned, fold_cal_matrix, weight_grid
    from src.config import FOLD_WEIGHTS_S1

    exps = list(MIX) + [a.new]
    Z, y, cut = aligned(exps)
    ly = np.log1p(y)
    folds = sorted(set(cut.tolist()))
    masks = [cut == c for c in folds]
    w_f = np.asarray(FOLD_WEIGHTS_S1, float)
    w_f = w_f / w_f.sum()
    ref = np.array([MIX[e] for e in MIX] + [0.0])
    ref_fc = fold_cal_matrix(Z, ly, masks, [ref])[0]
    print(f"опорная смесь S1-DIST-MIX: wCV={float(np.dot(w_f, ref_fc)):.5f}   "
          + " ".join(f"{v:.5f}" for v in ref_fc))

    i_new, i_cap = len(exps) - 1, list(MIX).index("S1-E03a")
    R = np.vstack([ly - z for z in Z])
    z_mix = np.average(Z, axis=0, weights=ref)
    r_mix = ly - z_mix
    print(f"\nразнообразие {a.new} против смеси: Var(dz)={np.var(Z[i_new] - z_mix):.5f} "
          f"(пол сидов 0.00712), corr остатков {np.corrcoef(R[i_new], r_mix)[0, 1]:.5f} "
          f"(потолок двух сидов 0.99885)")
    for i, e in enumerate(exps[:-1]):
        print(f"  против {e:<9} Var(dz)={np.var(Z[i_new] - Z[i]):.5f}  "
              f"corr остатков {np.corrcoef(R[i_new], R[i])[0, 1]:.5f}")

    # страховка E03a зафиксирована ненулевой (exp_016 §6)
    ws = [w for w in weight_grid(len(exps), a.step) if abs(w[i_cap] - MIX["S1-E03a"]) < 1e-9]
    FC = fold_cal_matrix(Z, ly, masks, ws)
    sc = FC @ w_f
    o = np.argsort(sc)
    print(f"\nлучшие смеси при фиксированном весе S1-E03a=0.10 ({len(ws)} комбинаций, "
          f"шаг {a.step}):")
    for i in o[:6]:
        d = " ".join(f"{v - r:+.5f}" for v, r in zip(FC[i], ref_fc))
        print(f"  w={np.round(ws[i], 2)}  wCV={sc[i]:.5f}  d={float(np.dot(w_f, FC[i] - ref_fc)):+.5f}"
              f"  [{d}]  {(FC[i] < ref_fc).sum()}/4")

    print("\nLOFO: веса подобраны без фолда, проверены на нём")
    held = np.zeros(len(folds))
    for h in range(len(folds)):
        idx = [i for i in range(len(folds)) if i != h]
        wh = w_f[idx] / w_f[idx].sum()
        b = int(np.argmin(FC[:, idx] @ wh))
        held[h] = FC[b, h]
        print(f"  {folds[h]}  веса {np.round(ws[b], 2)}  на нём {FC[b, h]:.5f}  "
              f"опорная {ref_fc[h]:.5f}  дельта {FC[b, h] - ref_fc[h]:+.5f}")
    print(f"  ЧЕСТНЫЙ выигрыш LOFO по wCV: {float(np.dot(w_f, held - ref_fc)):+.5f}"
          f"   (в выборке {sc[o[0]] - float(np.dot(w_f, ref_fc)):+.5f})")

    print("\nподстановка при ФИКСИРОВАННЫХ весах, страховка S1-E03a=0.10 не трогается:")
    print(f"{'источник доли':<16}{'вес нового':>11}{'wCV':>10}{'дельта':>10}{'фолдов':>8}")
    for take in a.slots:
        for src, name in ((0, "S1-E10"), (3, "S1-DIST"), (-1, "пропорц. (кроме E03a)")):
            w = np.array([MIX[e] for e in MIX] + [0.0])
            if src >= 0:
                if w[src] < take - 1e-9:
                    continue
                w[src] -= take
            else:
                free = np.array([1.0, 1.0, 0.0, 1.0, 0.0])
                w -= take * w * free / float((w * free).sum())
            w[i_new] += take
            fc = fold_cal_matrix(Z, ly, masks, [w])[0]
            print(f"{name:<16}{take:>11.2f}{float(np.dot(w_f, fc)):>10.5f}"
                  f"{float(np.dot(w_f, fc - ref_fc)):>+10.5f}{(fc < ref_fc).sum():>6}/4")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="STRATEGY_04: интенсив на всём календаре")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("audit")
    p.set_defaults(fn=cmd_audit)

    p = sub.add_parser("run")
    p.add_argument("--val", nargs="+", required=True)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--rounds", type=int, default=ROUNDS)
    p.add_argument("--variants", nargs="+", default=list(VARIANTS))
    p.add_argument("--max-clean", type=int, default=None, help="дым: обрезать CLEAN")
    p.add_argument("--no-diag", action="store_true")
    p.add_argument("--p-from-seed", type=int, default=None,
                   help="взять экстенсивную голову из готового прогона (усреднение по сидам)")
    p.set_defaults(fn=lambda a: [run_fold(dt.date.fromisoformat(v), a.seed, a.rounds,
                                          tuple(a.variants), a.max_clean, not a.no_diag,
                                          a.p_from_seed)
                                 for v in a.val])

    p = sub.add_parser("merge")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--rounds", type=int, default=ROUNDS)
    p.set_defaults(fn=cmd_merge)

    p = sub.add_parser("diag")
    p.add_argument("--seed", type=int, default=SEED)
    p.set_defaults(fn=cmd_diag)

    p = sub.add_parser("seg")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--rounds", type=int, default=ROUNDS)
    p.set_defaults(fn=cmd_seg)

    p = sub.add_parser("predict")
    p.add_argument("--variant", default="S04-B")
    p.add_argument("--rounds", type=int, default=ROUNDS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--no-extra", dest="extra", action="store_false",
                   help="контрольный прогон без EXTRA (это вариант A на тесте)")
    p.add_argument("--one", dest="both", action="store_false",
                   help="считать только --variant вместо пары S04-A / S04-B")
    p.set_defaults(fn=cmd_predict, extra=True, both=True)

    p = sub.add_parser("gap")
    p.add_argument("--val", default="2025-10-16")
    p.add_argument("--gaps", nargs="+", type=int, default=[30, 60, 90, 120])
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--rounds", type=int, default=ROUNDS)
    p.add_argument("--seed", type=int, default=SEED)
    p.set_defaults(fn=cmd_gap)

    p = sub.add_parser("blend")
    p.add_argument("--new", default="S04-B")
    p.add_argument("--step", type=float, default=0.05)
    p.add_argument("--slots", nargs="*", type=float, default=[0.05, 0.10, 0.15])
    p.set_defaults(fn=cmd_blend)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
