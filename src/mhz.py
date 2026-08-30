"""MHZ — multi-horizon hazard + count supervision (exp_024).

Проверяемая гипотеза. Bottleneck метрики — не точность величины GMV, а момент
следующей покупки: 74% остаточной дисперсии боевой смеси — бернуллиевский член
«попадёт ли покупка в это 30-дневное окно» (`research/rmsle_diagnostics` §4).
`two_part` и `dist` моделируют этот процесс одной меткой `1[y>0]`. Здесь та же
история размечается богаче — моментом следующей покупки на шести горизонтах и
числом покупательных дней, — и проверяется, извлекает ли это новую информацию
из тех же 227 агрегатов `S1-E10`.

Четыре вспомогательные головы (минимальный набор, см. карточку exp_024):

  haz   multiclass K=7 по интервалам дней до следующей покупки
        1-7 / 8-14 / 15-21 / 22-30 / 31-45 / 46-60 / >60      горизонт 60
  cnt   multiclass K=7 по числу покупательных дней в 30 днях
        0 / 1 / 2 / 3 / 4 / 5-7 / 8+                          горизонт 30
  b30   binary 1[y30 > 0]                                     горизонт 30
  val   L2 на log1p(y) по строкам y > 0                       горизонт 30
  self  L2 на log1p(y) по всем строкам — КОНТРОЛЬ ЧЕСТНОСТИ   горизонт 30

Отдельные бинарники `buy_7/14/21/45/60` не обучаются: это строго вложенное
семейство, и кумулятивы `haz` дают всю кривую одной моделью вместо шести. Их
качество меряется на настоящих метках `buy_h` (см. `diag`).

Строгий OOF вспомогательных предсказаний — кросс-фиттинг ПО ПОЛЬЗОВАТЕЛЯМ
(hash % 2). Предсказание для строки (u, T) всегда приходит от модели, которая
пользователя `u` не видела ни на одном cutoff'е. Временной канал закрыт
отдельно: голова с горизонтом `h` обучается только на cutoff'ах `T + h <= V`,
поэтому для фолда `V` не используется ни один бит после `V`.

Лестница аблаций (все арки — один и тот же `direct`, 600 раундов, 227 базовых
признаков + аугментация, различие ТОЛЬКО в наборе aux-колонок):

  base  aux-колонки обнулены      — внутренняя опорная точка, воспроизводит S1-E10
  self  + selfz                   — выигрыш от механики стекинга как таковой
  p30   + b30/val/two-part        — 30-дневная супервизия, то есть `two_part` в признаках
  full  + вся кривая + счёт       — проверяемая гипотеза

Запуск (один фолд = один процесс, ~45-60 мин на 6 потоках):
  LGB_THREADS=6 python -m src.mhz fold --val 2025-10-16
  python -m src.mhz merge
  python -m src.mhz diag
  python -m src.mhz meta
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import time

import numpy as np
import polars as pl

from src.config import (ARTIFACTS, CUTOFF_STEP, FOLD_WEIGHTS_S1, LGB_PARAMS, SEED,
                        TARGET_DAYS, VAL_FOLDS_S1, cutoff_grid)
from src.data import load
from src.features import feature_names, make_xy, panel_users, to_np
from src.report import evaluate, format_report, save_report
from src.tracking import save_oof
from src.validation import bias_z, calibrate, rmsle_z

T0 = time.time()

# --- разметка -------------------------------------------------------------------
HAZ_H = 60                      # горизонт головы hazard
HAZ_EDGES = (7, 14, 21, 30, 45, 60)         # 7 классов, последний = «> 60»
HAZ_MID = np.array([4.0, 11.0, 18.0, 26.0, 38.0, 53.0, 75.0])   # для E[days]
CNT_EDGES = (1, 2, 3, 4, 5, 8)              # 0 | 1 | 2 | 3 | 4 | 5-7 | 8+
CNT_MID = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 10.0])        # для E[N]
K_HAZ = len(HAZ_EDGES) + 1
K_CNT = len(CNT_EDGES) + 1
EPS = 1e-6

# «отравленная» область: панель отобрана по активности в каждом из трёх блоков
# 2025-11-16..2026-02-13, поэтому метка, чьё окно её задевает, смещена (eda §3.2).
POISON_START = dt.date(2025, 11, 16)


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def user_half(user_ids) -> np.ndarray:
    """Детерминированное расщепление пользователей на две половины.

    Тот же хеш, что в `train.row_sample_mask`: один user_id получает одну и ту же
    половину на всех cutoff'ах и во всех фолдах, поэтому «модель не видела этого
    пользователя» — утверждение про весь прогон, а не про одну матрицу.
    """
    ids = np.asarray(user_ids, dtype=np.uint64)
    return ((ids * np.uint64(2654435761)) % np.uint64(1000) >= np.uint64(500)).astype(np.int8)


def _window(T: dt.date, h: int):
    return T + dt.timedelta(days=1), T + dt.timedelta(days=h)


def buy_gap(T: dt.date, uid_sorted: np.ndarray, max_h: int = HAZ_H) -> np.ndarray:
    """Дни до первой покупки после T: min{d >= 1 : gmv(T + d) > 0}, иначе max_h + 1.

    Тот же фильтр окна и то же условие `gmv > 0`, что в `features.target`,
    поэтому по лукапу метка эквивалентна таргету с горизонтом `max_h`.
    """
    a, b = _window(T, max_h)
    g = (load().lazy()
         .filter((pl.col("event_date") >= a) & (pl.col("event_date") <= b) & (pl.col("gmv") > 0))
         .group_by("user_id")
         .agg((pl.col("event_date").min() - pl.lit(T)).dt.total_days().alias("gap"))
         .collect())
    out = (pl.DataFrame({"user_id": uid_sorted}).join(g, on="user_id", how="left")
           .with_columns(pl.col("gap").fill_null(max_h + 1)).sort("user_id"))
    return out["gap"].to_numpy().astype(np.int16)


def buy_days(T: dt.date, uid_sorted: np.ndarray, h: int = TARGET_DAYS) -> np.ndarray:
    """Число дней с покупкой в окне (T, T + h]. Лог дневной, поэтому это счёт строк."""
    a, b = _window(T, h)
    g = (load().lazy()
         .filter((pl.col("event_date") >= a) & (pl.col("event_date") <= b) & (pl.col("gmv") > 0))
         .group_by("user_id").agg(pl.len().alias("n")).collect())
    out = (pl.DataFrame({"user_id": uid_sorted}).join(g, on="user_id", how="left")
           .with_columns(pl.col("n").fill_null(0)).sort("user_id"))
    return out["n"].to_numpy().astype(np.int16)


def haz_class(gap) -> np.ndarray:
    return np.searchsorted(HAZ_EDGES, np.asarray(gap), side="left").astype(np.int32)


def cnt_class(n) -> np.ndarray:
    return np.searchsorted(CNT_EDGES, np.asarray(n), side="right").astype(np.int32)


# --- набор aux-колонок ----------------------------------------------------------
HAZ_COLS = ["haz_p7", "haz_p14", "haz_p21", "haz_p30", "haz_p45", "haz_p60", "haz_edays",
            "haz_h2", "haz_h3", "haz_h4", "haz_h5", "haz_h6",
            "haz_lo7", "haz_lo30", "haz_lo60", "haz_sl730", "haz_sl3060"]
CNT_COLS = ["cnt_p0", "cnt_en", "cnt_ge2", "cnt_ge4", "cnt_mix"]
P30_COLS = ["b30_p", "b30_lo", "val_mu", "tp30"]
COMBO_COLS = ["tp_cnt"]
SELF_COLS = ["selfz"]
AUX_COLS = HAZ_COLS + CNT_COLS + P30_COLS + COMBO_COLS + SELF_COLS
N_AUX = len(AUX_COLS)
IX = {c: i for i, c in enumerate(AUX_COLS)}

VARIANTS = {
    "BASE": [],
    "SELF": SELF_COLS,
    "P30": P30_COLS,
    "FULL": HAZ_COLS + CNT_COLS + P30_COLS + COMBO_COLS,
}


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def aux_from_heads(ph: np.ndarray, pc: np.ndarray, pb: np.ndarray, mu: np.ndarray,
                   zs: np.ndarray, m_cnt: np.ndarray) -> np.ndarray:
    """Матрица aux-признаков из сырых выходов голов.

    `ph` (n, 7) — hazard; `pc` (n, 7) — счёт; `pb` (n,) — P(y>0); `mu` (n,) —
    условная величина; `zs` (n,) — контрольное предсказание того же таргета;
    `m_cnt` (7,) — средний z в бакете счёта, посчитанный ТОЛЬКО на train.
    """
    n = len(pb)
    A = np.zeros((n, N_AUX), np.float32)
    cum = np.cumsum(ph, axis=1)
    for j, name in enumerate(["haz_p7", "haz_p14", "haz_p21", "haz_p30", "haz_p45", "haz_p60"]):
        A[:, IX[name]] = cum[:, j]
    A[:, IX["haz_edays"]] = ph @ HAZ_MID
    # условная вероятность интервала при дожитии до него — собственно hazard
    for j in range(1, 6):
        A[:, IX[f"haz_h{j + 1}"]] = ph[:, j] / (1.0 - cum[:, j - 1] + EPS)
    A[:, IX["haz_lo7"]] = _logit(cum[:, 0])
    A[:, IX["haz_lo30"]] = _logit(cum[:, 3])
    A[:, IX["haz_lo60"]] = _logit(cum[:, 5])
    # форма кривой в лог-шансах: то, что обязано отличать поведенческие состояния
    # при одинаковом P(buy30)
    A[:, IX["haz_sl730"]] = A[:, IX["haz_lo30"]] - A[:, IX["haz_lo7"]]
    A[:, IX["haz_sl3060"]] = A[:, IX["haz_lo60"]] - A[:, IX["haz_lo30"]]

    A[:, IX["cnt_p0"]] = pc[:, 0]
    A[:, IX["cnt_en"]] = pc @ CNT_MID
    A[:, IX["cnt_ge2"]] = 1.0 - pc[:, 0] - pc[:, 1]
    A[:, IX["cnt_ge4"]] = pc[:, 4:].sum(axis=1)
    A[:, IX["cnt_mix"]] = pc @ m_cnt

    A[:, IX["b30_p"]] = pb
    A[:, IX["b30_lo"]] = _logit(pb)
    A[:, IX["val_mu"]] = mu
    A[:, IX["tp30"]] = pb * mu
    A[:, IX["tp_cnt"]] = (1.0 - pc[:, 0]) * mu
    A[:, IX["selfz"]] = zs
    return A


# --- сборка матрицы фолда -------------------------------------------------------
def fold_cutoffs(V: dt.date, h: int) -> list[dt.date]:
    """Cutoff'ы, у которых окно метки длины `h` целиком в прошлом относительно `V`.

    Обобщение правила `T + TARGET_DAYS <= V` из `validation.get_folds` на
    произвольный горизонт. Для h = 60 это отрезает 4 самых свежих cutoff'а.
    """
    return [T for T in cutoff_grid(90, CUTOFF_STEP) if T + dt.timedelta(days=h) <= V]


class FoldData:
    """Матрица обучения фолда, отсортированная по (половина пользователя, cutoff).

    Порядок строк выбран так, чтобы каждая обучающая подвыборка была НЕПРЕРЫВНЫМ
    срезом: half = 0 идёт первой, внутри половины cutoff'ы по возрастанию даты,
    поэтому строки с `T + 60 <= V` образуют префикс половины. Срез
    C-непрерывной матрицы — тоже C-непрерывный, и LightGBM берёт его без копии;
    на матрице 5 ГБ это разница между 6 и 9 ГБ пика.

    Aux-колонки выделены сразу и до заполнения содержат нули: константный столбец
    LightGBM биннит в один бин, поэтому головы обучаются на тех же 227 признаках,
    а число колонок (а с ним и `feature_fraction`) одинаково во всех арках.
    """

    def __init__(self, V: dt.date, feats: list[str], stride: int = 1):
        self.V, self.feats = V, feats
        # stride > 1 — только прореживание для дымового прогона; в боевом прогоне 1
        self.cuts30 = fold_cutoffs(V, TARGET_DAYS)[::stride]
        self.cuts60 = [T for T in self.cuts30 if T + dt.timedelta(days=HAZ_H) <= V]
        assert self.cuts60 == self.cuts30[:len(self.cuts60)], "cuts60 не префикс cuts30"
        nb = len(feats)
        self.nb = nb

        sizes = {}
        for T in self.cuts30:
            hh = user_half(panel_users(T, 1)["user_id"].to_numpy())
            sizes[T] = (int((hh == 0).sum()), int((hh == 1).sum()))
        n_tot = sum(sum(v) for v in sizes.values())
        self.n = n_tot
        self.off = [0, sum(sizes[T][0] for T in self.cuts30)]
        self.cnt_half = [self.off[1], n_tot - self.off[1]]
        self.n60 = [sum(sizes[T][h] for T in self.cuts60) for h in (0, 1)]

        log(f"  матрица: {n_tot:,} строк x {nb + N_AUX} колонок "
            f"({(nb + N_AUX) * n_tot * 4 / 2 ** 30:.1f} ГБ), "
            f"cuts30={len(self.cuts30)} cuts60={len(self.cuts60)}, "
            f"половины {self.cnt_half[0]:,}/{self.cnt_half[1]:,}")

        self.X = np.zeros((n_tot, nb + N_AUX), np.float32)
        self.y = np.empty(n_tot, np.float32)
        self.n30 = np.empty(n_tot, np.int16)
        self.gap = np.full(n_tot, -1, np.int16)
        self.uid = np.empty(n_tot, np.int64)
        self.tix = np.empty(n_tot, np.int16)

        pos = [0, self.off[1]]
        for ci, T in enumerate(self.cuts30):
            Xb, yb = make_xy(T, None, 1, norm_long=True)
            uid = Xb["user_id"].to_numpy()
            A = to_np(Xb, feats)
            hh = user_half(uid)
            n30 = buy_days(T, uid, TARGET_DAYS)
            gp = buy_gap(T, uid, HAZ_H) if T in self.cuts60 else None
            for half in (0, 1):
                m = hh == half
                k = int(m.sum())
                s = slice(pos[half], pos[half] + k)
                self.X[s, :nb] = A[m]
                self.y[s] = yb[m]
                self.n30[s] = n30[m]
                if gp is not None:
                    self.gap[s] = gp[m]
                self.uid[s] = uid[m]
                self.tix[s] = ci
                pos[half] += k
            del Xb, A
        assert pos[0] == self.off[1] and pos[1] == n_tot, "сборка разъехалась с разметкой"
        # строки с горизонтом 60 обязаны быть префиксом каждой половины
        for half in (0, 1):
            s = self.sl(half)
            assert (self.gap[s][:self.n60[half]] >= 0).all(), "cuts60 не префикс половины"
            assert (self.gap[s][self.n60[half]:] < 0).all(), "cuts60 не префикс половины"
        gc.collect()

    def sl(self, half: int) -> slice:
        return slice(self.off[half], self.off[half] + self.cnt_half[half])

    def sl60(self, half: int) -> slice:
        return slice(self.off[half], self.off[half] + self.n60[half])


# --- головы ---------------------------------------------------------------------
def _p(**over):
    p = dict(LGB_PARAMS)
    p.update(over)
    return p


def _train(ds, params, rounds):
    import lightgbm as lgb
    return lgb.train(params, ds, num_boost_round=rounds)


def _ds(X, label, params):
    import lightgbm as lgb
    return lgb.Dataset(X, label=label, params=params, free_raw_data=True).construct()


def train_heads(fd: FoldData, Xv: np.ndarray, uid_v: np.ndarray, rounds: dict, seed: int):
    """Кросс-фиттинг по пользователям: модель половины `t` предсказывает половину `1-t`.

    Возвращает (aux матрица на обучающих строках, aux матрица на валидации).
    Ни одна строка не получает предсказание от модели, видевшей её пользователя.
    """
    nb = fd.nb
    # головы обязаны видеть ровно 227 базовых признаков: aux-колонки ещё не заполнены,
    # а константный столбец LightGBM биннит в один бин и никогда не выбирает в сплит
    assert not fd.X[:, nb:].any() and not Xv[:, nb:].any(), "aux-колонки заполнены до обучения голов"
    A_tr = np.zeros((fd.n, N_AUX), np.float32)
    A_va = np.zeros((len(uid_v), N_AUX), np.float32)
    hv = user_half(uid_v)
    m_cnt_log = {}

    for t in (0, 1):
        o = 1 - t                       # половина, которую предсказываем
        s_tr, s_tr60 = fd.sl(t), fd.sl60(t)
        s_pr = fd.sl(o)
        mv = hv == o
        Xp, Xvp = fd.X[s_pr], Xv[mv]
        log(f"  половина {t} -> {o}: обучение {fd.cnt_half[t]:,} строк "
            f"(hazard {fd.n60[t]:,}), предсказание {fd.cnt_half[o]:,} + {int(mv.sum()):,}")

        y_t = fd.y[s_tr]
        z_t = np.log1p(y_t)

        # 1. hazard: multiclass по интервалам дней до следующей покупки
        lab = haz_class(fd.gap[s_tr60])
        d = _ds(fd.X[s_tr60], lab, _p(objective="multiclass", metric="multi_logloss",
                                      num_class=K_HAZ, seed=seed))
        mh = _train(d, _p(objective="multiclass", metric="multi_logloss",
                          num_class=K_HAZ, seed=seed), rounds["haz"])
        del d
        gc.collect()
        ph, ph_v = mh.predict(Xp), mh.predict(Xvp)
        log(f"    haz  готова ({rounds['haz']} раундов)")
        del mh
        gc.collect()

        # 2. счёт: multiclass по числу покупательных дней в 30 днях
        lab_c = cnt_class(fd.n30[s_tr])
        m_cnt = np.zeros(K_CNT)
        last = 0.0
        for k in range(K_CNT):                      # средний z в бакете — только на train
            mk = lab_c == k
            if mk.any():
                last = float(z_t[mk].mean())
            m_cnt[k] = last
        m_cnt_log[t] = m_cnt.tolist()
        d = _ds(fd.X[s_tr], lab_c, _p(objective="multiclass", metric="multi_logloss",
                                      num_class=K_CNT, seed=seed))
        mc = _train(d, _p(objective="multiclass", metric="multi_logloss",
                          num_class=K_CNT, seed=seed), rounds["cnt"])
        del d
        gc.collect()
        pc, pc_v = mc.predict(Xp), mc.predict(Xvp)
        log(f"    cnt  готова ({rounds['cnt']} раундов)")
        del mc
        gc.collect()

        # 3. binary P(y>0) — контроль того, что hazard ничего не теряет на h=30
        d = _ds(fd.X[s_tr], (y_t > 0).astype(np.int8),
                _p(objective="binary", metric="binary_logloss", seed=seed))
        mb = _train(d, _p(objective="binary", metric="binary_logloss", seed=seed), rounds["b30"])
        del d
        gc.collect()
        pb, pb_v = mb.predict(Xp), mb.predict(Xvp)
        log(f"    b30  готова ({rounds['b30']} раундов)")
        del mb
        gc.collect()

        # 4. условная величина z | y > 0
        mpos = y_t > 0
        Xpos = fd.X[s_tr][mpos]
        d = _ds(Xpos, z_t[mpos], _p(seed=seed))
        del Xpos
        gc.collect()
        mv_reg = _train(d, _p(seed=seed), rounds["val"])
        del d
        gc.collect()
        mu, mu_v = mv_reg.predict(Xp), mv_reg.predict(Xvp)
        log(f"    val  готова ({rounds['val']} раундов)")
        del mv_reg
        gc.collect()

        # 5. КОНТРОЛЬ: то же самое предсказание того же таргета, без новой супервизии
        d = _ds(fd.X[s_tr], z_t, _p(seed=seed))
        ms = _train(d, _p(seed=seed), rounds["self"])
        del d
        gc.collect()
        zs, zs_v = ms.predict(Xp), ms.predict(Xvp)
        log(f"    self готова ({rounds['self']} раундов)")
        del ms, Xp, Xvp
        gc.collect()

        A_tr[s_pr] = aux_from_heads(ph, pc, pb, mu, zs, m_cnt)
        A_va[mv] = aux_from_heads(ph_v, pc_v, pb_v, mu_v, zs_v, m_cnt)
        del ph, pc, pb, mu, zs, ph_v, pc_v, pb_v, mu_v, zs_v
        gc.collect()

    return A_tr, A_va, m_cnt_log


# --- арки стекинга --------------------------------------------------------------
def run_variant(fd: FoldData, A_tr, Xv, A_va, yv, name: str, rounds: int, seed: int):
    """Одна арка лестницы: обнулить лишние aux-колонки, обучить `direct`, предсказать."""
    nb = fd.nb
    keep = VARIANTS[name]
    fd.X[:, nb:] = 0.0
    Xv[:, nb:] = 0.0
    for c in keep:
        fd.X[:, nb + IX[c]] = A_tr[:, IX[c]]
        Xv[:, nb + IX[c]] = A_va[:, IX[c]]
    t = time.time()
    d = _ds(fd.X, np.log1p(fd.y), _p(seed=seed))
    m = _train(d, _p(seed=seed), rounds)
    del d
    gc.collect()
    z = np.maximum(m.predict(Xv), 0.0)
    imp = sorted(zip(fd.feats + AUX_COLS, m.feature_importance("gain")), key=lambda p: -p[1])
    aux_gain = float(sum(g for f, g in imp if f in AUX_COLS))
    tot_gain = float(sum(g for _, g in imp)) + EPS
    log(f"  {name}: RMSLE={rmsle_z(yv, z):.5f} bias={bias_z(yv, z):+.4f} "
        f"cal={calibrate(yv, z)[1]:.5f}  aux gain {100 * aux_gain / tot_gain:.2f}%  "
        f"[{time.time() - t:.0f}s]")
    top = [(f, float(g)) for f, g in imp[:25]]
    del m
    gc.collect()
    return z, top, aux_gain / tot_gain


# --- прогон одного фолда --------------------------------------------------------
def run_fold(V: dt.date, seed: int, rounds: dict, variants: list[str], stride: int = 1,
             tag_suffix: str = ""):
    load()
    Xv_df, yv = make_xy(V, None, 3, norm_long=True)       # валидация: панель 3-блочная
    feats = feature_names(Xv_df)
    uid_v = Xv_df["user_id"].to_numpy()
    log(f"фолд {V}: валидация {len(yv):,} строк, {len(feats)} базовых признаков")
    Xv = np.zeros((len(yv), len(feats) + N_AUX), np.float32)
    Xv[:, :len(feats)] = to_np(Xv_df, feats)
    seg = Xv_df.select(["rec_buy", "w180_days_buy"]).to_numpy().astype(np.float32)
    del Xv_df
    gc.collect()

    fd = FoldData(V, feats, stride)
    # разметка валидации считается ДО обучения: после неё сырой лог (3.1 ГБ в RAM)
    # больше не нужен, а пик памяти приходится как раз на стекинг
    gap_v = buy_gap(V, uid_v, HAZ_H)
    n30_v = buy_days(V, uid_v, TARGET_DAYS)
    from src import data as _data
    _data._CACHE.pop("df", None)
    gc.collect()

    A_tr, A_va, m_cnt = train_heads(fd, Xv, uid_v, rounds, seed)

    tag = V.strftime("%m%d") + tag_suffix
    out = {}
    for name in variants:
        z, top, share = run_variant(fd, A_tr, Xv, A_va, yv, name, rounds["stack"], seed)
        exp_id = f"MHZ-{name}-V{tag}"
        save_oof(exp_id, uid_v, [V.isoformat()] * len(yv), z, yv)
        out[name] = dict(rmsle=rmsle_z(yv, z), cal=calibrate(yv, z)[1],
                         bias=bias_z(yv, z), aux_share=share, top=top)

    # разметка валидации посчитана выше; buy_h для всех горизонтов читается из gap
    np.savez_compressed(
        ARTIFACTS / f"mhz_val_{V.isoformat()}{tag_suffix}.npz",
        user_id=uid_v, y=yv, gap=gap_v, n30=n30_v, aux=A_va, aux_cols=np.array(AUX_COLS),
        rec_buy=seg[:, 0], w180_days_buy=seg[:, 1],
        z=np.vstack([np.load(ARTIFACTS / f"oof_MHZ-{n}-V{tag}.npz")["z"] for n in variants]),
        z_names=np.array(variants))
    (ARTIFACTS / f"mhz_fold_{V.isoformat()}{tag_suffix}.json").write_text(
        json.dumps(dict(val=V.isoformat(), seed=seed, rounds=rounds, n_val=len(yv),
                        n_train=fd.n, cuts30=len(fd.cuts30), cuts60=len(fd.cuts60),
                        halves=fd.cnt_half, m_cnt=m_cnt,
                        variants={k: {kk: vv for kk, vv in v.items() if kk != "top"}
                                  for k, v in out.items()},
                        top={k: v["top"] for k, v in out.items()}),
                   ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"фолд {V} готов, артефакты записаны")


# --- склейка --------------------------------------------------------------------
def merge(variants: list[str], vals: list[dt.date], ref: str = "S1-E10"):
    from src.merge_oof import auc_positive, diversity, load_parts, aligned_ref
    from src.tracking import log_from_report
    rows = []
    for name in variants:
        parts = [f"MHZ-{name}-V{V.strftime('%m%d')}" for V in vals]
        uid, cut, z, y = load_parts(parts)
        r = evaluate(y, z, cut)
        z_ref = aligned_ref(uid, cut, ref)
        rr = evaluate(y, z_ref, cut)
        d = diversity(z, y, z_ref)
        exp_id = f"MHZ-{name}"
        save_oof(exp_id, uid, cut, z, y)
        save_report(exp_id, r, extra=dict(description=f"exp_024 MHZ арка {name}", parts=parts,
                                          ref=ref, aux_cols=VARIANTS[name], **d))
        print(f"\n=== {exp_id} ===")
        print(format_report(r, rr))
        print(f"  AUC(y>0) = {auc_positive(y, z):.5f}   опорная {auc_positive(y, z_ref):.5f}")
        print(f"  Var(z - z_{ref}) = {d['var_delta']:.5f}  corr остатков {d['corr_resid']:.5f}")
        log_from_report(exp_id, f"exp_024 MHZ арка {name}: aux = {VARIANTS[name] or 'нет'}", r,
                        scenario="S1", n_features=227 + N_AUX, model="direct+aux",
                        params=json.dumps(dict(aux=VARIANTS[name]), ensure_ascii=False))
        rows.append((exp_id, r["wcv"], r["fold_cal"], d))
    print("\nсводка:")
    for e, w, fc, d in rows:
        print(f"  {e:<12} wCV={w:.5f}  " + " ".join(f"{v:.5f}" for v in fc)
              + f"   Var(Δ)={d['var_delta']:.5f}")


# --- мета-модель ----------------------------------------------------------------
MIX = {"S1-E10": 0.15, "S1-E02": 0.30, "S1-E03a": 0.10, "S1-DIST": 0.45}
META_COLS = HAZ_COLS + CNT_COLS + P30_COLS + COMBO_COLS      # selfz в мету не входит


def mix_z(V: dt.date):
    """z боевой смеси `S1-DIST-MIX` и user_id на фолде V, из сохранённых OOF членов."""
    from src.tracking import load_oof
    c, acc, uid0, y0 = V.isoformat(), None, None, None
    for e, w in MIX.items():
        d = load_oof(e)
        m = np.asarray(d["cutoff"], dtype="U10") == c
        u, z, y = np.asarray(d["user_id"])[m], np.asarray(d["z"])[m], np.asarray(d["y"])[m]
        o = np.argsort(u)
        u, z, y = u[o], z[o], y[o]
        if uid0 is None:
            uid0, y0 = u, y
        assert np.array_equal(u, uid0), f"{e}: другой набор строк на {c}"
        acc = w * z if acc is None else acc + w * z
    return acc, uid0, y0


def _meta_frame(V: dt.date):
    d = np.load(ARTIFACTS / f"mhz_val_{V.isoformat()}.npz", allow_pickle=False)
    o = np.argsort(np.asarray(d["user_id"]))
    uid, A = np.asarray(d["user_id"])[o], np.asarray(d["aux"])[o]
    zm, um, y = mix_z(V)
    assert np.array_equal(um, uid), f"{V}: смесь на другом наборе строк"
    X = np.column_stack([zm] + [A[:, IX[c]] for c in META_COLS]).astype(np.float32)
    # цель меты — ОСТАТОК боевой смеси, а не сам таргет. Дерево поверх `z_mix` как
    # признака не умеет воспроизвести тождественное отображение и проигрывает базе
    # даже при полном отсутствии сигнала; на остатке «сигнала нет» даёт ровно базу.
    return X, (np.log1p(y) - zm).astype(np.float32), y, zm, uid


def meta(vals: list[dt.date], rounds: int, leaves: int, seed: int):
    """Компактная модель поверх предсказания боевой смеси и hazard/count/activity.

    Схема temporal-safe: мета обучается ТОЛЬКО на предыдущих фолдах и применяется к
    следующему — ровно так же, как замерялась перекалибровка в
    `research/rmsle_diagnostics` §3. Поэтому фолд 2025-09-04 неоценим (перед ним
    ничего нет), и сводка даётся эквивалентом wCV с весами 2:4:8.

    Дополнительно печатается ОПТИМИСТИЧНАЯ оценка: мета обучена внутри того же
    фолда с кросс-фиттингом по пользователям. Это верхняя граница, а не результат.
    """
    import lightgbm as lgb
    p = _p(seed=seed, num_leaves=leaves, min_data_in_leaf=500, learning_rate=0.05)
    frames = {V: _meta_frame(V) for V in vals if (ARTIFACTS / f"mhz_val_{V.isoformat()}.npz").exists()}
    print(f"мета: {len(META_COLS) + 1} признаков (z смеси + {len(META_COLS)} aux), "
          f"{rounds} раундов, {leaves} листьев")
    rows = []
    order = [V for V in VAL_FOLDS_S1 if V in frames]
    for i, V in enumerate(order):
        X, res, y, zm, _ = frames[V]
        base = calibrate(y, zm)[1]
        prev = order[:i]
        safe = None
        if prev:
            Xtr = np.vstack([frames[P][0] for P in prev])
            rtr = np.concatenate([frames[P][1] for P in prev])
            m = lgb.train(p, lgb.Dataset(Xtr, rtr, params=p), num_boost_round=rounds)
            safe = calibrate(y, np.maximum(zm + m.predict(X), 0.0))[1]
        # оптимистичный контроль: обучение внутри фолда, кросс-фиттинг по пользователям
        uid = frames[V][4]
        h = user_half(uid)
        zo = np.empty(len(y))
        for t in (0, 1):
            tr, pr = h == t, h == (1 - t)
            m = lgb.train(p, lgb.Dataset(X[tr], res[tr], params=p), num_boost_round=rounds)
            zo[pr] = zm[pr] + m.predict(X[pr])
        opt = calibrate(y, np.maximum(zo, 0.0))[1]
        # сколько сигнала в поправке вообще есть: корреляция с истинным остатком и
        # лучший её масштаб. alpha* — один параметр на 190k строк, переобучиться
        # ему негде, поэтому RMSLE при alpha* — верхняя граница выигрыша меты.
        corr = float(np.corrcoef(zo - zm, res)[0, 1])
        grid = np.linspace(0.0, 1.5, 61)
        sc = [calibrate(y, np.maximum(zm + a * (zo - zm), 0.0))[1] for a in grid]
        ia = int(np.argmin(sc))
        rows.append((V, base, safe, opt, corr, float(grid[ia]), float(sc[ia])))
        s = f"{safe:.5f} ({safe - base:+.5f})" if safe is not None else "н/д (нет прошлых фолдов)"
        print(f"  {V}: смесь {base:.5f} | temporal-safe {s} | оптимистично "
              f"{opt:.5f} ({opt - base:+.5f})")
        print(f"      corr(поправка, остаток)={corr:+.4f}  alpha*={grid[ia]:.3f} -> "
              f"{sc[ia]:.5f} ({sc[ia] - base:+.5f})")
    if rows:
        wa = np.array([FOLD_WEIGHTS_S1[VAL_FOLDS_S1.index(r[0])] for r in rows], float)
        wa /= wa.sum()
        print(f"  по всем фолдам: corr(поправка, остаток) = "
              f"{float(np.dot(wa, [r[4] for r in rows])):+.4f}, "
              f"выигрыш при alpha* = {float(np.dot(wa, [r[6] - r[1] for r in rows])):+.5f} "
              f"(верхняя граница: обучение внутри фолда + подобранный масштаб)")
    ok = [r for r in rows if r[2] is not None]
    if ok:
        w = np.array([FOLD_WEIGHTS_S1[VAL_FOLDS_S1.index(r[0])] for r in ok], float)
        w /= w.sum()
        db = float(np.dot(w, [r[2] - r[1] for r in ok]))
        do = float(np.dot(w, [r[3] - r[1] for r in ok]))
        print(f"  эквивалент wCV (веса {':'.join(str(int(FOLD_WEIGHTS_S1[VAL_FOLDS_S1.index(r[0])])) for r in ok)}): "
              f"temporal-safe {db:+.5f}, оптимистично {do:+.5f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fold")
    f.add_argument("--val", required=True)
    f.add_argument("--seed", type=int, default=SEED)
    f.add_argument("--variants", nargs="*", default=list(VARIANTS))
    f.add_argument("--haz-rounds", type=int, default=200)
    f.add_argument("--cnt-rounds", type=int, default=200)
    f.add_argument("--aux-rounds", type=int, default=300)
    f.add_argument("--stack-rounds", type=int, default=600)
    f.add_argument("--stride", type=int, default=1, help="прореживание cutoff'ов; только дым")
    f.add_argument("--suffix", default="", help="суффикс имён артефактов; только дым")

    m = sub.add_parser("merge")
    m.add_argument("--variants", nargs="*", default=list(VARIANTS))
    m.add_argument("--ref", default="S1-E10")

    mt = sub.add_parser("meta")
    mt.add_argument("--rounds", type=int, default=200)
    mt.add_argument("--leaves", type=int, default=31)
    mt.add_argument("--seed", type=int, default=SEED)

    a = ap.parse_args()
    if a.cmd == "fold":
        rounds = dict(haz=a.haz_rounds, cnt=a.cnt_rounds, b30=a.aux_rounds,
                      val=a.aux_rounds, self=a.aux_rounds, stack=a.stack_rounds)
        run_fold(dt.date.fromisoformat(a.val), a.seed, rounds, a.variants, a.stride, a.suffix)
    elif a.cmd == "merge":
        merge(a.variants, VAL_FOLDS_S1, a.ref)
    elif a.cmd == "meta":
        meta(VAL_FOLDS_S1, a.rounds, a.leaves, a.seed)


if __name__ == "__main__":
    main()
