"""Анти-лукап и корректность разметки MHZ (exp_024).

Проверяется ровно то, на чём стоит вывод эксперимента:
  * метки читают ТОЛЬКО своё окно (T, T + h] — доказывается усечением лога;
  * метки согласованы с боевым `features.target` (то же условие gmv > 0);
  * границы классов hazard и счёта совпадают с объявленными;
  * обучающие cutoff'ы головы с горизонтом 60 легальны и по фолду, и по
    «отравленному» окну панели;
  * кросс-фиттинг по пользователям детерминирован и глобален.

Запуск: python -m pytest src/test_mhz.py -q
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from src import data
from src.config import CUTOFF_STEP, VAL_FOLDS_S1, cutoff_grid
from src.features import panel_users, target
from src.mhz import (AUX_COLS, CNT_MID, HAZ_H, HAZ_MID, N_AUX, POISON_START, VARIANTS,
                     aux_from_heads, buy_days, buy_gap, cnt_class, fold_cutoffs, haz_class,
                     user_half)

T_PROBE = dt.date(2025, 7, 3)          # чистый cutoff внутри коридора


@pytest.fixture(scope="module")
def probe():
    u = panel_users(T_PROBE, 1)["user_id"].to_numpy()
    return u, target(T_PROBE, pl.DataFrame({"user_id": u}))["y"].to_numpy()


# --- границы классов ------------------------------------------------------------
def test_haz_class_boundaries():
    gap = np.array([1, 7, 8, 14, 15, 21, 22, 30, 31, 45, 46, 60, 61])
    assert haz_class(gap).tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6]


def test_cnt_class_boundaries():
    n = np.array([0, 1, 2, 3, 4, 5, 7, 8, 30])
    assert cnt_class(n).tolist() == [0, 1, 2, 3, 4, 5, 5, 6, 6]
    assert len(CNT_MID) == 7 and len(HAZ_MID) == 7


# --- согласованность разметки с боевым таргетом ---------------------------------
def test_gap_matches_target(probe):
    """gap <= 30 равносильно y > 0: то же окно, то же условие gmv > 0."""
    u, y = probe
    gap = buy_gap(T_PROBE, u, HAZ_H)
    assert np.array_equal(gap <= 30, y > 0)


def test_count_matches_target(probe):
    u, y = probe
    n30 = buy_days(T_PROBE, u, 30)
    assert np.array_equal(n30 > 0, y > 0)
    assert n30.max() <= 30 and n30.min() >= 0


def test_gap_monotone_horizons(probe):
    """Кумулятивы вложены: {gap<=7} ⊂ {gap<=14} ⊂ ... ⊂ {gap<=60}."""
    u, _ = probe
    gap = buy_gap(T_PROBE, u, HAZ_H)
    cum = [(gap <= h).mean() for h in (7, 14, 21, 30, 45, 60)]
    assert all(a <= b + 1e-12 for a, b in zip(cum, cum[1:]))
    assert 0.0 < cum[0] < cum[-1] < 1.0


# --- анти-лукап: метка не видит ничего за пределами своего окна ------------------
@pytest.mark.parametrize("h", [30, 60])
def test_labels_read_only_their_window(probe, h):
    """Лог, усечённый по T + h, даёт ПОБИТОВО те же метки, что полный лог."""
    u, _ = probe
    full_gap = buy_gap(T_PROBE, u, h)
    full_n = buy_days(T_PROBE, u, h)
    df = data.load()
    saved = data._CACHE["df"]
    try:
        data._CACHE["df"] = df.filter(pl.col("event_date") <= T_PROBE + dt.timedelta(days=h))
        assert np.array_equal(buy_gap(T_PROBE, u, h), full_gap)
        assert np.array_equal(buy_days(T_PROBE, u, h), full_n)
    finally:
        data._CACHE["df"] = saved


def test_label_ignores_cutoff_day(probe):
    """Окно полуоткрыто слева: покупка В ДЕНЬ T меткой не считается."""
    u, _ = probe
    full_gap, full_n = buy_gap(T_PROBE, u, HAZ_H), buy_days(T_PROBE, u, 30)
    df = data.load()
    saved = data._CACHE["df"]
    try:
        data._CACHE["df"] = df.filter(pl.col("event_date") != T_PROBE)
        g2, n2 = buy_gap(T_PROBE, u, HAZ_H), buy_days(T_PROBE, u, 30)
    finally:
        data._CACHE["df"] = saved
    assert np.array_equal(g2, full_gap) and np.array_equal(n2, full_n)


# --- легальность обучающих cutoff'ов -------------------------------------------
@pytest.mark.parametrize("V", VAL_FOLDS_S1)
def test_fold_cutoffs_legal(V):
    c30, c60 = fold_cutoffs(V, 30), fold_cutoffs(V, HAZ_H)
    assert c30 and c60
    assert set(c60) <= set(c30), "горизонт 60 обязан быть подмножеством горизонта 30"
    assert max(c60) == c60[-1] and c60 == sorted(c60)
    assert c60 == c30[:len(c60)], "cuts60 обязаны быть ПРЕФИКСОМ cuts30 по дате"
    for T in c60:
        assert T + dt.timedelta(days=HAZ_H) <= V, "метка hazard залезает в валидацию"
        assert T + dt.timedelta(days=HAZ_H) < POISON_START, "метка hazard залезает в панель"
    for T in c30:
        assert T + dt.timedelta(days=30) <= V


def test_fold_cutoff_counts():
    """Зафиксированные объёмы: 60-дневная супервизия стоит 4 cutoff'а на фолд."""
    got = {V: (len(fold_cutoffs(V, 30)), len(fold_cutoffs(V, HAZ_H))) for V in VAL_FOLDS_S1}
    assert got == {VAL_FOLDS_S1[0]: (18, 14), VAL_FOLDS_S1[1]: (20, 16),
                   VAL_FOLDS_S1[2]: (22, 18), VAL_FOLDS_S1[3]: (24, 20)}


def test_grid_unchanged():
    """Сетка та же, что у боевых экспериментов: 29 cutoff'ов 04-03..10-16."""
    g = cutoff_grid(90, CUTOFF_STEP)
    assert len(g) == 29 and g[0] == dt.date(2025, 4, 3) and g[-1] == dt.date(2025, 10, 16)


# --- кросс-фиттинг --------------------------------------------------------------
def test_user_half_deterministic_and_global(probe):
    u, _ = probe
    h1, h2 = user_half(u), user_half(u)
    assert np.array_equal(h1, h2)
    other = panel_users(dt.date(2025, 9, 11), 1)["user_id"].to_numpy()
    common = np.intersect1d(u, other)
    assert len(common) > 1000
    a = dict(zip(u.tolist(), user_half(u).tolist()))
    b = dict(zip(other.tolist(), user_half(other).tolist()))
    assert all(a[k] == b[k] for k in common.tolist()), "половина зависит от cutoff'а"
    share = float(user_half(u).mean())
    assert 0.45 < share < 0.55


# --- сборка aux -----------------------------------------------------------------
def test_aux_columns_consistent():
    assert len(AUX_COLS) == N_AUX == len(set(AUX_COLS))
    union = set().union(*VARIANTS.values())
    assert union == set(AUX_COLS), "какая-то aux-колонка не входит ни в одну арку"
    assert VARIANTS["BASE"] == []
    assert "selfz" not in VARIANTS["FULL"], "контрольная колонка не должна течь в FULL"


def test_aux_from_heads_reconstruction():
    rng = np.random.default_rng(0)
    ph = rng.dirichlet(np.ones(7), size=64)
    pc = rng.dirichlet(np.ones(7), size=64)
    pb, mu, zs = rng.random(64), rng.random(64) * 5, rng.random(64) * 5
    m_cnt = np.array([0.0, 3.2, 4.0, 4.5, 4.9, 5.3, 5.8])
    A = aux_from_heads(ph, pc, pb, mu, zs, m_cnt)
    ix = {c: i for i, c in enumerate(AUX_COLS)}
    cum = np.cumsum(ph, axis=1)
    for j, name in enumerate(["haz_p7", "haz_p14", "haz_p21", "haz_p30", "haz_p45", "haz_p60"]):
        assert np.allclose(A[:, ix[name]], cum[:, j], atol=1e-6)
    assert np.allclose(A[:, ix["haz_edays"]], ph @ HAZ_MID, atol=1e-5)
    assert np.allclose(A[:, ix["cnt_p0"]], pc[:, 0], atol=1e-6)
    assert np.allclose(A[:, ix["cnt_en"]], pc @ CNT_MID, atol=1e-5)
    assert np.allclose(A[:, ix["cnt_mix"]], pc @ m_cnt, atol=1e-5)
    assert np.allclose(A[:, ix["tp30"]], pb * mu, atol=1e-5)
    assert np.allclose(A[:, ix["selfz"]], zs, atol=1e-5)
    # кумулятивы неубывают, условные hazard'ы в [0, 1]
    curve = A[:, [ix[c] for c in ("haz_p7", "haz_p14", "haz_p21", "haz_p30", "haz_p45", "haz_p60")]]
    assert (np.diff(curve, axis=1) >= -1e-6).all()
    hz = A[:, [ix[f"haz_h{j}"] for j in range(2, 7)]]
    assert (hz >= -1e-6).all() and (hz <= 1 + 1e-6).all()


def test_aux_shape_is_not_a_function_of_p30():
    """Форма кривой обязана различать состояния при ОДИНАКОВОМ P(buy30).

    Два пользователя с равным `p30`, но разной кривой (быстрый покупатель против
    «то ли пауза, то ли уход») дают разные `haz_sl730`/`haz_sl3060`. Если бы это
    было не так, вся гипотеза multi-horizon была бы пустой по построению.
    """
    fast = np.array([[0.30, 0.10, 0.05, 0.05, 0.05, 0.05, 0.40]])
    slow = np.array([[0.05, 0.10, 0.15, 0.20, 0.15, 0.10, 0.25]])
    m = np.zeros(7)
    A = aux_from_heads(np.vstack([fast, slow]), np.tile(np.full(7, 1 / 7), (2, 1)),
                       np.array([0.5, 0.5]), np.zeros(2), np.zeros(2), m)
    ix = {c: i for i, c in enumerate(AUX_COLS)}
    assert abs(A[0, ix["haz_p30"]] - A[1, ix["haz_p30"]]) < 1e-6, "p30 подобраны равными"
    assert A[0, ix["haz_sl730"]] < A[1, ix["haz_sl730"]] - 0.5
