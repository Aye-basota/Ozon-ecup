"""Проверки анти-лукапа для STRATEGY_04 (`src/calval.py`).

Стратегия — единственное место в проекте, где в обучение попадают cutoff'ы
ПОЗЖЕ валидационного. Поэтому канал переноса из будущего закрывается не
доводом, а этими проверками:

  1. легальность `EXTRA` относительно границы данных и относительно теста;
  2. признаки на cutoff'е T не зависят ни от одной строки после T (побитово);
  3. таргет на cutoff'е T зависит ровно от окна `(T, T+30]`;
  4. `CLEAN` по-прежнему подчиняется правилу `T + 30 <= V`;
  5. строки `EXTRA` приходят ТОЛЬКО от группы-донора, а метрика считается на
     другой группе — пересечения нет;
  6. расщепление по пользователям не вырождается в чётность `user_id`;
  7. панель и число строк валидации не изменились относительно проекта.

Запуск: python -m pytest src/test_calval.py -q
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from src import calval, data
from src.config import CORRIDOR_END, DATA_END, TARGET_DAYS, VAL_FOLDS_S1
from src.features import build_features, panel_users, target

EXTRA_SAMPLE = [dt.date(2025, 10, 22), dt.date(2025, 12, 3), dt.date(2026, 1, 14)]
VAL_ROWS = {dt.date(2025, 9, 4): 188_518, dt.date(2025, 9, 18): 191_025,
            dt.date(2025, 10, 2): 193_694, dt.date(2025, 10, 16): 197_379}


# ------------------------------------------------------------------ 1. легальность
def test_extra_cutoffs_are_legal():
    ce = calval.extra_cutoffs()
    assert len(ce) == 13
    assert ce[0] == dt.date(2025, 10, 22) and ce[-1] == dt.date(2026, 1, 14)
    for T in ce:
        # позже коридора — иначе это не «неиспользуемые дни», а дубль CLEAN
        assert T > CORRIDOR_END
        # target-окно целиком внутри данных: относительно ТЕСТА лукапа нет
        assert T + dt.timedelta(days=TARGET_DAYS) <= DATA_END
    assert not set(ce) & set(calval.clean_cutoffs())
    assert not set(calval.early_cutoffs()) & set(calval.clean_cutoffs())
    for T in calval.early_cutoffs():
        assert T <= CORRIDOR_END


def test_early_control_is_inside_corridor_and_earliest_first():
    ea = calval.early_cutoffs()
    assert ea == sorted(ea)
    assert ea[0] == dt.date(2025, 4, 1)
    for T in ea:
        assert T + dt.timedelta(days=TARGET_DAYS) <= min(VAL_FOLDS_S1) or T > min(VAL_FOLDS_S1)


# ---------------------------------------------------- 2. признаки не видят будущего
def _with_truncated_raw(cut: dt.date):
    """Подменяет кэш сырья усечённым по дате; возвращает функцию восстановления."""
    full = data.load()
    data._CACHE["df"] = full.filter(pl.col("event_date") <= cut)
    return lambda: data._CACHE.__setitem__("df", full)


def test_features_do_not_depend_on_future_rows():
    """Признаки, построенные на полных данных и на усечённых по T, совпадают побитово."""
    for T in EXTRA_SAMPLE:
        f_full = build_features(T, None, norm_long=True)
        restore = _with_truncated_raw(T)
        try:
            f_cut = build_features(T, None, norm_long=True)
        finally:
            restore()
        assert f_full.columns == f_cut.columns
        assert f_full.height == f_cut.height
        for c in f_full.columns:
            a, b = f_full[c].to_numpy(), f_cut[c].to_numpy()
            if a.dtype.kind == "f":
                assert np.allclose(a, b, equal_nan=True), f"{T}: {c} зависит от будущего"
            else:
                assert np.array_equal(a, b), f"{T}: {c} зависит от будущего"


def test_train_panel_does_not_depend_on_future_rows():
    T = EXTRA_SAMPLE[1]
    u_full = panel_users(T, calval.TRAIN_BLOCKS)["user_id"].to_numpy()
    restore = _with_truncated_raw(T)
    try:
        # кэш панели пишется на диск, поэтому сверяем правило напрямую
        df = data.load()
        a = T - dt.timedelta(days=29)
        u_cut = (df.filter((pl.col("event_date") >= a) & (pl.col("event_date") <= T))
                 .select("user_id").unique().sort("user_id")["user_id"].to_numpy())
    finally:
        restore()
    assert np.array_equal(u_full, u_cut)


# ------------------------------------------------------- 3. таргет = ровно окно 30д
def test_target_uses_exactly_the_30_day_window():
    T = EXTRA_SAMPLE[1]
    u = panel_users(T, calval.TRAIN_BLOCKS)
    y_full = target(T, u, TARGET_DAYS)["y"].to_numpy()
    restore = _with_truncated_raw(T + dt.timedelta(days=TARGET_DAYS))
    try:
        y_cut = target(T, u, TARGET_DAYS)["y"].to_numpy()
    finally:
        restore()
    assert np.array_equal(y_full, y_cut), "таргет зависит от данных после T+30"
    restore = _with_truncated_raw(T)
    try:
        y_zero = target(T, u, TARGET_DAYS)["y"].to_numpy()
    finally:
        restore()
    assert (y_zero == 0).all(), "таргет не пуст на данных, обрезанных по T"


def test_latest_extra_target_window_ends_at_data_boundary():
    T = calval.extra_cutoffs()[-1]
    assert T + dt.timedelta(days=TARGET_DAYS) == DATA_END
    df = data.load()
    assert int((df["event_date"] > DATA_END).sum()) == 0


# --------------------------------------------------------- 4. CLEAN не изменился
def test_clean_sample_still_obeys_the_project_rule():
    from src.validation import get_folds
    for tr, V in get_folds(90, 7, VAL_FOLDS_S1):
        assert calval.clean_cutoffs(V) == tr
        for T in calval.clean_cutoffs(V):
            assert T + dt.timedelta(days=TARGET_DAYS) <= V
            assert T <= CORRIDOR_END


# ------------------------------------------------- 5. группы не пересекаются
def test_extra_rows_never_come_from_the_scored_group():
    V = VAL_FOLDS_S1[-1]
    specs = calval.intensive_specs(calval.clean_cutoffs(V), calval.extra_cutoffs(), True)
    scored = set(np.asarray(calval.panel_users(V, calval.VAL_BLOCKS)["user_id"])
                 [~calval.user_group(calval.panel_users(V, calval.VAL_BLOCKS)["user_id"]
                                     .to_numpy())].tolist())
    extra = set(calval.extra_cutoffs())
    n_checked = 0
    for T, _, m in specs:
        if T not in extra:
            continue
        uid, _ = calval.panel_target(T)
        donors = set(uid[m].tolist())
        assert not donors & scored, f"{T}: строки от пользователей, на которых считается метрика"
        n_checked += 1
    assert n_checked == 13


def test_crossfit_halves_are_disjoint_and_cover_everything():
    V = VAL_FOLDS_S1[-1]
    uid = panel_users(V, calval.VAL_BLOCKS)["user_id"].to_numpy()
    g = calval.user_group(uid)
    assert g.sum() + (~g).sum() == len(uid)
    assert 0.45 < g.mean() < 0.55
    T = calval.extra_cutoffs()[0]
    m_b = calval.row_mask(T, True, True)
    m_a = calval.row_mask(T, True, False)
    assert not (m_b & m_a).any()
    assert np.array_equal(m_b | m_a, calval.row_mask(T, True, None))


# ------------------------------------------------- 6. расщепление не вырождено
def test_user_split_is_not_parity_of_user_id():
    uid = data.load()["user_id"].unique().sort().to_numpy()
    g = calval.user_group(uid)
    assert 0.49 < g.mean() < 0.51
    parity = (uid % 2 == 1)
    agree = float((g == parity).mean())
    assert 0.45 < agree < 0.55, f"расщепление совпало с чётностью user_id на {agree:.3f}"
    assert np.array_equal(g, calval.user_group(uid)), "расщепление недетерминировано"


# ------------------------------------------------- 7. схема валидации не тронута
def test_validation_panel_is_unchanged():
    for V, n in VAL_ROWS.items():
        assert panel_users(V, calval.VAL_BLOCKS).height == n
    assert calval.VAL_BLOCKS == 3
    assert list(VAL_FOLDS_S1) == [dt.date(2025, 9, 4), dt.date(2025, 9, 18),
                                  dt.date(2025, 10, 2), dt.date(2025, 10, 16)]


def test_centering_uses_only_training_rows():
    """c(T) считается по тем же строкам, что попали в обучение, и ни по каким другим."""
    T = calval.extra_cutoffs()[0]
    uid, y = calval.panel_target(T)
    m = calval.row_mask(T, True, True)
    feats = ["w30_gmv", "rec_buy"]
    _, v, levels = calval.assemble([(T, calval.TRAIN_BLOCKS, m)], feats, "centered")
    assert abs(levels[T] - float(np.log1p(y[m]).mean())) < 1e-9
    assert abs(float(v.mean())) < 1e-9
    assert not np.isclose(levels[T], float(np.log1p(y[y > 0]).mean()))
