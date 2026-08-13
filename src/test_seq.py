"""Анти-лукап и корректность представления SEQ-01.

Стоп-условие, а не пожелание (`STRATEGY_10` §Тесты причинности). Проверяется
три класса ошибок, каждая из которых даёт «слишком хороший» результат:

  1. в окно признаков попал день после cutoff'а;
  2. таргет посчитан не по тому окну, что у табличного пайплайна;
  3. свёртка видит будущее внутри последовательности (причинность).

Плюс тождества, без которых сравнение с существующими OOF невозможно:
набор строк валидации обязан совпадать с `S1-E10`, а сетка cutoff'ов — с
`validation.get_folds`.

Запуск: python -m pytest src/test_seq.py -q
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from src.config import ARTIFACTS, TARGET_DAYS, VAL_FOLDS_S1
from src.data import load
from src.features import panel_users, target
from src.seq import (CHANNELS, DEFAULT_CFG, LOG_COLS, N_CH, N_CH_STORED, N_DAYS, SCALE_END,
                     SEQ_L, build_index, day_index, extras_for, fold_cutoffs, gather,
                     panel, receptive_field, target_at, user_rows)
from src.validation import get_folds

V_LAST = VAL_FOLDS_S1[-1]
T_MID = dt.date(2025, 7, 3)
T_EARLY = dt.date(2025, 4, 3)


@pytest.fixture(scope="module")
def raw():
    return load()


# --------------------------------------------------------------------- индексация
def test_uid_index_roundtrip():
    _, _, uid, _ = panel()
    sample = uid[::5000]
    assert np.array_equal(uid[user_rows(sample)], sample)


def test_panel_shape_and_channels():
    p, g, uid, sc = panel()
    assert p.shape == (len(uid), N_DAYS, N_CH_STORED)
    assert g.shape == (len(uid), N_DAYS)
    assert len(sc) == N_CH_STORED
    assert N_CH == N_CH_STORED + 3
    assert np.isfinite(sc).all() and (sc > 0).all()


def test_panel_matches_raw_rows(raw):
    """present/gmv/searches в панели совпадают с сырым логом на выборке строк."""
    _, _, uid, _ = panel()
    s = raw.sample(n=20_000, seed=0)
    r = user_rows(s["user_id"].to_numpy())
    d = (s["event_date"].to_numpy() - np.datetime64("2025-01-01")
         ).astype("timedelta64[D]").astype(int)
    p, g, _, _ = panel()
    assert (p[r, d, CHANNELS.index("present")] == 1).all()
    assert np.allclose(g[r, d], s["gmv"].to_numpy(), rtol=1e-6, atol=1e-6)
    got = p[r, d, CHANNELS.index("searches")].astype(np.float32)
    assert np.allclose(got, np.log1p(s["searches"].to_numpy()).astype(np.float16), atol=1e-3)


def test_absent_day_is_all_zero(raw):
    """День без строки — нули во всех каналах, включая present."""
    p, g, uid, _ = panel()
    present = p[:, :, CHANNELS.index("present")]
    n_present = int((present > 0).sum())
    assert n_present == raw.height, f"present-ячеек {n_present}, строк лога {raw.height}"
    absent = present == 0
    # достаточно проверить, что суммарная масса всех каналов на пустых днях = 0
    assert float(p[:2000][absent[:2000]].astype(np.float32).sum()) == 0.0
    assert float(g[:2000][absent[:2000]].sum()) == 0.0


# --------------------------------------------------------------------- анти-лукап
@pytest.mark.parametrize("T", [T_EARLY, T_MID, V_LAST])
def test_window_has_no_day_after_cutoff(T, raw):
    """Ни одного активного дня позже cutoff'а в собранной последовательности.

    Сверка прямая: число present-дней в окне = число строк лога пользователя
    в полуинтервале (T-365, T].
    """
    u = panel_users(T, 3)["user_id"].to_numpy()[::4000]
    r = user_rows(u)
    x = gather(T, r).astype(np.float32)
    lo = T - dt.timedelta(days=SEQ_L - 1)
    cnt = (raw.lazy()
           .filter(pl.col("user_id").is_in(u.tolist()))
           .filter((pl.col("event_date") >= lo) & (pl.col("event_date") <= T))
           .group_by("user_id").agg(pl.len().alias("n")).collect())
    m = dict(zip(cnt["user_id"].to_list(), cnt["n"].to_list()))
    got = x[:, :, CHANNELS.index("present")].sum(axis=1)
    exp = np.array([m.get(int(v), 0) for v in u], float)
    assert np.array_equal(got, exp), "число дней в окне разошлось с сырым логом"


def test_window_gmv_mass_matches_window_only(raw):
    """Сумма log1p(gmv) в окне = сумма по строкам лога строго внутри окна."""
    T = V_LAST
    u = panel_users(T, 3)["user_id"].to_numpy()[::4000]
    r = user_rows(u)
    x = gather(T, r).astype(np.float32)
    lo = T - dt.timedelta(days=SEQ_L - 1)
    agg = (raw.lazy().filter(pl.col("user_id").is_in(u.tolist()))
           .filter((pl.col("event_date") >= lo) & (pl.col("event_date") <= T))
           .group_by("user_id").agg(pl.col("gmv").log1p().sum().alias("s")).collect())
    m = dict(zip(agg["user_id"].to_list(), agg["s"].to_list()))
    got = x[:, :, CHANNELS.index("gmv")].sum(axis=1)
    exp = np.array([m.get(int(v), 0.0) for v in u], float)
    assert np.allclose(got, exp, atol=2e-2)


def test_last_position_is_the_cutoff_day(raw):
    """Позиция SEQ_L-1 — это ровно день T, а не T+1 и не T-1."""
    T = V_LAST
    u = panel_users(T, 3)["user_id"].to_numpy()[::4000]
    r = user_rows(u)
    x = gather(T, r).astype(np.float32)
    on_T = (raw.lazy().filter(pl.col("user_id").is_in(u.tolist()))
            .filter(pl.col("event_date") == T).select("user_id").collect()["user_id"].to_list())
    exp = np.isin(u, np.array(on_T, dtype=u.dtype)).astype(float)
    assert np.array_equal(x[:, -1, CHANNELS.index("present")], exp)


@pytest.mark.parametrize("T", [T_EARLY, T_MID, V_LAST, dt.date(2026, 2, 13)])
def test_avail_marks_days_before_data_start(T):
    """`avail` = 0 ровно там, где день окна раньше 2025-01-01.

    Ни один cutoff коридора не даёт полного окна: на 2025-10-16 доступно 289 дней
    из 365, на самом раннем — 93. Полные 365 есть только на тестовом cutoff'е
    2026-02-13. Это та же асимметрия глубины истории, которую в табличном
    пайплайне закрывает `features.normalize_long`, и здесь её несёт канал `avail`.
    """
    e = extras_for(T).astype(np.float32)
    n_avail = min(day_index(T) + 1, SEQ_L)
    assert e[:SEQ_L - n_avail, 0].sum() == 0.0
    assert e[SEQ_L - n_avail:, 0].min() == 1.0
    assert e[:, 0].sum() == n_avail
    if T <= V_LAST:
        # каналы данных на недоступных днях — нули
        r = user_rows(panel_users(T, 1)["user_id"].to_numpy()[:512])
        x = gather(T, r).astype(np.float32)
        assert float(np.abs(x[:, :SEQ_L - n_avail, :N_CH_STORED]).sum()) == 0.0


def test_no_corridor_cutoff_has_a_full_window():
    """Диагностический инвариант: в коридоре окно 365 дней всегда неполное."""
    assert day_index(V_LAST) + 1 == 289 < SEQ_L
    assert day_index(dt.date(2026, 2, 13)) + 1 == 409 >= SEQ_L


def test_scale_window_is_in_the_past_of_every_fold():
    """RMS каналов посчитан по дням <= последнего обучающего cutoff'а раннего фолда."""
    assert SCALE_END == fold_cutoffs(VAL_FOLDS_S1[0])[-1]
    for V in VAL_FOLDS_S1:
        assert SCALE_END + dt.timedelta(days=TARGET_DAYS) <= V


# --------------------------------------------------------------------- таргет
@pytest.mark.parametrize("T", [T_MID, V_LAST])
def test_target_matches_features_target(T):
    """Таргет из плотной панели побитово равен таргету табличного пайплайна."""
    u = panel_users(T, 3)
    ref = target(T, u)["y"].to_numpy()
    got = target_at(T, user_rows(u["user_id"].to_numpy()))
    assert np.allclose(got, ref, rtol=1e-9, atol=1e-6)
    assert np.array_equal(got > 0, ref > 0)


def test_target_window_excludes_cutoff_day(raw):
    """Окно таргета — (T, T+30], день T в него НЕ входит."""
    T = T_MID
    u = panel_users(T, 3)["user_id"].to_numpy()[:5000]
    r = user_rows(u)
    _, g, _, _ = panel()
    d = day_index(T)
    direct = g[r, d + 1:d + 1 + TARGET_DAYS].sum(axis=1)
    assert np.allclose(target_at(T, r), direct)
    with_T = g[r, d:d + 1 + TARGET_DAYS].sum(axis=1)
    assert not np.allclose(with_T, direct), "день T обязан быть исключён из таргета"


# --------------------------------------------------------------------- схема фолдов
def test_fold_cutoffs_match_project_validation():
    ref = {V: tr for tr, V in get_folds(90, 7)}
    for V in VAL_FOLDS_S1:
        assert fold_cutoffs(V) == ref[V]
        assert max(fold_cutoffs(V)) + dt.timedelta(days=TARGET_DAYS) <= V


def test_train_panel_is_one_block_val_is_three():
    V = VAL_FOLDS_S1[0]
    cuts = fold_cutoffs(V)[-2:]
    ci, ri, zy = build_index(cuts, blocks=1)
    exp = sum(panel_users(T, 1).height for T in cuts)
    assert len(ri) == exp and len(zy) == exp
    assert panel_users(V, 3).height < panel_users(V, 1).height


def test_val_rows_match_existing_oof():
    """Набор строк валидации совпадает с S1-E10 — иначе blend.aligned не сойдётся."""
    f = ARTIFACTS / "oof_S1-E10.npz"
    if not f.exists():
        pytest.skip("нет artifacts/oof_S1-E10.npz")
    d = np.load(f, allow_pickle=False)
    cut = np.asarray(d["cutoff"], dtype="U10")
    for V in VAL_FOLDS_S1:
        ref = np.sort(d["user_id"][cut == V.isoformat()])
        got = np.sort(panel_users(V, 3)["user_id"].to_numpy())
        assert np.array_equal(ref, got), f"{V}: панель валидации разошлась с S1-E10"


# --------------------------------------------------------------------- причинность
def test_encoder_is_causal():
    """h[:, t] не меняется при изменении входа в позициях > t. Допуск ровно 0."""
    import torch
    from src.seq import build_model
    torch.manual_seed(0)
    m = build_model(dict(DEFAULT_CFG, hidden=16, blocks=4, z0=0.0)).eval()
    x = torch.randn(2, N_CH, SEQ_L)
    with torch.no_grad():
        h0 = m.encode(x)
        x2 = x.clone()
        t = 200
        x2[:, :, t + 1:] = torch.randn_like(x2[:, :, t + 1:])
        h1 = m.encode(x2)
    assert torch.equal(h0[:, :t + 1], h1[:, :t + 1]), "энкодер видит будущее"
    assert not torch.equal(h0[:, t + 1:], h1[:, t + 1:])


def test_receptive_field_covers_the_window():
    rf = receptive_field(DEFAULT_CFG["blocks"], DEFAULT_CFG["kernel"])
    assert rf >= SEQ_L, f"рецептивное поле {rf} дней < окна {SEQ_L}"


def test_prediction_does_not_depend_on_future_of_target_window():
    """Изменение данных в (T, T+30] не меняет вход модели: окно кончается на T."""
    T = T_MID
    r = user_rows(panel_users(T, 3)["user_id"].to_numpy()[:256])
    x = gather(T, r).copy()
    p, _, _, _ = panel()
    d = day_index(T)
    saved = p[r, d + 1:d + 1 + TARGET_DAYS, :].copy()
    p[r, d + 1:d + 1 + TARGET_DAYS, :] = 7.0
    try:
        assert np.array_equal(gather(T, r), x)
    finally:
        p[r, d + 1:d + 1 + TARGET_DAYS, :] = saved
