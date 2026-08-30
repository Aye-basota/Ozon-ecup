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

from src.config import (ARTIFACTS, CUTOFF_STEP, CUTOFF_TEST, TARGET_DAYS, VAL_FOLDS_S1,
                        cutoff_grid)
from src.data import load
from src.features import panel_users, target
from src.seq import (AUG_MODES, CHANNELS, DEFAULT_CFG, DEPTH_GRID, LOG_COLS, MIN_HISTORY,
                     N_CH, N_CH_STORED, N_DAYS, SCALE_END, SEQ_L, Batcher, build_index,
                     day_index, depth_spec, extras_for, fold_cutoffs, gather,
                     n_zero_positions, panel, receptive_field, sample_avail_shift,
                     sample_depth, target_at, user_rows)
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


# --------------------------------------------------------------------- глубина истории
@pytest.mark.parametrize("D", [180, 220, 254])
def test_depth_clip_truncates_only_from_the_left(D):
    """`--depth-clip D` обязан резать ТОЛЬКО прошлое.

    Последние D позиций окна (то есть дни `(T-D, T]`) должны совпадать с
    неурезанным окном побитово, а всё, что старше, — быть ровно нулём во всех
    хранимых каналах и в `avail`. Иначе диагностика по глубине измеряла бы не
    глубину, а испорченный вход.
    """
    T = V_LAST
    r = user_rows(panel_users(T, 3)["user_id"].to_numpy()[:512])
    full = gather(T, r)
    clip = gather(T, r, depth_clip=D)
    assert np.array_equal(clip[:, SEQ_L - D:, :], full[:, SEQ_L - D:, :])
    head = clip[:, :SEQ_L - D, :N_CH_STORED]
    assert float(np.abs(head).max()) == 0.0
    assert float(np.abs(clip[:, :SEQ_L - D, N_CH_STORED]).max()) == 0.0     # avail


def test_depth_clip_beyond_available_history_is_a_noop():
    """Клип глубже, чем есть данных, ничего не меняет — иначе «полная глубина» не эталон."""
    T = V_LAST
    r = user_rows(panel_users(T, 3)["user_id"].to_numpy()[:512])
    avail = day_index(T) + 1
    assert np.array_equal(gather(T, r, depth_clip=avail), gather(T, r))
    assert np.array_equal(gather(T, r, depth_clip=SEQ_L), gather(T, r))


def test_depth_clip_never_adds_days_after_cutoff(raw):
    """Ни при какой глубине в окне нет ни одной строки лога позже cutoff'а."""
    T = V_LAST
    u = panel_users(T, 3)["user_id"].to_numpy()[:2000]
    r = user_rows(u)
    late = raw.filter((pl.col("event_date") > T) & pl.col("user_id").is_in(u.tolist()))
    assert len(late) > 0, "контроль бессмыслен: после cutoff'а вообще нет строк"
    for D in (180, 289, None):
        x = gather(T, r, depth_clip=D)
        n_present = float(x[:, :, CHANNELS.index("present")].astype(np.float32).sum())
        window = raw.filter((pl.col("event_date") <= T) & pl.col("user_id").is_in(u.tolist())
                            & (pl.col("event_date") > T - dt.timedelta(days=D or SEQ_L)))
        assert n_present == float(len(window)), f"глубина {D}: {n_present} != {len(window)}"


# ------------------------------------------------- кросс-фолдовый стресс по глубине
def test_crossdepth_training_targets_are_strictly_before_the_late_panel():
    """Ранняя модель на поздней панели не может видеть её будущего.

    Обучающие cutoff'ы чекпойнта фолда `V_ck` кончаются на `V_ck − 30`, значит их
    таргет-окна кончаются на `V_ck`, а таргет поздней панели живёт в
    `(V_panel, V_panel + 30]` при `V_panel > V_ck`. Пересечения нет ни на день,
    и разрыв train→val здесь БОЛЬШЕ штатного — оценка консервативна.
    """
    for i, V_ck in enumerate(VAL_FOLDS_S1[:-1]):
        cuts = fold_cutoffs(V_ck)
        assert max(cuts) + dt.timedelta(days=TARGET_DAYS) <= V_ck
        for V in VAL_FOLDS_S1[i + 1:]:
            assert V > V_ck
            assert max(cuts) + dt.timedelta(days=TARGET_DAYS) <= V


def test_crossdepth_reproduces_the_test_extrapolation():
    """Ранняя модель на поздней панели обязана давать ТУ ЖЕ экстраполяцию, что тест.

    На тесте модель обучена до глубины 289 (последний чистый cutoff 2025-10-16) и
    получает окно 365 — это +76 дней за границей обученного. Ни один замер
    `exp_026` дальше +35 не заходил. Пара «модель 09-04 -> панель 10-16» даёт +77,
    то есть тот самый режим. Тест фиксирует конструкцию: если сетка cutoff'ов
    или окно поменяются, диагностика перестанет быть аналогом теста молча.
    """
    d_test_train = day_index(cutoff_grid(MIN_HISTORY, CUTOFF_STEP)[-1]) + 1
    assert d_test_train == 289
    assert SEQ_L - d_test_train == 76

    d_train = day_index(fold_cutoffs(VAL_FOLDS_S1[0])[-1]) + 1
    avail = day_index(VAL_FOLDS_S1[-1]) + 1
    assert (d_train, avail) == (212, 289)
    assert avail - d_train == 77, "пара 09-04 -> 10-16 перестала быть аналогом теста"


@pytest.mark.parametrize("D", [212, 230, 247, 261, 275])
def test_crossdepth_grid_truncates_only_the_past_on_the_late_panel(D):
    """Вся сетка глубин кросс-фолдового стресса режет только прошлое поздней панели."""
    T = V_LAST
    r = user_rows(panel_users(T, 3)["user_id"].to_numpy()[:512])
    full = gather(T, r)
    clip = gather(T, r, depth_clip=D)
    assert np.array_equal(clip[:, SEQ_L - D:, :], full[:, SEQ_L - D:, :])
    assert float(np.abs(clip[:, :SEQ_L - D, :N_CH_STORED]).max()) == 0.0
    assert float(np.abs(clip[:, :SEQ_L - D, N_CH_STORED]).max()) == 0.0     # avail


def test_crossdepth_input_does_not_depend_on_the_model():
    """Вход поздней панели — функция только (T, глубина): в paired-сравнении меняется
    ровно одна величина, а не «модель и вход сразу»."""
    T = V_LAST
    r = user_rows(panel_users(T, 3)["user_id"].to_numpy()[:256])
    for D in (212, 289):
        assert np.array_equal(gather(T, r, depth_clip=D), gather(T, r, depth_clip=D))
    assert not np.array_equal(gather(T, r, depth_clip=212), gather(T, r, depth_clip=289))


# ------------------------------------- аугментация границы `avail` (SEQ-AVAIL-AUG)
# `exp_027` доказал механизм провала на LB: режим `avail ≡ 1` на тесте не встречается
# ни на одном обучающем cutoff'е, и цена этого одного бита +0.0034…+0.0044. Здесь
# проверяется, что аугментация вводит этот режим в обучение, НЕ придумывая данных.
AV_COL = N_CH_STORED                       # индекс канала avail во входном тензоре


def _rows(T, n=384):
    return user_rows(panel_users(T, 3)["user_id"].to_numpy()[:n])


def _avail(x):
    return x[:, :, AV_COL].astype(np.float32)


@pytest.mark.parametrize("T", [T_EARLY, T_MID, V_LAST, CUTOFF_TEST])
def test_n_zero_positions_matches_the_window(T):
    """`n_zero_positions` — то же число, что `off` внутри `gather`, и оно же
    определяет, сколько позиций аугментация вправе трогать."""
    n0 = n_zero_positions(T)
    x = gather(T, _rows(T, 64))
    assert float(_avail(x)[:, :n0].max(initial=0.0)) == 0.0
    assert float(_avail(x)[:, n0:].min()) == 1.0
    assert n0 == SEQ_L - min(day_index(T) + 1, SEQ_L)
    # на тесте нулевых позиций нет вовсе — это и есть непрожитый режим
    assert (n0 == 0) == (T == CUTOFF_TEST)


def test_augmentation_changes_only_the_availability_channel():
    """Разрешено менять ровно один канал. Поведенческие и календарные — побитово те же."""
    T = V_LAST
    r = _rows(T)
    n0 = n_zero_positions(T)
    k = np.random.default_rng(0).integers(0, n0 + 1, len(r))
    x0, x1 = gather(T, r), gather(T, r, avail_shift=k)
    assert np.array_equal(x1[:, :, :N_CH_STORED], x0[:, :, :N_CH_STORED])
    assert np.array_equal(x1[:, :, AV_COL + 1:], x0[:, :, AV_COL + 1:])   # dow_sin/dow_cos
    assert not np.array_equal(x1[:, :, AV_COL], x0[:, :, AV_COL])


def test_augmented_positions_keep_behaviour_strictly_zero():
    """В искусственно «открытых» позициях поведение обязано остаться строго нулевым.

    Это и есть граница честности приёма: `avail` говорит «день существует», но
    ни одного события в него не дописано — иначе augmentation придумывала бы
    данные, которых в логе нет.
    """
    T = V_LAST
    r = _rows(T)
    n0 = n_zero_positions(T)
    x = gather(T, r, avail_shift=np.full(len(r), n0))
    assert float(np.abs(x[:, :n0, :N_CH_STORED]).max()) == 0.0
    assert float(_avail(x).min()) == 1.0                       # ровно режим теста


def test_augmentation_moves_the_boundary_and_keeps_it_a_boundary():
    """После сдвига `avail` остаётся префиксом нулей + суффиксом единиц.

    Иначе получился бы паттерн 1…1 0…0 1…1, которого нет ни в одном реальном
    окне, то есть один непрожитый режим сменился бы другим.
    """
    T = V_LAST
    r = _rows(T, 128)
    n0 = n_zero_positions(T)
    k = np.random.default_rng(1).integers(0, n0 + 1, len(r))
    a = _avail(gather(T, r, avail_shift=k))
    assert (np.diff(a, axis=1) >= 0).all(), "avail перестал быть монотонной границей"
    assert np.array_equal((a == 0).sum(axis=1), n0 - k)


def test_augmentation_is_clipped_to_the_zero_prefix():
    """`k` больше числа нулевых позиций не трогает НИ ОДНОЙ реальной позиции."""
    T = V_LAST
    r = _rows(T, 128)
    n0 = n_zero_positions(T)
    big = gather(T, r, avail_shift=np.full(len(r), n0 + 500))
    assert np.array_equal(big, gather(T, r, avail_shift=np.full(len(r), n0)))
    assert np.array_equal(big[:, :, :N_CH_STORED], gather(T, r)[:, :, :N_CH_STORED])


def test_augmentation_never_adds_a_day_after_the_cutoff(raw):
    """Анти-лукап: при любой аугментации число present-дней в окне равно числу
    строк лога в `(T-365, T]` — ни одного дня из будущего не появляется."""
    T = V_LAST
    u = panel_users(T, 3)["user_id"].to_numpy()[:2000]
    r = user_rows(u)
    n0 = n_zero_positions(T)
    window = raw.filter((pl.col("event_date") <= T) & pl.col("user_id").is_in(u.tolist())
                        & (pl.col("event_date") > T - dt.timedelta(days=SEQ_L)))
    late = raw.filter((pl.col("event_date") > T) & pl.col("user_id").is_in(u.tolist()))
    assert len(late) > 0, "контроль бессмыслен: после cutoff'а вообще нет строк"
    rng = np.random.default_rng(2)
    for k in (np.zeros(len(r), int), rng.integers(0, n0 + 1, len(r)), np.full(len(r), n0)):
        x = gather(T, r, avail_shift=k)
        n_present = float(x[:, :, CHANNELS.index("present")].astype(np.float32).sum())
        assert n_present == float(len(window))


def test_augmentation_is_a_noop_on_the_test_cutoff():
    """На тесте нулевых позиций нет: аугментировать нечего, вход тот же."""
    assert n_zero_positions(CUTOFF_TEST) == 0
    r = _rows(CUTOFF_TEST, 128)
    assert np.array_equal(gather(CUTOFF_TEST, r, avail_shift=np.full(len(r), 76)),
                          gather(CUTOFF_TEST, r))
    rng = np.random.default_rng(3)
    for mode in AUG_MODES:
        assert sample_avail_shift(mode, 0, 8, rng, 0.5, 0.5) is None


# --------------------------------------------------------- сэмплер сдвига границы
def test_sampler_none_mode_returns_nothing():
    assert sample_avail_shift("none", 76, 8, np.random.default_rng(0), 1.0, 1.0) is None


@pytest.mark.parametrize("p", [0.25, 0.5])
def test_sampler_avail_drop_is_all_or_nothing(p):
    """Вариант A: пример либо целиком в режиме теста, либо не тронут."""
    k = sample_avail_shift("avail_drop", 76, 40_000, np.random.default_rng(4), p, 0.0)
    assert set(np.unique(k).tolist()) == {0, 76}
    assert abs((k == 76).mean() - p) < 0.01


def test_sampler_avail_bnd_covers_the_whole_range_including_full():
    """Вариант B: train обязан увидеть широкий диапазон числа нулей, в том числе 0.

    Иначе `avail ≡ 1` останется таким же непрожитым режимом, как сейчас, только
    смещённым: `exp_027` показал, что вредит именно НЕПОКРЫТИЕ, а не величина.
    """
    n0, n = 76, 60_000
    k = sample_avail_shift("avail_bnd", n0, n, np.random.default_rng(5), 0.5, 0.5)
    assert k.min() == 0 and k.max() == n0
    assert abs((k == n0).mean() - 0.25) < 0.01          # p * full
    assert (k == 0).mean() > 0.5                        # 1 - p, плюс атом U{0}
    assert len(np.unique(k)) == n0 + 1, "покрыт не весь диапазон границ"
    mid = k[(k > 0) & (k < n0)]
    assert abs(mid.mean() - n0 / 2) < 3.0, "равномерности внутри диапазона нет"


def test_sampler_no_avail_is_constant():
    """Вариант C: канал константен у КАЖДОГО примера, границы не существует."""
    k = sample_avail_shift("no_avail", 76, 1000, np.random.default_rng(6), 0.0, 0.0)
    assert np.array_equal(k, np.full(1000, 76))


# ----------------------------------------------------- обучающий батч и инференс
def _one_group(T, n=512, chunk=256):
    """Один план-группа Batcher'а для cutoff'а T (без обращения к GPU)."""
    u = panel_users(T, 1)["user_id"].to_numpy()[:n]
    ri = user_rows(u)
    ci = np.zeros(len(ri), np.int16)
    y = np.arange(len(ri), dtype=np.float32)
    idx = np.arange(len(ri))
    return [T], ci, ri, y, [(0, idx[:chunk]), (0, idx[chunk:2 * chunk])]


def test_base_batch_is_bitwise_the_pre_augmentation_path():
    """BASE (`aug=none`) обязан собирать вход РОВНО как до этого эксперимента.

    Без этого «численный контроль» BASE ничего не контролирует: расхождение с
    `exp_026` нельзя было бы отнести ни к аугментации, ни к коду.
    """
    T = V_LAST
    cuts, ci, ri, y, group = _one_group(T)
    b = Batcher(cuts, ci, ri, y, 512, 256, np.random.default_rng(0), 1)
    x, yb = b._make(group, seed=12345)
    ref = np.concatenate([gather(T, ri[idx]) for _, idx in group])
    assert np.array_equal(x, ref)
    assert np.array_equal(yb, y[np.concatenate([idx for _, idx in group])])
    # и сид на сборку не влияет, пока аугментации нет
    assert np.array_equal(b._make(group, seed=999)[0], ref)


def test_augmentation_does_not_shift_the_batch_order_stream():
    """Порядок примеров в эпохе обязан быть один и тот же у BASE и у вариантов.

    Сиды аугментации берутся из ОТДЕЛЬНОГО потока. Иначе они сдвигали бы `rng`,
    из которого строится план эпохи, и (а) BASE перестал бы побитово повторять
    `exp_025`/`exp_026`, (б) разница между вариантами мешалась бы с шумом порядка
    данных, который у этой сети (std wCV 0.00250) больше проверяемого эффекта.
    """
    T = V_LAST
    cuts, ci, ri, y, _ = _one_group(T)
    plans = []
    for aug in (None, dict(mode="avail_bnd", p=0.5, full=0.5)):
        b = Batcher(cuts, ci, ri, y, 512, 256, np.random.default_rng(42), 1, aug=aug,
                    aug_seed=[42, 0xA7A1])
        for _ in b:                                   # эпоха 1 целиком
            pass
        plans.append([(k, idx.copy()) for grp in b._plan() for k, idx in grp])
    assert len(plans[0]) == len(plans[1])
    for (k0, i0), (k1, i1) in zip(*plans):
        assert k0 == k1 and np.array_equal(i0, i1), "план эпохи разошёлся из-за аугментации"


@pytest.mark.parametrize("mode,p,full", [("avail_drop", 0.5, 0.0), ("avail_bnd", 0.5, 0.5),
                                         ("no_avail", 0.0, 0.0)])
def test_augmented_batch_changes_only_avail_and_never_the_target(mode, p, full):
    """Таргет не зависит от аугментации, поведенческие каналы — тоже."""
    T = V_LAST
    cuts, ci, ri, y, group = _one_group(T)
    base = Batcher(cuts, ci, ri, y, 512, 256, np.random.default_rng(0), 1)
    aug = Batcher(cuts, ci, ri, y, 512, 256, np.random.default_rng(0), 1,
                  aug=dict(mode=mode, p=p, full=full))
    x0, y0 = base._make(group, seed=7)
    x1, y1 = aug._make(group, seed=7)
    assert np.array_equal(y0, y1), "таргет зависит от аугментации"
    assert np.array_equal(x1[:, :, :N_CH_STORED], x0[:, :, :N_CH_STORED])
    assert np.array_equal(x1[:, :, AV_COL + 1:], x0[:, :, AV_COL + 1:])
    assert not np.array_equal(x1[:, :, AV_COL], x0[:, :, AV_COL])
    n0 = n_zero_positions(T)
    assert float(np.abs(x1[:, :n0, :N_CH_STORED]).max()) == 0.0
    assert (np.diff(_avail(x1), axis=1) >= 0).all()


def test_augmented_batch_is_reproducible_per_seed():
    """Тот же сид -> тот же батч: обучение остаётся воспроизводимым, хотя сборка
    идёт в нескольких потоках."""
    T = V_LAST
    cuts, ci, ri, y, group = _one_group(T)
    b = Batcher(cuts, ci, ri, y, 512, 256, np.random.default_rng(0), 1,
                aug=dict(mode="avail_bnd", p=0.5, full=0.5))
    assert np.array_equal(b._make(group, seed=11)[0], b._make(group, seed=11)[0])
    assert not np.array_equal(b._make(group, seed=11)[0], b._make(group, seed=12)[0])


# ------------------------------------------- depth curriculum (EXP-030, SEQ-D3A)
# `exp_029` отверг аугментацию ОДНОГО канала `avail`: OOD-режим ушёл, но польза
# длинной истории сжалась, потому что поведенческие каналы всегда оставались на
# полной глубине. Здесь обрезается сама история, и главный инвариант приёма —
# обучающий вход обязан быть ТЕМ ЖЕ ОБЪЕКТОМ, что боевой `--depth-clip`.
def test_depth_sample_is_bitwise_the_production_depth_clip():
    """`depth_sample = D` для всех строк == `depth_clip = D`.

    Это и есть смысл эксперимента: сеть учится на тех же входах, которые ей
    выдаст `--depth-clip` на тесте. Если бы приёмы расходились хоть в одном
    канале, curriculum готовил бы модель к режиму, которого на тесте нет.
    """
    T = V_LAST
    r = _rows(T, 512)
    for D in (90, 150, 254, 289):
        assert np.array_equal(gather(T, r, depth_sample=np.full(len(r), D)),
                              gather(T, r, depth_clip=D)), f"глубина {D} разошлась с боевой"


@pytest.mark.parametrize("D", [90, 120, 180, 254])
def test_depth_sample_truncates_only_from_the_left(D):
    """Обрезаются только старые дни: хвост окна побитово тот же, голова — нули."""
    T = V_LAST
    r = _rows(T, 384)
    full = gather(T, r)
    cut = gather(T, r, depth_sample=np.full(len(r), D))
    assert np.array_equal(cut[:, SEQ_L - D:, :], full[:, SEQ_L - D:, :])
    assert float(np.abs(cut[:, :SEQ_L - D, :N_CH_STORED]).max()) == 0.0
    assert float(np.abs(cut[:, :SEQ_L - D, AV_COL]).max()) == 0.0            # avail
    # календарь — функция только даты, обрезка его не трогает (как у `depth_clip`)
    assert np.array_equal(cut[:, :, AV_COL + 1:], full[:, :, AV_COL + 1:])


def test_depth_sample_is_per_example():
    """Каждая строка обрезана СВОЕЙ глубиной, а не общей на батч."""
    T = V_LAST
    r = _rows(T, 256)
    ds = np.array([90, 150, 254, SEQ_L] * (len(r) // 4), np.int64)
    x = gather(T, r, depth_sample=ds)
    for D in (90, 150, 254, SEQ_L):
        m = ds == D
        assert np.array_equal(x[m], gather(T, r[m], depth_clip=None if D == SEQ_L else D))


def test_depth_sample_beyond_available_history_is_a_noop():
    """Глубина больше доступной ничего не меняет: дней из воздуха не появляется."""
    T = V_LAST
    r = _rows(T, 256)
    avail = day_index(T) + 1
    for D in (avail, avail + 40, SEQ_L):
        assert np.array_equal(gather(T, r, depth_sample=np.full(len(r), D)), gather(T, r))


def test_depth_sample_keeps_avail_a_monotone_boundary_and_never_reaches_all_ones():
    """`avail` остаётся префиксом нулей + суффиксом единиц, и нулей становится БОЛЬШЕ.

    Это ключевое отличие от `exp_029`: тот приём вводил в обучение непрожитый
    режим `avail ≡ 1`, а curriculum только добавляет нули, поэтому боевая
    политика теста остаётся `--depth-clip 289`, и ничего нового про `avail ≡ 1`
    сеть не узнаёт.
    """
    T = V_LAST
    r = _rows(T, 128)
    n0 = n_zero_positions(T)
    ds = sample_depth(1.0, DEPTH_GRID, len(r), np.random.default_rng(0))
    a = _avail(gather(T, r, depth_sample=ds))
    assert (np.diff(a, axis=1) >= 0).all(), "avail перестал быть монотонной границей"
    assert (a.min(axis=1) == 0).all(), "у примера пропал префикс avail=0"
    assert ((a == 0).sum(axis=1) >= n0).all(), "обрезка обязана только ДОБАВЛЯТЬ нули"


def test_depth_sample_never_adds_a_day_after_the_cutoff(raw):
    """Анти-лукап: при любой глубине число present-дней = число строк лога в `(T-D, T]`."""
    T = V_LAST
    u = panel_users(T, 3)["user_id"].to_numpy()[:2000]
    r = user_rows(u)
    late = raw.filter((pl.col("event_date") > T) & pl.col("user_id").is_in(u.tolist()))
    assert len(late) > 0, "контроль бессмыслен: после cutoff'а вообще нет строк"
    for D in (90, 180, 289, SEQ_L):
        x = gather(T, r, depth_sample=np.full(len(r), D))
        n_present = float(x[:, :, CHANNELS.index("present")].astype(np.float32).sum())
        window = raw.filter((pl.col("event_date") <= T) & pl.col("user_id").is_in(u.tolist())
                            & (pl.col("event_date") > T - dt.timedelta(days=D)))
        assert n_present == float(len(window)), f"глубина {D}: {n_present} != {len(window)}"


def test_depth_curriculum_and_avail_augmentation_are_exclusive():
    """Два приёма двигают `avail` в разные стороны; вместе они не определены."""
    T = V_LAST
    r = _rows(T, 64)
    with pytest.raises(AssertionError):
        gather(T, r, depth_sample=np.full(len(r), 150), avail_shift=np.full(len(r), 10))
    cuts, ci, ri, y, _ = _one_group(T, n=128, chunk=64)
    with pytest.raises(AssertionError):
        Batcher(cuts, ci, ri, y, 128, 64, np.random.default_rng(0), 1,
                aug=dict(mode="avail_bnd", p=0.5, full=0.5), depth=dict(p=0.5, grid=DEPTH_GRID))


# --------------------------------------------------------- сэмплер глубины
def test_depth_sampler_is_off_by_default():
    assert sample_depth(0.0, DEPTH_GRID, 8, np.random.default_rng(0)) is None
    assert sample_depth(0.5, DEPTH_GRID, 8, None) is None
    assert depth_spec(DEFAULT_CFG)["p"] == 0.0, "curriculum обязан быть выключен по умолчанию"


@pytest.mark.parametrize("p", [0.25, 0.5])
def test_depth_sampler_hits_the_grid_with_probability_p(p):
    """Доля обрезанных примеров = `p`, а глубины берутся ровно из сетки, равномерно."""
    n = 120_000
    d = sample_depth(p, DEPTH_GRID, n, np.random.default_rng(7))
    assert set(np.unique(d).tolist()) <= set(DEPTH_GRID) | {SEQ_L}
    clipped = d != SEQ_L
    # 289 из сетки неотличима от «не резать» на панели глубиной <= 289, поэтому
    # доля считается по числу выпавших значений сетки, а не по факту обрезки
    assert abs(clipped.mean() - p) < 0.01
    for D in DEPTH_GRID:
        assert abs((d == D).mean() - p / len(DEPTH_GRID)) < 0.005
    assert len(np.unique(d)) == len(DEPTH_GRID) + 1


def test_depth_grid_covers_the_real_train_depths():
    """Сетка обязана лежать внутри реально встречающихся глубин обучения.

    Иначе curriculum учил бы сеть режиму, которого нет ни на одном cutoff'е, —
    ровно та ошибка, за которую `exp_027` заплатил 0.005 LB.
    """
    cuts = cutoff_grid(MIN_HISTORY, CUTOFF_STEP)
    lo, hi = day_index(cuts[0]) + 1, day_index(cuts[-1]) + 1
    assert (lo, hi) == (93, 289)
    assert max(DEPTH_GRID) == hi, "верх сетки обязан совпадать с боевой политикой clip289"
    assert min(DEPTH_GRID) >= 90 and max(DEPTH_GRID) <= SEQ_L
    assert list(DEPTH_GRID) == sorted(set(DEPTH_GRID))
    # фолдовые модели видят меньшую глубину — 254 у последнего фолда
    assert day_index(fold_cutoffs(V_LAST)[-1]) + 1 == 254
    assert 254 in DEPTH_GRID


# ------------------------------------------- depth curriculum в обучающем батче
def test_depth_curriculum_batch_changes_history_and_never_the_target():
    """Меняется вход, но не таргет и не порядок строк."""
    T = V_LAST
    cuts, ci, ri, y, group = _one_group(T)
    base = Batcher(cuts, ci, ri, y, 512, 256, np.random.default_rng(0), 1)
    cur = Batcher(cuts, ci, ri, y, 512, 256, np.random.default_rng(0), 1,
                  depth=dict(p=0.5, grid=DEPTH_GRID))
    x0, y0 = base._make(group, seed=7)
    x1, y1 = cur._make(group, seed=7)
    assert np.array_equal(y0, y1), "таргет зависит от curriculum"
    assert not np.array_equal(x1[:, :, :N_CH_STORED], x0[:, :, :N_CH_STORED])
    assert np.array_equal(x1[:, :, AV_COL + 1:], x0[:, :, AV_COL + 1:])   # dow_sin/dow_cos
    # хвост окна не тронут ни у одной строки: режется только прошлое
    keep = SEQ_L - max(DEPTH_GRID)
    assert np.array_equal(x1[:, SEQ_L - min(DEPTH_GRID):, :], x0[:, SEQ_L - min(DEPTH_GRID):, :])
    assert keep >= 0


def test_depth_curriculum_batch_is_reproducible_per_seed():
    T = V_LAST
    cuts, ci, ri, y, group = _one_group(T)
    b = Batcher(cuts, ci, ri, y, 512, 256, np.random.default_rng(0), 1,
                depth=dict(p=0.5, grid=DEPTH_GRID))
    assert np.array_equal(b._make(group, seed=11)[0], b._make(group, seed=11)[0])
    assert not np.array_equal(b._make(group, seed=11)[0], b._make(group, seed=12)[0])


def test_depth_curriculum_leaves_the_base_path_bitwise_intact():
    """При `p = 0` вход обязан быть побитово тем же, что до EXP-030.

    Без этого BASE перестаёт быть контролем: разницу нельзя было бы отнести
    ни к curriculum, ни к коду сборки батча.
    """
    T = V_LAST
    cuts, ci, ri, y, group = _one_group(T)
    ref = np.concatenate([gather(T, ri[idx]) for _, idx in group])
    for depth in (None, dict(p=0.0, grid=DEPTH_GRID)):
        b = Batcher(cuts, ci, ri, y, 512, 256, np.random.default_rng(0), 1, depth=depth)
        assert np.array_equal(b._make(group, seed=12345)[0], ref)
        assert np.array_equal(b._make(group, seed=999)[0], ref)


def test_depth_curriculum_does_not_shift_the_batch_order_stream():
    """План эпохи у BASE и у curriculum обязан совпадать: сравнение парное.

    Сиды аугментации берутся из отдельного потока (`aug_seed`), поэтому
    `self.rng`, из которого строится план, curriculum не двигает. Иначе разница
    между вариантами мешалась бы с шумом порядка данных (std wCV TCN 0.00250).
    """
    T = V_LAST
    cuts, ci, ri, y, _ = _one_group(T)
    plans = []
    for depth in (None, dict(p=0.5, grid=DEPTH_GRID)):
        b = Batcher(cuts, ci, ri, y, 512, 256, np.random.default_rng(42), 1,
                    aug_seed=[42, 0xA7A1], depth=depth)
        for _ in b:                                   # эпоха 1 целиком
            pass
        plans.append([(k, idx.copy()) for grp in b._plan() for k, idx in grp])
    assert len(plans[0]) == len(plans[1])
    for (k0, i0), (k1, i1) in zip(*plans):
        assert k0 == k1 and np.array_equal(i0, i1), "план эпохи разошёлся из-за curriculum"


@pytest.fixture(scope="module")
def tiny_model():
    import torch
    from src.seq import build_model
    torch.manual_seed(0)
    cfg = dict(DEFAULT_CFG, hidden=16, blocks=4, kernel=3, dropout=0.0, batch=256, z0=2.3)
    m = build_model(cfg)
    # `build_model` зануляет последний слой (и residual-ветви), поэтому свежая
    # модель выдаёт константу и никакой вход её не сдвинет. Для проверки того,
    # что `avail` вообще доходит до выхода, веса нужно расшевелить.
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() > 1:
                p.add_(torch.randn_like(p) * 0.05)
    return m.eval(), cfg, torch.device("cpu")


@pytest.mark.parametrize("aug", ["none", "avail_drop", "avail_bnd"])
def test_inference_is_deterministic_and_never_augmented(tiny_model, aug):
    """Инференс не трогает `avail` ни в одном режиме обучения и повторяем побитово."""
    from src.seq import predict
    m, cfg, dev = tiny_model
    c = dict(cfg, aug=aug, aug_p=1.0, aug_full=1.0)
    r = _rows(V_LAST, 256)
    z1 = predict(m, V_LAST, r, c, dev)
    z2 = predict(m, V_LAST, r, c, dev)
    assert np.array_equal(z1, z2)
    assert np.array_equal(z1, predict(m, V_LAST, r, dict(cfg, aug="none"), dev))


def test_inference_never_applies_the_depth_curriculum(tiny_model):
    """Валидация и тест не аугментируются: `depth_aug` на инференс не влияет.

    Глубина на инференсе задаётся ТОЛЬКО явным `depth_clip` — боевой политикой
    `--depth-clip 289` или диагностикой. Иначе OOF перестал бы быть прогнозом
    того рецепта, который уходит в сабмит (`exp_027` §OOF/LOFO).
    """
    from src.seq import predict
    m, cfg, dev = tiny_model
    r = _rows(V_LAST, 256)
    z0 = predict(m, V_LAST, r, dict(cfg, depth_aug=0.0), dev)
    assert np.array_equal(z0, predict(m, V_LAST, r, dict(cfg, depth_aug=1.0), dev))
    assert np.array_equal(z0, predict(m, V_LAST, r, dict(cfg, depth_aug=1.0,
                                                         depth_grid=(90,)), dev))
    # а явный клип обязан двигать прогноз — иначе тест выше ничего не проверяет
    assert not np.array_equal(z0, predict(m, V_LAST, r, cfg, dev, depth_clip=150))


def test_availcurve_endpoint_equals_the_availprobe_point(tiny_model):
    """Кривая по положению границы обязана в крайней точке совпасть с `availprobe`.

    Иначе две диагностики мерили бы разные вещи, и «плоская кривая» не означала
    бы «нулевой availprobe».
    """
    from src.seq import n_zero_positions, predict
    m, cfg, dev = tiny_model
    r = _rows(V_LAST, 256)
    n0 = n_zero_positions(V_LAST)
    assert np.array_equal(predict(m, V_LAST, r, cfg, dev, avail_shift=n0),
                          predict(m, V_LAST, r, cfg, dev, avail_full=True))
    assert np.array_equal(predict(m, V_LAST, r, cfg, dev, avail_shift=0),
                          predict(m, V_LAST, r, cfg, dev))


def test_availprobe_is_the_same_operation_for_every_variant(tiny_model):
    """`availprobe` — один и тот же код у BASE и у новых вариантов.

    У всех режимов, кроме контроля `no_avail`, он переключает ровно один бит и
    обязан двигать прогноз. У `no_avail` канал константен ПО ОПРЕДЕЛЕНИЮ модели,
    поэтому Δ обязана быть ровно нулевой — это и есть проверка того, что контроль
    действительно лишён границы, а не «получил хороший результат».
    """
    from src.seq import predict
    m, cfg, dev = tiny_model
    r = _rows(V_LAST, 256)
    for aug in ("none", "avail_drop", "avail_bnd"):
        c = dict(cfg, aug=aug)
        assert not np.array_equal(predict(m, V_LAST, r, c, dev),
                                  predict(m, V_LAST, r, c, dev, avail_full=True))
    c = dict(cfg, aug="no_avail")
    assert np.array_equal(predict(m, V_LAST, r, c, dev),
                          predict(m, V_LAST, r, c, dev, avail_full=True))
    # и это именно режим avail ≡ 1, а не «пропущенный вызов»
    assert np.array_equal(predict(m, V_LAST, r, c, dev),
                          predict(m, V_LAST, r, dict(cfg, aug="none"), dev, avail_full=True))
