"""Анти-лукап и корректность событийного представления ETX-01.

Стоп-условие, а не пожелание (`STRATEGY_13` §Evaluation: «тесты причинности маски
и запрет на позиции после cutoff'а — с нулевым допуском»). Событийная
токенизация опаснее плотной ровно тем, что окно берётся поиском по
отсортированному ключу, а не срезом массива: ошибка на единицу в границе
`searchsorted` даёт «слишком хороший» результат и ничем другим себя не проявляет.

Проверяются четыре класса ошибок:

  1. в последовательность попал день ПОСЛЕ cutoff'а (или до начала окна);
  2. токен не равен строке панели за тот же день — то есть представление
     разошлось с тем, что читает `SEQ`, и сравнение архитектур перестало быть
     сравнением архитектур;
  3. внимание видит будущее внутри последовательности либо паддинг;
  4. ALiBi-смещение, свёрнутое в лишнее измерение `q`/`k`, не равно честной
     аддитивной матрице `−Δt/τ_h` (тогда «время во внимании» — это не время).

Запуск: python -m pytest src/test_etx.py -q
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from src.config import TARGET_DAYS, VAL_FOLDS_S1
from src.etx import (DEFAULT_CFG, N_STATIC, N_TOK_FEAT, TAU_UNIT, Batcher, Tokenizer,
                     build_model, events, n_params, predict, select)
from src.features import panel_users
from src.seq import (CHANNELS, N_CH_STORED, SEQ_L, build_index, day_index, fold_cutoffs,
                     gather, panel, target_at, user_rows)

V_LAST = VAL_FOLDS_S1[-1]
T_MID = dt.date(2025, 7, 3)
T_EARLY = dt.date(2025, 4, 3)
ROWS = np.arange(0, 250_000, 617, dtype=np.int64)      # 406 пользователей вразнобой


def _dev():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------- таблица событий
def test_event_table_is_globally_sorted_and_complete():
    """Ключ строго возрастает, а число событий пользователя равно числу его строк."""
    _, day, key, ptr = events()
    assert bool((np.diff(key) > 0).all())
    p, _, _, _ = panel()
    pi = CHANNELS.index("present")
    for r in ROWS[::40]:
        lo, hi = int(ptr[r]), int(ptr[r + 1])
        ref = np.flatnonzero(p[r, :, pi] > 0)
        assert np.array_equal(day[lo:hi].astype(np.int64), ref)


# --------------------------------------------------------------------- окно и причинность
@pytest.mark.parametrize("T", [T_EARLY, T_MID, V_LAST])
def test_no_event_after_cutoff(T):
    """Главный анти-лукап: ни одного дня строго позже cutoff'а, допуск 0."""
    _, day, _, _ = events()
    idx, cnt = select(T, ROWS, DEFAULT_CFG["n_tok"])
    d = day_index(T)
    for i in range(len(ROWS)):
        dd = day[idx[i, :cnt[i]]].astype(np.int64)
        assert dd.size == 0 or dd.max() <= d, f"событие после cutoff'а {T}"


@pytest.mark.parametrize("T", [T_EARLY, T_MID, V_LAST])
def test_window_bounds_and_strict_order(T):
    _, day, _, _ = events()
    idx, cnt = select(T, ROWS, DEFAULT_CFG["n_tok"])
    d = day_index(T)
    lo = max(0, d - SEQ_L + 1)
    for i in range(len(ROWS)):
        dd = day[idx[i, :cnt[i]]].astype(np.int64)
        assert (dd >= lo).all(), "событие старше окна 365 дней"
        assert (np.diff(dd) > 0).all(), "дни не строго возрастают"


@pytest.mark.parametrize("T", [T_EARLY, T_MID, V_LAST])
def test_selection_equals_present_days_of_dense_window(T):
    """Множество токенов = множество дней с `present = 1` в плотном окне `seq.gather`."""
    _, day, _, _ = events()
    K = DEFAULT_CFG["n_tok"]
    idx, cnt = select(T, ROWS, K)
    x = gather(T, ROWS)
    d = day_index(T)
    lo = max(0, d - SEQ_L + 1)
    off = SEQ_L - (d + 1 - lo)
    for i in range(len(ROWS)):
        dense = np.flatnonzero(x[i, :, CHANNELS.index("present")] > 0) - off + lo
        assert np.array_equal(day[idx[i, :cnt[i]]].astype(np.int64), dense[-K:])
        assert cnt[i] == min(len(dense), K)


def test_tokens_match_panel_rows_bitwise():
    """Признаки токена = строка панели за тот же день. Иначе представления разошлись."""
    p, _, _, _ = panel()
    x, day, _, _ = events()
    idx, cnt = select(T_MID, ROWS, DEFAULT_CFG["n_tok"])
    for i, r in enumerate(ROWS):
        sel = idx[i, :cnt[i]]
        assert np.array_equal(x[sel], p[r, day[sel].astype(np.int64), :])


def test_selection_does_not_depend_on_the_target_window():
    """Порча данных в (T, T+30] не меняет ни одного токена: окно кончается на T."""
    x, day, _, _ = events()
    T = T_MID
    d = day_index(T)
    idx, cnt = select(T, ROWS, DEFAULT_CFG["n_tok"])
    ref = (idx.copy(), cnt.copy())
    hit = np.isin(day.astype(np.int64), np.arange(d + 1, d + 1 + TARGET_DAYS))
    saved = x[hit].copy()
    x[hit] = 7.0
    try:
        a, c = select(T, ROWS, DEFAULT_CFG["n_tok"])
        assert np.array_equal(a, ref[0]) and np.array_equal(c, ref[1])
        # и сами токены: индексы указывают только на дни <= T
        for i in range(len(ROWS)):
            assert (x[a[i, :c[i]]] != 7.0).any(axis=1).all() or c[i] == 0
    finally:
        x[hit] = saved


def test_prediction_is_invariant_to_any_future_corruption():
    """End-to-end: порча ВСЕХ событий после cutoff'а не двигает прогноз ни на бит."""
    import torch
    x, day, _, _ = events()
    T = T_MID
    d = day_index(T)
    dev = _dev()
    torch.manual_seed(0)
    cfg = dict(DEFAULT_CFG, d_model=32, blocks=2, ffn=64, z0=2.7, batch=256)
    m = build_model(cfg).to(dev).eval()
    tk = Tokenizer(dev)
    rows = ROWS[:256]
    z0 = predict(m, tk, T, rows, cfg, dev)
    future = day.astype(np.int64) > d
    saved = x[future].copy()
    x[future] = 13.0
    try:
        tk2 = Tokenizer(dev)                       # таблица перезалита на устройство
        assert np.array_equal(z0, predict(m, tk2, T, rows, cfg, dev))
    finally:
        x[future] = saved


# --------------------------------------------------------------------- глубина истории
@pytest.mark.parametrize("D", [90, 180, 254])
def test_depth_clip_only_drops_the_oldest_events(D):
    _, day, _, _ = events()
    K = DEFAULT_CFG["n_tok"]
    a0, c0 = select(V_LAST, ROWS, K)
    a1, c1 = select(V_LAST, ROWS, K, depth_clip=D)
    d = day_index(V_LAST)
    for i in range(len(ROWS)):
        full = day[a0[i, :c0[i]]].astype(np.int64)
        cut = day[a1[i, :c1[i]]].astype(np.int64)
        assert np.array_equal(cut, full[full > d - D])
        assert cut.size == 0 or cut.max() <= d


def test_depth_clip_beyond_available_history_is_a_noop():
    a0, c0 = select(T_EARLY, ROWS, DEFAULT_CFG["n_tok"])
    a1, c1 = select(T_EARLY, ROWS, DEFAULT_CFG["n_tok"], depth_clip=SEQ_L)
    assert np.array_equal(a0, a1) and np.array_equal(c0, c1)


def test_token_limit_keeps_the_most_recent_events():
    """Переполнение окна прореживает СТАРЫЕ токены; свежие не трогаются никогда."""
    _, day, _, _ = events()
    big, cb = select(V_LAST, ROWS, 512)
    small, cs = select(V_LAST, ROWS, 32)
    for i in range(len(ROWS)):
        assert cs[i] == min(cb[i], 32)
        assert np.array_equal(day[small[i, :cs[i]]], day[big[i, cb[i] - cs[i]:cb[i]]])


def test_test_cutoff_clip_is_actually_applied():
    """Боевая политика теста: `--depth-clip 289` обязана РЕАЛЬНО резать вход.

    Ровно эта проверка отсутствовала в `exp_027`, где полная глубина на тесте
    стоила +0.0051 LB. На фолдах клип часто no-op (глубина и так меньше), а на
    тестовом cutoff'е доступны все 365 дней — если бы клип молча не срабатывал,
    ошибка была бы невидима до самого лидерборда.
    """
    from src.config import CUTOFF_TEST
    _, day, _, _ = events()
    d = day_index(CUTOFF_TEST)
    a0, c0 = select(CUTOFF_TEST, ROWS, 512)
    a1, c1 = select(CUTOFF_TEST, ROWS, 512, depth_clip=289)
    assert (c1 <= c0).all() and (c1 < c0).any(), "клип на тестовом cutoff'е ничего не срезал"
    for i in range(len(ROWS)):
        full = day[a0[i, :c0[i]]].astype(np.int64)
        cut = day[a1[i, :c1[i]]].astype(np.int64)
        assert cut.size == 0 or cut.max() <= d, "событие после тестового cutoff'а"
        assert np.array_equal(cut, full[full > d - 289])
        assert cut.size == 0 or cut.min() > d - 289


def test_test_corridor_targets_are_strictly_in_the_past():
    """Обучающие cutoff'ы тестовой модели не заходят в отравленное окно панели."""
    import datetime as _dt

    from src.config import CORRIDOR_END, CUTOFF_STEP, TARGET_DAYS, cutoff_grid
    from src.seq import MIN_HISTORY
    cuts = cutoff_grid(MIN_HISTORY, CUTOFF_STEP, CORRIDOR_END)
    assert max(cuts) == CORRIDOR_END
    assert all(T + _dt.timedelta(days=TARGET_DAYS) <= _dt.date(2025, 11, 15) for T in cuts)


# --------------------------------------------------------------------- модель
def test_parameter_budget():
    n = n_params()
    assert 1_000_000 <= n <= 1_500_000, f"параметров {n:,} вне бюджета гейта 1.0–1.5 млн"


def test_attention_is_causal_inside_the_sequence():
    """h[:, t] не меняется при изменении входа в позициях > t. Допуск ровно 0."""
    import torch
    torch.manual_seed(0)
    cfg = dict(DEFAULT_CFG, d_model=32, blocks=2, ffn=64, z0=0.0, dropout=0.0)
    m = build_model(cfg).eval()
    B, K, t = 3, 24, 10
    tok = torch.randn(B, K, N_TOK_FEAT)
    st = torch.randn(B, N_STATIC)
    age = torch.arange(K, 0, -1).float().unsqueeze(0).repeat(B, 1)
    n = torch.full((B,), K, dtype=torch.long)

    def hidden(x):
        h = torch.zeros(B, K + 1, cfg["d_model"])
        h[:, :K] = m.tok(x)
        h = h.scatter(1, n.view(B, 1, 1).expand(B, 1, cfg["d_model"]),
                      (m.cls + m.static(st)).unsqueeze(1))
        a = torch.zeros(B, K + 1)
        a[:, :K] = age
        a = a / TAU_UNIT
        for b in m.blocks:
            h = b(h, a)
        return h

    with torch.no_grad():
        h0 = hidden(tok)
        tok2 = tok.clone()
        tok2[:, t + 1:] = torch.randn_like(tok2[:, t + 1:])
        h1 = hidden(tok2)
    assert torch.equal(h0[:, :t + 1], h1[:, :t + 1]), "внимание видит будущее"
    assert not torch.equal(h0[:, t + 1:], h1[:, t + 1:])


def test_prediction_ignores_padding_slots():
    """Всё, что за позицией `n`, не влияет на прогноз: query стоит ровно на `n`."""
    import torch
    torch.manual_seed(1)
    cfg = dict(DEFAULT_CFG, d_model=32, blocks=2, ffn=64, z0=2.7, dropout=0.0)
    m = build_model(cfg).eval()
    B, K = 4, 20
    tok = torch.randn(B, K, N_TOK_FEAT)
    st = torch.randn(B, N_STATIC)
    age = torch.arange(K, 0, -1).float().unsqueeze(0).repeat(B, 1)
    n = torch.tensor([3, 7, 12, K], dtype=torch.long)
    with torch.no_grad():
        z0 = m(tok, st, age, n)
        tok2, age2 = tok.clone(), age.clone()
        for i, k in enumerate(n.tolist()):
            tok2[i, k:] = torch.randn(K - k, N_TOK_FEAT)
            age2[i, k:] = torch.rand(K - k) * 400
        z1 = m(tok2, st, age2, n)
    assert torch.equal(z0, z1), "паддинг влияет на прогноз"


def test_alibi_extra_dim_equals_explicit_time_bias():
    """Свёрнутое в измерение `q`/`k` смещение = честная матрица `−Δt/τ_h`.

    Проверяется на одном блоке: логиты, посчитанные трюком, обязаны совпасть с
    логитами `q·k·scale + b_h(i,j)`, где `b_h = −(age_j − age_i)/τ_h`.
    """
    import torch
    torch.manual_seed(2)
    cfg = dict(DEFAULT_CFG, d_model=32, blocks=1, heads=4, head_dim=8, ffn=64,
               z0=0.0, dropout=0.0)
    m = build_model(cfg).eval()
    blk = m.blocks[0]
    with torch.no_grad():
        blk.log_m.copy_(torch.tensor([0.3, -0.7, 1.1, 0.0]))
    B, L, H, dqk = 2, 9, blk.h, blk.dqk
    x = torch.randn(B, L, cfg["d_model"])
    age = torch.arange(L, 0, -1).float().unsqueeze(0).repeat(B, 1)
    with torch.no_grad():
        xn = blk.n1(x)
        q = blk.q(xn).view(B, L, H, dqk).transpose(1, 2)
        k = blk.k(xn).view(B, L, H, dqk).transpose(1, 2)
        mm = blk.log_m.exp().view(1, H, 1, 1).expand(B, H, L, 1)
        aa = (-age / TAU_UNIT).view(B, 1, L, 1).expand(B, H, L, 1)
        trick = (torch.cat([q, mm], -1) @ torch.cat([k, aa], -1).transpose(-2, -1)) * blk.scale
        tau = blk.taus().view(1, H, 1, 1)
        dtm = age.view(B, 1, 1, L) - age.view(B, 1, L, 1)      # Δt_ij = age_j − age_i
        honest = (q @ k.transpose(-2, -1)) * blk.scale - dtm / tau
    # softmax инвариантен к сдвигу по строке: сравниваем логиты, центрированные по j
    a = trick - trick.mean(-1, keepdim=True)
    b = honest - honest.mean(-1, keepdim=True)
    assert torch.allclose(a, b, atol=1e-4), (a - b).abs().max()


def test_tau_initialisation_spans_the_intended_scales():
    m = build_model(dict(DEFAULT_CFG, z0=0.0))
    t = m.blocks[0].taus().numpy()
    assert np.isclose(t.min(), DEFAULT_CFG["tau_lo"], rtol=1e-3)
    assert np.isclose(t.max(), DEFAULT_CFG["tau_hi"], rtol=1e-3)
    assert np.all(np.diff(t) > 0)


# --------------------------------------------------------------------- обвязка
def test_fold_cutoffs_and_panels_match_seq():
    """Схема валидации ETX обязана совпадать с SEQ — иначе числа несравнимы."""
    V = V_LAST
    cuts = fold_cutoffs(V)
    assert max(cuts) + dt.timedelta(days=TARGET_DAYS) <= V
    assert panel_users(V, 3).height > panel_users(V, 1).height * 0.5
    ci, ri, zy = build_index(cuts[-2:], blocks=1)
    assert len(ci) == len(ri) == len(zy)
    assert np.isfinite(zy).all() and zy.min() >= 0


def test_batcher_uses_every_example_exactly_once():
    V = V_LAST
    cuts = fold_cutoffs(V)[-2:]
    ci, ri, zy = build_index(cuts, blocks=1)
    b = Batcher(cuts, ci, ri, zy, 512, 128, 64, np.random.default_rng(0))
    seen, nb = 0, 0
    for idx, cnt, cd, yb in b:
        assert idx.shape == (len(yb), 64) and cnt.shape == (len(yb),)
        assert (cnt > 0).all() and (cnt <= 64).all()
        assert np.isin(cd, [day_index(T) for T in cuts]).all()
        seen += len(yb)
        nb += 1
    assert seen == len(zy)
    assert nb == b.n_batches()


def test_val_rows_match_the_seq_oof():
    """Панель и таргет фолда 10-16 совпадают со строками уже посчитанного SEQ."""
    from src.tracking import load_oof
    d = load_oof("SEQ-D3A-S42-V1016")
    uv = panel_users(V_LAST, 3)["user_id"].to_numpy()
    assert np.array_equal(np.asarray(d["user_id"]), uv)
    assert np.allclose(target_at(V_LAST, user_rows(uv)), np.asarray(d["y"], float))


# ------------------------------------------------- статик query-токена (`EXP-037`)
def _static_of(T, depth_cap=None, cdow_shift=0.0):
    """Статик query-токена для набора строк при заданной политике инференса."""
    import torch
    tk = Tokenizer(_dev())
    tk.depth_cap, tk.cdow_shift = depth_cap, cdow_shift
    idx, cnt = select(T, ROWS, DEFAULT_CFG["n_tok"])
    cd = np.full(len(ROWS), day_index(T), np.int32)
    _, st, _, _ = tk(torch.from_numpy(idx).to(tk.dev),
                     torch.from_numpy(cnt).to(tk.dev),
                     torch.from_numpy(cd).to(tk.dev))
    return st.float().cpu().numpy()


def test_static_policy_is_inert_by_default():
    """`depth_cap=None`, `cdow_shift=0` обязаны давать ПОБИТОВО прежний статик.

    Швы `EXP-037` добавлены в горячий путь, через который прошли все прогоны
    `exp_036`. Если поведение по умолчанию сдвинулось хоть на бит, все ранее
    сохранённые OOF перестают быть сравнимыми с новыми — а на них стоит весь
    LOFO. Поэтому инертность проверяется, а не декларируется.
    """
    from src.config import CUTOFF_TEST
    for T in (T_MID, VAL_FOLDS_S1[-1], CUTOFF_TEST):
        a = _static_of(T)
        b = _static_of(T, depth_cap=None, cdow_shift=0.0)
        assert np.array_equal(a, b), f"{T}: поведение по умолчанию изменилось"


def test_depth_cap_clips_only_the_depth_channels():
    """`depth_cap=D` трогает ровно два канала статика и ничего больше.

    Смысл шва — привести ПАРУ «окно / заявленная глубина» в тот вид, который
    встречался в обучении (`exp_027` сделал это для канала `avail` у TCN). Если
    бы он двигал ещё и число событий или календарь, политика теста меняла бы
    больше одного заявленного бита.
    """
    from src.config import CUTOFF_TEST
    raw = _static_of(CUTOFF_TEST)
    cap = _static_of(CUTOFF_TEST, depth_cap=289)
    assert np.array_equal(raw[:, 2:], cap[:, 2:]), "тронуты каналы помимо глубины"
    assert (cap[:, 0] < raw[:, 0]).all(), "глубина не уменьшилась"
    assert np.allclose(cap[:, 0], 289.0 / 365.0), "глубина не равна потолку"
    assert np.allclose(cap[:, 1], np.log1p(289.0) / np.log1p(365.0))


def test_depth_cap_above_calendar_depth_is_a_noop():
    """На фолде, где календарная глубина меньше потолка, шов обязан быть no-op."""
    V = VAL_FOLDS_S1[-1]                       # 2025-10-16, глубина ровно 289
    assert np.array_equal(_static_of(V), _static_of(V, depth_cap=289))
    assert np.array_equal(_static_of(V), _static_of(V, depth_cap=10 ** 6))


def test_cdow_shift_moves_only_the_cutoff_weekday():
    """`cdow_shift` меняет только календарь query и возвращается на место за 7 шагов."""
    from src.config import CUTOFF_TEST
    raw = _static_of(CUTOFF_TEST)
    sh = _static_of(CUTOFF_TEST, cdow_shift=-1.0)
    assert np.array_equal(raw[:, :4], sh[:, :4]), "тронуты каналы помимо дня недели"
    assert not np.allclose(raw[:, 4:], sh[:, 4:]), "день недели не сдвинулся"
    assert np.allclose(raw, _static_of(CUTOFF_TEST, cdow_shift=7.0), atol=1e-6)


def test_cutoff_weekday_is_constant_in_training_and_differs_on_test():
    """Обоснование шва: в обучении `cdow` — КОНСТАНТА, а тест — единственная точка,
    где этот вход меняется. Именно поэтому его нельзя оставлять «как есть»."""
    from src.config import CUTOFF_TEST, cutoff_grid
    wd = {T.weekday() for T in cutoff_grid()}
    assert len(wd) == 1, f"обучающие cutoff'ы уже не одного дня недели: {wd}"
    assert {V.weekday() for V in VAL_FOLDS_S1} == wd, "фолды разошлись с обучением"
    assert CUTOFF_TEST.weekday() not in wd, "тест совпал с обучением — шов не нужен"
