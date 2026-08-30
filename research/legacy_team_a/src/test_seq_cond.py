"""Анти-лукап и корректность EXP-032 (S04): conditional head на замороженном энкодере.

Этот эксперимент — единственное место в проекте, где в обучение сознательно
попадают cutoff'ы ПОСЛЕ `CORRIDOR_END`. Поэтому проверяется не «работает ли
код», а ровно те три канала, через которые он мог бы дать «слишком хороший»
результат:

  1. **строки EXTRA пришли от пользователей, на которых считается метрика** —
     их таргет-окна пересекают валидационное, и это была бы прямая утечка
     будущего того же человека (`N9`: корреляция таргетов одного пользователя
     на сдвиге 60 дней = 0.498);
  2. **EXTRA просочился в экстенсив** — именно вероятность активности отравлена
     правилом отбора панели (`exp_028`, `e08`), и обучать её на поздних
     cutoff'ах нельзя ни в каком виде;
  3. **эмбеддинг не тот, который видит боевая голова** — тогда «замороженный
     энкодер» перестал бы быть тем самым энкодером, и сравнение с базой
     потеряло бы смысл.

Плюс тождества, без которых варианты несравнимы: одинаковый шаговый бюджет и
выбор строк ТОЛЬКО через `rows`, а не через отдельную матрицу.

Запуск: python -m pytest src/test_seq_cond.py -q
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from src.config import (ARTIFACTS, CORRIDOR_END, CUTOFF_STEP, DATA_END, TARGET_DAYS,
                        VAL_FOLDS_S1)
from src.features import panel_users
from src.seq import (N_CH_STORED, SEQ_L, day_index, fold_cutoffs, gather, panel,
                     target_at, user_rows)
from src.seq_cond import (EMB_DIM_MULT, EXTRA_CUTOFFS, _pool, audit_extra, build_head,
                          collect, embed, fit_head, segments, user_group)

V = dt.date(2025, 10, 16)
CKPT = "SEQ-D3A-BASE-S42-V1016"
needs_ckpt = pytest.mark.skipif(not (ARTIFACTS / f"model_{CKPT}.pt").exists(),
                                reason=f"нет чекпойнта {CKPT}")


@pytest.fixture(scope="module")
def enc():
    from src.seq import load_ckpt
    model, cfg, Vc, dev = load_ckpt(CKPT)
    return model, cfg, Vc, dev


# ------------------------------------------------------------------ сетка EXTRA
def test_extra_cutoffs_are_outside_the_clean_corridor():
    assert len(EXTRA_CUTOFFS) == 13
    assert all(T > CORRIDOR_END for T in EXTRA_CUTOFFS)
    assert EXTRA_CUTOFFS[0] == dt.date(2025, 10, 22)
    assert EXTRA_CUTOFFS[-1] == dt.date(2026, 1, 14)


def test_extra_grid_is_step_seven_and_sorted():
    d = np.diff([day_index(T) for T in EXTRA_CUTOFFS])
    assert (d == CUTOFF_STEP).all(), f"шаг сетки EXTRA {set(d.tolist())}, ожидался {CUTOFF_STEP}"


def test_latest_extra_target_window_ends_exactly_at_the_data_border():
    """Данных после 2026-02-13 не существует, поэтому лукапа ОТНОСИТЕЛЬНО ТЕСТА нет."""
    assert EXTRA_CUTOFFS[-1] + dt.timedelta(days=TARGET_DAYS) == DATA_END
    for T in EXTRA_CUTOFFS:
        assert T + dt.timedelta(days=TARGET_DAYS) <= DATA_END


def test_overlap_flag_matches_the_dates_fold_by_fold():
    """Пересечение окон таргета есть у поздних фолдов и нет у ранних.

    Это диагностика, а НЕ условие применимости расщепления: строки EXTRA
    отдаёт группа B на любом фолде. Канал утечки — корреляция таргетов одного
    человека во времени (`N9`), она не требует пересечения окон; плюс признаки
    строки EXTRA всегда содержат валидационное окно фолда целиком.
    """
    for f in VAL_FOLDS_S1:
        a = audit_extra(f)
        want = EXTRA_CUTOFFS[0] < f + dt.timedelta(days=TARGET_DAYS)
        assert a["overlap_with_val_window"] is bool(want), f
    assert audit_extra(dt.date(2025, 10, 16))["overlap_with_val_window"] is True
    assert audit_extra(dt.date(2025, 9, 4))["overlap_with_val_window"] is False


def test_extra_features_always_reach_into_the_validation_target_window():
    """Окно признаков строки EXTRA всегда залезает в окно таргета фолда.

    Поэтому расщепление обязательно и на ранних фолдах: без него голова видела
    бы поведение тех же людей ВНУТРИ валидационного окна. И наоборот, полное
    покрытие окна таргета признаками EXTRA бывает ровно тогда, когда окна
    таргета НЕ пересекаются, — это одно и то же условие `T >= f + 30`.
    """
    T = EXTRA_CUTOFFS[0]
    for f in VAL_FOLDS_S1:
        assert T - dt.timedelta(days=SEQ_L - 1) <= f + dt.timedelta(days=1),             "окно признаков EXTRA начинается позже начала окна таргета фолда"
        assert T > f, "EXTRA обязан лежать в будущем от фолда"
        covers_fully = T >= f + dt.timedelta(days=TARGET_DAYS)
        assert covers_fully is not audit_extra(f)["overlap_with_val_window"]


def test_extra_is_never_a_training_cutoff_of_the_fold():
    assert not (set(fold_cutoffs(V)) & set(EXTRA_CUTOFFS))


# ------------------------------------------------------- расщепление по людям
def test_user_group_is_deterministic_and_order_independent():
    u = panel_users(V, 3)["user_id"].to_numpy()
    g = user_group(u)
    assert np.array_equal(g, user_group(u))
    p = np.random.default_rng(0).permutation(len(u))
    assert np.array_equal(g[p], user_group(u[p])), "группа зависит от порядка строк"


def test_user_group_is_balanced_and_binary():
    u = panel_users(V, 3)["user_id"].to_numpy()
    g = user_group(u)
    assert set(np.unique(g).tolist()) <= {0, 1}
    assert abs(float(g.mean()) - 0.5) < 0.01


def test_user_group_is_the_same_on_every_cutoff():
    """Иначе пользователь мог бы попасть в EXTRA на одном cutoff'е и в метрику на другом."""
    a = panel_users(V, 3)["user_id"].to_numpy()
    b = panel_users(EXTRA_CUTOFFS[-1], 1)["user_id"].to_numpy()
    common = np.intersect1d(a, b)
    assert len(common) > 10_000
    ga = user_group(a)[np.searchsorted(a, common)]
    gb = user_group(b)[np.searchsorted(b, common)]
    assert np.array_equal(ga, gb)


def test_metric_group_and_extra_group_are_disjoint():
    ua = panel_users(V, 3)["user_id"].to_numpy()
    metric = ua[user_group(ua) == 0]
    for T in (EXTRA_CUTOFFS[0], EXTRA_CUTOFFS[-1]):
        ue = panel_users(T, 1)["user_id"].to_numpy()
        donors = ue[user_group(ue) == 1]
        assert np.intersect1d(metric, donors).size == 0


# ------------------------------------------------------------------ эмбеддинги
@needs_ckpt
def test_pooled_embedding_is_bitwise_the_production_head_input(enc):
    """`head(pool(encode(x)))` обязан совпасть с `model(x)` — иначе это другой энкодер."""
    import torch
    model, cfg, Vc, dev = enc
    u = panel_users(Vc, 3)["user_id"].to_numpy()[:512]
    r = user_rows(u)
    _, _, _, sc_np = panel()
    sc = torch.from_numpy(sc_np).to(dev).view(1, N_CH_STORED, 1)
    x = gather(Vc, r)
    t = torch.from_numpy(x).to(dev).permute(0, 2, 1).contiguous().float()
    t[:, :N_CH_STORED] *= sc
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                         enabled=dev.type == "cuda"):
        z = model(t).float().cpu().numpy()
        z2 = model.head(_pool(model.encode(t))).squeeze(1).float().cpu().numpy()
    assert np.array_equal(z, z2)


@needs_ckpt
def test_embedding_shape_and_dtype(enc):
    model, cfg, Vc, dev = enc
    r = user_rows(panel_users(Vc, 3)["user_id"].to_numpy()[:256])
    E = embed(model, cfg, dev, Vc, r)
    assert E.shape == (256, cfg["hidden"] * EMB_DIM_MULT)
    assert E.dtype == np.float16


@needs_ckpt
def test_embedding_does_not_touch_the_encoder(enc):
    """Заморозка: контрольная сумма параметров не меняется, градиента нет."""
    model, cfg, Vc, dev = enc
    before = float(sum(float(p.detach().double().sum()) for p in model.parameters()))
    r = user_rows(panel_users(Vc, 3)["user_id"].to_numpy()[:256])
    embed(model, cfg, dev, Vc, r)
    after = float(sum(float(p.detach().double().sum()) for p in model.parameters()))
    assert after == before
    assert all(p.grad is None for p in model.parameters())


@needs_ckpt
def test_extra_embedding_uses_the_production_depth_policy(enc):
    """Вход EXTRA при `depth_clip=289` совпадает по структуре с панелью фолда.

    Без обрезки поздний cutoff даёт `avail = 1` на всём окне — режим, которого
    энкодер не видел (`exp_027`), и эксперимент про EXTRA смешался бы с
    экспериментом про глубину.
    """
    T = EXTRA_CUTOFFS[-1]
    r = user_rows(panel_users(T, 1)["user_id"].to_numpy()[:64])
    full = gather(T, r)
    clipped = gather(T, r, depth_clip=289)
    av_full = full[0, :, N_CH_STORED]
    av_clip = clipped[0, :, N_CH_STORED]
    assert av_full.min() == 1.0, "на позднем cutoff'е полная глубина = режим avail = 1"
    assert int((av_clip == 0).sum()) == 365 - 289
    rv = user_rows(panel_users(V, 3)["user_id"].to_numpy()[:64])
    assert int((gather(V, rv)[0, :, N_CH_STORED] == 0).sum()) == 365 - 289


@needs_ckpt
def test_extra_input_never_contains_a_day_after_its_cutoff(enc):
    """Стандартный анти-лукап `gather`, но на cutoff'ах вне коридора."""
    p, _, _, _ = panel()
    for T in (EXTRA_CUTOFFS[0], EXTRA_CUTOFFS[-1]):
        r = user_rows(panel_users(T, 1)["user_id"].to_numpy()[:128])
        x = gather(T, r, depth_clip=289)
        d = day_index(T)
        got = x[:, -1, :N_CH_STORED].astype(np.float32)
        assert np.allclose(got, p[r, d, :].astype(np.float32)), "последняя позиция != день T"


# --------------------------------------------------------------- сбор выборок
@needs_ckpt
def test_collect_extra_keeps_only_group_b_positives(enc):
    model, cfg, Vc, dev = enc
    cuts = EXTRA_CUTOFFS[-1:]
    X, z, u, c = collect(model, cfg, dev, cuts, 1, keep="y>0", group_keep=1,
                         depth_clip=289, tag="t")
    assert len(u) and int(user_group(u).min()) == 1, "в EXTRA попала группа A"
    assert (z > 0).all(), "в интенсивную выборку попали строки y = 0"
    assert len(X) == len(z) == len(u) == len(c)


@needs_ckpt
def test_extra_rows_never_come_from_the_metric_group(enc):
    model, cfg, Vc, dev = enc
    _, _, ue, _ = collect(model, cfg, dev, EXTRA_CUTOFFS[-1:], 1, keep="y>0",
                          group_keep=1, depth_clip=289, tag="t")
    uv = panel_users(V, 3)["user_id"].to_numpy()
    assert np.intersect1d(np.unique(ue), uv[user_group(uv) == 0]).size == 0


# ---------------------------------------------------------- голова и её бюджет
def _pool_xy(n_a=4000, n_b=4000, dim=8, seed=0, level=2.0):
    """Два непересекающихся блока строк с РАЗНЫМ уровнем цели."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_a + n_b, dim)).astype(np.float16)
    t = np.concatenate([np.zeros(n_a), np.full(n_b, level)]).astype(np.float32)
    return X, t, np.arange(n_a), np.arange(n_a, n_a + n_b)


def test_fit_head_uses_only_the_allowed_rows():
    """Вариант задаётся списком строк; чужой блок не должен влиять ни на что.

    Шагов и lr взято столько, чтобы голова успела дойти до уровня своего блока:
    последний слой стартует с нуля (как в боевой голове), и при коротком
    прогоне тест мерил бы скорость сходимости, а не выбор строк.
    """
    import torch
    from src.seq_cond import head_predict
    dev = torch.device("cpu")
    X, t, ia, ib = _pool_xy(level=2.0)
    kw = dict(steps=800, batch=256, lr=5e-2, wd=0.0, hidden=8, dropout=0.0,
              seed=42, binary=False, dev=dev, out_bias=0.0)
    pa = float(head_predict(fit_head(X, t, rows=ia, **kw)[0], X[ia], dev).mean())
    pb = float(head_predict(fit_head(X, t, rows=ib, **kw)[0], X[ib], dev).mean())
    assert abs(pa) < 0.3, f"голова A уехала к чужому уровню: {pa}"
    assert abs(pb - 2.0) < 0.3, f"голова B не выучила свой уровень: {pb}"
    assert pb - pa > 1.0, f"блоки не разделились: {pa} vs {pb}"


def test_fit_head_is_reproducible_per_seed():
    import torch
    from src.seq_cond import head_predict
    dev = torch.device("cpu")
    X, t, ia, _ = _pool_xy()
    kw = dict(steps=40, batch=128, lr=1e-3, wd=0.0, hidden=8, dropout=0.0,
              binary=False, dev=dev, out_bias=0.0, rows=ia)
    p1 = head_predict(fit_head(X, t, seed=7, **kw)[0], X[ia], dev)
    p2 = head_predict(fit_head(X, t, seed=7, **kw)[0], X[ia], dev)
    assert np.array_equal(p1, p2)


def test_head_shape_matches_the_production_head():
    """Форма головы — та же, что у TCN: Linear -> GELU -> Dropout -> Linear."""
    from torch import nn
    net = build_head(192, 64, 0.1, 0.0)
    kinds = [type(m) for m in net]
    assert kinds == [nn.Linear, nn.GELU, nn.Dropout, nn.Linear]
    assert net[0].in_features == 192 and net[0].out_features == 64
    assert net[-1].out_features == 1
    assert float(net[-1].weight.detach().abs().sum()) == 0.0, "последний слой стартует с нуля"


# ------------------------------------------------------------------- сегменты
def test_segments_match_the_gmv_panel():
    _, g, _, _ = panel()
    r = user_rows(panel_users(V, 3)["user_id"].to_numpy()[:2000])
    s = segments(V, r)
    d = day_index(V)
    lo = max(0, d - 364)
    win = g[r, lo:d + 1] > 0
    assert np.array_equal(s["w180_days_buy"], win[:, -180:].sum(1))
    never = ~win.any(1)
    assert (s["rec_buy"][never] > 365).all()
    for i in np.flatnonzero(~never)[:50]:
        last = int(np.flatnonzero(win[i])[-1])
        assert s["rec_buy"][i] == win.shape[1] - 1 - last


def test_segments_never_look_past_the_cutoff():
    """`rec_buy` считается строго по окну ДО cutoff'а включительно."""
    _, g, _, _ = panel()
    r = user_rows(panel_users(V, 3)["user_id"].to_numpy()[:500])
    d = day_index(V)
    before = segments(V, r)
    saved = g[r, d + 1:d + 31].copy()
    g[r, d + 1:d + 31] = 12345.0            # подделываем будущее
    try:
        after = segments(V, r)
        assert np.array_equal(before["rec_buy"], after["rec_buy"])
        assert np.array_equal(before["w180_days_buy"], after["w180_days_buy"])
    finally:
        g[r, d + 1:d + 31] = saved


# --------------------------------------------------------------- разложение
def test_two_part_decomposition_is_exact_in_log_space():
    """E[z] = P(y>0) * E[z | y>0] — точно, потому что z = 0 при y = 0."""
    r = user_rows(panel_users(V, 3)["user_id"].to_numpy())
    y = target_at(V, r)
    z = np.log1p(y)
    p = float((y > 0).mean())
    assert abs(float(z.mean()) - p * float(z[y > 0].mean())) < 1e-9


# ============================================================ EXP-032B: боевой экстенсив
# Замена головы `P(y>0)` на уже существующую CLEAN-модель открывает ровно один
# новый канал: экстенсив теперь приходит ИЗВНЕ этого эксперимента, и его надо
# проверять на те же два свойства, что и всё остальное, — он обучен только на
# чистом коридоре, и он выровнен по тем же строкам, на которых считается метрика.
needs_pact = pytest.mark.skipif(
    not (ARTIFACTS / f"PACT_dist_{V.isoformat()}.npz").exists(),
    reason="нет artifacts/PACT_dist_*.npz; сначала `python -m src.dist_pact`")


@needs_pact
def test_production_extensive_is_trained_only_on_clean_cutoffs():
    """Ни один cutoff обучения `P_prod` не выходит за коридор и не нарушает T+30<=V."""
    for Vf in VAL_FOLDS_S1:
        f = ARTIFACTS / f"PACT_dist_{Vf.isoformat()}.npz"
        if not f.exists():
            continue
        cuts = [dt.date.fromisoformat(str(c)) for c in np.load(f)["cuts"]]
        assert cuts, f"{Vf}: пустая сетка"
        assert all(T <= CORRIDOR_END for T in cuts), f"{Vf}: cutoff вне коридора"
        assert all(T + dt.timedelta(days=TARGET_DAYS) <= Vf for T in cuts)
        assert not set(cuts) & set(EXTRA_CUTOFFS), f"{Vf}: EXTRA просочился в экстенсив"


@needs_pact
def test_production_extensive_reproduces_the_shipped_dist_oof():
    """`1-p0` — сигнал боевой головы, а не её пересборки: ẑ совпал с OOF смеси."""
    d = np.load(ARTIFACTS / f"PACT_dist_{V.isoformat()}.npz")
    assert float(d["max_abs_dz"]) <= 1e-6
    assert np.allclose(d["z"], d["z_ref"], atol=1e-6)


@needs_pact
def test_production_extensive_is_a_probability_on_the_metric_panel():
    from src.seq_cond import load_pprod
    uv = panel_users(V, 3)["user_id"].to_numpy()
    for name, p in load_pprod(V, uv).items():
        assert p.shape == uv.shape, name
        assert np.isfinite(p).all() and p.min() >= 0.0 and p.max() <= 1.0, name


@needs_pact
def test_pprod_alignment_is_by_user_id_not_by_position():
    """Перемешанная панель обязана дать переставленный, но тот же вектор."""
    from src.seq_cond import load_pprod
    uv = panel_users(V, 3)["user_id"].to_numpy()
    straight = load_pprod(V, uv)
    perm = np.random.default_rng(0).permutation(len(uv))
    shuffled = load_pprod(V, uv[perm])
    for k in straight:
        assert np.array_equal(straight[k][perm], shuffled[k]), k


def test_composition_is_the_same_formula_as_the_pilot():
    """`P x mu` собирается ровно как в exp_032: клип mu, затем клип произведения."""
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 1000)
    mu = rng.normal(2.0, 3.0, 1000)
    z = np.maximum(p * np.maximum(mu, 0.0), 0.0)
    assert (z >= 0).all()
    assert np.allclose(z[mu <= 0], 0.0)
    assert np.allclose(z[mu > 0], p[mu > 0] * mu[mu > 0])


@needs_pact
def test_swapping_the_extensive_does_not_touch_the_intensive_head():
    """Контраст FRESH−CLEAN живёт в mu; замена P не может его создать из ничего."""
    d = np.load(ARTIFACTS / f"S04SEQ_PILOT-S42-V{V.strftime('%m%d')}_z.npz")
    mu_c, mu_f = d["mu_COND_CLEAN"].astype(float), d["mu_COND_FRESH"].astype(float)
    assert not np.allclose(mu_c, mu_f), "головы совпали — сравнивать нечего"
    p = np.load(ARTIFACTS / f"PACT_dist_{V.isoformat()}.npz")["p_act"]
    dz = np.maximum(p * np.maximum(mu_f, 0.0), 0.0) - np.maximum(p * np.maximum(mu_c, 0.0), 0.0)
    same = np.maximum(mu_f, 0.0) == np.maximum(mu_c, 0.0)
    assert np.allclose(dz[same], 0.0), "разница появилась там, где интенсив одинаков"


def test_control_extensive_sources_are_also_clean_only():
    """Контрольные `P_prod` тоже обязаны быть чистыми — иначе они не контроль.

    `S04LGB` (табличный S04, `src/calval.py`) хранит свою сетку обучения прямо в
    артефакте: экстенсивная голова там учится ТОЛЬКО на `cuts_clean`, а
    `cuts_extra`/`cuts_early` идут в интенсив. Проверяем, что чистая сетка
    действительно чистая и что поздние cutoff'ы в ней не оказались.
    """
    for Vf in VAL_FOLDS_S1:
        f = ARTIFACTS / f"S04_fold_{Vf.isoformat()}_s42.npz"
        if not f.exists():
            continue
        d = np.load(f, allow_pickle=False)
        clean = [dt.date.fromisoformat(str(c)) for c in d["cuts_clean"]]
        assert clean, f"{Vf}: пустая сетка"
        assert all(T <= CORRIDOR_END for T in clean)
        assert all(T + dt.timedelta(days=TARGET_DAYS) <= Vf for T in clean)
        assert not set(clean) & set(EXTRA_CUTOFFS)
        extra = [dt.date.fromisoformat(str(c)) for c in d["cuts_extra"]]
        assert all(T > CORRIDOR_END for T in extra), "cuts_extra обязан лежать вне коридора"


def test_dist_head_is_exactly_two_part_so_the_donor_is_comparable():
    """`ẑ_DIST = (1 − p0)·μ_DIST` — центроид нулевого бина равен нулю.

    Из этого следует, что `DIST-TAB` и `P_DIST × μ_SEQ` делят ОДИН экстенсив и
    различаются ровно интенсивом. Без этого тождества их сравнение ничего бы
    не изолировало.
    """
    from src.models import DIST_BINS, bin_centroids, bin_labels, z_bins
    rng = np.random.default_rng(0)
    z = np.where(rng.random(20_000) < 0.4, 0.0, rng.gamma(3.0, 1.2, 20_000))
    lab = bin_labels(z, z_bins(z))
    cent = bin_centroids(z, lab, DIST_BINS)
    assert cent[0] == 0.0, "нулевой бин обязан иметь нулевой центроид"
    p = rng.dirichlet(np.ones(DIST_BINS), 500)
    zh = p @ cent
    mu = zh / (1.0 - p[:, 0])
    assert np.allclose(zh, (1.0 - p[:, 0]) * mu, atol=1e-12)
