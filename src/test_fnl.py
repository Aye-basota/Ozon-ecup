"""Анти-лукап и контроль сопоставимости арок для EXP-038 (FNL).

Правило проекта: каждая новая МЕТКА проверяется на то, что она читает только своё
окно `(T, T+h]`, и что горизонт легален для фолда (`T + h <= V`). Здесь же лежат
контроли, без которых эксперимент не имеет смысла: при `lam = 0` арка обязана
побитово повторять BASE, порядок батчей и инициализация энкодера обязаны не
зависеть от арки — иначе разница между арками смешается с шумом порядка данных и
разной инициализации (`exp_030`).

Запуск: python -m pytest src/test_fnl.py -q
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from src.config import CORRIDOR_END, TARGET_DAYS, VAL_FOLDS_S1
from src.data import load
from src.features import panel_users
from src.fnl import (ARMS, FUT_COLS, POISON_START, Head, aux_labels, aux_scales,
                     build_index_aux, fold_cutoffs_for_heads, future)
from src.seq import DEFAULT_CFG, day_index, fold_cutoffs, target_at, user_rows

V = dt.date(2025, 10, 16)
T_TRAIN = dt.date(2025, 9, 4)


@pytest.fixture(scope="module")
def raw():
    return load()


@pytest.fixture(scope="module")
def uid_all(raw):
    return np.sort(raw["user_id"].unique().to_numpy())


def _rows(T: dt.date, n: int) -> np.ndarray:
    return user_rows(panel_users(T, 1)["user_id"].to_numpy()[:n])


def _window_ref(raw, T: dt.date, h: int, cols: tuple[str, ...]) -> pl.DataFrame:
    """Независимый пересчёт агрегатов окна `(T, T+h]` прямо из лога."""
    a, b = T + dt.timedelta(days=1), T + dt.timedelta(days=h)
    return (raw.lazy()
            .filter((pl.col("event_date") >= a) & (pl.col("event_date") <= b))
            .group_by("user_id")
            .agg([pl.col(c).sum().alias(c) for c in cols])
            .collect())


# --------------------------------------------------------------- сырые счётчики
def test_future_array_matches_raw_counts(raw, uid_all):
    """Плотный массив счётчиков совпадает с логом на случайной выборке строк."""
    f = future()
    s = raw.sample(3000, seed=0)
    ui = np.searchsorted(uid_all, s["user_id"].to_numpy())
    di = np.array([day_index(d) for d in s["event_date"].to_list()])
    for k, col in enumerate(FUT_COLS):
        assert np.array_equal(f[ui, di, k].astype(np.int64), s[col].to_numpy())


def test_absent_day_has_zero_counts(raw, uid_all):
    """У дня, которого нет в логе, счётчики строго нулевые (нет строки != ноль сверху)."""
    f = future()
    u0 = int(uid_all[0])
    have = set(raw.filter(pl.col("user_id") == u0)["event_date"].to_list())
    miss = [d for d in (dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(409))
            if d not in have][:20]
    assert miss, "у пользователя нет ни одного пропущенного дня — тест бессмысленен"
    for d in miss:
        assert f[0, day_index(d)].sum() == 0


# ------------------------------------------------------------------- анти-лукап
@pytest.mark.parametrize("h", [7, 14, 30])
def test_labels_read_only_their_window(raw, uid_all, h):
    """Бинарные метки горизонта h равны независимому пересчёту по `(T, T+h]`."""
    rows = _rows(T_TRAIN, 5000)
    heads = (Head(f"any_cart_{h}", "bin", "to_cart", h),
             Head(f"any_search_{h}", "bin", "searches", h))
    got = aux_labels(T_TRAIN, rows, heads)
    ref = _window_ref(raw, T_TRAIN, h, ("to_cart", "searches"))
    j = (pl.DataFrame(dict(user_id=uid_all[rows]))
         .join(ref, on="user_id", how="left").fill_null(0))
    assert np.array_equal(got[:, 0], (j["to_cart"].to_numpy() > 0).astype(np.float32))
    assert np.array_equal(got[:, 1], (j["searches"].to_numpy() > 0).astype(np.float32))


def test_label_does_not_depend_on_the_cutoff_day(raw, uid_all):
    """Окно полуоткрыто слева: активность самого дня T в метку не входит.

    Проверяется прямо: доля пользователей, у которых событие есть в день T и
    больше нигде в окне, положительна, и у них метка равна нулю.
    """
    h = 7
    rows = _rows(T_TRAIN, 30000)
    got = aux_labels(T_TRAIN, rows, (Head("c", "bin", "to_cart", h),))[:, 0]
    f = future()
    d = day_index(T_TRAIN)
    on_T = f[rows, d, FUT_COLS.index("to_cart")] > 0
    in_win = f[rows, d + 1:d + 1 + h, FUT_COLS.index("to_cart")].sum(axis=1) > 0
    only_T = on_T & ~in_win
    assert only_T.sum() > 0, "нет пользователей с событием ровно в день T"
    assert not got[only_T].any()


def test_regression_label_is_log1p_of_the_window_sum(raw, uid_all):
    rows = _rows(T_TRAIN, 5000)
    got = aux_labels(T_TRAIN, rows, (Head("log_cart_30", "reg", "to_cart", 30),))
    ref = _window_ref(raw, T_TRAIN, 30, ("to_cart",))
    j = (pl.DataFrame(dict(user_id=uid_all[rows]))
         .join(ref, on="user_id", how="left").fill_null(0))
    assert np.allclose(got[:, 0], np.log1p(j["to_cart"].to_numpy()), atol=1e-6)


def test_buy30_label_equals_the_main_target_being_positive():
    """`buy30` совпадает с `y30 > 0` строка в строку — тот же фильтр, что у таргета."""
    rows = _rows(T_TRAIN, 20000)
    got = aux_labels(T_TRAIN, rows, (Head("buy30", "bin", "gmv", 30),))[:, 0]
    assert np.array_equal(got, (target_at(T_TRAIN, rows) > 0).astype(np.float32))


@pytest.mark.parametrize("arm", ["BUYCTRL", "CART", "FUNNEL"])
def test_every_train_cutoff_is_legal_for_every_horizon(arm):
    """`T + h <= V` и `T <= CORRIDOR_END` для всех фолдов, cutoff'ов и горизонтов."""
    heads = ARMS[arm]
    for Vf in VAL_FOLDS_S1:
        cuts = fold_cutoffs_for_heads(Vf, heads)
        assert cuts
        for T in cuts:
            assert T <= CORRIDOR_END, f"грязный cutoff {T}"
            for hd in heads:
                assert T + dt.timedelta(days=hd.h) <= Vf, f"{hd.name}: {T}+{hd.h} > {Vf}"


@pytest.mark.parametrize("arm", ["BUYCTRL", "CART", "FUNNEL"])
def test_aux_horizons_never_exceed_the_main_target_horizon(arm):
    """h <= 30, поэтому правило `T+30<=V` уже достаточно и сетка не меняется."""
    heads = ARMS[arm]
    assert all(hd.h <= TARGET_DAYS for hd in heads)
    for Vf in VAL_FOLDS_S1:
        assert fold_cutoffs_for_heads(Vf, heads) == fold_cutoffs(Vf)


@pytest.mark.parametrize("arm", ["BUYCTRL", "CART", "FUNNEL"])
def test_validation_labels_do_not_touch_the_poisoned_window(arm):
    """Окно `(V, V+h]` не пересекает гарантированную область с 2025-11-16."""
    for Vf in VAL_FOLDS_S1:
        for hd in ARMS[arm]:
            assert Vf + dt.timedelta(days=hd.h) < POISON_START, f"{hd.name} на {Vf}"


def test_no_supervision_uses_a_dirty_late_cutoff():
    """Ни один cutoff супервизии не выходит за 2025-10-16 ни на день."""
    for arm in ARMS:
        for Vf in VAL_FOLDS_S1:
            assert max(fold_cutoffs_for_heads(Vf, ARMS[arm])) <= CORRIDOR_END


# ------------------------------------------------------------------- шкалы и индекс
def test_scales_are_the_constant_predictor_loss():
    A = np.zeros((1000, 2), np.float32)
    A[:300, 0] = 1.0
    A[:, 1] = np.arange(1000, dtype=np.float32) / 100.0
    heads = (Head("b", "bin", "to_cart", 7), Head("r", "reg", "to_cart", 30))
    s, b = aux_scales(A, heads)
    p = 0.3
    assert abs(s[0] - (-(p * np.log(p) + (1 - p) * np.log(1 - p)))) < 1e-6
    assert abs(s[1] - A[:, 1].var()) < 1e-3
    assert abs(b[0] - np.log(p / (1 - p))) < 1e-5
    assert abs(b[1] - A[:, 1].mean()) < 1e-3


def test_index_labels_line_up_with_rows_and_cutoffs():
    cuts = [dt.date(2025, 8, 7), dt.date(2025, 8, 14)]
    heads = ARMS["CART"]
    ci, ri, zy, A = build_index_aux(cuts, 1, heads)
    assert A.shape == (len(ri), len(heads)) and A.dtype == np.float32
    for k in (0, 1):
        m = ci == k
        assert np.array_equal(A[m], aux_labels(cuts[k], ri[m], heads))


# ---------------------------------------------- модель: BASE обязан не измениться
def _cfg(**kw):
    c = dict(DEFAULT_CFG, z0=2.6, seed=42)
    c.update(kw)
    return c


@pytest.mark.parametrize("arm", ["BASE", "BUYCTRL", "CART", "FUNNEL"])
def test_encoder_init_is_identical_across_arms(arm):
    """Веса энкодера и главной головы при одном сиде одинаковы у всех арок."""
    import torch
    from src.fnl import build_net
    from src.seq import build_model
    torch.manual_seed(42)
    ref = {k: v.clone() for k, v in build_model(_cfg()).state_dict().items()}
    torch.manual_seed(42)
    got = build_net(_cfg(), ARMS[arm], None).tcn.state_dict()
    assert set(got) == set(ref)
    for k in ref:
        assert torch.equal(got[k], ref[k]), f"{k} разошёлся у арки {arm}"


@pytest.mark.parametrize("arm", ["BUYCTRL", "CART", "FUNNEL"])
def test_aux_head_starts_at_the_constant_predictor(arm):
    import torch
    from src.fnl import build_net
    b = np.arange(len(ARMS[arm]), dtype=np.float32) - 1.0
    net = build_net(_cfg(), ARMS[arm], b)
    assert int(torch.count_nonzero(net.aux.weight)) == 0
    assert np.allclose(net.aux.bias.detach().numpy(), b, atol=1e-6)


@pytest.mark.parametrize("arm", ["BASE", "BUYCTRL", "CART", "FUNNEL"])
def test_forward_returns_the_same_z_for_every_arm(arm):
    """Прогноз на инференсе — только главная голова, и он не зависит от арки."""
    import torch
    from src.fnl import build_net
    from src.seq import N_CH, SEQ_L
    torch.manual_seed(42)
    a = build_net(_cfg(hidden=16, blocks=2), ARMS[arm],
                  np.zeros(len(ARMS[arm]), np.float32)).eval()
    torch.manual_seed(42)
    b = build_net(_cfg(hidden=16, blocks=2), ARMS["BASE"], None).eval()
    x = torch.randn(4, N_CH, SEQ_L, generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
        assert torch.equal(a(x), b(x))


# ------------------------------------------------ батчи: порядок не зависит от арки
def _batcher(arm, seed=42):
    from src.fnl import AuxBatcher
    cuts = [dt.date(2025, 8, 7), dt.date(2025, 8, 14)]
    heads = ARMS[arm]
    ci, ri, zy, A = build_index_aux(cuts, 1, heads)
    return AuxBatcher(cuts, ci, ri, zy, A, batch=2048, chunk=512,
                      rng=np.random.default_rng(seed), workers=1,
                      depth=dict(p=0.5, grid=(90, 150, 289)))


def test_batch_order_does_not_depend_on_the_arm():
    """`_plan()` при одном сиде одинаков у всех арок — иначе арки несравнимы."""
    plans = {}
    for arm in ("BASE", "BUYCTRL", "CART", "FUNNEL"):
        plans[arm] = [[(k, idx.tolist()) for k, idx in g] for g in _batcher(arm)._plan()]
    assert plans["BASE"] == plans["BUYCTRL"] == plans["CART"] == plans["FUNNEL"]


def test_batch_labels_match_the_rows_of_the_batch():
    b = _batcher("FUNNEL")
    group = b._plan()[0]
    x, y, a = b._make(group, seed=1)
    sel = np.concatenate([idx for _, idx in group])
    assert len(x) == len(y) == len(a) == len(sel)
    assert np.array_equal(a, b.A[sel])
    off = 0
    for k, idx in group:
        assert np.array_equal(a[off:off + len(idx)],
                              aux_labels(b.cuts[k], b.ri[idx], ARMS["FUNNEL"]))
        off += len(idx)


def test_realized_batch_order_is_a_race_above_one_worker():
    """Порядок ПОТРЕБЛЕНИЯ батчей не воспроизводим при `workers > 1`.

    `_plan()` детерминирован и от арки не зависит (тест выше), содержимое чанков
    тоже: сиды аугментации раздаются в главном потоке. Но `Batcher.__iter__`
    гонит `workers` потоков через общую очередь и отдаёт то, что положили первым,
    поэтому ПОРЯДОК шагов SGD меняется от прогона к прогону при одном сиде.

    Это фиксируется тестом, а не комментарием, потому что из этого следует
    практический вывод для всей линии SEQ: пара «BASE и вариант», посчитанная
    РАЗНЫМИ ПРОЦЕССАМИ, не является строго парной, и её дельту нужно мерить
    против цены самого прогона (`FNL-BASER2`), а не против нуля.

    Панель здесь не читается: `_make` подменён на метку чанка.
    """
    from src.seq import Batcher

    class OrderProbe(Batcher):
        def _make(self, group, seed=None):
            return tuple((int(k), int(idx[0])) for k, idx in group)

    cuts = [dt.date(2025, 8, 7), dt.date(2025, 8, 14), dt.date(2025, 8, 21)]
    n = 60000
    ci = np.repeat(np.arange(len(cuts), dtype=np.int16), n)
    ri = np.tile(np.arange(n, dtype=np.int32), len(cuts))
    y = np.zeros(len(ci), np.float32)

    def orders(workers, reps=4):
        out = []
        for _ in range(reps):
            b = OrderProbe(cuts, ci, ri, y, batch=1024, chunk=256,
                           rng=np.random.default_rng(42), workers=workers)
            out.append(list(b))
        return out

    one = orders(1)
    assert all(o == one[0] for o in one), "при одном потоке порядок обязан быть строгим"
    many = orders(3)
    assert all(sorted(map(str, o)) == sorted(map(str, many[0])) for o in many), \
        "набор батчей обязан совпадать при любом числе потоков"
    assert not all(o == many[0] for o in many), \
        "ожидалась гонка порядка при workers=3; если её нет, вывод о шуме нужно пересмотреть"


def test_inputs_are_bitwise_the_base_inputs():
    """Вход сети не зависит от арки ни на бит: aux-метки в него не попадают."""
    xs = {}
    for arm in ("BASE", "FUNNEL"):
        b = _batcher(arm)
        xs[arm] = b._make(b._plan()[0], seed=1)[0]
    assert np.array_equal(xs["BASE"], xs["FUNNEL"])


def test_aux_loss_is_one_at_the_constant_predictor():
    """При bias константного предсказателя нормированная aux-потеря равна 1."""
    import torch
    from src.fnl import aux_loss
    heads = ARMS["FUNNEL"]
    rng = np.random.default_rng(0)
    A = np.zeros((20000, len(heads)), np.float32)
    for j, hd in enumerate(heads):
        A[:, j] = ((rng.random(20000) < 0.6).astype(np.float32) if hd.kind == "bin"
                   else rng.normal(2.0, 1.3, 20000).astype(np.float32))
    s, b = aux_scales(A, heads)
    logits = torch.from_numpy(np.tile(b, (20000, 1)))
    v = aux_loss(logits, torch.from_numpy(A), [h.kind for h in heads],
                 torch.from_numpy(s))
    assert abs(float(v) - 1.0) < 0.01


# ------------------------------------------------------- главный контроль механики
@pytest.mark.slow
def test_lambda_zero_is_bitwise_the_base_run():
    """FUNNEL при lam=0 обязан дать те же z, что BASE: контроль всей механики."""
    from src.fnl import fit_model
    from src.seq import predict
    cuts = [dt.date(2025, 8, 7), dt.date(2025, 8, 14)]
    cfg = _cfg(epochs=1, batch=256, chunk=256, workers=1, hidden=16, blocks=3,
               depth_aug=0.5)
    rows = _rows(V, 512)
    out = {}
    for arm in ("BASE", "FUNNEL"):
        m, dev, c, _ = fit_model(cuts, dict(cfg), ARMS[arm], 0.0)
        out[arm] = predict(m, V, rows, c, dev)
    assert np.array_equal(out["BASE"], out["FUNNEL"])


# ------------------------------------------------------------------- реестр арок
def test_arms_are_exactly_the_four_specified():
    """Список голов зафиксирован спекой: ни одной лишней головы «для количества»."""
    assert set(ARMS) == {"BASE", "BUYCTRL", "CART", "FUNNEL"}
    assert [h.name for h in ARMS["BASE"]] == []
    assert [h.name for h in ARMS["BUYCTRL"]] == ["buy30"]
    assert [h.name for h in ARMS["CART"]] == [
        "any_cart_7", "any_cart_14", "any_cart_30", "log_cart_30"]
    assert [h.name for h in ARMS["FUNNEL"]] == [
        "any_cart_7", "any_cart_14", "any_cart_30", "log_cart_30",
        "any_search_7", "any_search_14", "any_search_30", "log_search_30"]


def test_cli_exposes_build_smoke_fold():
    import inspect
    from src.fnl import main
    src = inspect.getsource(main)
    for c in ("build", "smoke", "fold"):
        assert f'"{c}"' in src
