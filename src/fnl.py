"""EXP-038 (FNL) — future-funnel supervision для энкодера SEQ-D3A.

Проверяемое утверждение. `exp_024` закрыл БОГАТУЮ разметку самой покупки
(hazard до покупки, распределение счёта покупательных дней, величина): три
независимые параметризации активности дали AUC 0.8467 ± 0.0002, то есть
перечитывание одного и того же purchase-сигнала, и `FULL` = 1.75234 против
контроля `SELF` = 1.75235. Но сырые данные содержат более РАННИЕ стадии
воронки `Search -> Cart -> Order`, и они размечают ДРУГОЕ событие: не «купил»,
а «искал» и «положил в корзину». Вопрос эксперимента ровно один:

    заставляет ли будущая поисковая/корзинная активность энкодер выучить
    состояние НАМЕРЕНИЯ, которого нет в `z30`/`buy30`?

Четыре арки с ОДНИМ энкодером, ОДНИМ порядком батчей и ОДНИМ бюджетом:

    BASE     L = MSE(z30)
    BUYCTRL  L = MSE(z30) + lam * s_z * BCE(buy30)/s_buy
    CART     L = MSE(z30) + lam * s_z * mean(cart-головы)
    FUNNEL   L = MSE(z30) + lam * s_z * mean(cart + search-головы)

`BUYCTRL` играет ту же роль, что контроль `SELF` в `exp_024`: вторая голова
есть, нового ИСТОЧНИКА информации нет. Поэтому главное сравнение эксперимента —
`FUNNEL - BUYCTRL`, а не `FUNNEL - BASE`: иначе польза multi-task регуляризации
неотделима от пользы нового сигнала. Второе сравнение — `CART` против `FUNNEL`:
добавляет ли Search что-то сверх Cart.

## Метки и лукап

Все метки читаются строго из полуинтервала `(T, T+h]` — тем же фильтром, что
`features.target`. Горизонты 7/14/30, то есть `h <= TARGET_DAYS`, поэтому
проектное правило фолда `T + 30 <= V` УЖЕ гарантирует `T + h <= V`, и сетка
обучающих cutoff'ов не меняется ни на один cutoff (это утверждение проверяется
тестом, а не принимается на веру). Ни один cutoff позже `CORRIDOR_END`
(2025-10-16) не используется: отравление панели (`exp_028`, `STATE.md`) закрывает
эту область для ЛЮБОЙ супервизии, а не только для таргета. Валидационные метки
`(V, V+h]` при h <= 30 тоже чисты: `2025-10-16 + 30 = 2025-11-15`, ровно на день
раньше начала гарантированного окна.

## Источник счётчиков

`seq_panel_v1.npy` хранит `log1p` поканально, а метке `log1p(sum)` нужна сырая
сумма: сумма логарифмов — не логарифм суммы. Поэтому рядом с `seq_gmv_v1.npy`
строится `seq_fut_v1.npy` `(250 000, 409, 2)` uint16 с сырыми дневными
`searches` и `to_cart` (максимумы в данных 630 и 1535, в uint16 влезают с
запасом). Бинарные метки «было ли событие» берутся оттуда же — сумма окна > 0,
чтобы у обоих видов меток был один источник и один код.

## Что здесь НЕ меняется

Энкодер, depth policy, входные каналы, сетка cutoff'ов, панели, рецепт
оптимизатора и основной таргет `z30` — ровно те же, что у подтверждённого
`SEQ-D3A` (`exp_030c`). Меняется РОВНО supervision. `src/seq.py` не
редактируется: он боевой код `SEQ-AVG3`, этот модуль его только импортирует.
`torch.compile` не поддержан (локальная машина Windows его и не тянет).

Запуск:
  python -m src.fnl build                                       # массив счётчиков, один раз
  python -m src.fnl smoke                                       # быстрая проверка
  python -m src.fnl fold --val 2025-10-16 --arm FUNNEL --lam 0.3 --curve
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import time
from dataclasses import dataclass

import numpy as np

from src.config import (ARTIFACTS, CORRIDOR_END, DATA_PROCESSED, DATA_START, TARGET_DAYS,
                        VAL_FOLDS_S1)
from src.data import load
from src.features import panel_users
from src.report import evaluate, format_report
from src.seq import (DEFAULT_CFG, N_CH_STORED, N_DAYS, N_USERS, Batcher, aug_spec,
                     build_model, day_index, depth_spec, fold_cutoffs, gather, log, panel,
                     predict, target_at, user_rows)
from src.tracking import save_oof
from src.validation import bias_z, calibrate, rmsle_z

FUT_COLS = ("searches", "to_cart")
FUT_NPY = DATA_PROCESSED / "seq_fut_v1.npy"
# Начало гарантированного окна активности тестовой панели (eda §3.2, exp_028):
# любая метка, чьё окно его задевает, отравлена отбором панели.
POISON_START = dt.date(2025, 11, 16)


# =============================================================== сырые счётчики будущего
def build_future(force: bool = False) -> None:
    """Плотный (пользователи x дни x 2) uint16 с сырыми `searches` и `to_cart`.

    Строится тем же одним проходом по логу, что `seq_gmv_v1.npy` в
    `src.seq.build_panel`: `flat = user_index * N_DAYS + day_index`. Обрезки по
    cutoff'у здесь нет и быть не может — это сырой лог; окна режутся в
    `aux_labels`, тем же полуинтервалом, что `features.target`.
    """
    if FUT_NPY.exists() and not force:
        log(f"массив будущих счётчиков уже собран: {FUT_NPY.name}")
        return
    df = load()
    uid = np.sort(df["user_id"].unique().to_numpy()).astype(np.int64)
    assert len(uid) == N_USERS, f"пользователей {len(uid)}, ожидалось {N_USERS}"
    ui = np.searchsorted(uid, df["user_id"].to_numpy())
    di = (df["event_date"].to_numpy() - np.datetime64(DATA_START.isoformat())
          ).astype("timedelta64[D]").astype(np.int32)
    assert di.min() >= 0 and di.max() < N_DAYS
    flat = ui.astype(np.int64) * N_DAYS + di
    del ui, di
    out = np.zeros((N_USERS * N_DAYS, len(FUT_COLS)), dtype=np.uint16)
    for k, c in enumerate(FUT_COLS):
        v = df[c].to_numpy()
        assert v.min() >= 0 and v.max() < 65536, f"{c} не влезает в uint16: max {v.max()}"
        out[flat, k] = v.astype(np.uint16)
        del v
    np.save(FUT_NPY, out.reshape(N_USERS, N_DAYS, len(FUT_COLS)))
    log(f"записано: {FUT_NPY.name} ({FUT_NPY.stat().st_size / 1e6:.0f} МБ)")


_F: dict = {}


def future() -> np.ndarray:
    """(N_USERS, N_DAYS, 2) uint16 — сырые дневные счётчики, кэш в памяти процесса."""
    if "fut" not in _F:
        assert FUT_NPY.exists(), "сначала: python -m src.fnl build"
        _F["fut"] = np.load(FUT_NPY)
    return _F["fut"]


# =============================================================================== головы
@dataclass(frozen=True)
class Head:
    """Одна вспомогательная голова: что размечаем, каким лоссом, на каком горизонте."""
    name: str
    kind: str          # "bin" -> BCE, "reg" -> MSE
    src: str           # "searches" | "to_cart" | "gmv"
    h: int             # горизонт окна (T, T+h]


def _cart(h: int) -> Head:
    return Head(f"any_cart_{h}", "bin", "to_cart", h)


def _search(h: int) -> Head:
    return Head(f"any_search_{h}", "bin", "searches", h)


CART_HEADS = (_cart(7), _cart(14), _cart(30),
              Head("log_cart_30", "reg", "to_cart", 30))
SEARCH_HEADS = (_search(7), _search(14), _search(30),
                Head("log_search_30", "reg", "searches", 30))

# Ровно четыре арки спеки. Ни одной головы «для количества»: эксперимент проверяет
# ИСТОЧНИК разметки, и лишняя голова смешала бы его с силой регуляризации.
ARMS: dict[str, tuple[Head, ...]] = {
    "BASE": (),
    "BUYCTRL": (Head("buy30", "bin", "gmv", TARGET_DAYS),),
    "CART": CART_HEADS,
    "FUNNEL": CART_HEADS + SEARCH_HEADS,
}


def aux_labels(T: dt.date, rows: np.ndarray, heads: tuple[Head, ...]) -> np.ndarray:
    """(n, M) float32 — метки строго из окна `(T, T+h]` для каждой головы.

    `gmv` читается из `seq_gmv_v1.npy` — того же массива, из которого считается
    боевой таргет, поэтому `buy30` совпадает с `y30 > 0` строка в строку (это
    проверяется тестом). `searches`/`to_cart` — из `seq_fut_v1.npy`. Окно
    полуоткрыто слева: день `T` в него не входит, ровно как в `features.target`.
    """
    if not heads:
        return np.zeros((len(rows), 0), np.float32)
    _, g, _, _ = panel()
    d = day_index(T)
    out = np.empty((len(rows), len(heads)), np.float32)
    f = None
    for j, hd in enumerate(heads):
        assert 1 <= hd.h <= TARGET_DAYS, f"горизонт {hd.h} вне 1..{TARGET_DAYS}"
        assert d + hd.h < N_DAYS, f"окно {T}+{hd.h} выходит за данные"
        if hd.src == "gmv":
            s = g[rows, d + 1:d + 1 + hd.h].sum(axis=1)
        else:
            if f is None:
                f = future()
            k = FUT_COLS.index(hd.src)
            s = f[rows, d + 1:d + 1 + hd.h, k].sum(axis=1, dtype=np.int64)
        out[:, j] = (s > 0).astype(np.float32) if hd.kind == "bin" else np.log1p(s)
    return out


def aux_scales(A: np.ndarray, heads: tuple[Head, ...]):
    """(s, b): потеря КОНСТАНТНОГО предсказателя и её оптимальный параметр.

    `s_m` уравнивает шкалы BCE и MSE и делает нормированную потерю головы равной
    ровно 1.0 в начале обучения, поэтому `lam` читается прямо, а не через
    случайное соотношение масштабов. `b_m` — начальный bias головы (logit доли
    для бинарной, среднее для регрессии): тот же приём, что `z0` у главной
    головы, сеть стартует из константного предсказания.
    """
    s = np.empty(len(heads), np.float32)
    b = np.empty(len(heads), np.float32)
    for j, hd in enumerate(heads):
        v = A[:, j].astype(np.float64)
        if hd.kind == "bin":
            p = float(np.clip(v.mean(), 1e-6, 1 - 1e-6))
            s[j] = -(p * math.log(p) + (1 - p) * math.log(1 - p))
            b[j] = math.log(p / (1 - p))
        else:
            s[j] = max(float(v.var()), 1e-6)
            b[j] = float(v.mean())
    return s, b


def fold_cutoffs_for_heads(V: dt.date, heads: tuple[Head, ...]) -> list[dt.date]:
    """Обучающие cutoff'ы фолда, легальные ДЛЯ ВСЕХ горизонтов арки.

    Правило проекта `T + TARGET_DAYS <= V` обобщено до `T + max(h) <= V`, как в
    `exp_024`. Поскольку все горизонты арок `h <= TARGET_DAYS`, множество
    совпадает с `seq.fold_cutoffs(V)`; функция существует затем, чтобы это было
    ПРОВЕРЯЕМЫМ утверждением, а не соглашением.
    """
    hmax = max([TARGET_DAYS] + [hd.h for hd in heads])
    out = [T for T in fold_cutoffs(V) if T + dt.timedelta(days=hmax) <= V]
    assert out, f"пустая сетка обучающих cutoff'ов для фолда {V}"
    assert all(T <= CORRIDOR_END for T in out), "грязный cutoff в сетке обучения"
    return out


def build_index_aux(cuts: list[dt.date], blocks: int, heads: tuple[Head, ...]):
    """(ci, ri, zy, A) — то же, что `seq.build_index`, плюс матрица aux-меток."""
    ci, ri, ys, aa = [], [], [], []
    for k, T in enumerate(cuts):
        u = panel_users(T, blocks)["user_id"].to_numpy()
        r = user_rows(u)
        ci.append(np.full(len(r), k, np.int16))
        ri.append(r)
        ys.append(target_at(T, r))
        aa.append(aux_labels(T, r, heads))
    n = sum(len(x) for x in ri)
    return (np.concatenate(ci), np.concatenate(ri),
            np.log1p(np.concatenate(ys)).astype(np.float32),
            np.concatenate(aa) if heads else np.zeros((n, 0), np.float32))


# =============================================================================== модель
def build_net(cfg: dict, heads: tuple[Head, ...], aux_bias):
    """TCN проекта + линейный aux-зонд на том же pooled-векторе.

    Порядок создания модулей принципиален: сначала `seq.build_model` (он вытягивает
    из глобального генератора ровно те же случайные числа, что и у BASE), и только
    ПОТОМ aux-голова. Поэтому энкодер и главная голова при одном сиде побитово
    совпадают у всех четырёх арок, и разница между арками — это разница loss'а, а
    не инициализации.

    Голова — ОДИН `Linear(3H -> M)` без скрытого слоя: эксперимент проверяет, меняет
    ли разметка ПРЕДСТАВЛЕНИЕ, а ёмкая голова решала бы задачу сама, не трогая
    энкодер. Цена — +193 / +772 / +1544 параметра против 245 633 у энкодера.
    Веса нулевые, bias — константный предсказатель: на шаге 0 нормированная
    aux-потеря равна ровно 1.0, а её градиент в энкодер равен ровно нулю.
    """
    import torch
    from torch import nn

    tcn = build_model(cfg)

    class FunnelNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.tcn = tcn
            self.aux = nn.Linear(3 * cfg["hidden"], len(heads)) if heads else None
            if self.aux is not None:
                nn.init.zeros_(self.aux.weight)
                b = (np.zeros(len(heads), np.float32) if aux_bias is None
                     else np.asarray(aux_bias, np.float32))
                with torch.no_grad():
                    self.aux.bias.copy_(torch.from_numpy(b))

        def _pooled(self, x):
            h = self.tcn.encode(x)
            return torch.cat([h[:, -1], h.mean(1), h.amax(1)], dim=1)

        def forward(self, x):
            """Только главная голова — `seq.predict` работает без единой правки."""
            return self.tcn.head(self._pooled(x)).squeeze(1)

        def forward_all(self, x):
            p = self._pooled(x)
            z = self.tcn.head(p).squeeze(1)
            return z, (None if self.aux is None else self.aux(p))

    return FunnelNet()


# =============================================================================== батчи
class AuxBatcher(Batcher):
    """`seq.Batcher`, дополнительно отдающий матрицу aux-меток батча.

    План эпохи (`_plan`) наследуется без изменений и зависит ТОЛЬКО от `self.rng`,
    которого aux-метки не касаются. Поэтому у всех четырёх арок при одном сиде
    порядок примеров совпадает, и разница между арками — это разница loss'а, а не
    порядка данных. Ровно тот же приём, которым `exp_030` удержал BASE контролем.
    """

    def __init__(self, cuts, ci, ri, y, A, batch, chunk, rng, workers=2, aug=None,
                 aug_seed=None, depth=None):
        super().__init__(cuts, ci, ri, y, batch, chunk, rng, workers, aug, aug_seed, depth)
        self.A = A

    def _make(self, group, seed=None):
        x, yb = super()._make(group, seed)
        sel = np.concatenate([idx for _, idx in group])
        return x, yb, self.A[sel]


# =============================================================================== loss
def aux_loss(logits, targets, kinds, scales):
    """mean_m( L_m / s_m ) — равный БЮДЖЕТ арок при любом числе голов.

    Деление на `s_m` уравнивает шкалы BCE и MSE. Усреднение по головам, а не
    суммирование, уравнивает арки: у BUYCTRL одна голова, у FUNNEL восемь, и при
    суммировании FUNNEL получал бы восьмикратный aux-градиент — тогда сравнение
    мерило бы силу регуляризации, а не источник сигнала.
    """
    import torch
    from torch.nn import functional as F
    terms = []
    for j, kind in enumerate(kinds):
        o, t = logits[:, j], targets[:, j]
        li = (F.binary_cross_entropy_with_logits(o, t) if kind == "bin"
              else F.mse_loss(o, t))
        terms.append(li / scales[j])
    return torch.stack(terms).mean()


# =============================================================================== обучение
def _device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fit_model(cuts, cfg: dict, heads: tuple[Head, ...], lam: float, eval_fn=None):
    """Рецепт `SEQ-D3A` без изменений плюс один слагаемый член в loss.

        L = MSE(z30) + lam * s_z * mean_m( L_m / s_m ),   s_z = Var(z30) на обучении

    Главный член НЕ нормируется: его градиент обязан остаться таким же, как у
    BASE, иначе `lam = 0` перестал бы ТОЧНО воспроизводить базу (это проверяется
    тестом). Множитель `s_z` переводит нормированный aux в шкалу главного члена,
    поэтому `lam` читается прямо: `lam = 0.3` — вспомогательная задача весит 30%
    от главной в начале обучения, потому что при инициализации bias главной
    головы средним `z` выполняется `MSE(z30) = Var(z30) = s_z`.

    `clip_grad_norm_` остаётся ОДНИМ глобальным клипом по всем параметрам: рецепт
    оптимизатора менять запрещено. Средняя норма градиента ДО клиппинга пишется в
    лог поэпохно, чтобы в карточке можно было сказать, стал ли клип различать арки.
    """
    import torch

    assert not cfg.get("compile"), "torch.compile в fnl не поддержан"
    dev = _device()
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ci, ri, zy, A = build_index_aux(cuts, 1, heads)          # train-панель 1-блочная
    s_np, b_np = (aux_scales(A, heads) if heads
                  else (np.zeros(0, np.float32), np.zeros(0, np.float32)))
    s_z = float(zy.var())
    log(f"{len(cuts)} обучающих cutoff'ов {cuts[0]}..{cuts[-1]}, {len(zy):,} примеров, "
        f"mean z = {zy.mean():.4f}, s_z = Var(z30) = {s_z:.4f}")
    for j, hd in enumerate(heads):
        m = float(A[:, j].mean())
        log(f"  голова {hd.name:<15} kind={hd.kind} h={hd.h:<3} "
            f"{'доля' if hd.kind == 'bin' else 'среднее'} {m:.4f}  s = {s_np[j]:.4f}")

    cfg = dict(cfg, z0=float(zy.mean()))
    model = build_net(cfg, heads, b_np).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    log(f"  модель: hidden={cfg['hidden']} blocks={cfg['blocks']} параметров {n_par:,}, "
        f"aux-голов {len(heads)}, lam = {lam}")

    decay = [p for n, p in model.named_parameters() if p.dim() > 1]
    nodecay = [p for n, p in model.named_parameters() if p.dim() <= 1]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=cfg["wd"]),
                             dict(params=nodecay, weight_decay=0.0)], lr=cfg["lr"],
                            betas=(0.9, 0.98))
    rng = np.random.default_rng(cfg["seed"])
    bat = AuxBatcher(cuts, ci, ri, zy, A, cfg["batch"], cfg["chunk"], rng, cfg["workers"],
                     aug=aug_spec(cfg), aug_seed=[cfg["seed"], 0xA7A1], depth=depth_spec(cfg))
    if bat.depth["p"] > 0:
        log(f"  depth curriculum: p={bat.depth['p']}, сетка {list(bat.depth['grid'])}")
    total = bat.n_batches() * cfg["epochs"]
    log(f"  шагов всего {total:,} ({bat.n_batches():,} на эпоху), batch={cfg['batch']}")

    sc = torch.from_numpy(panel()[3]).to(dev).view(1, N_CH_STORED, 1)
    kinds = [hd.kind for hd in heads]
    s_t = torch.from_numpy(s_np).to(dev)
    use_aux = bool(heads) and bool(lam)
    step, hist = 0, []
    for ep in range(cfg["epochs"]):
        t_ep, run, run_a, run_g, seen, nstep = time.time(), None, None, 0.0, 0, 0
        for x, yb, ab in bat:
            lr = cfg["lr"] * (min(1.0, (step + 1) / cfg["warmup"])
                              * 0.5 * (1 + math.cos(math.pi * min(1.0, step / total))))
            for gp in opt.param_groups:
                gp["lr"] = lr
            n = len(yb)
            t = torch.from_numpy(x).to(dev).permute(0, 2, 1).contiguous().float()
            t[:, :N_CH_STORED] *= sc
            tgt = torch.from_numpy(yb).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                z, logits = model.forward_all(t)
                main = torch.nn.functional.mse_loss(z, tgt)
                if use_aux:
                    at = torch.from_numpy(ab).to(dev)
                    aux = aux_loss(logits.float(), at, kinds, s_t)
                    loss = main + lam * s_z * aux
                else:
                    aux = None
                    loss = main
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run = main.detach() * n if run is None else run + main.detach() * n
            if aux is not None:
                run_a = aux.detach() * n if run_a is None else run_a + aux.detach() * n
            run_g += float(gn)
            nstep += 1
            seen += n
            step += 1
        rec = dict(epoch=ep + 1, train_mse=float(run) / seen,
                   train_aux=(float(run_a) / seen if run_a is not None else None),
                   grad_norm=run_g / max(nstep, 1))
        msg = (f"  эпоха {ep + 1}/{cfg['epochs']}: train MSE {rec['train_mse']:.5f}"
               + (f" aux {rec['train_aux']:.5f}" if rec["train_aux"] is not None else "")
               + f" |g| {rec['grad_norm']:.3f} [{time.time() - t_ep:.0f}s]")
        if eval_fn is not None:
            r = eval_fn(model, dev, ep, cfg)
            if r:
                msg += "  " + r.pop("_msg", "")
                rec.update(r)
        hist.append(rec)
        log(msg)
    return model, dev, cfg, hist


def predict_aux(model, T: dt.date, rows: np.ndarray, cfg: dict, dev) -> np.ndarray:
    """(n, M) логиты вспомогательных голов — ТОЛЬКО для диагностики.

    Боевой прогноз всегда идёт через главную голову (`seq.predict`); ни одна
    aux-голова не участвует ни в постобработке, ни в сабмите.
    """
    import torch
    if getattr(model, "aux", None) is None:
        return np.zeros((len(rows), 0), np.float32)
    model.eval()
    outs = []
    sc = torch.from_numpy(panel()[3]).to(dev).view(1, N_CH_STORED, 1)
    B = cfg["batch"]
    with torch.no_grad():
        for i in range(0, len(rows), B):
            x = gather(T, rows[i:i + B])
            t = torch.from_numpy(x).to(dev).permute(0, 2, 1).contiguous().float()
            t[:, :N_CH_STORED] *= sc
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                _, lg = model.forward_all(t)
            outs.append(lg.float().cpu().numpy())
    model.train()
    return np.concatenate(outs)


# =============================================================================== фолд
def train_fold(V: dt.date, cfg: dict, heads: tuple[Head, ...], lam: float,
               curve: bool = False, ckpt: str | None = None,
               n_cutoffs: int | None = None, val_frac: float = 1.0) -> dict:
    """Один фолд одной арки. Всё, что нужно диагностике, возвращается одним словарём."""
    cuts = fold_cutoffs_for_heads(V, heads)
    if n_cutoffs:
        cuts = cuts[-n_cutoffs:]
    uv = panel_users(V, 3)["user_id"].to_numpy()             # val-панель 3-блочная
    if val_frac < 1.0:
        uv = uv[::int(round(1 / val_frac))]
    rv = user_rows(uv)
    yv = target_at(V, rv)
    for hd in heads:
        assert V + dt.timedelta(days=hd.h) < POISON_START, (
            f"валидационная метка {hd.name} на фолде {V} задевает отравленное окно")
    av = aux_labels(V, rv, heads)
    log(f"фолд {V}: val-панель {len(uv):,} пользователей, доля y>0 = {(yv > 0).mean():.4f}")

    def ev(model, dev, ep, c):
        if not (curve or ep == c["epochs"] - 1):
            return None
        z = np.maximum(predict(model, V, rv, c, dev), 0.0)
        o, sc_o = calibrate(yv, z)
        return dict(rmsle=rmsle_z(yv, z), rmsle_cal=sc_o, bias=bias_z(yv, z), offset=o,
                    _msg=f"val RMSLE {rmsle_z(yv, z):.5f} -> калибр. {sc_o:.5f} "
                         f"(сдвиг {o:+.3f})")

    model, dev, c, hist = fit_model(cuts, cfg, heads, lam, ev)
    z = np.maximum(predict(model, V, rv, c, dev), 0.0)
    ap = predict_aux(model, V, rv, c, dev)
    if ckpt:
        import torch
        torch.save(dict(state=model.state_dict(), cfg={k: v for k, v in c.items()},
                        val=V.isoformat(), heads=[hd.name for hd in heads], lam=lam),
                   ARTIFACTS / f"model_{ckpt}.pt")
        log(f"веса сохранены: artifacts/model_{ckpt}.pt")
    return dict(user_id=uv, z=z, y=yv, aux_pred=ap, aux_true=av, hist=hist,
                head_names=[hd.name for hd in heads],
                head_kinds=[hd.kind for hd in heads])


# =============================================================================== CLI
def cmd_fold(a):
    cfg = dict(DEFAULT_CFG)
    for k in ("hidden", "blocks", "kernel", "dropout", "batch", "chunk", "lr", "wd",
              "epochs", "seed", "workers", "depth_aug", "depth_grid"):
        v = getattr(a, k, None)
        if v is not None:
            cfg[k] = v
    heads = ARMS[a.arm]
    lam = 0.0 if a.arm == "BASE" else float(a.lam)
    V = dt.date.fromisoformat(a.val)
    exp = a.exp or f"FNL-{a.arm}-L{int(round(lam * 100)):02d}-S{cfg['seed']}"
    part = f"{exp}-V{V.strftime('%m%d')}"
    full = a.val_frac >= 1.0 and not a.n_cutoffs
    r = train_fold(V, cfg, heads, lam, curve=a.curve, n_cutoffs=a.n_cutoffs,
                   val_frac=a.val_frac, ckpt=part if (full and not a.no_ckpt) else None)
    if full:
        save_oof(part, r["user_id"], [V.isoformat()] * len(r["user_id"]), r["z"], r["y"])
        np.savez_compressed(
            ARTIFACTS / f"fnl_{part}.npz", user_id=r["user_id"],
            y=r["y"].astype(np.float32), z=r["z"].astype(np.float32),
            aux_pred=r["aux_pred"], aux_true=r["aux_true"],
            head_names=np.array(r["head_names"], dtype="U20"),
            head_kinds=np.array(r["head_kinds"], dtype="U4"))
        log(f"OOF: artifacts/oof_{part}.npz, диагностика: artifacts/fnl_{part}.npz")
    rep = evaluate(r["y"], r["z"], np.array([V.isoformat()] * len(r["user_id"])))
    print(format_report(rep))
    (ARTIFACTS / f"curve_{part}.json").write_text(
        json.dumps(dict(cfg={k: v for k, v in cfg.items()}, val=V.isoformat(),
                        arm=a.arm, lam=lam, heads=r["head_names"], hist=r["hist"]),
                   indent=1, default=str), encoding="utf-8")


def cmd_smoke(a):
    """Быстрая проверка: доли меток, шкалы и один короткий прогон на двух cutoff'ах."""
    build_future()
    T = dt.date(2025, 9, 4)
    rows = user_rows(panel_users(T, 1)["user_id"].to_numpy()[:20000])
    for arm, heads in ARMS.items():
        if not heads:
            continue
        A = aux_labels(T, rows, heads)
        s, b = aux_scales(A, heads)
        log(f"{arm:<8} " + ", ".join(f"{h.name}={A[:, j].mean():.3f}/s={s[j]:.3f}"
                                     for j, h in enumerate(heads)))
    for V in VAL_FOLDS_S1:
        for hd in ARMS["FUNNEL"]:
            assert V + dt.timedelta(days=hd.h) < POISON_START
    cfg = dict(DEFAULT_CFG, epochs=1, batch=256, chunk=256, workers=1, depth_aug=0.5)
    r = train_fold(dt.date(2025, 10, 16), cfg, ARMS["FUNNEL"], 0.3, n_cutoffs=2,
                   val_frac=0.05)
    log(f"smoke ok: mean z {r['z'].mean():.4f}, aux {r['aux_pred'].shape}")


def main():
    ap = argparse.ArgumentParser(description="EXP-038 FNL: future funnel supervision")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--force", action="store_true")

    sub.add_parser("smoke")

    f = sub.add_parser("fold")
    f.add_argument("--val", required=True)
    f.add_argument("--arm", required=True, choices=sorted(ARMS))
    f.add_argument("--lam", type=float, default=0.3)
    f.add_argument("--exp", default=None)
    f.add_argument("--curve", action="store_true", help="валидировать после каждой эпохи")
    f.add_argument("--n-cutoffs", type=int, default=None, help="только последние N (отладка)")
    f.add_argument("--val-frac", type=float, default=1.0, help="доля val-панели (отладка)")
    f.add_argument("--no-ckpt", action="store_true")
    f.add_argument("--depth-aug", type=float, default=0.5)
    f.add_argument("--depth-grid", type=int, nargs="+", default=None)
    for k, t in [("hidden", int), ("blocks", int), ("kernel", int), ("dropout", float),
                 ("batch", int), ("chunk", int), ("lr", float), ("wd", float),
                 ("epochs", int), ("seed", int), ("workers", int)]:
        f.add_argument(f"--{k}", type=t, default=None)

    a = ap.parse_args()
    if a.cmd == "build":
        build_future(a.force)
    elif a.cmd == "smoke":
        cmd_smoke(a)
    elif a.cmd == "fold":
        cmd_fold(a)


if __name__ == "__main__":
    main()
