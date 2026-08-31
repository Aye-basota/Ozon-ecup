"""SEQ-01 — энкодер сырой дневной последовательности (dilated TCN).

Проверяемая гипотеза. `exp_021` закрыл новые ФУНКЦИИ от тех же событий,
`exp_024` — новую РАЗМЕТКУ будущего. Осталась третья ось: 227 признаков `S1-E10`
это оконные агрегаты (`w7..w365`, `rec_*`, `trend_*`), то есть свёртка истории
фиксированными box-ядрами. Вопрос эксперимента ровно один:

    теряется ли полезная информация ИМЕННО ПРИ АГРЕГАЦИИ временного ряда?

Поэтому вход модели — не агрегаты, а сырая дневная последовательность 365 дней
перед cutoff'ом, а голова ровно одна: `z30 = log1p(GMV в (T, T+30])`. Никаких
hazard/count/multi-task голов: `exp_024` показал, что сами по себе эти таргеты
поверх текущего представления нового сигнала не дают, и их присутствие здесь
смешало бы два вопроса.

## Представление данных

Наивная материализация «строка = (пользователь, cutoff) x 365 дней» это
5 млн x 365 = 1.8 млрд ячеек на один фолд. Вместо неё строится ОДНА плотная
панель `(250 000 пользователей, 409 дней, 14 каналов)` в fp16 = 2.9 ГБ: все
пользователи, все дни датасета, каждый день ровно один раз. Последовательность
пары `(u, T)` — это срез `panel[u, dT-364 : dT+1]`, то есть представление
переиспользуется всеми 29 cutoff'ами без копирования. Наивная форма больше
в 18 раз и не нужна.

Каналы (14 хранимых + 3 вычисляемых на лету):

    present  1[есть строка в этот день]        — «нет строки» != «строка из нулей»
    searches cat  s2cart s2ord c2cart c2ord  cart ord      — log1p, кроме cat (0/1)
    gmv_search gmv_cat gmv                                 — log1p
    buy      1[gmv > 0]
    ponly    1[день без searches/cat/cart/ord]
    avail    1[день внутри диапазона данных]   — отделяет «до начала данных»
    dow_sin dow_cos                            — день недели

`avail` обязателен: на раннем cutoff'е (2025-04-03) доступно 92 дня из 365, и без
этого канала «до начала данных» неотличимо от «пользователь не заходил» — та же
проблема глубины истории, что решает `normalize_long` в табличном пайплайне.

Масштабирование — ТОЛЬКО деление на поканальный RMS, посчитанный по существующим
дням <= 2025-07-31 (последний обучающий cutoff самого раннего фолда, то есть
прошлое для всех четырёх фолдов). Без вычитания среднего: ноль обязан оставаться
нулём, иначе отсутствующий день перестаёт быть отсутствующим.

## Валидация

Штатная схема проекта без изменений: фолды 09-04/09-18/10-02/10-16, обучающие
cutoff'ы `T + 30 <= V`, train-панель 1-блочная, val-панель 3-блочная, метрика —
калиброванный wCV с весами 1:2:4:8. OOF пишется в общий формат
`artifacts/oof_<EXP>.npz`, поэтому `blend.py`/`report.py`/`ptime_eval.py`
работают без правок.

Запуск:
  python -m src.seq build                       # плотная панель, один раз (~3 мин)
  python -m src.seq smoke                       # быстрая проверка корректности
  python -m src.seq fold --val 2025-10-16 --curve
  python -m src.seq fold --val 2025-10-16       # боевой прогон одного фолда
  python -m src.seq merge --exp SEQ-01
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import threading
import time
from queue import Queue

import numpy as np
import polars as pl

from src.config import (ARTIFACTS, CUTOFF_STEP, DATA_END, DATA_PROCESSED, DATA_START, SEED,
                        TARGET_DAYS, VAL_FOLDS_S1, cutoff_grid)
from src.data import load
from src.features import panel_users
from src.report import evaluate, format_report, save_report
from src.tracking import save_oof
from src.validation import bias_z, calibrate, rmsle_z

T0 = time.time()

N_USERS = 250_000
N_DAYS = (DATA_END - DATA_START).days + 1          # 409
SEQ_L = 365                                        # окно последовательности
MIN_HISTORY = 90                                   # сетка cutoff'ов как у S1-E10

# --- каналы ---------------------------------------------------------------------
# LOG_COLS хранятся как log1p(значение); BIN_COLS — как есть (0/1).
LOG_COLS = ["searches", "search_to_cart", "search_to_ord", "cat_to_cart", "cat_to_ord",
            "to_cart", "to_ord", "gmv_search", "gmv_cat", "gmv"]
CHANNELS = ["present", "cat", "buy", "ponly"] + LOG_COLS
N_CH_STORED = len(CHANNELS)                        # 14
EXTRA = ["avail", "dow_sin", "dow_cos"]
N_CH = N_CH_STORED + len(EXTRA)                    # 17

PANEL_NPY = DATA_PROCESSED / "seq_panel_v1.npy"
GMV_NPY = DATA_PROCESSED / "seq_gmv_v1.npy"
UID_NPY = DATA_PROCESSED / "seq_uid_v1.npy"
SCALE_JSON = DATA_PROCESSED / "seq_scale_v1.json"

# std считается по дням <= этой даты. Это последний обучающий cutoff самого
# раннего фолда (2025-09-04), поэтому окно легально для ВСЕХ четырёх фолдов.
SCALE_END = dt.date(2025, 7, 31)


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def day_index(d: dt.date) -> int:
    return (d - DATA_START).days


# =============================================================================== панель
def build_panel(force: bool = False) -> None:
    """Плотная панель (пользователи x дни x каналы) fp16 + матрица сырого GMV.

    Строится один раз и переиспользуется всеми cutoff'ами и фолдами. Порядок
    пользователей — по возрастанию `user_id`, тот же, что у `panel_users`,
    поэтому индекс пользователя ищется одним `searchsorted`.
    """
    if PANEL_NPY.exists() and not force:
        log(f"панель уже собрана: {PANEL_NPY}")
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

    log(f"строк {len(flat):,}, пользователей {N_USERS:,}, дней {N_DAYS}; "
        f"панель {N_USERS * N_DAYS * N_CH_STORED * 2 / 1e9:.2f} ГБ")
    out = np.zeros((N_USERS * N_DAYS, N_CH_STORED), dtype=np.float16)

    gmv = df["gmv"].to_numpy().astype(np.float64)
    cat = df["cat"].to_numpy().astype(np.float32)
    acts = (df["searches"].to_numpy() + df["cat"].to_numpy()
            + df["to_cart"].to_numpy() + df["to_ord"].to_numpy())
    out[flat, CHANNELS.index("present")] = 1.0
    out[flat, CHANNELS.index("cat")] = cat
    out[flat, CHANNELS.index("buy")] = (gmv > 0).astype(np.float32)
    out[flat, CHANNELS.index("ponly")] = (acts == 0).astype(np.float32)
    for c in LOG_COLS:
        v = np.log1p(df[c].to_numpy().astype(np.float32))
        out[flat, CHANNELS.index(c)] = v
        del v
    out = out.reshape(N_USERS, N_DAYS, N_CH_STORED)

    # сырой GMV отдельно: таргет обязан считаться в float64 без log-искажений
    g = np.zeros(N_USERS * N_DAYS, dtype=np.float64)
    g[flat] = gmv
    g = g.reshape(N_USERS, N_DAYS)

    np.save(UID_NPY, uid)
    np.save(GMV_NPY, g.astype(np.float64))
    np.save(PANEL_NPY, out)
    log(f"записано: {PANEL_NPY.name}, {GMV_NPY.name}, {UID_NPY.name}")
    _build_scale(out)


def _build_scale(panel: np.ndarray) -> None:
    """Поканальный RMS по существующим дням <= SCALE_END.

    Только масштаб, без сдвига: отсутствующий день обязан остаться нулём.
    Бинарные каналы не трогаются (масштаб = 1).
    """
    d_end = day_index(SCALE_END) + 1
    sub = panel[:, :d_end, :]
    pres = sub[:, :, CHANNELS.index("present")].astype(np.float32) > 0
    n = int(pres.sum())
    scale = {}
    for i, c in enumerate(CHANNELS):
        if c in ("present", "cat", "buy", "ponly"):
            scale[c] = 1.0
            continue
        v = sub[:, :, i].astype(np.float32)[pres]
        scale[c] = float(max(np.sqrt(float((v.astype(np.float64) ** 2).mean())), 1e-3))
    SCALE_JSON.write_text(json.dumps(dict(scale=scale, n_present_cells=n,
                                          scale_end=SCALE_END.isoformat()), indent=1),
                          encoding="utf-8")
    log(f"масштаб каналов посчитан по {n:,} существующим дням <= {SCALE_END}")


_P: dict = {}


def panel():
    """(panel fp16, gmv float64, uid int64, scale float32[C])."""
    if "panel" not in _P:
        assert PANEL_NPY.exists(), "сначала: python -m src.seq build"
        _P["panel"] = np.load(PANEL_NPY)
        _P["gmv"] = np.load(GMV_NPY)
        _P["uid"] = np.load(UID_NPY)
        sc = json.loads(SCALE_JSON.read_text(encoding="utf-8"))["scale"]
        _P["scale"] = np.array([1.0 / sc[c] for c in CHANNELS], dtype=np.float32)
    return _P["panel"], _P["gmv"], _P["uid"], _P["scale"]


def user_rows(user_ids: np.ndarray) -> np.ndarray:
    """Индексы строк панели по user_id."""
    _, _, uid, _ = panel()
    idx = np.searchsorted(uid, np.asarray(user_ids))
    assert np.array_equal(uid[idx], np.asarray(user_ids)), "неизвестный user_id"
    return idx.astype(np.int32)


def target_at(T: dt.date, rows: np.ndarray) -> np.ndarray:
    """GMV в окне (T, T+TARGET_DAYS]. То же определение, что `features.target`."""
    _, g, _, _ = panel()
    d = day_index(T)
    assert d + TARGET_DAYS < N_DAYS, f"окно таргета {T} выходит за данные"
    return g[rows, d + 1:d + 1 + TARGET_DAYS].sum(axis=1)


def extras_for(T: dt.date) -> np.ndarray:
    """(SEQ_L, 3): avail / dow_sin / dow_cos — функции только календаря."""
    d = day_index(T)
    days = np.arange(d - SEQ_L + 1, d + 1)
    e = np.zeros((SEQ_L, 3), dtype=np.float16)
    e[:, 0] = (days >= 0).astype(np.float16)
    dow = (days + DATA_START.weekday()) % 7
    e[:, 1] = np.sin(2 * np.pi * dow / 7.0)
    e[:, 2] = np.cos(2 * np.pi * dow / 7.0)
    return e


def gather(T: dt.date, rows: np.ndarray, buf: np.ndarray | None = None,
           depth_clip: int | None = None) -> np.ndarray:
    """(n, SEQ_L, N_CH) fp16 — окно 365 дней, заканчивающееся ДНЁМ T включительно.

    Дни до начала данных остаются нулями, и канал `avail` их помечает.

    `depth_clip` — искусственно ограничить глубину истории D днями: всё старше
    обнуляется, `avail` там тоже 0. Нужен ТОЛЬКО на тесте: в коридоре обучения
    глубина 93..254 дня, на валидации максимум 289, а на тестовом cutoff'е
    доступны все 365. Прогноз на непрожитой глубине — экстраполяция, и `--depth-clip`
    это ровно та же страховка, что `S1-E03a`/`S1-CAP` в табличной смеси.
    """
    p, _, _, _ = panel()
    d = day_index(T)
    lo, hi = max(0, d - SEQ_L + 1), d + 1
    if depth_clip is not None:
        lo = max(lo, hi - depth_clip)
    off = SEQ_L - (hi - lo)
    n = len(rows)
    out = np.zeros((n, SEQ_L, N_CH), dtype=np.float16) if buf is None else buf[:n]
    if buf is not None:
        out[:] = 0
    out[:, off:, :N_CH_STORED] = p[rows, lo:hi, :]
    e = extras_for(T)
    if depth_clip is not None:
        e = e.copy()
        e[:off, 0] = 0.0
    out[:, :, N_CH_STORED:] = e
    return out


# =============================================================================== выборка
def fold_cutoffs(V: dt.date, step: int = CUTOFF_STEP) -> list[dt.date]:
    """Обучающие cutoff'ы фолда: то же правило, что `validation.get_folds`."""
    return [T for T in cutoff_grid(MIN_HISTORY, step)
            if T + dt.timedelta(days=TARGET_DAYS) <= V]


def build_index(cuts: list[dt.date], blocks: int):
    """Список обучающих примеров: (индекс cutoff'а, строка панели, таргет)."""
    ci, ri, ys = [], [], []
    for k, T in enumerate(cuts):
        u = panel_users(T, blocks)["user_id"].to_numpy()
        r = user_rows(u)
        ci.append(np.full(len(r), k, np.int16))
        ri.append(r)
        ys.append(target_at(T, r))
    return (np.concatenate(ci), np.concatenate(ri),
            np.log1p(np.concatenate(ys)).astype(np.float32))


class Batcher:
    """Батчи из нескольких cutoff'ов сразу, с фоновой подготовкой на CPU.

    Один срез панели дёшев только если у всех строк чанка один и тот же cutoff
    (тогда это один fancy-index по пользователям при фиксированном окне дней).
    Поэтому батч собирается из `batch // chunk` чанков РАЗНЫХ cutoff'ов:
    и градиент перемешан по времени, и сборка остаётся векторной.
    """

    def __init__(self, cuts, ci, ri, y, batch: int, chunk: int, rng, workers: int = 2):
        self.cuts, self.ci, self.ri, self.y = cuts, ci, ri, y
        self.batch, self.chunk, self.rng, self.workers = batch, chunk, rng, workers
        self.per = max(batch // chunk, 1)

    def _plan(self):
        chunks = []
        for k in range(len(self.cuts)):
            idx = np.flatnonzero(self.ci == k)
            self.rng.shuffle(idx)
            chunks += [(k, idx[i:i + self.chunk]) for i in range(0, len(idx), self.chunk)]
        chunks = [chunks[i] for i in self.rng.permutation(len(chunks))]
        return [chunks[i:i + self.per] for i in range(0, len(chunks), self.per)]

    def _make(self, group):
        xs = [gather(self.cuts[k], self.ri[idx]) for k, idx in group]
        sel = np.concatenate([idx for _, idx in group])
        return np.concatenate(xs), self.y[sel]

    def __iter__(self):
        plan = self._plan()
        q: Queue = Queue(maxsize=4)

        def work(lo, hi):
            for g in plan[lo:hi]:
                q.put(self._make(g))

        # чанки раздаются потокам блоками, порядок внутри эпохи и так случайный
        bounds = np.linspace(0, len(plan), self.workers + 1).astype(int)
        ths = [threading.Thread(target=work, args=(bounds[i], bounds[i + 1]), daemon=True)
               for i in range(self.workers)]
        for t in ths:
            t.start()
        for _ in range(len(plan)):
            yield q.get()
        for t in ths:
            t.join()

    def n_batches(self):
        tot = sum(math.ceil(int((self.ci == k).sum()) / self.chunk) for k in range(len(self.cuts)))
        return math.ceil(tot / self.per)


# =============================================================================== модель
def build_model(cfg):
    import torch
    from torch import nn

    class Block(nn.Module):
        """Pre-norm residual: LayerNorm -> причинная dilated conv -> GLU -> 1x1."""

        def __init__(self, h, dil, k, p):
            super().__init__()
            self.pad = dil * (k - 1)
            self.norm = nn.LayerNorm(h)
            self.conv = nn.Conv1d(h, 2 * h, k, dilation=dil)
            self.pw = nn.Conv1d(h, h, 1)
            self.drop = nn.Dropout(p)
            nn.init.zeros_(self.pw.weight)
            nn.init.zeros_(self.pw.bias)

        def forward(self, x):                       # x: (B, H, L)
            y = self.norm(x.transpose(1, 2)).transpose(1, 2)
            y = nn.functional.pad(y, (self.pad, 0))
            y = self.conv(y)
            a, b = y.chunk(2, dim=1)
            y = a * torch.sigmoid(b)
            return x + self.drop(self.pw(y))

    class TCN(nn.Module):
        def __init__(self, c_in, h, n_blocks, k, p, z0):
            super().__init__()
            self.stem = nn.Conv1d(c_in, h, 1)       # temporal projection
            dil = [2 ** i for i in range(n_blocks)]
            self.blocks = nn.ModuleList([Block(h, d, k, p) for d in dil])
            self.norm = nn.LayerNorm(h)
            self.head = nn.Sequential(nn.Linear(3 * h, h), nn.GELU(), nn.Dropout(p),
                                      nn.Linear(h, 1))
            nn.init.zeros_(self.head[-1].weight)
            nn.init.constant_(self.head[-1].bias, z0)

        def encode(self, x):                        # x: (B, C, L) -> (B, L, H)
            h = self.stem(x)
            for b in self.blocks:
                h = b(h)
            return self.norm(h.transpose(1, 2))

        def forward(self, x):
            h = self.encode(x)
            pooled = torch.cat([h[:, -1], h.mean(1), h.amax(1)], dim=1)
            return self.head(pooled).squeeze(1)

    return TCN(N_CH, cfg["hidden"], cfg["blocks"], cfg["kernel"], cfg["dropout"], cfg["z0"])


def receptive_field(n_blocks: int, k: int) -> int:
    return 1 + sum((k - 1) * 2 ** i for i in range(n_blocks))


DEFAULT_CFG = dict(hidden=64, blocks=8, kernel=3, dropout=0.10, batch=1024, chunk=256,
                   lr=3e-3, wd=1e-2, epochs=3, warmup=300, seed=SEED, workers=3)


# =============================================================================== обучение
def _device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(model, T, rows, cfg, dev, depth_clip: int | None = None):
    """z для одного cutoff'а, батчами; возвращает np.float32."""
    import torch
    model.eval()
    out = np.empty(len(rows), np.float32)
    sc = torch.from_numpy(panel()[3]).to(dev).view(1, N_CH_STORED, 1)
    B = cfg["batch"]
    with torch.no_grad():
        for i in range(0, len(rows), B):
            x = gather(T, rows[i:i + B], depth_clip=depth_clip)
            t = torch.from_numpy(x).to(dev).permute(0, 2, 1).contiguous().float()
            t[:, :N_CH_STORED] *= sc
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                z = model(t)
            out[i:i + B] = z.float().cpu().numpy()
    model.train()
    return out


def fit_model(cuts: list[dt.date], cfg: dict, eval_fn=None):
    """Обучение на списке cutoff'ов. `eval_fn(model, dev)` — диагностика по эпохам."""
    import torch

    dev = _device()
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ci, ri, zy = build_index(cuts, blocks=1)          # train-панель 1-блочная
    log(f"{len(cuts)} обучающих cutoff'ов {cuts[0]}..{cuts[-1]}, "
        f"{len(zy):,} примеров, mean z = {zy.mean():.4f}")

    cfg = dict(cfg, z0=float(zy.mean()))
    model = build_model(cfg).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    log(f"  модель: hidden={cfg['hidden']} blocks={cfg['blocks']} k={cfg['kernel']} "
        f"параметров {n_par:,}, рецептивное поле {receptive_field(cfg['blocks'], cfg['kernel'])} дней")

    decay = [p for n, p in model.named_parameters() if p.dim() > 1]
    nodecay = [p for n, p in model.named_parameters() if p.dim() <= 1]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=cfg["wd"]),
                             dict(params=nodecay, weight_decay=0.0)], lr=cfg["lr"],
                            betas=(0.9, 0.98))
    rng = np.random.default_rng(cfg["seed"])
    bat = Batcher(cuts, ci, ri, zy, cfg["batch"], cfg["chunk"], rng, cfg["workers"])
    total = bat.n_batches() * cfg["epochs"]
    log(f"  шагов всего {total:,} ({bat.n_batches():,} на эпоху), batch={cfg['batch']}")

    sc = torch.from_numpy(panel()[3]).to(dev).view(1, N_CH_STORED, 1)
    step, hist = 0, []
    for ep in range(cfg["epochs"]):
        t_ep, run, seen = time.time(), 0.0, 0
        for x, yb in bat:
            lr = cfg["lr"] * (min(1.0, (step + 1) / cfg["warmup"])
                              * 0.5 * (1 + math.cos(math.pi * min(1.0, step / total))))
            for g in opt.param_groups:
                g["lr"] = lr
            t = torch.from_numpy(x).to(dev).permute(0, 2, 1).contiguous().float()
            t[:, :N_CH_STORED] *= sc
            tgt = torch.from_numpy(yb).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                loss = torch.nn.functional.mse_loss(model(t), tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += float(loss.detach()) * len(yb)
            seen += len(yb)
            step += 1
        msg = (f"  эпоха {ep + 1}/{cfg['epochs']}: train MSE {run / seen:.5f} "
               f"[{time.time() - t_ep:.0f}s]")
        if eval_fn is not None:
            r = eval_fn(model, dev, ep, cfg)
            if r:
                msg += "  " + r.pop("_msg", "")
                hist.append(dict(epoch=ep + 1, train_mse=run / seen, **r))
        log(msg)
    return model, dev, cfg, hist


def train_fold(V: dt.date, cfg: dict, curve: bool = False, n_cutoffs: int | None = None,
               val_frac: float = 1.0):
    """Обучение одного фолда. Возвращает (user_id, z, y, история кривой)."""
    cuts = fold_cutoffs(V)
    if n_cutoffs:
        cuts = cuts[-n_cutoffs:]
    uv = panel_users(V, 3)["user_id"].to_numpy()      # val-панель 3-блочная
    if val_frac < 1.0:
        uv = uv[::int(round(1 / val_frac))]
    rv = user_rows(uv)
    yv = target_at(V, rv)
    log(f"фолд {V}: val-панель {len(uv):,} пользователей, доля y>0 = {(yv > 0).mean():.4f}")

    def ev(model, dev, ep, c):
        if not (curve or ep == c["epochs"] - 1):
            return None
        z = np.maximum(predict(model, V, rv, c, dev), 0.0)
        o, sc_o = calibrate(yv, z)
        return dict(rmsle=rmsle_z(yv, z), rmsle_cal=sc_o, bias=bias_z(yv, z), offset=o,
                    _msg=f"val RMSLE {rmsle_z(yv, z):.5f} -> калибр. {sc_o:.5f} (сдвиг {o:+.3f})")

    model, dev, c, hist = fit_model(cuts, cfg, ev)
    z = np.maximum(predict(model, V, rv, c, dev), 0.0)
    return uv, z, yv, hist


def train_test(cfg: dict, depth_clip: int | None, n_cutoffs: int | None = None):
    """Тестовая модель: весь чистый коридор -> прогноз на 2026-02-13.

    Обучающие cutoff'ы — вся сетка до `CORRIDOR_END` (29 штук), как у `S1-DIST`:
    отказ от свежих cutoff'ов стоил +0.0004 на LB (`exp_015`).
    """
    from src.config import CUTOFF_TEST
    cuts = cutoff_grid(MIN_HISTORY, CUTOFF_STEP)
    if n_cutoffs:
        cuts = cuts[-n_cutoffs:]
    model, dev, c, _ = fit_model(cuts, cfg)
    ut = panel_users(CUTOFF_TEST, 3)["user_id"].to_numpy()
    rt = user_rows(ut)
    log(f"тестовая панель {len(ut):,} пользователей, cutoff {CUTOFF_TEST}")
    out = {}
    for tag, dc in [("FULL", None), (f"D{depth_clip}", depth_clip)]:
        if dc is None and depth_clip is None and tag != "FULL":
            continue
        z = np.maximum(predict(model, CUTOFF_TEST, rt, c, dev, depth_clip=dc), 0.0)
        out[tag] = z
        log(f"  глубина {tag}: mean(z) = {z.mean():.4f}, доля нулей {float((z == 0).mean()):.4%}")
    if len(out) == 2:
        a, b = list(out.values())
        log(f"  corr(FULL, клип) = {np.corrcoef(a, b)[0, 1]:.5f}, "
            f"Var(разности) = {np.var(a - b):.5f}")
    return ut, out


# =============================================================================== CLI
def cmd_fold(a):
    cfg = dict(DEFAULT_CFG)
    for k in ("hidden", "blocks", "kernel", "dropout", "batch", "lr", "epochs", "seed",
              "workers", "chunk", "wd"):
        v = getattr(a, k, None)
        if v is not None:
            cfg[k] = v
    V = dt.date.fromisoformat(a.val)
    uv, z, yv, hist = train_fold(V, cfg, curve=a.curve, n_cutoffs=a.n_cutoffs,
                                 val_frac=a.val_frac)
    exp = a.exp or f"SEQ-01-S{cfg['seed']}"
    part = f"{exp}-V{V.strftime('%m%d')}"
    if a.val_frac >= 1.0 and not a.n_cutoffs:
        save_oof(part, uv, [V.isoformat()] * len(uv), z, yv)
        log(f"OOF сохранён: artifacts/oof_{part}.npz")
    rep = evaluate(yv, z, np.array([V.isoformat()] * len(uv)))
    print(format_report(rep))
    (ARTIFACTS / f"curve_{part}.json").write_text(
        json.dumps(dict(cfg={k: v for k, v in cfg.items()}, val=V.isoformat(), hist=hist),
                   indent=1), encoding="utf-8")


def cmd_predict(a):
    """Тестовая модель и `artifacts/ztest_*.npy` для `src.submit`."""
    cfg = dict(DEFAULT_CFG)
    for k in ("hidden", "blocks", "epochs", "seed", "batch", "lr", "workers"):
        v = getattr(a, k, None)
        if v is not None:
            cfg[k] = v
    ut, out = train_test(cfg, a.depth_clip, getattr(a, "n_cutoffs", None))
    np.save(ARTIFACTS / f"uid_{a.exp}.npy", ut)
    for tag, z in out.items():
        name = a.exp if tag != "FULL" else f"{a.exp}-FULL"
        np.save(ARTIFACTS / f"ztest_{name}.npy", z.astype(np.float64))
        np.save(ARTIFACTS / f"uid_{name}.npy", ut)
        log(f"сохранено: artifacts/ztest_{name}.npy (mean z {z.mean():.4f})")


def cmd_merge(a):
    from src.merge_oof import load_parts, merge_arrays
    from src.merge_oof import auc_positive
    parts = [f"{a.exp}-V{d.strftime('%m%d')}" for d in VAL_FOLDS_S1]
    uid, cut, z, y = load_parts(parts)
    rep = merge_arrays(uid, cut, z, y)
    print(format_report(rep))
    print(f"  AUC(1[y>0]) = {auc_positive(y, z):.5f}")
    save_oof(a.exp, uid, cut, z, y)
    save_report(a.exp, rep, extra=dict(description=a.desc, parts=parts))
    log(f"OOF: artifacts/oof_{a.exp}.npz")


def cmd_smoke(a):
    """Быстрая проверка: 3 cutoff'а, 1 эпоха, 5% валидации."""
    cfg = dict(DEFAULT_CFG, epochs=1, hidden=32, blocks=6)
    uv, z, yv, hist = train_fold(dt.date(2025, 10, 16), cfg, curve=True, n_cutoffs=3,
                                 val_frac=0.05)
    log(f"smoke: RMSLE {rmsle_z(yv, z):.5f}, калибр. {calibrate(yv, z)[1]:.5f}, "
        f"mean z {z.mean():.4f}, y>0 {(yv > 0).mean():.4f}")


def main():
    ap = argparse.ArgumentParser(description="SEQ-01: dilated TCN на сырой дневной истории")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--force", action="store_true")

    sub.add_parser("smoke")

    f = sub.add_parser("fold")
    f.add_argument("--val", required=True)
    f.add_argument("--exp", default=None)
    f.add_argument("--curve", action="store_true", help="валидировать после каждой эпохи")
    f.add_argument("--n-cutoffs", type=int, default=None, help="только последние N (отладка)")
    f.add_argument("--val-frac", type=float, default=1.0, help="доля val-панели (отладка)")
    for k, t in [("hidden", int), ("blocks", int), ("kernel", int), ("dropout", float),
                 ("batch", int), ("chunk", int), ("lr", float), ("wd", float),
                 ("epochs", int), ("seed", int), ("workers", int)]:
        f.add_argument(f"--{k}", type=t, default=None)

    p = sub.add_parser("predict")
    p.add_argument("--exp", default="SEQ-01")
    p.add_argument("--depth-clip", type=int, default=289,
                   help="ограничить глубину истории на тесте (289 = максимум, "
                        "прожитый моделью на валидации); 0 = не ограничивать")
    p.add_argument("--n-cutoffs", type=int, default=None, help="только последние N (отладка)")
    for k, t in [("hidden", int), ("blocks", int), ("epochs", int), ("seed", int),
                 ("batch", int), ("lr", float), ("workers", int)]:
        p.add_argument(f"--{k}", type=t, default=None)

    m = sub.add_parser("merge")
    m.add_argument("--exp", default="SEQ-01-S42")
    m.add_argument("--desc", default="SEQ-01: dilated TCN на сырой дневной последовательности")

    a = ap.parse_args()
    if a.cmd == "build":
        build_panel(a.force)
    elif a.cmd == "smoke":
        cmd_smoke(a)
    elif a.cmd == "fold":
        cmd_fold(a)
    elif a.cmd == "predict":
        a.depth_clip = a.depth_clip or None
        cmd_predict(a)
    elif a.cmd == "merge":
        cmd_merge(a)


if __name__ == "__main__":
    main()
