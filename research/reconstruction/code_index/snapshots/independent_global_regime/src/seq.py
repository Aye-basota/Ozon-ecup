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

...и он же оказался механизмом провала на LB. `exp_027`: у всех 29 обучающих
cutoff'ов и всех 4 фолдов в окне не меньше 76 позиций `avail = 0`, а на тесте при
полной глубине их РОВНО НОЛЬ, и цена этого одного бита +0.0034…+0.0044 RMSLE.
`SEQ-AVAIL-AUG` вводит режим `avail ≡ 1` в обучение аугментацией границы
(`--aug`, `sample_avail_shift`): поведенческие каналы при этом остаются строго
нулевыми, меняется только состояние доступности. Инференс не аугментируется
никогда, поэтому прогноз детерминирован.

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


def n_zero_positions(T: dt.date, depth_clip: int | None = None) -> int:
    """Сколько ВЕДУЩИХ позиций окна имеют `avail = 0`.

    Это дни до начала данных (и, если задан `depth_clip`, срезанные глубиной).
    Поведенческие каналы там нули по построению — строк лога за эти дни не
    существует. Число ровно то же, что `off` внутри `gather`.
    """
    d = day_index(T)
    lo, hi = max(0, d - SEQ_L + 1), d + 1
    if depth_clip is not None:
        lo = max(lo, hi - depth_clip)
    return SEQ_L - (hi - lo)


# --------------------------------------------------- аугментация границы `avail`
AUG_MODES = ("none", "avail_drop", "avail_bnd", "no_avail")


def sample_avail_shift(mode: str, n_zero: int, n: int, rng, p: float, full: float):
    """`k_i` — сколько ведущих позиций `avail = 0` перевести в `avail = 1`.

    `exp_027` намерил механизм провала на LB: у всех 29 обучающих cutoff'ов и
    всех 4 фолдов в окне ≥ 76 позиций `avail = 0`, а на тесте при полной глубине
    их РОВНО НОЛЬ. Цена этого одного бита — +0.0034…+0.0044 калиброванного
    RMSLE. Здесь режим `avail ≡ 1` вводится в обучение, чтобы он перестал быть
    непрожитым.

    Данные при этом не придумываются: переводимые позиции лежат до начала
    данных, все поведенческие каналы там строго нули, и меняется ровно один бит
    состояния — «этих дней не существует» -> «дни есть, активности не было».
    Таргет не участвует вовсе.

    Режимы:
      `avail_drop`  — вариант A: с вероятностью `p` весь пример получает
                      `avail ≡ 1` (маскирование канала целиком), иначе ничего.
      `avail_bnd`   — вариант B: с вероятностью `p` граница сдвигается, и внутри
                      этого доля `full` уходит в `avail ≡ 1`, остальные получают
                      равномерное `k ~ U{0..n_zero}`. Train видит весь диапазон
                      числа нулей, включая 0.
      `no_avail`    — вариант C (контроль): `avail ≡ 1` ВСЕГДА, и на обучении, и
                      на инференсе, то есть канал константен и границы не несёт.
    """
    assert mode in AUG_MODES, f"неизвестный режим аугментации {mode}"
    if mode == "none" or n_zero <= 0:
        return None                        # на тесте нулевых позиций нет — no-op
    if mode == "no_avail":
        return np.full(n, n_zero, np.int64)
    if mode == "avail_drop":
        return np.where(rng.random(n) < p, n_zero, 0).astype(np.int64)
    on = rng.random(n) < p
    k = np.where(rng.random(n) < full, n_zero, rng.integers(0, n_zero + 1, n))
    return np.where(on, k, 0).astype(np.int64)


def gather(T: dt.date, rows: np.ndarray, buf: np.ndarray | None = None,
           depth_clip: int | None = None, avail_full: bool = False,
           avail_shift: np.ndarray | None = None) -> np.ndarray:
    """(n, SEQ_L, N_CH) fp16 — окно 365 дней, заканчивающееся ДНЁМ T включительно.

    Дни до начала данных остаются нулями, и канал `avail` их помечает.

    `depth_clip` — искусственно ограничить глубину истории D днями: всё старше
    обнуляется, `avail` там тоже 0. Нужен ТОЛЬКО на тесте: в коридоре обучения
    глубина 93..254 дня, на валидации максимум 289, а на тестовом cutoff'е
    доступны все 365. Прогноз на непрожитой глубине — экстраполяция, и `--depth-clip`
    это ровно та же страховка, что `S1-E03a`/`S1-CAP` в табличной смеси.

    `avail_full` — ДИАГНОСТИКА, не боевой режим: принудительно `avail = 1` на всём
    окне при неизменных данных. Нужен потому, что у всех 29 обучающих cutoff'ов и
    всех 4 фолдов в окне не меньше 76 позиций с `avail = 0` (данные начинаются
    2025-01-01), а на тесте при полной глубине их РОВНО НОЛЬ. Это единственная
    точка задачи, где канал `avail` постоянен, и обучающих примеров у неё нет
    ни одного. Флаг позволяет измерить, насколько сеть на этот канал опирается.

    `avail_shift` — АУГМЕНТАЦИЯ (только обучение), по одному числу `k_i` на
    пример: последние `k_i` из ведущих позиций с `avail = 0` переводятся в
    `avail = 1`. Граница остаётся границей — префикс нулей и суффикс единиц, —
    но её позиция становится случайной, вплоть до `avail ≡ 1`. Поведенческие
    каналы НЕ трогаются: в этих позициях они нули по построению и нулями
    остаются, то есть новых данных не появляется ни бита.
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
    if avail_full:
        e = e.copy()
        e[:, 0] = 1.0
    out[:, :, N_CH_STORED:] = e
    if avail_shift is not None and off > 0:
        k = np.clip(np.asarray(avail_shift, np.int64), 0, off)
        assert len(k) == n, f"avail_shift длины {len(k)} при {n} примерах"
        pos = np.arange(SEQ_L)[None, :]
        m = (pos >= (off - k)[:, None]) & (pos < off)
        out[:, :, N_CH_STORED][m] = 1.0
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

    def __init__(self, cuts, ci, ri, y, batch: int, chunk: int, rng, workers: int = 2,
                 aug: dict | None = None, aug_seed=None):
        self.cuts, self.ci, self.ri, self.y = cuts, ci, ri, y
        self.batch, self.chunk, self.rng, self.workers = batch, chunk, rng, workers
        self.per = max(batch // chunk, 1)
        # Аугментация применяется ТОЛЬКО здесь, то есть только на обучении:
        # `predict` собирает вход через `gather` без `avail_shift` и остаётся
        # детерминированным.
        self.aug = dict(aug or dict(mode="none"))
        # ОТДЕЛЬНЫЙ поток случайности. Иначе раздача сидов аугментации сдвигала бы
        # `self.rng`, и порядок примеров в эпохах 2..4 у BASE разошёлся бы с
        # exp_025/exp_026 — тогда BASE перестал бы быть контролем, а разница
        # между вариантами мешалась бы с шумом порядка данных. При этом потоке
        # BASE побитово повторяет прежний прогон, а варианты видят ТЕ ЖЕ батчи в
        # ТОМ ЖЕ порядке и отличаются ровно каналом `avail`.
        self.arng = np.random.default_rng(aug_seed)

    def _plan(self):
        chunks = []
        for k in range(len(self.cuts)):
            idx = np.flatnonzero(self.ci == k)
            self.rng.shuffle(idx)
            chunks += [(k, idx[i:i + self.chunk]) for i in range(0, len(idx), self.chunk)]
        chunks = [chunks[i] for i in self.rng.permutation(len(chunks))]
        return [chunks[i:i + self.per] for i in range(0, len(chunks), self.per)]

    def _shift(self, T, n, g):
        """`k_i` для одного чанка. `None` = вход собирается ровно как раньше."""
        if g is None or self.aug["mode"] == "none":
            return None
        return sample_avail_shift(self.aug["mode"], n_zero_positions(T), n, g,
                                  self.aug.get("p", 0.0), self.aug.get("full", 0.0))

    def _make(self, group, seed=None):
        # Свой Generator на батч: `_make` вызывается из нескольких потоков, а
        # общий rng не потокобезопасен. Сиды раздаются в главном потоке, поэтому
        # обучение остаётся воспроизводимым при том же `cfg["seed"]`.
        g = np.random.default_rng(seed) if seed is not None else None
        xs = [gather(self.cuts[k], self.ri[idx],
                     avail_shift=self._shift(self.cuts[k], len(idx), g))
              for k, idx in group]
        sel = np.concatenate([idx for _, idx in group])
        return np.concatenate(xs), self.y[sel]

    def __iter__(self):
        plan = self._plan()
        seeds = self.arng.integers(0, 2 ** 62, size=len(plan))
        q: Queue = Queue(maxsize=4)

        def work(lo, hi):
            for i in range(lo, hi):
                q.put(self._make(plan[i], int(seeds[i])))

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
                   lr=3e-3, wd=1e-2, epochs=3, warmup=300, seed=SEED, workers=3,
                   compile=False, aug="none", aug_p=0.5, aug_full=0.5)


def aug_spec(cfg: dict) -> dict:
    return dict(mode=cfg.get("aug", "none"), p=float(cfg.get("aug_p", 0.0)),
                full=float(cfg.get("aug_full", 0.0)))


# =============================================================================== обучение
def _device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(model, T, rows, cfg, dev, depth_clip: int | None = None,
            avail_full: bool = False, avail_shift: int | None = None):
    """z для одного cutoff'а, батчами; возвращает np.float32.

    Обучающая аугментация сюда не попадает никогда: по умолчанию вход собирается
    без `avail_shift`, поэтому инференс детерминирован при любом режиме обучения.
    Исключение ровно одно и оно не случайное: у контроля `no_avail` канал
    константен ПО ОПРЕДЕЛЕНИЮ модели, и на инференсе он обязан быть таким же,
    иначе модель получила бы вход, которого не видела.

    `avail_shift` — ДИАГНОСТИКА (`availcurve`), одно и то же число для всех
    строк: сдвинуть границу на `k` позиций и посмотреть, насколько прогноз от
    этого зависит. Боевые пути (`fold`, `predict`, `merge`) его не передают.
    """
    import torch
    if cfg.get("aug") == "no_avail":
        avail_full = True
    model.eval()
    out = np.empty(len(rows), np.float32)
    sc = torch.from_numpy(panel()[3]).to(dev).view(1, N_CH_STORED, 1)
    B = cfg["batch"]
    with torch.no_grad():
        for i in range(0, len(rows), B):
            n = len(rows[i:i + B])
            sh = None if avail_shift is None else np.full(n, avail_shift, np.int64)
            x = gather(T, rows[i:i + B], depth_clip=depth_clip, avail_full=avail_full,
                       avail_shift=sh)
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
    # torch.compile ускоряет шаг вдвое (128.6 -> 64.4 мс на A10, замер `seq bench`).
    # Компилируется ТОЛЬКО обучающий проход: валидация идёт через `model` в eager,
    # у неё другая форма батча, и её доля во времени меньше процента.
    net = torch.compile(model) if cfg.get("compile") else model
    n_par = sum(p.numel() for p in model.parameters())
    log(f"  модель: hidden={cfg['hidden']} blocks={cfg['blocks']} k={cfg['kernel']} "
        f"параметров {n_par:,}, рецептивное поле {receptive_field(cfg['blocks'], cfg['kernel'])} дней")

    decay = [p for n, p in model.named_parameters() if p.dim() > 1]
    nodecay = [p for n, p in model.named_parameters() if p.dim() <= 1]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=cfg["wd"]),
                             dict(params=nodecay, weight_decay=0.0)], lr=cfg["lr"],
                            betas=(0.9, 0.98))
    rng = np.random.default_rng(cfg["seed"])
    bat = Batcher(cuts, ci, ri, zy, cfg["batch"], cfg["chunk"], rng, cfg["workers"],
                  aug=aug_spec(cfg), aug_seed=[cfg["seed"], 0xA7A1])
    if bat.aug["mode"] != "none":
        log(f"  аугментация `avail`: {bat.aug['mode']}, p={bat.aug['p']}, "
            f"full={bat.aug['full']}; нулевых позиций в окне "
            f"{n_zero_positions(cuts[0])}..{n_zero_positions(cuts[-1])}")
    total = bat.n_batches() * cfg["epochs"]
    log(f"  шагов всего {total:,} ({bat.n_batches():,} на эпоху), batch={cfg['batch']}")

    sc = torch.from_numpy(panel()[3]).to(dev).view(1, N_CH_STORED, 1)
    pad_to = cfg["batch"] if cfg.get("compile") else 0
    step, hist = 0, []
    for ep in range(cfg["epochs"]):
        t_ep, run, seen = time.time(), None, 0
        for x, yb in bat:
            lr = cfg["lr"] * (min(1.0, (step + 1) / cfg["warmup"])
                              * 0.5 * (1 + math.cos(math.pi * min(1.0, step / total))))
            for g in opt.param_groups:
                g["lr"] = lr
            n = len(yb)
            t = torch.from_numpy(x).to(dev).permute(0, 2, 1).contiguous().float()
            t[:, :N_CH_STORED] *= sc
            tgt = torch.from_numpy(yb).to(dev)
            # Батчи из неполных чанков дают ~25 разных форм за эпоху, и без
            # добивки dynamo перекомпилирует граф на каждую. Добивка НУЛЕВЫМИ
            # строками ничего не меняет по существу: все слои сети поэлементны по
            # примеру (conv по времени, LayerNorm по каналам, голова — линейная),
            # межпримерных операций нет, а лишние строки срезаются до loss.
            if pad_to and n < pad_to:
                t = torch.nn.functional.pad(t, (0, 0, 0, 0, 0, pad_to - n))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                loss = torch.nn.functional.mse_loss(net(t)[:n], tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            # накопление на GPU: `float(loss)` синхронизировал бы CPU с GPU на
            # КАЖДОМ шаге и мешал бы перекрытию H2D со счётом. На обучение это
            # не влияет никак — `run` идёт только в лог.
            run = loss.detach() * n if run is None else run + loss.detach() * n
            seen += n
            step += 1
        run = float(run)
        msg = (f"  эпоха {ep + 1}/{cfg['epochs']}: train MSE {run / seen:.5f} "
               f"[{time.time() - t_ep:.0f}s]")
        if eval_fn is not None:
            r = eval_fn(model, dev, ep, cfg)
            if r:
                msg += "  " + r.pop("_msg", "")
                hist.append(dict(epoch=ep + 1, train_mse=run / seen, **r))
        log(msg)
    return model, dev, cfg, hist


def save_ckpt(part: str, model, cfg: dict, V: dt.date) -> None:
    """Веса фолда на диск.

    `SEQ-01` их не сохранял, поэтому диагностика по глубине истории (`depth`)
    была невозможна без переобучения. Файл маленький (245 633 параметра, ~1 МБ),
    сохранять его всегда дешевле, чем ещё раз считать фолд.
    """
    import torch
    p = ARTIFACTS / f"model_{part}.pt"
    torch.save(dict(state=model.state_dict(), cfg={k: v for k, v in cfg.items()},
                    val=V.isoformat()), p)
    log(f"веса сохранены: artifacts/{p.name}")


def load_ckpt(part: str):
    """(model на устройстве, cfg, дата фолда) из `artifacts/model_<part>.pt`."""
    import torch
    p = ARTIFACTS / f"model_{part}.pt"
    assert p.exists(), f"нет чекпойнта {p}; фолд нужно посчитать с сохранением весов"
    d = torch.load(p, map_location="cpu", weights_only=False)
    dev = _device()
    model = build_model(d["cfg"]).to(dev)
    model.load_state_dict(d["state"])
    model.eval()
    return model, d["cfg"], dt.date.fromisoformat(d["val"]), dev


def train_fold(V: dt.date, cfg: dict, curve: bool = False, n_cutoffs: int | None = None,
               val_frac: float = 1.0, ckpt: str | None = None):
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
    if ckpt:
        save_ckpt(ckpt, model, c, V)
    return uv, z, yv, hist


def train_test(cfg: dict, depth_clip: int | None, n_cutoffs: int | None = None,
               ckpt: str | None = None):
    """Тестовая модель: весь чистый коридор -> прогноз на 2026-02-13.

    Обучающие cutoff'ы — вся сетка до `CORRIDOR_END` (29 штук), как у `S1-DIST`:
    отказ от свежих cutoff'ов стоил +0.0004 на LB (`exp_015`).

    Веса сохраняются ВСЕГДА. `exp_027` уперся ровно в их отсутствие: чтобы
    пересчитать тестовый прогноз при другой политике глубины, пришлось бы заново
    обучать модель, хотя это чистый inference на 40 секунд. Фолды свои веса
    сохраняют с `exp_026`, тестовая модель — нет, и это стоило эксперименту
    целой ветки анализа.
    """
    from src.config import CUTOFF_TEST
    cuts = cutoff_grid(MIN_HISTORY, CUTOFF_STEP)
    if n_cutoffs:
        cuts = cuts[-n_cutoffs:]
    model, dev, c, _ = fit_model(cuts, cfg)
    if ckpt:
        save_ckpt(ckpt, model, c, CUTOFF_TEST)
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
              "workers", "chunk", "wd", "compile", "aug", "aug_p", "aug_full"):
        v = getattr(a, k, None)
        if v is not None:
            cfg[k] = v
    V = dt.date.fromisoformat(a.val)
    exp = a.exp or f"SEQ-01-S{cfg['seed']}"
    part = f"{exp}-V{V.strftime('%m%d')}"
    full = a.val_frac >= 1.0 and not a.n_cutoffs
    uv, z, yv, hist = train_fold(V, cfg, curve=a.curve, n_cutoffs=a.n_cutoffs,
                                 val_frac=a.val_frac,
                                 ckpt=part if (full and not a.no_ckpt) else None)
    if full:
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
    for k in ("hidden", "blocks", "epochs", "seed", "batch", "lr", "workers", "compile",
              "aug", "aug_p", "aug_full"):
        v = getattr(a, k, None)
        if v is not None:
            cfg[k] = v
    ut, out = train_test(cfg, a.depth_clip, getattr(a, "n_cutoffs", None),
                         ckpt=None if getattr(a, "no_ckpt", False) else f"{a.exp}-TEST")
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


def cmd_depth(a):
    """Чувствительность прогноза к глубине истории — БЕЗ переобучения.

    Главный неизмеримый риск `SEQ-01`: обучающие cutoff'ы живут на глубинах
    93..254 дня, валидационные видят максимум 289, а на тестовом cutoff'е
    доступны все 365. Здесь на сохранённом чекпойнте фолда прогноз считается
    при нескольких искусственно урезанных глубинах и сравнивается с полной.
    Данных позже cutoff'а не появляется ни при какой глубине: `gather` режет
    окно только слева.
    """
    model, cfg, V, dev = load_ckpt(a.ckpt)
    uv = panel_users(V, 3)["user_id"].to_numpy()
    rv = user_rows(uv)
    yv = target_at(V, rv)
    avail = day_index(V) + 1
    # самая глубокая история, которую модель ВИДЕЛА на обучении: последний
    # обучающий cutoff фолда. Всё, что глубже, — экстраполяция.
    d_train = day_index(fold_cutoffs(V)[-1]) + 1
    log(f"{a.ckpt}: фолд {V}, доступная глубина {avail} дней, максимум на обучении "
        f"{d_train}, {len(uv):,} пользователей")

    depths = sorted({d for d in list(a.depths) + [d_train] if d < avail} | {avail})
    zs, rows = {}, []
    for D in depths:
        z = np.maximum(predict(model, V, rv, cfg, dev, depth_clip=D), 0.0)
        zs[D] = z
        o, cal = calibrate(yv, z)
        rows.append(dict(fold=V.isoformat(), depth=D, full=int(D == avail),
                         train_max=d_train, seen_in_train=int(D <= d_train),
                         rmsle=rmsle_z(yv, z), rmsle_cal=cal, offset=o,
                         mean_z=float(z.mean()), std_z=float(z.std()),
                         q99=float(np.quantile(z, 0.99)), zero_share=float((z == 0).mean())))
        log(f"  глубина {D:>3}: RMSLE {rows[-1]['rmsle']:.5f} -> калибр. {cal:.5f} "
            f"(сдвиг {o:+.3f}), mean z {z.mean():.4f}, std {z.std():.4f}")

    ref = depths[-1]
    for r in rows:
        d = zs[r["depth"]] - zs[ref]
        r["var_vs_full"] = float(np.var(d))
        r["corr_vs_full"] = 1.0 if r["depth"] == ref else float(
            np.corrcoef(zs[r["depth"]], zs[ref])[0, 1])
        r["best_cal"] = min(x["rmsle_cal"] for x in rows)
    best = min(rows, key=lambda r: r["rmsle_cal"])
    log(f"минимум калиброванного RMSLE на глубине {best['depth']} "
        f"({best['rmsle_cal']:.5f}); полная глубина {ref} даёт "
        f"{[r['rmsle_cal'] for r in rows if r['depth'] == ref][0]:.5f}")
    out = ARTIFACTS / f"depth_{a.ckpt}.csv"
    pl.DataFrame(rows).write_csv(out)
    log(f"записано: {out}")


def cmd_crossdepth(a):
    """Стресс-тест ЭКСТРАПОЛЯЦИИ по глубине: ранняя модель на поздней панели.

    `depth` (`exp_026`) мерил глубину только на своём фолде, где за границей
    обученной глубины остаётся всего 35 дней. На тесте модель обучена до глубины
    289 и получает 365 — экстраполяция **+76 дней**, вдвое больше всего, что
    когда-либо проверялось. Прямо этот режим воспроизводится так: взять модель
    РАННЕГО фолда (её обученная глубина мала) и применить её к ПОЗДНЕЙ панели
    (там доступно больше истории). Модель `V0904` (обучена до 212) на панели
    `10-16` (доступно 289) даёт **+77** — то есть почти точную копию теста.

    Лукапа нет и быть не может: обучающие cutoff'ы модели кончаются на
    `V_ckpt − 30`, а поздняя панель целиком в их будущем. Разрыв train→val здесь
    БОЛЬШЕ штатного, то есть оценка консервативна.

    Внутри одного запуска модель, панель, таргет и пользователи фиксированы —
    меняется ровно одна величина, глубина входа. Сохраняются все `z`, поэтому
    политика усадки `z(α) = (1−α)·z_clip + α·z_full` считается потом без GPU.
    """
    model, cfg, V_ck, dev = load_ckpt(a.ckpt)
    V = dt.date.fromisoformat(a.val)
    cuts_ck = fold_cutoffs(V_ck)
    d_train = day_index(cuts_ck[-1]) + 1
    avail = day_index(V) + 1
    assert V > V_ck, f"{a.ckpt} обучен под фолд {V_ck}: панель {V} обязана быть ПОЗЖЕ"
    assert cuts_ck[-1] + dt.timedelta(days=TARGET_DAYS) <= V_ck, "чужое правило фолдов"
    assert avail > d_train, "экстраполяции нет: доступная глубина не больше обученной"

    uv = panel_users(V, 3)["user_id"].to_numpy()
    rv = user_rows(uv)
    yv = target_at(V, rv)
    log(f"{a.ckpt} (обучен под {V_ck}, максимум глубины на обучении {d_train}) "
        f"-> панель {V}: доступно {avail} дней, ЭКСТРАПОЛЯЦИЯ +{avail - d_train}, "
        f"{len(uv):,} пользователей, доля y>0 = {(yv > 0).mean():.4f}")

    depths = sorted({d for d in list(a.depths) + [d_train] if d < avail} | {avail})
    zs, rows = {}, []
    for D in depths:
        z = np.maximum(predict(model, V, rv, cfg, dev, depth_clip=D), 0.0)
        zs[D] = z
        o, cal = calibrate(yv, z)
        rows.append(dict(ckpt=a.ckpt, fold_ckpt=V_ck.isoformat(), fold_panel=V.isoformat(),
                         depth=D, train_max=d_train, avail=avail,
                         over_train=D - d_train, seen_in_train=int(D <= d_train),
                         rmsle=rmsle_z(yv, z), rmsle_cal=cal, offset=o,
                         bias=bias_z(yv, z), mean_z=float(z.mean()), std_z=float(z.std()),
                         zero_share=float((z == 0).mean())))
        log(f"  глубина {D:>3} ({D - d_train:+4d} к обученной): RMSLE {rows[-1]['rmsle']:.5f}"
            f" -> калибр. {cal:.5f} (сдвиг {o:+.3f}), mean z {z.mean():.4f}")

    ref = zs[d_train]                       # опора — обученная глубина, а не полная
    for r in rows:
        d = zs[r["depth"]] - ref
        r["var_vs_clip"] = float(np.var(d))
        r["corr_vs_clip"] = 1.0 if r["depth"] == d_train else float(
            np.corrcoef(zs[r["depth"]], ref)[0, 1])
        r["d_cal_vs_clip"] = r["rmsle_cal"] - [x for x in rows
                                               if x["depth"] == d_train][0]["rmsle_cal"]
    best = min(rows, key=lambda r: r["rmsle_cal"])
    log(f"минимум калиброванного RMSLE на глубине {best['depth']} "
        f"({best['over_train']:+d} к обученной, {best['rmsle_cal']:.5f}); "
        f"полная {avail} даёт {[r['rmsle_cal'] for r in rows if r['depth'] == avail][0]:.5f}")

    tag = f"{a.ckpt}_on{V.strftime('%m%d')}"
    np.savez_compressed(ARTIFACTS / f"xdepth_{tag}.npz", user_id=uv, y=yv.astype(np.float32),
                        depths=np.array(depths), train_max=d_train, avail=avail,
                        z=np.vstack([zs[D] for D in depths]).astype(np.float32))
    pl.DataFrame(rows).write_csv(ARTIFACTS / f"xdepth_{tag}.csv")
    log(f"записано: artifacts/xdepth_{tag}.npz, artifacts/xdepth_{tag}.csv")


def cmd_availprobe(a):
    """Насколько сеть опирается на канал `avail` там, где на тесте он уходит вне прожитого режима.

    Постановка. `--depth-clip 289` считался «страховкой от экстраполяции глубины»,
    но делал он и вторую вещь, которую никто не мерил: держал канал `avail` в
    прожитом режиме. У ВСЕХ 29 обучающих cutoff'ов и всех 4 фолдов в окне не
    меньше 76 позиций с `avail = 0` (данные начинаются 2025-01-01, а окно 365
    дней). На тесте при полной глубине их ровно ноль — `avail ≡ 1`. Такого входа
    сеть не видела ни разу, и воспроизвести его на истории НЕЛЬЗЯ: ни один
    исторический cutoff не имеет 365 дней прошлого.

    Что здесь мерится. На валидационной панели данные оставляются как есть, а
    `avail` принудительно ставится в 1 на всём окне. Данные в «долистовых»
    позициях при этом нули, то есть сеть получает сигнал «эти 76 дней доступны,
    и пользователь в них не сделал ничего» вместо «этих дней не существует».
    Это ровно тот бит, который на тесте переключается вместе с глубиной, и его
    вклад можно отделить от вклада новых данных.
    """
    model, cfg, V_ck, dev = load_ckpt(a.ckpt)
    V = dt.date.fromisoformat(a.val) if a.val else V_ck
    uv = panel_users(V, 3)["user_id"].to_numpy()
    rv = user_rows(uv)
    yv = target_at(V, rv)
    avail = day_index(V) + 1
    n_zero = SEQ_L - avail
    aug = cfg.get("aug", "none")
    log(f"{a.ckpt} на панели {V}: доступно {avail} дней, позиций с avail=0 в окне "
        f"{n_zero} (на тесте при полной глубине их 0), {len(uv):,} пользователей, "
        f"режим обучения avail = {aug}")
    if aug == "no_avail":
        log("  контроль `no_avail`: канал константен и на инференсе, поэтому Δ обязана "
            "быть РОВНО нулевой — это проверка кода, а не результат модели")

    rows = []
    z0 = np.maximum(predict(model, V, rv, cfg, dev), 0.0)
    z1 = np.maximum(predict(model, V, rv, cfg, dev, avail_full=True), 0.0)
    for tag, z in [("avail как в обучении", z0), ("avail ≡ 1 (режим теста)", z1)]:
        o, cal = calibrate(yv, z)
        rows.append(dict(ckpt=a.ckpt, panel=V.isoformat(), mode=tag, aug=aug,
                         n_zero_pos=n_zero,
                         rmsle=rmsle_z(yv, z), rmsle_cal=cal, offset=o,
                         mean_z=float(z.mean()), std_z=float(z.std())))
        log(f"  {tag:>24}: RMSLE {rows[-1]['rmsle']:.5f} -> калибр. {cal:.5f} "
            f"(сдвиг {o:+.3f}), mean z {z.mean():.4f}")
    d = z1 - z0
    rows[1]["var_vs_base"] = float(np.var(d))
    rows[1]["corr_vs_base"] = float(np.corrcoef(z0, z1)[0, 1])
    rows[1]["d_cal"] = rows[1]["rmsle_cal"] - rows[0]["rmsle_cal"]
    log(f"  переключение одного бита: Var(Δz) = {np.var(d):.5f}, mean Δ = {d.mean():+.4f}, "
        f"corr = {rows[1]['corr_vs_base']:.5f}, Δ калибр. RMSLE = {rows[1]['d_cal']:+.5f}")
    p = ARTIFACTS / f"availprobe_{a.ckpt}_on{V.strftime('%m%d')}.csv"
    pl.DataFrame(rows).write_csv(p)
    log(f"записано: {p}")


def cmd_availcurve(a):
    """Насколько прогноз зависит от ПОЛОЖЕНИЯ границы `avail`, а не только от её ухода.

    `availprobe` меряет одну точку — крайний режим `avail ≡ 1`. Гипотеза
    `SEQ-AVAIL-AUG` шире: энкодер должен стать устойчив к положению границы, и тогда
    полные 365 дней перестают быть особой точкой. Здесь граница двигается на
    `k = 0..n_zero` позиций при неизменных данных (в открываемых позициях
    поведение остаётся строго нулевым — там нет ни одной строки лога), и
    смотрится, плоская ли кривая.

    У BASE ожидается монотонный рост штрафа с максимумом на `k = n_zero`
    (это и есть число `availprobe` из `exp_027`); у удачного варианта — плато
    около нуля.
    """
    model, cfg, V_ck, dev = load_ckpt(a.ckpt)
    V = dt.date.fromisoformat(a.val) if a.val else V_ck
    uv = panel_users(V, 3)["user_id"].to_numpy()
    rv = user_rows(uv)
    yv = target_at(V, rv)
    n_zero = n_zero_positions(V)
    ks = sorted({min(k, n_zero) for k in list(a.shifts) + [0, n_zero]})
    log(f"{a.ckpt} на панели {V}: {n_zero} позиций avail=0, сетка сдвигов {ks}, "
        f"режим обучения avail = {cfg.get('aug', 'none')}")

    rows, z0 = [], None
    for k in ks:
        z = np.maximum(predict(model, V, rv, cfg, dev, avail_shift=k), 0.0)
        if z0 is None:
            z0 = z
        o, cal = calibrate(yv, z)
        rows.append(dict(ckpt=a.ckpt, panel=V.isoformat(), aug=cfg.get("aug", "none"),
                         shift=k, n_zero_left=n_zero - k, rmsle=rmsle_z(yv, z),
                         rmsle_cal=cal, offset=o, mean_z=float(z.mean()),
                         var_vs_k0=float(np.var(z - z0)),
                         corr_vs_k0=1.0 if k == ks[0] else float(np.corrcoef(z, z0)[0, 1])))
        rows[-1]["d_cal"] = cal - rows[0]["rmsle_cal"]
        log(f"  сдвиг {k:>3} (осталось нулей {n_zero - k:>3}): калибр. {cal:.5f} "
            f"({rows[-1]['d_cal']:+.5f}), Var(Δz) = {rows[-1]['var_vs_k0']:.5f}")
    p = ARTIFACTS / f"availcurve_{a.ckpt}_on{V.strftime('%m%d')}.csv"
    pl.DataFrame(rows).write_csv(p)
    log(f"записано: {p}")


def cmd_avg(a):
    """Усреднение сидов В ЛОГ-ПРОСТРАНСТВЕ (как `blend.py`: среднее сырого GMV сместило бы уровень).

    Пишет и пофолдовые OOF, и склеенный, чтобы `blend.py`/`report.py`/`analyze.py`
    работали с результатом ровно так же, как с одиночным сидом.
    """
    from src.merge_oof import auc_positive, load_parts, merge_arrays
    from src.tracking import load_oof
    tags = [f"V{d.strftime('%m%d')}" for d in VAL_FOLDS_S1]
    for t in tags:
        ds = [load_oof(f"{a.src}-S{s}-{t}") for s in a.seeds]
        u0 = np.asarray(ds[0]["user_id"])
        for d in ds[1:]:
            assert np.array_equal(np.asarray(d["user_id"]), u0), f"{t}: разные наборы строк"
            assert np.allclose(d["y"], ds[0]["y"]), f"{t}: разные таргеты"
        z = np.mean([d["z"] for d in ds], axis=0)
        save_oof(f"{a.exp}-{t}", u0, ds[0]["cutoff"], z, ds[0]["y"])
        log(f"{t}: усреднено {len(a.seeds)} сидов, "
            f"Var(z_i - z_avg) = {np.mean([np.var(d['z'] - z) for d in ds]):.5f}")
    parts = [f"{a.exp}-{t}" for t in tags]
    uid, cut, z, y = load_parts(parts)
    rep = merge_arrays(uid, cut, z, y)
    print(format_report(rep))
    print(f"  AUC(1[y>0]) = {auc_positive(y, z):.5f}")
    save_oof(a.exp, uid, cut, z, y)
    save_report(a.exp, rep, extra=dict(description=a.desc, parts=parts, seeds=a.seeds))
    log(f"OOF: artifacts/oof_{a.exp}.npz")


def cmd_bench(a):
    """Профиль шага обучения: сборка батча / H2D / forward / backward+шаг.

    Измерение, а не обучение: ни один артефакт эксперимента не пишется.
    Нужен, чтобы оптимизации выбирались по числам. `exp_025` утверждал «узкое
    место — GPU, не данные», здесь это проверяется поканально и проверяются
    варианты ускорения самого GPU-шага.
    """
    import torch
    cfg = dict(DEFAULT_CFG, z0=2.28)
    for k in ("hidden", "blocks", "batch", "chunk", "workers"):
        v = getattr(a, k, None)
        if v is not None:
            cfg[k] = v
    dev = _device()
    cuts = fold_cutoffs(dt.date(2025, 10, 16))[-a.n_cutoffs:]
    ci, ri, zy = build_index(cuts, blocks=1)
    rng = np.random.default_rng(0)
    bat = Batcher(cuts, ci, ri, zy, cfg["batch"], cfg["chunk"], rng, cfg["workers"])

    # 1. сборка батча на CPU, один поток
    plan = bat._plan()
    t0 = time.time()
    xs = [bat._make(g) for g in plan[:a.iters]]
    t_cpu = (time.time() - t0) / a.iters
    x, yb = xs[0]
    log(f"сборка батча (1 поток): {t_cpu * 1e3:7.1f} мс -> "
        f"{cfg['batch'] / t_cpu:,.0f} примеров/с, форма {x.shape}")

    def timeit(fn, n=a.iters, warm=5):
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.time() - t) / n

    sc = torch.from_numpy(panel()[3]).to(dev).view(1, N_CH_STORED, 1)
    xp = torch.from_numpy(x).pin_memory()
    xn = torch.from_numpy(x)
    log(f"H2D непинованный:  {timeit(lambda: xn.to(dev)) * 1e3:7.2f} мс "
        f"({xn.numel() * 2 / 1e6:.1f} МБ)")
    log(f"H2D пинованный:    {timeit(lambda: xp.to(dev, non_blocking=True)) * 1e3:7.2f} мс")

    tgt = torch.from_numpy(yb).to(dev)
    for tag, compile_ in [("eager", False)] + ([("compile", True)] if a.compile else []):
        model = build_model(cfg).to(dev)
        net = torch.compile(model) if compile_ else model
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        t = torch.from_numpy(x).to(dev).permute(0, 2, 1).contiguous().float()
        t[:, :N_CH_STORED] *= sc

        def fwd():
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                                 enabled=dev.type == "cuda"):
                net(t)

        def step():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                loss = torch.nn.functional.mse_loss(net(t), tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        t_f, t_s = timeit(fwd, warm=8), timeit(step, warm=8)
        log(f"[{tag:>7}] forward {t_f * 1e3:7.1f} мс | forward+backward+шаг {t_s * 1e3:7.1f} мс "
            f"| эффективная сборка {t_cpu / cfg['workers'] * 1e3:.1f} мс "
            f"| GPU-доля {t_s / max(t_s, t_cpu / cfg['workers']):.0%}")
        log(f"          пик памяти {torch.cuda.max_memory_allocated() / 1e9:.2f} ГБ; "
            f"полный сид (~67 400 шагов) = {t_s * 67400 / 3600:.2f} ч")
        del model, net, opt
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def cmd_smoke(a):
    """Быстрая проверка: 3 cutoff'а, 1 эпоха, 5% валидации."""
    cfg = dict(DEFAULT_CFG, epochs=1, hidden=32, blocks=6,
               compile=bool(getattr(a, "compile", False)))
    for k in ("aug", "aug_p", "aug_full"):
        v = getattr(a, k, None)
        if v is not None:
            cfg[k] = v
    uv, z, yv, hist = train_fold(dt.date(2025, 10, 16), cfg, curve=True, n_cutoffs=3,
                                 val_frac=0.05)
    log(f"smoke: RMSLE {rmsle_z(yv, z):.5f}, калибр. {calibrate(yv, z)[1]:.5f}, "
        f"mean z {z.mean():.4f}, y>0 {(yv > 0).mean():.4f}")
    if getattr(a, "save_z", None):
        np.save(ARTIFACTS / f"smokez_{a.save_z}.npy", z)
        log(f"z сохранён: artifacts/smokez_{a.save_z}.npy")


def _aug_args(sp):
    """Флаги аугментации границы `avail` — одинаковые у `fold` и `predict`."""
    sp.add_argument("--aug", choices=list(AUG_MODES), default=None,
                    help="avail_drop = маскировать канал целиком с вероятностью p; "
                         "avail_bnd = случайная граница (доля full уходит в avail≡1); "
                         "no_avail = контроль, канал константен всегда")
    sp.add_argument("--aug-p", type=float, default=None, help="вероятность аугментации примера")
    sp.add_argument("--aug-full", type=float, default=None,
                    help="доля аугментированных, получающих avail ≡ 1 (только avail_bnd)")


def main():
    ap = argparse.ArgumentParser(description="SEQ-01: dilated TCN на сырой дневной истории")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--force", action="store_true")

    s = sub.add_parser("smoke")
    s.add_argument("--compile", action="store_true")
    s.add_argument("--save-z", default=None, help="сохранить z для сверки eager/compile")
    _aug_args(s)

    f = sub.add_parser("fold")
    f.add_argument("--val", required=True)
    f.add_argument("--exp", default=None)
    f.add_argument("--curve", action="store_true", help="валидировать после каждой эпохи")
    f.add_argument("--n-cutoffs", type=int, default=None, help="только последние N (отладка)")
    f.add_argument("--val-frac", type=float, default=1.0, help="доля val-панели (отладка)")
    f.add_argument("--no-ckpt", action="store_true", help="не сохранять веса фолда")
    f.add_argument("--compile", action="store_true", help="torch.compile обучающего прохода (x2)")
    _aug_args(f)
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
    p.add_argument("--compile", action="store_true")
    p.add_argument("--no-ckpt", action="store_true", help="не сохранять веса тестовой модели")
    _aug_args(p)
    for k, t in [("hidden", int), ("blocks", int), ("epochs", int), ("seed", int),
                 ("batch", int), ("lr", float), ("workers", int)]:
        p.add_argument(f"--{k}", type=t, default=None)

    m = sub.add_parser("merge")
    m.add_argument("--exp", default="SEQ-01-S42")
    m.add_argument("--desc", default="SEQ-01: dilated TCN на сырой дневной последовательности")

    d = sub.add_parser("depth", help="inference-only: чувствительность к глубине истории")
    d.add_argument("--ckpt", required=True, help="имя части, напр. SEQ-01-S43-V1016")
    d.add_argument("--depths", type=int, nargs="*", default=[180, 220, 254, 289])

    x = sub.add_parser("crossdepth",
                       help="inference-only: ранняя модель на поздней панели (+60..80 дней)")
    x.add_argument("--ckpt", required=True, help="чекпойнт РАННЕГО фолда, напр. SEQ-01-S43-V0904")
    x.add_argument("--val", required=True, help="ПОЗДНЯЯ валидационная панель, напр. 2025-10-16")
    x.add_argument("--depths", type=int, nargs="*",
                   default=[212, 230, 247, 261, 275, 289])

    q = sub.add_parser("availprobe",
                       help="inference-only: сколько стоит уход канала avail вне прожитого режима")
    q.add_argument("--ckpt", required=True)
    q.add_argument("--val", default=None, help="панель; по умолчанию — фолд чекпойнта")

    ac = sub.add_parser("availcurve",
                        help="inference-only: штраф как функция ПОЛОЖЕНИЯ границы avail")
    ac.add_argument("--ckpt", required=True)
    ac.add_argument("--val", default=None, help="панель; по умолчанию — фолд чекпойнта")
    ac.add_argument("--shifts", type=int, nargs="*", default=[0, 19, 38, 57, 76])

    v = sub.add_parser("avg", help="усреднение сидов в лог-пространстве")
    v.add_argument("--exp", default="SEQ-AVG2")
    v.add_argument("--src", default="SEQ-01")
    v.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    v.add_argument("--desc", default="SEQ-02: усреднение сидов dilated TCN")

    n = sub.add_parser("bench", help="профиль шага обучения (ничего не пишет)")
    n.add_argument("--n-cutoffs", type=int, default=3)
    n.add_argument("--iters", type=int, default=20)
    n.add_argument("--compile", action="store_true")
    for k, t in [("hidden", int), ("blocks", int), ("batch", int), ("chunk", int),
                 ("workers", int)]:
        n.add_argument(f"--{k}", type=t, default=None)

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
    elif a.cmd == "depth":
        cmd_depth(a)
    elif a.cmd == "crossdepth":
        cmd_crossdepth(a)
    elif a.cmd == "availprobe":
        cmd_availprobe(a)
    elif a.cmd == "availcurve":
        cmd_availcurve(a)
    elif a.cmd == "avg":
        cmd_avg(a)
    elif a.cmd == "bench":
        cmd_bench(a)


if __name__ == "__main__":
    main()
