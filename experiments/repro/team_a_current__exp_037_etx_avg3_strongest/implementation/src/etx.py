"""ETX-01 — Sparse Event Transformer над СОБЫТИЙНОЙ историей (`STRATEGY_13`, вариант B).

Проверяемая гипотеза. `SEQ-01`/`SEQ-D3A` читают историю дилатированной причинной
свёрткой по ПЛОТНОЙ дневной сетке 365 дней, где 70% ячеек — «строки нет». Вопрос
этого эксперимента ровно один:

    извлекает ли внимание по РЕАЛЬНЫМ событиям существенно больше temporal signal,
    чем свёртка по плотной сетке нулей?

Отличие от `src/seq.py` ровно два и оба архитектурные:

  1. **представление** — токен это существующая строка лога (день, в который у
     пользователя хоть что-то было), а не ячейка календаря. У панели фолда 10-16
     в окне 289 дней медиана 83 события при среднем 95, то есть последовательность
     втрое короче сетки, и «нет строки» перестаёт быть каналом: оно перестаёт быть
     токеном. Расстояние между событиями подаётся явно (`age`, `gap`);
  2. **модель** — pre-LN трансформер (`d_model=128`, 5 блоков, 8 голов, GEGLU 384)
     с причинным `scaled_dot_product_attention` вместо TCN.

Всё остальное намеренно совпадает с `SEQ`: та же панель, те же фолды, тот же
таргет `z30 = log1p(GMV в (T, T+30])`, та же метрика, тот же формат OOF, тот же
поканальный RMS-масштаб (`data/processed/seq_scale_v1.json`), тот же оптимизатор,
то же число эпох. Сравнение обязано отвечать за архитектуру, а не за обвязку.

## Чего здесь СОЗНАТЕЛЬНО нет

* **Multi-horizon / hazard / count голов** — `exp_024` закрыл эти таргеты поверх
  текущего представления (+0.00286 wCV, 0/4), и их присутствие смешало бы два вопроса.
* **Causal dense supervision по нескольким историческим позициям.** Причинная
  маска её допускает (позиция `j` видит только события `<= j`), но:
  ближайший замер проекта — `exp_022` (плотная сетка cutoff'ов при равном объёме,
  +0.001263 wCV, 0/4) — говорит, что лишние temporal-позиции сигнала не добавляют;
  а панель на исторической позиции `t_j` отбиралась бы не правилом организатора
  (3 блока перед `t_j`), а правилом на `T`, то есть распределение примеров
  разъехалось бы с боевым. **Семантика супервизии здесь ровно одна: один
  forward = один честный пример `(история <= T) -> y30(T)`**, как у `SEQ`.
* **Смещение внимания в ЛИЧНОМ времени** (`τ_h · ρ_i`, гипотеза `G1` стратегии) —
  `exp_021` закрыл личное время как представление (30 признаков `pt_*`,
  −0.000006 wCV, контроль с перемешанной ρ не хуже настоящей). Здесь стоит его
  календарный аналог (см. ниже), и он же — дешёвый гейт для `G1`.

## Токен события

    14 хранимых каналов панели (`seq.CHANNELS`), поканальный RMS-масштаб SEQ
    + log1p(age)/log1p(365), age/365, exp(-age/7), exp(-age/30), exp(-age/90)
    + log1p(gap до предыдущего события)/log1p(365)
    + dow_sin, dow_cos                                        итого 22 признака

`present` в событийном представлении тождественно 1 и оставлен только ради
побитовой сверки с панелью (`smoke`, `test_etx.py`).

Последний токен последовательности — **query cutoff'а** (`age = 0`): обучаемый
вектор плюс проекция статик-контекста
`[depth/365, log1p(depth), n_events/365, log1p(n_events), dow_sin(T), dow_cos(T)]`.
`depth` — сколько дней истории вообще доступно на этом cutoff'е; в плотном
представлении ровно эту роль играет канал `avail` (`exp_027`).

## Время во внимании (ALiBi в календарном времени, без материализации маски)

Линейное смещение `b_h(i,j) = −Δt_ij/τ_h` для причинного внимания сводится к
смещению, зависящему ТОЛЬКО от ключа: `Δt_ij = age_j − age_i`, а слагаемое
`+age_i/τ_h` одинаково для всех `j` и сокращается в softmax. Поэтому смещение
реализуется одним дополнительным измерением в `q`/`k` (`q_extra = m_h`,
`k_extra = −age_j/64`) вместо матрицы `(B,H,L,L)` — и flash-ядро
`scaled_dot_product_attention(is_causal=True)` остаётся доступным.
`τ_h = 64 / (scale · m_h)`, `m_h = exp(log_m_h)`, инициализация разносит головы
по шкалам 4..512 дней. Значения `τ_h` логируются каждую эпоху: расхождение голов
— фальсифицируемое предсказание `STRATEGY_13` §Evaluation.

## Причинность и утечки

* окно события: `T − 364 <= day <= T`, отбор одним `searchsorted` по глобально
  отсортированному ключу `user * 512 + day` — дней после cutoff'а в выборке не
  существует по построению;
* внимание причинное (`is_causal=True`), паддинг СПРАВА: реальные токены никогда
  не смотрят на паддинг, query стоит ровно на позиции `n_events`;
* инференс детерминирован, аугментаций нет вовсе;
* проверки — `src/test_etx.py`, включая end-to-end: любая порча панели строго
  после cutoff'а обязана оставить вход и прогноз побитово теми же.

Запуск:
  python -m src.etx build                       # событийная таблица, один раз (~2 мин)
  python -m src.etx smoke                       # быстрая проверка корректности
  python -m src.etx bench                       # шаг/с, VRAM, число параметров
  python -m src.etx fold --val 2025-10-16 --curve
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import time

import numpy as np

from src import seq
from src.config import ARTIFACTS, DATA_PROCESSED, DATA_START, SEED, VAL_FOLDS_S1
from src.report import evaluate, format_report, save_report
from src.seq import CHANNELS, N_CH_STORED, N_DAYS, N_USERS, SEQ_L, day_index
from src.tracking import save_oof
from src.validation import bias_z, calibrate, rmsle_z

T0 = time.time()

EV_X = DATA_PROCESSED / "etx_ev_x_v1.npy"
EV_DAY = DATA_PROCESSED / "etx_ev_day_v1.npy"
EV_PTR = DATA_PROCESSED / "etx_ev_ptr_v1.npy"

DAY_STRIDE = 512                      # > N_DAYS: ключ `user * STRIDE + day` глобально сортирован
N_TOK_FEAT = N_CH_STORED + 8          # 14 каналов + 6 временных + 2 календарных
N_STATIC = 6                          # контекст query-токена
TAU_UNIT = 64.0                       # единица нормировки age в ALiBi-смещении

LOG365 = math.log1p(365.0)


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


# =============================================================================== таблица событий
def build_events(force: bool = False) -> None:
    """Событийная таблица из плотной панели: (значения, день, границы пользователей).

    Событие = день, в который у пользователя есть строка лога (`present > 0`).
    Порядок строк — по пользователю, внутри пользователя по дню (`np.nonzero`
    отдаёт индексы в C-порядке), поэтому ключ `user * DAY_STRIDE + day`
    глобально возрастает и окно любого cutoff'а берётся двумя `searchsorted`.
    """
    if EV_X.exists() and not force:
        log(f"событийная таблица уже собрана: {EV_X}")
        return
    p, _, _, _ = seq.panel()
    pi = CHANNELS.index("present")
    step = 20_000

    cnt = np.empty(N_USERS, np.int64)
    for i in range(0, N_USERS, step):
        cnt[i:i + step] = (p[i:i + step, :, pi] > 0).sum(1)
    ptr = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int64)
    n_ev = int(ptr[-1])
    log(f"событий {n_ev:,} из {N_USERS * N_DAYS:,} ячеек "
        f"(плотность {n_ev / (N_USERS * N_DAYS):.3f}); "
        f"таблица {n_ev * N_CH_STORED * 2 / 1e9:.2f} ГБ")
    assert cnt.min() > 0, "есть пользователь без единого события — панель битая"

    x = np.lib.format.open_memmap(EV_X, mode="w+", dtype=np.float16,
                                  shape=(n_ev, N_CH_STORED))
    day = np.empty(n_ev, np.int16)
    for i in range(0, N_USERS, step):
        sub = p[i:i + step]
        r, c = np.nonzero(sub[:, :, pi] > 0)
        lo, hi = int(ptr[i]), int(ptr[min(i + step, N_USERS)])
        assert hi - lo == len(r)
        x[lo:hi] = sub[r, c, :]
        day[lo:hi] = c.astype(np.int16)
    x.flush()
    np.save(EV_DAY, day)
    np.save(EV_PTR, ptr)
    log(f"записано: {EV_X.name}, {EV_DAY.name}, {EV_PTR.name}")


_E: dict = {}


def events():
    """(x fp16 (n_ev, 14), day int16 (n_ev,), key int64 (n_ev,), ptr int64 (N_USERS+1))."""
    if "x" not in _E:
        assert EV_X.exists(), "сначала: python -m src.etx build"
        _E["x"] = np.load(EV_X)
        _E["day"] = np.load(EV_DAY)
        _E["ptr"] = np.load(EV_PTR)
        uidx = np.repeat(np.arange(N_USERS, dtype=np.int64), np.diff(_E["ptr"]))
        _E["key"] = uidx * DAY_STRIDE + _E["day"].astype(np.int64)
        assert bool((np.diff(_E["key"]) > 0).all()), "ключ событий не строго возрастает"
    return _E["x"], _E["day"], _E["key"], _E["ptr"]


def select(T: dt.date, rows: np.ndarray, n_tok: int, depth_clip: int | None = None):
    """Индексы последних `n_tok` событий окна `[T − 364 .. T]` для каждой строки панели.

    Возвращает `(idx (n, n_tok) int32, cnt (n,) int32)`: `idx[i, :cnt[i]]` — глобальные
    позиции событий в таблице от СТАРОГО к СВЕЖЕМУ, остальное — паддинг (значение 0,
    маскируется по `cnt`). Событий после `T` не существует по построению: верхняя
    граница поиска — ключ `user * DAY_STRIDE + day(T) + 1`.

    `depth_clip` — та же боевая страховка глубины, что у `seq.gather`: история
    ограничивается `D` последними днями. На фолдах не используется, нужна тесту
    и будущей тестовой модели (`exp_027`: на тесте всегда `--depth-clip 289`).
    """
    _, _, key, _ = events()
    d = day_index(T)
    lo = max(0, d - SEQ_L + 1)
    if depth_clip is not None:
        lo = max(lo, d + 1 - depth_clip)
    r = np.asarray(rows, np.int64)
    start = np.searchsorted(key, r * DAY_STRIDE + lo, side="left")
    end = np.searchsorted(key, r * DAY_STRIDE + d + 1, side="left")
    cnt = np.minimum(end - start, n_tok)
    j = np.arange(n_tok, dtype=np.int64)[None, :]
    idx = np.where(j < cnt[:, None], (end - cnt)[:, None] + j, 0)
    return idx.astype(np.int32), cnt.astype(np.int32)


# =============================================================================== выборка
class Batcher:
    """Батчи из нескольких cutoff'ов сразу; на CPU считаются ТОЛЬКО индексы.

    Как и в `seq.Batcher`, батч собирается из `batch // chunk` чанков РАЗНЫХ
    cutoff'ов — градиент перемешан по времени. Отличие в цене: здесь на CPU
    остаются два `searchsorted` на чанк, а признаки токенов собираются на GPU из
    резидентной таблицы событий, поэтому фоновые потоки не нужны вовсе.
    """

    def __init__(self, cuts, ci, ri, y, batch: int, chunk: int, n_tok: int, rng):
        self.cuts, self.ci, self.ri, self.y = cuts, ci, ri, y
        self.batch, self.chunk, self.n_tok, self.rng = batch, chunk, n_tok, rng
        self.per = max(batch // chunk, 1)

    def _plan(self):
        chunks = []
        for k in range(len(self.cuts)):
            idx = np.flatnonzero(self.ci == k)
            self.rng.shuffle(idx)
            chunks += [(k, idx[i:i + self.chunk]) for i in range(0, len(idx), self.chunk)]
        chunks = [chunks[i] for i in self.rng.permutation(len(chunks))]
        return [chunks[i:i + self.per] for i in range(0, len(chunks), self.per)]

    def __iter__(self):
        for group in self._plan():
            ids, cns, cds = [], [], []
            for k, sel in group:
                T = self.cuts[k]
                a, c = select(T, self.ri[sel], self.n_tok)
                ids.append(a)
                cns.append(c)
                cds.append(np.full(len(sel), day_index(T), np.int32))
            sel = np.concatenate([s for _, s in group])
            yield (np.concatenate(ids), np.concatenate(cns), np.concatenate(cds),
                   self.y[sel])

    def n_batches(self):
        tot = sum(math.ceil(int((self.ci == k).sum()) / self.chunk)
                  for k in range(len(self.cuts)))
        return math.ceil(tot / self.per)


# =============================================================================== токены на GPU
class Tokenizer:
    """Резидентная на GPU событийная таблица + сборка признаков токенов.

    Через PCIe едут только индексы (`int32`), а не признаки: таблица событий
    (30.6 млн x 14 fp16 = 0.86 ГБ) лежит на устройстве целиком. Это на порядок
    дешевле, чем гнать плотное окно, и именно поэтому событийное представление
    на 8 ГБ VRAM вообще помещается в бюджет гейта.
    """

    def __init__(self, dev):
        import torch
        x, day, _, _ = events()
        self.dev = dev
        self.x = torch.from_numpy(x).to(dev)                       # (n_ev, 14) fp16
        self.day = torch.from_numpy(day.astype(np.int32)).to(dev)  # (n_ev,)
        self.scale = torch.from_numpy(seq.panel()[3]).to(dev).view(1, 1, N_CH_STORED)
        self.dow0 = DATA_START.weekday()
        # `depth_cap` — потолок СТАТИК-признака глубины query-токена. None = как было:
        # глубина берётся из календаря (`cut_day + 1`, потолок 365). Смысл появляется
        # ровно тогда, когда окно урезано `depth_clip`: у TCN обрезка автоматически
        # гасит канал `avail` на срезанных днях (`seq.gather`), а здесь статик
        # остался бы календарным, и модель получила бы ПАРУ (окно D, глубина 365),
        # которой в обучении не существует. См. EXP-037 и `predict(depth_static=...)`.
        self.depth_cap = None
        # `cdow_shift` — ДИАГНОСТИКА (EXP-037): сдвиг дня недели cutoff'а в статике
        # query-токена. Все обучающие cutoff'ы проекта — четверги, тестовый —
        # пятница, то есть sin/cos(cdow) в обучении КОНСТАНТЫ и их вес ничем не
        # закреплён. Ноль = как было.
        self.cdow_shift = 0.0

    def __call__(self, idx, cnt, cut_day):
        """(B, K, 22) признаки событий, (B, 6) статик query, (B, K) age, (B,) n."""
        import torch
        i = idx.long()
        ch = self.x[i].float() * self.scale                        # (B,K,14)
        day = self.day[i].float()
        age = (cut_day.float().unsqueeze(1) - day).clamp_min(0.0)  # (B,K)
        gap = torch.zeros_like(age)
        gap[:, 1:] = (day[:, 1:] - day[:, :-1]).clamp_min(0.0)
        dow = (day + float(self.dow0)) % 7.0
        f = torch.stack([
            torch.log1p(age) / LOG365,
            age / 365.0,
            torch.exp(-age / 7.0),
            torch.exp(-age / 30.0),
            torch.exp(-age / 90.0),
            torch.log1p(gap) / LOG365,
            torch.sin(2 * math.pi * dow / 7.0),
            torch.cos(2 * math.pi * dow / 7.0),
        ], dim=-1)
        tok = torch.cat([ch, f], dim=-1)                           # (B,K,22)

        cap = SEQ_L if self.depth_cap is None else min(SEQ_L, self.depth_cap)
        depth = (cut_day.float() + 1.0).clamp(max=float(cap))
        cdow = (cut_day.float() + float(self.dow0) + float(self.cdow_shift)) % 7.0
        nf = cnt.float()
        static = torch.stack([
            depth / 365.0, torch.log1p(depth) / LOG365,
            nf / 365.0, torch.log1p(nf) / LOG365,
            torch.sin(2 * math.pi * cdow / 7.0), torch.cos(2 * math.pi * cdow / 7.0),
        ], dim=-1)                                                 # (B,6)
        return tok, static, age, cnt.long()


# =============================================================================== модель
def build_model(cfg):
    import torch
    from torch import nn
    from torch.nn import functional as F

    class Block(nn.Module):
        """Pre-LN: причинное внимание с ALiBi в календарном времени -> GEGLU FFN."""

        def __init__(self, d, heads, dh, ffn, p, tau_init):
            super().__init__()
            self.h, self.dh, self.dqk = heads, dh, dh - 1
            self.scale = dh ** -0.5
            self.n1 = nn.LayerNorm(d)
            self.q = nn.Linear(d, heads * self.dqk, bias=False)
            self.k = nn.Linear(d, heads * self.dqk, bias=False)
            self.v = nn.Linear(d, heads * dh, bias=False)
            self.o = nn.Linear(heads * dh, d, bias=False)
            # b_h(i,j) = -Δt/τ_h реализовано лишним измерением q/k (см. docstring
            # модуля): логит получает scale * m_h * (-age_j/TAU_UNIT), то есть
            # τ_h = TAU_UNIT / (scale * m_h).
            m0 = TAU_UNIT / (self.scale * torch.as_tensor(tau_init, dtype=torch.float32))
            self.log_m = nn.Parameter(m0.log())
            self.n2 = nn.LayerNorm(d)
            self.w_in = nn.Linear(d, 2 * ffn)
            self.w_out = nn.Linear(ffn, d)
            self.drop = nn.Dropout(p)
            # обе ветви стартуют тождественным отображением: pre-LN трансформер
            # такой глубины иначе требует куда более длинного прогрева
            nn.init.zeros_(self.w_out.weight)
            nn.init.zeros_(self.w_out.bias)
            nn.init.zeros_(self.o.weight)

        def taus(self):
            return (TAU_UNIT / (self.scale * self.log_m.detach().exp())).float().cpu()

        def forward(self, h, age_n):
            B, L, _ = h.shape
            x = self.n1(h)
            q = self.q(x).view(B, L, self.h, self.dqk).transpose(1, 2)
            k = self.k(x).view(B, L, self.h, self.dqk).transpose(1, 2)
            v = self.v(x).view(B, L, self.h, self.dh).transpose(1, 2)
            m = self.log_m.exp().to(q.dtype).view(1, self.h, 1, 1).expand(B, self.h, L, 1)
            a = (-age_n).to(k.dtype).view(B, 1, L, 1).expand(B, self.h, L, 1)
            y = F.scaled_dot_product_attention(
                torch.cat([q, m], -1), torch.cat([k, a], -1), v,
                is_causal=True, scale=self.scale)
            h = h + self.drop(self.o(y.transpose(1, 2).reshape(B, L, self.h * self.dh)))
            g, u = self.w_in(self.n2(h)).chunk(2, dim=-1)
            return h + self.drop(self.w_out(F.gelu(g) * u))

    class ETX(nn.Module):
        def __init__(self, c):
            super().__init__()
            d = c["d_model"]
            self.tok = nn.Linear(N_TOK_FEAT, d)
            self.static = nn.Linear(N_STATIC, d)
            self.cls = nn.Parameter(torch.zeros(d))
            taus = np.geomspace(c["tau_lo"], c["tau_hi"], c["heads"])
            self.blocks = nn.ModuleList([
                Block(d, c["heads"], c["head_dim"], c["ffn"], c["dropout"], taus)
                for _ in range(c["blocks"])])
            self.norm = nn.LayerNorm(d)
            self.head = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(),
                                      nn.Dropout(c["dropout"]), nn.Linear(d, 1))
            nn.init.zeros_(self.head[-1].weight)
            nn.init.constant_(self.head[-1].bias, c["z0"])

        def forward(self, tok, static, age, n):
            B, K, _ = tok.shape
            d = self.cls.numel()
            ev = torch.arange(K, device=tok.device).unsqueeze(0) < n.unsqueeze(1)
            h = torch.zeros(B, K + 1, d, dtype=tok.dtype, device=tok.device)
            h[:, :K] = self.tok(tok) * ev.unsqueeze(-1)
            qtok = (self.cls + self.static(static)).unsqueeze(1)
            # query-токен встаёт РОВНО на позицию n: при причинном внимании он
            # видит все n событий и ни одного паддинга
            h = h.scatter(1, n.view(B, 1, 1).expand(B, 1, d), qtok.to(h.dtype))
            a = torch.zeros(B, K + 1, dtype=age.dtype, device=age.device)
            a[:, :K] = age * ev                                    # query и паддинг: age = 0
            a = a / TAU_UNIT
            for b in self.blocks:
                h = b(h, a)
            h = self.norm(h)
            zq = h.gather(1, n.view(B, 1, 1).expand(B, 1, d)).squeeze(1)
            zl = h.gather(1, (n - 1).clamp_min(0).view(B, 1, 1).expand(B, 1, d)).squeeze(1)
            w = ev.to(h.dtype).unsqueeze(-1)
            zm = (h[:, :K] * w).sum(1) / w.sum(1).clamp_min(1.0)
            return self.head(torch.cat([zq, zm, zl], dim=1)).squeeze(1)

    return ETX(cfg)


DEFAULT_CFG = dict(d_model=128, blocks=5, heads=8, head_dim=16, ffn=384, dropout=0.10,
                   n_tok=192, tau_lo=4.0, tau_hi=512.0,
                   batch=512, chunk=128, lr=1.5e-3, wd=1e-2, epochs=4, warmup=500,
                   seed=SEED, compile=False)
# batch 512, а не 1024 как у SEQ: при 1024 пик VRAM 7.29 ГБ на карте с 8 ГБ, и шаг
# деградирует 163 -> 1946 мс (вытеснение в системную память, замер `bench`).
# epochs 4 — как у всех прогонов SEQ (`exp_025` §конфиг), чтобы разница не читалась
# как «трансформер недоучен».

CFG_KEYS = ("d_model", "blocks", "heads", "head_dim", "ffn", "dropout", "n_tok",
            "batch", "chunk", "lr", "wd", "epochs", "warmup", "seed", "compile")


def n_params(**over) -> int:
    m = build_model(dict(DEFAULT_CFG, z0=0.0, **over))
    return sum(p.numel() for p in m.parameters())


# =============================================================================== обучение
def _device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(model, tk, T, rows, cfg, dev, depth_clip: int | None = None) -> np.ndarray:
    """z для одного cutoff'а батчами. Аугментаций нет, прогноз детерминирован."""
    import torch
    model.eval()
    out = np.empty(len(rows), np.float32)
    B = cfg["batch"]
    with torch.no_grad():
        for i in range(0, len(rows), B):
            r = rows[i:i + B]
            idx, cnt = select(T, r, cfg["n_tok"], depth_clip=depth_clip)
            cd = np.full(len(r), day_index(T), np.int32)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                tok, st, age, n = tk(torch.from_numpy(idx).to(dev),
                                     torch.from_numpy(cnt).to(dev),
                                     torch.from_numpy(cd).to(dev))
                z = model(tok, st, age, n)
            out[i:i + B] = z.float().cpu().numpy()
    model.train()
    return out


def fit_model(cuts: list[dt.date], cfg: dict, eval_fn=None):
    """Обучение на списке cutoff'ов. `eval_fn(model, tk, dev, ep, cfg)` — диагностика."""
    import torch

    dev = _device()
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ci, ri, zy = seq.build_index(cuts, blocks=1)       # train-панель 1-блочная, как у SEQ
    log(f"{len(cuts)} обучающих cutoff'ов {cuts[0]}..{cuts[-1]}, "
        f"{len(zy):,} примеров, mean z = {zy.mean():.4f}")

    cfg = dict(cfg, z0=float(zy.mean()))
    model = build_model(cfg).to(dev)
    net = torch.compile(model) if cfg.get("compile") else model
    tk = Tokenizer(dev)
    n_par = sum(p.numel() for p in model.parameters())
    log(f"  модель: d={cfg['d_model']} blocks={cfg['blocks']} heads={cfg['heads']} "
        f"ffn={cfg['ffn']} n_tok={cfg['n_tok']}, параметров {n_par:,}")

    named = dict(model.named_parameters())
    decay = [p for n, p in named.items() if p.dim() > 1]
    nodecay = [p for n, p in named.items() if p.dim() <= 1 and "log_m" not in n]
    tau = [p for n, p in named.items() if "log_m" in n]
    # Отдельная (x10) скорость для шкал внимания — требование `STRATEGY_13`:
    # иначе `τ_h` не успевают разойтись, и гипотеза осталась бы непроверенной
    # по чисто оптимизационной причине, а не по существу.
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=cfg["wd"]),
                             dict(params=nodecay, weight_decay=0.0),
                             dict(params=tau, weight_decay=0.0, lr_mult=10.0)],
                            lr=cfg["lr"], betas=(0.9, 0.98))
    mult = [g.get("lr_mult", 1.0) for g in opt.param_groups]

    rng = np.random.default_rng(cfg["seed"])
    bat = Batcher(cuts, ci, ri, zy, cfg["batch"], cfg["chunk"], cfg["n_tok"], rng)
    total = bat.n_batches() * cfg["epochs"]
    log(f"  шагов всего {total:,} ({bat.n_batches():,} на эпоху), batch={cfg['batch']}")

    pad_to = cfg["batch"] if cfg.get("compile") else 0
    step, hist = 0, []
    for ep in range(cfg["epochs"]):
        t_ep, run, seen = time.time(), None, 0
        for idx, cnt, cd, yb in bat:
            lr = cfg["lr"] * (min(1.0, (step + 1) / cfg["warmup"])
                              * 0.5 * (1 + math.cos(math.pi * min(1.0, step / total))))
            for g, mu in zip(opt.param_groups, mult):
                g["lr"] = lr * mu
            nrow = len(yb)
            ti = torch.from_numpy(idx).to(dev, non_blocking=True)
            tc = torch.from_numpy(cnt).to(dev, non_blocking=True)
            td = torch.from_numpy(cd).to(dev, non_blocking=True)
            tgt = torch.from_numpy(yb).to(dev, non_blocking=True)
            if pad_to and nrow < pad_to:
                # добивка ПОВТОРОМ первой строки: все слои поэлементны по примеру,
                # межпримерных операций нет, лишние строки срезаются до loss
                rep = pad_to - nrow
                ti = torch.cat([ti, ti[:1].expand(rep, -1)])
                tc = torch.cat([tc, tc[:1].expand(rep)])
                td = torch.cat([td, td[:1].expand(rep)])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                tok, st, age, n = tk(ti, tc, td)
                loss = torch.nn.functional.mse_loss(net(tok, st, age, n)[:nrow], tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            # накопление на GPU: `float(loss)` синхронизировал бы CPU с GPU на каждом шаге
            run = loss.detach() * nrow if run is None else run + loss.detach() * nrow
            seen += nrow
            step += 1
        run = float(run)
        el = time.time() - t_ep
        tt = np.concatenate([b.taus().numpy() for b in model.blocks])
        msg = (f"  эпоха {ep + 1}/{cfg['epochs']}: train MSE {run / seen:.5f} "
               f"[{el:.0f}s, {seen / el:,.0f} примеров/с]  "
               f"tau {tt.min():.1f}..{tt.max():.1f}д (медиана {np.median(tt):.1f})")
        r = eval_fn(model, tk, dev, ep, cfg) if eval_fn is not None else None
        if r:
            msg += "  " + r.pop("_msg", "")
            hist.append(dict(epoch=ep + 1, train_mse=run / seen, epoch_s=el,
                             rows_s=seen / el, tau=[round(float(v), 2) for v in tt], **r))
        log(msg)
    return model, tk, dev, cfg, hist


def save_ckpt(part: str, model, cfg: dict, V: dt.date) -> None:
    import torch
    p = ARTIFACTS / f"model_{part}.pt"
    torch.save(dict(state=model.state_dict(), cfg={k: v for k, v in cfg.items()},
                    val=V.isoformat()), p)
    log(f"веса сохранены: artifacts/{p.name}")


def load_ckpt(part: str):
    """(model, Tokenizer, cfg, дата фолда, dev) из `artifacts/model_<part>.pt`."""
    import torch
    p = ARTIFACTS / f"model_{part}.pt"
    assert p.exists(), f"нет чекпойнта {p}"
    d = torch.load(p, map_location="cpu", weights_only=False)
    dev = _device()
    model = build_model(d["cfg"]).to(dev)
    model.load_state_dict(d["state"])
    model.eval()
    return model, Tokenizer(dev), d["cfg"], dt.date.fromisoformat(d["val"]), dev


def train_fold(V: dt.date, cfg: dict, curve: bool = False, n_cutoffs: int | None = None,
               val_frac: float = 1.0, ckpt: str | None = None):
    from src.features import panel_users
    cuts = seq.fold_cutoffs(V)
    if n_cutoffs:
        cuts = cuts[-n_cutoffs:]
    uv = panel_users(V, 3)["user_id"].to_numpy()      # val-панель 3-блочная, как у SEQ
    if val_frac < 1.0:
        uv = uv[::int(round(1 / val_frac))]
    rv = seq.user_rows(uv)
    yv = seq.target_at(V, rv)
    log(f"фолд {V}: val-панель {len(uv):,} пользователей, доля y>0 = {(yv > 0).mean():.4f}")

    def ev(model, tk, dev, ep, c):
        if not (curve or ep == c["epochs"] - 1):
            return None
        z = np.maximum(predict(model, tk, V, rv, c, dev), 0.0)
        o, sc_o = calibrate(yv, z)
        return dict(rmsle=rmsle_z(yv, z), rmsle_cal=sc_o, bias=bias_z(yv, z), offset=o,
                    _msg=f"val RMSLE {rmsle_z(yv, z):.5f} -> калибр. {sc_o:.5f} "
                         f"(сдвиг {o:+.3f})")

    model, tk, dev, c, hist = fit_model(cuts, cfg, ev)
    z = np.maximum(predict(model, tk, V, rv, c, dev), 0.0)
    if ckpt:
        save_ckpt(ckpt, model, c, V)
    return uv, z, yv, hist, model, c


def train_test(cfg: dict, depth_clip: int | None, n_cutoffs: int | None = None,
               ckpt: str | None = None):
    """Тестовая модель: весь чистый коридор -> прогноз на 2026-02-13.

    Обучающие cutoff'ы — вся сетка до `CORRIDOR_END` (29 штук), как у `S1-DIST`
    и `SEQ`: отказ от свежих cutoff'ов стоил +0.0004 на LB (`exp_015`).

    **`depth_clip` здесь обязателен и по той же причине, что у TCN** (`exp_027`,
    LB +0.0051). Более того, `exp_036` намерил, что ETX опирается на длинную
    историю СИЛЬНЕЕ TCN (обрезка до 180 дней стоит +0.01259 против +0.00841),
    то есть цена экстраполяции на непрожитую глубину у него не меньше, а больше.
    На фолдах глубина 93..289 дней, на тестовом cutoff'е доступны все 365 —
    режим, которого обучение не видело ни разу. Политика: `--depth-clip 289`.
    `FULL` считается только как диагностика и в сабмит не идёт никогда.

    Веса сохраняются ВСЕГДА: пересчитать прогноз при другой политике глубины —
    это 40 секунд инференса, а переобучение модели — два часа (`exp_027` §4).
    """
    from src.config import CORRIDOR_END, CUTOFF_STEP, CUTOFF_TEST, cutoff_grid
    from src.features import panel_users
    cuts = cutoff_grid(seq.MIN_HISTORY, CUTOFF_STEP, CORRIDOR_END)
    if n_cutoffs:
        cuts = cuts[-n_cutoffs:]
    model, tk, dev, c, _ = fit_model(cuts, cfg)
    if ckpt:
        save_ckpt(ckpt, model, c, CUTOFF_TEST)
    ut = panel_users(CUTOFF_TEST, 3)["user_id"].to_numpy()
    rt = seq.user_rows(ut)
    log(f"тестовая панель {len(ut):,} пользователей, cutoff {CUTOFF_TEST}")
    out = {}
    for tag, dc in [("FULL", None), (f"D{depth_clip}", depth_clip)]:
        if dc is None and tag != "FULL":
            continue
        z = np.maximum(predict(model, tk, CUTOFF_TEST, rt, c, dev, depth_clip=dc), 0.0)
        out[tag] = z
        log(f"  глубина {tag}: mean(z) = {z.mean():.4f}, "
            f"доля нулей {float((z == 0).mean()):.4%}")
    if len(out) == 2:
        a, b = list(out.values())
        log(f"  corr(FULL, клип) = {np.corrcoef(a, b)[0, 1]:.5f}, "
            f"Var(разности) = {np.var(a - b):.5f}")
    return ut, out


# =============================================================================== CLI
def _cfg_from(a) -> dict:
    cfg = dict(DEFAULT_CFG)
    for k in CFG_KEYS:
        v = getattr(a, k, None)
        if v is not None:
            cfg[k] = v
    return cfg


def cmd_build(a):
    build_events(a.force)


def cmd_fold(a):
    import torch
    cfg = _cfg_from(a)
    V = dt.date.fromisoformat(a.val)
    exp = a.exp or f"ETX-01-S{cfg['seed']}"
    part = f"{exp}-V{V.strftime('%m%d')}"
    full = a.val_frac >= 1.0 and not a.n_cutoffs
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    uv, z, yv, hist, model, c = train_fold(
        V, cfg, curve=a.curve, n_cutoffs=a.n_cutoffs, val_frac=a.val_frac,
        ckpt=part if (full and not a.no_ckpt) else None)
    rt = time.time() - t0
    vram = (float(torch.cuda.max_memory_allocated() / 2 ** 30)
            if torch.cuda.is_available() else 0.0)
    npar = sum(p.numel() for p in model.parameters())
    if full:
        save_oof(part, uv, [V.isoformat()] * len(uv), z, yv)
        log(f"OOF сохранён: artifacts/oof_{part}.npz")
    rep = evaluate(yv, z, np.array([V.isoformat()] * len(uv)))
    print(format_report(rep))
    tt = np.concatenate([b.taus().numpy() for b in model.blocks]).tolist()
    (ARTIFACTS / f"curve_{part}.json").write_text(
        json.dumps(dict(cfg={k: v for k, v in c.items()}, val=V.isoformat(), hist=hist,
                        n_params=npar, runtime_s=rt, peak_vram_gb=vram, n_val=len(uv),
                        throughput_rows_s=(np.mean([h["rows_s"] for h in hist])
                                           if hist else None),
                        tau_final=[round(float(v), 3) for v in tt]), indent=1),
        encoding="utf-8")
    log(f"время {rt / 60:.1f} мин, пик VRAM {vram:.2f} ГБ, параметров {npar:,}")


def cmd_predict(a):
    """Тестовая модель и `artifacts/ztest_*.npy` для `src.submit`."""
    cfg = _cfg_from(a)
    ut, out = train_test(cfg, a.depth_clip, getattr(a, "n_cutoffs", None),
                         ckpt=None if getattr(a, "no_ckpt", False) else f"{a.exp}-TEST")
    np.save(ARTIFACTS / f"uid_{a.exp}.npy", ut)
    for tag, z in out.items():
        name = a.exp if tag != "FULL" else f"{a.exp}-FULL"
        np.save(ARTIFACTS / f"ztest_{name}.npy", z.astype(np.float64))
        np.save(ARTIFACTS / f"uid_{name}.npy", ut)
        log(f"сохранено: artifacts/ztest_{name}.npy (mean z {z.mean():.4f})")


def cmd_merge(a):
    from src.merge_oof import auc_positive, load_parts, merge_arrays
    parts = [f"{a.exp}-V{d.strftime('%m%d')}" for d in VAL_FOLDS_S1]
    uid, cut, z, y = load_parts(parts)
    rep = merge_arrays(uid, cut, z, y)
    print(format_report(rep))
    print(f"  AUC(1[y>0]) = {auc_positive(y, z):.5f}")
    save_oof(a.exp, uid, cut, z, y)
    save_report(a.exp, rep, extra=dict(description=a.desc, parts=parts))


def cmd_depth(a):
    """Чувствительность прогноза к глубине истории и к лимиту токенов — БЕЗ переобучения.

    Две разные оси, которые легко спутать:

    * `depth_clip` — сколько ДНЕЙ истории видно. Тот же смысл, что у
      `seq.py depth` (`exp_027`): на тесте глубина 365, а обучающие cutoff'ы
      живут на 93..254, и цена экстраполяции измеряется здесь.
    * `n_tok` — сколько СОБЫТИЙ помещается в окно. В плотном представлении такой
      оси нет вовсе; у 8.5% пользователей панели фолда событий больше 192, и надо
      знать, сколько стоит их обрезка. Веса не меняются: лимит — свойство входа.

    Обе оси считаются на сохранённом чекпойнте, поэтому стоят минуты инференса.
    """
    model, tk, cfg, V, dev = load_ckpt(a.ckpt)
    from src.features import panel_users
    uv = panel_users(V, 3)["user_id"].to_numpy()
    rv = seq.user_rows(uv)
    yv = seq.target_at(V, rv)
    z0 = np.maximum(predict(model, tk, V, rv, cfg, dev), 0.0)
    _, cal0 = calibrate(yv, z0)
    rows = [dict(axis="full", value=SEQ_L, rmsle_cal=cal0, d_cal=0.0,
                 var_vs_full=0.0, corr_vs_full=1.0, mean_z=float(z0.mean()))]
    log(f"{a.ckpt}: полная глубина {seq.day_index(V) + 1} дней, "
        f"калибр. RMSLE {cal0:.5f}")
    for D in a.depths:
        z = np.maximum(predict(model, tk, V, rv, cfg, dev, depth_clip=D), 0.0)
        _, cal = calibrate(yv, z)
        rows.append(dict(axis="depth_clip", value=D, rmsle_cal=cal, d_cal=cal - cal0,
                         var_vs_full=float(np.var(z - z0)),
                         corr_vs_full=float(np.corrcoef(z, z0)[0, 1]),
                         mean_z=float(z.mean())))
        log(f"  глубина {D:>3}д: {cal:.5f} ({cal - cal0:+.5f}), "
            f"Var(Δ)={np.var(z - z0):.5f}")
    for K in a.tokens:
        z = np.maximum(predict(model, tk, V, rv, dict(cfg, n_tok=K), dev), 0.0)
        _, cal = calibrate(yv, z)
        rows.append(dict(axis="n_tok", value=K, rmsle_cal=cal, d_cal=cal - cal0,
                         var_vs_full=float(np.var(z - z0)),
                         corr_vs_full=float(np.corrcoef(z, z0)[0, 1]),
                         mean_z=float(z.mean())))
        log(f"  лимит {K:>3} токенов: {cal:.5f} ({cal - cal0:+.5f}), "
            f"Var(Δ)={np.var(z - z0):.5f}")
    import polars as pl
    p = ARTIFACTS / f"etx_depth_{a.ckpt}.csv"
    pl.DataFrame(rows).write_csv(p)
    log(f"записано: {p}")


def cmd_bench(a):
    """Мс/шаг, примеров/с, пик VRAM и число параметров — без полного обучения."""
    import torch
    cfg = dict(_cfg_from(a), z0=2.7)
    dev = _device()
    torch.manual_seed(cfg["seed"])
    model = build_model(cfg).to(dev)
    net = torch.compile(model) if cfg.get("compile") else model
    tk = Tokenizer(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    V = dt.date.fromisoformat(a.val)
    cuts = seq.fold_cutoffs(V)[-a.n_cutoffs:]
    ci, ri, zy = seq.build_index(cuts, blocks=1)
    bat = Batcher(cuts, ci, ri, zy, cfg["batch"], cfg["chunk"], cfg["n_tok"],
                  np.random.default_rng(cfg["seed"]))
    print(f"параметров {sum(p.numel() for p in model.parameters()):,}, "
          f"n_tok={cfg['n_tok']}, batch={cfg['batch']}, "
          f"{len(zy):,} примеров на {a.n_cutoffs} cutoff'ах")
    torch.cuda.reset_peak_memory_stats()
    it, t0, done = iter(bat), None, 0
    for i in range(a.iters + 5):
        idx, cnt, cd, yb = next(it)
        ti, tc, td = (torch.from_numpy(v).to(dev) for v in (idx, cnt, cd))
        tgt = torch.from_numpy(yb).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            tok, st, age, n = tk(ti, tc, td)
            loss = torch.nn.functional.mse_loss(net(tok, st, age, n)[:len(yb)], tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if i == 4:
            torch.cuda.synchronize()
            t0 = time.time()
        elif i > 4:
            done += len(yb)
    torch.cuda.synchronize()
    el = time.time() - t0
    rate = done / el
    print(f"{a.iters} шагов за {el:.1f}с: {el / a.iters * 1000:.1f} мс/шаг, "
          f"{rate:,.0f} примеров/с, пик VRAM "
          f"{torch.cuda.max_memory_allocated() / 2 ** 30:.2f} ГБ")
    full = sum(len(seq.build_index([T], blocks=1)[2]) for T in seq.fold_cutoffs(V))
    print(f"  оценка полной эпохи фолда {V}: {full:,} примеров -> {full / rate / 60:.1f} мин")


def cmd_smoke(a):
    """Быстрая проверка корректности представления и одного forward."""
    import torch
    V = dt.date(2025, 10, 16)
    p, _, _, _ = seq.panel()
    x, day, _, _ = events()
    rows = np.array([0, 1, 7, 12345, 249999], np.int64)
    idx, cnt = select(V, rows, 192)
    d = day_index(V)
    lo = max(0, d - SEQ_L + 1)
    print(f"cutoff {V} (день {d}), событий в окне: {cnt.tolist()}")
    for i, r in enumerate(rows):
        dd = day[idx[i, :cnt[i]]]
        assert (dd <= d).all(), "событие ПОСЛЕ cutoff'а попало во вход"
        assert (dd >= lo).all(), "событие вне окна 365 дней"
        assert (np.diff(dd) > 0).all(), "дни не строго возрастают"
        ref = np.flatnonzero(p[r, lo:d + 1, CHANNELS.index("present")] > 0) + lo
        assert np.array_equal(dd, ref[-192:]), "набор событий не совпал с плотной панелью"
        assert np.allclose(x[idx[i, :cnt[i]]].astype(np.float32),
                           p[r, dd, :].astype(np.float32)), "признаки токенов не совпали"
    print("представление: события, окно, порядок и признаки совпадают с панелью — ОК")

    cfg = dict(DEFAULT_CFG, z0=2.7)
    dev = _device()
    model = build_model(cfg).to(dev)
    print(f"параметров {sum(q.numel() for q in model.parameters()):,}")
    tk = Tokenizer(dev)
    ti, tc, td = (torch.from_numpy(v).to(dev) for v in
                  (idx, cnt, np.full(len(rows), d, np.int32)))
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                         enabled=dev.type == "cuda"):
        tok, st, age, n = tk(ti, tc, td)
        z = model(tok, st, age, n)
    print(f"z = {z.float().cpu().numpy().round(4).tolist()}")
    print("smoke: ОК")


def main():
    ap = argparse.ArgumentParser(prog="src.etx")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="событийная таблица из плотной панели")
    b.add_argument("--force", action="store_true")
    b.set_defaults(fn=cmd_build)

    s = sub.add_parser("smoke", help="проверка представления и одного forward")
    s.set_defaults(fn=cmd_smoke)

    f = sub.add_parser("fold", help="обучить один фолд")
    f.add_argument("--val", required=True)
    f.add_argument("--exp", default=None)
    f.add_argument("--curve", action="store_true")
    f.add_argument("--n-cutoffs", type=int, default=None)
    f.add_argument("--val-frac", type=float, default=1.0)
    f.add_argument("--no-ckpt", action="store_true")
    f.set_defaults(fn=cmd_fold)

    p = sub.add_parser("predict", help="тестовая модель на всём коридоре -> 2026-02-13")
    p.add_argument("--exp", default="ETX-01-S42")
    p.add_argument("--depth-clip", type=int, default=289,
                   dest="depth_clip", help="боевая политика глубины (`exp_027`/`exp_036`)")
    p.add_argument("--n-cutoffs", type=int, default=None, help="только последние N (отладка)")
    p.add_argument("--no-ckpt", action="store_true")
    p.set_defaults(fn=cmd_predict)

    m = sub.add_parser("merge", help="склеить фолды в общий OOF")
    m.add_argument("--exp", default="ETX-01-S42")
    m.add_argument("--desc", default="ETX-01: sparse event transformer")
    m.set_defaults(fn=cmd_merge)

    d = sub.add_parser("depth", help="глубина истории и лимит токенов на готовом чекпойнте")
    d.add_argument("--ckpt", required=True, help="напр. ETX-01-S42-V1016")
    d.add_argument("--depths", type=int, nargs="*", default=[90, 120, 150, 180, 220, 254, 289])
    d.add_argument("--tokens", type=int, nargs="*", default=[64, 96, 128, 160])
    d.set_defaults(fn=cmd_depth)

    n = sub.add_parser("bench", help="скорость шага и VRAM")
    n.add_argument("--val", default="2025-10-16")
    n.add_argument("--n-cutoffs", type=int, default=3)
    n.add_argument("--iters", type=int, default=20)
    n.set_defaults(fn=cmd_bench)

    for sp in (f, n, p):
        for k, t in (("d_model", int), ("blocks", int), ("heads", int), ("head_dim", int),
                     ("ffn", int), ("dropout", float), ("n_tok", int), ("batch", int),
                     ("chunk", int), ("lr", float), ("wd", float), ("epochs", int),
                     ("warmup", int), ("seed", int)):
            sp.add_argument(f"--{k.replace('_', '-')}", dest=k, type=t, default=None)
        sp.add_argument("--compile", action="store_true", default=None)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
