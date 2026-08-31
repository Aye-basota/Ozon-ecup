"""ETX2 — общие куски этапов 2–5: сегменты, число событий, тестовая сторона.

Один модуль на все скрипты `ETX2/`, чтобы определение сегмента было ФИЗИЧЕСКИ
одним и тем же на OOF и на тесте: расхождение определений — ровно тот класс
ошибки, из-за которого сегментный гейт может выглядеть работающим на фолдах и
разъехаться на тестовом cutoff'е.

Признаки гейта — только cutoff-safe и только уже существующие (`w180_days_buy`,
`rec_buy`, число событийных дней в окне), из кэша `feat_<T>_LNone.parquet`,
который считает `build_features`. Ничего нового не строится.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from src.config import ARTIFACTS, CUTOFF_TEST, DATA_PROCESSED

SEG_COLS = ["rec_buy", "w180_days_buy"]
DEPTH_TEST = 289          # боевая политика глубины (`exp_027`)


def feats(T: dt.date, uid: np.ndarray) -> pl.DataFrame:
    """`rec_buy` / `w180_days_buy` в порядке строк `uid`."""
    f = pl.read_parquet(DATA_PROCESSED / f"feat_{T.strftime('%Y%m%d')}_LNone.parquet",
                        columns=["user_id"] + SEG_COLS)
    out = pl.DataFrame({"user_id": np.asarray(uid)}).join(f, on="user_id", how="left")
    assert out.height == len(uid)
    return out


def seg3(f: pl.DataFrame) -> np.ndarray:
    """Три КРУПНЫХ сегмента по частоте покупок за 180 дней — разбиение, не пересечения.

    Границы взяты не из перебора, а из таблицы сегментов `exp_036`: там ETX
    выигрывает у трёхсидового TCN на КРАЯХ активности (`w180_days_buy` 0–1:
    −0.00293 против −0.00214; 16+: −0.00708 против −0.00619) и ПРОИГРЫВАЕТ в
    середине (2–15: −0.00241 против −0.00256). Доли 0.216 / 0.565 / 0.219.
    """
    nb = f["w180_days_buy"].to_numpy()
    s = np.full(len(nb), 1, np.int8)          # 1 = середина 2..15
    s[nb <= 1] = 0
    s[nb >= 16] = 2
    return s


def seg4(f: pl.DataFrame) -> np.ndarray:
    """seg3 с расщеплением середины по `rec_buy` 15–60 — самой проблемной полосе.

    `rec_buy` 15–60 держит 35.3% MSE проекта и это единственный сегмент, где
    трёхсидовый TCN сильнее ETX заметно (−0.00232 против −0.00169).
    """
    s = seg3(f).astype(np.int8)
    rb = f["rec_buy"].to_numpy()
    known = ~np.isnan(rb)
    mid = s == 1
    s = np.where(mid & known & (rb >= 15) & (rb <= 60), 3, s).astype(np.int8)
    return s


SEG_NAMES = {
    "seg3": ["w180_days_buy 0-1", "w180_days_buy 2-15", "w180_days_buy 16+"],
    "seg4": ["w180_days_buy 0-1", "2-15 вне rec 15-60", "w180_days_buy 16+",
             "2-15 и rec_buy 15-60"],
}


def event_counts(T: dt.date, uid: np.ndarray, depth: int | None = DEPTH_TEST) -> np.ndarray:
    """Сколько событийных ДНЕЙ у пользователя в окне `[T − depth + 1 .. T]`.

    Считается тем же ключом `user * DAY_STRIDE + day`, что и выборка токенов ETX
    (`src.etx.select`), но без ограничения `n_tok`: это ось активности, а не вход
    модели. Определение одно и то же на фолдах и на тесте.
    """
    from src import etx, seq
    from src.seq import day_index
    _, _, key, _ = etx.events()
    d = day_index(T)
    lo = max(0, d - 364)
    if depth is not None:
        lo = max(lo, d + 1 - depth)
    r = np.asarray(seq.user_rows(np.asarray(uid)), np.int64)   # СТРОКИ панели, не user_id
    start = np.searchsorted(key, r * etx.DAY_STRIDE + lo, side="left")
    end = np.searchsorted(key, r * etx.DAY_STRIDE + d + 1, side="left")
    return (end - start).astype(np.int32)


ACT_BINS = [0, 5, 15, 40, 80, 150, 10 ** 9]
ACT_NAMES = ["0-5", "5-15", "15-40", "40-80", "80-150", "150+"]


def act_bin(cnt: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(cnt, ACT_BINS[1:-1], right=True), 0, len(ACT_NAMES) - 1)


def ztest(name: str) -> np.ndarray:
    return np.load(ARTIFACTS / f"ztest_{name}.npy")


def test_uid() -> np.ndarray:
    return np.load(ARTIFACTS / "uid_S1-DIST.npy")


TEST_T = CUTOFF_TEST
