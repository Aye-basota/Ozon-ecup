"""Единая конфигурация проекта (Strategy 1).

Здесь и только здесь: seed, пути, cutoff-даты валидации, гиперпараметры по умолчанию.

Обоснование значений — research/strategy_1.md и research/eda_findings.md:
  * панель теста отобрана правилом «активен в каждом из 3 последних 30-дневных блоков»
    (eda_findings §3.1) -> PANEL_BLOCKS = 3;
  * гарантированное окно активности [2025-11-16..2026-02-13] отравляет любой cutoff
    T >= 2025-10-17 (eda_findings §3.2) -> CORRIDOR_END = 2025-10-16;
  * глубина истории на cutoff'е меняется с 89 до 409 дней -> усечение до HISTORY_L
    на КАЖДОМ cutoff'е, включая тестовый (eda_findings §7.4).
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

SEED = 42

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
SUBMISSIONS = ROOT / "submissions"
EXPERIMENTS = ROOT / "experiments"
ARTIFACTS = ROOT / "artifacts"          # oof / preds / model dumps (gitignored)

RAW_PARQUET = DATA_RAW / "train.parquet"
SAMPLE_SUBMIT = DATA_RAW / "sample_submit.csv"

# --- временные границы датасета -------------------------------------------------
DATA_START = dt.date(2025, 1, 1)
DATA_END = dt.date(2026, 2, 13)          # он же тестовый cutoff
CUTOFF_TEST = DATA_END

TARGET_DAYS = 30
PANEL_BLOCKS = 3                          # правило отбора панели (eda §3.1)

# --- коридор чистых cutoff'ов ---------------------------------------------------
# T >= DATA_START + 30*PANEL_BLOCKS  -> панель строится
# T <= 2025-10-16                    -> target-окно (T, T+30] не пересекает
#                                       гарантированное окно [2025-11-16..2026-02-13]
CORRIDOR_END = dt.date(2025, 10, 16)
CUTOFF_STEP = 7

# --- глубина истории ------------------------------------------------------------
# Усечение до фиксированного L на каждом cutoff'е убирает артефакт «глубина истории».
# Нижняя граница коридора при усечении: T >= DATA_START + L (иначе окно неполное).
HISTORY_L = 180
WINDOWS_BY_L = {90: [7, 14, 30, 60, 90], 180: [7, 14, 30, 60, 90, 180],
                270: [7, 14, 30, 60, 90, 180, 270], None: [7, 14, 30, 60, 90, 180, 365]}

# --- валидация ------------------------------------------------------------------
# S1 (rolling out-of-time): 4 фолда в конце коридора, шаг 14 дней.
VAL_FOLDS_S1 = [dt.date(2025, 9, 4), dt.date(2025, 9, 18), dt.date(2025, 10, 2),
                dt.date(2025, 10, 16)]
# S2 (max-gap): разрыв train->val, сопоставимый с тестовым (120 дней).
S2_TRAIN_END = dt.date(2025, 6, 15)
S2_VAL = dt.date(2025, 10, 16)
# S3 (сезонный, календарный аналог теста): 1-блочная панель.
S3_VAL = dt.date(2025, 2, 13)

CUTOFF_VAL = VAL_FOLDS_S1[-1]

# --- модель ---------------------------------------------------------------------
LGB_PARAMS = dict(
    objective="regression", metric="rmse", learning_rate=0.05, num_leaves=127,
    min_data_in_leaf=200, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
    lambda_l2=5.0, verbose=-1, seed=SEED, num_threads=12, max_bin=63,
    force_row_wise=True,
)
LGB_ROUNDS = 600

# --- якорь уровня из sample_submit (eda §2.2, §5.3) ------------------------------
ANCHOR_M_X = 2.2421              # mean(log1p(gmv 30d)) на тестовом cutoff'е
ANCHOR_BAND = (2.28, 2.41)       # допустимый коридор mean(log1p(pred)) на сабмите


def cutoff_grid(min_history: int = HISTORY_L, step: int = CUTOFF_STEP,
                end: dt.date = CORRIDOR_END) -> list[dt.date]:
    """Сетка чистых cutoff'ов, отсчитанная НАЗАД от конца коридора.

    `min_history` — сколько дней истории обязано быть доступно на cutoff'е.
    Минимум 90 (правило панели = 3 блока по 30 дней). Если min_history == L,
    окно признаков на каждом cutoff'е полное; если меньше — на старых cutoff'ах
    окно длины L частично выходит за начало данных (компромисс «данные против
    сопоставимости», проверяется экспериментом S1-E04).
    """
    start = DATA_START + dt.timedelta(days=max(30 * PANEL_BLOCKS, min_history))
    out, T = [], end
    while T >= start:
        out.append(T)
        T -= dt.timedelta(days=step)
    return sorted(out)


for _p in (DATA_PROCESSED, SUBMISSIONS, ARTIFACTS):
    _p.mkdir(parents=True, exist_ok=True)
