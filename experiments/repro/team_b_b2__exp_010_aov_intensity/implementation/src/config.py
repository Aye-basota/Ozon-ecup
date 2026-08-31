"""Единая конфигурация проекта.

Здесь и только здесь: seed, пути, cutoff-даты валидации.
Менять — только по явному запросу (см. AGENTS.md).
"""
from pathlib import Path

SEED = 42

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
SUBMISSIONS = ROOT / "submissions"

# Таргет: сумма заказов за TARGET_DAYS дней после cutoff.
TARGET_DAYS = 30

# Out-of-time валидация: фичи считаются на данных до cutoff,
# таргет — за 30 дней после. Даты уточнить по данным соревнования.
CUTOFFS_TRAIN = []  # cutoff-даты для обучения, напр. ["2024-01-01", ...]
CUTOFF_VAL = None   # cutoff-дата для валидации
CUTOFF_TEST = None  # cutoff-дата для теста (сабмит)
