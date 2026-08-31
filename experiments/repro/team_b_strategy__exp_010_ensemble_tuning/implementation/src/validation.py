"""Единая схема валидации и метрика для всех.

Out-of-time: фолды по cutoff-датам из config.CUTOFFS_TRAIN + CUTOFF_VAL.
Никаких случайных сплитов. Менять — только по явному запросу.
"""
from src.config import CUTOFF_VAL, CUTOFFS_TRAIN


def get_folds():
    """Вернуть список (train_cutoffs, val_cutoff) для CV."""
    raise NotImplementedError


def metric(y_true, y_pred) -> float:
    """Метрика соревнования. Уточнить на странице e-cup."""
    raise NotImplementedError
