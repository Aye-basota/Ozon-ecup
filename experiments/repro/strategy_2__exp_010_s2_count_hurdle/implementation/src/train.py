"""Обучение одной модели с CV по схеме из validation.py.

Запуск: python src/train.py
Печатает CV по фолдам и среднее — эти числа идут в STATE.md и experiments/.
"""
from src.config import SEED, TARGET_DAYS
from src.features import build_features
from src.validation import get_folds, metric


def main():
    """Построить фичи по фолдам, обучить модель, вывести CV."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
