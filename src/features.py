"""Единый пайплайн фичей: build_features(cutoff_date).

Один и тот же код строит фичи для train, val и test.
Фичи считаются ТОЛЬКО на данных до cutoff — никакого лукапа.
Новые фичи — только новые колонки, чужие не переписываем.

Запуск: python src/features.py  (строит и сохраняет фичи в data/processed/)
"""
import pandas as pd

from src.config import DATA_PROCESSED, DATA_RAW


def build_features(cutoff_date: str) -> pd.DataFrame:
    """Построить фичи для всех пользователей на дату cutoff.

    Возвращает DataFrame: index — user_id, колонки — фичи.
    Таргет (сумма заказов за 30 дней после cutoff) добавляет train.py,
    здесь его нет, чтобы не было лукапа.
    """
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit("Сначала реализуйте build_features()")
