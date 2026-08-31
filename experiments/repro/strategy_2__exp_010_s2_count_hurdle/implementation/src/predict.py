"""Сабмит одной модели: обучение на всех train-cutoff, предсказание на test.

Запуск: python src/predict.py
Результат — файл в submissions/ (не коммитим).
"""
from src.config import CUTOFF_TEST, SUBMISSIONS
from src.features import build_features


def main():
    """Обучить модель на полном train и сохранить сабмит."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
