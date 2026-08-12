"""Сабмит одной модели: обучение на всех train-cutoff, предсказание на test.

Запуск: python src/predict.py
Результат — файл в submissions/ (не коммитим).
"""
import numpy as np
import pandas as pd
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import SUBMISSIONS
from src.features import DEFAULT_FEATURE_SET, FEATURE_SETS, build_features
from src.train import DEFAULT_MODEL, DEFAULT_SCALE, TEST_CUTOFF, TRAIN_CUTOFFS, make_dataset, make_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-set", default=DEFAULT_FEATURE_SET, choices=sorted(FEATURE_SETS))
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=["hgbr", "lightgbm"])
    parser.add_argument("--scale", default=DEFAULT_SCALE, type=float)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main():
    """Обучить модель на полном train и сохранить сабмит."""
    args = parse_args()
    train_parts = []
    target_parts = []
    for cutoff in TRAIN_CUTOFFS:
        x_cutoff, y_cutoff = make_dataset(cutoff, feature_set=args.feature_set)
        train_parts.append(x_cutoff)
        target_parts.append(y_cutoff)
        print(f"train cutoff {cutoff}: X={x_cutoff.shape}, target_mean={y_cutoff.mean():.4f}")

    x_train = pd.concat(train_parts, axis=0)
    y_train = pd.concat(target_parts, axis=0)

    model = make_model(args.model)
    model.fit(x_train, np.log1p(y_train))

    x_test = build_features(TEST_CUTOFF, feature_set=args.feature_set)
    x_test = x_test.replace([np.inf, -np.inf], 0).fillna(0)
    x_test = x_test.astype("float32")
    predict = np.expm1(model.predict(x_test))
    predict = predict * args.scale
    predict = np.clip(predict, 0, None)

    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    output_name = args.output or f"{args.feature_set}_{args.model}_scale_{args.scale:g}.csv"
    out_path = SUBMISSIONS / output_name
    submission = pd.DataFrame({"user_id": x_test.index, "predict": predict})
    submission.to_csv(out_path, index=False)
    print(f"saved {out_path} rows={len(submission)}")


if __name__ == "__main__":
    main()
