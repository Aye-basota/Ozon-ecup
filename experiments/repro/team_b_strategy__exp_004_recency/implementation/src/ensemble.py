"""Ансамбль моделей: объединение предсказаний в финальный сабмит.

Запуск: python src/ensemble.py
Владелец — B2. Веса фиксируются в эксперименте (experiments/exp_NNN).
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
from src.features import build_features
from src.train import TEST_CUTOFF, TRAIN_CUTOFFS, make_dataset, make_model

ENSEMBLE_CONFIG = [
    {"feature_set": "recency", "model": "lightgbm", "scale": 0.64, "weight": 0.5},
    {"feature_set": "long_buy", "model": "lightgbm", "scale": 0.62, "weight": 0.5},
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recency-weight", type=float, default=0.5)
    parser.add_argument("--global-scale", type=float, default=1.0)
    parser.add_argument("--output", default="exp_009_recency_long_buy_lgbm_logens.csv")
    return parser.parse_args()


def main():
    """Загрузить предсказания моделей, объединить, сохранить сабмит."""
    args = parse_args()
    config = [
        {**ENSEMBLE_CONFIG[0], "weight": args.recency_weight},
        {**ENSEMBLE_CONFIG[1], "weight": 1.0 - args.recency_weight},
    ]
    pred_parts = []
    total_weight = sum(item["weight"] for item in config)

    for item in config:
        train_parts = []
        target_parts = []
        for cutoff in TRAIN_CUTOFFS:
            x_cutoff, y_cutoff = make_dataset(cutoff, feature_set=item["feature_set"])
            train_parts.append(x_cutoff)
            target_parts.append(y_cutoff)
            print(
                f"{item['feature_set']} train cutoff {cutoff}: "
                f"X={x_cutoff.shape}, target_mean={y_cutoff.mean():.4f}"
            )

        x_train = pd.concat(train_parts, axis=0)
        y_train = pd.concat(target_parts, axis=0)
        model = make_model(item["model"])
        model.fit(x_train, np.log1p(y_train))

        x_test = build_features(TEST_CUTOFF, feature_set=item["feature_set"])
        x_test = x_test.replace([np.inf, -np.inf], 0).fillna(0).astype("float32")
        predict = np.clip(np.expm1(model.predict(x_test)) * item["scale"], 0, None)
        pred_parts.append(pd.Series(np.log1p(predict), index=x_test.index) * item["weight"])

    z_pred = sum(pred_parts) / total_weight
    predict = np.clip(np.expm1(z_pred) * args.global_scale, 0, None)
    submission = pd.DataFrame({"user_id": z_pred.index, "predict": predict})

    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    out_path = SUBMISSIONS / args.output
    submission.to_csv(out_path, index=False)
    print(f"saved {out_path} rows={len(submission)}")


if __name__ == "__main__":
    main()
