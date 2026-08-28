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
from src.train import (DEFAULT_MODEL, DEFAULT_SCALE, HANDOFF_LEVEL, HANDOFF_WEIGHTS,
                       TEST_CUTOFF, TRAIN_CUTOFFS, make_dataset, make_model, train_models)


def _prepare_features(cutoff_date: str, feature_set: str, features: list[str]) -> pd.DataFrame:
    x = build_features(cutoff_date, feature_set=feature_set)
    x = x.replace([np.inf, -np.inf], 0).fillna(0).astype("float32")
    return x.reindex(columns=features, fill_value=0)


def _component_log_prediction(component: dict, cutoff_date: str) -> pd.Series:
    x = _prepare_features(cutoff_date, component["feature_set"], component["features"])
    if component["kind"] == "dist":
        z_raw = component["model"].predict_proba(x) @ component["centroids"]
    elif component["kind"] == "regressor":
        z_raw = component["model"].predict(x)
    else:
        raise ValueError(f"Unknown component kind: {component['kind']}")
    pred = np.clip(np.expm1(z_raw) * component["scale"], 0, None)
    return pd.Series(np.log1p(pred), index=x.index)


def predict_log(models: dict, meta: dict | None = None, cutoff_date: str = TEST_CUTOFF,
                level: float | None = HANDOFF_LEVEL) -> pd.Series:
    """Predict final ensemble in log1p space."""
    weights = (meta or {}).get("weights", HANDOFF_WEIGHTS)
    z_parts = []
    total_weight = 0.0
    for name, weight in weights.items():
        if weight <= 0:
            continue
        z_parts.append(_component_log_prediction(models[name], cutoff_date) * weight)
        total_weight += weight
    z = sum(z_parts) / total_weight
    if level is not None:
        z = pd.Series(np.maximum(z.to_numpy() + (level - float(z.mean())), 0.0), index=z.index)
    return z


def predict_gmv(models: dict, meta: dict | None = None, cutoff_date: str = TEST_CUTOFF,
                level: float | None = HANDOFF_LEVEL) -> pd.Series:
    """Predict final ensemble in raw GMV space."""
    z = predict_log(models, meta=meta, cutoff_date=cutoff_date, level=level)
    return pd.Series(np.maximum(np.expm1(z.to_numpy()), 0.0), index=z.index)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--handoff", action="store_true", help="train final exp024 handoff ensemble")
    parser.add_argument("--feature-set", default=DEFAULT_FEATURE_SET, choices=sorted(FEATURE_SETS))
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=["hgbr", "lightgbm"])
    parser.add_argument("--scale", default=DEFAULT_SCALE, type=float)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main():
    """Обучить модель на полном train и сохранить сабмит."""
    args = parse_args()
    if args.handoff:
        models, meta = train_models()
        predict = predict_gmv(models, meta=meta, cutoff_date=TEST_CUTOFF, level=meta["level"])
        SUBMISSIONS.mkdir(parents=True, exist_ok=True)
        output_name = args.output or "exp_024_handoff_level_e19.csv"
        out_path = SUBMISSIONS / output_name
        pd.DataFrame({"user_id": predict.index, "predict": predict.to_numpy()}).to_csv(out_path, index=False)
        print(f"saved {out_path} rows={len(predict)} mean_log1p={np.log1p(predict).mean():.6f}")
        return

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
