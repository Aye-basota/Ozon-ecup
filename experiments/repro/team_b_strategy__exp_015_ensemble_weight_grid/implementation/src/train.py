"""Обучение первой baseline-модели.

Запуск: python src/train.py
Печатает CV по фолдам и среднее — эти числа идут в STATE.md и experiments/.
"""
import duckdb
import numpy as np
import pandas as pd
import sys
import argparse
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATA_RAW, SEED, TARGET_DAYS
from src.features import DEFAULT_FEATURE_SET, FEATURE_SETS, build_features

RAW_TRAIN = DATA_RAW / "train.parquet"

# В config.py cutoff пока не зафиксированы командой, поэтому baseline держит их локально.
# Валидационный cutoff выбран так, чтобы полностью видеть следующие 30 дней до 2026-02-13.
TRAIN_CUTOFFS = ["2025-10-01", "2025-11-01", "2025-12-01"]
VAL_CUTOFF = "2026-01-15"
TEST_CUTOFF = "2026-02-14"
DEFAULT_MODEL = "lightgbm"
DEFAULT_SCALE = 0.64


def build_target(cutoff_date: str) -> pd.Series:
    """Сумма gmv за TARGET_DAYS дней после cutoff, без включения самого cutoff."""
    query = f"""
        SELECT
            user_id,
            SUM(gmv) AS target
        FROM read_parquet($path)
        WHERE event_date >= CAST($cutoff AS DATE)
          AND event_date < CAST($cutoff AS DATE) + INTERVAL {TARGET_DAYS} DAY
        GROUP BY user_id
    """
    with duckdb.connect() as con:
        target = con.execute(
            query,
            {"path": str(RAW_TRAIN), "cutoff": cutoff_date},
        ).fetchdf()
    return target.set_index("user_id")["target"]


def make_dataset(cutoff_date: str, feature_set: str = DEFAULT_FEATURE_SET):
    """Фичи на дату cutoff + будущий 30-дневный target."""
    x = build_features(cutoff_date, feature_set=feature_set)
    y = build_target(cutoff_date)
    y = y.reindex(x.index).fillna(0.0)
    x = x.replace([np.inf, -np.inf], 0).fillna(0)
    x = x.astype("float32")
    return x, y


def rmsle(y_true, y_pred) -> float:
    """Официальная метрика соревнования: RMSE по log1p, отрицательные прогнозы зануляются."""
    y_pred = np.clip(y_pred, 0, None)
    return float(root_mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))


def make_model(model_name: str = "hgbr"):
    """Baseline-модель для train.py и predict.py."""
    if model_name == "hgbr":
        return HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_leaf_nodes=31,
            max_iter=250,
            l2_regularization=0.05,
            random_state=SEED,
        )
    if model_name == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            objective="regression",
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=0.05,
            random_state=SEED,
            n_jobs=-1,
            verbosity=-1,
        )
    raise ValueError(f"Неизвестная модель: {model_name}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-set", default=DEFAULT_FEATURE_SET, choices=sorted(FEATURE_SETS))
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=["hgbr", "lightgbm"])
    parser.add_argument("--scale", default=DEFAULT_SCALE, type=float)
    return parser.parse_args()


def main():
    """Построить фичи по фолдам, обучить модель, вывести CV."""
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
    x_val, y_val = make_dataset(VAL_CUTOFF, feature_set=args.feature_set)

    model = make_model(args.model)
    model.fit(x_train, np.log1p(y_train))

    pred_val = np.expm1(model.predict(x_val))
    pred_val = pred_val * args.scale
    pred_val = np.clip(pred_val, 0, None)

    rmse = root_mean_squared_error(y_val, pred_val)
    mae = mean_absolute_error(y_val, pred_val)
    log_rmse = rmsle(y_val, pred_val)

    print(f"feature_set: {args.feature_set}")
    print(f"model: {args.model}")
    print(f"scale: {args.scale}")
    print(f"validation cutoff {VAL_CUTOFF}: X={x_val.shape}, target_mean={y_val.mean():.4f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"RMSLE: {log_rmse:.6f}")


if __name__ == "__main__":
    main()
