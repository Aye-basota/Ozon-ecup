"""Обучение первой baseline-модели.

Запуск: python src/train.py
Печатает CV по фолдам и среднее — эти числа идут в STATE.md и experiments/.
"""
import duckdb
import gc
import numpy as np
import pandas as pd
import sys
import argparse
from datetime import date, timedelta
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

CLEAN_START = date(2025, 4, 3)
CLEAN_END = date(2025, 10, 16)

HANDOFF_WEIGHTS = {
    "recency": 0.25,
    "post_order_dist": 0.10,
    "behavior_dist": 0.20,
    "xgb_behavior": 0.25,
    "cat_behavior": 0.20,
}

HANDOFF_LEVEL = 2.370966
DIST_BINS = 16
ZERO_EDGE = 1e-9


def clean_grid(start: date = CLEAN_START, end: date = CLEAN_END, step_days: int = 7) -> list[str]:
    """Clean weekly cutoffs used for final team-b ensemble training."""
    out = []
    cur = start
    while cur <= end:
        out.append(cur.isoformat())
        cur += timedelta(days=step_days)
    return out


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


def _concat_xy(train_cutoffs: list[str], feature_set: str):
    x_parts = []
    y_parts = []
    for cutoff in train_cutoffs:
        x_cutoff, y_cutoff = make_dataset(cutoff, feature_set=feature_set)
        x_parts.append(x_cutoff)
        y_parts.append(y_cutoff)
        print(f"{feature_set} train cutoff {cutoff}: X={x_cutoff.shape}, target_mean={y_cutoff.mean():.4f}")
    x_train = pd.concat(x_parts, axis=0)
    y_train = pd.concat(y_parts, axis=0)
    del x_parts, y_parts
    gc.collect()
    return x_train, y_train


def _z_bin_edges(z: np.ndarray, n_bins: int = DIST_BINS) -> np.ndarray:
    positive = z[z > 0]
    quantiles = np.quantile(positive, np.linspace(0, 1, n_bins)[1:-1])
    return np.concatenate([[ZERO_EDGE], quantiles])


def _bin_labels(z: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges, z, side="right").astype("int32")


def _bin_centroids(z: np.ndarray, labels: np.ndarray, n_bins: int = DIST_BINS) -> np.ndarray:
    counts = np.bincount(labels, minlength=n_bins)
    totals = np.bincount(labels, weights=z, minlength=n_bins)
    centroids = np.zeros(n_bins, dtype="float64")
    last = 0.0
    for idx in range(n_bins):
        if counts[idx] > 0:
            last = totals[idx] / counts[idx]
        centroids[idx] = last
    return centroids


def _make_lgbm_regressor():
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


def _make_lgbm_dist():
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        objective="multiclass",
        num_class=DIST_BINS,
        n_estimators=250,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=0.05,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )


def _make_xgb_regressor():
    from xgboost import XGBRegressor

    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=450,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=20.0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        random_state=SEED,
        n_jobs=-1,
        tree_method="hist",
        verbosity=1,
    )


def _make_catboost_regressor():
    from catboost import CatBoostRegressor

    return CatBoostRegressor(
        loss_function="RMSE",
        iterations=500,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=SEED,
        task_type="CPU",
        thread_count=-1,
        allow_writing_files=False,
        verbose=100,
    )


def _fit_regressor(train_cutoffs: list[str], feature_set: str, scale: float, factory):
    x_train, y_train = _concat_xy(train_cutoffs, feature_set)
    features = list(x_train.columns)
    model = factory()
    model.fit(x_train, np.log1p(y_train))
    del x_train, y_train
    gc.collect()
    return {"kind": "regressor", "model": model, "feature_set": feature_set, "features": features, "scale": scale}


def _fit_dist(train_cutoffs: list[str], feature_set: str, scale: float):
    x_train, y_train = _concat_xy(train_cutoffs, feature_set)
    features = list(x_train.columns)
    z_train = np.log1p(y_train.to_numpy())
    labels = _bin_labels(z_train, _z_bin_edges(z_train))
    centroids = _bin_centroids(z_train, labels)
    print(
        f"dist {feature_set}: bins={DIST_BINS}, positive_rate={(z_train > 0).mean():.4f}, "
        f"centroids[0]={centroids[0]:.4f}, centroids[-1]={centroids[-1]:.4f}"
    )
    model = _make_lgbm_dist()
    model.fit(x_train, labels)
    del x_train, y_train, z_train, labels
    gc.collect()
    return {
        "kind": "dist",
        "model": model,
        "centroids": centroids,
        "feature_set": feature_set,
        "features": features,
        "scale": scale,
    }


def train_models(df_raw=None, cutoff_date: str | None = None, target_days: int = TARGET_DAYS,
                 train_cutoffs: list[str] | None = None, recent_train_cutoffs: int = 8):
    """Train the final team-b handoff ensemble.

    `df_raw`, `cutoff_date` and `target_days` are accepted for API symmetry with
    team-b-B2. The current feature pipeline reads `data/raw/train.parquet`
    internally, so callers can simply use `train_models()`.
    """
    if target_days != TARGET_DAYS:
        raise ValueError(f"Only TARGET_DAYS={TARGET_DAYS} is supported")
    if train_cutoffs is None:
        train_cutoffs = clean_grid()[-recent_train_cutoffs:]
    print(f"handoff train_cutoffs={len(train_cutoffs)} {train_cutoffs[0]}..{train_cutoffs[-1]}")

    models = {
        "recency": _fit_regressor(train_cutoffs, "recency", 0.64, _make_lgbm_regressor),
        "post_order_dist": _fit_dist(train_cutoffs, "long_buy_post_order", 0.62),
        "behavior_dist": _fit_dist(train_cutoffs, "behavior_v1", 0.62),
        "xgb_behavior": _fit_regressor(train_cutoffs, "behavior_v1", 0.62, _make_xgb_regressor),
        "cat_behavior": _fit_regressor(train_cutoffs, "behavior_v1", 0.62, _make_catboost_regressor),
    }
    meta = {
        "weights": HANDOFF_WEIGHTS.copy(),
        "level": HANDOFF_LEVEL,
        "train_cutoffs": list(train_cutoffs),
    }
    return models, meta


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
