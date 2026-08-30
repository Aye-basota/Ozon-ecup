from datetime import date, timedelta
import gc

import numpy as np
import pandas as pd

from src.config import SEED, TARGET_DAYS
from src.team_features import build_features


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
    cutoffs = []
    current = start
    while current <= end:
        cutoffs.append(current.isoformat())
        current += timedelta(days=step_days)
    return cutoffs


def _format_cutoff(cutoff_date) -> str:
    return pd.to_datetime(cutoff_date).date().isoformat()


def build_target(df_raw: pd.DataFrame, cutoff_date: str) -> pd.Series:
    cutoff_date = pd.to_datetime(cutoff_date)
    event_date = pd.to_datetime(df_raw["event_date"])
    mask = (
        (event_date >= cutoff_date)
        & (event_date < cutoff_date + pd.Timedelta(days=TARGET_DAYS))
    )
    return df_raw.loc[mask].groupby("user_id")["gmv"].sum()


def make_dataset(df_raw: pd.DataFrame, cutoff_date: str, feature_set: str):
    x = build_features(df_raw, cutoff_date, feature_set=feature_set)
    y = build_target(df_raw, cutoff_date).reindex(x.index).fillna(0.0)
    x = x.replace([np.inf, -np.inf], 0).fillna(0).astype("float32")
    return x, y


def _concat_xy(df_raw: pd.DataFrame, train_cutoffs: list[str], feature_set: str):
    x_parts = []
    y_parts = []
    for cutoff in train_cutoffs:
        x_cutoff, y_cutoff = make_dataset(df_raw, cutoff, feature_set)
        x_parts.append(x_cutoff)
        y_parts.append(y_cutoff)
        print(
            f"team {feature_set} cutoff {cutoff}: X={x_cutoff.shape}, target_mean={y_cutoff.mean():.4f}",
            flush=True,
        )
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
        verbosity=0,
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
        verbose=False,
    )


def _fit_regressor(df_raw: pd.DataFrame, train_cutoffs: list[str], feature_set: str, scale: float, factory):
    x_train, y_train = _concat_xy(df_raw, train_cutoffs, feature_set)
    features = list(x_train.columns)
    model = factory()
    model.fit(x_train, np.log1p(y_train))
    del x_train, y_train
    gc.collect()
    return {
        "kind": "regressor",
        "model": model,
        "feature_set": feature_set,
        "features": features,
        "scale": scale,
    }


def _fit_dist(df_raw: pd.DataFrame, train_cutoffs: list[str], feature_set: str, scale: float):
    x_train, y_train = _concat_xy(df_raw, train_cutoffs, feature_set)
    features = list(x_train.columns)
    z_train = np.log1p(y_train.to_numpy())
    labels = _bin_labels(z_train, _z_bin_edges(z_train))
    centroids = _bin_centroids(z_train, labels)
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


def train_models(df_raw: pd.DataFrame, train_cutoff=None, recent_train_cutoffs: int = 8):
    if train_cutoff is None:
        train_cutoffs = clean_grid()[-recent_train_cutoffs:]
    else:
        train_cutoffs = [_format_cutoff(train_cutoff)]
    print(f"team train cutoffs: {train_cutoffs[0]}..{train_cutoffs[-1]}", flush=True)

    models = {
        "recency": _fit_regressor(df_raw, train_cutoffs, "recency", 0.64, _make_lgbm_regressor),
        "post_order_dist": _fit_dist(df_raw, train_cutoffs, "long_buy_post_order", 0.62),
        "behavior_dist": _fit_dist(df_raw, train_cutoffs, "behavior_v1", 0.62),
        "xgb_behavior": _fit_regressor(df_raw, train_cutoffs, "behavior_v1", 0.62, _make_xgb_regressor),
        "cat_behavior": _fit_regressor(df_raw, train_cutoffs, "behavior_v1", 0.62, _make_catboost_regressor),
    }
    meta = {
        "weights": HANDOFF_WEIGHTS.copy(),
        "level": HANDOFF_LEVEL,
        "train_cutoffs": list(train_cutoffs),
    }
    return models, meta


def _prepare_features(df_raw: pd.DataFrame, cutoff_date: str, component: dict) -> pd.DataFrame:
    x = build_features(df_raw, cutoff_date, feature_set=component["feature_set"])
    x = x.replace([np.inf, -np.inf], 0).fillna(0).astype("float32")
    return x.reindex(columns=component["features"], fill_value=0)


def _component_log_prediction(component: dict, df_raw: pd.DataFrame, cutoff_date: str) -> pd.Series:
    x = _prepare_features(df_raw, cutoff_date, component)
    if component["kind"] == "dist":
        z_raw = component["model"].predict_proba(x) @ component["centroids"]
    elif component["kind"] == "regressor":
        z_raw = component["model"].predict(x)
    else:
        raise ValueError(f"Unknown component kind: {component['kind']}")

    pred = np.clip(np.expm1(z_raw) * component["scale"], 0, None)
    return pd.Series(np.log1p(pred), index=x.index)


def predict_log(
    models: dict,
    meta: dict,
    df_raw: pd.DataFrame,
    cutoff_date: str,
    level: float | None = HANDOFF_LEVEL,
) -> pd.Series:
    weights = meta.get("weights", HANDOFF_WEIGHTS)
    z = None
    total_weight = 0.0

    for name, weight in weights.items():
        if weight <= 0:
            continue
        part = _component_log_prediction(models[name], df_raw, cutoff_date) * weight
        z = part if z is None else z.add(part, fill_value=0)
        total_weight += weight

    z = z / total_weight
    if level is not None:
        z = z + (level - float(z.mean()))
        z = pd.Series(np.maximum(z.to_numpy(), 0.0), index=z.index)
    return z


def predict_gmv(
    models: dict,
    meta: dict,
    df_raw: pd.DataFrame,
    cutoff_date: str,
    level: float | None = HANDOFF_LEVEL,
) -> pd.Series:
    z = predict_log(models, meta, df_raw, cutoff_date, level)
    return pd.Series(np.maximum(np.expm1(z.to_numpy()), 0.0), index=z.index)
