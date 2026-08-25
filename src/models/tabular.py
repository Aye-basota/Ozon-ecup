"""Reusable tabular model families retained from the historical project."""
from __future__ import annotations

import os
from typing import Any

import numpy as np

from src.settings import competition


DIST_BINS = 16
ZERO_EDGE = 1e-9


def _lightgbm():
    import lightgbm as lgb

    return lgb


def lightgbm_params(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = competition()
    params = dict(cfg["lightgbm"])
    params.pop("rounds", None)
    params.update(
        {
            "verbose": -1,
            "seed": int(cfg["seed"]),
            "num_threads": int(os.environ.get("LGB_THREADS", "12")),
        }
    )
    if overrides:
        params.update(overrides)
    return params


def _bin_edges(z: np.ndarray, n_bins: int = DIST_BINS) -> np.ndarray:
    positive = z[z > 0]
    if len(positive) == 0:
        raise ValueError("Distribution head requires positive targets")
    quantiles = np.quantile(positive, np.linspace(0, 1, n_bins)[1:-1])
    return np.concatenate([[ZERO_EDGE], quantiles])


def _bin_centroids(z: np.ndarray, labels: np.ndarray, n_bins: int = DIST_BINS) -> np.ndarray:
    counts = np.bincount(labels, minlength=n_bins)
    totals = np.bincount(labels, weights=z, minlength=n_bins)
    centroids = np.zeros(n_bins)
    last = 0.0
    for index in range(n_bins):
        if counts[index] > 0:
            last = totals[index] / counts[index]
        centroids[index] = last
    return centroids


def fit_model(
    family: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray | None = None,
    rounds: int | None = None,
    params: dict[str, Any] | None = None,
):
    cfg = competition()
    rounds = int(rounds or cfg["lightgbm"]["rounds"])
    lgb = _lightgbm()
    base_params = lightgbm_params(params)
    if family == "direct":
        dataset = lgb.Dataset(x_train, np.log1p(y_train), weight=sample_weight, params=base_params)
        return lgb.train(base_params, dataset, num_boost_round=rounds)
    if family == "two_part":
        positive = y_train > 0
        classifier_params = {**base_params, "objective": "binary", "metric": "binary_logloss"}
        classifier = lgb.train(
            classifier_params,
            lgb.Dataset(x_train, positive.astype(np.int8), weight=sample_weight, params=classifier_params),
            num_boost_round=rounds,
        )
        regressor = lgb.train(
            base_params,
            lgb.Dataset(
                x_train[positive],
                np.log1p(y_train[positive]),
                weight=None if sample_weight is None else sample_weight[positive],
                params=base_params,
            ),
            num_boost_round=rounds,
        )
        return classifier, regressor
    if family == "dist":
        z = np.log1p(y_train)
        labels = np.searchsorted(_bin_edges(z), z, side="right").astype(np.int32)
        centroids = _bin_centroids(z, labels)
        dist_params = {
            **base_params,
            "objective": "multiclass",
            "metric": "multi_logloss",
            "num_class": len(centroids),
        }
        model = lgb.train(
            dist_params,
            lgb.Dataset(x_train, labels, weight=sample_weight, params=dist_params),
            num_boost_round=rounds,
        )
        return model, centroids
    if family == "catboost":
        from catboost import CatBoostRegressor

        cat_params = {
            "loss_function": "RMSE",
            "iterations": rounds,
            "learning_rate": 0.05,
            "depth": 8,
            "l2_leaf_reg": 5.0,
            "random_seed": int(cfg["seed"]),
            "verbose": 0,
            "thread_count": int(os.environ.get("CATBOOST_THREADS", "12")),
            "border_count": 64,
            "langevin": True,
            **(params or {}),
        }
        model = CatBoostRegressor(**cat_params)
        model.fit(np.nan_to_num(x_train, nan=-999.0), np.log1p(y_train), sample_weight=sample_weight)
        return model
    raise ValueError(f"Unknown model family: {family}")


def predict_model(family: str, model, features: np.ndarray) -> np.ndarray:
    if family == "two_part":
        classifier, regressor = model
        return np.asarray(classifier.predict(features)) * np.asarray(regressor.predict(features))
    if family == "dist":
        booster, centroids = model
        return np.asarray(booster.predict(features)) @ np.asarray(centroids)
    if family == "catboost":
        return np.asarray(model.predict(np.nan_to_num(features, nan=-999.0)))
    return np.asarray(model.predict(features))
