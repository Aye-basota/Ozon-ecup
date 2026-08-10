"""Модели Strategy 1.

Метрика — MSE в переменной z = log1p(y) (eda_findings §8), поэтому L2-регрессия
на log1p(y) оптимизирует её ТОЧНО. Кастомный лосс не нужен.

Три постановки:
  direct    — LightGBM regression на log1p(y);
  two_part  — P(y>0) (binary) x E[log1p(y) | y>0]; источник декорреляции ошибок;
  catboost  — другая схема построения деревьев (oblivious), тот же таргет.
"""
from __future__ import annotations

import numpy as np

from src.config import LGB_PARAMS, LGB_ROUNDS, SEED


def _lgb():
    import lightgbm as lgb
    return lgb


def _params(params, **over):
    p = dict(LGB_PARAMS)
    if params:
        p.update(params)
    p.update(over)
    return p


def make_datasets(kind: str, Xtr, ytr, wtr=None, params: dict | None = None):
    """Заранее биннит матрицу, чтобы вызывающий мог освободить Xtr до бустинга.

    На плотной сетке cutoff'ов Xtr занимает 5+ ГБ; без этого шага он живёт
    в памяти всё обучение.
    """
    lgb = _lgb()
    p = _params(params)
    if kind == "direct":
        return [lgb.Dataset(Xtr, np.log1p(ytr), weight=wtr, params=p).construct()]
    if kind == "two_part":
        m = ytr > 0
        return [lgb.Dataset(Xtr, (ytr > 0).astype(np.int8), weight=wtr, params=p).construct(),
                lgb.Dataset(Xtr[m], np.log1p(ytr[m]),
                            weight=None if wtr is None else wtr[m], params=p).construct()]
    raise ValueError(kind)


def train_direct_ds(ds, params: dict | None = None, rounds: int = LGB_ROUNDS):
    return _lgb().train(_params(params), ds, num_boost_round=rounds)


def train_two_part_ds(dss, params: dict | None = None, rounds: int = LGB_ROUNDS):
    lgb = _lgb()
    clf = lgb.train(_params(params, objective="binary", metric="binary_logloss"), dss[0],
                    num_boost_round=rounds)
    reg = lgb.train(_params(params), dss[1], num_boost_round=rounds)
    return clf, reg


def train_direct(Xtr, ytr, wtr=None, params: dict | None = None, rounds: int = LGB_ROUNDS):
    return train_direct_ds(make_datasets("direct", Xtr, ytr, wtr, params)[0], params, rounds)


def train_two_part(Xtr, ytr, wtr=None, params: dict | None = None, rounds: int = LGB_ROUNDS):
    return train_two_part_ds(make_datasets("two_part", Xtr, ytr, wtr, params), params, rounds)


def predict_two_part(models, X) -> np.ndarray:
    clf, reg = models
    return clf.predict(X) * reg.predict(X)


def train_catboost(Xtr, ytr, wtr=None, params: dict | None = None, rounds: int = LGB_ROUNDS):
    from catboost import CatBoostRegressor
    p = dict(loss_function="RMSE", iterations=rounds, learning_rate=0.05, depth=8,
             l2_leaf_reg=5.0, random_seed=SEED, verbose=0, thread_count=12,
             border_count=64, langevin=True)
    if params:
        p.update(params)
    m = CatBoostRegressor(**p)
    m.fit(np.nan_to_num(Xtr, nan=-999.0), np.log1p(ytr), sample_weight=wtr)
    return m


def predict_catboost(model, X) -> np.ndarray:
    return model.predict(np.nan_to_num(X, nan=-999.0))


def importance(model, feats: list[str], top: int = 30):
    imp = sorted(zip(feats, model.feature_importance("gain")), key=lambda t: -t[1])
    return imp[:top]
