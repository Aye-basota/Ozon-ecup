"""Модели Strategy 1.

Метрика — MSE в переменной z = log1p(y) (eda_findings §8), поэтому L2-регрессия
на log1p(y) оптимизирует её ТОЧНО. Кастомный лосс не нужен.

Четыре постановки:
  direct    — LightGBM regression на log1p(y);
  two_part  — P(y>0) (binary) x E[log1p(y) | y>0]; источник декорреляции ошибок;
  catboost  — другая схема построения деревьев (oblivious), тот же таргет;
  dist      — голова распределения: softmax по 16 бинам z, прогноз = среднее
              этого распределения (эксперимент E0, research/strategy_NN_1.md §1.4).

Про `dist`: метрике нужно E[z|x], а L2 по таргету с точечной массой 0.39 в нуле
и тяжёлым хвостом оценивает это среднее неэффективно. Кросс-энтропия по бинам
даёт ограниченные градиенты, а ожидание восстанавливается точно — при условии,
что центроиды считаются НА TRAIN и прогноз берётся как сумма p_k*m_k, а не как
самый вероятный бин.
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


# --------------------------------------------------------------- голова распределения
DIST_BINS = 16          # 1 бин на ровный ноль + 15 квантильных бинов z|z>0
_ZERO_EDGE = 1e-9       # граница «ноль против положительного»


def z_bins(z, n_bins: int = DIST_BINS) -> np.ndarray:
    """Границы бинов z: бин 0 — ровно ноль, остальные — квантили z|z>0.

    Возвращает n_bins-1 границ (по числу переходов между n_bins бинами).
    """
    q = np.quantile(np.asarray(z)[np.asarray(z) > 0], np.linspace(0, 1, n_bins)[1:-1])
    return np.concatenate([[_ZERO_EDGE], q])


def bin_labels(z, edges) -> np.ndarray:
    return np.searchsorted(edges, z, "right").astype(np.int32)


def bin_centroids(z, labels, n_bins: int = DIST_BINS) -> np.ndarray:
    """Среднее z внутри каждого бина. Считается ТОЛЬКО на train.

    Пустой бин (совпавшие квантили) наследует значение ближайшего непустого слева:
    NaN здесь отравил бы всё предсказание, а вероятность такого бина всё равно ~0.
    """
    z = np.asarray(z, dtype=np.float64)
    cnt = np.bincount(labels, minlength=n_bins)
    tot = np.bincount(labels, weights=z, minlength=n_bins)
    out, last = np.zeros(n_bins), 0.0
    for k in range(n_bins):
        if cnt[k] > 0:
            last = tot[k] / cnt[k]
        out[k] = last
    return out


def dist_targets(ytr, n_bins: int = DIST_BINS):
    """(метки бинов, центроиды) для головы распределения — всё по train."""
    z = np.log1p(ytr)
    lab = bin_labels(z, z_bins(z, n_bins))
    return lab, bin_centroids(z, lab, n_bins)


def dist_expectation(proba, centroids) -> np.ndarray:
    """E[z|x] = сумма p_k*m_k. Именно среднее, а не самый вероятный бин."""
    return np.asarray(proba) @ np.asarray(centroids)


def make_datasets(kind: str, Xtr, ytr, wtr=None, params: dict | None = None):
    """Заранее биннит матрицу, чтобы вызывающий мог освободить Xtr до бустинга.

    На плотной сетке cutoff'ов Xtr занимает 5+ ГБ; без этого шага он живёт
    в памяти всё обучение.

    Для `dist` второй элемент списка — не Dataset, а центроиды бинов: они
    считаются на train вместе с метками и обязаны дожить до инференса.
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
    if kind == "dist":
        lab, cent = dist_targets(ytr)
        return [lgb.Dataset(Xtr, lab, weight=wtr, params=p).construct(), cent]
    raise ValueError(kind)


def train_direct_ds(ds, params: dict | None = None, rounds: int = LGB_ROUNDS):
    return _lgb().train(_params(params), ds, num_boost_round=rounds)


def train_two_part_ds(dss, params: dict | None = None, rounds: int = LGB_ROUNDS):
    lgb = _lgb()
    clf = lgb.train(_params(params, objective="binary", metric="binary_logloss"), dss[0],
                    num_boost_round=rounds)
    reg = lgb.train(_params(params), dss[1], num_boost_round=rounds)
    return clf, reg


def train_dist_ds(dss, params: dict | None = None, rounds: int = LGB_ROUNDS):
    """dss = [Dataset с метками бинов, центроиды]. Возвращает (booster, центроиды)."""
    ds, cent = dss
    p = _params(params, objective="multiclass", metric="multi_logloss", num_class=len(cent))
    return _lgb().train(p, ds, num_boost_round=rounds), cent


def train_direct(Xtr, ytr, wtr=None, params: dict | None = None, rounds: int = LGB_ROUNDS):
    return train_direct_ds(make_datasets("direct", Xtr, ytr, wtr, params)[0], params, rounds)


def train_two_part(Xtr, ytr, wtr=None, params: dict | None = None, rounds: int = LGB_ROUNDS):
    return train_two_part_ds(make_datasets("two_part", Xtr, ytr, wtr, params), params, rounds)


def train_dist(Xtr, ytr, wtr=None, params: dict | None = None, rounds: int = LGB_ROUNDS):
    return train_dist_ds(make_datasets("dist", Xtr, ytr, wtr, params), params, rounds)


def predict_two_part(models, X, num_iteration: int | None = None) -> np.ndarray:
    clf, reg = models
    return clf.predict(X, num_iteration=num_iteration) * reg.predict(X, num_iteration=num_iteration)


def predict_dist(model, X, num_iteration: int | None = None) -> np.ndarray:
    booster, cent = model
    return dist_expectation(booster.predict(X, num_iteration=num_iteration), cent)


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
