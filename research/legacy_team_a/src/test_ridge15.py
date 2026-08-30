import datetime as dt

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge

from src.config import VAL_FOLDS_S1
from src.ridge15 import (BASE_WEIGHTS, CONTROL_WEIGHTS, EXP_ID, LAMBDAS, MAIN_WEIGHTS,
                         clean_train_cutoffs, fit_ridge_arrays, select_lofo_lambdas,
                         select_numeric_features, standardize)


def test_train_only_normalization_and_nonfinite_to_mean():
    Xtr = np.array([[1.0, np.nan], [3.0, 2.0], [5.0, np.inf]])
    y = np.array([0.0, 1.0, 2.0])
    mean, std, _ = fit_ridge_arrays(Xtr, y, 0.1)
    before = mean.copy(), std.copy()
    Zv = standardize(np.array([[1000.0, np.nan]]), mean, std)
    assert np.array_equal(mean, before[0])
    assert np.array_equal(std, before[1])
    assert Zv[0, 1] == 0.0
    assert Zv[0, 0] > 100.0


def test_no_future_lookup_in_clean_corridor():
    for val in VAL_FOLDS_S1:
        cuts = clean_train_cutoffs(val)
        assert val not in cuts
        assert all(cut + dt.timedelta(days=30) <= val for cut in cuts)


def test_user_id_is_never_a_feature():
    X = pl.DataFrame({"user_id": [1, 2], "x": [1.0, 2.0], "n": [3, 4]})
    assert select_numeric_features(X) == ["x", "n"]


def test_solver_matches_sklearn_mean_squared_formulation():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(120, 8))
    X[::11, 2] = np.nan
    y = rng.normal(size=120)
    lam = 1e-2
    mean, std, coef = fit_ridge_arrays(X, y, lam)
    Z = standardize(X, mean, std)
    ref = Ridge(alpha=len(X) * lam, fit_intercept=True, solver="cholesky").fit(Z, y)
    assert np.allclose(coef[0], ref.intercept_, atol=1e-10)
    assert np.allclose(coef[1:], ref.coef_, atol=1e-10)
    assert np.allclose(coef[0] + Z @ coef[1:], ref.predict(Z), atol=1e-10)


def test_lambda_selection_is_reproducible_and_from_fixed_grid():
    curve = np.array([[2.0, 2.0, 2.0, 2.0], [1.9, 1.8, 1.7, 1.6],
                      [1.8, 1.7, 1.6, 1.5], [1.81, 1.71, 1.61, 1.51],
                      [1.9, 1.9, 1.9, 1.9], [2.1, 2.1, 2.1, 2.1]])
    a = select_lofo_lambdas(curve)
    b = select_lofo_lambdas(curve.copy())
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])
    assert a[2] == b[2]
    assert set(a[1]).issubset(set(LAMBDAS))


def test_fixed_blend_contract():
    for weights in (BASE_WEIGHTS, MAIN_WEIGHTS, CONTROL_WEIGHTS):
        assert abs(sum(weights.values()) - 1.0) < 1e-12
        assert weights["S1-E03a"] == 0.10
    assert MAIN_WEIGHTS[EXP_ID] == 0.15
    assert CONTROL_WEIGHTS[EXP_ID] == 0.15
    assert MAIN_WEIGHTS["S1-DIST"] == 0.25
    assert MAIN_WEIGHTS["ETX-AVG3"] == CONTROL_WEIGHTS["ETX-AVG3"] == 0.225
    assert MAIN_WEIGHTS["SEQ-AVG3"] == CONTROL_WEIGHTS["SEQ-AVG3"] == 0.225
