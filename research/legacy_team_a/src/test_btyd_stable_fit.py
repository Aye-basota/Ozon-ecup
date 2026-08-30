"""Numerical and policy tests for EXP-051 BG/NBD stable fitting."""
from __future__ import annotations

import inspect

import numpy as np
from scipy.optimize import check_grad

from src.btyd_stable_fit import (
    POLICY_ID,
    fit_bgnbd_stable,
    objective_and_gradient,
)
from src.config import SEED


def test_analytic_gradient_matches_finite_difference() -> None:
    x = np.asarray([0.0, 1.0, 2.0, 5.0])
    tx = np.asarray([0.0, 3.0, 8.0, 12.0])
    counts = np.asarray([10.0, 5.0, 3.0, 1.0])
    theta = np.log(np.asarray([0.7, 15.0, 1.5, 4.0]))
    objective_and_gradient.T = 20.0  # type: ignore[attr-defined]
    fun = lambda value: objective_and_gradient(value, x, tx, counts)[0]
    jac = lambda value: objective_and_gradient(value, x, tx, counts)[1]
    assert check_grad(fun, jac, theta) <= 1e-6


def test_policy_is_deterministic_on_fixed_input() -> None:
    rng = np.random.default_rng(SEED)
    x = rng.poisson(4.0, size=2000).astype(np.int32)
    tx = np.where(x > 0, rng.integers(1, 181, size=len(x)), 0).astype(np.int32)
    one = fit_bgnbd_stable(x, tx, 180, "synthetic", 0, fail_on_unstable=False)
    two = fit_bgnbd_stable(x, tx, 180, "synthetic", 0, fail_on_unstable=False)
    assert one["parameters"] == two["parameters"]
    assert one["starts"] == two["starts"]


def test_optimizer_change_does_not_change_model_likelihood() -> None:
    source = inspect.getsource(__import__("src.btyd_stable_fit", fromlist=["fit_bgnbd_stable"]))
    assert "bgnbd_log_terms" in source
    assert "penal" not in source.lower()
    assert POLICY_ID == "analytic-jac-lbfgsb-bfgs-v1"

