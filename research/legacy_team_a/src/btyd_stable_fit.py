"""Deterministic, strict optimizer policy for the unchanged BG/NBD likelihood.

The statistical model and inputs are inherited from EXP-047.  This module only
changes numerical optimization: an analytic log-parameter gradient replaces
finite differences and every fixed start receives an L-BFGS-B solve followed
by a deterministic BFGS polish.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import digamma, logsumexp

from src.btyd_day_bgnbd import (
    LOG_BOUNDS,
    MAX_GRAD_NORM,
    MAX_START_LOG_PARAM_SPREAD,
    MAX_START_NLL_SPREAD,
    OPT_STARTS,
    _compressed_summary,
    bgnbd_log_terms,
    distribution_summary,
    jsonable,
    sha256_array,
)

PARAMETER_NAMES = ("r", "alpha", "a", "b")
POLICY_ID = "analytic-jac-lbfgsb-bfgs-v1"


def _log_likelihood_and_param_gradient(
    x: np.ndarray, tx: np.ndarray, T: float, params: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-summary log likelihood and d(loglik)/d(raw params)."""
    r, alpha, a, b = np.asarray(params, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    tx = np.asarray(tx, dtype=np.float64)
    alive, dead = bgnbd_log_terms(x, tx, T, params)
    terms = np.vstack((alive, dead))
    ll = logsumexp(terms, axis=0)
    weights = np.exp(terms - ll)

    grad_alive = np.empty((len(x), 4), dtype=np.float64)
    grad_alive[:, 0] = digamma(r + x) - digamma(r) + np.log(alpha) - np.log(alpha + T)
    grad_alive[:, 1] = r / alpha - (r + x) / (alpha + T)
    grad_alive[:, 2] = digamma(a + b) - digamma(a + b + x)
    grad_alive[:, 3] = (-digamma(b) + digamma(a + b) + digamma(b + x)
                        - digamma(a + b + x))

    grad_dead = np.zeros_like(grad_alive)
    positive = x > 0
    xp = x[positive]
    txp = tx[positive]
    grad_dead[positive, 0] = (digamma(r + xp) - digamma(r) + np.log(alpha)
                              - np.log(alpha + txp))
    grad_dead[positive, 1] = r / alpha - (r + xp) / (alpha + txp)
    grad_dead[positive, 2] = (-digamma(a) + digamma(a + b) + digamma(a + 1.0)
                              - digamma(a + b + xp))
    grad_dead[positive, 3] = (-digamma(b) + digamma(a + b)
                              + digamma(b + xp - 1.0) - digamma(a + b + xp))
    grad = weights[0, :, None] * grad_alive + weights[1, :, None] * grad_dead
    return ll, grad


def objective_and_gradient(
    theta: np.ndarray,
    x: np.ndarray,
    tx: np.ndarray,
    counts: np.ndarray,
) -> tuple[float, np.ndarray]:
    params = np.exp(theta)
    ll, grad_params = _log_likelihood_and_param_gradient(x, tx, float(objective_and_gradient.T), params)
    n_users = float(counts.sum())
    value = -float(np.dot(counts, ll) / n_users)
    grad = -(counts @ grad_params) * params / n_users
    return value, np.asarray(grad, dtype=np.float64)


# Set immediately before a solve.  Keeping T out of repeated optimizer args
# reduces allocation in scipy's tight loop; execution is single-process/serial.
objective_and_gradient.T = 0.0  # type: ignore[attr-defined]


def _within_bounds(theta: np.ndarray) -> bool:
    return all(lo <= value <= hi for value, (lo, hi) in zip(theta, LOG_BOUNDS))


def fit_bgnbd_stable(
    x: np.ndarray, tx: np.ndarray, T: int, fold: str, donor_group: int,
    *, fail_on_unstable: bool = True,
) -> dict[str, Any]:
    """Fit unchanged BG/NBD using the EXP-051 deterministic optimizer policy."""
    ux, utx, counts = _compressed_summary(x, tx)
    n_users = int(counts.sum())
    objective_and_gradient.T = float(T)  # type: ignore[attr-defined]

    def fun(theta: np.ndarray) -> float:
        return objective_and_gradient(theta, ux, utx, counts)[0]

    def jac(theta: np.ndarray) -> np.ndarray:
        return objective_and_gradient(theta, ux, utx, counts)[1]

    starts: list[dict[str, Any]] = []
    for initial in OPT_STARTS:
        first = minimize(
            fun, np.log(np.asarray(initial, dtype=np.float64)), jac=jac,
            method="L-BFGS-B", bounds=LOG_BOUNDS,
            options={"maxiter": 10_000, "ftol": 1e-15, "gtol": 1e-10, "maxls": 100},
        )
        second = minimize(
            fun, first.x, jac=jac, method="BFGS",
            options={"maxiter": 10_000, "gtol": 1e-10},
        )
        candidates = [first]
        if _within_bounds(second.x) and np.isfinite(second.fun):
            candidates.append(second)
        result = min(candidates, key=lambda item: float(item.fun))
        params = np.exp(result.x)
        gradient = jac(result.x)
        starts.append({
            "initial": list(initial),
            "success": bool(np.all(np.isfinite(result.x)) and np.isfinite(result.fun)),
            "reported_success": bool(result.success),
            "status": int(result.status), "message": str(result.message),
            "nit": int(result.nit), "nfev": int(result.nfev),
            "mean_nll": float(result.fun), "log_likelihood": float(-result.fun * n_users),
            "gradient_norm": float(np.linalg.norm(gradient)),
            "gradient_max_abs": float(np.max(np.abs(gradient))),
            "parameters": dict(zip(PARAMETER_NAMES, params)),
            "log_parameters": result.x.tolist(),
            "hit_bound": bool(any(abs(v - lo) < 1e-6 or abs(v - hi) < 1e-6
                                  for v, (lo, hi) in zip(result.x, LOG_BOUNDS))),
            "stage1": {"success": bool(first.success), "status": int(first.status),
                       "message": str(first.message), "nit": int(first.nit),
                       "mean_nll": float(first.fun),
                       "gradient_norm": float(np.linalg.norm(jac(first.x)))},
            "stage2": {"success": bool(second.success), "status": int(second.status),
                       "message": str(second.message), "nit": int(second.nit),
                       "mean_nll": float(second.fun),
                       "gradient_norm": float(np.linalg.norm(jac(second.x))),
                       "within_bounds": _within_bounds(second.x)},
        })

    nll = np.asarray([s["mean_nll"] for s in starts])
    log_params = np.asarray([s["log_parameters"] for s in starts])
    nll_spread = float(np.ptp(nll))
    param_spread = np.ptp(log_params, axis=0)
    max_gradient = max(s["gradient_norm"] for s in starts)
    stable = (all(s["success"] and not s["hit_bound"] for s in starts)
              and nll_spread <= MAX_START_NLL_SPREAD
              and float(param_spread.max()) <= MAX_START_LOG_PARAM_SPREAD
              and max_gradient <= MAX_GRAD_NORM)
    best_i = int(np.argmin(nll))
    best = starts[best_i]
    out = {
        "policy_id": POLICY_ID,
        "fold": fold, "donor_group": donor_group, "recipient_group": 1 - donor_group,
        "T": int(T), "n_users": n_users, "n_unique_summaries": len(ux),
        "x_distribution": distribution_summary(x),
        "input_summary_sha256": sha256_array(np.column_stack((x, tx)).astype(np.int32)),
        "starts": starts, "best_start_index": best_i,
        "parameters": best["parameters"], "log_likelihood": best["log_likelihood"],
        "mean_nll_spread": nll_spread,
        "max_log_parameter_spread": float(param_spread.max()),
        "log_parameter_spread": dict(zip(PARAMETER_NAMES, param_spread)),
        "max_gradient_norm": max_gradient,
        "stability_thresholds": {
            "mean_nll_spread": MAX_START_NLL_SPREAD,
            "max_log_parameter_spread": MAX_START_LOG_PARAM_SPREAD,
            "gradient_norm": MAX_GRAD_NORM,
        },
        "stable": bool(stable),
    }
    if fail_on_unstable and not stable:
        raise RuntimeError("TECHNICAL_FAIL_UNSTABLE_MLE: " + json.dumps(jsonable(out)))
    return out
