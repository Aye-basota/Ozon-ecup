"""Strategy 2: structural Poisson x empirical-Bayes LogNormal pipeline.

One-command entry points (all use the same leakage-safe feature builder):

    python src/strategy_2.py aggregation
    python src/strategy_2.py count-screen
    python src/strategy_2.py cv --folds 2025-09-18 2025-10-16 --ks 1 2 3 5 8 15
    python src/strategy_2.py season
    python src/strategy_2.py blend
    python src/strategy_2.py final

Protected project files ``src/config.py`` and ``src/validation.py`` are imported
but never modified.  The seed comes only from ``src.config``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy.special import ndtri
from scipy.stats import qmc

# ``python src/strategy_2.py`` puts src/ rather than the repository root on
# sys.path.  Keep the promised invocation working without requiring installation.
if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_PROCESSED, DATA_RAW, SEED, SUBMISSIONS, TARGET_DAYS
from src.features import HISTORY_DAYS, build_features


ROOT = Path(__file__).resolve().parent.parent
RAW_FILE = DATA_RAW / "train.parquet"
ARTIFACTS = ROOT / "artifacts"
DATA_START = dt.date(2025, 1, 1)
GRID_START = dt.date(2025, 4, 3)
GRID_END = dt.date(2025, 10, 16)
TEST_CUTOFF = dt.date(2026, 2, 13)
VAL_FOLDS = [
    dt.date(2025, 9, 4),
    dt.date(2025, 9, 18),
    dt.date(2025, 10, 2),
    dt.date(2025, 10, 16),
]
K_GRID = [1.0, 2.0, 3.0, 5.0, 8.0, 15.0]
NMAX = 30
QN = 11
LEVEL_TEST = 2.3293
CAL_FACTOR = 1.1628
CAL_MU_LIFT = 0.0804
ROUNDS = 600
T0 = time.time()
_SMALL_N_TABLE: dict[str, np.ndarray] | None = None


def log(message: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {message}", flush=True)


def date_grid() -> list[dt.date]:
    out: list[dt.date] = []
    current = GRID_START
    while current <= GRID_END:
        out.append(current)
        current += dt.timedelta(days=7)
    return out


def train_cutoffs(validation_cutoff: dt.date) -> list[dt.date]:
    return [cutoff for cutoff in date_grid()
            if cutoff + dt.timedelta(days=TARGET_DAYS) <= validation_cutoff]


def _tag(cutoff: dt.date) -> str:
    return cutoff.strftime("%Y%m%d")


def _ensure_dirs() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)


def panel_users(cutoff: dt.date, n_blocks: int) -> pl.DataFrame:
    """Organiser panel: active in every trailing, non-overlapping 30-day block."""
    cache = DATA_PROCESSED / f"panel_{_tag(cutoff)}_b{n_blocks}.parquet"
    if cache.exists():
        return pl.read_parquet(cache).select("user_id").sort("user_id")
    users: pl.DataFrame | None = None
    for block in range(n_blocks):
        end = cutoff - dt.timedelta(days=30 * block)
        start = end - dt.timedelta(days=29)
        active = (
            pl.scan_parquet(RAW_FILE)
            .select("user_id", "event_date")
            .filter((pl.col("event_date") >= start) & (pl.col("event_date") <= end))
            .select("user_id").unique().collect()
        )
        users = active if users is None else users.join(active, on="user_id", how="inner")
    assert users is not None
    users = users.sort("user_id")
    users.write_parquet(cache)
    return users


def target_table(cutoff: dt.date) -> pl.DataFrame:
    """All target components in (T, T+30], independent of a panel definition."""
    cache = DATA_PROCESSED / f"s2_target_{_tag(cutoff)}.parquet"
    if cache.exists():
        return pl.read_parquet(cache)
    start = cutoff + dt.timedelta(days=1)
    end = cutoff + dt.timedelta(days=TARGET_DAYS)
    target = (
        pl.scan_parquet(RAW_FILE)
        .select("user_id", "event_date", "gmv")
        .filter((pl.col("event_date") >= start) & (pl.col("event_date") <= end)
                & (pl.col("gmv") > 0))
        .group_by("user_id")
        .agg(
            pl.col("gmv").sum().alias("y"),
            pl.len().cast(pl.Int16).alias("n"),
            pl.col("gmv").log().mean().alias("target_mu"),
        )
        .collect().sort("user_id")
    )
    target.write_parquet(cache)
    return target


def history_table(cutoff: dt.date) -> pl.DataFrame:
    """Sufficient statistics of log(daily GMV) in (T-180, T]."""
    cache = DATA_PROCESSED / f"s2_value_{_tag(cutoff)}_L{HISTORY_DAYS}.parquet"
    if cache.exists():
        return pl.read_parquet(cache)
    start = cutoff - dt.timedelta(days=HISTORY_DAYS)
    history = (
        pl.scan_parquet(RAW_FILE)
        .select("user_id", "event_date", "gmv")
        .filter((pl.col("event_date") > start) & (pl.col("event_date") <= cutoff)
                & (pl.col("gmv") > 0))
        .with_columns(log_gmv=pl.col("gmv").log())
        .group_by("user_id")
        .agg(
            pl.len().cast(pl.Int16).alias("k"),
            pl.col("log_gmv").sum().alias("sum_log"),
            (pl.col("log_gmv") ** 2).sum().alias("sum_sq_log"),
        )
        .collect().sort("user_id")
    )
    history.write_parquet(cache)
    return history


@dataclass
class Dataset:
    cutoff: dt.date
    users: np.ndarray
    features: list[str]
    x: np.ndarray
    y: np.ndarray
    n: np.ndarray
    target_mu: np.ndarray


def make_dataset(cutoff: dt.date, n_blocks: int, with_target: bool = True) -> Dataset:
    """Panel -> build_features(T) -> targets.  No feature is built elsewhere."""
    users = panel_users(cutoff, n_blocks)
    features = build_features(cutoff)
    frame = users.join(features, on="user_id", how="left").sort("user_id")
    names = [column for column in frame.columns if column != "user_id"]
    x = frame.select(names).to_numpy().astype(np.float32)
    if with_target:
        joined = (
            users.join(target_table(cutoff), on="user_id", how="left")
            .with_columns(
                pl.col("y").fill_null(0.0),
                pl.col("n").fill_null(0),
            )
            .sort("user_id")
        )
        y = joined["y"].to_numpy().astype(np.float32)
        n = joined["n"].to_numpy().astype(np.float32)
        target_mu = joined["target_mu"].to_numpy().astype(np.float64)
    else:
        y = np.empty(0, np.float32)
        n = np.empty(0, np.float32)
        target_mu = np.empty(0, np.float64)
    return Dataset(cutoff, frame["user_id"].to_numpy(), names, x, y, n, target_mu)


def assemble(cutoffs: Iterable[dt.date], n_blocks: int = 1) -> tuple[np.ndarray, np.ndarray, list[str]]:
    matrices: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    feature_names: list[str] | None = None
    total = 0
    for cutoff in cutoffs:
        dataset = make_dataset(cutoff, n_blocks)
        if feature_names is None:
            feature_names = dataset.features
        else:
            assert feature_names == dataset.features, f"feature mismatch at {cutoff}"
        matrices.append(dataset.x)
        labels.append(dataset.n)
        total += len(dataset.n)
        log(f"assembled {cutoff}: {len(dataset.n):,} rows (total {total:,})")
    assert feature_names is not None and matrices
    x = np.vstack(matrices)
    n = np.concatenate(labels)
    matrices.clear(); labels.clear(); gc.collect()
    return x, n, feature_names


def lgb_params(objective: str) -> dict:
    params = {
        "objective": objective,
        "metric": "poisson" if objective == "poisson" else "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 5.0,
        "max_bin": 63,
        "force_row_wise": True,
        "num_threads": 10,
        "verbosity": -1,
        "seed": SEED,
        "feature_fraction_seed": SEED,
        "bagging_seed": SEED,
        "data_random_seed": SEED,
    }
    return params


def _offset_from_x(x: np.ndarray, feature_names: list[str]) -> np.ndarray:
    index = feature_names.index("n_expected_L")
    # The formula in the strategy is k*30/L.  A small numerical floor is needed
    # only because log(0) is not a valid LightGBM init_score.
    return np.maximum(x[:, index].astype(np.float64), 0.02)


def fit_count(cutoffs: list[dt.date], mode: str) -> tuple[lgb.Booster, list[str]]:
    x, n, names = assemble(cutoffs, n_blocks=1)
    params = lgb_params("poisson")
    init_score = None
    if mode == "offset":
        init_score = np.log(_offset_from_x(x, names))
        params["boost_from_average"] = False
    dataset = lgb.Dataset(x, label=n, init_score=init_score, feature_name=names,
                          free_raw_data=True)
    del x, n, init_score
    gc.collect()
    log(f"training {mode} Poisson on {len(cutoffs)} cutoffs")
    model = lgb.train(params, dataset, num_boost_round=ROUNDS)
    return model, names


def predict_count(model: lgb.Booster, dataset: Dataset, mode: str) -> np.ndarray:
    if mode == "offset":
        raw = model.predict(dataset.x, raw_score=True)
        return np.maximum(np.exp(np.clip(raw, -20, 20))
                          * _offset_from_x(dataset.x, dataset.features), 1e-8)
    return np.maximum(model.predict(dataset.x), 1e-8)


def fit_classifier(cutoffs: list[dt.date]) -> tuple[lgb.Booster, list[str]]:
    x, n, names = assemble(cutoffs, n_blocks=1)
    dataset = lgb.Dataset(x, label=(n > 0).astype(np.int8), feature_name=names,
                          free_raw_data=True)
    del x, n
    gc.collect()
    log(f"training hurdle P(n>0) on {len(cutoffs)} cutoffs")
    model = lgb.train(lgb_params("binary"), dataset, num_boost_round=ROUNDS)
    return model, names


def _prediction_cache(kind: str, mode: str, cutoffs: list[dt.date], validation: dt.date) -> Path:
    return ARTIFACTS / (
        f"s2_{kind}_{mode}_tr{_tag(cutoffs[0])}-{_tag(cutoffs[-1])}_"
        f"n{len(cutoffs)}_val{_tag(validation)}.npz"
    )


def cached_count_prediction(cutoffs: list[dt.date], validation: Dataset, mode: str) -> np.ndarray:
    cache = _prediction_cache("lambda", mode, cutoffs, validation.cutoff)
    if cache.exists():
        saved = np.load(cache)
        assert np.array_equal(saved["user_id"], validation.users)
        log(f"loaded cached lambda: {cache.name}")
        return saved["lambda"].astype(np.float64)
    model, names = fit_count(cutoffs, mode)
    assert names == validation.features
    lam = predict_count(model, validation, mode)
    np.savez_compressed(cache, user_id=validation.users, **{"lambda": lam.astype(np.float32)})
    del model
    gc.collect()
    return lam


def cached_positive_probability(cutoffs: list[dt.date], validation: Dataset) -> np.ndarray:
    cache = _prediction_cache("ppos", "binary", cutoffs, validation.cutoff)
    if cache.exists():
        saved = np.load(cache)
        assert np.array_equal(saved["user_id"], validation.users)
        log(f"loaded cached ppos: {cache.name}")
        return saved["ppos"].astype(np.float64)
    model, names = fit_classifier(cutoffs)
    assert names == validation.features
    ppos = np.clip(model.predict(validation.x), 1e-6, 1 - 1e-6)
    np.savez_compressed(cache, user_id=validation.users, ppos=ppos.astype(np.float32))
    del model
    gc.collect()
    return ppos


@dataclass
class ValueHistory:
    k: np.ndarray
    sum_log: np.ndarray
    sum_sq_log: np.ndarray
    mu_pop: float
    var_pop: float


def value_history(cutoff: dt.date, users: np.ndarray) -> ValueHistory:
    user_frame = pl.DataFrame({"user_id": users})
    joined = (
        user_frame.join(history_table(cutoff), on="user_id", how="left")
        .with_columns(
            pl.col("k").fill_null(0),
            pl.col("sum_log").fill_null(0.0),
            pl.col("sum_sq_log").fill_null(0.0),
        )
        .sort("user_id")
    )
    k = joined["k"].to_numpy().astype(np.float64)
    sums = joined["sum_log"].to_numpy().astype(np.float64)
    squares = joined["sum_sq_log"].to_numpy().astype(np.float64)
    total = max(k.sum(), 1.0)
    mu_pop = float(sums.sum() / total)
    var_pop = float(max(squares.sum() / total - mu_pop ** 2, 1e-3))
    return ValueHistory(k, sums, squares, mu_pop, var_pop)


def empirical_bayes(history: ValueHistory, k_prior: float) -> tuple[np.ndarray, np.ndarray]:
    k = history.k
    mu = (history.sum_log + k_prior * history.mu_pop) / (k + k_prior)
    individual_mean = np.divide(history.sum_log, k, out=np.zeros_like(k), where=k > 0)
    numerator = np.maximum(history.sum_sq_log - k * individual_mean ** 2, 0.0)
    within = np.where(k >= 2, numerator / np.maximum(k - 1, 1), history.var_pop)
    variance = np.maximum((k * within + k_prior * history.var_pop) / (k + k_prior), 1e-3)
    return mu, np.sqrt(variance)


def _quadrature(qn: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.hermite_e.hermegauss(qn)
    return nodes, weights / weights.sum()


def _small_n_table() -> dict[str, np.ndarray]:
    """QMC lookup for E[log1p(sum LogNormal)] when n=1..4.

    Fenton-Wilkinson failed the mandatory 0.01 check for n=2..4.  A scrambled
    Sobol sequence with common random numbers makes the replacement both stable
    and cheap at inference: the expensive calculation is cached once, inference
    is bilinear interpolation in (mu, sigma).
    """
    global _SMALL_N_TABLE
    if _SMALL_N_TABLE is not None:
        return _SMALL_N_TABLE
    cache = DATA_PROCESSED / "s2_small_n_qmc_v1.npz"
    if cache.exists():
        loaded = np.load(cache)
        _SMALL_N_TABLE = {key: loaded[key] for key in loaded.files}
        return _SMALL_N_TABLE

    mu_grid = np.linspace(-1.0, 9.0, 101)
    sigma_grid = np.linspace(0.2, 3.0, 57)
    # 2**15 low-discrepancy points; clipping avoids ndtri(0/1).
    uniforms = qmc.Sobol(d=4, scramble=True, seed=SEED).random_base2(15)
    normals = ndtri(np.clip(uniforms, 1e-12, 1 - 1e-12))
    table: dict[str, np.ndarray] = {"mu_grid": mu_grid, "sigma_grid": sigma_grid}
    for n in (1, 2, 3, 4):
        values = np.empty((len(sigma_grid), len(mu_grid)), dtype=np.float64)
        for sigma_index, sigma in enumerate(sigma_grid):
            scaled = sigma * normals[:, :n]
            maximum = scaled.max(axis=1)
            log_sum = maximum + np.log(np.exp(scaled - maximum[:, None]).sum(axis=1))
            for start in range(0, len(mu_grid), 20):
                stop = min(start + 20, len(mu_grid))
                values[sigma_index, start:stop] = np.logaddexp(
                    0.0, log_sum[:, None] + mu_grid[None, start:stop]
                ).mean(axis=0)
        table[f"n{n}"] = values
        log(f"built small-n QMC lookup for n={n}")
    np.savez_compressed(cache, **table)
    _SMALL_N_TABLE = table
    return table


def small_n_qmc(mu: np.ndarray, sigma: np.ndarray, n: int) -> np.ndarray:
    table = _small_n_table()
    mu_grid = table["mu_grid"]
    sigma_grid = table["sigma_grid"]
    mu_clipped = np.clip(mu, mu_grid[0], mu_grid[-1])
    sigma_clipped = np.clip(sigma, sigma_grid[0], sigma_grid[-1])
    mu_position = (mu_clipped - mu_grid[0]) / (mu_grid[1] - mu_grid[0])
    sigma_position = (sigma_clipped - sigma_grid[0]) / (sigma_grid[1] - sigma_grid[0])
    mu_low = np.minimum(np.floor(mu_position).astype(int), len(mu_grid) - 2)
    sigma_low = np.minimum(np.floor(sigma_position).astype(int), len(sigma_grid) - 2)
    mu_weight = mu_position - mu_low
    sigma_weight = sigma_position - sigma_low
    values = table[f"n{n}"]
    lower = values[sigma_low, mu_low] * (1 - mu_weight) + values[sigma_low, mu_low + 1] * mu_weight
    upper = (values[sigma_low + 1, mu_low] * (1 - mu_weight)
             + values[sigma_low + 1, mu_low + 1] * mu_weight)
    return lower * (1 - sigma_weight) + upper * sigma_weight


def expected_log1p_poisson(
    lam: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    nmax: int = NMAX,
    qn: int = QN,
) -> np.ndarray:
    """E[log1p(S)]: QMC for n<=4, Fenton-Wilkinson for n>=5."""
    nodes, weights = _quadrature(qn)
    ns = np.arange(1, nmax + 1, dtype=np.float64)
    log_factorials = np.array([math.lgamma(number + 1) for number in ns])
    output = np.zeros(len(lam), dtype=np.float64)
    for start in range(0, len(lam), 30_000):
        stop = min(start + 30_000, len(lam))
        local_lam = np.maximum(lam[start:stop, None], 1e-12)
        local_mu = mu[start:stop, None]
        variance = np.minimum(sigma[start:stop, None] ** 2, 20.0)
        pmf = np.exp(-local_lam + np.log(local_lam) * ns[None, :] - log_factorials[None, :])
        sum_variance = np.log1p(np.expm1(variance) / ns[None, :])
        sum_mu = np.log(ns)[None, :] + local_mu + variance / 2 - sum_variance / 2
        sum_sigma = np.sqrt(sum_variance)
        conditional = np.zeros_like(pmf)
        for node, weight in zip(nodes, weights):
            log1p_sum = np.logaddexp(0.0, sum_mu + sum_sigma * node)
            conditional += weight * log1p_sum
        base_mu = mu[start:stop]
        base_sigma = sigma[start:stop]
        for n in range(1, min(4, nmax) + 1):
            conditional[:, n - 1] = small_n_qmc(base_mu, base_sigma, n)
        output[start:stop] = np.sum(pmf * conditional, axis=1)
    return output


def expected_log1p_hurdle(
    lam: np.ndarray,
    p_positive: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    nmax: int = NMAX,
    qn: int = QN,
) -> np.ndarray:
    """Hurdle count law: P(N>0) from GBM, positive Poisson shape renormalised."""
    poisson_value = expected_log1p_poisson(lam, mu, sigma, nmax=nmax, qn=qn)
    positive_mass = np.maximum(-np.expm1(-np.maximum(lam, 1e-12)), 1e-12)
    return p_positive * poisson_value / positive_mass


def fixed_n_fw(mu: np.ndarray, sigma: np.ndarray, n: np.ndarray, qn: int = QN) -> np.ndarray:
    nodes, weights = _quadrature(qn)
    variance = sigma ** 2
    sum_variance = np.log1p(np.expm1(variance) / n)
    sum_mu = np.log(n) + mu + variance / 2 - sum_variance / 2
    out = np.zeros(len(mu))
    for node, weight in zip(nodes, weights):
        out += weight * np.logaddexp(0.0, sum_mu + np.sqrt(sum_variance) * node)
    return out


def fixed_n_hybrid(mu: np.ndarray, sigma: np.ndarray, n: np.ndarray, qn: int = QN) -> np.ndarray:
    out = fixed_n_fw(mu, sigma, n, qn)
    for small_n in (1, 2, 3, 4):
        mask = n == small_n
        if mask.any():
            out[mask] = small_n_qmc(mu[mask], sigma[mask], small_n)
    return out


def rmsle_z(y: np.ndarray, z: np.ndarray) -> float:
    effective = np.maximum(np.asarray(z, np.float64), 0.0)
    return float(np.sqrt(np.mean((np.log1p(y) - effective) ** 2)))


def bias_z(y: np.ndarray, z: np.ndarray) -> float:
    return float(np.mean(np.log1p(y) - np.maximum(z, 0.0)))


def poisson_deviance(n: np.ndarray, lam: np.ndarray) -> float:
    lam = np.maximum(lam, 1e-12)
    term = np.where(n > 0, n * np.log(np.maximum(n, 1e-12) / lam) - (n - lam), lam)
    return float(2 * np.mean(term))


def count_diagnostics(n: np.ndarray, lam: np.ndarray, ppos: np.ndarray | None = None) -> dict:
    predicted_zero = float(np.mean(np.exp(-lam))) if ppos is None else float(np.mean(1 - ppos))
    return {
        "mean_true": float(np.mean(n)),
        "mean_pred": float(np.mean(lam)),
        "var_true": float(np.var(n)),
        "p0_true": float(np.mean(n == 0)),
        "p0_pred": predicted_zero,
        "p0_gap_pp": 100 * (predicted_zero - float(np.mean(n == 0))),
        "poisson_deviance": poisson_deviance(n, lam),
        "pearson_dispersion": float(np.mean((n - lam) ** 2 / np.maximum(lam, 1e-8))),
        "lambda_max": float(np.max(lam)),
    }


def structural_prediction(
    lam: np.ndarray,
    history: ValueHistory,
    k_prior: float,
    mu_shift: float = 0.0,
    sigma_scale: float = 1.0,
    ppos: np.ndarray | None = None,
    qn: int = QN,
) -> np.ndarray:
    mu, sigma = empirical_bayes(history, k_prior)
    if ppos is None:
        return expected_log1p_poisson(lam, mu + mu_shift, sigma * sigma_scale, qn=qn)
    return expected_log1p_hurdle(lam, ppos, mu + mu_shift, sigma * sigma_scale, qn=qn)


def calibration_grid(
    y: np.ndarray,
    lam: np.ndarray,
    history: ValueHistory,
    ks: list[float],
    sigma_scales: list[float],
    ppos: np.ndarray | None,
) -> dict[float, dict]:
    coarse_shifts = np.arange(-0.60, 0.051, 0.10)
    results: dict[float, dict] = {}
    for k_prior in ks:
        best: tuple[float, float, float] | None = None
        for sigma_scale in sigma_scales:
            for mu_shift in coarse_shifts:
                z = structural_prediction(lam, history, k_prior, float(mu_shift),
                                          sigma_scale, ppos)
                score = rmsle_z(y, z)
                candidate = (score, sigma_scale, float(mu_shift))
                if best is None or candidate < best:
                    best = candidate
        assert best is not None
        refine = np.arange(best[2] - 0.075, best[2] + 0.076, 0.025)
        for mu_shift in refine:
            z = structural_prediction(lam, history, k_prior, float(mu_shift), best[1], ppos)
            candidate = (rmsle_z(y, z), best[1], float(mu_shift))
            if candidate < best:
                best = candidate
        results[k_prior] = {
            "score": best[0],
            "sigma_scale": best[1],
            "mu_shift": best[2],
        }
        log(f"train-only calibration K={k_prior:g}: RMSLE={best[0]:.5f}, "
            f"sigma*{best[1]:.2f}, mu{best[2]:+.3f}")
    return results


def aggregation_experiment(samples: int = 10_000) -> dict:
    """Mandatory Strategy 2 Experiment 1: FW/quadrature against Monte Carlo."""
    rng = np.random.default_rng(SEED)
    grid_rows = []
    for n in (2, 3, 4):
        for sigma in (0.8, 1.1, 1.4):
            mu = 3.318
            draws = rng.lognormal(mu, sigma, size=(samples, n))
            monte_carlo = float(np.mean(np.log1p(draws.sum(axis=1))))
            # A larger independent reference separates approximation error from
            # the ~0.01 sampling noise of the prescribed 10k diagnostic.
            verify_rng = np.random.default_rng(SEED + 10_000 * n + int(100 * sigma))
            verify_total = 0.0
            verify_count = 0
            for _ in range(4):
                verify_draws = verify_rng.lognormal(mu, sigma, size=(50_000, n))
                verify_total += np.log1p(verify_draws.sum(axis=1)).sum()
                verify_count += len(verify_draws)
            monte_carlo_verify = float(verify_total / verify_count)
            fw = float(fixed_n_fw(np.array([mu]), np.array([sigma]),
                                  np.array([float(n)]), qn=QN)[0])
            hybrid = float(fixed_n_hybrid(np.array([mu]), np.array([sigma]),
                                          np.array([float(n)]), qn=QN)[0])
            grid_rows.append({"n": n, "sigma": sigma, "mc": monte_carlo,
                              "fw": fw, "fw_abs_error": abs(fw - monte_carlo),
                              "hybrid": hybrid,
                              "hybrid_abs_error": abs(hybrid - monte_carlo),
                              "mc_verify_200k": monte_carlo_verify,
                              "hybrid_verify_abs_error": abs(hybrid - monte_carlo_verify)})
            log(f"FW n={n} sigma={sigma:.1f}: MC={monte_carlo:.6f} "
                f"FW={fw:.6f} |diff|={abs(fw - monte_carlo):.6f}; "
                f"hybrid={hybrid:.6f} |diff|={abs(hybrid - monte_carlo):.6f}; "
                f"verify200k |diff|={abs(hybrid - monte_carlo_verify):.6f}")

    # The requested 1000-user stress test, evaluated in bounded chunks.
    user_count = 1000
    user_n = rng.integers(1, 5, user_count)
    user_mu = rng.uniform(2.5, 4.2, user_count)
    user_sigma = rng.uniform(0.8, 1.4, user_count)
    fw_user = fixed_n_fw(user_mu, user_sigma, user_n.astype(float), qn=QN)
    hybrid_user = fixed_n_hybrid(user_mu, user_sigma, user_n.astype(float), qn=QN)
    mc_user = np.empty(user_count)
    for start in range(0, user_count, 25):
        stop = min(start + 25, user_count)
        for index in range(start, stop):
            draws = rng.lognormal(user_mu[index], user_sigma[index],
                                  size=(samples, int(user_n[index])))
            mc_user[index] = np.mean(np.log1p(draws.sum(axis=1)))
    errors = np.abs(fw_user - mc_user)
    hybrid_errors = np.abs(hybrid_user - mc_user)

    lam = rng.lognormal(math.log(2.0), 0.8, 10_000)
    mu = rng.uniform(2.5, 4.2, len(lam))
    sigma = rng.uniform(0.7, 1.5, len(lam))
    q11 = expected_log1p_poisson(lam, mu, sigma, qn=11)
    q21 = expected_log1p_poisson(lam, mu, sigma, qn=21)
    qdiff = np.abs(q11 - q21)
    result = {
        "grid": grid_rows,
        "fw_grid_max_abs_error": max(row["fw_abs_error"] for row in grid_rows),
        "hybrid_grid_max_abs_error": max(row["hybrid_abs_error"] for row in grid_rows),
        "hybrid_verify_grid_max_abs_error": max(row["hybrid_verify_abs_error"] for row in grid_rows),
        "fw_users_mean_abs_error": float(errors.mean()),
        "fw_users_p95_abs_error": float(np.quantile(errors, 0.95)),
        "fw_users_max_abs_error": float(errors.max()),
        "hybrid_users_mean_abs_error": float(hybrid_errors.mean()),
        "hybrid_users_p95_abs_error": float(np.quantile(hybrid_errors, 0.95)),
        "hybrid_users_max_abs_error": float(hybrid_errors.max()),
        "qn11_vs_21_mean": float(qdiff.mean()),
        "qn11_vs_21_max": float(qdiff.max()),
        "pass_grid_001": max(row["hybrid_verify_abs_error"] for row in grid_rows) <= 0.01,
    }
    log("1000-user raw FW errors: mean={:.5f}, p95={:.5f}, max={:.5f}".format(
        result["fw_users_mean_abs_error"], result["fw_users_p95_abs_error"],
        result["fw_users_max_abs_error"]))
    log("1000-user hybrid errors: mean={:.5f}, p95={:.5f}, max={:.5f}".format(
        result["hybrid_users_mean_abs_error"], result["hybrid_users_p95_abs_error"],
        result["hybrid_users_max_abs_error"]))
    log("QN11 vs QN21: mean={:.8f}, max={:.8f}".format(
        result["qn11_vs_21_mean"], result["qn11_vs_21_max"]))
    _ensure_dirs()
    (ARTIFACTS / "s2_aggregation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def count_screen(validation_cutoff: dt.date) -> dict:
    """Compare the Strategy 2 init-score hypothesis with plain Poisson."""
    _ensure_dirs()
    cuts = train_cutoffs(validation_cutoff)
    validation = make_dataset(validation_cutoff, 3)
    history = value_history(validation_cutoff, validation.users)
    result: dict[str, dict] = {}
    for mode in ("plain", "offset"):
        lam = cached_count_prediction(cuts, validation, mode)
        z = structural_prediction(lam, history, 3.0)
        diagnostics = count_diagnostics(validation.n, lam)
        diagnostics.update({
            "structural_raw_rmsle": rmsle_z(validation.y, z),
            "structural_raw_bias": bias_z(validation.y, z),
            "structural_raw_mean_z": float(z.mean()),
        })
        result[mode] = diagnostics
        log(f"{mode}: dev={diagnostics['poisson_deviance']:.5f}, "
            f"p0={diagnostics['p0_pred']:.4f}/{diagnostics['p0_true']:.4f}, "
            f"struct={diagnostics['structural_raw_rmsle']:.5f}")
    result["selected"] = min((result[mode]["poisson_deviance"], mode)
                             for mode in ("plain", "offset"))[1]
    (ARTIFACTS / "s2_count_screen.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def outer_fold(
    validation_cutoff: dt.date,
    mode: str,
    ks: list[float],
    sigma_scales: list[float],
    use_hurdle: bool,
) -> tuple[dict, dict[float, np.ndarray], Dataset]:
    cuts = train_cutoffs(validation_cutoff)
    if len(cuts) < 2:
        raise ValueError(f"not enough train cutoffs for {validation_cutoff}")
    calibration_cutoff = cuts[-1]
    calibration_train = [cutoff for cutoff in cuts
                         if cutoff + dt.timedelta(days=TARGET_DAYS) <= calibration_cutoff]
    if not calibration_train:
        raise ValueError(f"no nested calibration train for {validation_cutoff}")

    calibration = make_dataset(calibration_cutoff, 3)
    lam_cal = cached_count_prediction(calibration_train, calibration, mode)
    ppos_cal = cached_positive_probability(calibration_train, calibration) if use_hurdle else None
    history_cal = value_history(calibration_cutoff, calibration.users)
    calibrated = calibration_grid(calibration.y, lam_cal, history_cal, ks,
                                  sigma_scales, ppos_cal)

    validation = make_dataset(validation_cutoff, 3)
    lam = cached_count_prediction(cuts, validation, mode)
    ppos = cached_positive_probability(cuts, validation) if use_hurdle else None
    history = value_history(validation_cutoff, validation.users)
    diagnostics = count_diagnostics(validation.n, lam, ppos)
    by_k: dict[float, dict] = {}
    predictions: dict[float, np.ndarray] = {}
    for k_prior in ks:
        config = calibrated[k_prior]
        z = structural_prediction(lam, history, k_prior, config["mu_shift"],
                                  config["sigma_scale"], ppos)
        predictions[k_prior] = z
        mask_mid = (validation.x[:, validation.features.index("w180_days_buy")] >= 2)
        mask_mid &= (validation.x[:, validation.features.index("w180_days_buy")] <= 7)
        mu, _ = empirical_bayes(history, k_prior)
        buying = validation.n > 0
        by_k[k_prior] = {
            **config,
            "rmsle": rmsle_z(validation.y, z),
            "bias": bias_z(validation.y, z),
            "mean_z": float(z.mean()),
            "mid_2_7_rmsle": rmsle_z(validation.y[mask_mid], z[mask_mid]),
            "value_mu_rmse_buyers": float(np.sqrt(np.nanmean(
                (mu[buying] - validation.target_mu[buying]) ** 2))),
        }
        log(f"outer {validation_cutoff} K={k_prior:g}: RMSLE={by_k[k_prior]['rmsle']:.5f}, "
            f"bias={by_k[k_prior]['bias']:+.4f}, mid={by_k[k_prior]['mid_2_7_rmsle']:.5f}")
    result = {
        "validation": validation_cutoff.isoformat(),
        "calibration_cutoff": calibration_cutoff.isoformat(),
        "calibration_train": [cutoff.isoformat() for cutoff in calibration_train],
        "train_cutoffs": [cutoff.isoformat() for cutoff in cuts],
        "count_mode": mode,
        "hurdle": use_hurdle,
        "count": diagnostics,
        "by_k": {str(k): value for k, value in by_k.items()},
    }
    return result, predictions, validation


def cv_experiment(
    folds: list[dt.date],
    mode: str,
    ks: list[float],
    sigma_scales: list[float],
    use_hurdle: bool,
    output_name: str,
) -> dict:
    _ensure_dirs()
    results = []
    saved: dict[str, list[np.ndarray]] = {"user_id": [], "cutoff": [], "y": [], "n": []}
    for k_prior in ks:
        saved[f"z_K{k_prior:g}"] = []
    for validation_cutoff in folds:
        fold_result, predictions, validation = outer_fold(
            validation_cutoff, mode, ks, sigma_scales, use_hurdle)
        results.append(fold_result)
        saved["user_id"].append(validation.users)
        saved["cutoff"].append(np.full(len(validation.users), validation_cutoff.isoformat(), dtype="U10"))
        saved["y"].append(validation.y)
        saved["n"].append(validation.n)
        for k_prior in ks:
            saved[f"z_K{k_prior:g}"].append(predictions[k_prior].astype(np.float32))
        del predictions, validation
        gc.collect()

    summary: dict[str, dict] = {}
    for k_prior in ks:
        scores = [fold["by_k"][str(k_prior)]["rmsle"] for fold in results]
        summary[str(k_prior)] = {
            "fold_scores": scores,
            "cv_mean": float(np.mean(scores)),
            "cv_std": float(np.std(scores)),
            "all_folds_bias_mean": float(np.mean([
                fold["by_k"][str(k_prior)]["bias"] for fold in results])),
        }
        log(f"SUMMARY K={k_prior:g}: {scores}, CV={np.mean(scores):.5f} +/- {np.std(scores):.5f}")
    best_k = min((value["cv_mean"], float(k)) for k, value in summary.items())[1]
    result = {
        "folds": results,
        "summary": summary,
        "selected_k": best_k,
        "count_mode": mode,
        "hurdle": use_hurdle,
        "sigma_scales": sigma_scales,
    }
    artifact = ARTIFACTS / f"{output_name}.npz"
    np.savez_compressed(artifact, **{key: np.concatenate(value) for key, value in saved.items()})
    (ARTIFACTS / f"{output_name}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"saved {artifact}")
    return result


def season_experiment(
    mode: str,
    k_prior: float,
    mu_shift: float,
    sigma_scale: float,
    use_hurdle: bool,
) -> dict:
    """Prescribed one-block calendar diagnostic; not part of primary CV."""
    _ensure_dirs()
    validation_cutoff = dt.date(2025, 2, 13)
    # Strategy 1/2 explicitly define this as a reverse-time calibration scenario:
    # spring/summer clean targets fit the model, the February fold only measures
    # the direction and safe strength of the calendar correction.
    cuts = [cutoff for cutoff in date_grid() if cutoff <= dt.date(2025, 6, 26)]
    validation = make_dataset(validation_cutoff, 1)
    lam = cached_count_prediction(cuts, validation, mode)
    ppos = cached_positive_probability(cuts, validation) if use_hurdle else None
    history = value_history(validation_cutoff, validation.users)
    rows = []
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        adjusted_lam = lam * CAL_FACTOR ** alpha
        adjusted_mu = mu_shift + alpha * CAL_MU_LIFT
        z = structural_prediction(adjusted_lam, history, k_prior, adjusted_mu, sigma_scale, ppos)
        rows.append({"alpha": alpha, "rmsle": rmsle_z(validation.y, z),
                     "bias": bias_z(validation.y, z), "mean_z": float(z.mean())})
        log(f"season alpha={alpha:.2f}: RMSLE={rows[-1]['rmsle']:.5f}, "
            f"bias={rows[-1]['bias']:+.4f}")
    best = min(rows, key=lambda row: row["rmsle"])
    result = {
        "diagnostic_only": True,
        "validation": validation_cutoff.isoformat(),
        "panel_blocks": 1,
        "hurdle": use_hurdle,
        "train_cutoffs": [cutoff.isoformat() for cutoff in cuts],
        "rows": rows,
        "best_alpha": best["alpha"],
        "strategy_rule_alpha": 0.5 if rows[2]["rmsle"] < rows[0]["rmsle"] else 0.0,
    }
    (ARTIFACTS / "s2_season.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _load_s1_oof(s1_artifacts: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    experiments = ["S1-E10", "S1-E02", "S1-E03a"]
    weights = [0.45, 0.45, 0.10]
    loaded = [np.load(s1_artifacts / f"oof_{experiment}.npz") for experiment in experiments]
    base_keys = np.char.add(loaded[0]["cutoff"].astype("U10"),
                            np.char.zfill(loaded[0]["user_id"].astype("U10"), 10))
    order = np.argsort(base_keys)
    z = np.zeros(len(order), dtype=np.float64)
    for data, weight in zip(loaded, weights):
        keys = np.char.add(data["cutoff"].astype("U10"),
                           np.char.zfill(data["user_id"].astype("U10"), 10))
        local_order = np.argsort(keys)
        assert np.array_equal(keys[local_order], base_keys[order])
        z += weight * data["z"][local_order]
    return (loaded[0]["user_id"][order], loaded[0]["cutoff"][order],
            loaded[0]["y"][order], z)


def _best_delta(y: np.ndarray, z: np.ndarray) -> float:
    # Iteration handles the tiny set clipped at zero after applying the shift.
    delta = float(np.mean(np.log1p(y) - z))
    for _ in range(8):
        active = z + delta > 0
        delta = float(np.mean(np.log1p(y[active]) - z[active])) if active.any() else 0.0
    return delta


def blend_experiment(s2_oof: Path, k_prior: float, s1_artifacts: Path) -> dict:
    _ensure_dirs()
    s2 = np.load(s2_oof)
    z_key = f"z_K{k_prior:g}"
    s2_keys = np.char.add(s2["cutoff"].astype("U10"),
                          np.char.zfill(s2["user_id"].astype("U10"), 10))
    s2_order = np.argsort(s2_keys)
    s1_uid, s1_cutoff, s1_y, z_s1 = _load_s1_oof(s1_artifacts)
    s1_keys = np.char.add(s1_cutoff.astype("U10"), np.char.zfill(s1_uid.astype("U10"), 10))
    positions = np.searchsorted(s1_keys, s2_keys[s2_order])
    assert np.array_equal(s1_keys[positions], s2_keys[s2_order])
    z_s1 = z_s1[positions]
    y = s2["y"][s2_order]
    assert np.allclose(y, s1_y[positions])
    z_struct = s2[z_key][s2_order].astype(np.float64)
    cutoffs = s2["cutoff"][s2_order]
    unique_folds = np.unique(cutoffs)

    cross = []
    fold_pairs = list(zip(unique_folds[:-1], unique_folds[1:]))
    fold_pairs += [(right, left) for left, right in fold_pairs]
    for fit_fold, eval_fold in fold_pairs:
        fit_mask = cutoffs == fit_fold
        eval_mask = cutoffs == eval_fold
        candidates = []
        for weight in np.arange(0, 0.61, 0.05):
            mixed = (1 - weight) * z_s1[fit_mask] + weight * z_struct[fit_mask]
            delta = _best_delta(y[fit_mask], mixed)
            candidates.append((rmsle_z(y[fit_mask], mixed + delta), float(weight), delta))
        _, weight, delta = min(candidates)
        evaluation = (1 - weight) * z_s1[eval_mask] + weight * z_struct[eval_mask] + delta
        base_delta = _best_delta(y[fit_mask], z_s1[fit_mask])
        base_eval = rmsle_z(y[eval_mask], z_s1[eval_mask] + base_delta)
        score = rmsle_z(y[eval_mask], evaluation)
        row = {"fit_fold": str(fit_fold), "eval_fold": str(eval_fold),
               "structural_weight": weight, "delta": delta,
               "eval_rmsle": score, "s1_eval_rmsle": base_eval,
               "gain": base_eval - score}
        cross.append(row)
        log(f"blend {fit_fold}->{eval_fold}: w_struct={weight:.2f}, "
            f"gain={row['gain']:+.5f}, RMSLE={score:.5f}")

    selected_weight = float(np.median([row["structural_weight"] for row in cross]))
    z_blend = (1 - selected_weight) * z_s1 + selected_weight * z_struct
    residual_corr = float(np.corrcoef(np.log1p(y) - z_s1,
                                      np.log1p(y) - z_struct)[0, 1])
    result = {
        "cross_fold": cross,
        "selected_weight": selected_weight,
        "residual_correlation": residual_corr,
        "s1_raw_rmsle": rmsle_z(y, z_s1),
        "structural_raw_rmsle": rmsle_z(y, z_struct),
        "blend_raw_rmsle": rmsle_z(y, z_blend),
        "acceptance": bool(selected_weight >= 0.15
                           and sum(row["gain"] > 0.003 for row in cross) >= 2),
    }
    log(f"blend summary: w={selected_weight:.2f}, corr(resid)={residual_corr:.4f}, "
        f"raw={result['blend_raw_rmsle']:.5f}, accepted={result['acceptance']}")
    (ARTIFACTS / "s2_blend.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _align_test_component(s1_artifacts: Path, name: str, target_users: np.ndarray) -> np.ndarray:
    users = np.load(s1_artifacts / f"uid_{name}.npy")
    prediction = np.load(s1_artifacts / f"ztest_{name}.npy")
    order = np.argsort(users)
    positions = np.searchsorted(users[order], target_users)
    assert np.array_equal(users[order][positions], target_users)
    return prediction[order][positions].astype(np.float64)


def _shift_to_level(z: np.ndarray, target_level: float) -> tuple[np.ndarray, float]:
    low, high = -5.0, 5.0
    for _ in range(80):
        middle = (low + high) / 2
        if np.mean(np.maximum(z + middle, 0.0)) < target_level:
            low = middle
        else:
            high = middle
    delta = (low + high) / 2
    return np.maximum(z + delta, 0.0), delta


def final_fit(
    mode: str,
    k_prior: float,
    sigma_scales: list[float],
    use_hurdle: bool,
    calendar_alpha: float,
    structural_weight: float,
    s1_artifacts: Path,
    target_level: float,
) -> dict:
    _ensure_dirs()
    all_cuts = date_grid()
    calibration_cutoff = all_cuts[-1]
    calibration_train = [cutoff for cutoff in all_cuts
                         if cutoff + dt.timedelta(days=TARGET_DAYS) <= calibration_cutoff]
    calibration = make_dataset(calibration_cutoff, 3)
    lam_cal = cached_count_prediction(calibration_train, calibration, mode)
    ppos_cal = cached_positive_probability(calibration_train, calibration) if use_hurdle else None
    history_cal = value_history(calibration_cutoff, calibration.users)
    calibrated = calibration_grid(calibration.y, lam_cal, history_cal, [k_prior],
                                  sigma_scales, ppos_cal)[k_prior]

    test = make_dataset(TEST_CUTOFF, 3, with_target=False)
    model, names = fit_count(all_cuts, mode)
    assert names == test.features
    lam = predict_count(model, test, mode)
    model.save_model(str(ARTIFACTS / "s2_count_model.txt"))
    ppos = None
    if use_hurdle:
        classifier, classifier_names = fit_classifier(all_cuts)
        assert classifier_names == test.features
        ppos = np.clip(classifier.predict(test.x), 1e-6, 1 - 1e-6)
        classifier.save_model(str(ARTIFACTS / "s2_hurdle_model.txt"))
        del classifier

    lam *= CAL_FACTOR ** calendar_alpha
    test_history = value_history(TEST_CUTOFF, test.users)
    mu_shift = calibrated["mu_shift"] + calendar_alpha * CAL_MU_LIFT
    z_struct = structural_prediction(lam, test_history, k_prior, mu_shift,
                                     calibrated["sigma_scale"], ppos)
    np.save(ARTIFACTS / "uid_S2-STRUCT.npy", test.users)
    np.save(ARTIFACTS / "ztest_S2-STRUCT.npy", z_struct.astype(np.float32))

    if structural_weight < 1.0:
        z_norm = _align_test_component(s1_artifacts, "S1-NORM", test.users)
        z_unc = _align_test_component(s1_artifacts, "S1-UNC", test.users)
        z_cap = _align_test_component(s1_artifacts, "S1-CAP", test.users)
        z_s1 = 0.45 * z_norm + 0.45 * z_unc + 0.10 * z_cap
        z_raw = (1 - structural_weight) * z_s1 + structural_weight * z_struct
    else:
        z_raw = z_struct
    z_final, level_delta = _shift_to_level(z_raw, target_level)
    prediction = np.expm1(z_final)
    prediction = np.maximum(prediction, 0.0)

    sample = pl.read_csv(DATA_RAW / "sample_submit.csv")
    mapping = pl.DataFrame({"user_id": test.users, "predict_s2": prediction})
    submission = (
        sample.select("user_id")
        .join(mapping, on="user_id", how="left")
        .rename({"predict_s2": "predict"})
    )
    output = SUBMISSIONS / "submission_strategy_2.csv"
    submission.write_csv(output)

    values = submission["predict"].to_numpy()
    sample_users = sample["user_id"].to_numpy()
    checks = {
        "rows": submission.height,
        "columns": submission.columns,
        "order_matches_sample": bool(np.array_equal(submission["user_id"].to_numpy(), sample_users)),
        "unique_users": int(submission["user_id"].n_unique()),
        "nan": int(np.isnan(values).sum()),
        "inf": int(np.isinf(values).sum()),
        "negative": int((values < 0).sum()),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean_log1p": float(np.mean(np.log1p(values))),
    }
    assert checks["rows"] == sample.height == 250_000
    assert checks["unique_users"] == 250_000
    assert checks["order_matches_sample"]
    assert checks["nan"] == checks["inf"] == checks["negative"] == 0
    result = {
        "count_mode": mode,
        "hurdle": use_hurdle,
        "k": k_prior,
        "sigma_scale": calibrated["sigma_scale"],
        "mu_shift_train_only": calibrated["mu_shift"],
        "calendar_alpha": calendar_alpha,
        "calendar_lambda_factor": CAL_FACTOR ** calendar_alpha,
        "calendar_mu_shift": calendar_alpha * CAL_MU_LIFT,
        "structural_weight": structural_weight,
        "raw_structural_level": float(z_struct.mean()),
        "raw_blend_level": float(z_raw.mean()),
        "level_target": target_level,
        "level_delta": level_delta,
        "submission": str(output),
        "checks": checks,
    }
    (ARTIFACTS / "s2_final.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"submission written: {output}")
    log(f"checks: {json.dumps(checks, ensure_ascii=False)}")
    return result


def _dates(values: list[str] | None) -> list[dt.date]:
    return VAL_FOLDS if not values else [dt.date.fromisoformat(value) for value in values]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Strategy 2 experiments")
    sub = parser.add_subparsers(dest="command", required=True)

    aggregation = sub.add_parser("aggregation")
    aggregation.add_argument("--samples", type=int, default=10_000)

    count = sub.add_parser("count-screen")
    count.add_argument("--fold", default="2025-10-16")

    cv = sub.add_parser("cv")
    cv.add_argument("--folds", nargs="*")
    cv.add_argument("--mode", choices=["plain", "offset"], default="offset")
    cv.add_argument("--ks", nargs="+", type=float, default=K_GRID)
    cv.add_argument("--sigma-scales", nargs="+", type=float, default=[1.0])
    cv.add_argument("--hurdle", action="store_true")
    cv.add_argument("--output", default="s2_oof")

    season = sub.add_parser("season")
    season.add_argument("--mode", choices=["plain", "offset"], default="offset")
    season.add_argument("--k", type=float, default=3.0)
    season.add_argument("--mu-shift", type=float, default=-0.3)
    season.add_argument("--sigma-scale", type=float, default=1.0)
    season.add_argument("--hurdle", action="store_true")

    blend = sub.add_parser("blend")
    blend.add_argument("--s2-oof", type=Path, default=ARTIFACTS / "s2_oof_best.npz")
    blend.add_argument("--k", type=float, default=3.0)
    blend.add_argument("--s1-artifacts", type=Path,
                       default=ROOT.parent / "OZON-E-CUP" / "artifacts")

    final = sub.add_parser("final")
    final.add_argument("--mode", choices=["plain", "offset"], default="offset")
    final.add_argument("--k", type=float, default=3.0)
    final.add_argument("--sigma-scales", nargs="+", type=float, default=[0.8, 0.9, 1.0])
    final.add_argument("--hurdle", action="store_true")
    final.add_argument("--calendar-alpha", type=float, default=0.5)
    final.add_argument("--structural-weight", type=float, default=0.2)
    final.add_argument("--level", type=float, default=LEVEL_TEST)
    final.add_argument("--s1-artifacts", type=Path,
                       default=ROOT.parent / "OZON-E-CUP" / "artifacts")

    args = parser.parse_args()
    _ensure_dirs()
    if args.command == "aggregation":
        aggregation_experiment(args.samples)
    elif args.command == "count-screen":
        count_screen(dt.date.fromisoformat(args.fold))
    elif args.command == "cv":
        cv_experiment(_dates(args.folds), args.mode, args.ks, args.sigma_scales,
                      args.hurdle, args.output)
    elif args.command == "season":
        season_experiment(args.mode, args.k, args.mu_shift, args.sigma_scale, args.hurdle)
    elif args.command == "blend":
        blend_experiment(args.s2_oof, args.k, args.s1_artifacts)
    elif args.command == "final":
        final_fit(args.mode, args.k, args.sigma_scales, args.hurdle,
                  args.calendar_alpha, args.structural_weight,
                  args.s1_artifacts, args.level)


if __name__ == "__main__":
    main()
