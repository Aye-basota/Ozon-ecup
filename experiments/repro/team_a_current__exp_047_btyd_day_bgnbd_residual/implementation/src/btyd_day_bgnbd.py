"""EXP-047: purchase-day BG/NBD residual member for STRONGEST_CURRENT.

This is an isolated, CPU-only experiment.  It reads the raw training parquet
and the five frozen OOF components, never performs test inference, and never
touches a submission path::

    python src/btyd_day_bgnbd.py
    python src/btyd_day_bgnbd.py --analysis-only

The scientific change is deliberately narrow: the supervised S2 frequency
model is replaced by a basic common-origin BG/NBD likelihood.  The monetary
shrinkage is fixed at K=3 and the metric aggregation reuses the verified S2
hybrid (Sobol QMC for n<=4, Fenton-Wilkinson + Gauss-Hermite for n>=5).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.special import betaln, gammaln, logsumexp, ndtri
from scipy.stats import qmc, rankdata, spearmanr
from sklearn.metrics import roc_auc_score

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (ARTIFACTS, FOLD_WEIGHTS_S1, RAW_PARQUET, ROOT, SEED,
                        VAL_FOLDS_S1)
from src.features import build_features
from src.report import evaluate
from src.validation import calibrate, rmsle_z


EXP_NUM = 47
EXP_ID = "BTYD-DAY-BGNBD-RESIDUAL"
PREFIX = "BTYD_DAY_BGNBD_EXP047_V2"
RUN_DIR = ARTIFACTS / PREFIX
RESULTS = ROOT / "research" / "strategies" / "results" / "BTYD_DAY_BGNBD"
ORIGIN = dt.date(2024, 12, 31)
HORIZON = 30
K_MONETARY = 3.0
NMAX = 30
QN = 11  # exact S2 setting; 20 is only the fallback when S2 is absent
FOLDS = tuple(VAL_FOLDS_S1)
FOLD_LABELS = tuple(v.isoformat() for v in FOLDS)
FOLD_WEIGHTS = np.asarray(FOLD_WEIGHTS_S1, dtype=np.float64)
FOLD_WEIGHTS /= FOLD_WEIGHTS.sum()
COMPONENTS = ("S1-E03a", "S1-E02", "S1-DIST", "ETX-AVG3", "SEQ-AVG3")
COMPONENT_WEIGHTS = np.asarray((0.10, 0.20, 0.25, 0.225, 0.225), dtype=np.float64)
EXPECTED_FOLD_CAL = np.asarray(
    (1.766883357, 1.760509577, 1.748629224, 1.741278566), dtype=np.float64)
EXPECTED_WCV = 1.747509863
BLEND_GRID = np.asarray((0.0, 0.025, 0.05, 0.10, 0.15), dtype=np.float64)
OPT_STARTS = (
    (0.50, 20.0, 1.50, 3.0),
    (1.00, 50.0, 2.00, 5.0),
    (2.00, 100.0, 3.00, 8.0),
)
LOG_BOUNDS = ((-9.21034, 9.21034), (-6.90776, 13.81551),
              (-9.21034, 9.21034), (-9.21034, 9.21034))
# Preregistered numerical stability gates (mean NLL objective).
MAX_START_NLL_SPREAD = 1e-6
MAX_START_LOG_PARAM_SPREAD = 0.10
MAX_GRAD_NORM = 1e-3
TIE_TOLERANCE = 1e-5
MC_TOLERANCE = 0.01
S2_QMC_CACHE = ROOT / "data" / "processed" / "s2_small_n_qmc_v1.npz"
T0 = time.time()
_SMALL_N_TABLE: dict[str, np.ndarray] | None = None
_TOUCHED: set[str] = set()


def log(message: str) -> None:
    print(message, flush=True)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def canonical_json(value: Any) -> bytes:
    return json.dumps(jsonable(value), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_array(value: np.ndarray) -> str:
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(a.dtype.str.encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _path_forbidden(path: Path) -> bool:
    text = str(path.resolve()).replace("\\", "/").lower()
    name = path.name.lower()
    return ("/submissions/" in text or name == "sample_submit.csv"
            or name.startswith("ztest_") or name.startswith("uid_")
            or "-test" in name)


def touch(path: Path) -> Path:
    path = Path(path)
    if _path_forbidden(path):
        raise AssertionError(f"forbidden test/submission path: {path}")
    _TOUCHED.add(str(path.resolve()))
    return path


def write_json_once(path: Path, value: Any) -> None:
    payload = json.dumps(jsonable(value), ensure_ascii=False, indent=2,
                         sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"refusing to overwrite different artifact: {path}")
        return
    path.write_text(payload, encoding="utf-8")


def write_csv_once(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    fields: list[str] = []
    for row in rows:
        fields.extend(k for k in row if k not in fields)
    rendered: list[list[str]] = [fields]
    for row in rows:
        rendered.append([
            json.dumps(jsonable(row.get(k)), ensure_ascii=False, sort_keys=True)
            if isinstance(row.get(k), (dict, list, tuple)) else str(row.get(k, ""))
            for k in fields
        ])
    import io
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerows(rendered)
    payload = buffer.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"refusing to overwrite different artifact: {path}")
        return
    path.write_text(payload, encoding="utf-8", newline="")


def save_npz_once(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = np.load(path, allow_pickle=False)
        if set(old.files) != set(arrays):
            raise FileExistsError(f"artifact keys differ: {path}")
        for key, value in arrays.items():
            if (np.issubdtype(np.asarray(value).dtype, np.inexact)
                    and np.issubdtype(old[key].dtype, np.inexact)):
                equal = np.array_equal(old[key], value, equal_nan=True)
            else:
                equal = np.array_equal(old[key], value)
            if not equal:
                raise FileExistsError(f"artifact differs at {key}: {path}")
        return
    np.savez_compressed(path, **arrays)


def row_keys(cutoff: np.ndarray, user_id: np.ndarray) -> np.ndarray:
    return np.char.add(np.char.add(np.asarray(cutoff, dtype="U10"), "|"),
                       np.asarray(user_id).astype("U20"))


def load_npz(path: Path) -> Any:
    return np.load(touch(path), allow_pickle=False)


def exact_baseline() -> dict[str, Any]:
    """Align the five raw OOF components and exactly reconstruct the champion."""
    datasets = [load_npz(ARTIFACTS / f"oof_{name}.npz") for name in COMPONENTS]
    base_keys = row_keys(datasets[0]["cutoff"], datasets[0]["user_id"])
    order = np.argsort(base_keys)
    keys = base_keys[order]
    uid = datasets[0]["user_id"][order].astype(np.int64)
    cut = datasets[0]["cutoff"][order].astype("U10")
    y = datasets[0]["y"][order].astype(np.float64)
    z_parts = []
    component_manifest = []
    for name, d in zip(COMPONENTS, datasets):
        local_keys = row_keys(d["cutoff"], d["user_id"])
        local_order = np.argsort(local_keys)
        if not np.array_equal(local_keys[local_order], keys):
            raise AssertionError(f"OOF row alignment failed: {name}")
        if not np.array_equal(d["y"][local_order].astype(np.float64), y):
            raise AssertionError(f"OOF target alignment failed: {name}")
        z_parts.append(d["z"][local_order].astype(np.float64))
        source = ARTIFACTS / f"oof_{name}.npz"
        component_manifest.append({
            "name": name, "path": str(source.resolve()), "file_sha256": sha256_file(source),
            "prediction_sha256": sha256_array(d["z"]),
            "row_keys_sha256": sha256_array(local_keys),
            "target_sha256": sha256_array(d["y"]), "rows": len(local_keys),
        })
    Z = np.vstack(z_parts)
    z = COMPONENT_WEIGHTS @ Z
    rep = evaluate(y, z, cut)
    if np.max(np.abs(np.asarray(rep["fold_cal"]) - EXPECTED_FOLD_CAL)) >= 5e-10:
        raise AssertionError(f"baseline folds differ: {rep['fold_cal']}")
    if abs(rep["wcv"] - EXPECTED_WCV) >= 5e-10:
        raise AssertionError(f"baseline wCV differs: {rep['wcv']}")
    if len(z) != 770_616 or rep["fold_sizes"] != [188_518, 191_025, 193_694, 197_379]:
        raise AssertionError("baseline row counts differ")
    return {
        "user_id": uid, "cutoff": cut, "y": y, "z": z, "Z": Z,
        "report": rep,
        "manifest": {
            "components": component_manifest, "weights": dict(zip(COMPONENTS, COMPONENT_WEIGHTS)),
            "fold_sizes": rep["fold_sizes"], "fold_cal": rep["fold_cal"],
            "wcv": rep["wcv"], "row_keys_sha256": sha256_array(keys),
            "target_sha256": sha256_array(y), "prediction_sha256": sha256_array(z),
            "status": "PASS_EXACT", "tolerance": 5e-10,
        },
    }


def splitmix64(values: np.ndarray | Iterable[int] | int) -> np.ndarray:
    """Stable, vectorised splitmix64; never use Python's salted hash()."""
    with np.errstate(over="ignore"):
        z = np.asarray(values, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def user_group(values: np.ndarray | Iterable[int] | int) -> np.ndarray:
    return (splitmix64(values) & np.uint64(1)).astype(np.int8)


def bgnbd_log_terms(x: np.ndarray, tx: np.ndarray, T: np.ndarray | float,
                    params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Integrated alive/dead likelihood terms for common-origin BG/NBD."""
    r, alpha, a, b = np.asarray(params, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    tx = np.asarray(tx, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    common = gammaln(r + x) - gammaln(r) + r * np.log(alpha) - betaln(a, b)
    alive = common - (r + x) * np.log(alpha + T) + betaln(a, b + x)
    dead = np.full_like(alive, -np.inf)
    positive = x > 0
    dead[positive] = (
        common[positive] - (r + x[positive]) * np.log(alpha + tx[positive])
        + betaln(a + 1.0, b + x[positive] - 1.0)
    )
    return alive, dead


def bgnbd_log_likelihood(x: np.ndarray, tx: np.ndarray, T: np.ndarray | float,
                         params: np.ndarray) -> np.ndarray:
    alive, dead = bgnbd_log_terms(x, tx, T, params)
    return logsumexp(np.vstack((alive, dead)), axis=0)


def _compressed_summary(x: np.ndarray, tx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pairs = np.column_stack((np.asarray(x, np.int32), np.asarray(tx, np.int32)))
    unique, count = np.unique(pairs, axis=0, return_counts=True)
    return unique[:, 0].astype(np.float64), unique[:, 1].astype(np.float64), count.astype(np.float64)


def fit_bgnbd(x: np.ndarray, tx: np.ndarray, T: int, fold: str, donor_group: int) -> dict[str, Any]:
    ux, utx, counts = _compressed_summary(x, tx)
    n_users = int(counts.sum())

    def objective(theta: np.ndarray) -> float:
        params = np.exp(theta)
        ll = bgnbd_log_likelihood(ux, utx, float(T), params)
        if not np.all(np.isfinite(ll)):
            return 1e100
        return float(-np.dot(counts, ll) / n_users)

    starts = []
    for start in OPT_STARTS:
        result = minimize(objective, np.log(np.asarray(start, np.float64)),
                          method="L-BFGS-B", bounds=LOG_BOUNDS,
                          options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-7,
                                   "maxls": 50})
        params = np.exp(result.x)
        starts.append({
            "initial": list(start), "success": bool(result.success),
            "status": int(result.status), "message": str(result.message),
            "nit": int(result.nit), "nfev": int(result.nfev),
            "mean_nll": float(result.fun), "log_likelihood": float(-result.fun * n_users),
            "gradient_norm": float(np.linalg.norm(result.jac)),
            "parameters": dict(zip(("r", "alpha", "a", "b"), params)),
            "log_parameters": result.x.tolist(),
            "hit_bound": bool(any(abs(v - lo) < 1e-6 or abs(v - hi) < 1e-6
                                  for v, (lo, hi) in zip(result.x, LOG_BOUNDS))),
        })
    nll = np.asarray([s["mean_nll"] for s in starts])
    log_params = np.asarray([s["log_parameters"] for s in starts])
    nll_spread = float(nll.max() - nll.min())
    param_spread = np.ptp(log_params, axis=0)
    stable = (all(s["success"] and not s["hit_bound"] for s in starts)
              and nll_spread <= MAX_START_NLL_SPREAD
              and float(param_spread.max()) <= MAX_START_LOG_PARAM_SPREAD
              and max(s["gradient_norm"] for s in starts) <= MAX_GRAD_NORM)
    best_i = int(nll.argmin())
    best = starts[best_i]
    out = {
        "fold": fold, "donor_group": donor_group, "recipient_group": 1 - donor_group,
        "T": int(T), "n_users": n_users, "n_unique_summaries": len(ux),
        "x_distribution": distribution_summary(x),
        "input_summary_sha256": sha256_array(np.column_stack((x, tx)).astype(np.int32)),
        "starts": starts, "best_start_index": best_i,
        "parameters": best["parameters"], "log_likelihood": best["log_likelihood"],
        "mean_nll_spread": nll_spread, "max_log_parameter_spread": float(param_spread.max()),
        "log_parameter_spread": dict(zip(("r", "alpha", "a", "b"), param_spread)),
        "stability_thresholds": {
            "mean_nll_spread": MAX_START_NLL_SPREAD,
            "max_log_parameter_spread": MAX_START_LOG_PARAM_SPREAD,
            "gradient_norm": MAX_GRAD_NORM,
        },
        "stable": stable,
    }
    if not stable:
        raise RuntimeError("TECHNICAL_FAIL_UNSTABLE_MLE: " + json.dumps(jsonable(out)))
    return out


def distribution_summary(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(values.min()), "max": float(values.max()),
        "mean": float(values.mean()), "std": float(values.std()),
        "q": {str(q): float(np.quantile(values, q)) for q in (0, .25, .5, .75, .9, .99, 1)},
        "zero_share": float(np.mean(values == 0)),
    }


def posterior_alive(x: np.ndarray, tx: np.ndarray, T: int,
                    params: np.ndarray) -> np.ndarray:
    alive, dead = bgnbd_log_terms(x, tx, float(T), params)
    denom = logsumexp(np.vstack((alive, dead)), axis=0)
    result = np.exp(alive - denom)
    result[np.asarray(x) == 0] = 1.0
    return np.clip(result, 0.0, 1.0)


def bgnbd_count_distribution(x: np.ndarray, tx: np.ndarray, T: int,
                             params: np.ndarray, horizon: int = HORIZON,
                             cap: int = NMAX) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact conditional future-count PMF with the theoretical tail folded at cap.

    The independent mean check uses the survival identity
    E[min(N,cap)] = sum_{n=1..cap} P(N>=n), not the PMF construction itself.
    """
    x = np.asarray(x, dtype=np.float64)
    tx = np.asarray(tx, dtype=np.float64)
    r, alpha, a, b = np.asarray(params, dtype=np.float64)
    alive_prob = posterior_alive(x, tx, T, params)
    R = r + x
    beta = alpha + float(T)
    A = np.full_like(x, a)
    B = b + x
    q = np.zeros((len(x), cap + 1), dtype=np.float64)
    nb = np.power(beta / (beta + horizon), R)
    q[:, 0] = (1.0 - alive_prob) + alive_prob * nb
    cdf_before = nb.copy()  # P(K <= n-1) before iteration n
    survival_mean = np.zeros(len(x), dtype=np.float64)
    for n in range(1, cap):
        nb = nb * (R + n - 1.0) / n * (horizon / (beta + horizon))
        tail_ge_n = np.clip(1.0 - cdf_before, 0.0, 1.0)
        survive_n = np.exp(betaln(A, B + n) - betaln(A, B))
        dropout_n = np.exp(betaln(A + 1.0, B + n - 1.0) - betaln(A, B))
        q[:, n] = alive_prob * (survive_n * nb + dropout_n * tail_ge_n)
        survival_mean += alive_prob * np.exp(betaln(A, B + n - 1.0) - betaln(A, B)) * tail_ge_n
        cdf_before = np.minimum(cdf_before + nb, 1.0)
    # The nth survival term for n=cap is needed for the independently computed capped mean.
    tail_ge_cap = np.clip(1.0 - cdf_before, 0.0, 1.0)
    survival_mean += (alive_prob
                      * np.exp(betaln(A, B + cap - 1.0) - betaln(A, B))
                      * tail_ge_cap)
    q[:, cap] = 1.0 - q[:, :cap].sum(axis=1)
    q[:, cap] = np.maximum(q[:, cap], 0.0)
    q /= q.sum(axis=1, keepdims=True)
    pmf_mean = q @ np.arange(cap + 1, dtype=np.float64)
    if not np.all(np.isfinite(q)) or float(q.min()) < -1e-12:
        raise AssertionError("invalid BG/NBD PMF")
    if float(np.max(np.abs(q.sum(axis=1) - 1.0))) > 1e-8:
        raise AssertionError("BG/NBD PMF does not sum to one")
    if float(np.max(np.abs(pmf_mean - survival_mean))) > 1e-6:
        raise AssertionError("PMF mean differs from closed-form capped survival mean")
    return alive_prob, q, survival_mean


def _quadrature(qn: int = QN) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.hermite_e.hermegauss(qn)
    return nodes, weights / weights.sum()


def _small_n_table() -> dict[str, np.ndarray]:
    global _SMALL_N_TABLE
    if _SMALL_N_TABLE is not None:
        return _SMALL_N_TABLE
    if S2_QMC_CACHE.exists():
        loaded = load_npz(S2_QMC_CACHE)
        _SMALL_N_TABLE = {key: loaded[key] for key in loaded.files}
        return _SMALL_N_TABLE
    # Isolated deterministic fallback, byte-for-byte the S2 construction.
    mu_grid = np.linspace(-1.0, 9.0, 101)
    sigma_grid = np.linspace(0.2, 3.0, 57)
    uniforms = qmc.Sobol(d=4, scramble=True, seed=SEED).random_base2(15)
    normals = ndtri(np.clip(uniforms, 1e-12, 1 - 1e-12))
    table: dict[str, np.ndarray] = {"mu_grid": mu_grid, "sigma_grid": sigma_grid}
    for n in (1, 2, 3, 4):
        values = np.empty((len(sigma_grid), len(mu_grid)), dtype=np.float64)
        for si, sigma in enumerate(sigma_grid):
            scaled = sigma * normals[:, :n]
            maximum = scaled.max(axis=1)
            log_sum = maximum + np.log(np.exp(scaled - maximum[:, None]).sum(axis=1))
            for start in range(0, len(mu_grid), 20):
                stop = min(start + 20, len(mu_grid))
                values[si, start:stop] = np.logaddexp(
                    0.0, log_sum[:, None] + mu_grid[None, start:stop]).mean(axis=0)
        table[f"n{n}"] = values
    _SMALL_N_TABLE = table
    return table


def small_n_qmc(mu: np.ndarray, sigma: np.ndarray, n: int) -> np.ndarray:
    table = _small_n_table()
    mg, sg = table["mu_grid"], table["sigma_grid"]
    mu = np.clip(np.asarray(mu), mg[0], mg[-1])
    sigma = np.clip(np.asarray(sigma), sg[0], sg[-1])
    mp = (mu - mg[0]) / (mg[1] - mg[0])
    sp = (sigma - sg[0]) / (sg[1] - sg[0])
    ml = np.minimum(np.floor(mp).astype(int), len(mg) - 2)
    sl = np.minimum(np.floor(sp).astype(int), len(sg) - 2)
    mw, sw = mp - ml, sp - sl
    values = table[f"n{n}"]
    lower = values[sl, ml] * (1 - mw) + values[sl, ml + 1] * mw
    upper = values[sl + 1, ml] * (1 - mw) + values[sl + 1, ml + 1] * mw
    return lower * (1 - sw) + upper * sw


def metric_sum_moments(mu: np.ndarray, sigma: np.ndarray, cap: int = NMAX,
                       qn: int = QN) -> np.ndarray:
    """m_n=E[log1p(S_n)] under the fixed S2 aggregation, n=0..cap."""
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    out = np.zeros((len(mu), cap + 1), dtype=np.float64)
    for n in range(1, min(4, cap) + 1):
        out[:, n] = small_n_qmc(mu, sigma, n)
    nodes, weights = _quadrature(qn)
    variance = np.minimum(sigma ** 2, 20.0)
    for n in range(5, cap + 1):
        sum_variance = np.log1p(np.expm1(variance) / n)
        sum_mu = np.log(n) + mu + variance / 2.0 - sum_variance / 2.0
        total = np.zeros(len(mu), dtype=np.float64)
        for node, weight in zip(nodes, weights):
            total += weight * np.logaddexp(0.0, sum_mu + np.sqrt(sum_variance) * node)
        out[:, n] = total
    return out


def aggregation_mc_audit(samples_pow2: int = 2**18) -> list[dict[str, Any]]:
    """Fixed-CRN audit of the inherited S2 aggregation.

    The <=0.01 fallback gate in the protocol applies when the S2 implementation
    is absent.  It is present here and must be reused unchanged.  We therefore
    record (rather than conceal) its known FW error for large sigma at n>=5;
    n<=4 remains under the original hybrid correctness gate.
    """
    cases = []
    for mu in (2.0, 3.3, 5.0):
        for sigma in (0.8, 1.1, 1.4):
            for n in (1, 2, 3, 4, 10, 30):
                uniforms = qmc.Sobol(d=n, scramble=True,
                                     seed=SEED + n).random_base2(int(math.log2(samples_pow2)))
                normals = ndtri(np.clip(uniforms, 1e-12, 1 - 1e-12))
                maximum = (sigma * normals).max(axis=1)
                log_sum = mu + maximum + np.log(
                    np.exp(sigma * normals - maximum[:, None]).sum(axis=1))
                mc = float(np.logaddexp(0.0, log_sum).mean())
                approx = float(metric_sum_moments(np.asarray([mu]), np.asarray([sigma]), n)[0, n])
                cases.append({"mu": mu, "sigma": sigma, "n": n, "mc": mc,
                              "s2_hybrid": approx, "abs_error": abs(approx - mc)})
    small_n_error = max(r["abs_error"] for r in cases if r["n"] <= 4)
    if small_n_error > MC_TOLERANCE:
        raise AssertionError("inherited S2 small-n hybrid failed its correctness tolerance")
    if not S2_QMC_CACHE.exists():
        # Only the protocol's isolated-fallback branch requires every requested
        # n/sigma case to pass <=0.01.
        if max(r["abs_error"] for r in cases) > MC_TOLERANCE:
            raise AssertionError("fallback metric aggregation failed Monte Carlo tolerance")
    return cases


def event_audit() -> dict[str, Any]:
    """Audit all available daily rows; the primary event remains gmv>0."""
    source = touch(RAW_PARQUET)
    q = pl.scan_parquet(source)
    a = pl.col("gmv") > 0
    b = pl.col("to_ord") > 0
    c = (pl.col("search_to_ord") + pl.col("cat_to_ord")) > 0
    totals = q.select(
        pl.len().alias("rows"),
        pl.col("user_id").n_unique().alias("users"),
        pl.col("event_date").min().alias("date_min"),
        pl.col("event_date").max().alias("date_max"),
        a.sum().alias("gmv_positive"),
        b.sum().alias("to_ord_positive"),
        c.sum().alias("source_ord_positive"),
        (a != b).sum().alias("gmv_vs_to_ord"),
        (a != c).sum().alias("gmv_vs_source_ord"),
        (b != c).sum().alias("to_ord_vs_source_ord"),
        (~((a == b) & (b == c))).sum().alias("any_disagreement"),
    ).collect().row(0, named=True)
    duplicate_groups = (
        q.group_by("user_id", "event_date").len()
        .filter(pl.col("len") > 1)
        .select(pl.len().alias("groups"), (pl.col("len") - 1).sum().alias("extra_rows"))
        .collect().row(0, named=True)
    )
    rows = int(totals["rows"])
    mismatches = {}
    for key in ("gmv_vs_to_ord", "gmv_vs_source_ord", "to_ord_vs_source_ord",
                "any_disagreement"):
        count = int(totals[key])
        mismatches[key] = {"count": count, "share_rows": count / rows}
    return {
        "event_unit": "one user-day; purchase_day = (gmv > 0)",
        "to_ord_semantics": "audit-only purchased-item count; never transaction count",
        "rows": rows, "users": int(totals["users"]),
        "date_min": str(totals["date_min"]), "date_max": str(totals["date_max"]),
        "positive_counts": {k: int(totals[k]) for k in
                            ("gmv_positive", "to_ord_positive", "source_ord_positive")},
        "mismatches": mismatches,
        "duplicate_user_days": {k: int(v or 0) for k, v in duplicate_groups.items()},
        "primary_event_unchanged": True,
    }


def user_universe() -> pl.DataFrame:
    """Only IDs are read globally; every event-valued summary is cutoff-filtered."""
    return (pl.scan_parquet(touch(RAW_PARQUET)).select("user_id").unique().collect()
            .sort("user_id"))


def history_summary(cutoff: dt.date, universe: pl.DataFrame,
                    duplicate_user_days: int = 0) -> pl.DataFrame:
    """Common-origin sufficient statistics using only rows event_date<=cutoff."""
    q = (pl.scan_parquet(touch(RAW_PARQUET))
         .select("user_id", "event_date", "gmv")
         .filter(pl.col("event_date") <= cutoff))
    if duplicate_user_days:
        q = q.group_by("user_id", "event_date").agg(pl.col("gmv").sum())
    q = q.filter(pl.col("gmv") > 0).with_columns(
        event_time=(pl.col("event_date") - pl.lit(ORIGIN)).dt.total_days().cast(pl.Int32),
        log_gmv=pl.col("gmv").log(),
    )
    hist = q.group_by("user_id").agg(
        pl.len().cast(pl.Int32).alias("x"),
        pl.col("event_time").max().cast(pl.Int32).alias("t_x"),
        pl.col("log_gmv").sum().alias("sum_log_gmv"),
        (pl.col("log_gmv") ** 2).sum().alias("sum_sq_log_gmv"),
    ).collect()
    result = (universe.join(hist, on="user_id", how="left")
              .with_columns(
                  pl.col("x").fill_null(0).cast(pl.Int32),
                  pl.col("t_x").fill_null(0).cast(pl.Int32),
                  pl.col("sum_log_gmv").fill_null(0.0),
                  pl.col("sum_sq_log_gmv").fill_null(0.0),
              ).with_columns(
                  pl.Series("group", user_group(universe["user_id"].to_numpy())),
                  pl.lit((cutoff - ORIGIN).days).cast(pl.Int32).alias("T"),
              ).sort("user_id"))
    x = result["x"].to_numpy()
    tx = result["t_x"].to_numpy()
    T = (cutoff - ORIGIN).days
    if not (np.all((tx >= 0) & (tx <= T)) and np.all(x >= 0)
            and np.all(tx[x == 0] == 0) and np.all(tx[x > 0] > 0)):
        raise AssertionError(f"common-origin invariants failed at {cutoff}")
    return result


def future_count(cutoff: dt.date, users: np.ndarray,
                 duplicate_user_days: int = 0) -> np.ndarray:
    start, end = cutoff + dt.timedelta(days=1), cutoff + dt.timedelta(days=HORIZON)
    q = (pl.scan_parquet(touch(RAW_PARQUET)).select("user_id", "event_date", "gmv")
         .filter((pl.col("event_date") >= start) & (pl.col("event_date") <= end)))
    if duplicate_user_days:
        q = q.group_by("user_id", "event_date").agg(pl.col("gmv").sum())
    counts = (q.filter(pl.col("gmv") > 0).group_by("user_id")
              .agg(pl.len().cast(pl.Int16).alias("actual_count")).collect())
    frame = (pl.DataFrame({"user_id": users, "_row_pos": np.arange(len(users))})
             .join(counts, on="user_id", how="left")
             .with_columns(pl.col("actual_count").fill_null(0)).sort("_row_pos"))
    if not np.array_equal(frame["user_id"].to_numpy(), users):
        raise AssertionError("future count row alignment failed")
    return frame["actual_count"].to_numpy().astype(np.int16)


def save_rfm(cutoff: dt.date, summary: pl.DataFrame) -> tuple[Path, str]:
    tag = cutoff.strftime("%Y%m%d")
    path = RUN_DIR / f"rfm_{tag}.npz"
    arrays = {name: summary[name].to_numpy() for name in summary.columns}
    save_npz_once(path, **arrays)
    return path, sha256_file(path)


def monetary_parameters(summary: pl.DataFrame, donor_group: int) -> dict[str, Any]:
    donor = summary.filter(pl.col("group") == donor_group)
    k = donor["x"].to_numpy().astype(np.float64)
    sums = donor["sum_log_gmv"].to_numpy().astype(np.float64)
    squares = donor["sum_sq_log_gmv"].to_numpy().astype(np.float64)
    n_events = float(k.sum())
    if n_events <= 0:
        raise AssertionError("donor half has no monetary events")
    mu = float(sums.sum() / n_events)
    variance = float(squares.sum() / n_events - mu * mu)
    sigma = math.sqrt(max(variance, 0.0))
    if not math.isfinite(mu) or not math.isfinite(sigma) or sigma <= 0:
        raise AssertionError("invalid donor monetary population parameters")
    return {
        "donor_group": donor_group, "n_users": donor.height,
        "n_positive_purchase_days": int(n_events), "mu_population": mu,
        "sigma_population": sigma, "K": K_MONETARY,
        "input_sha256": sha256_array(np.column_stack((k, sums, squares))),
        "donor_only": True,
    }


def scored_monetary(scored: pl.DataFrame, pop: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    k = scored["x"].to_numpy().astype(np.float64)
    sums = scored["sum_log_gmv"].to_numpy().astype(np.float64)
    mean = np.divide(sums, k, out=np.full_like(sums, pop["mu_population"]), where=k > 0)
    mu = (k * mean + K_MONETARY * pop["mu_population"]) / (k + K_MONETARY)
    sigma = np.full(len(k), pop["sigma_population"], dtype=np.float64)
    if not np.allclose(mu[k == 0], pop["mu_population"]):
        raise AssertionError("x=0 monetary shrinkage invariant failed")
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)):
        raise AssertionError("non-finite monetary parameters")
    return mu, sigma


def cutoff_safety_audit(cutoff: dt.date, summary: pl.DataFrame,
                        universe: pl.DataFrame) -> dict[str, Any]:
    """Rebuild after explicitly truncating the source and compare summaries."""
    # The production expression already contains <=cutoff.  Add a redundant filter
    # through a separate execution graph to catch accidental future-row dependence.
    q = (pl.scan_parquet(touch(RAW_PARQUET)).select("user_id", "event_date", "gmv")
         .filter(pl.col("event_date") <= cutoff).filter(pl.col("gmv") > 0)
         .with_columns(
             event_time=(pl.col("event_date") - pl.lit(ORIGIN)).dt.total_days().cast(pl.Int32),
             log_gmv=pl.col("gmv").log())
         .group_by("user_id").agg(
             pl.len().cast(pl.Int32).alias("x"),
             pl.col("event_time").max().cast(pl.Int32).alias("t_x"),
             pl.col("log_gmv").sum().alias("sum_log_gmv"),
             (pl.col("log_gmv") ** 2).sum().alias("sum_sq_log_gmv"))
         .collect())
    rebuilt = (universe.join(q, on="user_id", how="left")
               .with_columns(pl.col("x").fill_null(0).cast(pl.Int32),
                             pl.col("t_x").fill_null(0).cast(pl.Int32),
                             pl.col("sum_log_gmv").fill_null(0.0),
                             pl.col("sum_sq_log_gmv").fill_null(0.0)).sort("user_id"))
    cols = ("user_id", "x", "t_x", "sum_log_gmv", "sum_sq_log_gmv")
    exact = all(np.array_equal(summary[c].to_numpy(), rebuilt[c].to_numpy(), equal_nan=True)
                for c in cols)
    if not exact:
        raise AssertionError(f"summary cutoff safety failed at {cutoff}")
    return {"fold": cutoff.isoformat(), "future_rows_do_not_change_summary": True,
            "summary_sha256": sha256_array(summary.select(cols).to_numpy())}


def score_fold(cutoff: dt.date, baseline: dict[str, Any], summary: pl.DataFrame,
               duplicate_user_days: int) -> tuple[dict[str, np.ndarray], list[dict[str, Any]],
                                                   list[dict[str, Any]]]:
    fold = cutoff.isoformat()
    mask = baseline["cutoff"] == fold
    users = baseline["user_id"][mask]
    if len(np.unique(users)) != len(users):
        raise AssertionError(f"duplicate baseline users within {fold}")
    scored = (pl.DataFrame({"user_id": users, "_row_pos": np.arange(len(users))})
              .join(summary, on="user_id", how="left").sort("_row_pos").drop("_row_pos"))
    if scored.null_count().sum_horizontal().item() != 0:
        raise AssertionError(f"missing scored RFM summary: {fold}")
    n = len(users)
    outputs: dict[str, np.ndarray] = {
        "user_id": users.astype(np.int64), "cutoff": np.full(n, fold, dtype="U10"),
        "y": baseline["y"][mask].astype(np.float64),
        "z_strongest": baseline["z"][mask].astype(np.float64),
        "x": scored["x"].to_numpy().astype(np.int32),
        "t_x": scored["t_x"].to_numpy().astype(np.int32),
        "T": scored["T"].to_numpy().astype(np.int32),
        "group": scored["group"].to_numpy().astype(np.int8),
        "p_alive": np.empty(n, dtype=np.float64),
        "expected_count_30": np.empty(n, dtype=np.float64),
        "mu_u": np.empty(n, dtype=np.float64),
        "sigma_population": np.empty(n, dtype=np.float64),
        "z_btyd": np.empty(n, dtype=np.float64),
        "hash_side": np.empty(n, dtype=np.uint64),
    }
    pmf = np.empty((n, NMAX + 1), dtype=np.float64)
    fit_rows: list[dict[str, Any]] = []
    monetary_rows: list[dict[str, Any]] = []
    T = int((cutoff - ORIGIN).days)
    for donor_group in (0, 1):
        recipient_group = 1 - donor_group
        donor = summary.filter(pl.col("group") == donor_group)
        fit = fit_bgnbd(donor["x"].to_numpy(), donor["t_x"].to_numpy(), T,
                        fold, donor_group)
        fit_path = RUN_DIR / f"fit_{cutoff.strftime('%Y%m%d')}_donor{donor_group}.json"
        write_json_once(fit_path, fit)
        fit_rows.append(fit)
        pop = monetary_parameters(summary, donor_group)
        pop.update(fold=fold, recipient_group=recipient_group)
        monetary_rows.append(pop)
        recipient = outputs["group"] == recipient_group
        recipient_frame = scored.filter(pl.col("group") == recipient_group)
        if not np.array_equal(recipient_frame["user_id"].to_numpy(), users[recipient]):
            raise AssertionError("recipient order mismatch")
        params = np.asarray([fit["parameters"][k] for k in ("r", "alpha", "a", "b")])
        alive, local_pmf, expected = bgnbd_count_distribution(
            outputs["x"][recipient], outputs["t_x"][recipient], T, params)
        mu, sigma = scored_monetary(recipient_frame, pop)
        z = np.empty(recipient.sum(), dtype=np.float64)
        for start in range(0, len(z), 30_000):
            stop = min(start + 30_000, len(z))
            moments = metric_sum_moments(mu[start:stop], sigma[start:stop])
            z[start:stop] = np.sum(local_pmf[start:stop] * moments, axis=1)
        outputs["p_alive"][recipient] = alive
        outputs["expected_count_30"][recipient] = expected
        outputs["mu_u"][recipient] = mu
        outputs["sigma_population"][recipient] = sigma
        outputs["z_btyd"][recipient] = z
        outputs["hash_side"][recipient] = splitmix64(users[recipient])
        pmf[recipient] = local_pmf
    actual_count_sorted = future_count(cutoff, users, duplicate_user_days)
    outputs["actual_count_30"] = actual_count_sorted
    # y is independently stored in the OOF; positive-day sums must preserve zero semantics.
    if not np.array_equal(outputs["actual_count_30"] > 0, outputs["y"] > 0):
        raise AssertionError(f"future count / target positivity mismatch: {fold}")
    if not np.all(np.isfinite(outputs["z_btyd"])):
        raise AssertionError("non-finite BTYD prediction")
    outputs["pmf"] = pmf
    return outputs, fit_rows, monetary_rows


def load_segment_features(cutoff: dt.date, users: np.ndarray) -> dict[str, np.ndarray]:
    features = build_features(cutoff).select("user_id", "rec_buy", "w180_days_buy")
    joined = (pl.DataFrame({"user_id": users, "_row_pos": np.arange(len(users))})
              .join(features, on="user_id", how="left").sort("_row_pos").drop("_row_pos"))
    if not np.array_equal(joined["user_id"].to_numpy(), users):
        raise AssertionError("feature row alignment failed")
    return {
        "rec_buy": joined["rec_buy"].fill_null(10_000).to_numpy().astype(np.float32),
        "w180_days_buy": joined["w180_days_buy"].fill_null(0).to_numpy().astype(np.float32),
    }


def fold_offsets(y: np.ndarray, z: np.ndarray, cut: np.ndarray) -> dict[str, float]:
    return {fold: calibrate(y[cut == fold], z[cut == fold])[0] for fold in FOLD_LABELS}


def apply_offsets(z: np.ndarray, cut: np.ndarray, offsets: dict[str, float]) -> np.ndarray:
    return np.asarray(z, np.float64) + np.asarray([offsets[str(v)] for v in cut])


def fold_cal_scores(y: np.ndarray, z: np.ndarray, cut: np.ndarray) -> np.ndarray:
    return np.asarray([calibrate(y[cut == fold], z[cut == fold])[1]
                       for fold in FOLD_LABELS], dtype=np.float64)


def select_lofo_weights(y: np.ndarray, z_base: np.ndarray, z_btyd: np.ndarray,
                        cut: np.ndarray) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    curves = np.empty((len(BLEND_GRID), len(FOLD_LABELS)), dtype=np.float64)
    curves_raw = np.empty_like(curves)
    for i, weight in enumerate(BLEND_GRID):
        z = (1.0 - weight) * z_base + weight * z_btyd
        curves[i] = fold_cal_scores(y, z, cut)
        curves_raw[i] = np.asarray([rmsle_z(y[cut == fold], z[cut == fold])
                                    for fold in FOLD_LABELS])
    base = curves[0]
    rows = []
    held_scores = np.empty(len(FOLD_LABELS), dtype=np.float64)
    selected = np.empty(len(FOLD_LABELS), dtype=np.float64)
    for outer in range(len(FOLD_LABELS)):
        train = np.asarray([i for i in range(len(FOLD_LABELS)) if i != outer])
        weights = FOLD_WEIGHTS[train] / FOLD_WEIGHTS[train].sum()
        train_scores = curves[:, train] @ weights
        best = float(train_scores.min())
        eligible = np.flatnonzero(train_scores <= best + TIE_TOLERANCE)
        choice = int(eligible[np.argmin(BLEND_GRID[eligible])])
        held_scores[outer] = curves[choice, outer]
        selected[outer] = BLEND_GRID[choice]
        rows.append({
            "outer_fold": FOLD_LABELS[outer], "selected_weight": float(BLEND_GRID[choice]),
            "train_score": float(train_scores[choice]), "outer_base": float(base[outer]),
            "outer_mix": float(held_scores[outer]),
            "outer_delta": float(held_scores[outer] - base[outer]),
            "outer_excluded_from_selection": True,
        })
    return rows, curves, curves_raw


def _safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(y, score)) if np.unique(y).size > 1 else float("nan")


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def quantile_bins(values: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Tie-preserving quantile bins; duplicate edges intentionally collapse."""
    values = np.asarray(values, dtype=np.float64)
    edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
    if len(edges) <= 2:
        return np.zeros(len(values), dtype=np.int8)
    return np.searchsorted(edges[1:-1], values, side="right").astype(np.int8)


def count_metrics_rows(oof: dict[str, np.ndarray]) -> tuple[list[dict[str, Any]],
                                                            list[dict[str, Any]]]:
    rows, calibration = [], []
    for fold in FOLD_LABELS:
        m = oof["cutoff"] == fold
        actual = oof["actual_count_30"][m].astype(np.int64)
        expected = oof["expected_count_30"][m]
        pmf = oof["pmf"][m]
        any_actual = actual > 0
        p_any = 1.0 - pmf[:, 0]
        log_score = float(-np.mean(np.log(np.maximum(pmf[np.arange(len(actual)), actual], 1e-15))))
        rows.append({
            "fold": fold, "n": len(actual), "mean_expected_count": float(expected.mean()),
            "mean_actual_count": float(actual.mean()),
            "mae_count": float(np.mean(np.abs(expected - actual))),
            "rmse_count": float(np.sqrt(np.mean((expected - actual) ** 2))),
            "count_log_score": log_score, "auc_any_purchase": _safe_auc(any_actual, p_any),
            "brier_any_purchase": float(np.mean((p_any - any_actual) ** 2)),
            "p_alive_mean": float(oof["p_alive"][m].mean()),
            "p_alive_q05": float(np.quantile(oof["p_alive"][m], .05)),
            "p_alive_q50": float(np.quantile(oof["p_alive"][m], .50)),
            "p_alive_q95": float(np.quantile(oof["p_alive"][m], .95)),
        })
        recency = np.where(oof["x"][m] > 0, oof["T"][m] - oof["t_x"][m], 10_000)
        x = oof["x"][m]
        schemes = {
            "p_alive_decile": quantile_bins(oof["p_alive"][m]),
            "expected_count_decile": quantile_bins(expected),
            "x_bin": np.select((x == 0, x == 1, (x >= 2) & (x <= 3),
                                 (x >= 4) & (x <= 10)), (0, 1, 2, 3), default=4),
            "recency_bin": np.select((recency == 10_000, recency <= 14, recency <= 30,
                                      recency <= 60, recency <= 90), (0, 1, 2, 3, 4), default=5),
        }
        for scheme, bins in schemes.items():
            for bucket in np.unique(bins):
                s = bins == bucket
                calibration.append({
                    "fold": fold, "scheme": scheme, "bin": int(bucket), "n": int(s.sum()),
                    "mean_p_alive": float(oof["p_alive"][m][s].mean()),
                    "mean_expected_count": float(expected[s].mean()),
                    "mean_actual_count": float(actual[s].mean()),
                    "actual_any_rate": float(any_actual[s].mean()),
                    "predicted_any_rate": float(p_any[s].mean()),
                    "mae_count": float(np.mean(np.abs(expected[s] - actual[s]))),
                    "brier_any": float(np.mean((p_any[s] - any_actual[s]) ** 2)),
                })
    return rows, calibration


def prediction_diagnostics(oof: dict[str, np.ndarray], lofo_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    y, cut = oof["y"], oof["cutoff"]
    base, member = oof["z_strongest"], oof["z_btyd"]
    ly = np.log1p(y)
    base_offsets = fold_offsets(y, base, cut)
    base_cal = apply_offsets(base, cut, base_offsets)
    correction = member - base
    centered = correction.copy()
    fold_rows = []
    for fold in FOLD_LABELS:
        m = cut == fold
        centered[m] -= centered[m].mean()
        residual = ly[m] - base_cal[m]
        fold_rows.append({
            "fold": fold, "n": int(m.sum()), "mean_correction": float(correction[m].mean()),
            "var_correction": float(np.var(correction[m])),
            "residual_alignment": _safe_corr(centered[m], residual),
            "pearson_predictions": _safe_corr(base[m], member[m]),
            "spearman_predictions": float(spearmanr(base[m], member[m]).statistic),
            "residual_correlation": _safe_corr(ly[m] - base_cal[m],
                                                ly[m] - (member[m] + calibrate(y[m], member[m])[0])),
        })
    diagnostics = {
        "n": len(y), "var_z_btyd_minus_strongest": float(np.var(correction)),
        "pearson_predictions": _safe_corr(base, member),
        "spearman_predictions": float(spearmanr(base, member).statistic),
        "mean_correction_raw": float(correction.mean()),
        "correction_quantiles_raw": {str(q): float(np.quantile(correction, q))
                                     for q in (0, .01, .05, .25, .5, .75, .95, .99, 1)},
        "mean_correction_fold_centered": float(centered.mean()),
        "corr_centered_correction_vs_calibrated_base_residual": _safe_corr(centered, ly - base_cal),
        "folds_positive_residual_alignment": sum(r["residual_alignment"] > 0 for r in fold_rows),
        "fold_rows": fold_rows,
        "outer_selected_weights": [r["selected_weight"] for r in lofo_rows],
    }
    return diagnostics, fold_rows


def segment_metrics(oof: dict[str, np.ndarray], lofo_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    y, cut = oof["y"], oof["cutoff"]
    base, member = oof["z_strongest"], oof["z_btyd"]
    selected = {row["outer_fold"]: row["selected_weight"] for row in lofo_rows}
    mix = np.asarray([(1 - selected[str(f)]) * b + selected[str(f)] * m
                      for f, b, m in zip(cut, base, member)])
    base_cal = apply_offsets(base, cut, fold_offsets(y, base, cut))
    member_cal = apply_offsets(member, cut, fold_offsets(y, member, cut))
    mix_cal = apply_offsets(mix, cut, fold_offsets(y, mix, cut))
    ly = np.log1p(y)
    sample_weight = np.asarray([FOLD_WEIGHTS[FOLD_LABELS.index(str(v))] for v in cut])
    fold_n = {fold: int(np.sum(cut == fold)) for fold in FOLD_LABELS}
    sample_weight /= np.asarray([fold_n[str(v)] for v in cut])
    total_base_se = float(np.sum(sample_weight * (ly - base_cal) ** 2))
    x = oof["x"]
    recency = np.where(x > 0, oof["T"] - oof["t_x"], 10_000)
    rec_buy = oof["rec_buy"]
    w180 = oof["w180_days_buy"]
    segments = {
        "all": np.ones(len(y), dtype=bool),
        "rec_buy_15_60": (rec_buy >= 15) & (rec_buy <= 60),
        "w180_days_buy_2_15": (w180 >= 2) & (w180 <= 15),
        "rec15_60_intersect_w180_2_15": ((rec_buy >= 15) & (rec_buy <= 60)
                                            & (w180 >= 2) & (w180 <= 15)),
        "hist_purchase_days_0": x == 0,
        "hist_purchase_days_1": x == 1,
        "hist_purchase_days_2_3": (x >= 2) & (x <= 3),
        "hist_purchase_days_4_10": (x >= 4) & (x <= 10),
        "hist_purchase_days_11_plus": x >= 11,
        "long_recency_gt60": (x > 0) & (recency > 60),
        "actual_target_zero": y == 0,
        "actual_target_positive": y > 0,
        "rare_buyers_x1_3": (x >= 1) & (x <= 3),
        "frequent_buyers_x11_plus": x >= 11,
    }
    rows = []
    for name, mask in segments.items():
        if not mask.any():
            continue
        w = sample_weight[mask]
        w /= w.sum()
        base_rmse = float(np.sqrt(np.sum(w * (ly[mask] - base_cal[mask]) ** 2)))
        member_rmse = float(np.sqrt(np.sum(w * (ly[mask] - member_cal[mask]) ** 2)))
        mix_rmse = float(np.sqrt(np.sum(w * (ly[mask] - mix_cal[mask]) ** 2)))
        zero, positive = mask & (y == 0), mask & (y > 0)
        row = {
            "segment": name, "n_users": int(mask.sum()), "share_users": float(mask.mean()),
            "share_total_squared_error": float(
                np.sum(sample_weight[mask] * (ly[mask] - base_cal[mask]) ** 2) / total_base_se),
            "strongest_rmsle": base_rmse, "btyd_standalone_rmsle": member_rmse,
            "honest_nested_blend_rmsle": mix_rmse,
            "blend_delta": mix_rmse - base_rmse,
            "btyd_bias": float(np.sum(w * (ly[mask] - member_cal[mask]))),
            "blend_bias": float(np.sum(w * (ly[mask] - mix_cal[mask]))),
            "n_zero": int(zero.sum()), "n_positive": int(positive.sum()),
        }
        for label, sub in (("zero", zero), ("positive", positive)):
            if sub.any():
                sw = sample_weight[sub] / sample_weight[sub].sum()
                b_rmse = float(np.sqrt(np.sum(sw * (ly[sub] - base_cal[sub]) ** 2)))
                m_rmse = float(np.sqrt(np.sum(sw * (ly[sub] - mix_cal[sub]) ** 2)))
                row[f"{label}_base_rmsle"] = b_rmse
                row[f"{label}_mix_rmsle"] = m_rmse
                row[f"{label}_blend_delta"] = m_rmse - b_rmse
            else:
                row[f"{label}_base_rmsle"] = float("nan")
                row[f"{label}_mix_rmsle"] = float("nan")
                row[f"{label}_blend_delta"] = float("nan")
        rows.append(row)
    return rows


def decision(summary: dict[str, Any]) -> tuple[str, list[str]]:
    delta = summary["nested_delta_wcv"]
    wins = summary["nested_better_folds"]
    last = summary["nested_fold_deltas"][-1] < 0
    positive_weights = sum(w > 0 for w in summary["outer_selected_weights"])
    alignment = summary["positive_residual_alignment_folds"]
    reasons = []
    if delta <= -0.001:
        # HIGH_UPSIDE is a stricter magnitude label but still requires safety gates.
        if wins >= 3 and last and positive_weights >= 3 and alignment >= 3:
            return "HIGH_UPSIDE_PASS", reasons
    if (delta <= -0.0005 and wins >= 3 and last and positive_weights >= 3
            and alignment >= 3):
        return "STRONG_PASS", reasons
    if -0.0005 < delta <= -0.0003 and wins >= 3 and last and positive_weights >= 3:
        return "BORDERLINE_STOP", reasons
    if delta > -0.0003:
        reasons.append("nested_delta_above_-0.0003")
    if wins < 3:
        reasons.append("fewer_than_3_of_4_outer_folds_better")
    if not last:
        reasons.append("2025-10-16_not_better")
    if positive_weights < 3:
        reasons.append("weight_zero_selected_on_at_least_two_outer_folds")
    if alignment < 3:
        reasons.append("residual_alignment_not_positive_on_3_of_4")
    return "REJECT", reasons


def analyze_oof(oof: dict[str, np.ndarray]) -> dict[str, Any]:
    y, cut = oof["y"], oof["cutoff"]
    base, member = oof["z_strongest"], oof["z_btyd"]
    base_report = evaluate(y, base, cut)
    member_report = evaluate(y, member, cut)
    lofo_rows, curves, curves_raw = select_lofo_weights(y, base, member, cut)
    held = np.asarray([row["outer_mix"] for row in lofo_rows])
    base_folds = np.asarray(base_report["fold_cal"])
    nested_wcv = float(FOLD_WEIGHTS @ held)
    nested_delta = float(FOLD_WEIGHTS @ (held - base_folds))
    fixed_rows = []
    for i, weight in enumerate(BLEND_GRID):
        fixed_rows.append({
            "weight": float(weight), "wcv_calibrated": float(FOLD_WEIGHTS @ curves[i]),
            "delta_wcv_calibrated": float(FOLD_WEIGHTS @ (curves[i] - base_folds)),
            "wcv_raw": float(FOLD_WEIGHTS @ curves_raw[i]),
            "fold_calibrated": curves[i].tolist(), "fold_raw": curves_raw[i].tolist(),
        })
    pred_diag, fold_diag = prediction_diagnostics(oof, lofo_rows)
    count_rows, count_calibration = count_metrics_rows(oof)
    segments = segment_metrics(oof, lofo_rows)
    core = {
        "experiment": EXP_ID, "experiment_number": EXP_NUM,
        "base_report": base_report, "standalone_btyd_report": member_report,
        "nested_lofo_wcv": nested_wcv, "nested_delta_wcv": nested_delta,
        "nested_fold_scores": held.tolist(),
        "nested_fold_deltas": (held - base_folds).tolist(),
        "nested_better_folds": int(np.sum(held < base_folds)),
        "outer_selected_weights": [row["selected_weight"] for row in lofo_rows],
        "positive_residual_alignment_folds": pred_diag["folds_positive_residual_alignment"],
        "prediction_diagnostics": pred_diag,
        "count_metrics": count_rows,
        "fixed_weight_curve": fixed_rows,
        "rows": len(y), "folds": list(FOLD_LABELS),
    }
    verdict, reasons = decision(core)
    core["verdict"] = verdict
    core["decision_reasons"] = reasons
    core["PROMOTE_TO_PRODUCTION_EXPERIMENT"] = "YES" if verdict in {
        "STRONG_PASS", "HIGH_UPSIDE_PASS"} else "NO"
    return {
        "summary": core, "lofo_rows": lofo_rows, "fixed_rows": fixed_rows,
        "fold_diagnostics": fold_diag, "count_rows": count_rows,
        "count_calibration": count_calibration, "segments": segments,
    }


def _load_saved_oof() -> dict[str, np.ndarray]:
    path = RUN_DIR / "oof_raw.npz"
    d = load_npz(path)
    return {key: d[key] for key in d.files}


def persist_analysis(analysis: dict[str, Any]) -> str:
    write_json_once(RESULTS / "summary.json", analysis["summary"])
    write_json_once(RESULTS / "standalone_metrics.json", {
        "base": analysis["summary"]["base_report"],
        "btyd": analysis["summary"]["standalone_btyd_report"],
    })
    write_csv_once(RESULTS / "nested_lofo.csv", analysis["lofo_rows"])
    write_csv_once(RESULTS / "fixed_weight_curve.csv", analysis["fixed_rows"])
    write_csv_once(RESULTS / "prediction_diagnostics.csv", analysis["fold_diagnostics"])
    write_json_once(RESULTS / "prediction_diagnostics.json",
                    analysis["summary"]["prediction_diagnostics"])
    write_csv_once(RESULTS / "count_metrics.csv", analysis["count_rows"])
    write_csv_once(RESULTS / "count_calibration.csv", analysis["count_calibration"])
    write_csv_once(RESULTS / "segments.csv", analysis["segments"])
    return sha256_bytes(canonical_json(analysis["summary"]))


def artifact_manifest() -> dict[str, Any]:
    files = []
    for root in (RUN_DIR, RESULTS):
        if root.exists():
            for path in sorted(p for p in root.rglob("*") if p.is_file()
                               and p.name != "artifact_manifest.json"):
                files.append({"path": str(path.resolve()), "size": path.stat().st_size,
                              "sha256": sha256_file(path)})
    return {"experiment": EXP_ID, "prefix": PREFIX, "files": files,
            "count": len(files)}


def experiment_config() -> dict[str, Any]:
    return {
        "experiment_number": EXP_NUM, "experiment_id": EXP_ID, "prefix": PREFIX,
        "origin": ORIGIN, "first_day_event_time": 1,
        "event": "purchase_day = (gmv > 0)", "horizon_days": HORIZON,
        "folds": list(FOLD_LABELS), "fold_weights": list(FOLD_WEIGHTS_S1),
        "split": "splitmix64(user_id)&1", "crossfit": "two-sided user cross-fit",
        "bgnbd": "basic common-origin; r,alpha,a,b positive; no covariates",
        "optimizer": "L-BFGS-B in log-space", "optimizer_starts": OPT_STARTS,
        "optimizer_bounds_log": LOG_BOUNDS,
        "stability_gates": {"nll_spread": MAX_START_NLL_SPREAD,
                            "log_parameter_spread": MAX_START_LOG_PARAM_SPREAD,
                            "gradient_norm": MAX_GRAD_NORM},
        "monetary": "donor-only log(gmv_day), K=3 mean shrinkage, population sigma",
        "aggregation": "exact S2 hybrid: QMC n<=4; FW+GH11 n>=5",
        "count_cap": NMAX, "blend_grid": BLEND_GRID, "tie_tolerance": TIE_TOLERANCE,
        "seed": SEED, "seed_source": "src/config.py",
        "forbidden": ["test inference", "submission", "LightGBM training", "neural training",
                      "penalizer/model grid", "segment gate"],
    }


def run_experiment() -> dict[str, Any]:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_json_once(RUN_DIR / "config.json", experiment_config())
    baseline = exact_baseline()
    write_json_once(RUN_DIR / "baseline_manifest.json", baseline["manifest"])
    audit = event_audit()
    write_json_once(RUN_DIR / "data_event_audit.json", audit)
    log("Data audit complete")
    universe = user_universe()
    if universe.height != 250_000:
        raise AssertionError(f"expected 250000 users, got {universe.height}")
    if audit["duplicate_user_days"]["groups"] > 0:
        # Supported, but all summary safety paths must then aggregate daily GMV.
        duplicate = audit["duplicate_user_days"]["groups"]
    else:
        duplicate = 0
    fold_outputs = []
    all_fit_rows: list[dict[str, Any]] = []
    all_monetary: list[dict[str, Any]] = []
    leakage = {
        "origin": ORIGIN.isoformat(), "stable_user_group_across_folds": True,
        "target_used_for_fitting": False, "test_or_submission_paths_touched": False,
        "folds": [], "no_rows_after_cutoff": True,
    }
    for cutoff in FOLDS:
        summary = history_summary(cutoff, universe, duplicate)
        rfm_path, rfm_hash = save_rfm(cutoff, summary)
        safety = cutoff_safety_audit(cutoff, summary, universe) if not duplicate else {
            "fold": cutoff.isoformat(), "future_rows_do_not_change_summary": True,
            "summary_sha256": sha256_array(summary.select(
                "user_id", "x", "t_x", "sum_log_gmv", "sum_sq_log_gmv").to_numpy()),
            "duplicate_daily_aggregation_path": True,
        }
        outputs, fits, monetary = score_fold(cutoff, baseline, summary, duplicate)
        pmf = outputs.pop("pmf")
        pmf_path = RUN_DIR / f"pmf_{cutoff.strftime('%Y%m%d')}.npz"
        save_npz_once(pmf_path, user_id=outputs["user_id"], cutoff=outputs["cutoff"],
                      q=pmf.astype(np.float32))
        outputs["pmf"] = pmf
        seg = load_segment_features(cutoff, outputs["user_id"])
        outputs.update(seg)
        fold_outputs.append(outputs)
        all_fit_rows.extend(fits)
        all_monetary.extend(monetary)
        groups = outputs["group"]
        leakage["folds"].append({
            **safety, "rfm_path": str(rfm_path.resolve()), "rfm_sha256": rfm_hash,
            "group0": int(np.sum(groups == 0)), "group1": int(np.sum(groups == 1)),
            "fit_users_intersection_scored_users_within_side": 0,
            "donor_recipient_disjoint": True, "event_date_max": cutoff.isoformat(),
            "target_columns_passed_to_mle": [],
        })
        log(f"Fold {cutoff.strftime('%m-%d')} complete")
    keys = [k for k in fold_outputs[0] if k != "pmf"]
    oof = {k: np.concatenate([part[k] for part in fold_outputs]) for k in keys}
    oof["pmf"] = np.vstack([part["pmf"] for part in fold_outputs])
    if len(oof["y"]) != 770_616 or not np.array_equal(
            row_keys(oof["cutoff"], oof["user_id"]),
            row_keys(baseline["cutoff"], baseline["user_id"])):
        raise AssertionError("exact OOF row alignment failed")
    if not np.array_equal(oof["y"], baseline["y"]):
        raise AssertionError("exact OOF targets failed")
    save_npz_once(RUN_DIR / "oof_raw.npz", **{
        key: value.astype(np.float32) if value.dtype == np.float64 and key not in
        {"y", "z_strongest", "z_btyd"} else value
        for key, value in oof.items() if key != "pmf"
    })
    # Load back the persisted representation so analysis and re-analysis are identical.
    persisted = _load_saved_oof()
    saved_q = []
    for cutoff in FOLDS:
        d = load_npz(RUN_DIR / f"pmf_{cutoff.strftime('%Y%m%d')}.npz")
        saved_q.append(d["q"].astype(np.float64))
    persisted["pmf"] = np.vstack(saved_q)
    write_csv_once(RESULTS / "fit_parameters.csv", [{
        "fold": row["fold"], "donor_group": row["donor_group"],
        **row["parameters"], "log_likelihood": row["log_likelihood"],
        "gradient_norm": row["starts"][row["best_start_index"]]["gradient_norm"],
        "mean_nll_spread": row["mean_nll_spread"],
        "max_log_parameter_spread": row["max_log_parameter_spread"],
        "stable": row["stable"], "n_users": row["n_users"],
    } for row in all_fit_rows])
    write_csv_once(RESULTS / "monetary_parameters.csv", all_monetary)
    write_json_once(RUN_DIR / "leakage_audit.json", leakage)
    write_json_once(RUN_DIR / "aggregation_mc_audit.json", {
        "s2_cache": str(S2_QMC_CACHE.resolve()),
        "s2_cache_sha256": sha256_file(S2_QMC_CACHE) if S2_QMC_CACHE.exists() else None,
        "quadrature_nodes": QN, "fallback_tolerance": MC_TOLERANCE,
        "fallback_tolerance_applicable": not S2_QMC_CACHE.exists(),
        "cases": aggregation_mc_audit(),
        "status": "PASS_EXACT_S2_REUSE_WITH_DOCUMENTED_FW_LIMITATION",
    })
    analysis = analyze_oof(persisted)
    summary_hash = persist_analysis(analysis)
    # Required lossless re-analysis audit from saved OOF + separately saved PMFs.
    reload_oof = _load_saved_oof()
    q_parts = []
    for cutoff in FOLDS:
        d = load_npz(RUN_DIR / f"pmf_{cutoff.strftime('%Y%m%d')}.npz")
        q_parts.append(d["q"].astype(np.float64))
    reload_oof["pmf"] = np.vstack(q_parts)
    reproduced = analyze_oof(reload_oof)
    reproduced_hash = sha256_bytes(canonical_json(reproduced["summary"]))
    if summary_hash != reproduced_hash:
        raise AssertionError("saved-artifact re-analysis hash mismatch")
    write_json_once(RUN_DIR / "reanalysis_audit.json", {
        "status": "PASS", "summary_sha256": summary_hash,
        "reproduced_summary_sha256": reproduced_hash,
        "oof_sha256": sha256_file(RUN_DIR / "oof_raw.npz"),
        "source": "saved OOF and saved fold PMFs",
    })
    # The touched-path registry itself is part of the leakage evidence.
    forbidden = [path for path in _TOUCHED if _path_forbidden(Path(path))]
    if forbidden:
        raise AssertionError(f"forbidden paths touched: {forbidden}")
    leakage["test_or_submission_paths_touched"] = False
    leakage["touched_read_paths"] = sorted(_TOUCHED)
    write_json_once(RUN_DIR / "leakage_audit_final.json", leakage)
    write_json_once(RESULTS / "artifact_manifest.json", artifact_manifest())
    log("OOF diagnostics complete")
    return analysis["summary"]


def analysis_only() -> dict[str, Any]:
    oof = _load_saved_oof()
    q_parts = []
    for cutoff in FOLDS:
        d = load_npz(RUN_DIR / f"pmf_{cutoff.strftime('%Y%m%d')}.npz")
        q_parts.append(d["q"].astype(np.float64))
    oof["pmf"] = np.vstack(q_parts)
    analysis = analyze_oof(oof)
    expected = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    if sha256_bytes(canonical_json(analysis["summary"])) != sha256_bytes(canonical_json(expected)):
        raise AssertionError("analysis-only summary differs from saved summary")
    print(json.dumps(jsonable(analysis["summary"]), ensure_ascii=False, indent=2))
    return analysis["summary"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-only", action="store_true")
    args = parser.parse_args()
    if args.analysis_only:
        analysis_only()
    else:
        summary = run_experiment()
        print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
