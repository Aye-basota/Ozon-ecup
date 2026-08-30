"""EXP087: cross-user dynamic-factor residual experiment.

The mandatory oracle stage is deliberately self-contained and CPU-only.  It
uses the audited raw-derived dense panel from EXP075/EXP085, the exact purged
fold IDs/targets and the faithful five-family production reconstruction from
EXP082.  No target is touched until the factor representation and its rank have
been frozen for a fold.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OZON = Path(r"C:\Users\Admin\Desktop\OZON-E-CUP")
PANEL_PATH = OZON / "data" / "processed" / "seq_panel_v1.npy"
UID_PATH = OZON / "data" / "processed" / "seq_uid_v1.npy"
EXP082 = ROOT / "research" / "new_directions" / "EXP082_PURGED_TEMPORAL_RESIDUAL"
PROD = EXP082 / "production_components"

DATA_START = np.datetime64("2025-01-01")
FOLDS = ("2025-07-03", "2025-08-07", "2025-09-11", "2025-10-16")
FOLD_WEIGHTS = {fold: float(2**i) for i, fold in enumerate(FOLDS)}
PROD_WEIGHTS = {"cap": 0.10, "unc": 0.20, "dist": 0.25, "seq": 0.225, "etx": 0.225}

PANEL_CHANNELS = (
    "present", "cat", "buy", "ponly", "searches", "search_to_cart",
    "search_to_ord", "cat_to_cart", "cat_to_ord", "to_cart", "to_ord",
    "gmv_search", "gmv_cat", "gmv",
)
CHANNELS = (
    "searches", "cat", "search_to_cart", "search_to_ord",
    "cat_to_cart", "cat_to_ord", "present", "buy",
)
COUNT_CHANNELS = frozenset(CHANNELS[:6])
CHANNEL_INDEX = {name: PANEL_CHANNELS.index(name) for name in CHANNELS}

HISTORY_DAYS = 180
ORACLE_DAYS = 30
MAX_RANK = 16
PA_USERS = 1024
PA_REPS = 100
PA_PERCENTILE = 99.0
SEED = 870029
EPS = 1e-12
ORACLE_GATE = 0.0010
T0 = time.time()


def log(*parts: object) -> None:
    print(f"[{time.time() - T0:8.1f}s]", *parts, flush=True)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, np.float64)
    return float(np.sqrt(np.mean(x * x)))


def corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, np.float64) - float(np.mean(x))
    y = np.asarray(y, np.float64) - float(np.mean(y))
    den = math.sqrt(float(x @ x) * float(y @ y))
    return 0.0 if den <= 1e-300 else float(x @ y / den)


def day_index(cutoff: str) -> int:
    return int((np.datetime64(cutoff) - DATA_START).astype(int))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load_production_fold(fold: str) -> dict[str, np.ndarray]:
    parts: dict[str, np.ndarray] = {}
    ref_uid: np.ndarray | None = None
    ref_target: np.ndarray | None = None
    for family in PROD_WEIGHTS:
        record = load_npz(PROD / f"{family}_{fold}.npz")
        uid = record["user_id"].astype(np.int64, copy=False)
        target = record["target_log"].astype(np.float64, copy=False)
        if ref_uid is None:
            ref_uid, ref_target = uid, target
        elif not np.array_equal(uid, ref_uid) or not np.allclose(target, ref_target, atol=1e-10, rtol=0):
            raise AssertionError(f"production alignment failed for {fold}/{family}")
        parts[family] = record["z"].astype(np.float64, copy=False)
    assert ref_uid is not None and ref_target is not None
    components = np.column_stack([parts[name] for name in PROD_WEIGHTS])
    baseline = sum(PROD_WEIGHTS[name] * parts[name] for name in PROD_WEIGHTS)
    # Baseline is included explicitly even though it lies in the component span;
    # this reproduces the requested full production-like historical span.
    span = np.column_stack([np.ones(len(ref_uid)), components, baseline])
    return {
        "user_id": ref_uid,
        "target_log": ref_target,
        "components": components,
        "baseline": baseline,
        "residual": ref_target - baseline,
        "span": span,
    }


def robust_scale(x: np.ndarray, sample_rows: np.ndarray) -> tuple[float, dict[str, float]]:
    sample = np.asarray(x[sample_rows], np.float32).reshape(-1)
    q25, median, q75 = np.quantile(sample, [0.25, 0.50, 0.75])
    scale = float((q75 - q25) / 1.349)
    fallback = False
    if not np.isfinite(scale) or scale < 1e-6:
        scale = float(np.sqrt(np.mean(np.asarray(sample, np.float64) ** 2)))
        fallback = True
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
        fallback = True
    return scale, {
        "q25": float(q25), "median": float(median), "q75": float(q75),
        "robust_scale": scale, "fallback": bool(fallback),
        "scale_sample_rows": int(len(sample_rows)),
    }


def normalize_channel(
    panel: np.ndarray,
    rows: np.ndarray,
    hist_slice: slice,
    future_slice: slice,
    channel: str,
    scale_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], np.ndarray]:
    idx = CHANNEL_INDEX[channel]
    hist = np.asarray(panel[rows, hist_slice, idx], dtype=np.float32)
    future = np.asarray(panel[rows, future_slice, idx], dtype=np.float32)
    if channel in COUNT_CHANNELS:
        np.log1p(hist, out=hist)
        np.log1p(future, out=future)
    user_mean = hist.mean(axis=1, dtype=np.float64).astype(np.float32)
    hist -= user_mean[:, None]
    hist_day_mean = hist.mean(axis=0, dtype=np.float64).astype(np.float32)
    hist -= hist_day_mean[None, :]
    scale, diag = robust_scale(hist, scale_rows)
    hist /= np.float32(scale)

    # Frozen historical user mean/scale; future cohort-day centering is allowed
    # only in the oracle diagnostic because the future channels are observed.
    future -= user_mean[:, None]
    future_day_mean = future.mean(axis=0, dtype=np.float64).astype(np.float32)
    future -= future_day_mean[None, :]
    future /= np.float32(scale)
    diag.update({
        "historical_user_mean_mean": float(np.mean(user_mean)),
        "historical_user_mean_sd": float(np.std(user_mean)),
        "historical_day_mean_abs_max_before_removal": float(np.max(np.abs(hist_day_mean))),
        "future_day_mean_abs_max_before_removal": float(np.max(np.abs(future_day_mean))),
        "post_normalization_day_mean_abs_max": float(np.max(np.abs(hist.mean(axis=0)))),
        "history_rms": rms(hist),
        "future_oracle_rms": rms(future),
    })
    return hist, future, diag, user_mean


def parallel_analysis(
    matrices: list[np.ndarray], sample_rows: np.ndarray, seed: int,
) -> dict[str, Any]:
    """Target-free parallel analysis using within-day/channel permutations.

    A random circular shift is a genuine permutation of users.  Independent
    shifts for every (day, channel) destroy identity synchrony while preserving
    every univariate day/channel empirical distribution exactly.
    """
    rng = np.random.default_rng(seed)
    blocks: list[np.ndarray] = []
    for matrix in matrices:
        block = np.asarray(matrix[sample_rows].T, np.float32)
        block -= block.mean(axis=1, keepdims=True)
        blocks.append(block)
    real_gram = np.zeros((HISTORY_DAYS, HISTORY_DAYS), np.float64)
    for block in blocks:
        real_gram += np.asarray(block @ block.T, np.float64)
    real_values = np.linalg.eigvalsh(real_gram)[::-1]
    real_sv = np.sqrt(np.maximum(real_values, 0.0))

    n = len(sample_rows)
    base = np.arange(n, dtype=np.int32)[None, :]
    null_spectra = np.empty((PA_REPS, MAX_RANK), np.float64)
    for rep in range(PA_REPS):
        gram = np.zeros_like(real_gram)
        for block in blocks:
            shifts = rng.integers(0, n, size=HISTORY_DAYS, dtype=np.int32)[:, None]
            take = (base + shifts) % n
            shuffled = np.take_along_axis(block, take, axis=1)
            gram += np.asarray(shuffled @ shuffled.T, np.float64)
        values = np.linalg.eigvalsh(gram)[::-1][:MAX_RANK]
        null_spectra[rep] = np.sqrt(np.maximum(values, 0.0))
        if (rep + 1) % 20 == 0:
            log("parallel analysis", rep + 1, "/", PA_REPS)
    null99 = np.percentile(null_spectra, PA_PERCENTILE, axis=0)
    passed = real_sv[:MAX_RANK] > null99
    k = int(np.sum(passed))
    return {
        "real_singular_values": real_sv[:MAX_RANK],
        "null_p99_singular_values": null99,
        "real_to_null_p99_ratio": real_sv[:MAX_RANK] / np.maximum(null99, EPS),
        "passed": passed,
        "K": min(k, MAX_RANK),
        "null_spectra": null_spectra,
        "method": "100 independent random circular user permutations per day/channel",
        "sample_users": int(n),
    }


def factorize(
    matrices: list[np.ndarray], day_gram: np.ndarray, k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    values, vectors = np.linalg.eigh(day_gram)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0.0)
    temporal_basis = vectors[:, order[:k]].astype(np.float64)
    # Remove the arbitrary eigenvector sign deterministically.
    for j in range(k):
        ix = int(np.argmax(np.abs(temporal_basis[:, j])))
        if temporal_basis[ix, j] < 0:
            temporal_basis[:, j] *= -1

    n = matrices[0].shape[0]
    c = len(matrices)
    loadings = np.empty((n, k), np.float64)
    channel_profiles = np.empty((c, k), np.float64)
    separable_fraction = np.empty(k, np.float64)
    for j in range(k):
        projected = np.column_stack([matrix @ temporal_basis[:, j] for matrix in matrices])
        channel_cov = projected.T @ projected
        cv, cw = np.linalg.eigh(channel_cov)
        w = cw[:, -1]
        ix = int(np.argmax(np.abs(w)))
        if w[ix] < 0:
            w = -w
        l = projected @ w
        l -= l.mean()
        l_scale = rms(l)
        if l_scale < EPS:
            raise RuntimeError(f"degenerate loading at factor {j}")
        loadings[:, j] = l / l_scale
        channel_profiles[:, j] = w
        separable_fraction[j] = float(max(cv[-1], 0.0) / max(float(np.sum(np.maximum(cv, 0.0))), EPS))

    cross_gram = (loadings.T @ loadings) * (channel_profiles.T @ channel_profiles)
    cross_pinv = np.linalg.pinv(cross_gram, rcond=1e-10)
    rhs = np.zeros((HISTORY_DAYS, k), np.float64)
    for ci, matrix in enumerate(matrices):
        rhs += (matrix.T @ loadings) * channel_profiles[ci][None, :]
    factors = rhs @ cross_pinv
    total_variance = float(np.sum(values))
    diag = {
        "full_singular_values_top16": np.sqrt(values[:MAX_RANK]),
        "explained_variance_fraction_topK": float(np.sum(values[:k]) / max(total_variance, EPS)),
        "separable_user_channel_fraction": separable_fraction,
        "loading_cross_gram_condition": float(np.linalg.cond(cross_gram)),
        "factor_std": factors.std(axis=0),
    }
    return loadings, factors, channel_profiles, cross_pinv, diag


def project_future(
    future_matrices: list[np.ndarray], loadings: np.ndarray,
    channel_profiles: np.ndarray, cross_pinv: np.ndarray,
) -> np.ndarray:
    k = loadings.shape[1]
    rhs = np.zeros((ORACLE_DAYS, k), np.float64)
    for ci, matrix in enumerate(future_matrices):
        rhs += (matrix.T @ loadings) * channel_profiles[ci][None, :]
    return rhs @ cross_pinv


def project_two_pass(raw: np.ndarray, span: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    raw = np.asarray(raw, np.float64)
    centered = raw - raw.mean(axis=0, keepdims=True)
    coef1, *_ = np.linalg.lstsq(span, centered, rcond=1e-10)
    first = centered - span @ coef1
    coef2, *_ = np.linalg.lstsq(span, first, rcond=1e-10)
    removed2 = span @ coef2
    out = first - removed2
    return out, {
        "raw_rms_frobenius": rms(raw),
        "centered_rms_frobenius": rms(centered),
        "perp_rms_frobenius": rms(out),
        "perp_fraction": rms(out) / max(rms(centered), EPS),
        "second_pass_projection_rms": rms(removed2),
        "second_pass_relative_error": rms(removed2) / max(rms(out), EPS),
    }


def oracle_headroom(residual: np.ndarray, features: np.ndarray) -> dict[str, Any]:
    n = len(residual)
    gram = features.T @ features / n
    covariance = features.T @ residual / n
    coefficients = np.linalg.pinv(gram, rcond=1e-10) @ covariance
    correction = features @ coefficients
    gain = float(covariance @ coefficients)
    rank = int(np.linalg.matrix_rank(gram, tol=max(float(np.max(np.diag(gram), initial=0)), 1.0) * 1e-10))
    finite_sample_bias = rank * float(np.mean(residual * residual)) / n
    return {
        "oracle_gain": gain,
        "oracle_gain_debiased": max(gain - finite_sample_bias, 0.0),
        "finite_sample_df_bias": finite_sample_bias,
        "rank": rank,
        "rho": corr(correction, residual),
        "correction_rms": rms(correction),
        "coefficients": coefficients,
        "G": gram,
        "b": covariance,
    }


def run_oracle_fold(
    fold: str, panel: np.ndarray, all_uid: np.ndarray, fold_number: int,
) -> dict[str, Any]:
    prod = load_production_fold(fold)
    uid = prod["user_id"]
    rows = np.searchsorted(all_uid, uid)
    if rows.max(initial=0) >= len(all_uid) or not np.array_equal(all_uid[rows], uid):
        raise AssertionError(f"panel user alignment failed for {fold}")
    d = day_index(fold)
    hist_slice = slice(d - HISTORY_DAYS + 1, d + 1)
    future_slice = slice(d + 1, d + ORACLE_DAYS + 1)
    if hist_slice.start < 0 or future_slice.stop > panel.shape[1]:
        raise AssertionError(f"date bounds failed for {fold}")

    rng = np.random.default_rng(SEED + 1000 * fold_number)
    pa_n = min(PA_USERS, len(uid))
    scale_n = min(8192, len(uid))
    pa_rows = np.sort(rng.choice(len(uid), size=pa_n, replace=False))
    scale_rows = np.sort(rng.choice(len(uid), size=scale_n, replace=False))
    matrices: list[np.ndarray] = []
    future_matrices: list[np.ndarray] = []
    normalization: dict[str, Any] = {}
    activity_days: np.ndarray | None = None
    day_gram = np.zeros((HISTORY_DAYS, HISTORY_DAYS), np.float64)
    for channel in CHANNELS:
        matrix, future, diag, user_mean = normalize_channel(
            panel, rows, hist_slice, future_slice, channel, scale_rows,
        )
        matrices.append(matrix)
        future_matrices.append(future)
        normalization[channel] = diag
        day_gram += np.asarray(matrix.T @ matrix, np.float64)
        if channel == "present":
            # Binary input was not log-transformed, so 180*mean is active days.
            activity_days = user_mean.astype(np.float64) * HISTORY_DAYS
        log(fold, "normalized", channel, "history RMS", f"{diag['history_rms']:.4f}")
    assert activity_days is not None

    pa = parallel_analysis(matrices, pa_rows, SEED + 1000 * fold_number + 17)
    k = int(pa["K"])
    if k < 1:
        # This is a scientific zero-structure outcome, not a rank sweep.
        raise RuntimeError(f"parallel analysis retained no factors at {fold}")
    log(fold, "parallel-analysis K", k)
    loadings, factors, profiles, cross_pinv, factor_diag = factorize(matrices, day_gram, k)
    oracle_factors = project_future(future_matrices, loadings, profiles, cross_pinv)
    g_oracle = oracle_factors.sum(axis=0)
    z_oracle = loadings * g_oracle[None, :]
    z_perp, projection = project_two_pass(z_oracle, prod["span"])
    oracle = oracle_headroom(prod["residual"], z_perp)

    level_audit = {
        "max_abs_loading_corr_with_baseline": float(max(abs(corr(loadings[:, j], prod["baseline"])) for j in range(k))),
        "max_abs_oracle_feature_corr_with_baseline": float(max(abs(corr(z_oracle[:, j], prod["baseline"])) for j in range(k))),
        "max_abs_loading_corr_with_activity_days": float(max(abs(corr(loadings[:, j], activity_days)) for j in range(k))),
        "max_abs_oracle_feature_corr_with_activity_days": float(max(abs(corr(z_oracle[:, j], activity_days)) for j in range(k))),
        "gmv_channels_used": False,
    }
    output = HERE / f"factor_oracle_{fold}.npz"
    np.savez_compressed(
        output,
        user_id=uid,
        L=loadings.astype(np.float32),
        W=profiles.astype(np.float32),
        F_history=factors.astype(np.float32),
        F_oracle=oracle_factors.astype(np.float32),
        G_oracle=g_oracle.astype(np.float64),
        Z_oracle_perp=z_perp.astype(np.float32),
        activity_days=activity_days.astype(np.float32),
        pa_real_sv=np.asarray(pa["real_singular_values"], np.float64),
        pa_null_p99=np.asarray(pa["null_p99_singular_values"], np.float64),
    )
    result = {
        "fold": fold,
        "n": int(len(uid)),
        "history_start": str(DATA_START + np.timedelta64(hist_slice.start, "D")),
        "history_end": fold,
        "oracle_start": str(np.datetime64(fold) + np.timedelta64(1, "D")),
        "oracle_end": str(np.datetime64(fold) + np.timedelta64(ORACLE_DAYS, "D")),
        "K": k,
        "normalization": normalization,
        "parallel_analysis": {key: value for key, value in pa.items() if key != "null_spectra"},
        "factor_diagnostics": factor_diag,
        "projection": projection,
        "level_audit": level_audit,
        "oracle": oracle,
        "artifact": str(output),
        "artifact_sha256": sha256(output),
    }
    # Release the approximately 1.2 GB fold tensor before the next fold.
    del matrices, future_matrices, loadings, factors, profiles, oracle_factors, z_oracle, z_perp
    gc.collect()
    return result


def run_oracle() -> None:
    for required in (PANEL_PATH, UID_PATH, PROD):
        if not required.exists():
            raise FileNotFoundError(required)
    panel = np.load(PANEL_PATH, mmap_mode="r")
    all_uid = np.load(UID_PATH, mmap_mode="r")
    if panel.shape != (250_000, 409, 14) or panel.dtype != np.float16:
        raise AssertionError(f"unexpected panel {panel.shape}/{panel.dtype}")
    if all_uid.shape != (250_000,) or not np.all(all_uid[1:] > all_uid[:-1]):
        raise AssertionError("unexpected uid panel")

    results: list[dict[str, Any]] = []
    for i, fold in enumerate(FOLDS):
        log("starting oracle fold", fold)
        result = run_oracle_fold(fold, panel, all_uid, i)
        results.append(result)
        write_json(HERE / "oracle_partial.json", {"completed": results})
        log(fold, "oracle gain", f"{result['oracle']['oracle_gain']:.9f}")

    weights = np.asarray([FOLD_WEIGHTS[fold] for fold in FOLDS], np.float64)
    gains = np.asarray([row["oracle"]["oracle_gain"] for row in results], np.float64)
    debiased = np.asarray([row["oracle"]["oracle_gain_debiased"] for row in results], np.float64)
    weighted = float(np.average(gains, weights=weights))
    weighted_debiased = float(np.average(debiased, weights=weights))
    fold_passes = int(np.sum(gains >= ORACLE_GATE))
    gate_pass = bool(weighted >= ORACLE_GATE)
    summary = {
        "experiment": "EXP087_DYNAMIC_FACTOR_RESIDUAL",
        "stage": "ORACLE_GATE",
        "channels": CHANNELS,
        "count_transform": "log1p on six count channels; binary presence/buy unchanged",
        "normalization": "per-user history mean, cohort-day mean, per-channel robust IQR scale",
        "history_days": HISTORY_DAYS,
        "parallel_analysis": {
            "users": PA_USERS, "permutations": PA_REPS, "percentile": PA_PERCENTILE,
            "hard_cap": MAX_RANK,
        },
        "fold_weights": FOLD_WEIGHTS,
        "folds": results,
        "weighted_oracle_gain": weighted,
        "weighted_oracle_gain_debiased_diagnostic": weighted_debiased,
        "folds_at_or_above_0.001": fold_passes,
        "oracle_gate_threshold": ORACLE_GATE,
        "oracle_gate_pass": gate_pass,
        "verdict": "ORACLE_PASS" if gate_pass else "REJECT_ORACLE",
        "forecast_authorized": gate_pass,
        "gpu_used": False,
        "leaderboard_used": False,
        "runtime_seconds": time.time() - T0,
        "source_artifacts": {
            "panel": str(PANEL_PATH), "panel_sha256": sha256(PANEL_PATH),
            "uid": str(UID_PATH), "uid_sha256": sha256(UID_PATH),
            "production_components": str(PROD),
        },
    }
    write_json(HERE / "oracle_results.json", summary)
    pd.DataFrame([
        {
            "fold": row["fold"], "n": row["n"], "K": row["K"],
            "oracle_gain": row["oracle"]["oracle_gain"],
            "oracle_gain_debiased": row["oracle"]["oracle_gain_debiased"],
            "oracle_rho": row["oracle"]["rho"],
            "oracle_correction_rms": row["oracle"]["correction_rms"],
            "perp_fraction": row["projection"]["perp_fraction"],
            "passes_0.001": row["oracle"]["oracle_gain"] >= ORACLE_GATE,
        }
        for row in results
    ]).to_csv(HERE / "oracle_fold_metrics.csv", index=False)
    log("weighted oracle gain", f"{weighted:.9f}", "verdict", summary["verdict"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("oracle",), default="oracle")
    args = parser.parse_args()
    if args.stage == "oracle":
        run_oracle()


if __name__ == "__main__":
    main()
