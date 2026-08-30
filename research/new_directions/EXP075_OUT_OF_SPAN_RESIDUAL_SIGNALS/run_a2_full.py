from __future__ import annotations

import gc
import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


EXP = Path(__file__).resolve().parent


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, EXP / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


A1 = load_module("exp075_a1_full", "run_a1_clean_forward.py")
A2 = load_module("exp075_a2_pilot", "run_a2_cnn_pilot.py")
FW = A1.FOLD_WEIGHTS
T0 = time.time()


def log(*x: object) -> None:
    print(f"[{time.time() - T0:7.1f}s]", *x, flush=True)


def fit_fold(data: A1.CleanData, cutoff: A1.dt.date, canonical: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    train_cutoffs = [cutoff - A1.dt.timedelta(days=lag) for lag in A1.TRAIN_LAGS]
    frames = [data.raw_cutoff_frame(x) for x in train_cutoffs]
    val = data.raw_cutoff_frame(cutoff)
    log(cutoff, "baseline")
    Xc, y, uid, _ = A1.concat_context(data, frames)
    halves = A1.stable_half(uid)
    base_cf = np.empty(len(y), dtype=float)
    for side in (0, 1):
        fit, pred_idx = halves != side, halves == side
        model = A1.train_lgb(Xc[fit], y[fit], "baseline", 260)
        base_cf[pred_idx] = model.predict(Xc[pred_idx])
        del model
    offset = float(np.mean(y - base_cf))
    residual = y - (base_cf + offset)
    full_base = A1.train_lgb(Xc, y, "baseline", 260)
    val_ids = val.user_id.to_numpy(np.int64)
    Xc_val = data.context_features(data.rows(val_ids), cutoff)
    baseline_z = full_base.predict(Xc_val) + offset
    del full_base
    canon = canonical[canonical.cutoff == cutoff.isoformat()].sort_values("user_id")
    if not np.array_equal(canon.user_id.to_numpy(), val_ids):
        raise AssertionError("A2/A1 validation row mismatch")
    base_err = float(np.max(np.abs(canon.baseline_z.to_numpy(float) - baseline_z)))
    if base_err > 1e-10:
        raise AssertionError(f"A2 baseline replay mismatch at {cutoff}: {base_err}")

    log(cutoff, "weekly tensors")
    Xs = A2.build_weekly(data, frames)
    Xs_val = A2.build_weekly(data, [val])
    channel_rms = np.sqrt(np.mean(Xs.astype(np.float64) ** 2, axis=(0, 1)))
    channel_rms = np.maximum(channel_rms, 1e-3)
    Xs = (Xs / channel_rms).astype(np.float16)
    Xs_val = (Xs_val / channel_rms).astype(np.float16)
    context_mean = Xc.mean(axis=0, dtype=np.float64)
    context_std = np.maximum(Xc.std(axis=0, dtype=np.float64), 1e-3)
    Xc = ((Xc - context_mean) / context_std).astype(np.float16)
    Xc_val = ((Xc_val - context_mean) / context_std).astype(np.float16)
    bucket = A2.hash64(uid) % np.uint64(10)
    internal_valid = np.flatnonzero(bucket == 0)
    internal_train = np.flatnonzero(bucket != 0)
    device = torch.device("cuda")
    _, best_epoch, curve = A2.train_model(Xs, Xc, residual, internal_train, internal_valid, device)
    model = A2.train_full_epochs(Xs, Xc, residual, best_epoch, device)
    u_raw = A2.predict(model, Xs_val, Xc_val, device)
    u_perp, projection = A1.project_candidate(u_raw, baseline_z)
    r = val.target_log.to_numpy(float) - baseline_z
    b = float(np.mean(u_perp * r))
    G = float(np.mean(u_perp * u_perp))
    metrics = {
        "cutoff": cutoff.isoformat(),
        "train_cutoffs": [x.isoformat() for x in train_cutoffs],
        "train_rows": int(len(y)),
        "validation_rows": int(len(val)),
        "best_epoch": int(best_epoch),
        "curve": curve,
        "rho": A1.correlation(u_perp, r),
        "b": b,
        "G": G,
        "oracle_amplitude": b / G if G else 0.0,
        "oracle_MSE_gain": -(b * b / G) if G else 0.0,
        "corr_with_A1_365": A1.correlation(u_perp, canon.u_perp_365.to_numpy(float)),
        "projection": projection,
        "baseline_replay_max_abs_error": base_err,
    }
    log(cutoff, "rho", metrics["rho"], "corr A1", metrics["corr_with_A1_365"])
    del Xs, Xs_val, Xc, Xc_val, model
    gc.collect()
    torch.cuda.empty_cache()
    return u_raw, u_perp, metrics


def summarize(d: pd.DataFrame) -> tuple[dict, np.ndarray, np.ndarray]:
    fold_rows = []
    prior_u, prior_r, prior_w = [], [], []
    for fi, cutoff in enumerate([x.isoformat() for x in A1.FOLDS]):
        idx = d.index[d.cutoff == cutoff]
        p = d.loc[idx]
        u = p.u_perp_A2.to_numpy(float)
        r = p.residual.to_numpy(float)
        if fi == 0:
            amp, source = 1.0, "fixed_residual_objective"
        else:
            up, rp, wp = np.concatenate(prior_u), np.concatenate(prior_r), np.concatenate(prior_w)
            amp = float(np.sum(wp * up * rp) / np.sum(wp * up * up))
            source = "strictly_earlier_heldout_folds"
        delta = (r - amp * u) ** 2 - r ** 2
        d.loc[idx, "amplitude_A2"] = amp
        d.loc[idx, "delta_mse_A2"] = delta
        base = math.sqrt(float(np.mean(r * r)))
        corrected = math.sqrt(float(np.mean((r - amp * u) ** 2)))
        fold_rows.append({
            "cutoff": cutoff,
            "rho": A1.correlation(u, r),
            "b": float(np.mean(u * r)),
            "G": float(np.mean(u * u)),
            "deployable_amplitude": amp,
            "amplitude_source": source,
            "delta_MSE": float(np.mean(delta)),
            "baseline_RMSLE": base,
            "corrected_RMSLE": corrected,
            "delta_RMSLE": corrected - base,
            "corr_with_A1_365": A1.correlation(u, p.u_perp_365.to_numpy(float)),
        })
        prior_u.append(u)
        prior_r.append(r)
        prior_w.append(np.full(len(u), FW[fi] / len(u)))

    fidx = np.asarray([[x.isoformat() for x in A1.FOLDS].index(v) for v in d.cutoff], dtype=int)
    counts = np.bincount(fidx, minlength=4)
    w = FW[fidx] / counts[fidx]
    w /= w.sum()
    u, r = d.u_perp_A2.to_numpy(float), d.residual.to_numpy(float)
    mu_u, mu_r = np.sum(w * u), np.sum(w * r)
    rho = float(np.sum(w * (u - mu_u) * (r - mu_r)) /
                math.sqrt(np.sum(w * (u - mu_u) ** 2) * np.sum(w * (r - mu_r) ** 2)))
    nested_mse = float(sum(FW[i] * fold_rows[i]["delta_MSE"] for i in range(4)) / FW.sum())
    nested_rmsle = float(sum(FW[i] * fold_rows[i]["delta_RMSLE"] for i in range(4)) / FW.sum())

    uid, inv = np.unique(d.user_id.to_numpy(np.int64), return_inverse=True)
    stats = np.column_stack([w, w * u, w * r, w * u * u, w * r * r, w * u * r,
                             w * d.delta_mse_A2.to_numpy(float)])
    cluster = np.zeros((len(uid), stats.shape[1]), dtype=float)
    for j in range(stats.shape[1]):
        cluster[:, j] = np.bincount(inv, weights=stats[:, j], minlength=len(uid))
    rng = np.random.default_rng(20260828 + 202)
    rho_draws, delta_draws = [], []
    for _ in range(50):
        mult = rng.poisson(1.0, size=(20, len(uid))).astype(float)
        sums = mult @ cluster
        sw = sums[:, 0]
        uu, rr = sums[:, 1] / sw, sums[:, 2] / sw
        vu = sums[:, 3] / sw - uu ** 2
        vr = sums[:, 4] / sw - rr ** 2
        cv = sums[:, 5] / sw - uu * rr
        rho_draws.extend((cv / np.sqrt(np.maximum(vu * vr, 1e-300))).tolist())
        delta_draws.extend((sums[:, 6] / sw).tolist())
    rho_draws, delta_draws = np.asarray(rho_draws), np.asarray(delta_draws)
    se = float(np.std(rho_draws, ddof=1))
    result = {
        "experiment": "A2_WEEKLY_RESIDUAL_CNN",
        "fold_rows": fold_rows,
        "weighted_clean_forward_rho": rho,
        "latest_rho": fold_rows[-1]["rho"],
        "nested_delta_MSE": nested_mse,
        "nested_delta_RMSLE": nested_rmsle,
        "bootstrap": {
            "replicates": 1000,
            "unique_users": int(len(uid)),
            "rho_ci_2_5": float(np.quantile(rho_draws, 0.025)),
            "rho_ci_97_5": float(np.quantile(rho_draws, 0.975)),
            "rho_bootstrap_se": se,
            "t_rho": rho / se,
            "delta_MSE_ci_2_5": float(np.quantile(delta_draws, 0.025)),
            "delta_MSE_ci_97_5": float(np.quantile(delta_draws, 0.975)),
            "P_delta_MSE_lt_0": float(np.mean(delta_draws < 0)),
        },
    }
    result["verdict"] = ("STRONG_SIGNAL" if rho >= 0.025 and result["latest_rho"] >= 0.020
                         and result["bootstrap"]["t_rho"] >= 3
                         and result["bootstrap"]["P_delta_MSE_lt_0"] >= 0.95
                         else "PROMISING" if rho >= 0.020 and nested_mse < 0 else "REJECT")
    return result, rho_draws, delta_draws


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    canonical = pd.read_parquet(EXP / "clean_forward_predictions.parquet")
    canonical["u_raw_A2"] = np.nan
    canonical["u_perp_A2"] = np.nan
    data = A1.CleanData()
    metas = []
    for cutoff in A1.FOLDS[:-1]:
        raw, perp, meta = fit_fold(data, cutoff, canonical)
        idx = canonical.index[canonical.cutoff == cutoff.isoformat()]
        canonical.loc[idx, "u_raw_A2"] = raw
        canonical.loc[idx, "u_perp_A2"] = perp
        metas.append(meta)

    pilot = pd.read_parquet(EXP / "a2_latest_pilot_predictions.parquet").sort_values("user_id")
    idx = canonical.index[canonical.cutoff == A1.FOLDS[-1].isoformat()]
    canon_latest = canonical.loc[idx].sort_values("user_id")
    if not np.array_equal(canon_latest.user_id.to_numpy(), pilot.user_id.to_numpy()):
        raise AssertionError("Pilot latest row mismatch")
    canonical.loc[canon_latest.index, "u_raw_A2"] = pilot.u_raw_A2.to_numpy(float)
    canonical.loc[canon_latest.index, "u_perp_A2"] = pilot.u_perp_A2.to_numpy(float)
    metas.append(json.loads((EXP / "a2_pilot_metrics.json").read_text(encoding="utf-8")))
    if canonical[["u_raw_A2", "u_perp_A2"]].isna().any().any():
        raise AssertionError("Incomplete A2 OOF")

    result, rho_draws, delta_draws = summarize(canonical)
    canonical[[
        "user_id", "cutoff", "target_y30", "target_log", "baseline_prediction",
        "baseline_z", "residual", "u_raw_A2", "u_perp_A2", "amplitude_A2",
        "delta_mse_A2",
    ]].to_parquet(EXP / "a2_clean_forward_predictions.parquet", index=False)
    pd.DataFrame(result["fold_rows"]).to_csv(EXP / "a2_fold_metrics.csv", index=False)
    (EXP / "a2_rho_analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (EXP / "a2_training_meta.json").write_text(json.dumps(metas, indent=2), encoding="utf-8")
    np.savez_compressed(EXP / "a2_bootstrap_draws.npz", rho=rho_draws, delta_mse=delta_draws)
    runtime = {
        "wall_seconds_full_additional": time.time() - T0,
        "pilot_seconds": json.loads((EXP / "a2_pilot_metrics.json").read_text())["runtime_seconds"],
        "gpu": torch.cuda.get_device_name(0),
        "all_gpu_runs_below_six_hours": True,
    }
    (EXP / "a2_runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
