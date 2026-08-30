from __future__ import annotations

import gc
import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd


EXP = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("exp075_a1_c", EXP / "run_a1_clean_forward.py")
A1 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(A1)
T0 = time.time()


def log(*x: object) -> None:
    print(f"[{time.time() - T0:7.1f}s]", *x, flush=True)


def state_axes(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Context layout for every window is [11 sums, 11 nonzero-day counts].
    offset = A1.WINDOWS.index(90) * len(A1.RAW_CHANNELS) * 2
    counts = X[:, offset + len(A1.RAW_CHANNELS):offset + 2 * len(A1.RAW_CHANNELS)]
    idx = {name: A1.RAW_CHANNELS.index(name) for name in A1.RAW_CHANNELS}
    activity = np.log1p(counts[:, idx["searches"]] + counts[:, idx["cat"]]
                        + counts[:, idx["to_cart"]] + counts[:, idx["to_ord"]])
    buy_days = counts[:, idx["gmv"]]
    return activity, buy_days


def labels_from_validation(X: np.ndarray, activity_edges: np.ndarray) -> np.ndarray:
    activity, buy_days = state_axes(X)
    a = np.digitize(activity, activity_edges[1:-1], right=True)
    b = np.where(buy_days == 0, 0, np.where(buy_days <= 2, 1, 2))
    return (a * 3 + b).astype(np.int16)


def matched_indices(X: np.ndarray, cutoff_index: np.ndarray, Xval: np.ndarray) -> tuple[np.ndarray, dict]:
    val_activity, _ = state_axes(Xval)
    activity_edges = np.quantile(val_activity, [0.0, 0.25, 0.5, 0.75, 1.0])
    # Tie-safe strictly increasing internal edges.
    for i in range(1, len(activity_edges)):
        if activity_edges[i] <= activity_edges[i - 1]:
            activity_edges[i] = np.nextafter(activity_edges[i - 1], np.inf)
    val_label = labels_from_validation(Xval, activity_edges)
    p = np.bincount(val_label, minlength=12).astype(float)
    p /= p.sum()
    rng = np.random.default_rng(20260828)
    chosen = []
    diagnostics = {"activity_edges": activity_edges.tolist(), "validation_proportions": p.tolist(), "cutoffs": []}
    for ci in np.unique(cutoff_index):
        idx = np.flatnonzero(cutoff_index == ci)
        lab = labels_from_validation(X[idx], activity_edges)
        avail = np.bincount(lab, minlength=12)
        feasible = [avail[k] / p[k] for k in range(12) if p[k] > 0]
        total = int(min(len(idx), math.floor(min(feasible))))
        if total < 0.5 * len(idx):
            raise AssertionError(f"Matched cohort support too small: {total}/{len(idx)}")
        target = np.floor(total * p).astype(int)
        # Allocate rounding remainder to bins with largest fractional parts.
        remainder = total - int(target.sum())
        fractional = total * p - target
        for k in np.argsort(-fractional)[:remainder]:
            target[k] += 1
        local_choice = []
        for k in range(12):
            pool = idx[lab == k]
            if target[k] > len(pool):
                raise AssertionError("Target-free matched construction exceeded support")
            if target[k]:
                local_choice.append(rng.choice(pool, size=target[k], replace=False))
        local_choice = np.sort(np.concatenate(local_choice))
        before = avail / avail.sum()
        after_lab = labels_from_validation(X[local_choice], activity_edges)
        after = np.bincount(after_lab, minlength=12) / len(local_choice)
        diagnostics["cutoffs"].append({
            "cutoff_index": int(ci),
            "rows_before": int(len(idx)),
            "rows_after": int(len(local_choice)),
            "retained_fraction": float(len(local_choice) / len(idx)),
            "TV_before_vs_validation": float(0.5 * np.abs(before - p).sum()),
            "TV_after_vs_validation": float(0.5 * np.abs(after - p).sum()),
        })
        chosen.append(local_choice)
    return np.sort(np.concatenate(chosen)), diagnostics


def fit_fold(data: A1.CleanData, cutoff: A1.dt.date, canonical: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    train_cutoffs = [cutoff - A1.dt.timedelta(days=lag) for lag in A1.TRAIN_LAGS]
    frames = [data.raw_cutoff_frame(x) for x in train_cutoffs]
    val = data.raw_cutoff_frame(cutoff)
    Xb, y, uid, cutoff_index = A1.concat_context(data, frames)
    halves = A1.stable_half(uid)
    base_cf = np.empty(len(y), dtype=float)
    for side in (0, 1):
        fit, pred = halves != side, halves == side
        model = A1.train_lgb(Xb[fit], y[fit], "baseline", 260)
        base_cf[pred] = model.predict(Xb[pred])
        del model
    offset = float(np.mean(y - base_cf))
    residual = y - (base_cf + offset)
    full_base = A1.train_lgb(Xb, y, "baseline", 260)
    val_ids = val.user_id.to_numpy(np.int64)
    Xb_val = data.context_features(data.rows(val_ids), cutoff)
    baseline_z = full_base.predict(Xb_val) + offset
    del full_base
    canon = canonical[canonical.cutoff == cutoff.isoformat()].sort_values("user_id")
    if not np.array_equal(canon.user_id.to_numpy(), val_ids):
        raise AssertionError("C/A1 row mismatch")
    replay = float(np.max(np.abs(canon.baseline_z.to_numpy(float) - baseline_z)))
    if replay > 1e-10:
        raise AssertionError(f"C baseline replay mismatch {replay}")

    selected, match_diag = matched_indices(Xb, cutoff_index, Xb_val)
    matched_frames, matched_residual = [], []
    pos = 0
    for ci, frame in enumerate(frames):
        n = len(frame)
        local_global = selected[(selected >= pos) & (selected < pos + n)]
        local = local_global - pos
        matched_frames.append(frame.iloc[local].reset_index(drop=True))
        matched_residual.append(residual[local_global])
        pos += n
    matched_residual = np.concatenate(matched_residual)
    log(cutoff, "matched rows", len(matched_residual), "/", len(residual))
    Xm = A1.concat_candidate(data, matched_frames, 365)
    model = A1.train_lgb(Xm, matched_residual, "candidate", 300)
    del Xm
    gc.collect()
    Xval = data.candidate_features(data.rows(val_ids), cutoff, 365, Xb_val)
    u_raw = model.predict(Xval)
    del Xval, model, Xb, Xb_val
    gc.collect()
    u_perp, projection = A1.project_candidate(u_raw, baseline_z)
    r = val.target_log.to_numpy(float) - baseline_z
    b, G = float(np.mean(u_perp * r)), float(np.mean(u_perp * u_perp))
    meta = {
        "cutoff": cutoff.isoformat(),
        "rho_matched": A1.correlation(u_perp, r),
        "rho_normal": A1.correlation(canon.u_perp_365.to_numpy(float), r),
        "b": b,
        "G": G,
        "oracle_amplitude": b / G if G else 0.0,
        "corr_matched_normal": A1.correlation(u_perp, canon.u_perp_365.to_numpy(float)),
        "match_diagnostics": match_diag,
        "projection": projection,
        "baseline_replay_max_abs_error": replay,
    }
    return u_raw, u_perp, meta


def summarize(d: pd.DataFrame, metas: list[dict]) -> dict:
    fold_rows, prior_u, prior_r, prior_w = [], [], [], []
    for fi, cutoff in enumerate([x.isoformat() for x in A1.FOLDS]):
        idx = d.index[d.cutoff == cutoff]
        p = d.loc[idx]
        u, r = p.u_perp_C.to_numpy(float), p.residual.to_numpy(float)
        if fi == 0:
            amp = 1.0
        else:
            up, rp, wp = np.concatenate(prior_u), np.concatenate(prior_r), np.concatenate(prior_w)
            amp = float(np.sum(wp * up * rp) / np.sum(wp * up * up))
        delta = (r - amp * u) ** 2 - r ** 2
        d.loc[idx, "delta_mse_C"] = delta
        base, corrected = math.sqrt(np.mean(r * r)), math.sqrt(np.mean((r - amp * u) ** 2))
        fold_rows.append({
            "cutoff": cutoff,
            "rho_matched": A1.correlation(u, r),
            "rho_normal": metas[fi]["rho_normal"],
            "rho_relative_change": A1.correlation(u, r) / metas[fi]["rho_normal"] - 1,
            "deployable_amplitude": amp,
            "delta_MSE": float(np.mean(delta)),
            "delta_RMSLE": float(corrected - base),
            "corr_matched_normal": metas[fi]["corr_matched_normal"],
        })
        prior_u.append(u)
        prior_r.append(r)
        prior_w.append(np.full(len(u), A1.FOLD_WEIGHTS[fi] / len(u)))
    weighted_rho = float(sum(A1.FOLD_WEIGHTS[i] * fold_rows[i]["rho_matched"] for i in range(4)) / A1.FOLD_WEIGHTS.sum())
    normal = json.loads((EXP / "rho_analysis.json").read_text())["A1_TREE_TRAJ_365"]
    nested_mse = float(sum(A1.FOLD_WEIGHTS[i] * fold_rows[i]["delta_MSE"] for i in range(4)) / A1.FOLD_WEIGHTS.sum())
    nested_rmsle = float(sum(A1.FOLD_WEIGHTS[i] * fold_rows[i]["delta_RMSLE"] for i in range(4)) / A1.FOLD_WEIGHTS.sum())
    improve = weighted_rho / normal["weighted_clean_forward_rho"] - 1
    latest_ok = fold_rows[-1]["rho_matched"] >= normal["latest_rho"]
    nested_ok = nested_mse < normal["nested_delta_MSE"]
    return {
        "experiment": "C_TEST_REGIME_MATCHED_A1_365",
        "construction": "per-cutoff without-replacement maximum-support subset matching held-out target-free activity-quartile x purchase-day class proportions",
        "fold_rows": fold_rows,
        "weighted_rho_matched": weighted_rho,
        "weighted_rho_normal": normal["weighted_clean_forward_rho"],
        "rho_relative_change": improve,
        "latest_not_worse": latest_ok,
        "nested_delta_MSE_matched": nested_mse,
        "nested_delta_MSE_normal": normal["nested_delta_MSE"],
        "nested_delta_RMSLE_matched": nested_rmsle,
        "nested_better": nested_ok,
        "verdict": "CONTINUE_MATCHED" if improve >= 0.10 and latest_ok and nested_ok else "REJECT_MATCHED",
    }


def main() -> None:
    canonical = pd.read_parquet(EXP / "clean_forward_predictions.parquet")
    canonical["u_raw_C"] = np.nan
    canonical["u_perp_C"] = np.nan
    data = A1.CleanData()
    metas = []
    for cutoff in A1.FOLDS:
        raw, perp, meta = fit_fold(data, cutoff, canonical)
        idx = canonical.index[canonical.cutoff == cutoff.isoformat()]
        canonical.loc[idx, "u_raw_C"] = raw
        canonical.loc[idx, "u_perp_C"] = perp
        metas.append(meta)
        log(cutoff, meta["rho_matched"], "normal", meta["rho_normal"])
    result = summarize(canonical, metas)
    canonical[["user_id", "cutoff", "target_log", "baseline_z", "residual",
               "u_raw_C", "u_perp_C", "delta_mse_C"]].to_parquet(
                   EXP / "c_matched_clean_forward_predictions.parquet", index=False)
    (EXP / "c_matched_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (EXP / "c_matched_construction_audit.json").write_text(json.dumps(metas, indent=2), encoding="utf-8")
    pd.DataFrame(result["fold_rows"]).to_csv(EXP / "c_matched_fold_metrics.csv", index=False)
    (EXP / "c_runtime.json").write_text(json.dumps({"wall_seconds": time.time() - T0, "gpu_used": False}, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
