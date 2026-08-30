from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


EXP = Path(__file__).resolve().parent
FOLDS = ["2025-09-04", "2025-09-18", "2025-10-02", "2025-10-16"]
FW = np.asarray([1.0, 2.0, 4.0, 8.0])


def corr(x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> float:
    if w is None:
        w = np.full(len(x), 1.0 / len(x))
    else:
        w = w / w.sum()
    mx, my = np.sum(w * x), np.sum(w * y)
    return float(np.sum(w * (x - mx) * (y - my)) /
                 math.sqrt(np.sum(w * (x - mx) ** 2) * np.sum(w * (y - my) ** 2)))


def main() -> None:
    d = pd.read_parquet(EXP / "clean_forward_predictions.parquet")
    fold_rows = []
    prior_U, prior_r, prior_w = [], [], []
    d["joint_delta_mse"] = np.nan
    for fi, cutoff in enumerate(FOLDS):
        idx = d.index[d.cutoff == cutoff]
        p = d.loc[idx]
        U = p[["u_perp_180", "u_perp_365"]].to_numpy(float)
        r = p.residual.to_numpy(float)
        if fi == 0:
            a = np.asarray([0.5, 0.5])
            source = "fixed_equal_average"
        else:
            Up = np.concatenate(prior_U)
            rp = np.concatenate(prior_r)
            wp = np.concatenate(prior_w)
            G = (Up * wp[:, None]).T @ Up
            b = (Up * wp[:, None]).T @ rp
            a = np.linalg.solve(G, b)
            source = "strictly_earlier_heldout_folds"
        c = U @ a
        delta = (r - c) ** 2 - r ** 2
        d.loc[idx, "joint_delta_mse"] = delta
        base = math.sqrt(float(np.mean(r * r)))
        corrected = math.sqrt(float(np.mean((r - c) ** 2)))
        fold_rows.append({
            "cutoff": cutoff,
            "a180": float(a[0]),
            "a365": float(a[1]),
            "coefficient_source": source,
            "rho_joint_correction": corr(c, r),
            "corr_180_365": corr(U[:, 0], U[:, 1]),
            "delta_MSE": float(np.mean(delta)),
            "baseline_RMSLE": base,
            "corrected_RMSLE": corrected,
            "delta_RMSLE": corrected - base,
        })
        prior_U.append(U)
        prior_r.append(r)
        prior_w.append(np.full(len(U), FW[fi] / len(U)))

    # Weighted population moments: every fold has its fixed weight.
    fidx = np.asarray([FOLDS.index(x) for x in d.cutoff], dtype=int)
    nfold = np.bincount(fidx, minlength=4)
    w = FW[fidx] / nfold[fidx]
    w /= w.sum()
    U = d[["u_perp_180", "u_perp_365"]].to_numpy(float)
    r = d.residual.to_numpy(float)
    Uc = U - np.sum(w[:, None] * U, axis=0)
    rc = r - np.sum(w * r)
    G = (Uc * w[:, None]).T @ Uc
    b = (Uc * w[:, None]).T @ rc
    var_r = float(np.sum(w * rc * rc))
    a_oracle = np.linalg.solve(G, b)
    rho2_full = float(b @ a_oracle / var_r)
    rho2_180 = float(b[0] ** 2 / (G[0, 0] * var_r))
    rho2_365 = float(b[1] ** 2 / (G[1, 1] * var_r))
    nested_delta_mse = float(sum(FW[i] * fold_rows[i]["delta_MSE"] for i in range(4)) / FW.sum())
    nested_delta_rmsle = float(sum(FW[i] * fold_rows[i]["delta_RMSLE"] for i in range(4)) / FW.sum())

    # Cluster bootstrap of the already-deployed rolling correction.
    uid, inv = np.unique(d.user_id.to_numpy(np.int64), return_inverse=True)
    stats = np.column_stack([w, w * d.joint_delta_mse.to_numpy(float)])
    cluster = np.zeros((len(uid), 2), dtype=float)
    for j in range(2):
        cluster[:, j] = np.bincount(inv, weights=stats[:, j], minlength=len(uid))
    rng = np.random.default_rng(20260828 + 999)
    draws = []
    for _ in range(50):
        counts = rng.poisson(1.0, size=(20, len(uid))).astype(float)
        sums = counts @ cluster
        draws.extend((sums[:, 1] / sums[:, 0]).tolist())
    draws = np.asarray(draws)

    individual = json.loads((EXP / "rho_analysis.json").read_text(encoding="utf-8"))
    result = {
        "directions": ["A1_TREE_TRAJ_180", "A1_TREE_TRAJ_365"],
        "fold_rows": fold_rows,
        "G_weighted": G.tolist(),
        "b_weighted": b.tolist(),
        "oracle_joint_coefficients": a_oracle.tolist(),
        "condition_number": float(np.linalg.cond(G)),
        "weighted_correction_correlation": corr(U[:, 0], U[:, 1], w),
        "rho_joint": math.sqrt(max(rho2_full, 0.0)),
        "rho2_joint": rho2_full,
        "rho2_180_alone": rho2_180,
        "rho2_365_alone": rho2_365,
        "incremental_rho2_180_given_365": rho2_full - rho2_365,
        "incremental_rho2_365_given_180": rho2_full - rho2_180,
        "nested_joint_delta_MSE": nested_delta_mse,
        "nested_joint_delta_RMSLE": nested_delta_rmsle,
        "individual_LOFO_delta_MSE": {
            "drop_180_use_365": individual["A1_TREE_TRAJ_365"]["nested_delta_MSE"],
            "drop_365_use_180": individual["A1_TREE_TRAJ_180"]["nested_delta_MSE"],
        },
        "bootstrap": {
            "replicates": 1000,
            "delta_MSE_ci_2_5": float(np.quantile(draws, 0.025)),
            "delta_MSE_ci_97_5": float(np.quantile(draws, 0.975)),
            "P_delta_MSE_lt_0": float(np.mean(draws < 0)),
        },
    }
    (EXP / "joint_analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(fold_rows).to_csv(EXP / "joint_fold_metrics.csv", index=False)
    np.savez_compressed(EXP / "joint_bootstrap_draws.npz", delta_mse=draws)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
