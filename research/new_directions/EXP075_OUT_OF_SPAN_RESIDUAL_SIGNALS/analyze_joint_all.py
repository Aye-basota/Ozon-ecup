from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


EXP = Path(__file__).resolve().parent
FOLDS = ["2025-09-04", "2025-09-18", "2025-10-02", "2025-10-16"]
FW = np.asarray([1.0, 2.0, 4.0, 8.0])


def corr(x, y, w=None):
    if w is None:
        w = np.full(len(x), 1 / len(x))
    else:
        w = w / w.sum()
    x0, y0 = x - np.sum(w * x), y - np.sum(w * y)
    return float(np.sum(w * x0 * y0) / math.sqrt(np.sum(w * x0 * x0) * np.sum(w * y0 * y0)))


def evaluate(d: pd.DataFrame, cols: list[str], name: str) -> dict:
    fold_rows, prior_U, prior_r, prior_w = [], [], [], []
    delta_all = np.empty(len(d), dtype=float)
    for fi, cutoff in enumerate(FOLDS):
        idx = d.index[d.cutoff == cutoff]
        p = d.loc[idx]
        U, r = p[cols].to_numpy(float), p.residual.to_numpy(float)
        if fi == 0:
            a = np.full(len(cols), 1 / len(cols))
            source = "fixed_equal_average"
        else:
            Up, rp, wp = np.concatenate(prior_U), np.concatenate(prior_r), np.concatenate(prior_w)
            Gp = (Up * wp[:, None]).T @ Up
            bp = (Up * wp[:, None]).T @ rp
            a = np.linalg.solve(Gp, bp)
            source = "strictly_earlier_heldout_folds"
        correction = U @ a
        delta = (r - correction) ** 2 - r ** 2
        delta_all[idx] = delta
        base, corrected = math.sqrt(np.mean(r * r)), math.sqrt(np.mean((r - correction) ** 2))
        fold_rows.append({
            "cutoff": cutoff,
            "coefficients": a.tolist(),
            "coefficient_source": source,
            "rho_joint": corr(correction, r),
            "delta_MSE": float(np.mean(delta)),
            "delta_RMSLE": float(corrected - base),
        })
        prior_U.append(U)
        prior_r.append(r)
        prior_w.append(np.full(len(U), FW[fi] / len(U)))

    fidx = np.asarray([FOLDS.index(x) for x in d.cutoff], dtype=int)
    counts = np.bincount(fidx, minlength=4)
    w = FW[fidx] / counts[fidx]
    w /= w.sum()
    U, r = d[cols].to_numpy(float), d.residual.to_numpy(float)
    Uc, rc = U - np.sum(w[:, None] * U, axis=0), r - np.sum(w * r)
    G = (Uc * w[:, None]).T @ Uc
    b = (Uc * w[:, None]).T @ rc
    a_oracle = np.linalg.solve(G, b)
    var_r = float(np.sum(w * rc * rc))
    rho2 = float(b @ a_oracle / var_r)
    nested_mse = float(sum(FW[i] * fold_rows[i]["delta_MSE"] for i in range(4)) / FW.sum())
    nested_rmsle = float(sum(FW[i] * fold_rows[i]["delta_RMSLE"] for i in range(4)) / FW.sum())

    uid, inv = np.unique(d.user_id.to_numpy(np.int64), return_inverse=True)
    stats = np.column_stack([w, w * delta_all])
    cluster = np.zeros((len(uid), 2), dtype=float)
    for j in range(2):
        cluster[:, j] = np.bincount(inv, weights=stats[:, j], minlength=len(uid))
    rng = np.random.default_rng(20260828 + len(cols) * 100)
    draws = []
    for _ in range(50):
        mult = rng.poisson(1.0, size=(20, len(uid))).astype(float)
        sums = mult @ cluster
        draws.extend((sums[:, 1] / sums[:, 0]).tolist())
    draws = np.asarray(draws)
    return {
        "name": name,
        "directions": cols,
        "fold_rows": fold_rows,
        "G": G.tolist(),
        "b": b.tolist(),
        "oracle_coefficients": a_oracle.tolist(),
        "condition_number": float(np.linalg.cond(G)),
        "correction_correlation_matrix": np.corrcoef(U.T).tolist(),
        "rho_joint": math.sqrt(max(rho2, 0)),
        "rho2_joint": rho2,
        "nested_delta_MSE": nested_mse,
        "nested_delta_RMSLE": nested_rmsle,
        "bootstrap": {
            "replicates": 1000,
            "delta_MSE_ci_2_5": float(np.quantile(draws, 0.025)),
            "delta_MSE_ci_97_5": float(np.quantile(draws, 0.975)),
            "P_delta_MSE_lt_0": float(np.mean(draws < 0)),
        },
    }


def main() -> None:
    a1 = pd.read_parquet(EXP / "clean_forward_predictions.parquet")
    a2 = pd.read_parquet(EXP / "a2_clean_forward_predictions.parquet")
    d = a1.merge(a2[["user_id", "cutoff", "u_perp_A2"]], on=["user_id", "cutoff"], validate="one_to_one")
    definitions = {
        "A1_365_PLUS_A2": ["u_perp_365", "u_perp_A2"],
        "A1_180_PLUS_A2": ["u_perp_180", "u_perp_A2"],
        "A1_180_365_PLUS_A2": ["u_perp_180", "u_perp_365", "u_perp_A2"],
    }
    results = {name: evaluate(d, cols, name) for name, cols in definitions.items()}
    full = results["A1_180_365_PLUS_A2"]
    full["incremental_rho2"] = {}
    for dropped in definitions["A1_180_365_PLUS_A2"]:
        remain = [x for x in definitions["A1_180_365_PLUS_A2"] if x != dropped]
        key = "A1_365_PLUS_A2" if remain == definitions["A1_365_PLUS_A2"] else (
            "A1_180_PLUS_A2" if remain == definitions["A1_180_PLUS_A2"] else None
        )
        other = results[key] if key else evaluate(d, remain, "drop_one")
        full["incremental_rho2"][dropped] = full["rho2_joint"] - other["rho2_joint"]
    base = results["A1_365_PLUS_A2"]
    single_a1 = json.loads((EXP / "rho_analysis.json").read_text())["A1_TREE_TRAJ_365"]
    single_a2 = json.loads((EXP / "a2_rho_analysis.json").read_text())
    base["individual_LOFO"] = {
        "drop_A1_365_use_A2_nested_delta_MSE": single_a2["nested_delta_MSE"],
        "drop_A2_use_A1_365_nested_delta_MSE": single_a1["nested_delta_MSE"],
        "incremental_rho2_A1_365_given_A2": base["rho2_joint"] - single_a2["weighted_clean_forward_rho"] ** 2,
        "incremental_rho2_A2_given_A1_365": base["rho2_joint"] - single_a1["weighted_clean_forward_rho"] ** 2,
    }
    (EXP / "joint_all_analysis.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    rows = []
    for name, result in results.items():
        for row in result["fold_rows"]:
            rows.append({"joint": name, **row})
    pd.DataFrame(rows).to_csv(EXP / "joint_all_fold_metrics.csv", index=False)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
