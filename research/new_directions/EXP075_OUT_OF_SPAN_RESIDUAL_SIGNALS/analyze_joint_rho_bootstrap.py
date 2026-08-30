from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


EXP = Path(__file__).resolve().parent
FOLDS = ["2025-09-04", "2025-09-18", "2025-10-02", "2025-10-16"]
FW = np.asarray([1.0, 2.0, 4.0, 8.0], dtype=np.float64)


def main() -> None:
    a1 = pd.read_parquet(EXP / "clean_forward_predictions.parquet", columns=["user_id", "cutoff", "residual", "u_perp_365"])
    a2 = pd.read_parquet(EXP / "a2_clean_forward_predictions.parquet", columns=["user_id", "cutoff", "u_perp_A2"])
    d = a1.merge(a2, on=["user_id", "cutoff"], validate="one_to_one")
    fold_index = np.asarray([FOLDS.index(x) for x in d.cutoff], dtype=np.int8)
    counts = np.bincount(fold_index, minlength=4)
    w = FW[fold_index] / counts[fold_index]
    w /= w.sum()
    u1 = d.u_perp_365.to_numpy(np.float64)
    u2 = d.u_perp_A2.to_numpy(np.float64)
    r = d.residual.to_numpy(np.float64)
    uid, inv = np.unique(d.user_id.to_numpy(np.int64), return_inverse=True)
    row_stats = np.column_stack([
        w, w * u1, w * u2, w * r,
        w * u1 * u1, w * u1 * u2, w * u2 * u2,
        w * u1 * r, w * u2 * r, w * r * r,
    ])
    cluster = np.empty((len(uid), row_stats.shape[1]), dtype=np.float64)
    for j in range(row_stats.shape[1]):
        cluster[:, j] = np.bincount(inv, weights=row_stats[:, j], minlength=len(uid))

    rng = np.random.default_rng(2026082802)
    draws: list[float] = []
    for _ in range(50):
        mult = rng.poisson(1.0, size=(20, len(uid))).astype(np.float64)
        for s in mult @ cluster:
            sw = s[0]
            mu1, mu2, mur = s[1] / sw, s[2] / sw, s[3] / sw
            G = np.asarray([
                [s[4] / sw - mu1 * mu1, s[5] / sw - mu1 * mu2],
                [s[5] / sw - mu1 * mu2, s[6] / sw - mu2 * mu2],
            ])
            b = np.asarray([s[7] / sw - mu1 * mur, s[8] / sw - mu2 * mur])
            vr = s[9] / sw - mur * mur
            rho2 = float(b @ np.linalg.solve(G, b) / vr)
            draws.append(np.sqrt(max(rho2, 0.0)))
    draws_array = np.asarray(draws)
    point = json.loads((EXP / "joint_all_analysis.json").read_text(encoding="utf-8"))["A1_365_PLUS_A2"]["rho_joint"]
    result = {
        "method": "Poisson user-cluster bootstrap of weighted oracle joint rho; all rows for a user share one count",
        "replicates": int(len(draws_array)),
        "unique_users": int(len(uid)),
        "rho_point": point,
        "rho_ci_2_5": float(np.quantile(draws_array, 0.025)),
        "rho_ci_97_5": float(np.quantile(draws_array, 0.975)),
        "rho_bootstrap_se": float(np.std(draws_array, ddof=1)),
        "t_rho": float(point / np.std(draws_array, ddof=1)),
    }
    (EXP / "joint_rho_bootstrap.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(EXP / "joint_rho_bootstrap_draws.npz", rho=draws_array)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
