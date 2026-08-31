from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


EXP = Path(__file__).resolve().parent
FOLDS = ["2025-09-04", "2025-09-18", "2025-10-02", "2025-10-16"]
FW = np.asarray([1.0, 2.0, 4.0, 8.0], dtype=np.float64)


def weighted_corr(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    w = w / w.sum()
    x0 = x - np.sum(w * x)
    y0 = y - np.sum(w * y)
    return float(np.sum(w * x0 * y0) / np.sqrt(np.sum(w * x0 * x0) * np.sum(w * y0 * y0)))


def main() -> None:
    a1 = pd.read_parquet(EXP / "clean_forward_predictions.parquet")
    a2 = pd.read_parquet(EXP / "a2_clean_forward_predictions.parquet", columns=["user_id", "cutoff", "u_perp_A2", "amplitude_A2"])
    d = a1.merge(a2, on=["user_id", "cutoff"], validate="one_to_one")
    fi = np.asarray([FOLDS.index(x) for x in d.cutoff], dtype=np.int8)
    counts = np.bincount(fi, minlength=4)
    w = FW[fi] / counts[fi]
    r = d.residual.to_numpy(np.float64)
    joint = json.loads((EXP / "joint_all_analysis.json").read_text(encoding="utf-8"))["A1_365_PLUS_A2"]
    coefs = np.asarray([row["coefficients"] for row in joint["fold_rows"]])
    corrections = {
        "A1_TREE_TRAJ_180": d.amplitude_180.to_numpy(float) * d.u_perp_180.to_numpy(float),
        "A1_TREE_TRAJ_365": d.amplitude_365.to_numpy(float) * d.u_perp_365.to_numpy(float),
        "A2_WEEKLY_RESIDUAL_CNN": d.amplitude_A2.to_numpy(float) * d.u_perp_A2.to_numpy(float),
        "JOINT_A1_365_A2": coefs[fi, 0] * d.u_perp_365.to_numpy(float) + coefs[fi, 1] * d.u_perp_A2.to_numpy(float),
    }
    test = json.loads((EXP / "test_span_projection.json").read_text(encoding="utf-8"))["candidates"]
    result = {}
    for name, correction in corrections.items():
        result[name] = {
            "corr_candidate_corrected_residual_vs_existing_baseline_residual": weighted_corr(r - correction, r, w),
            "corr_candidate_correction_vs_existing_baseline_residual": weighted_corr(correction, r, w),
            "test_corr_perp_correction_vs_current_ORTH_correction": test[name]["corr_D_perp_current_ORTH"],
        }
    result["A1_365_vs_A2_correction_correlation"] = joint["correction_correlation_matrix"][0][1]
    (EXP / "residual_correlation_analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
