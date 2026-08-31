"""Build the final risk-adjusted three-solution E-Cup ensemble."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "submissions" / "SUBMIT_FINAL_3WAY_V1.csv"
REPORT = ROOT / "research" / "FINAL_THREEWAY_ENSEMBLE.json"

SOURCES = {
    "local_cap_unc_dist_seq_etx": ROOT / "submissions" / "submission_STRONGEST_CURRENT.csv",
    "joint_v2": ROOT / "submissions" / "SUBMIT_JOINT_V2.csv",
    "team_b_final": ROOT / "team-b-final" / "submissions" / "final_classic_ml.csv",
}
WEIGHTS = {
    "local_cap_unc_dist_seq_etx": 0.05,
    "joint_v2": 0.90,
    "team_b_final": 0.05,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def shift_nonnegative_log_to_mean(z: np.ndarray, target_mean: float) -> tuple[np.ndarray, float]:
    """Apply one constant log shift with z>=0 clipping and exact mean matching."""
    low, high = -float(np.max(z)) - 1.0, float(target_mean) + 1.0
    for _ in range(100):
        middle = 0.5 * (low + high)
        if float(np.maximum(z + middle, 0.0).mean()) < target_mean:
            low = middle
        else:
            high = middle
    shift = 0.5 * (low + high)
    aligned = np.maximum(z + shift, 0.0)
    return aligned, shift


def main() -> None:
    frames = {name: pd.read_csv(path) for name, path in SOURCES.items()}
    anchor = frames["joint_v2"]
    expected_columns = ["user_id", "predict"]
    anchor_ids = anchor["user_id"].to_numpy(np.int64)

    for name, frame in frames.items():
        if list(frame.columns) != expected_columns:
            raise AssertionError(f"{name}: columns={frame.columns.tolist()}")
        if len(frame) != 250_000 or frame["user_id"].nunique() != 250_000:
            raise AssertionError(f"{name}: row or user count mismatch")
        if not np.array_equal(frame["user_id"].to_numpy(np.int64), anchor_ids):
            raise AssertionError(f"{name}: row order mismatch")
        pred = frame["predict"].to_numpy(np.float64)
        if not np.isfinite(pred).all() or np.any(pred < 0):
            raise AssertionError(f"{name}: invalid predictions")

    sample = pd.read_csv(ROOT / "team-b" / "data" / "raw" / "sample_submit.csv")
    if not np.array_equal(sample["user_id"].to_numpy(np.int64), anchor_ids):
        raise AssertionError("sample submission order mismatch")

    z = {
        name: np.log1p(frame["predict"].to_numpy(np.float64))
        for name, frame in frames.items()
    }

    # The two incumbent solutions independently agree on the production level.
    # team-b-final is structurally useful but has a validated positive level bias,
    # so only its level (not its internal CURRENT_LOG_SCALE or row structure) is
    # aligned before the outer blend.
    reference_level = 0.5 * (
        float(z["local_cap_unc_dist_seq_etx"].mean())
        + float(z["joint_v2"].mean())
    )
    z_team_raw = z["team_b_final"].copy()
    z["team_b_final"], team_level_shift = shift_nonnegative_log_to_mean(
        z_team_raw,
        reference_level,
    )

    z_final = sum(WEIGHTS[name] * z[name] for name in WEIGHTS)
    z_final = np.maximum(z_final, 0.0)
    prediction = np.maximum(np.expm1(z_final), 0.0)
    submission = pd.DataFrame({"user_id": anchor_ids, "predict": prediction})

    if list(submission.columns) != expected_columns:
        raise AssertionError("final column order mismatch")
    if submission.dtypes["user_id"] != np.dtype("int64"):
        raise AssertionError(f"unexpected user_id dtype: {submission.dtypes['user_id']}")
    if submission.dtypes["predict"] != np.dtype("float64"):
        raise AssertionError(f"unexpected predict dtype: {submission.dtypes['predict']}")
    if len(submission) != 250_000 or submission.user_id.nunique() != 250_000:
        raise AssertionError("final row or user count mismatch")
    if not np.isfinite(prediction).all() or np.any(prediction < 0):
        raise AssertionError("final predictions are invalid")
    if not np.array_equal(submission.user_id.to_numpy(np.int64), sample.user_id.to_numpy(np.int64)):
        raise AssertionError("final sample order mismatch")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUTPUT, index=False)

    current_log_scale_source = ROOT / "team-b-final" / "src" / "predict.py"
    if "CURRENT_LOG_SCALE = 1.12" not in current_log_scale_source.read_text(encoding="utf-8"):
        raise AssertionError("CURRENT_LOG_SCALE=1.12 contract was changed")

    validation_path = ROOT / "team-b-final" / "validation_summary.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    correlations = pd.DataFrame(z).corr()
    source_stats = {}
    for name in SOURCES:
        source_stats[name] = {
            "path": str(SOURCES[name].resolve()),
            "sha256": sha256(SOURCES[name]),
            "weight": WEIGHTS[name],
            "mean_log1p_raw": float(z_team_raw.mean()) if name == "team_b_final" else float(z[name].mean()),
            "mean_log1p_used": float(z[name].mean()),
            "std_log1p_used": float(z[name].std()),
            "zeros_raw": int((frames[name]["predict"].to_numpy() == 0).sum()),
        }

    report = {
        "output": str(OUTPUT.resolve()),
        "output_sha256": sha256(OUTPUT),
        "rows": int(len(submission)),
        "columns": expected_columns,
        "dtypes": {col: str(dtype) for col, dtype in submission.dtypes.items()},
        "nan_count": int(submission.isna().sum().sum()),
        "finite": bool(np.isfinite(prediction).all()),
        "nonnegative": bool(np.all(prediction >= 0)),
        "same_order_as_sample": True,
        "unique_user_id": int(submission.user_id.nunique()),
        "zeros": int((prediction == 0).sum()),
        "min_predict": float(prediction.min()),
        "max_predict": float(prediction.max()),
        "mean_predict": float(prediction.mean()),
        "mean_log1p": float(z_final.mean()),
        "std_log1p": float(z_final.std()),
        "blend_space": "log1p",
        "weights": WEIGHTS,
        "reference_level": reference_level,
        "team_b_level_shift": team_level_shift,
        "team_b_internal_current_log_scale": 1.12,
        "sources": source_stats,
        "prediction_correlations_after_level_alignment": correlations.to_dict(),
        "team_b_validation": validation,
        "known_public_lb": {
            "submission_STRONGEST_CURRENT.csv": 1.6496571902356205,
            "SUBMIT_JOINT_V2.csv": 1.6459363044782171,
            "team_b_final": None,
        },
        "offline_evidence": {
            "local_cap_unc_dist_seq_etx": {
                "validation": "canonical four-fold OOF with per-fold log-offset calibration",
                "fold_rmsle": [
                    1.7668833567997195,
                    1.7605095767798136,
                    1.748629223964952,
                    1.7412785664479717,
                ],
                "fold_weights": [1.0, 2.0, 4.0, 8.0],
                "weighted_cv": 1.7475098625201952,
                "public_geometry_optimal_weight_from_joint_v2": 0.010101354671869523,
            },
            "joint_v2": {
                "exact_common_oof_score": None,
                "predecessor_exp075_nested_delta_rmsle": -0.00125067,
                "predecessor_all_four_fold_signs_positive": True,
                "joint_v2_plane_only_delta_vs_exp075_oof": 0.0003779151228466837,
                "out_of_plane_local_value": None,
            },
            "team_b_final": {
                "two_fold_mean_raw_rmsle": validation["mean"]["final_rmsle"],
                "two_fold_mean_level_corrected_rmsle": validation["mean"]["final_shifted_rmsle"],
                "two_fold_mean_team_only_rmsle": validation["mean"]["team_rmsle"],
                "two_fold_mean_current_team_error_corr": validation["mean"]["current_team_error_corr"],
                "scored_span_optimal_weight_range_rcond_1e_4_to_1e_6": [
                    0.08558791561384882,
                    0.10781536701752063,
                ],
                "chosen_weight_after_local_risk_shrinkage": WEIGHTS["team_b_final"],
            },
            "common_oof_final_threeway_rmsle": None,
            "common_oof_limitation": "JOINT_V2 and team-b-final do not have predictions on the canonical same-row OOF panel; no synthetic final CV was reported",
        },
        "test_geometry": {
            "raw_prediction_correlations": {
                "local_vs_joint_v2": 0.9980468812562309,
                "local_vs_team_b_final": 0.9960477127961432,
                "joint_v2_vs_team_b_final": 0.9961066418444048,
            },
            "local_vs_joint_v2_post_scored_span_rms_fraction": 0.00034304601106224206,
            "local_vs_joint_v2_post_scored_span_energy_fraction": 1.176805657057159e-07,
            "team_b_final_vs_joint_v2_post_scored_span_rms_fraction": 0.29450682186259836,
            "team_b_final_vs_joint_v2_post_scored_span_energy_fraction": 0.08673426812360824,
        },
        "weight_rationale": {
            "joint_v2": "dominant anchor: best exact LB and already absorbs nearly all scored-span information",
            "local_cap_unc_dist_seq_etx": "small robustness reserve: strongest canonical OOF evidence, but almost entirely duplicated by the scored span",
            "team_b_final": "half-shrunk scored-span optimum: real test novelty, but fixed blend loses on both OOT folds and the novel residual has no target-alignment evidence",
        },
        "public_lb_forecast": {
            "point": 1.64590,
            "reasonable_range": [1.64586, 1.64605],
            "status": "estimate, not fact",
            "assumption": "50% shrinkage of the stable scored-span optimum for team-b-final; no value assigned to its unvalidated post-span residual",
        },
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({
        "output": str(OUTPUT),
        "sha256": report["output_sha256"],
        "weights": WEIGHTS,
        "team_b_level_shift": team_level_shift,
        "mean_log1p": report["mean_log1p"],
        "zeros": report["zeros"],
        "min": report["min_predict"],
        "max": report["max_predict"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
