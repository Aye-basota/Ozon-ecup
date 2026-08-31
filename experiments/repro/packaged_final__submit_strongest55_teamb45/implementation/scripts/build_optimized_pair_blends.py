"""Build two LB-calibrated pair blends with the level-aligned team-b-final."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "team-b" / "data" / "raw" / "sample_submit.csv"
STRONGEST = ROOT / "submissions" / "submission_STRONGEST_CURRENT.csv"
JOINT_V2 = ROOT / "submissions" / "SUBMIT_JOINT_V2.csv"
TEAM_B = ROOT / "team-b-final" / "submissions" / "final_classic_ml.csv"
REPORT = ROOT / "research" / "OPTIMIZED_PAIR_BLENDS.json"

CONFIGS = {
    "strongest55_teamb45": {
        "anchor": "strongest",
        "anchor_weight": 0.55,
        "team_weight": 0.45,
        "output": ROOT / "submissions" / "SUBMIT_STRONGEST55_TEAMB45.csv",
        "forecast": 1.64823,
        "forecast_range": [1.64818, 1.64834],
    },
    "joint86_teamb14": {
        "anchor": "joint_v2",
        "anchor_weight": 0.86,
        "team_weight": 0.14,
        "output": ROOT / "submissions" / "SUBMIT_JOINT86_TEAMB14.csv",
        "forecast": 1.64582,
        "forecast_range": [1.64579, 1.64588],
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def align_log_mean(z: np.ndarray, target_mean: float) -> tuple[np.ndarray, float]:
    """Shift in log space, clip to z>=0, and match target mean exactly."""
    low, high = -float(np.max(z)) - 1.0, target_mean + 1.0
    for _ in range(100):
        middle = 0.5 * (low + high)
        if float(np.maximum(z + middle, 0.0).mean()) < target_mean:
            low = middle
        else:
            high = middle
    shift = 0.5 * (low + high)
    aligned = np.maximum(z + shift, 0.0)
    return aligned, shift


def audit_source(name: str, frame: pd.DataFrame, expected_ids: np.ndarray) -> None:
    if list(frame.columns) != ["user_id", "predict"]:
        raise AssertionError(f"{name}: wrong columns {frame.columns.tolist()}")
    if len(frame) != 250_000 or frame.user_id.nunique() != 250_000:
        raise AssertionError(f"{name}: wrong row/user count")
    if not np.array_equal(frame.user_id.to_numpy(np.int64), expected_ids):
        raise AssertionError(f"{name}: row order mismatch")
    values = frame.predict.to_numpy(np.float64)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise AssertionError(f"{name}: invalid predictions")


def main() -> None:
    sample = pd.read_csv(SAMPLE)
    expected_ids = sample.user_id.to_numpy(np.int64)
    frames = {
        "strongest": pd.read_csv(STRONGEST),
        "joint_v2": pd.read_csv(JOINT_V2),
        "team_b": pd.read_csv(TEAM_B),
    }
    for name, frame in frames.items():
        audit_source(name, frame, expected_ids)

    scale_source = (ROOT / "team-b-final" / "src" / "predict.py").read_text(
        encoding="utf-8"
    )
    if "CURRENT_LOG_SCALE = 1.12" not in scale_source:
        raise AssertionError("CURRENT_LOG_SCALE=1.12 contract changed")

    logs = {
        name: np.log1p(frame.predict.to_numpy(np.float64))
        for name, frame in frames.items()
    }
    results: dict[str, object] = {}

    for name, config in CONFIGS.items():
        anchor_name = str(config["anchor"])
        anchor = logs[anchor_name]
        team_aligned, shift = align_log_mean(logs["team_b"], float(anchor.mean()))
        anchor_weight = float(config["anchor_weight"])
        team_weight = float(config["team_weight"])
        if not np.isclose(anchor_weight + team_weight, 1.0):
            raise AssertionError(f"{name}: weights do not sum to one")

        z_final = anchor_weight * anchor + team_weight * team_aligned
        prediction = np.maximum(np.expm1(z_final), 0.0)
        submission = pd.DataFrame({"user_id": expected_ids, "predict": prediction})

        if list(submission.columns) != list(sample.columns):
            raise AssertionError(f"{name}: final columns/order mismatch")
        if submission.dtypes.to_dict() != {
            "user_id": np.dtype("int64"),
            "predict": np.dtype("float64"),
        }:
            raise AssertionError(f"{name}: unexpected dtypes")
        if submission.isna().any().any() or not np.isfinite(prediction).all():
            raise AssertionError(f"{name}: NaN/Inf")
        if np.any(prediction < 0) or not submission.user_id.is_unique:
            raise AssertionError(f"{name}: invalid predictions/user ids")

        output = Path(config["output"])
        submission.to_csv(output, index=False)
        results[name] = {
            "output": str(output.resolve()),
            "sha256": sha256(output),
            "blend_space": "log1p",
            "weights": {anchor_name: anchor_weight, "team_b_final": team_weight},
            "team_b_level_shift": shift,
            "team_b_internal_current_log_scale": 1.12,
            "source_log_correlation": float(np.corrcoef(anchor, logs["team_b"])[0, 1]),
            "rows": int(len(submission)),
            "unique_user_id": int(submission.user_id.nunique()),
            "same_order_as_sample": bool(submission.user_id.equals(sample.user_id)),
            "columns": submission.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in submission.dtypes.items()},
            "nan_count": int(submission.isna().sum().sum()),
            "finite": bool(np.isfinite(prediction).all()),
            "nonnegative": bool(np.all(prediction >= 0)),
            "min_predict": float(prediction.min()),
            "max_predict": float(prediction.max()),
            "mean_predict": float(prediction.mean()),
            "mean_log1p": float(z_final.mean()),
            "std_log1p": float(z_final.std()),
            "public_lb_forecast": {
                "point": float(config["forecast"]),
                "reasonable_range": config["forecast_range"],
                "status": "estimate calibrated with observed STRONGEST80/TEAM20 LB",
            },
        }

    report = {
        "known_public_lb": {
            "submission_STRONGEST_CURRENT.csv": 1.6496571902356205,
            "SUBMIT_STRONGEST80_TEAMB20.csv": 1.6486550051601747,
            "SUBMIT_JOINT_V2.csv": 1.6459363044782171,
        },
        "weight_method": (
            "quadratic RMSLE geometry calibrated by the three known public scores, "
            "then rounded to stable weights near the inferred minima"
        ),
        "results": results,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
