"""Build an 80/20 log-space blend of STRONGEST_CURRENT and team-b-final."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STRONGEST = ROOT / "submissions" / "submission_STRONGEST_CURRENT.csv"
TEAM_B = ROOT / "team-b-final" / "submissions" / "final_classic_ml.csv"
SAMPLE = ROOT / "team-b" / "data" / "raw" / "sample_submit.csv"
OUTPUT = ROOT / "submissions" / "SUBMIT_STRONGEST80_TEAMB20.csv"
REPORT = ROOT / "research" / "STRONGEST80_TEAMB20.json"

STRONGEST_WEIGHT = 0.80
TEAM_B_WEIGHT = 0.20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def align_log_mean(z: np.ndarray, target_mean: float) -> tuple[np.ndarray, float]:
    low, high = -float(np.max(z)) - 1.0, target_mean + 1.0
    for _ in range(100):
        middle = 0.5 * (low + high)
        if float(np.maximum(z + middle, 0.0).mean()) < target_mean:
            low = middle
        else:
            high = middle
    shift = 0.5 * (low + high)
    return np.maximum(z + shift, 0.0), shift


def main() -> None:
    strongest = pd.read_csv(STRONGEST)
    team_b = pd.read_csv(TEAM_B)
    sample = pd.read_csv(SAMPLE)

    for name, frame in {"strongest": strongest, "team_b": team_b}.items():
        if list(frame.columns) != ["user_id", "predict"]:
            raise AssertionError(f"{name}: wrong columns")
        if len(frame) != 250_000 or frame.user_id.nunique() != 250_000:
            raise AssertionError(f"{name}: wrong row/user count")
        pred = frame.predict.to_numpy(np.float64)
        if not np.isfinite(pred).all() or np.any(pred < 0):
            raise AssertionError(f"{name}: invalid predictions")

    uid = strongest.user_id.to_numpy(np.int64)
    if not np.array_equal(team_b.user_id.to_numpy(np.int64), uid):
        raise AssertionError("source row order mismatch")
    if not np.array_equal(sample.user_id.to_numpy(np.int64), uid):
        raise AssertionError("sample row order mismatch")

    z_strongest = np.log1p(strongest.predict.to_numpy(np.float64))
    z_team_raw = np.log1p(team_b.predict.to_numpy(np.float64))
    z_team, level_shift = align_log_mean(z_team_raw, float(z_strongest.mean()))
    z_final = STRONGEST_WEIGHT * z_strongest + TEAM_B_WEIGHT * z_team
    prediction = np.maximum(np.expm1(z_final), 0.0)

    submission = pd.DataFrame({"user_id": uid, "predict": prediction})
    if submission.dtypes.to_dict() != {
        "user_id": np.dtype("int64"),
        "predict": np.dtype("float64"),
    }:
        raise AssertionError(f"unexpected dtypes: {submission.dtypes.to_dict()}")
    if submission.isna().any().any() or not np.isfinite(prediction).all():
        raise AssertionError("NaN/Inf in final submission")
    if np.any(prediction < 0):
        raise AssertionError("negative prediction")

    scale_source = (ROOT / "team-b-final" / "src" / "predict.py").read_text(encoding="utf-8")
    if "CURRENT_LOG_SCALE = 1.12" not in scale_source:
        raise AssertionError("CURRENT_LOG_SCALE=1.12 contract changed")

    submission.to_csv(OUTPUT, index=False)
    report = {
        "output": str(OUTPUT.resolve()),
        "sha256": sha256(OUTPUT),
        "blend_space": "log1p",
        "weights": {
            "submission_STRONGEST_CURRENT": STRONGEST_WEIGHT,
            "team_b_final": TEAM_B_WEIGHT,
        },
        "team_b_internal_current_log_scale": 1.12,
        "team_b_level_shift": level_shift,
        "source_correlation": float(np.corrcoef(z_strongest, z_team_raw)[0, 1]),
        "centered_source_difference_rms": float(np.std(z_strongest - z_team_raw)),
        "rows": int(len(submission)),
        "unique_user_id": int(submission.user_id.nunique()),
        "same_order_as_sample": True,
        "columns": submission.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in submission.dtypes.items()},
        "nan_count": int(submission.isna().sum().sum()),
        "finite": bool(np.isfinite(prediction).all()),
        "nonnegative": bool(np.all(prediction >= 0)),
        "zeros": int((prediction == 0).sum()),
        "min_predict": float(prediction.min()),
        "max_predict": float(prediction.max()),
        "mean_predict": float(prediction.mean()),
        "mean_log1p": float(z_final.mean()),
        "std_log1p": float(z_final.std()),
        "known_strongest_public_lb": 1.6496571902356205,
        "public_lb_forecast": {
            "point": 1.64875,
            "reasonable_range": [1.64855, 1.64915],
            "status": "estimate, not fact",
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
