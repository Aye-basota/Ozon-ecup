"""Rebuild SUBMIT_STRONGEST55_TEAMB45 from its two source predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_STRONGEST = ROOT / "frozen" / "strongest" / "submission_STRONGEST_CURRENT.csv"
DEFAULT_TEAM_B = ROOT / "frozen" / "team_b" / "final_classic_ml.csv"
DEFAULT_SAMPLE = ROOT / "reference" / "sample_submit.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "SUBMIT_STRONGEST55_TEAMB45.csv"
EXPECTED_SHA256 = "1ce85203e3069363e3d2ba425078213d1a723a895e3c684573a6c1b998a14fb4"

STRONGEST_WEIGHT = 0.55
TEAM_B_WEIGHT = 0.45


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def align_log_mean(z: np.ndarray, target_mean: float) -> tuple[np.ndarray, float]:
    """Historical level alignment, copied without numerical changes."""
    low, high = -float(np.max(z)) - 1.0, target_mean + 1.0
    for _ in range(100):
        middle = 0.5 * (low + high)
        if float(np.maximum(z + middle, 0.0).mean()) < target_mean:
            low = middle
        else:
            high = middle
    shift = 0.5 * (low + high)
    return np.maximum(z + shift, 0.0), shift


def audit_source(name: str, frame: pd.DataFrame, expected_ids: np.ndarray) -> None:
    if list(frame.columns) != ["user_id", "predict"]:
        raise AssertionError(f"{name}: wrong columns {frame.columns.tolist()}")
    if len(frame) != 250_000 or frame.user_id.nunique() != 250_000:
        raise AssertionError(f"{name}: wrong row/user count")
    if not np.array_equal(frame.user_id.to_numpy(np.int64), expected_ids):
        raise AssertionError(f"{name}: user_id order mismatch")
    values = frame.predict.to_numpy(np.float64)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise AssertionError(f"{name}: predictions must be finite and nonnegative")


def build(strongest_path: Path, team_b_path: Path, sample_path: Path, output: Path) -> dict:
    sample = pd.read_csv(sample_path)
    strongest = pd.read_csv(strongest_path)
    team_b = pd.read_csv(team_b_path)
    expected_ids = sample.user_id.to_numpy(np.int64)

    for name, frame in {"sample": sample, "strongest": strongest, "team_b": team_b}.items():
        audit_source(name, frame, expected_ids)

    z_strongest = np.log1p(strongest.predict.to_numpy(np.float64))
    z_team_raw = np.log1p(team_b.predict.to_numpy(np.float64))
    z_team, shift = align_log_mean(z_team_raw, float(z_strongest.mean()))
    z_final = STRONGEST_WEIGHT * z_strongest + TEAM_B_WEIGHT * z_team
    prediction = np.maximum(np.expm1(z_final), 0.0)

    submission = pd.DataFrame({"user_id": expected_ids, "predict": prediction})
    if submission.dtypes.to_dict() != {
        "user_id": np.dtype("int64"),
        "predict": np.dtype("float64"),
    }:
        raise AssertionError(f"unexpected dtypes: {submission.dtypes.to_dict()}")
    output.parent.mkdir(parents=True, exist_ok=True)
    # The archived reference was written by pandas on Windows.  Pin CRLF so
    # the output remains byte-identical on every platform.
    submission.to_csv(output, index=False, lineterminator="\r\n")

    return {
        "output": str(output.resolve()),
        "sha256": sha256(output),
        "rows": len(submission),
        "weights": {"submission_STRONGEST_CURRENT": STRONGEST_WEIGHT, "team_b_final": TEAM_B_WEIGHT},
        "blend_space": "log1p",
        "team_b_level_shift": shift,
        "mean_log1p": float(z_final.mean()),
        "min_predict": float(prediction.min()),
        "max_predict": float(prediction.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strongest", type=Path, default=DEFAULT_STRONGEST)
    parser.add_argument("--team-b", type=Path, default=DEFAULT_TEAM_B)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-nonreference",
        action="store_true",
        help="do not require the historical byte-exact SHA (used for retraining audits)",
    )
    args = parser.parse_args()

    result = build(args.strongest, args.team_b, args.sample, args.output)
    if not args.allow_nonreference and result["sha256"] != EXPECTED_SHA256:
        raise AssertionError(f"SHA256 mismatch: {result['sha256']} != {EXPECTED_SHA256}")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
