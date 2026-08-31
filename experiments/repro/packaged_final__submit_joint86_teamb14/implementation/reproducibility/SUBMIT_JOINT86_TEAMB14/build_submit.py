"""Rebuild SUBMIT_JOINT86_TEAMB14.csv from the two frozen source predictions."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
JOINT = ROOT / "inputs" / "SUBMIT_JOINT_V2.csv"
TEAM = ROOT / "team-b-final" / "submissions" / "final_classic_ml.csv"
SAMPLE = ROOT / "team-b-final" / "data" / "sample_submit.csv"
OUTPUT = ROOT / "outputs" / "SUBMIT_JOINT86_TEAMB14_REBUILT.csv"

JOINT_WEIGHT = 0.86
TEAM_WEIGHT = 0.14
EXPECTED_SHA256 = "85d9cd645e14a7895da9ad8cc89065714606266be588c762d37487d2b4edac02"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def align_log_mean(z: np.ndarray, target_mean: float) -> np.ndarray:
    low, high = -float(np.max(z)) - 1.0, target_mean + 1.0
    for _ in range(100):
        middle = 0.5 * (low + high)
        if float(np.maximum(z + middle, 0.0).mean()) < target_mean:
            low = middle
        else:
            high = middle
    return np.maximum(z + 0.5 * (low + high), 0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint", type=Path, default=JOINT)
    parser.add_argument("--team-b", type=Path, default=TEAM)
    parser.add_argument("--sample", type=Path, default=SAMPLE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--allow-nonreference", action="store_true")
    args = parser.parse_args()

    joint = pd.read_csv(args.joint)
    team = pd.read_csv(args.team_b)
    sample = pd.read_csv(args.sample)

    for name, frame in {"joint": joint, "team": team, "sample": sample}.items():
        if list(frame.columns) != ["user_id", "predict"]:
            raise AssertionError(f"{name}: wrong columns")
        if len(frame) != 250_000 or frame.user_id.nunique() != 250_000:
            raise AssertionError(f"{name}: wrong row/user count")
        if not np.array_equal(frame.user_id.to_numpy(np.int64), sample.user_id.to_numpy(np.int64)):
            raise AssertionError(f"{name}: user_id order mismatch")

    z_joint = np.log1p(joint.predict.to_numpy(np.float64))
    z_team_raw = np.log1p(team.predict.to_numpy(np.float64))
    z_team = align_log_mean(z_team_raw, float(z_joint.mean()))
    prediction = np.maximum(
        np.expm1(JOINT_WEIGHT * z_joint + TEAM_WEIGHT * z_team),
        0.0,
    )

    submission = pd.DataFrame({"user_id": sample.user_id.to_numpy(np.int64), "predict": prediction})
    if submission.isna().any().any() or not np.isfinite(prediction).all():
        raise AssertionError("NaN/Inf in final submission")
    if np.any(prediction < 0):
        raise AssertionError("negative predictions")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # The archived reference was written by pandas on Windows.  Pin CRLF so
    # the output remains byte-identical on every platform.
    submission.to_csv(args.output, index=False, lineterminator="\r\n")
    digest = file_sha256(args.output)
    if not args.allow_nonreference and digest != EXPECTED_SHA256:
        raise AssertionError(f"SHA256 mismatch: {digest} != {EXPECTED_SHA256}")
    print(f"created: {args.output}")
    print(f"rows: {len(submission)}")
    print(f"sha256: {digest}")


if __name__ == "__main__":
    main()
