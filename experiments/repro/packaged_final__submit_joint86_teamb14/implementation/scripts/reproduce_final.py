"""Single entry point for the two packaged Team-A final submissions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_SHA256 = "5f3aa90992652b8a4f0f398e735a3ba11c2ea6ccf9e8fb1d236436e9a49167c0"
SOLUTIONS = {
    "SUBMIT_STRONGEST55_TEAMB45": {
        "package": ROOT / "reproducibility" / "SUBMIT_STRONGEST55_TEAMB45",
        "reference": "reference/SUBMIT_STRONGEST55_TEAMB45.csv",
        "precomputed_output": "outputs/SUBMIT_STRONGEST55_TEAMB45.csv",
        "expected": "1ce85203e3069363e3d2ba425078213d1a723a895e3c684573a6c1b998a14fb4",
    },
    "SUBMIT_JOINT86_TEAMB14": {
        "package": ROOT / "reproducibility" / "SUBMIT_JOINT86_TEAMB14",
        "reference": "reference/SUBMIT_JOINT86_TEAMB14.csv",
        "precomputed_output": "outputs/SUBMIT_JOINT86_TEAMB14_REBUILT.csv",
        "expected": "85d9cd645e14a7895da9ad8cc89065714606266be588c762d37487d2b4edac02",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], cwd: Path) -> None:
    print("RUN", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def audit(reference_path: Path, reproduced_path: Path, expected: str) -> dict[str, object]:
    reference = pd.read_csv(reference_path)
    reproduced = pd.read_csv(reproduced_path)
    required = ["user_id", "predict"]
    for name, frame in {"reference": reference, "reproduced": reproduced}.items():
        if list(frame.columns) != required:
            raise AssertionError(f"{name}: columns={frame.columns.tolist()}")
        if len(frame) != 250_000 or frame.user_id.nunique() != 250_000:
            raise AssertionError(f"{name}: row/user count mismatch")
        values = frame.predict.to_numpy(np.float64)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise AssertionError(f"{name}: invalid predictions")
    if not np.array_equal(reference.user_id.to_numpy(np.int64), reproduced.user_id.to_numpy(np.int64)):
        raise AssertionError("user_id alignment mismatch")
    actual = reproduced.predict.to_numpy(np.float64)
    target = reference.predict.to_numpy(np.float64)
    log_diff = np.log1p(actual) - np.log1p(target)
    reproduced_sha = sha256(reproduced_path)
    return {
        "EXPECTED_SHA256": expected,
        "REPRODUCED_SHA256": reproduced_sha,
        "BYTE_IDENTICAL": "YES" if reproduced_sha == expected else "NO",
        "max_abs_prediction_difference": float(np.max(np.abs(actual - target))),
        "RMS_log_space_difference": float(np.sqrt(np.mean(log_diff**2))),
        "rows": int(len(reproduced)),
        "unique_user_id": int(reproduced.user_id.nunique()),
        "finite_predictions": True,
        "nonnegative_predictions": True,
        "reference": str(reference_path),
        "reproduced": str(reproduced_path),
    }


def reproduce_precomputed(solution: str, python: Path) -> Path:
    config = SOLUTIONS[solution]
    package = Path(config["package"])
    run([str(python), str(package / "build_submit.py")], package)
    return package / str(config["precomputed_output"])


def require_raw(path: Path | None) -> Path:
    raw = (path or ROOT / "data" / "raw" / "train.parquet").expanduser().resolve()
    if not raw.is_file():
        raise FileNotFoundError(
            f"raw data not found: {raw}; pass --raw-data or place train.parquet in data/raw"
        )
    digest = sha256(raw)
    if digest != RAW_SHA256:
        raise AssertionError(f"raw SHA256 mismatch: {digest} != {RAW_SHA256}")
    return raw


def reproduce_raw(
    solution: str,
    raw_data: Path,
    strongest_python: Path,
    team_b_python: Path,
) -> Path:
    config = SOLUTIONS[solution]
    package = Path(config["package"])
    if solution == "SUBMIT_STRONGEST55_TEAMB45":
        work = package / "work" / "tabular"
        run(
            [
                str(team_b_python),
                str(package / "scripts" / "run_tabular.py"),
                "--raw-data", str(raw_data),
                "--work-dir", str(work),
                "--team-processed-dir", str(work / "team_b" / "processed"),
                "--strongest-python", str(strongest_python),
                "--team-python", str(team_b_python),
                "--skip-model-pickle",
            ],
            package,
        )
        return work / "SUBMIT_STRONGEST55_TEAMB45_RETRAINED.csv"

    team_work = package / "work" / "team_b"
    team_csv = team_work / "final_classic_ml_RETRAINED.csv"
    output = package / "work" / "SUBMIT_JOINT86_TEAMB14_RETRAINED.csv"
    run(
        [
            str(team_b_python),
            "-m", "src.predict",
            "--raw-data", str(raw_data),
            "--output", str(team_csv),
        ],
        package / "team-b-final",
    )
    run(
        [
            str(team_b_python),
            str(package / "build_submit.py"),
            "--team-b", str(team_csv),
            "--output", str(output),
            "--allow-nonreference",
        ],
        package,
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", required=True, choices=sorted(SOLUTIONS))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--from-precomputed", action="store_true")
    mode.add_argument("--from-raw", action="store_true")
    parser.add_argument("--raw-data", type=Path)
    parser.add_argument("--strongest-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--team-b-python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()

    config = SOLUTIONS[args.solution]
    package = Path(config["package"])
    if args.from_precomputed:
        reproduced = reproduce_precomputed(args.solution, args.team_b_python)
        mode_name = "from_precomputed"
    else:
        reproduced = reproduce_raw(
            args.solution,
            require_raw(args.raw_data),
            args.strongest_python,
            args.team_b_python,
        )
        mode_name = "from_raw"
    result = audit(package / str(config["reference"]), reproduced, str(config["expected"]))
    result.update({"solution": args.solution, "mode": mode_name})
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
