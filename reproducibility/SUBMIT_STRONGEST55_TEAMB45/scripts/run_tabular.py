"""Actually retrain and audit every tabular component used by the final submit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
STRONGEST_CODE = ROOT / "strongest"
TEAM_B_CODE = ROOT / "team_b"
FROZEN_PREDICTIONS = ROOT / "frozen" / "strongest" / "predictions"
FROZEN_STRONGEST = ROOT / "frozen" / "strongest" / "submission_STRONGEST_CURRENT.csv"
FROZEN_TEAM_B = ROOT / "frozen" / "team_b" / "final_classic_ml.csv"
REFERENCE_FINAL = ROOT / "reference" / "SUBMIT_STRONGEST55_TEAMB45.csv"
SAMPLE = ROOT / "reference" / "sample_submit.csv"
DEFAULT_RAW = (
    REPO_ROOT
    / "delivery"
    / "submission_STRONGEST_CURRENT_training_bundle_v2"
    / "pipeline"
    / "data"
    / "raw"
    / "train.parquet"
)
DEFAULT_TEAM_CACHE = REPO_ROOT / "team-b" / "data" / "processed"
EXPECTED_RAW_SHA256 = "5f3aa90992652b8a4f0f398e735a3ba11c2ea6ccf9e8fb1d236436e9a49167c0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print("RUN:", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def remove_generated_feature_cache(processed: Path) -> None:
    """Keep disk bounded between historically distinct tabular variants."""
    paths = list(processed.glob("feat_*.parquet"))
    total = sum(path.stat().st_size for path in paths)
    for path in paths:
        path.unlink()
    if paths:
        print(f"removed {len(paths)} generated feature caches ({total / 2**30:.2f} GiB)", flush=True)


def compare_vectors(actual_dir: Path) -> dict:
    result = {}
    for name in ["S1-UNC", "S1-CAP", "S1-DIST"]:
        actual_uid = np.load(actual_dir / f"uid_{name}.npy")
        expected_uid = np.load(FROZEN_PREDICTIONS / f"uid_{name}.npy")
        if not np.array_equal(actual_uid, expected_uid):
            raise AssertionError(f"user_id mismatch for {name}")
        actual = np.load(actual_dir / f"ztest_{name}.npy").astype(np.float64)
        expected = np.load(FROZEN_PREDICTIONS / f"ztest_{name}.npy").astype(np.float64)
        if not np.isfinite(actual).all() or len(actual) != 250_000:
            raise AssertionError(f"invalid retrained vector: {name}")
        diff = actual - expected
        result[name] = {
            "bitwise_equal": bool(np.array_equal(actual, expected)),
            "max_abs_log_error": float(np.max(np.abs(diff))),
            "mean_abs_log_error": float(np.mean(np.abs(diff))),
            "rms_log_error": float(np.sqrt(np.mean(diff**2))),
            "correlation": float(np.corrcoef(actual, expected)[0, 1]),
        }
    return result


def compare_csv(actual_path: Path, expected_path: Path) -> dict:
    actual = pd.read_csv(actual_path)
    expected = pd.read_csv(expected_path)
    if list(actual.columns) != ["user_id", "predict"] or len(actual) != 250_000:
        raise AssertionError(f"invalid CSV: {actual_path}")
    if not np.array_equal(actual.user_id.to_numpy(np.int64), expected.user_id.to_numpy(np.int64)):
        raise AssertionError(f"user_id mismatch: {actual_path}")
    za = np.log1p(actual.predict.to_numpy(np.float64))
    ze = np.log1p(expected.predict.to_numpy(np.float64))
    diff = za - ze
    return {
        "actual_sha256": sha256(actual_path),
        "expected_sha256": sha256(expected_path),
        "byte_equal": sha256(actual_path) == sha256(expected_path),
        "max_abs_log_error": float(np.max(np.abs(diff))),
        "mean_abs_log_error": float(np.mean(np.abs(diff))),
        "rms_log_error": float(np.sqrt(np.mean(diff**2))),
        "correlation": float(np.corrcoef(za, ze)[0, 1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=["all", "strongest", "team-b"], default="all")
    parser.add_argument("--raw-data", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--work-dir", type=Path, default=ROOT / "work" / "tabular")
    parser.add_argument("--team-processed-dir", type=Path, default=DEFAULT_TEAM_CACHE)
    parser.add_argument("--strongest-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--team-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--lgb-threads", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument("--skip-model-pickle", action="store_true")
    args = parser.parse_args()

    raw_data = args.raw_data.expanduser().resolve()
    if not raw_data.is_file():
        raise FileNotFoundError(raw_data)
    raw_digest = sha256(raw_data)
    if raw_digest != EXPECTED_RAW_SHA256:
        raise AssertionError(f"raw data SHA256 mismatch: {raw_digest}")

    work = args.work_dir.expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "raw_data": str(raw_data),
        "raw_data_sha256": raw_digest,
        "started_at_epoch": time.time(),
    }

    strongest_csv = FROZEN_STRONGEST
    team_csv = FROZEN_TEAM_B
    if args.branch in {"all", "strongest"}:
        strongest_work = work / "strongest"
        artifacts = strongest_work / "artifacts"
        processed = strongest_work / "processed"
        submissions = strongest_work / "submissions"
        for path in [artifacts, processed, submissions]:
            path.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update({
            "ECUP_RAW_DATA_DIR": str(raw_data.parent),
            "ECUP_PROCESSED_DIR": str(processed),
            "ECUP_ARTIFACTS_DIR": str(artifacts),
            "ECUP_SUBMISSIONS_DIR": str(submissions),
            "LGB_THREADS": str(args.lgb_threads),
            "PYTHONIOENCODING": "utf-8",
        })
        commands = [
            ["-m", "src.predict", "--exp", "S1-UNC", "--variant", "S1-UNC", "--L", "0", "--min-history", "90", "--train-blocks", "1"],
            ["-m", "src.predict", "--exp", "S1-CAP", "--variant", "S1-CAP", "--L", "180", "--min-history", "90", "--train-blocks", "1"],
            ["-m", "src.predict", "--exp", "S1-DIST", "--variant", "S1-DIST", "--L", "0", "--norm-long", "--min-history", "90", "--step", "7", "--train-blocks", "1", "--rounds", "250", "--blend", "dist"],
        ]
        model_dir = strongest_work / "models"
        started = time.perf_counter()
        for tail in commands:
            command = [str(args.strongest_python), *tail]
            if not args.skip_model_pickle:
                command.extend(["--model-dir", str(model_dir)])
            run(command, cwd=STRONGEST_CODE, env=env)
            remove_generated_feature_cache(processed)
        report["strongest_components"] = compare_vectors(artifacts)
        report["strongest_train_seconds"] = time.perf_counter() - started
        strongest_csv = strongest_work / "submission_STRONGEST_CURRENT_RETRAINED.csv"
        run(
            [
                str(args.strongest_python),
                str(ROOT / "scripts" / "build_strongest.py"),
                "--tabular-dir", str(artifacts),
                "--output", str(strongest_csv),
            ],
            cwd=ROOT,
        )
        report["strongest_submission"] = compare_csv(strongest_csv, FROZEN_STRONGEST)

    if args.branch in {"all", "team-b"}:
        team_work = work / "team_b"
        runtime = team_work / "runtime"
        runtime.mkdir(parents=True, exist_ok=True)
        team_csv = team_work / "final_classic_ml_RETRAINED.csv"
        command = [
            str(args.team_python),
            str(TEAM_B_CODE / "train_and_predict.py"),
            "--raw-data", str(raw_data),
            "--processed-dir", str(args.team_processed_dir.expanduser().resolve()),
            "--output", str(team_csv),
            "--report", str(team_work / "training_report.json"),
        ]
        if not args.skip_model_pickle:
            command.extend(["--model-output", str(team_work / "team_b_models.pkl")])
        started = time.perf_counter()
        run(command, cwd=runtime)
        report["team_b_train_seconds"] = time.perf_counter() - started
        report["team_b_submission"] = compare_csv(team_csv, FROZEN_TEAM_B)

    if args.branch == "all":
        final_csv = work / "SUBMIT_STRONGEST55_TEAMB45_RETRAINED.csv"
        run(
            [
                str(args.team_python),
                str(ROOT / "build_submit.py"),
                "--strongest", str(strongest_csv),
                "--team-b", str(team_csv),
                "--output", str(final_csv),
                "--allow-nonreference",
            ],
            cwd=ROOT,
        )
        report["final_submission"] = compare_csv(final_csv, REFERENCE_FINAL)

    report["total_seconds"] = time.time() - float(report["started_at_epoch"])
    report_path = work / "TABULAR_REPRO_AUDIT.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    print(f"audit: {report_path}", flush=True)


if __name__ == "__main__":
    main()
