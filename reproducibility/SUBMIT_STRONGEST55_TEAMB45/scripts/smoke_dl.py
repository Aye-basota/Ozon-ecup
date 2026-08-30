"""Bounded checkpoint/unit smoke test for SEQ and ETX (no full training)."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
CODE = ROOT / "strongest"
FROZEN = ROOT / "frozen" / "strongest" / "predictions"
DEFAULT_RAW_DIR = (
    REPO_ROOT
    / "delivery"
    / "submission_STRONGEST_CURRENT_training_bundle_v2"
    / "pipeline"
    / "data"
    / "raw"
)
DEFAULT_PROCESSED = (
    REPO_ROOT
    / "delivery"
    / "submission_STRONGEST_CURRENT_training_bundle_v2"
    / "pipeline"
    / "data"
    / "processed"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--max-gpu-minutes", type=float, default=30.0)
    parser.add_argument("--skip-unit-tests", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "work" / "dl" / "DL_SMOKE_AUDIT.json")
    args = parser.parse_args()
    if not 1 <= args.rows <= 4096:
        raise ValueError("--rows must be in 1..4096")

    raw_dir = args.raw_dir.expanduser().resolve()
    processed_dir = args.processed_dir.expanduser().resolve()
    env = os.environ.copy()
    env.update({
        "ECUP_RAW_DATA_DIR": str(raw_dir),
        "ECUP_PROCESSED_DIR": str(processed_dir),
        "ECUP_ARTIFACTS_DIR": str((CODE / "artifacts").resolve()),
        "PYTHONIOENCODING": "utf-8",
    })
    required = [
        raw_dir / "train.parquet",
        processed_dir / "seq_panel_v1.npy",
        processed_dir / "seq_gmv_v1.npy",
        processed_dir / "seq_uid_v1.npy",
        processed_dir / "etx_ev_x_v1.npy",
        processed_dir / "etx_ev_day_v1.npy",
        processed_dir / "etx_ev_ptr_v1.npy",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing smoke inputs:\n" + "\n".join(missing))

    unit_test_seconds = 0.0
    if not args.skip_unit_tests:
        started = time.perf_counter()
        subprocess.run(
            [sys.executable, "-m", "pytest", "src/test_etx.py", "src/test_seq.py", "-q"],
            cwd=CODE,
            env=env,
            check=True,
        )
        unit_test_seconds = time.perf_counter() - started

    os.environ.update({key: value for key, value in env.items() if key.startswith("ECUP_")})
    sys.path.insert(0, str(CODE))
    import torch
    from src import etx, seq
    from src.config import CUTOFF_TEST
    from src.features import panel_users

    user_id = panel_users(CUTOFF_TEST, 3)["user_id"].to_numpy()
    rows = seq.user_rows(user_id)
    positions = np.unique(np.linspace(0, len(user_id) - 1, args.rows, dtype=np.int64))
    sample_rows = rows[positions]
    results: dict[str, object] = {}
    inference_started = time.perf_counter()

    for checkpoint in ["ETX-01-S42-TEST", "ETX-01-S43-TEST", "ETX-01-S44-TEST"]:
        production_name = checkpoint.replace("-TEST", "-DCW")
        frozen_uid = np.load(FROZEN / f"uid_{production_name}.npy")
        frozen_z = np.load(FROZEN / f"ztest_{production_name}.npy")
        if not np.array_equal(frozen_uid, user_id):
            raise AssertionError(f"user_id mismatch: {production_name}")
        model, tokenizer, cfg, val, device = etx.load_ckpt(checkpoint)
        tokenizer.depth_cap = 289
        tokenizer.cdow_shift = -1.0
        try:
            actual = np.maximum(
                etx.predict(model, tokenizer, CUTOFF_TEST, sample_rows, cfg, device, depth_clip=289),
                0.0,
            )
        finally:
            tokenizer.depth_cap = None
            tokenizer.cdow_shift = 0.0
        expected = frozen_z[positions]
        diff = actual.astype(np.float64) - expected.astype(np.float64)
        results[production_name] = {
            "checkpoint_sha256": sha256(CODE / "artifacts" / f"model_{checkpoint}.pt"),
            "rows_checked": len(positions),
            "checkpoint_val": val.isoformat(),
            "device": str(device),
            "max_abs_log_error": float(np.max(np.abs(diff))),
            "rms_log_error": float(np.sqrt(np.mean(diff**2))),
            "allclose_atol_5e-2": bool(np.allclose(actual, expected, rtol=0.0, atol=5e-2)),
        }
        if not np.isfinite(actual).all() or not results[production_name]["allclose_atol_5e-2"]:
            raise AssertionError(f"ETX smoke failed: {production_name}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for checkpoint in ["SEQ-C289-S43-TEST", "SEQ-C289-S44-TEST"]:
        production_name = checkpoint.replace("-TEST", "")
        frozen_uid = np.load(FROZEN / f"uid_{production_name}.npy")
        frozen_z = np.load(FROZEN / f"ztest_{production_name}.npy")
        if not np.array_equal(frozen_uid, user_id):
            raise AssertionError(f"user_id mismatch: {production_name}")
        model, cfg, val, device = seq.load_ckpt(checkpoint)
        actual = np.maximum(seq.predict(model, CUTOFF_TEST, sample_rows, cfg, device, depth_clip=289), 0.0)
        expected = frozen_z[positions]
        diff = actual.astype(np.float64) - expected.astype(np.float64)
        results[production_name] = {
            "checkpoint_sha256": sha256(CODE / "artifacts" / f"model_{checkpoint}.pt"),
            "rows_checked": len(positions),
            "checkpoint_val": val.isoformat(),
            "device": str(device),
            "max_abs_log_error": float(np.max(np.abs(diff))),
            "rms_log_error": float(np.sqrt(np.mean(diff**2))),
            "bitwise_equal": bool(np.array_equal(actual, expected)),
        }
        if not np.isfinite(actual).all():
            raise AssertionError(f"SEQ smoke failed: {production_name}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    inference_seconds = time.perf_counter() - inference_started
    total_seconds = unit_test_seconds + inference_seconds
    if total_seconds > args.max_gpu_minutes * 60:
        raise AssertionError(
            f"smoke wall time {total_seconds / 60:.2f} min exceeded budget {args.max_gpu_minutes:.2f} min"
        )
    results["SEQ-01"] = {
        "checkpoint_available": False,
        "frozen_prediction_available": (FROZEN / "ztest_SEQ-01.npy").is_file(),
        "note": "Historical seed-42 checkpoint was not saved; full retraining code is included.",
    }
    report = {
        "status": "PASS",
        "scope": "unit tests plus bounded checkpoint inference; no full SEQ/ETX training",
        "rows_per_checkpoint": len(positions),
        "unit_test_seconds": unit_test_seconds,
        "checkpoint_inference_seconds": inference_seconds,
        "total_wall_seconds_charged_to_budget": total_seconds,
        "budget_minutes": args.max_gpu_minutes,
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "polars": importlib.metadata.version("polars"),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "checks": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
