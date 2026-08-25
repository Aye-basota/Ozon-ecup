"""Reproduce the frozen EXP-037 offline baseline without training."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.settings import competition
from src.utils.artifacts import load_oof
from src.validation.evaluate import evaluate_oof, expected_fold_names


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    baseline_cfg = competition()["offline_baseline"]
    path = ROOT / str(baseline_cfg["artifact"])
    frame = load_oof(path)
    metrics = evaluate_oof(frame["y_true"], frame["z_pred"], frame["cutoff"])
    expected = float(baseline_cfg["expected_wcv"])
    tolerance = float(baseline_cfg["tolerance"])

    if metrics["folds"] != expected_fold_names():
        raise AssertionError(f"Unexpected folds: {metrics['folds']}")
    if metrics["fold_sizes"] != [188_518, 191_025, 193_694, 197_379]:
        raise AssertionError(f"Unexpected fold sizes: {metrics['fold_sizes']}")
    if frame.duplicated(["cutoff", "user_id"]).any():
        raise AssertionError("Duplicate row keys")
    if not np.isfinite(frame[["y_true", "z_pred"]].to_numpy()).all():
        raise AssertionError("Non-finite baseline values")
    error = abs(float(metrics["wCV"]) - expected)
    if error > tolerance:
        raise AssertionError(f"wCV mismatch: observed={metrics['wCV']}, expected={expected}, error={error}")

    result = {
        "status": "PASS",
        "artifact": str(path),
        "sha256": sha256(path),
        "rows": len(frame),
        "unique_row_keys": int(frame[["cutoff", "user_id"]].drop_duplicates().shape[0]),
        "folds": metrics["folds"],
        "fold_sizes": metrics["fold_sizes"],
        "fold_cal": metrics["fold_cal"],
        "wCV": metrics["wCV"],
        "expected_wCV": expected,
        "absolute_error": error,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
