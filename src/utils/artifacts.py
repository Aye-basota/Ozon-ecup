"""Read and write the standardized prediction artifact schemas."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.settings import ROOT
from src.validation.evaluate import assert_unique_row_keys


OOF_COLUMNS = ["cutoff", "user_id", "y_true", "z_pred"]
TEST_COLUMNS = ["user_id", "z_pred"]


def oof_path(experiment_id: str) -> Path:
    return ROOT / "artifacts" / "oof" / f"{experiment_id}.parquet"


def test_path(experiment_id: str) -> Path:
    return ROOT / "artifacts" / "test" / f"{experiment_id}.parquet"


def write_oof(experiment_id: str, cutoff, user_id, y_true, z_pred) -> Path:
    frame = pd.DataFrame(
        {
            "cutoff": np.asarray(cutoff, dtype="U10"),
            "user_id": np.asarray(user_id, dtype=np.int64),
            "y_true": np.asarray(y_true, dtype=np.float64),
            "z_pred": np.asarray(z_pred, dtype=np.float64),
        }
    )
    assert_unique_row_keys(frame)
    if not np.isfinite(frame[["y_true", "z_pred"]].to_numpy()).all():
        raise ValueError("OOF contains NaN or infinity")
    path = oof_path(experiment_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path, compression="zstd")
    return path


def write_test(experiment_id: str, user_id, z_pred) -> Path:
    frame = pd.DataFrame(
        {
            "user_id": np.asarray(user_id, dtype=np.int64),
            "z_pred": np.asarray(z_pred, dtype=np.float64),
        }
    )
    if frame["user_id"].duplicated().any():
        raise ValueError("TEST artifact contains duplicate user_id")
    if not np.isfinite(frame["z_pred"]).all():
        raise ValueError("TEST artifact contains NaN or infinity")
    path = test_path(experiment_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path, compression="zstd")
    return path


def load_oof(path_or_experiment: str | Path) -> pd.DataFrame:
    path = Path(path_or_experiment)
    if not path.exists():
        path = oof_path(str(path_or_experiment))
    frame = pd.read_parquet(path)
    missing = set(OOF_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"OOF artifact {path} lacks columns: {sorted(missing)}")
    assert_unique_row_keys(frame)
    return frame[OOF_COLUMNS]


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
