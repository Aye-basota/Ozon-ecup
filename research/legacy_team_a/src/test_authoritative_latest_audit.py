from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.authoritative_latest_audit import blend_log_predictions, read_submission


def write_submit(path: Path, user_id: list[int], predict: list[float]) -> None:
    pd.DataFrame({"user_id": user_id, "predict": predict}).to_csv(path, index=False)


def test_blend_is_log_space_and_applies_zero_floor() -> None:
    parts = {
        "friend": np.asarray([1.0, -2.0]),
        "occ_meta_B": np.asarray([2.0, -1.0]),
        "occ_raw_X3": np.asarray([3.0, -3.0]),
    }
    before, after = blend_log_predictions(parts)
    np.testing.assert_allclose(before, [2.6, -2.56])
    np.testing.assert_allclose(after, [2.6, 0.0])


def test_read_submission_rejects_duplicate_user_ids(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.csv"
    write_submit(path, [1, 1], [0.0, 1.0])
    with pytest.raises(ValueError, match="duplicate user_id"):
        read_submission(path, expected_rows=2)


def test_read_submission_rejects_negative_predictions(tmp_path: Path) -> None:
    path = tmp_path / "negative.csv"
    write_submit(path, [1, 2], [0.0, -0.1])
    with pytest.raises(ValueError, match="negative prediction"):
        read_submission(path, expected_rows=2)


def test_read_submission_requires_exact_schema(tmp_path: Path) -> None:
    path = tmp_path / "schema.csv"
    pd.DataFrame({"predict": [0.0], "user_id": [1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="columns"):
        read_submission(path, expected_rows=1)
