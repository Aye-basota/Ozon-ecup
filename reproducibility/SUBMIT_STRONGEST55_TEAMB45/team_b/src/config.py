"""Runtime paths and constants for the standalone team-b-final handoff."""

import os
from pathlib import Path


SEED = 42
TARGET_DAYS = 30

ROOT = Path(__file__).resolve().parent.parent


def _path_from_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


# Only runtime paths are parameterized. Seeds, features, estimators and blend
# constants are unchanged from the historical team-b-final handoff.
DATA_RAW = _path_from_env("ECUP_RAW_DATA_DIR", ROOT / "data" / "raw")
DATA_PROCESSED = _path_from_env("ECUP_TEAM_PROCESSED_DIR", ROOT / "data" / "processed")
SUBMISSIONS = _path_from_env("ECUP_TEAM_SUBMISSIONS_DIR", ROOT / "submissions")
RAW_PARQUET = DATA_RAW / "train.parquet"
SAMPLE_SUBMIT = DATA_RAW / "sample_submit.csv"
