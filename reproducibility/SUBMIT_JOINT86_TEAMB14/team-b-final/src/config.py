"""Runtime paths and constants for the standalone team-b-final handoff."""

from pathlib import Path


SEED = 42
TARGET_DAYS = 30

ROOT = Path(__file__).resolve().parent.parent

# Portable bundle: raw data and regenerated feature caches stay inside team-b-final.
DATA_RAW = ROOT / "data"
DATA_PROCESSED = ROOT / "data" / "processed"
SUBMISSIONS = ROOT / "submissions"
