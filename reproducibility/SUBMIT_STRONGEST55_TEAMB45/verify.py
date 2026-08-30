"""Verify package files, frozen sources and optional external training data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
DEFAULT_RAW = (
    REPO_ROOT
    / "delivery"
    / "submission_STRONGEST_CURRENT_training_bundle_v2"
    / "pipeline"
    / "data"
    / "raw"
    / "train.parquet"
)
DEFAULT_STRONGEST_DATA = DEFAULT_RAW.parent.parent
DEFAULT_TEAM_PROCESSED = REPO_ROOT / "team-b" / "data" / "processed"
EXPECTED = {
    "raw": "5f3aa90992652b8a4f0f398e735a3ba11c2ea6ccf9e8fb1d236436e9a49167c0",
    "sample": "06a433b0ac32f7c0292ce3cb994c1684b4156b392f30fe537ea6a44d0bc4c1b1",
    "strongest": "abc2218b1a3d55d41121b7b5a22db7e95ffd45283b42ff0006c5e6e731e04bda",
    "final": "1ce85203e3069363e3d2ba425078213d1a723a895e3c684573a6c1b998a14fb4",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_csv(path: Path) -> dict:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["user_id", "predict"]:
        raise AssertionError(f"wrong columns: {path}")
    if len(frame) != 250_000 or frame.user_id.nunique() != 250_000:
        raise AssertionError(f"wrong user count: {path}")
    values = frame.predict.to_numpy(np.float64)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise AssertionError(f"invalid predictions: {path}")
    return {"rows": len(frame), "sha256": sha256(path), "mean_log1p": float(np.log1p(values).mean())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--skip-raw", action="store_true")
    parser.add_argument("--full-data", action="store_true", help="also hash all external caches")
    parser.add_argument("--strongest-data", type=Path, default=DEFAULT_STRONGEST_DATA)
    parser.add_argument("--team-processed", type=Path, default=DEFAULT_TEAM_PROCESSED)
    args = parser.parse_args()

    manifest = ROOT / "MANIFEST.sha256"
    checked = 0
    if manifest.is_file():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            expected, relative = line.split("  ", 1)
            path = ROOT / Path(relative)
            if not path.is_file() or sha256(path) != expected:
                raise AssertionError(f"manifest mismatch: {relative}")
            checked += 1

    paths = {
        "sample": ROOT / "reference" / "sample_submit.csv",
        "strongest": ROOT / "frozen" / "strongest" / "submission_STRONGEST_CURRENT.csv",
        "team_b": ROOT / "frozen" / "team_b" / "final_classic_ml.csv",
        "final": ROOT / "reference" / "SUBMIT_STRONGEST55_TEAMB45.csv",
    }
    result = {"manifest_files_checked": checked, "files": {}}
    for name, path in paths.items():
        item = check_csv(path)
        if name in EXPECTED and item["sha256"] != EXPECTED[name]:
            raise AssertionError(f"{name} SHA256 mismatch")
        result["files"][name] = item

    sample_ids = pd.read_csv(paths["sample"], usecols=["user_id"]).user_id.to_numpy(np.int64)
    for name in ["strongest", "team_b", "final"]:
        ids = pd.read_csv(paths[name], usecols=["user_id"]).user_id.to_numpy(np.int64)
        if not np.array_equal(ids, sample_ids):
            raise AssertionError(f"row order mismatch: {name}")

    if not args.skip_raw:
        raw = args.raw_data.expanduser().resolve()
        if not raw.is_file():
            raise FileNotFoundError(raw)
        digest = sha256(raw)
        if digest != EXPECTED["raw"]:
            raise AssertionError(f"raw SHA256 mismatch: {digest}")
        result["raw_data"] = {"path": str(raw), "sha256": digest}

    if args.full_data:
        external = json.loads((ROOT / "EXTERNAL_DATA_MANIFEST.json").read_text(encoding="utf-8"))
        checked_external = 0
        for section, base in [
            ("raw", args.strongest_data.expanduser().resolve() / "raw"),
            ("strongest_dl_cache", args.strongest_data.expanduser().resolve() / "processed"),
            ("team_b_feature_cache", args.team_processed.expanduser().resolve()),
        ]:
            for name, expected in external[section].items():
                path = base / name
                if not path.is_file() or path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
                    raise AssertionError(f"external data mismatch: {section}/{name}")
                checked_external += 1
        result["external_cache_files_checked"] = checked_external

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
