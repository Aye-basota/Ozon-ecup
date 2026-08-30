"""Create hashes for external raw data and caches used by the verified run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
DEFAULT_STRONGEST_DATA = (
    REPO_ROOT
    / "delivery"
    / "submission_STRONGEST_CURRENT_training_bundle_v2"
    / "pipeline"
    / "data"
)
DEFAULT_TEAM_PROCESSED = REPO_ROOT / "team-b" / "data" / "processed"
TEAM_CUTOFFS = [
    "2025-08-28", "2025-09-04", "2025-09-11", "2025-09-18",
    "2025-09-25", "2025-10-02", "2025-10-09", "2025-10-16", "2026-02-14",
]
TEAM_FEATURE_SETS = ["recency", "long_buy_post_order", "behavior_v1"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def describe(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strongest-data", type=Path, default=DEFAULT_STRONGEST_DATA)
    parser.add_argument("--team-processed", type=Path, default=DEFAULT_TEAM_PROCESSED)
    parser.add_argument("--output", type=Path, default=ROOT / "EXTERNAL_DATA_MANIFEST.json")
    args = parser.parse_args()

    strongest_data = args.strongest_data.expanduser().resolve()
    team_processed = args.team_processed.expanduser().resolve()
    result = {
        "raw": {
            "train.parquet": describe(strongest_data / "raw" / "train.parquet"),
            "sample_submit.csv": describe(strongest_data / "raw" / "sample_submit.csv"),
        },
        "strongest_dl_cache": {},
        "team_b_feature_cache": {},
    }
    for name in [
        "seq_panel_v1.npy", "seq_gmv_v1.npy", "seq_uid_v1.npy", "seq_scale_v1.json",
        "etx_ev_x_v1.npy", "etx_ev_day_v1.npy", "etx_ev_ptr_v1.npy",
    ]:
        result["strongest_dl_cache"][name] = describe(strongest_data / "processed" / name)
    for feature_set in TEAM_FEATURE_SETS:
        for cutoff in TEAM_CUTOFFS:
            name = f"features_{feature_set}_{cutoff}.parquet"
            result["team_b_feature_cache"][name] = describe(team_processed / name)

    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"created: {args.output}")
    print(f"files: {2 + len(result['strongest_dl_cache']) + len(result['team_b_feature_cache'])}")


if __name__ == "__main__":
    main()
