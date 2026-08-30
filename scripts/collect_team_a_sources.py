"""Copy the reviewable Team-A source subset selected by the forensic inventory.

Binary predictions, model weights, raw data, caches and byte-identical duplicate
files remain at their original paths and are represented by the inventory.  The
collector preserves relative paths and refuses to overwrite a different file.
"""

from __future__ import annotations

import csv
import hashlib
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "docs" / "TEAM_A_SOURCE_INVENTORY.csv"
TEXT_SUFFIXES = {
    ".csv",
    ".gitignore",
    ".json",
    ".md",
    ".mjs",
    ".ps1",
    ".py",
    ".rst",
    ".sha256",
    ".sh",
    ".toml",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}
MAX_COPY_BYTES = 2 * 1024 * 1024

EXPLICIT_COPIES = {
    Path(r"C:\Users\Admin\Desktop\submission_geometry_research\current_best\SUBMIT_v2_shrunk.csv"):
        ROOT / "research" / "submission_geometry" / "reference" / "SUBMIT_v2_shrunk.csv",
    Path(r"C:\Users\Admin\Desktop\submission_geometry_research\submission_geometry\SUBMIT_NEXT_BEST.csv"):
        ROOT / "research" / "submission_geometry" / "reference" / "SUBMIT_NEXT_BEST.csv",
    Path(r"C:\Users\Admin\Downloads\Telegram Desktop\TEAMMATE_REPRO_REQUEST.md"):
        ROOT / "research" / "provenance" / "downloads" / "TEAMMATE_REPRO_REQUEST.md",
    Path(r"C:\Users\Admin\Downloads\SUBMIT_ORTH_FINAL_reasoning.md"):
        ROOT / "research" / "provenance" / "downloads" / "SUBMIT_ORTH_FINAL_reasoning.md",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(source) == sha256(destination):
            return "same"
        raise FileExistsError(f"refusing to overwrite different file: {destination}")
    shutil.copy2(source, destination)
    return "copied"


def main() -> None:
    copied = same = skipped = 0
    with INVENTORY.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            source = Path(row["source_path"])
            destination = row["destination"]
            if source.is_relative_to(ROOT):
                skipped += 1
                continue
            if row["artifact_type"] == "DUPLICATE":
                skipped += 1
                continue
            if destination.startswith(("EXTERNAL_ONLY", "SOURCE_ONLY")):
                skipped += 1
                continue
            if source.suffix.lower() not in TEXT_SUFFIXES or source.stat().st_size > MAX_COPY_BYTES:
                skipped += 1
                continue
            result = safe_copy(source, ROOT / Path(destination))
            copied += result == "copied"
            same += result == "same"

    for source, destination in EXPLICIT_COPIES.items():
        if not source.exists():
            continue
        result = safe_copy(source, destination)
        copied += result == "copied"
        same += result == "same"

    print({"copied": copied, "already_identical": same, "skipped": skipped})


if __name__ == "__main__":
    main()
