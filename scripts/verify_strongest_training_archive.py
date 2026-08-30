"""Verify the extended manifest directly inside the Zip64 delivery archive."""

from __future__ import annotations

import hashlib
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "delivery" / "submission_STRONGEST_CURRENT_training_bundle_v2.zip"
PREFIX = "submission_STRONGEST_CURRENT_training_bundle_v2/"
EXPECTED_SUBMISSION_SHA256 = "abc2218b1a3d55d41121b7b5a22db7e95ffd45283b42ff0006c5e6e731e04bda"


def stream_sha256(handle) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(8 << 20), b""):
        digest.update(block)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return stream_sha256(handle)


def main() -> None:
    with zipfile.ZipFile(ARCHIVE, "r", allowZip64=True) as archive:
        bad_crc = archive.testzip()
        if bad_crc is not None:
            raise AssertionError(f"bad CRC: {bad_crc}")
        names = set(archive.namelist())
        manifest_name = PREFIX + "EXTENDED_MANIFEST.sha256"
        manifest = archive.read(manifest_name).decode("utf-8")
        checked = 0
        for line in manifest.splitlines():
            expected, relative = line.split("  ", 1)
            entry_name = PREFIX + relative
            if entry_name not in names:
                raise AssertionError(f"missing ZIP entry: {entry_name}")
            with archive.open(entry_name) as handle:
                actual = stream_sha256(handle)
            if actual != expected:
                raise AssertionError(f"SHA256 mismatch in ZIP: {relative}")
            checked += 1

        submission_name = PREFIX + "submission/submission_STRONGEST_CURRENT.csv"
        with archive.open(submission_name) as handle:
            submission_hash = stream_sha256(handle)
        if submission_hash != EXPECTED_SUBMISSION_SHA256:
            raise AssertionError("embedded submission hash mismatch")

        critical = {
            "train.parquet": PREFIX + "pipeline/data/raw/train.parquet",
            "seq_panel_v1.npy": PREFIX + "pipeline/data/processed/seq_panel_v1.npy",
            "etx_ev_x_v1.npy": PREFIX + "pipeline/data/processed/etx_ev_x_v1.npy",
            "README": PREFIX + "START_HERE.md",
            "DL audit": PREFIX + "DL_REPRO_AUDIT_RUNTIME.json",
        }
        critical_sizes = {label: archive.getinfo(name).file_size for label, name in critical.items()}
        uncompressed = sum(info.file_size for info in archive.infolist())

    print(
        {
            "archive_check": "PASS",
            "entries": len(names),
            "manifest_files_checked": checked,
            "uncompressed_bytes": uncompressed,
            "archive_bytes": ARCHIVE.stat().st_size,
            "archive_sha256": file_sha256(ARCHIVE),
            "embedded_submission_sha256": submission_hash,
            "critical_sizes": critical_sizes,
        }
    )


if __name__ == "__main__":
    main()
