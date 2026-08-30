"""Create a Zip64 archive for the large STRONGEST_CURRENT training bundle."""

from __future__ import annotations

from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "delivery" / "submission_STRONGEST_CURRENT_training_bundle_v2"
OUTPUT = ROOT / "delivery" / "submission_STRONGEST_CURRENT_training_bundle_v2.zip"
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache"}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    files = sorted(
        path
        for path in SOURCE.rglob("*")
        if path.is_file() and not any(part in EXCLUDED_PARTS for part in path.parts)
    )
    with zipfile.ZipFile(
        OUTPUT,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        for index, path in enumerate(files, 1):
            arcname = (Path(SOURCE.name) / path.relative_to(SOURCE)).as_posix()
            archive.write(path, arcname)
            if index % 20 == 0 or index == len(files):
                print(f"packed {index}/{len(files)}: {path.relative_to(SOURCE)}", flush=True)
    print(f"archive: {OUTPUT}", flush=True)
    print(f"bytes: {OUTPUT.stat().st_size}", flush=True)


if __name__ == "__main__":
    main()
