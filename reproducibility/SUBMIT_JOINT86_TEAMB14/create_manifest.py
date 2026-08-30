"""Regenerate MANIFEST.sha256 for committed package files."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "MANIFEST.sha256"
EXCLUDED = {"outputs", "work", "__pycache__", ".pytest_cache", "catboost_info"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    files = sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path != OUTPUT
        and not any(part in EXCLUDED for part in path.relative_to(ROOT).parts)
        and path.suffix not in {".pyc", ".log"}
    )
    OUTPUT.write_text(
        "".join(f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}\n" for path in files),
        encoding="utf-8",
        newline="\n",
    )
    print(f"created {OUTPUT} with {len(files)} entries")


if __name__ == "__main__":
    main()
