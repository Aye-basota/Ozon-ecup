"""Проверка SHA256 всех файлов, перечисленных в MANIFEST.sha256."""
from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST.sha256"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    checked = 0
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, rel = line.split("  ", 1)
        path = ROOT / Path(rel)
        assert path.is_file(), f"Нет файла: {rel}"
        actual = sha256(path)
        assert actual == expected, f"SHA256 не совпал: {rel}"
        checked += 1
    print(f"OK: проверено файлов: {checked}")


if __name__ == "__main__":
    main()
