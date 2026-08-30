"""Create the immutable artifact inventory for EXP080."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
EXCLUDE = {"artifact_manifest.csv", "checksums.sha256", "__pycache__"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def role(path: Path) -> str:
    if path.name == "REPORT.md":
        return "final_report"
    if path.suffix == ".py":
        return "reproduction_code"
    if "audit" in path.name or path.name == "config.json":
        return "audit_or_config"
    if path.suffix in {".csv", ".json"}:
        return "metric_artifact"
    if path.suffix in {".npz", ".parquet"}:
        return "working_prediction_artifact"
    return "other"


def main() -> None:
    files = [p for p in sorted(HERE.iterdir()) if p.is_file() and p.name not in EXCLUDE]
    rows = [{"file": p.name, "bytes": p.stat().st_size, "sha256": sha256(p), "role": role(p)}
            for p in files]
    pd.DataFrame(rows).to_csv(HERE / "artifact_manifest.csv", index=False, lineterminator="\n")
    lines = [f"{row['sha256']}  {row['file']}" for row in rows]
    (HERE / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
