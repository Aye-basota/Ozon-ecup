from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
COMPONENTS = EXP / "production_components"
MANIFEST = EXP / "artifact_manifest.csv"
SUMS = EXP / "SHA256SUMS.txt"
RUNTIME = EXP / "runtime_summary.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def included_files() -> list[Path]:
    excluded = {SUMS.resolve()}
    return sorted(
        path for path in EXP.rglob("*")
        if path.is_file()
        and path.resolve() not in excluded
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    )


def main() -> None:
    component_rows = []
    for path in sorted(COMPONENTS.glob("*.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        component_rows.append({
            "family": meta.get("family"),
            "cutoff": meta.get("cutoff"),
            "runtime_seconds": meta.get("runtime_seconds", 0.0),
            "mode": meta.get("mode"),
            "config_changed": meta.get("config_changed"),
            "metadata_file": path.relative_to(EXP).as_posix(),
        })
    runtime_payload = {
        "component_runs": component_rows,
        "component_runtime_seconds_sum": float(sum(row["runtime_seconds"] for row in component_rows)),
        "note": "Canonical reused components retain their recorded replay/copy runtime; wall clock is not inferred from this sum.",
    }
    RUNTIME.write_text(json.dumps(runtime_payload, indent=2), encoding="utf-8")

    rows = []
    for path in included_files():
        if path.resolve() == MANIFEST.resolve():
            continue
        rows.append({
            "path": path.relative_to(EXP).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    with MANIFEST.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)

    sum_rows = []
    for path in included_files():
        sum_rows.append(f"{sha256(path)}  {path.relative_to(EXP).as_posix()}")
    SUMS.write_text("\n".join(sum_rows) + "\n", encoding="utf-8")
    print(f"finalized {len(rows)} manifest artifacts; SHA256SUMS entries={len(sum_rows)}")


if __name__ == "__main__":
    main()
