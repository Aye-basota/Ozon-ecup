from __future__ import annotations

import hashlib
import json
from pathlib import Path


EXP = Path(__file__).resolve().parent
ROOT = Path(r"C:\Users\Admin\Desktop\e-cup-research-clean")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    excluded = {"artifact_manifest_all.json", "checksums_final.sha256"}
    files = [p for p in EXP.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.name not in excluded]
    files += sorted((ROOT / "submissions").glob("SUBMIT_EXP075*.csv"))
    manifest = []
    lines = []
    for path in sorted(files, key=lambda x: str(x).lower()):
        sha = digest(path)
        try:
            label = str(path.relative_to(ROOT)).replace("\\", "/")
        except ValueError:
            label = str(path)
        manifest.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha})
        lines.append(f"{sha}  {label}")
    (EXP / "artifact_manifest_all.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (EXP / "checksums_final.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"files": len(manifest), "bytes": sum(x["bytes"] for x in manifest)}, indent=2))


if __name__ == "__main__":
    main()
