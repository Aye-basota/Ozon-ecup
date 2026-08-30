from __future__ import annotations

import csv
import hashlib
import subprocess
import sys
from pathlib import Path


SOURCE = Path(sys.argv[1]).resolve()
DEST = Path(sys.argv[2]).resolve()

SOURCES = {
    "independent_domain": ("codex/domain-01", "research/domain_01/results/", "independent_domain:exp_028"),
    "independent_calendar": ("codex/calendar-placebo-01", "research/calendar_placebo_01/results/", "independent_calendar:exp_029"),
}


def git(*args: str) -> bytes:
    result = subprocess.run(["git", *args], cwd=SOURCE, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
    return result.stdout


def main() -> None:
    rows: list[dict] = []
    for namespace, (ref_name, prefix, experiment_id) in SOURCES.items():
        paths = git("ls-tree", "-r", "--name-only", ref_name, "--", prefix.rstrip("/")).decode("utf-8", errors="replace").splitlines()
        for source_path in sorted(paths):
            if Path(source_path).suffix.lower() not in {".csv", ".json", ".npz", ".npy", ".log"}:
                continue
            raw = git("show", f"{ref_name}:{source_path}")
            output = DEST / "evidence" / "git_machine_artifacts" / namespace / source_path
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(raw)
            rows.append({
                "namespace": namespace,
                "experiment_id": experiment_id,
                "source_ref": ref_name,
                "source_path": source_path,
                "clean_path": output.relative_to(DEST).as_posix(),
                "size_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "type": Path(source_path).suffix.lower(),
            })
    fields = ["namespace", "experiment_id", "source_ref", "source_path", "clean_path", "size_bytes", "sha256", "type"]
    output = DEST / "evidence" / "git_machine_artifacts_manifest.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"exported={len(rows)} bytes={sum(int(r['size_bytes']) for r in rows)}")


if __name__ == "__main__":
    main()
