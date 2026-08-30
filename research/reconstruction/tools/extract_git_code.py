from __future__ import annotations

import csv
import hashlib
import re
import subprocess
import sys
from pathlib import Path


SOURCE = Path(sys.argv[1]).resolve()
DEST = Path(sys.argv[2]).resolve()

NAMESPACE_REFS = {
    "team_a_s2": "team-a-strategy-2-impl",
    "team_b_core": "origin/team-b-B2",
    "team_b_alt": "origin/team-b-strategy-1-impl",
    "independent_renewal": "codex/renewal-01",
    "independent_domain": "codex/domain-01",
    "independent_calendar": "codex/calendar-placebo-01",
    "independent_global_regime": "exp/057-global-regime-occ",
}


def git(*args: str, check: bool = True) -> bytes:
    result = subprocess.run(["git", *args], cwd=SOURCE, capture_output=True, check=False)
    if check and result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
    return result.stdout


def main() -> None:
    rows: list[dict] = []
    for namespace, ref_name in NAMESPACE_REFS.items():
        report_paths = git("ls-tree", "-r", "--name-only", ref_name, "--", "experiments").decode("utf-8", errors="replace").splitlines()
        references: dict[str, set[str]] = {}
        for report_path in report_paths:
            if not re.match(r"(?i)^experiments/exp[_-]?\d+.*\.md$", report_path):
                continue
            text = git("show", f"{ref_name}:{report_path}", check=False).decode("utf-8", errors="replace")
            for code_path in re.findall(r"(?i)(?:`|\b)((?:src|research)/[A-Za-z0-9_./-]+\.(?:py|sh|mjs))(?:`|\b)", text):
                references.setdefault(code_path, set()).add(report_path)
        # Config and validation establish seeds/protocols and are indexed even when reports omit an explicit link.
        for required in ("src/config.py", "src/validation.py"):
            references.setdefault(required, set()).add("protocol_dependency")
        for code_path, reports in sorted(references.items()):
            raw = git("show", f"{ref_name}:{code_path}", check=False)
            if not raw:
                rows.append({
                    "namespace": namespace, "source_ref": ref_name, "source_path": code_path,
                    "sha256": "missing", "size_bytes": 0, "referenced_by": ";".join(sorted(reports)),
                    "clean_snapshot_path": "missing", "status": "referenced_but_absent_at_ref",
                })
                continue
            sha = hashlib.sha256(raw).hexdigest()
            output = DEST / "code_index" / "snapshots" / namespace / code_path
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(raw)
            rows.append({
                "namespace": namespace, "source_ref": ref_name, "source_path": code_path,
                "sha256": sha, "size_bytes": len(raw), "referenced_by": ";".join(sorted(reports)),
                "clean_snapshot_path": output.relative_to(DEST).as_posix(), "status": "snapshotted",
            })
    output = DEST / "code_index" / "git_referenced_code.csv"
    fields = ["namespace", "source_ref", "source_path", "sha256", "size_bytes", "referenced_by", "clean_snapshot_path", "status"]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"indexed={len(rows)} snapshotted={sum(r['status'] == 'snapshotted' for r in rows)} missing={sum(r['status'] != 'snapshotted' for r in rows)}")


if __name__ == "__main__":
    main()
