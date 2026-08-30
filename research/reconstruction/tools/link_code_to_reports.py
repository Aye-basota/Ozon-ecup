from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


DEST = Path(sys.argv[1]).resolve()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    reports = read_csv(DEST / "registry" / "report_catalog.csv")
    links: list[dict] = []
    linked_paths: dict[str, set[str]] = defaultdict(set)
    pattern = re.compile(r"(?i)(?<![A-Za-z0-9_])((?:src|research|artifacts|submissions)/[A-Za-z0-9_./-]+\.(?:py|sh|mjs|ipynb))(?![A-Za-z0-9_])")
    for report in reports:
        evidence_path = DEST / report["clean_evidence_path"]
        if not evidence_path.exists():
            continue
        text = evidence_path.read_text(encoding="utf-8", errors="replace")
        for source_path in sorted(set(match.group(1).strip("`*.,;:()") for match in pattern.finditer(text))):
            linked_paths[source_path].add(report["experiment_id"])
            links.append({
                "experiment_id": report["experiment_id"],
                "namespace": report["namespace"],
                "source_ref": report["source_ref"],
                "code_path": source_path,
                "link_kind": "explicit_primary_report_reference",
                "source_report": report["source_path"],
            })
    write_csv(
        DEST / "code_index" / "report_code_links.csv",
        links,
        ["experiment_id", "namespace", "source_ref", "code_path", "link_kind", "source_report"],
    )

    scripts = read_csv(DEST / "code_index" / "scripts.csv")
    shared = {
        "src/config.py", "src/data.py", "src/features.py", "src/models.py", "src/validation.py",
        "src/train.py", "src/predict.py", "src/tracking.py", "src/report.py", "src/submit.py",
        "src/__init__.py",
    }
    orphans: list[dict] = []
    for script in scripts:
        path = script["original_path"]
        ids = sorted(linked_paths.get(path, set()))
        status = "linked_to_primary_report" if ids else ("shared_pipeline_dependency" if path in shared else "no_explicit_primary_report_link")
        orphans.append({
            "code_path": path,
            "type": script["type"],
            "sha256": script["sha256"],
            "linked_experiments": ";".join(ids) or "unknown",
            "link_status": status,
            "notes": "An absent explicit link does not prove the script was unused; it is an orphan candidate.",
        })
    write_csv(
        DEST / "code_index" / "script_linkage_audit.csv",
        orphans,
        ["code_path", "type", "sha256", "linked_experiments", "link_status", "notes"],
    )
    print(json.dumps({
        "explicit_report_code_links": len(links),
        "unique_linked_code_paths": len(linked_paths),
        "main_scripts": len(scripts),
        "orphan_candidates": sum(r["link_status"] == "no_explicit_primary_report_link" for r in orphans),
        "shared_dependencies": sum(r["link_status"] == "shared_pipeline_dependency" for r in orphans),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
