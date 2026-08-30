from __future__ import annotations

import csv
import json
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
    main_files = read_csv(DEST / "inventory" / "files.csv")
    wt_files = read_csv(DEST / "inventory" / "worktree_files.csv")
    main_sha = {row["sha256"] for row in main_files if row["sha256"] != "unknown"}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in main_files:
        if row["sha256"] != "unknown":
            grouped[row["sha256"]].append({
                "root": "main_worktree", "namespace": "team_a_current", "path": row["original_path"],
                "size_bytes": row["size_bytes"], "type": row["type"],
            })
    for row in wt_files:
        grouped[row["sha256"]].append({
            "root": row["worktree_root"], "namespace": row["namespace"], "path": row["relative_path"],
            "size_bytes": row["size_bytes"], "type": row["type"],
        })
    duplicates: list[dict] = []
    for sha, group in grouped.items():
        namespaces = sorted({row["namespace"] for row in group})
        if len(namespaces) < 2:
            continue
        canonical = sorted(group, key=lambda r: (r["namespace"] != "team_a_current", len(r["path"]), r["path"]))[0]
        for row in group:
            duplicates.append({
                "sha256": sha,
                "copies": len(group),
                "namespaces": ";".join(namespaces),
                "canonical_namespace": canonical["namespace"],
                "canonical_path": canonical["path"],
                **row,
                "duplicate_of": "" if row is canonical else f"{canonical['namespace']}:{canonical['path']}",
            })
    unique_rows = [
        {
            "namespace": row["namespace"], "worktree_root": row["worktree_root"], "branch": row["branch"],
            "relative_path": row["relative_path"], "size_bytes": row["size_bytes"], "sha256": row["sha256"],
            "type": row["type"], "schema_or_arrays": row["schema_or_arrays"],
        }
        for row in wt_files
        if row["sha256"] not in main_sha
    ]
    write_csv(
        DEST / "artifacts" / "cross_worktree_duplicates.csv",
        duplicates,
        ["sha256", "copies", "namespaces", "canonical_namespace", "canonical_path", "root", "namespace", "path", "size_bytes", "type", "duplicate_of"],
    )
    write_csv(
        DEST / "artifacts" / "worktree_unique_files.csv",
        unique_rows,
        ["namespace", "worktree_root", "branch", "relative_path", "size_bytes", "sha256", "type", "schema_or_arrays"],
    )
    summary = {
        "linked_worktree_files": len(wt_files),
        "linked_worktree_files_unique_vs_main": len(unique_rows),
        "cross_namespace_duplicate_sha_groups": len({row["sha256"] for row in duplicates}),
        "cross_namespace_duplicate_occurrences": len(duplicates),
        "unique_artifact_or_prediction_files": sum("artifact" in row["type"] or "prediction" in row["type"] for row in unique_rows),
    }
    (DEST / "inventory" / "cross_worktree_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
