#!/usr/bin/env python3
"""Re-hash source/worktree content and prove the forensic pass made no edits."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


SKIP_PARTS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def enumerate_main(root: Path) -> set[str]:
    result: set[str] = set()
    for directory, dirs, files in os.walk(root):
        dirs[:] = [name for name in dirs if name not in SKIP_PARTS]
        base = Path(directory)
        for name in files:
            path = base / name
            rel = path.relative_to(root).as_posix()
            if any(part in SKIP_PARTS for part in Path(rel).parts):
                continue
            result.add(rel)
    return result


def enumerate_linked(root: Path) -> set[str]:
    result: set[str] = set()
    for directory, dirs, files in os.walk(root, followlinks=False):
        base = Path(directory)
        rel_dir = base.relative_to(root)
        if not rel_dir.parts:
            dirs[:] = [name for name in dirs if name != "data" and name not in SKIP_PARTS]
        else:
            dirs[:] = [name for name in dirs if name not in SKIP_PARTS]
        for name in files:
            rel = (rel_dir / name).as_posix()
            if rel == ".git" or any(part in SKIP_PARTS for part in Path(rel).parts):
                continue
            result.add(rel)
    return result


def audit_expected(root: Path, expected_rows: list[dict[str, str]], current_paths: set[str]) -> dict[str, Any]:
    expected = {row["relative_path"].replace("\\", "/"): row for row in expected_rows}
    missing: list[str] = []
    size_changed: list[dict[str, Any]] = []
    hash_changed: list[dict[str, Any]] = []
    for rel, row in expected.items():
        path = root.joinpath(*rel.split("/"))
        if not path.is_file():
            missing.append(rel)
            continue
        actual_size = path.stat().st_size
        expected_size = int(row["size_bytes"])
        if actual_size != expected_size:
            size_changed.append({"path": rel, "expected": expected_size, "actual": actual_size})
            continue
        actual_hash = sha256(path)
        if actual_hash != row["sha256"]:
            hash_changed.append({"path": rel, "expected": row["sha256"], "actual": actual_hash})
    unexpected = sorted(current_paths - set(expected))
    return {
        "expected_files": len(expected),
        "current_files": len(current_paths),
        "missing": missing,
        "unexpected": unexpected,
        "size_changed": size_changed,
        "hash_changed": hash_changed,
        "content_unchanged": not (missing or unexpected or size_changed or hash_changed),
    }


def git_status(root: Path) -> list[str]:
    process = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return sorted(line.rstrip() for line in process.stdout.splitlines() if line.rstrip())


def main() -> None:
    clean = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
    source = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else Path(r"C:\Users\Admin\Desktop\OZON-E-CUP")
    main_expected = load_csv(clean / "inventory" / "source_checksums_sha256.csv")
    main_result = audit_expected(source, main_expected, enumerate_main(source))

    before_status = sorted(line.rstrip() for line in (clean / "inventory" / "git_status_before.txt").read_text(encoding="utf-8-sig").splitlines() if line.rstrip())
    after_status = git_status(source)
    main_result["git_status_before_lines"] = len(before_status)
    main_result["git_status_after_lines"] = len(after_status)
    main_result["git_status_unchanged"] = before_status == after_status
    main_result["git_status_added_lines"] = sorted(set(after_status) - set(before_status))
    main_result["git_status_removed_lines"] = sorted(set(before_status) - set(after_status))

    worktree_expected = load_csv(clean / "inventory" / "worktree_files.csv")
    worktree_meta = {row["namespace"]: row for row in load_csv(clean / "inventory" / "worktrees.csv")}
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in worktree_expected:
        grouped.setdefault(row["namespace"], []).append(row)
    worktree_results: dict[str, Any] = {}
    for namespace, rows in grouped.items():
        root = Path(rows[0]["worktree_root"])
        result = audit_expected(root, rows, enumerate_linked(root))
        actual_status = git_status(root)
        before_raw = worktree_meta[namespace].get("git_status_before", "clean")
        before = [] if before_raw == "clean" else sorted(item.strip() for item in before_raw.split("|") if item.strip())
        result["git_status_before"] = before
        result["git_status_after"] = actual_status
        result["git_status_unchanged"] = before == actual_status
        worktree_results[namespace] = result

    summary = {
        "source_root": str(source),
        "main": main_result,
        "linked_worktrees": worktree_results,
        "all_content_unchanged": main_result["content_unchanged"] and all(item["content_unchanged"] for item in worktree_results.values()),
        "all_git_statuses_unchanged": main_result["git_status_unchanged"] and all(item["git_status_unchanged"] for item in worktree_results.values()),
    }
    summary["source_untouched_verified"] = summary["all_content_unchanged"] and summary["all_git_statuses_unchanged"]
    out = clean / "reports" / "source_unchanged_verification.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "main_expected_files": main_result["expected_files"],
        "main_content_unchanged": main_result["content_unchanged"],
        "main_git_status_unchanged": main_result["git_status_unchanged"],
        "linked_worktrees": {key: {"files": value["expected_files"], "content_unchanged": value["content_unchanged"], "git_status_unchanged": value["git_status_unchanged"]} for key, value in worktree_results.items()},
        "source_untouched_verified": summary["source_untouched_verified"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
