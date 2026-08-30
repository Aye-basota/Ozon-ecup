from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


MAIN = Path(sys.argv[1]).resolve()
DEST = Path(sys.argv[2]).resolve()

NAMESPACE_BY_DIR = {
    "exp058-exact-anniversary": "independent_anniversary",
    "OZON-E-CUP-calendar-placebo-01": "independent_calendar",
    "OZON-E-CUP-domain-01": "independent_domain",
    "OZON-E-CUP-exp057-global-regime-occ": "independent_global_regime",
    "OZON-E-CUP-renewal-01": "independent_renewal",
    "OZON-E-CUP-s2": "team_a_s2",
}


def run_git(cwd: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, check=False)
    return result.stdout.decode("utf-8", errors="replace")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def np_header(stream):
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
    else:
        shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
    return shape, str(dtype)


def schema(path: Path) -> str:
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
                first = handle.readline().strip()
            return "columns=" + first[:8000]
        if suffix == ".json":
            with path.open("r", encoding="utf-8-sig") as handle:
                value = json.load(handle)
            return "keys=" + "|".join(list(value)[:100]) if isinstance(value, dict) else f"type={type(value).__name__}"
        if suffix == ".parquet":
            meta = pq.ParquetFile(path).metadata
            return f"rows={meta.num_rows};row_groups={meta.num_row_groups};columns=" + "|".join(meta.schema.names[:100])
        if suffix == ".npy":
            with path.open("rb") as handle:
                shape, dtype = np_header(handle)
            return f"array={shape}:{dtype}"
        if suffix == ".npz":
            values: list[str] = []
            with zipfile.ZipFile(path) as archive:
                for info in archive.infolist():
                    if info.filename.endswith(".npy"):
                        with archive.open(info) as handle:
                            shape, dtype = np_header(handle)
                        values.append(f"{Path(info.filename).stem}:{shape}:{dtype}")
                        if len(values) >= 80:
                            break
            return "arrays=" + "|".join(values)
    except Exception as exc:
        return f"schema_error={type(exc).__name__}"
    return "unknown"


def classify(relative: str) -> str:
    lower = relative.lower()
    suffix = Path(relative).suffix.lower()
    roles: list[str] = []
    if lower.startswith("artifacts/"):
        roles.append("artifact")
    if "oof" in Path(relative).name.lower():
        roles.append("oof_prediction")
    if "test" in Path(relative).name.lower() and suffix in {".npz", ".npy", ".parquet", ".csv"}:
        roles.append("test_prediction")
    if "fold" in Path(relative).name.lower():
        roles.append("fold_artifact")
    if "metric" in lower or "summary" in lower or "report_" in lower or "verdict" in lower:
        roles.append("metrics_or_report")
    if lower.startswith("submissions/") or "submission" in Path(relative).name.lower():
        roles.append("submission")
    if suffix in {".py", ".sh", ".mjs"}:
        roles.append("code")
    if lower.startswith("experiments/") and re.match(r"(?i)exp[_-]?\d+.*\.md$", Path(relative).name):
        roles.append("experiment_report")
    return ";".join(sorted(set(roles))) or "other"


def parse_worktrees() -> list[dict]:
    raw = run_git(MAIN, "worktree", "list", "--porcelain")
    blocks = [block for block in raw.strip().split("\n\n") if block.strip()]
    rows: list[dict] = []
    for block in blocks:
        values: dict[str, str] = {}
        for line in block.splitlines():
            key, _, value = line.partition(" ")
            values[key] = value
        root = Path(values["worktree"]).resolve()
        if root == MAIN:
            continue
        rows.append({
            "root": root,
            "namespace": NAMESPACE_BY_DIR.get(root.name, root.name),
            "head": values.get("HEAD", "unknown"),
            "branch": values.get("branch", "unknown").removeprefix("refs/heads/"),
        })
    return rows


def main() -> None:
    worktrees = parse_worktrees()
    wt_rows: list[dict] = []
    file_rows: list[dict] = []
    copied_rows: list[dict] = []
    for worktree in worktrees:
        root: Path = worktree["root"]
        namespace = worktree["namespace"]
        status = run_git(root, "status", "--short", "--untracked-files=all")
        local_files: list[Path] = []
        for top in root.iterdir():
            if top.name in {".git", ".pytest_cache", "data", "__pycache__"}:
                continue
            if top.is_file():
                local_files.append(top)
            elif top.is_dir() and not top.is_symlink():
                local_files.extend(p for p in top.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
        total_bytes = 0
        for index, path in enumerate(sorted(local_files), 1):
            relative = path.relative_to(root).as_posix()
            digest = sha256(path)
            size = path.stat().st_size
            total_bytes += size
            roles = classify(relative)
            file_rows.append({
                "namespace": namespace,
                "worktree_root": str(root),
                "branch": worktree["branch"],
                "relative_path": relative,
                "size_bytes": size,
                "sha256": digest,
                "type": roles,
                "schema_or_arrays": schema(path),
            })

            should_copy = False
            if relative.startswith("artifacts/") and path.suffix.lower() in {".json", ".csv", ".log"} and size <= 5_000_000:
                should_copy = True
            if "experiment_report" in roles and namespace == "independent_anniversary" and "exact_anniversary" in path.name.lower():
                should_copy = True
            if "code" in roles and namespace == "independent_anniversary" and "anniversary" in path.name.lower():
                should_copy = True
            if "submission" in roles and namespace == "team_a_s2" and size <= 10_000_000:
                should_copy = True
            if should_copy:
                output = DEST / "evidence" / "worktree_artifacts" / namespace / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, output)
                copied_rows.append({
                    "namespace": namespace,
                    "original_path": str(root / relative),
                    "clean_path": output.relative_to(DEST).as_posix(),
                    "sha256": digest,
                    "size_bytes": size,
                    "type": roles,
                })
        wt_rows.append({
            "namespace": namespace,
            "root": str(root),
            "branch": worktree["branch"],
            "head": worktree["head"],
            "local_file_count_excluding_data_junction": len(local_files),
            "local_bytes_excluding_data_junction": total_bytes,
            "git_status_before": status.replace("\r", "").replace("\n", " | ").strip() or "clean",
            "data_link_target": str((root / "data").resolve()) if (root / "data").exists() else "missing",
        })
        print(f"{namespace}: {len(local_files)} files, {total_bytes} bytes", flush=True)

    write_specs = [
        (DEST / "inventory" / "worktrees.csv", wt_rows, ["namespace", "root", "branch", "head", "local_file_count_excluding_data_junction", "local_bytes_excluding_data_junction", "git_status_before", "data_link_target"]),
        (DEST / "inventory" / "worktree_files.csv", file_rows, ["namespace", "worktree_root", "branch", "relative_path", "size_bytes", "sha256", "type", "schema_or_arrays"]),
        (DEST / "evidence" / "worktree_artifacts_manifest.csv", copied_rows, ["namespace", "original_path", "clean_path", "sha256", "size_bytes", "type"]),
    ]
    for path, rows, fields in write_specs:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"worktrees": len(wt_rows), "files": len(file_rows), "copied_evidence": len(copied_rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
