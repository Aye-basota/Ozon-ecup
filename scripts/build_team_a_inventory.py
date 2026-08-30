"""Build the machine-readable forensic inventory for the Team-A package.

The scanner is intentionally read-only.  It records source files in the clean
repository, the active Team-A worktree, discovered historical worktrees,
submission-geometry research, downloads that match competition-specific names,
and the two standalone forensic bundles.  Environments and transient caches are
excluded, while raw/model/prediction artifacts are inventoried but not copied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path


ROOTS = (
    Path(r"C:\Users\Admin\Desktop\e-cup-research-clean"),
    Path(r"C:\Users\Admin\Desktop\OZON-E-CUP"),
    Path(r"C:\Users\Admin\Desktop\submission_geometry_research"),
    Path(r"C:\Users\Admin\Desktop\latest_pipeline_bundle"),
    Path(r"C:\Users\Admin\Desktop\research_clean"),
    Path(r"C:\Users\Admin\Desktop\OZON-E-CUP-calendar-placebo-01"),
    Path(r"C:\Users\Admin\Desktop\OZON-E-CUP-domain-01"),
    Path(r"C:\Users\Admin\Desktop\OZON-E-CUP-exp057-global-regime-occ"),
    Path(r"C:\Users\Admin\Desktop\OZON-E-CUP-renewal-01"),
    Path(r"C:\Users\Admin\Desktop\OZON-E-CUP-s2"),
    Path(r"C:\Users\Admin\Desktop\OZON-ECUP2-WORK"),
    Path(r"C:\Users\Admin\Downloads"),
)

EXCLUDED_PARTS = {
    ".git",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    "__pycache__",
    "node_modules",
    "catboost_info",
    ".codex_sheet_audit",
    "_tmp_strongest_training_v2",
}

DOWNLOAD_PATTERN = re.compile(
    r"(?i)(ozon|e.?cup|submit|joint|strongest|team.?a|team.?b|"
    r"exp.?0(?:[0-9]{2})|orth|occurrence|sample_submit|teammate_repro)"
)
EXPERIMENT_PATTERN = re.compile(r"(?i)\b(exp[_ -]?\d{3}[a-z]?|s\d+-e\d+[a-z]?)\b")
FINAL_STRONGEST = "SUBMIT_STRONGEST55_TEAMB45"
FINAL_JOINT = "SUBMIT_JOINT86_TEAMB14"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_metadata(root: Path) -> tuple[str, str]:
    probe = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode:
        return "", ""
    repo = probe.stdout.strip()
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return repo, commit


def include_file(root: Path, path: Path) -> bool:
    rel = path.relative_to(root)
    if any(part in EXCLUDED_PARTS or part.startswith(".venv") for part in rel.parts):
        return False
    if root.name == "Downloads":
        return bool(DOWNLOAD_PATTERN.search(str(rel)))
    return True


def infer_experiment(path: Path) -> str:
    matches = EXPERIMENT_PATTERN.findall(str(path))
    return matches[-1].upper().replace(" ", "_") if matches else ""


def used_by_final(path: Path) -> str:
    text = str(path).upper()
    strongest = any(
        token in text
        for token in (
            FINAL_STRONGEST,
            "STRONGEST_CURRENT",
            "S1-CAP",
            "S1-UNC",
            "S1-DIST",
            "SEQ-AVG3",
            "SEQ-C289",
            "ETX-AVG3",
            "ETX-01",
        )
    )
    joint = any(token in text for token in (FINAL_JOINT, "SUBMIT_JOINT_V2"))
    team_b = "FINAL_CLASSIC_ML" in text or "TEAM-B-FINAL" in text or "TEAM_B" in text
    if team_b:
        strongest = joint = True
    if strongest and joint:
        return f"{FINAL_STRONGEST};{FINAL_JOINT}"
    if strongest:
        return FINAL_STRONGEST
    if joint:
        return FINAL_JOINT
    return "NO"


def classify(path: Path) -> str:
    text = str(path).lower()
    name = path.name.lower()
    suffix = path.suffix.lower()
    if name in {f"{FINAL_STRONGEST.lower()}.csv", f"{FINAL_JOINT.lower()}.csv"}:
        return "FINAL_SUBMISSION"
    if any(token in name for token in ("build_submit", "reproduce_final", "build_strongest")):
        return "FINAL_PIPELINE"
    if "oof" in name or "aligned_oof" in text:
        return "OOF"
    if suffix in {".pt", ".pth", ".joblib", ".pkl", ".cbm"} or "models" in path.parts:
        return "MODEL"
    if suffix in {".npy", ".npz", ".parquet"} and any(
        token in text for token in ("ztest", "test_pred", "aligned_test", "predictions")
    ):
        return "TEST_PREDICTION"
    if any(part.lower().startswith("exp") for part in path.parts) or EXPERIMENT_PATTERN.search(name):
        return "EXPERIMENT"
    if name in {"requirements.txt", "pyproject.toml", "environment.yml"} or suffix in {
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
    } or "config" in name:
        return "CONFIG"
    if suffix in {".md", ".rst"} or any(token in name for token in ("report", "manifest", "audit")):
        return "REPORT"
    if suffix == ".csv" and any(token in name for token in ("submit", "submission")):
        return "FINAL_SUBMISSION" if used_by_final(path) != "NO" else "LEGACY"
    if suffix in {".py", ".sh", ".ps1", ".mjs"}:
        return "REUSABLE_COMPONENT"
    if any(token in text for token in ("legacy", "history", "archive")):
        return "LEGACY"
    return "UNKNOWN"


def destination(root: Path, path: Path, artifact_type: str) -> str:
    rel = path.relative_to(root)
    if root.name == "e-cup-research-clean":
        return rel.as_posix()
    if artifact_type in {"MODEL", "OOF", "TEST_PREDICTION"} or "data" in {
        part.lower() for part in rel.parts
    }:
        return "EXTERNAL_ONLY (documented in inventory)"
    if root.name == "OZON-E-CUP":
        if rel.parts and rel.parts[0] == "experiments":
            return (Path("experiments/team_a") / Path(*rel.parts[1:])).as_posix()
        return (Path("research/legacy_team_a") / rel).as_posix()
    if root.name == "submission_geometry_research":
        return (Path("research/submission_geometry") / rel).as_posix()
    if root.name == "latest_pipeline_bundle":
        return (Path("research/provenance/latest_pipeline_bundle") / rel).as_posix()
    if root.name == "research_clean":
        return (Path("research/reconstruction") / rel).as_posix()
    if root.name == "Downloads":
        return "SOURCE_ONLY (copy selected provenance files)"
    if root.name == "OZON-ECUP2-WORK":
        return "EXTERNAL_ONLY (raw/audit workspace)"
    return (Path("research/worktree_snapshots") / root.name / rel).as_posix()


def base_status(path: Path, artifact_type: str) -> str:
    lower_parts = {part.lower() for part in path.parts}
    if "data" in lower_parts and path.suffix.lower() in {".parquet", ".csv", ".xlsx"}:
        return "EXTERNAL_RAW_OR_CACHE"
    if path.stat().st_size > 50 * 1024 * 1024:
        return "EXTERNAL_LARGE"
    if artifact_type == "UNKNOWN":
        return "PROVENANCE_INCOMPLETE"
    return "FOUND"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs" / "TEAM_A_SOURCE_INVENTORY.csv",
    )
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    for root in ROOTS:
        if not root.exists():
            continue
        repo, commit = git_metadata(root)
        print(f"scan {root}", flush=True)
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            base = Path(dirpath)
            is_junction = getattr(os.path, "isjunction", lambda _: False)
            dirnames[:] = [
                name
                for name in dirnames
                if name not in EXCLUDED_PARTS and not name.startswith(".venv")
                and not (base / name).is_symlink()
                and not is_junction(base / name)
            ]
            for filename in filenames:
                path = base / filename
                if not include_file(root, path) or not path.is_file():
                    continue
                artifact_type = classify(path)
                try:
                    digest = sha256(path)
                except OSError as exc:
                    digest = f"ERROR:{exc.__class__.__name__}"
                rows.append(
                    {
                        "source_path": str(path.resolve()),
                        "artifact_type": artifact_type,
                        "experiment": infer_experiment(path),
                        "description": f"{artifact_type.lower().replace('_', ' ')}: {path.name}",
                        "git_repo": repo,
                        "git_commit_if_known": commit,
                        "used_by_final_solution": used_by_final(path),
                        "destination": destination(root, path, artifact_type),
                        "sha256": digest,
                        "status": base_status(path, artifact_type),
                    }
                )

    by_hash: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if not row["sha256"].startswith("ERROR:"):
            by_hash[row["sha256"]].append(index)
    for indexes in by_hash.values():
        if len(indexes) < 2:
            continue
        canonical = indexes[0]
        canonical_path = rows[canonical]["source_path"]
        for index in indexes[1:]:
            rows[index]["artifact_type"] = "DUPLICATE"
            rows[index]["description"] += f"; byte-identical to {canonical_path}"
            if rows[index]["status"] == "FOUND":
                rows[index]["status"] = "DUPLICATE_SHA256"

    rows.sort(key=lambda row: row["source_path"].lower())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "source_path",
        "artifact_type",
        "experiment",
        "description",
        "git_repo",
        "git_commit_if_known",
        "used_by_final_solution",
        "destination",
        "sha256",
        "status",
    ]
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
