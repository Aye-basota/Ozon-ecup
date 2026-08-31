"""Fast, non-training verification for the materialized experiment archive.

Checks syntax without producing ``__pycache__``, catalogue/folder consistency,
README coverage, git-history card coverage and the no-new-data/no-new-submission rule.

Run:
    python tools/verify_experiment_archive.py
"""
from __future__ import annotations

import json
import csv
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "experiments" / "repro" / "catalog.json"
README = ROOT / "README.md"
REPORT = ROOT / "docs" / "EXPERIMENT_ARCHIVE_VERIFICATION.json"


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    ).stdout


def check_python_syntax() -> tuple[int, list[str]]:
    excluded = {".git", "__pycache__", ".pytest_cache", "data", "artifacts", "submissions", "catboost_info"}
    paths = [
        path for path in ROOT.rglob("*.py")
        if not any(part in excluded for part in path.relative_to(ROOT).parts)
    ]
    errors: list[str] = []
    preserved_invalid: list[str] = []
    known_forensic_wip = {
        "research/reconstruction/code_index/snapshots/team_b_core/src/train.py",
    }
    for index, path in enumerate(paths, start=1):
        try:
            source = path.read_text(encoding="utf-8")
            compile(source, str(path.relative_to(ROOT)), "exec")
        except Exception as exc:  # report every file in a single pass
            relative = path.relative_to(ROOT).as_posix()
            message = f"{relative}: {type(exc).__name__}: {exc}"
            if relative in known_forensic_wip:
                preserved_invalid.append(message)
            else:
                errors.append(message)
        if index % 500 == 0:
            print(f"syntax: {index}/{len(paths)}", flush=True)
    if errors:
        raise RuntimeError("Python syntax failures:\n" + "\n".join(errors))
    return len(paths) - len(preserved_invalid), preserved_invalid


def main() -> None:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    identifiers = [entry["catalog_id"] for entry in catalog]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("catalog_id values are not unique")
    readme = README.read_text(encoding="utf-8")
    for entry in catalog:
        folder = ROOT / "experiments" / "repro" / entry["catalog_id"]
        for name in ("README.md", "experiment.json", "run.py"):
            if not (folder / name).is_file():
                raise FileNotFoundError(folder / name)
        if entry["catalog_id"] not in readme:
            raise RuntimeError(f"README does not mention {entry['catalog_id']}")

    history_cards = {
        line for line in git("log", "--all", "--name-only", "--pretty=format:").splitlines()
        if line.startswith("experiments/exp_") and line.endswith(".md")
    }
    origins = [str(entry["original_path"]) for entry in catalog]
    missing = [
        path for path in sorted(history_cards)
        if path not in origins and not any(origin.endswith(f":{path}") for origin in origins)
    ]
    if missing:
        raise RuntimeError(f"historical cards missing from catalogue: {missing}")

    new_direction_root = ROOT / "research" / "new_directions"
    expected_directions = {
        str(path.relative_to(ROOT)).replace("\\", "/")
        for path in new_direction_root.iterdir() if path.is_dir()
    } if new_direction_root.exists() else set()
    covered_directions = {
        str(entry["original_path"]).rstrip("/")
        for entry in catalog if entry["namespace"] == "new_direction"
    }
    if expected_directions != covered_directions:
        raise RuntimeError(
            "late new-direction coverage mismatch: "
            f"missing={sorted(expected_directions - covered_directions)}, "
            f"extra={sorted(covered_directions - expected_directions)}"
        )
    if expected_directions and not any(
        entry["namespace"] == "independent_anniversary" for entry in catalog
    ):
        raise RuntimeError("reconstructed independent anniversary report is missing")

    reconstruction_catalog = ROOT / "research" / "reconstruction" / "registry" / "report_catalog.csv"
    reconstruction_rows: list[dict[str, str]] = []
    if reconstruction_catalog.exists():
        with reconstruction_catalog.open(encoding="utf-8-sig", newline="") as stream:
            reconstruction_rows = list(csv.DictReader(stream))
        origins = [str(entry["original_path"]) for entry in catalog]
        uncovered_primary = [
            row["source_path"] for row in reconstruction_rows
            if row["source_path"] not in origins
            and not any(origin.endswith(f":{row['source_path']}") for origin in origins)
        ]
        if uncovered_primary:
            raise RuntimeError(f"reconstruction primary reports missing: {uncovered_primary}")

    packaged_cards = list((ROOT / "experiments" / "team_a").glob("exp_*.md"))
    semantically_divergent_cards: list[str] = []
    for packaged in packaged_cards:
        canonical = ROOT / "experiments" / packaged.name
        if not canonical.exists():
            semantically_divergent_cards.append(packaged.name + " (canonical missing)")
            continue
        normalize = lambda path: re.sub(
            r"\s+", " ", path.read_text(encoding="utf-8", errors="replace")
        ).strip()
        if normalize(packaged) != normalize(canonical):
            semantically_divergent_cards.append(packaged.name)
    if semantically_divergent_cards:
        raise RuntimeError(
            "packaged Team-A cards diverge from canonical cards: "
            f"{semantically_divergent_cards}"
        )

    tracked = git("ls-files").replace("\\", "/").splitlines()
    forbidden = [path for path in tracked if path.startswith("data/") or path.startswith("submissions/")]
    # The fetched Team-A reconstruction commit already contains reviewed frozen
    # final evidence under these roots.  This archive must not add any *new*
    # raw-data/submission paths, while preserving the fetched history verbatim.
    remote_tracked = set(
        git("ls-tree", "-r", "--name-only", "origin/team-a").replace("\\", "/").splitlines()
    )
    added_forbidden = [path for path in forbidden if path not in remote_tracked]
    if added_forbidden:
        raise RuntimeError(f"archive added forbidden data/submission files: {added_forbidden}")

    compiled, preserved_invalid = check_python_syntax()
    result = {
        "catalog_entries": len(catalog),
        "historical_cards": len(history_cards),
        "python_files_compiled": compiled,
        "preserved_forensic_wip_syntax_failures": preserved_invalid,
        "inherited_reviewed_data_submission_files": len(forbidden),
        "archive_added_forbidden_files": len(added_forbidden),
        "late_new_direction_packages": len(expected_directions),
        "reconstruction_primary_reports_covered": len(reconstruction_rows),
        "semantic_duplicate_packaged_team_a_cards": len(packaged_cards),
        "status": "PASS",
    }
    REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
