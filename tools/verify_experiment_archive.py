"""Fast, non-training verification for the materialized experiment archive.

Checks syntax without producing ``__pycache__``, catalogue/folder consistency,
README coverage, git-history card coverage and the no-data/no-submission rule.

Run:
    python tools/verify_experiment_archive.py
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "experiments" / "repro" / "catalog.json"
README = ROOT / "README.md"


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    ).stdout


def check_python_syntax() -> int:
    excluded = {".git", "__pycache__", ".pytest_cache", "data", "artifacts", "submissions", "catboost_info"}
    paths = [
        path for path in ROOT.rglob("*.py")
        if not any(part in excluded for part in path.relative_to(ROOT).parts)
    ]
    errors: list[str] = []
    for index, path in enumerate(paths, start=1):
        try:
            source = path.read_text(encoding="utf-8")
            compile(source, str(path.relative_to(ROOT)), "exec")
        except Exception as exc:  # report every file in a single pass
            errors.append(f"{path.relative_to(ROOT)}: {type(exc).__name__}: {exc}")
        if index % 500 == 0:
            print(f"syntax: {index}/{len(paths)}", flush=True)
    if errors:
        raise RuntimeError("Python syntax failures:\n" + "\n".join(errors))
    return len(paths)


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

    tracked = git("ls-files").replace("\\", "/").splitlines()
    forbidden = [path for path in tracked if path.startswith("data/") or path.startswith("submissions/")]
    if forbidden:
        raise RuntimeError(f"forbidden data/submission files are tracked: {forbidden}")

    compiled = check_python_syntax()
    print(json.dumps({
        "catalog_entries": len(catalog),
        "historical_cards": len(history_cards),
        "python_files_compiled": compiled,
        "tracked_forbidden_files": 0,
        "status": "PASS",
    }, indent=2))


if __name__ == "__main__":
    main()
