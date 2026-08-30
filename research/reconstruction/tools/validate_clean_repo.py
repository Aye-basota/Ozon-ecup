#!/usr/bin/env python3
"""Internal consistency checks for the reconstructed research repository."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED_EXPERIMENT_FIELDS = [
    "experiment_id", "canonical_name", "family", "date", "parent_baseline",
    "change", "model_family", "validation_protocol", "cv_score", "delta_cv",
    "folds_positive", "folds_total", "lb_score", "runtime", "status",
    "evidence_strength", "artifacts", "duplicate_of", "compatible_tags", "notes",
]


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def jsonl_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def main() -> None:
    root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    required_paths = [
        "README.md", "registry/experiments.csv", "registry/experiments.jsonl",
        "registry/run_metrics.csv", "registry/components.csv", "registry/family_summary.csv",
        "baselines/chronology.csv", "submissions/registry.csv", "leaderboard/chronology.csv",
        "ensembles/ancestry_edges.csv", "contradictions/registry.csv",
        "reports/REPOSITORY_RECONSTRUCTION.md", "reports/START_HERE.md",
    ]
    missing = [path for path in required_paths if not (root / path).is_file()]
    check("required_paths_present", not missing, missing or "all present")

    experiments_csv = csv_rows(root / "registry" / "experiments.csv")
    experiments_jsonl = jsonl_rows(root / "registry" / "experiments.jsonl")
    check("experiment_csv_jsonl_row_parity", len(experiments_csv) == len(experiments_jsonl), {"csv": len(experiments_csv), "jsonl": len(experiments_jsonl)})
    ids = [row["experiment_id"] for row in experiments_csv]
    check("global_experiment_ids_unique", len(ids) == len(set(ids)), [item for item, count in Counter(ids).items() if count > 1])
    missing_fields = {field: sum(not row.get(field) for row in experiments_csv) for field in REQUIRED_EXPERIMENT_FIELDS}
    check("required_registry_fields_nonblank", all(value == 0 for value in missing_fields.values()), missing_fields)
    check("unknown_is_explicit", all(all(value is not None and str(value) != "" for value in row.values()) for row in experiments_csv), "no blank CSV cells")

    report_catalog = csv_rows(root / "registry" / "report_catalog.csv")
    report_backed = [row for row in experiments_csv if row.get("report_sha256") not in {"", "unknown"}]
    check("report_catalog_registry_parity", len(report_catalog) == len(report_backed) == 124, {"catalog": len(report_catalog), "registry": len(report_backed)})

    design_rows = jsonl_rows(root / "evidence" / "experiment_design_fields.jsonl")
    design_ids = [row.get("experiment_id") for row in design_rows]
    report_ids = {row["experiment_id"] for row in report_backed}
    required_design_fields = {
        "experiment_id", "report_path", "train_construction", "features",
        "target", "folds", "seeds", "hyperparameters",
    }
    valid_design_schema = all(required_design_fields == set(row) for row in design_rows)
    check(
        "one_design_evidence_record_per_primary_report",
        len(design_rows) == len(set(design_ids)) == 124
        and set(design_ids) == report_ids and valid_design_schema,
        {
            "design_rows": len(design_rows),
            "unique_ids": len(set(design_ids)),
            "matched_report_ids": len(set(design_ids) & report_ids),
            "schema_valid": valid_design_schema,
        },
    )

    normalized_cards = list((root / "experiments" / "normalized").glob("*.md"))
    check("one_normalized_card_per_registry_row", len(normalized_cards) == len(experiments_csv), {"cards": len(normalized_cards), "registry": len(experiments_csv)})
    families = csv_rows(root / "registry" / "family_summary.csv")
    family_docs = [path for path in (root / "families").glob("*.md") if path.name.lower() != "readme.md"]
    check("one_family_page_per_family", len(family_docs) == len(families), {"pages": len(family_docs), "families": len(families)})

    leaderboard = csv_rows(root / "leaderboard" / "chronology.csv")
    submissions = csv_rows(root / "submissions" / "registry.csv")
    submission_hashes = {row["sha256"] for row in submissions}
    missing_lb_hashes = [row["artifact_sha256"] for row in leaderboard if row["artifact_sha256"] not in submission_hashes]
    check("confirmed_lb_artifacts_in_submission_registry", not missing_lb_hashes, missing_lb_hashes or f"{len(leaderboard)} links matched")
    check("confirmed_lb_has_no_platform_overclaim", all(row.get("external_platform_export_present") in {"False", "false", "no", "0"} for row in leaderboard), "all marked repository-internal only")
    unknown_recipes = [row["path"] for row in submissions if row.get("recipe", "").strip().lower() in {"", "unknown"}]
    check("all_existing_submissions_have_forensic_recipe", not unknown_recipes, unknown_recipes or f"{len(submissions)} recipes present")

    excluded_use = csv_rows(root / "inventory" / "excluded_sources_used_for_facts.csv")
    check("excluded_interpretive_sources_not_used", not excluded_use, excluded_use or "zero rows")

    secondary_conflicts = jsonl_rows(root / "evidence" / "secondary_summary_conflicts.jsonl")
    contradiction_rows = csv_rows(root / "contradictions" / "registry.csv")
    secondary_registry_rows = [
        row for row in contradiction_rows
        if row.get("impact", "").startswith("secondary-only;")
    ]
    check(
        "secondary_summaries_are_conflict_objects_not_fact_sources",
        len(secondary_conflicts) == len(secondary_registry_rows) == 22
        and all("used_for_facts=no" in row.get("impact", "") for row in secondary_registry_rows),
        {
            "secondary_audit_rows": len(secondary_conflicts),
            "contradiction_registry_rows": len(secondary_registry_rows),
            "all_marked_used_for_facts_no": all(
                "used_for_facts=no" in row.get("impact", "")
                for row in secondary_registry_rows
            ),
        },
    )

    derived_json = [
        root / "registry" / "registry_build_summary.json",
        root / "reports" / "completeness_summary.json",
    ]
    json_errors: list[str] = []
    for path in derived_json:
        try:
            json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception as exc:  # noqa: BLE001 - audit should report every parse error
            json_errors.append(f"{path.relative_to(root)}: {exc}")
    for path in list((root / "registry").glob("*.jsonl")) + list((root / "leaderboard").glob("*.jsonl")) + list((root / "contradictions").glob("*.jsonl")) + list((root / "baselines").glob("*.jsonl")) + list((root / "ensembles").glob("*.jsonl")):
        try:
            jsonl_rows(path)
        except Exception as exc:  # noqa: BLE001
            json_errors.append(f"{path.relative_to(root)}: {exc}")
    check("derived_json_and_jsonl_parse", not json_errors, json_errors or "all parsed")

    integrity_path = root / "reports" / "source_unchanged_verification.json"
    integrity = json.loads(integrity_path.read_text(encoding="utf-8-sig")) if integrity_path.exists() else {}
    check("source_and_linked_worktrees_untouched", integrity.get("source_untouched_verified") is True, {
        "source_untouched_verified": integrity.get("source_untouched_verified", False),
        "main_files": integrity.get("main", {}).get("expected_files", "unknown"),
        "linked_worktrees": len(integrity.get("linked_worktrees", {})),
    })

    large = [{"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size} for path in root.rglob("*") if path.is_file() and ".git" not in path.parts and path.stat().st_size > 50 * 1024 * 1024]
    check("no_unnecessary_large_artifacts_copied", not large, large or "no file above 50 MiB")

    failed = [row for row in checks if not row["passed"]]
    result = {"checks": checks, "passed": not failed, "failed_checks": len(failed)}
    (root / "reports" / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Reconstruction validation", "", f"Overall: **{'PASS' if not failed else 'FAIL'}**", "", "| Check | Result | Detail |", "|---|---|---|"]
    for row in checks:
        detail = json.dumps(row["detail"], ensure_ascii=False, separators=(",", ":")) if not isinstance(row["detail"], str) else row["detail"]
        lines.append(f"| {row['check']} | {'PASS' if row['passed'] else 'FAIL'} | {detail.replace('|', '/')} |")
    lines.append("")
    (root / "reports" / "VALIDATION.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"passed": not failed, "checks": len(checks), "failed": [row["check"] for row in failed]}, ensure_ascii=False, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
