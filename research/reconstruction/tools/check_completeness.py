#!/usr/bin/env python3
"""Create explicit completeness/orphan audits for the reconstructed repository."""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


UNKNOWN = "unknown"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, separators=(",", ":")) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def value_known(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in {"", "unknown", "none", "not_applicable", "not applicable", "n/a", "null"}
    return True


def metric_number_present(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, list):
        return any(metric_number_present(item) for item in value)
    if isinstance(value, dict):
        return any(metric_number_present(item) for item in value.values())
    if not isinstance(value, str) or not value_known(value):
        return False
    return bool(re.search(r"(?<![A-Za-z])[-+]?(?:\d+\.\d+|\.\d+)(?:[eE][-+]?\d+)?", value))


def report_linkage(root: Path) -> tuple[list[dict[str, Any]], int]:
    catalog = read_csv(root / "registry" / "report_catalog.csv")
    candidates = read_csv(root / "inventory" / "report_candidates.csv")
    by_source = {row["source_path"].replace("\\", "/"): row for row in catalog}
    by_hash: dict[str, list[dict[str, str]]] = {}
    for row in catalog:
        by_hash.setdefault(row["sha256"], []).append(row)
    result: list[dict[str, Any]] = []
    orphan_count = 0
    for row in candidates:
        path = row["original_path"].replace("\\", "/")
        sha = row["sha256"]
        association = row.get("experiment_association", UNKNOWN)
        if path in by_source:
            link_status = "primary_report_in_catalog"
            linked_to = by_source[path]["experiment_id"]
            evidence = by_source[path]["clean_evidence_path"]
        elif sha in by_hash:
            link_status = "exact_content_duplicate_of_catalog_report"
            linked_to = [item["experiment_id"] for item in by_hash[sha]]
            evidence = [item["clean_evidence_path"] for item in by_hash[sha]]
        elif path.startswith("research/strategies/results/"):
            link_status = "auxiliary_strategy_report_linked_to_machine_audit"
            linked_to = association
            evidence = "evidence/strategy_results_audit.md"
        elif path.startswith("пайплайн сокомандника/friend_original/"):
            link_status = "teammate_provenance_copy_linked_to_package_audit"
            linked_to = association
            evidence = "evidence/teammate_lb_audit.md"
        else:
            link_status = "orphan_report_candidate"
            linked_to = UNKNOWN
            evidence = UNKNOWN
            orphan_count += 1
        result.append({
            "original_path": path,
            "sha256": sha,
            "inventory_association": association,
            "link_status": link_status,
            "linked_to": linked_to,
            "evidence": evidence,
        })
    return result, orphan_count


def prediction_resolution(root: Path) -> tuple[list[dict[str, Any]], int]:
    candidates = read_csv(root / "artifacts" / "orphan_prediction_candidates.csv")
    result: list[dict[str, Any]] = []
    orphan_count = 0
    rules = [
        (re.compile(r"^ETX-AVG[23]-V\d+$"), "team_a_current:EXP-037", "EXP037 report plus ETX2 LOFO code/result tables"),
        (re.compile(r"^ETX-CKPT"), "team_a_current:EXP-037", "EXP036→EXP037 ETX checkpoint/depth-consistency lineage"),
        (re.compile(r"^L(?:180|90|None)_norm0_tb1$"), "team_a_current:EXP-027", "src/drift.py naming rule plus EXP027 depth-transfer diagnostic"),
        (re.compile(r"^S04-[ABC]$"), "team_a_current:RUN-S04-LGB", "machine S04 A/B/C reports and exact submission reconstruction"),
        (re.compile(r"^S04_ptest_s42$"), "team_a_current:RUN-S04-LGB", "S04 test prediction and reconstructed S04 submission formulas"),
        (re.compile(r"^TIER-A-DIRECT-AVG3-R300$"), "derived_component:TIER-A-CHECKPOINT", "TIER_A_REPORT plus tracked submission recipe; not promoted to experiment without run manifest"),
        (re.compile(r"^ZERO2D_DIST$"), "team_a_current:EXP-042", "src/zero2d_shrink.py explicitly writes/loads ZERO2D_DIST_test.npz"),
    ]
    for row in candidates:
        component = row["component_id"]
        linked_to = UNKNOWN
        evidence = UNKNOWN
        for pattern, target, reason in rules:
            if pattern.match(component):
                linked_to, evidence = target, reason
                break
        is_orphan = linked_to == UNKNOWN
        if is_orphan:
            orphan_count += 1
        result.append({
            **row,
            "forensic_resolution": linked_to,
            "resolution_evidence": evidence,
            "orphan_after_forensic_resolution": "yes" if is_orphan else "no",
        })
    return result, orphan_count


def no_primary_metric(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        fields = [row.get("cv_score"), row.get("delta_cv"), row.get("lb_score"), row.get("per_fold_cv")]
        if any(metric_number_present(value) for value in fields):
            continue
        status = str(row.get("status", UNKNOWN)).lower()
        if any(token in status for token in ("blocked", "preflight", "manifest", "inconclusive")):
            kind = "blocked_preflight_or_manifest_without_canonical_numeric_metric"
        elif row.get("namespace") == "teammate_review":
            kind = "runtime_backed_run_unit_without_standalone_metric"
        else:
            kind = "no_canonical_numeric_cv_delta_or_lb"
        result.append({
            "experiment_id": row["experiment_id"],
            "canonical_name": row["canonical_name"],
            "status": row["status"],
            "result_gap_kind": kind,
            "facts_preserved": row.get("facts", UNKNOWN),
            "source": row.get("source_report", UNKNOWN),
            "note": "Diagnostic or technical facts may still exist; this table only flags the canonical CV/delta/LB fields.",
        })
    return result


def main() -> None:
    root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
    experiments = read_jsonl(root / "registry" / "experiments.jsonl")
    reports = read_csv(root / "registry" / "report_catalog.csv")
    report_audit, orphan_reports = report_linkage(root)
    prediction_audit, orphan_predictions = prediction_resolution(root)
    metric_gaps = no_primary_metric(experiments)
    scripts = read_csv(root / "code_index" / "script_linkage_audit.csv")
    script_counts = Counter(row.get("link_status", UNKNOWN) for row in scripts)
    excluded = read_csv(root / "inventory" / "excluded_interpretive_documents.csv")
    excluded_paths = {row.get("original_path", row.get("path", "")).replace("\\", "/") for row in excluded}
    used_excluded: list[dict[str, Any]] = []
    for row in experiments:
        source = str(row.get("source_report", "")).replace("\\", "/")
        origin = str(row.get("source_origin", "")).replace("\\", "/")
        for path in excluded_paths:
            if path and (source == path or origin == path):
                used_excluded.append({"experiment_id": row["experiment_id"], "excluded_source": path})

    global_ids = [row["experiment_id"] for row in experiments]
    duplicate_global_ids = sorted(item for item, count in Counter(global_ids).items() if count > 1)
    primary_source_reports = [row for row in experiments if row.get("report_sha256") not in {None, UNKNOWN, ""}]
    submissions = read_csv(root / "submissions" / "registry.csv")
    if submissions:
        unknown_recipes = [row for row in submissions if not value_known(row.get("recipe"))]
    else:
        audit = read_csv(root / "submissions" / "audit.csv")
        unknown_recipes = [row for row in audit if row.get("exists") == "yes" and row.get("valid_submission_schema") == "yes" and not value_known(row.get("recipe"))]

    unverified_lb = read_csv(root / "leaderboard" / "report_only_claims.csv")
    verified_lb = read_csv(root / "leaderboard" / "chronology.csv")
    dedup = read_csv(root / "registry" / "deduplication.csv")
    components = read_csv(root / "registry" / "components.csv")
    run_metrics = read_csv(root / "registry" / "run_metrics.csv")
    design_fields = read_jsonl(root / "evidence" / "experiment_design_fields.jsonl")
    contradictions = read_csv(root / "contradictions" / "registry.csv")
    secondary_conflicts = read_jsonl(root / "evidence" / "secondary_summary_conflicts.jsonl")

    write_csv(root / "inventory" / "report_linkage_audit.csv", report_audit)
    write_csv(root / "artifacts" / "orphan_prediction_resolution.csv", prediction_audit)
    write_csv(root / "registry" / "experiments_without_canonical_numeric_metric.csv", metric_gaps)
    write_csv(root / "inventory" / "excluded_sources_used_for_facts.csv", used_excluded, ["experiment_id", "excluded_source"])

    summary = {
        "primary_report_catalog_rows": len(reports),
        "primary_report_registry_rows": len(primary_source_reports),
        "primary_reports_with_structured_design_evidence": len(design_fields),
        "current_worktree_report_candidates": len(report_audit),
        "orphan_report_candidates_after_linkage": orphan_reports,
        "central_experiment_registry_rows": len(experiments),
        "duplicate_global_experiment_ids": len(duplicate_global_ids),
        "cross_namespace_local_id_collisions": len(read_csv(root / "registry" / "id_collisions.csv")),
        "dedup_rerun_reuse_clusters": len(dedup),
        "component_groups": len(components),
        "granular_run_metric_records": len(run_metrics),
        "initial_orphan_prediction_candidates": len(prediction_audit),
        "orphan_predictions_after_forensic_resolution": orphan_predictions,
        "main_scripts": len(scripts),
        "scripts_linked_to_primary_report": script_counts.get("linked_to_primary_report", 0),
        "shared_pipeline_dependencies": script_counts.get("shared_pipeline_dependency", 0),
        "scripts_without_explicit_primary_report_link": script_counts.get("no_explicit_primary_report_link", 0),
        "experiments_without_canonical_numeric_cv_delta_or_lb": len(metric_gaps),
        "verified_repository_internal_lb_links": len(verified_lb),
        "platform_independently_verified_lb_links": sum(row.get("external_platform_export_present") in {"True", "true", "yes"} for row in verified_lb),
        "unverified_lb_claim_records": len(unverified_lb),
        "contradiction_and_caveat_rows": len(contradictions),
        "secondary_summary_conflict_rows_used_for_facts_no": len(secondary_conflicts),
        "existing_submissions_without_forensic_recipe": len(unknown_recipes),
        "excluded_interpretive_documents_inventoried": len(excluded),
        "excluded_interpretive_documents_used_as_fact_sources": len(used_excluded),
    }
    (root / "reports" / "completeness_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (root / "inventory" / "duplicate_global_experiment_ids.json").write_text(json.dumps(duplicate_global_ids, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
