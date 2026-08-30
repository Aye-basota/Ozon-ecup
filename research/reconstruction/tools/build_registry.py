#!/usr/bin/env python3
"""Build the normalized research registry from forensic evidence extracts.

This script performs no work in the source repository. It only reads evidence
already copied/indexed in research_clean and writes derived text/CSV/JSONL here.
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


UNKNOWN = "unknown"
NONE_LIKE = {"", "unknown", "none", "null", "n/a", "na", "not_applicable", "not applicable"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSONL {path}:{line_no}: {exc}") from exc
    return rows


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key, UNKNOWN)) for key in fields})


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def csv_value(value: Any) -> Any:
    if value is None:
        return UNKNOWN
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def clean_value(value: Any) -> Any:
    if value is None:
        return UNKNOWN
    if isinstance(value, str):
        value = value.strip()
        return value if value else UNKNOWN
    return value


def text_value(value: Any) -> str:
    value = clean_value(value)
    if isinstance(value, str):
        return value
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def is_unknown(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in NONE_LIKE
    return False


def as_list(value: Any) -> list[Any]:
    if value is None or is_unknown(value):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def pure_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", candidate):
        return None
    number = float(candidate)
    return number if math.isfinite(number) else None


def local_id(raw: str) -> str:
    value = raw.strip()
    if ":" in value:
        value = value.rsplit(":", 1)[1]
    value = value.replace("_", "-").upper()
    match = re.fullmatch(r"EXP-?(\d+)([A-Z]*)", value)
    if match:
        return f"EXP-{int(match.group(1)):03d}{match.group(2)}"
    if value.startswith("EXP-"):
        return value
    return value


def namespace_for_report(path: str) -> str:
    normalized = path.replace("\\", "/")
    match = re.match(r"experiments/([^/]+)/", normalized)
    if match and match.group(1) in {
        "team_a_s2",
        "team_b_core",
        "team_b_alt",
        "independent_anniversary",
        "independent_renewal",
        "independent_domain",
        "independent_calendar",
        "independent_global_regime",
    }:
        return match.group(1)
    return "team_a_current"


def global_id(raw: str, namespace: str) -> str:
    if ":" in raw:
        possible_ns, possible_id = raw.split(":", 1)
        if possible_ns:
            namespace = possible_ns.lower().replace("-", "_")
            raw = possible_id
    return f"{namespace}:{local_id(raw)}"


def normalize_family(raw: Any, name: str, change: str) -> str:
    text = " ".join([text_value(raw), name, change]).lower()
    if any(key in text for key in ("submission", "production integration", "artifact provenance", "artifact_provenance", "artifact package")):
        return "production_integration_and_provenance"
    if any(key in text for key in ("validation", "reproducib", "diagnostic", "audit", "fingerprint", "experiment_manifest")):
        return "validation_reproducibility_and_diagnostics"
    if any(key in text for key in ("ensemble", "mix", "stack", "shapley", "seed aver", "seed_aver", "slot replacement", "cached meta", "cached_meta")):
        return "ensembles_stacking_and_component_selection"
    if any(key in text for key in ("postprocess", "calibr", "shrink", "level minus", "detrend", "reweight")):
        return "calibration_and_postprocessing"
    if any(key in text for key in ("btyd", "bgnbd", "bg/nbd", "occurrence", "renewal")):
        return "behavioral_occurrence_and_btyd"
    if any(key in text for key in ("residual", "ridge15", "ridge 15")):
        return "residual_and_correction_models"
    if any(key in text for key in ("domain", "unlabeled", "dataset identity", "dataset shift")):
        return "domain_shift_unlabeled_and_dataset_identity"
    if any(key in text for key in ("sequence", "transformer", "etx", "tcn", "event order", "landmark", "burst-gap")):
        return "neural_sequence_and_event_models"
    if any(key in text for key in ("target", "distribution", "hurdle", "count", "zero", "aggregation", "funnel", "hazard")):
        return "target_distribution_and_decomposition"
    if any(key in text for key in ("panel", "train example", "dense cutoff", "train block", "coverage", "construction")):
        return "train_example_construction"
    if any(key in text for key in ("temporal", "calendar", "holiday", "history", "recency", "gap-axis", "personal-time")):
        return "temporal_history_and_calendar"
    if any(key in text for key in ("feature", "cohort", "capacity", "minimal", "tabular", "lightgbm", "hgbr", "hyperparameter")):
        return "tabular_models_and_feature_engineering"
    return "other_evidenced_experiment"


STANDARD_TAGS = {
    "changes_data_construction",
    "changes_representation",
    "changes_model",
    "changes_target",
    "changes_postprocessing",
    "changes_ensemble",
    "same_prediction_source",
    "nested_modification",
    "mutually_exclusive",
    "potentially_additive",
    "unknown",
}


def normalize_tags(raw: Any, family: str, change: str) -> tuple[list[str], list[str]]:
    raw_tags = [str(item) for item in as_list(raw)]
    combined = " ".join(raw_tags + [family, change]).lower()
    tags: set[str] = set()
    aliases = {
        "changes_data_construction": ("data_construction", "train_example", "panel", "cutoff", "train block"),
        "changes_representation": ("representation", "sequence", "feature", "temporal", "calendar"),
        "changes_model": ("changes_model", "model", "lightgbm", "transformer", "tcn", "ridge", "bgnbd"),
        "changes_target": ("changes_target", "target", "hurdle", "distribution", "count", "zero"),
        "changes_postprocessing": ("postprocess", "calibr", "shrink", "level", "clip", "reweight", "detrend"),
        "changes_ensemble": ("ensemble", "mix", "stack", "blend", "slot"),
        "same_prediction_source": ("same_prediction_source", "same prediction", "same source"),
        "nested_modification": ("nested", "continuation", "follow-up", "followup"),
        "mutually_exclusive": ("mutually_exclusive", "mutually exclusive"),
        "potentially_additive": ("potentially_additive", "potentially additive", "complement"),
    }
    for target, needles in aliases.items():
        if any(needle in combined for needle in needles):
            tags.add(target)
    if not tags:
        tags.add("unknown")
    return sorted(tags & STANDARD_TAGS), raw_tags


def fold_dates(row: dict[str, Any]) -> list[str]:
    text = " ".join(text_value(row.get(key, UNKNOWN)) for key in ("folds", "validation_protocol"))
    return sorted(set(re.findall(r"20\d\d-\d\d-\d\d", text)))


def comparison_class(row: dict[str, Any], machine: dict[str, Any] | None) -> str:
    ns = row["namespace"]
    validation_parts = [text_value(row.get("validation_protocol", UNKNOWN))]
    if machine:
        validation_parts.append(text_value(machine.get("validation", UNKNOWN)))
    validation = " | ".join(validation_parts).lower()
    folds = fold_dates(row)
    joined_folds = "_".join(date.replace("-", "") for date in folds)
    if "simulation" in validation or "analytic" in validation or "not ml cv" in validation:
        return f"{ns}:simulation_or_analytic_check"
    if "attempted comparison mixed" in validation or "mixed a four-fold" in validation:
        return f"{ns}:invalid_mixed_fold_comparison"
    if any(token in validation for token in ("no new cv", "no local cv", "production-only", "no cv run", "canonical outer lofo")):
        return f"{ns}:no_comparable_cv"
    if "group a only" in validation or "on group a only" in validation or "on the same group-a" in validation:
        return f"{ns}:group_a_half_panel_4fold"
    if "stress scores are not ordinary" in validation or ("depth stress" in validation and "no new canonical" in validation):
        return f"{ns}:diagnostic_stress_not_ordinary_cv"
    if "pseudo-production" in validation or "pseudo production" in validation or "half split" in validation:
        return f"{ns}:pseudo_production_half_split"
    if "auc" in validation and "rmsle" not in validation:
        return f"{ns}:diagnostic_auc"
    if "three folds" in validation or ("three-fold" in validation and "four-fold" not in validation):
        return f"{ns}:three_fold_protocol"
    if "one-fold" in validation or "one fold" in validation or "single-fold" in validation or "single fold" in validation or len(folds) == 1:
        return f"{ns}:single_fold_protocol"
    if "informational_only" in validation or "marked informational" in validation:
        return f"{ns}:informational_4fold_not_decision_metric"
    if "oracle" in validation or "preflight" in validation:
        return f"{ns}:preflight_or_auxiliary_metric"
    if "nested lofo" in validation or "honest lofo" in validation or "nested" in validation and "four" in validation:
        return f"{ns}:calibrated_temporal_4fold_nested_lofo"
    if "1:2:4:8" in validation and ("four" in validation or len(folds) == 4):
        return f"{ns}:calibrated_temporal_4fold_wcv_1_2_4_8"
    if ns == "team_a_current" and "canonical four s1 folds" in validation:
        return f"{ns}:calibrated_temporal_4fold_wcv_1_2_4_8"
    if ns == "team_a_current" and len(folds) == 4 and {"2025-07-24", "2025-08-21", "2025-09-18", "2025-10-16"}.issubset(set(folds)):
        return f"{ns}:temporal_4fold_equal_mean_0724_1016"
    if ns == "team_b_core":
        if "no local" in validation or "public lb only" in validation:
            return f"{ns}:public_lb_only_no_local_cv"
        if any(token in validation for token in ("folds1-2", "two decision", "same folds1-2")):
            return f"{ns}:two_fold_decision_mean_with_fold3_diagnostic"
    if ns == "team_b_alt":
        if "4-fold" in validation or "four-fold" in validation:
            return f"{ns}:four_fold_equal_weight_temporal"
        if "fold" in validation and any(token in validation for token in ("two", "2-fold")):
            return f"{ns}:two_fold_temporal_rmsle"
    if ns == "team_a_s2" and len(folds) == 4:
        return f"{ns}:structural_four_fold_rmsle"
    if "three" in validation and "fold" in validation:
        return f"{ns}:three_fold_protocol"
    if "single" in validation or "1-fold" in validation or len(folds) == 1:
        return f"{ns}:single_fold_protocol"
    if "leaderboard" in validation or validation.strip() == "public lb":
        return f"{ns}:public_leaderboard"
    if any(token in validation for token in ("no new", "not applicable", "not_applicable", "preflight", "blocked")):
        return f"{ns}:no_comparable_cv"
    # Conservative fallback: no two unclear protocols are pooled.
    return f"{ns}:{row['local_id'].lower()}:comparability_unconfirmed"


def outcome_bucket(status: str, delta: Any) -> str:
    lower = status.lower()
    if any(word in lower for word in ("blocked", "inconclusive", "unknown", "manifest", "baseline", "technical")):
        return "inconclusive"
    if any(word in lower for word in ("reject", "fail", "negative", "no_go", "no-go")):
        return "negative"
    if any(word in lower for word in ("accept", "pass", "promot", "positive", "continued", "created", "final")):
        return "positive"
    number = pure_float(delta)
    if number is not None:
        if number < 0:
            return "positive"
        if number > 0:
            return "negative"
    return "inconclusive"


def resolve_catalog(record: dict[str, Any], catalogs: list[dict[str, str]]) -> dict[str, str] | None:
    path = text_value(record.get("report_path", UNKNOWN)).replace("\\", "/")
    exact: list[dict[str, str]] = []
    for item in catalogs:
        candidates = {
            item.get("source_path", "").replace("\\", "/"),
            item.get("clean_evidence_path", "").replace("\\", "/"),
        }
        if path in candidates:
            exact.append(item)
    if len(exact) == 1:
        return exact[0]
    basename = Path(path).name.lower()
    by_name = [item for item in catalogs if Path(item.get("clean_evidence_path", "")).name.lower() == basename]
    if len(by_name) == 1:
        return by_name[0]
    return exact[0] if exact else None


def prefix_duplicate(value: Any, namespace: str) -> str:
    if is_unknown(value):
        return UNKNOWN
    text = text_value(value)
    match = re.search(r"EXP[-_ ]?(\d+)([A-Za-z]*)", text, flags=re.I)
    if match:
        target = f"EXP-{int(match.group(1)):03d}{match.group(2).upper()}"
        return f"{namespace}:{target}"
    return text


def merge_primary_records(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    primary = load_jsonl(root / "evidence" / "primary_reports_records.jsonl")
    branch = load_jsonl(root / "evidence" / "branch_reports_records.jsonl")
    catalogs = load_csv(root / "registry" / "report_catalog.csv")
    strategy = load_jsonl(root / "evidence" / "strategy_results_records.jsonl")
    machine_map = {global_id(str(item.get("id", UNKNOWN)), "team_a_current"): item for item in strategy}
    design_rows = load_jsonl(root / "evidence" / "experiment_design_fields.jsonl")
    design_map: dict[str, dict[str, Any]] = {}
    for item in design_rows:
        report_path = text_value(item.get("report_path", UNKNOWN))
        namespace = namespace_for_report(report_path)
        key = global_id(text_value(item.get("experiment_id", UNKNOWN)), namespace)
        design_map[key] = item

    rows: list[dict[str, Any]] = []
    for source in primary + branch:
        report_path = text_value(source.get("report_path", UNKNOWN))
        namespace = namespace_for_report(report_path)
        gid = global_id(text_value(source.get("experiment_id", source.get("id", UNKNOWN))), namespace)
        namespace, lid = gid.split(":", 1)
        catalog = resolve_catalog(source, catalogs)
        machine = machine_map.get(gid)
        design = design_map.get(gid, {})

        raw_cv = clean_value(source.get("cv_score", UNKNOWN))
        raw_delta = clean_value(source.get("delta_cv", UNKNOWN))
        raw_per_fold = clean_value(source.get("per_fold_cv", UNKNOWN))
        if machine:
            if pure_float(machine.get("cv")) is not None:
                raw_cv = machine["cv"]
            if pure_float(machine.get("delta")) is not None:
                raw_delta = machine["delta"]
            if machine.get("per_fold") not in (None, [], UNKNOWN):
                raw_per_fold = machine["per_fold"]

        name = text_value(source.get("canonical_name", source.get("name", UNKNOWN)))
        change = text_value(source.get("change", UNKNOWN))
        family_raw = text_value(source.get("family", UNKNOWN))
        family = normalize_family(family_raw, name, change)
        tags, raw_tags = normalize_tags(source.get("compatible_tags", source.get("tags", UNKNOWN)), family, change)

        artifacts: list[Any] = as_list(source.get("artifacts", UNKNOWN))
        if machine:
            artifacts.extend(as_list(machine.get("artifacts", UNKNOWN)))
            artifacts.extend(as_list(machine.get("evidence", UNKNOWN)))
        artifact_values = sorted({text_value(item) for item in artifacts if not is_unknown(item)}) or [UNKNOWN]

        conflicts: list[str] = []
        for candidate in as_list(source.get("conflicts", UNKNOWN)) + as_list(machine.get("conflicts", UNKNOWN) if machine else UNKNOWN):
            candidate_text = text_value(candidate)
            if candidate_text.lower() not in {"none", "none found", "unknown", "no conflict found"}:
                conflicts.append(candidate_text)

        facts: dict[str, Any] = {"primary_report": clean_value(source.get("facts", UNKNOWN))}
        interpretation: dict[str, Any] = {"primary_report": clean_value(source.get("interpretation", UNKNOWN))}
        if machine:
            facts["machine_audit"] = clean_value(machine.get("facts", UNKNOWN))
            interpretation["machine_audit"] = clean_value(machine.get("interpretation", UNKNOWN))

        evidence_strength = text_value(source.get("evidence_strength", UNKNOWN))
        if machine:
            evidence_strength = "machine_artifact_or_run_metric_plus_primary_report"

        duplicate = prefix_duplicate(source.get("duplicate_of", UNKNOWN), namespace)
        if machine and not is_unknown(machine.get("duplicate_of", UNKNOWN)):
            duplicate = prefix_duplicate(machine["duplicate_of"], namespace)

        row: dict[str, Any] = {
            "experiment_id": gid,
            "namespace": namespace,
            "local_id": lid,
            "canonical_name": name,
            "family": family,
            "source_family": family_raw,
            "date": clean_value(source.get("date", catalog.get("date_reported", UNKNOWN) if catalog else UNKNOWN)),
            "parent_baseline": clean_value(source.get("baseline", source.get("parent", UNKNOWN))),
            "change": change,
            "train_construction": clean_value(design.get("train_construction", source.get("train_construction", UNKNOWN))),
            "features": clean_value(design.get("features", source.get("features", UNKNOWN))),
            "target": clean_value(design.get("target", source.get("target", UNKNOWN))),
            "model_family": clean_value(source.get("model_family", source.get("model", UNKNOWN))),
            "validation_protocol": clean_value(source.get("validation_protocol", source.get("validation", UNKNOWN))),
            "comparison_class": UNKNOWN,
            "folds": clean_value(design.get("folds", source.get("folds", UNKNOWN))) if not is_unknown(design.get("folds", UNKNOWN)) else clean_value(source.get("folds", UNKNOWN)),
            "seeds": clean_value(design.get("seeds", source.get("seeds", UNKNOWN))) if not is_unknown(design.get("seeds", UNKNOWN)) else clean_value(source.get("seeds", UNKNOWN)),
            "hyperparameters": clean_value(design.get("hyperparameters", source.get("hyperparameters", UNKNOWN))) if not is_unknown(design.get("hyperparameters", UNKNOWN)) else clean_value(source.get("hyperparameters", UNKNOWN)),
            "cv_score": raw_cv,
            "cv_score_numeric": pure_float(raw_cv) if pure_float(raw_cv) is not None else UNKNOWN,
            "cv_score_report": clean_value(source.get("cv_score", UNKNOWN)),
            "per_fold_cv": raw_per_fold,
            "delta_cv": raw_delta,
            "folds_positive": clean_value(source.get("folds_positive", UNKNOWN)),
            "folds_total": clean_value(source.get("folds_total", UNKNOWN)),
            "lb_score": clean_value(source.get("lb_score", UNKNOWN)),
            "submission": clean_value(source.get("submission", UNKNOWN)),
            "runtime": clean_value(source.get("runtime", UNKNOWN)),
            "status": clean_value(source.get("status", UNKNOWN)),
            "outcome_bucket": UNKNOWN,
            "evidence_strength": evidence_strength,
            "artifacts": artifact_values,
            "duplicate_of": duplicate,
            "relation_type": "duplicate_or_rerun" if duplicate != UNKNOWN else "canonical",
            "compatible_tags": tags,
            "raw_compatible_tags": raw_tags or [UNKNOWN],
            "notes": clean_value(source.get("confounders", UNKNOWN)),
            "reproducibility_status": clean_value(source.get("reproducibility_status", UNKNOWN)),
            "facts": facts,
            "interpretation": interpretation,
            "confounders": clean_value(source.get("confounders", UNKNOWN)),
            "conflicts": conflicts or [UNKNOWN],
            "source_report": catalog.get("clean_evidence_path", report_path) if catalog else report_path,
            "source_origin": catalog.get("source_ref", UNKNOWN) if catalog else UNKNOWN,
            "report_sha256": catalog.get("sha256", UNKNOWN) if catalog else UNKNOWN,
            "machine_audit_id": text_value(machine.get("id", UNKNOWN)) if machine else UNKNOWN,
        }
        row["comparison_class"] = comparison_class(row, machine)
        row["outcome_bucket"] = outcome_bucket(text_value(row["status"]), row["delta_cv"])
        rows.append(row)

    # Curated relations where reports describe actual reruns rather than a new hypothesis.
    curated = {
        "team_a_current:EXP-030B": ("team_a_current:EXP-030", "seed_rerun"),
        "team_a_current:EXP-030C": ("team_a_current:EXP-030", "multiseed_rerun"),
        "team_b_core:EXP-031": ("team_b_core:EXP-005", "exact_replay"),
    }
    by_id = {row["experiment_id"]: row for row in rows}
    for child, (parent, relation) in curated.items():
        if child in by_id and parent in by_id:
            by_id[child]["duplicate_of"] = parent
            by_id[child]["relation_type"] = relation

    if "team_a_current:EXP-032-MANIFEST" in by_id:
        by_id["team_a_current:EXP-032-MANIFEST"]["relation_type"] = "duplicate_document_for_same_experiment"
    if "team_a_current:EXP-051" in by_id and by_id["team_a_current:EXP-051"]["duplicate_of"] != UNKNOWN:
        by_id["team_a_current:EXP-051"]["relation_type"] = "numerical_oof_replay_with_distinct_production_artifacts"

    rows.sort(key=lambda row: (text_value(row["date"]), row["namespace"], row["local_id"], row["source_report"]))
    return rows, strategy


def teammate_run_units(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    evidence = load_jsonl(root / "evidence" / "teammate_records.jsonl")
    selected = [row for row in evidence if row.get("record_type") in {"review_run", "review_training_unit"}]
    dates = {text_value(row.get("experiment_id", UNKNOWN)): clean_value(row.get("date", UNKNOWN)) for row in selected}
    result: list[dict[str, Any]] = []
    for source in selected:
        raw_id = text_value(source.get("experiment_id", UNKNOWN))
        parent = clean_value(source.get("parent_baseline", source.get("parent_run", UNKNOWN)))
        date = clean_value(source.get("date", dates.get(text_value(parent), UNKNOWN)))
        name = text_value(source.get("canonical_name", raw_id))
        change = text_value(source.get("change", "separately logged training unit inside teammate review run"))
        source_family = text_value(source.get("family", source.get("model_family", "teammate review training")))
        family = normalize_family(source_family, name, change)
        tags, raw_tags = normalize_tags(source.get("compatible_tags", UNKNOWN), family, change)
        runtime: Any = UNKNOWN
        runtime_parts = {key: value for key, value in source.items() if key.startswith("runtime_")}
        if runtime_parts:
            runtime = runtime_parts
        artifacts = [text_value(item) for item in as_list(source.get("artifacts", UNKNOWN))] or [UNKNOWN]
        gid = f"teammate_review:{raw_id}"
        row: dict[str, Any] = {
            "experiment_id": gid,
            "namespace": "teammate_review",
            "local_id": raw_id,
            "canonical_name": name,
            "family": family,
            "source_family": source_family,
            "date": date,
            "parent_baseline": parent,
            "change": change,
            "train_construction": UNKNOWN,
            "features": UNKNOWN,
            "target": UNKNOWN,
            "model_family": clean_value(source.get("model_family", UNKNOWN)),
            "validation_protocol": clean_value(source.get("validation_protocol", UNKNOWN)),
            "comparison_class": "teammate_review:walk_forward_4fold_wcv_1_2_4_8" if source.get("record_type") == "review_run" else "teammate_review:training_unit_no_standalone_metric",
            "folds": clean_value(source.get("folds_or_test", UNKNOWN)),
            "seeds": UNKNOWN,
            "hyperparameters": UNKNOWN,
            "cv_score": UNKNOWN,
            "cv_score_numeric": UNKNOWN,
            "cv_score_report": UNKNOWN,
            "per_fold_cv": UNKNOWN,
            "delta_cv": UNKNOWN,
            "folds_positive": UNKNOWN,
            "folds_total": 4 if "four" in text_value(source.get("validation_protocol", UNKNOWN)).lower() else UNKNOWN,
            "lb_score": UNKNOWN,
            "submission": clean_value(source.get("materialized_submissions", UNKNOWN)),
            "runtime": runtime,
            "status": clean_value(source.get("status", UNKNOWN)),
            "outcome_bucket": "inconclusive",
            "evidence_strength": "run_manifest_and_runtime_artifact",
            "artifacts": artifacts,
            "duplicate_of": UNKNOWN,
            "relation_type": "canonical_artifact_derived_run_unit",
            "compatible_tags": tags,
            "raw_compatible_tags": raw_tags or [UNKNOWN],
            "notes": clean_value(source.get("notes", UNKNOWN)),
            "reproducibility_status": "archive_extraction_exact" if source.get("archive_extraction_exact") is True else "runtime_record_present",
            "facts": {"machine_manifest": source},
            "interpretation": {"forensic_interpretation": "Completed run unit; no public LB result is bound to this unit."},
            "confounders": clean_value(source.get("notes", "No standalone score is reported for this run unit.")),
            "conflicts": [UNKNOWN],
            "source_report": f"evidence/teammate_records.jsonl#{raw_id}",
            "source_origin": "teammate package manifest/runtime evidence",
            "report_sha256": UNKNOWN,
            "machine_audit_id": raw_id,
        }
        result.append(row)
    result.sort(key=lambda row: (text_value(row["date"]), row["experiment_id"]))

    package_links: list[dict[str, Any]] = []
    for source in evidence:
        if source.get("record_type") != "package_experiment":
            continue
        raw_id = text_value(source.get("experiment_id", UNKNOWN))
        match = re.search(r"EXP[-_ ]?(\d+)([A-Za-z]*)", raw_id, flags=re.I)
        canonical = UNKNOWN
        if match:
            canonical = f"team_a_current:EXP-{int(match.group(1)):03d}{match.group(2).upper()}"
        package_links.append({
            "teammate_package_record": raw_id,
            "canonical_experiment": canonical,
            "relation_type": "packaged_copy_or_provenance_bundle_of_existing_experiment",
            "artifacts": clean_value(source.get("artifacts", UNKNOWN)),
            "lb_score_recorded": clean_value(source.get("lb_score", UNKNOWN)),
            "evidence_strength": clean_value(source.get("evidence_strength", UNKNOWN)),
            "central_registry_row_added": "no; canonical experiment row already exists",
        })
    return result, package_links


def machine_only_strategy_units(strategy: list[dict[str, Any]], report_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    represented = {row["machine_audit_id"] for row in report_rows if not is_unknown(row["machine_audit_id"])}
    result: list[dict[str, Any]] = []
    for source in strategy:
        raw_id = text_value(source.get("id", UNKNOWN))
        if raw_id in represented:
            continue
        name = text_value(source.get("name", raw_id))
        change = text_value(source.get("change", UNKNOWN))
        source_family = text_value(source.get("family", UNKNOWN))
        family = normalize_family(source_family, name, change)
        tags, raw_tags = normalize_tags(source.get("tags", UNKNOWN), family, change)
        facts = clean_value(source.get("facts", UNKNOWN))
        folds_positive = facts.get("folds_positive", UNKNOWN) if isinstance(facts, dict) else UNKNOWN
        artifacts = sorted({text_value(item) for item in as_list(source.get("artifacts", UNKNOWN)) + as_list(source.get("evidence", UNKNOWN)) if not is_unknown(item)}) or [UNKNOWN]
        gid = f"team_a_current:{local_id(raw_id)}"
        row: dict[str, Any] = {
            "experiment_id": gid,
            "namespace": "team_a_current",
            "local_id": local_id(raw_id),
            "canonical_name": name,
            "family": family,
            "source_family": source_family,
            "date": clean_value(source.get("date", UNKNOWN)),
            "parent_baseline": clean_value(source.get("parent", UNKNOWN)),
            "change": change,
            "train_construction": UNKNOWN,
            "features": UNKNOWN,
            "target": UNKNOWN,
            "model_family": clean_value(source.get("model", UNKNOWN)),
            "validation_protocol": clean_value(source.get("validation", UNKNOWN)),
            "comparison_class": UNKNOWN,
            "folds": UNKNOWN,
            "seeds": UNKNOWN,
            "hyperparameters": UNKNOWN,
            "cv_score": clean_value(source.get("cv", UNKNOWN)),
            "cv_score_numeric": pure_float(source.get("cv")) if pure_float(source.get("cv")) is not None else UNKNOWN,
            "cv_score_report": UNKNOWN,
            "per_fold_cv": clean_value(source.get("per_fold", UNKNOWN)),
            "delta_cv": clean_value(source.get("delta", UNKNOWN)),
            "folds_positive": folds_positive,
            "folds_total": len(source.get("per_fold", [])) if isinstance(source.get("per_fold"), list) else UNKNOWN,
            "lb_score": clean_value(source.get("lb", UNKNOWN)),
            "submission": UNKNOWN,
            "runtime": clean_value(source.get("runtime", UNKNOWN)),
            "status": clean_value(source.get("status", UNKNOWN)),
            "outcome_bucket": UNKNOWN,
            "evidence_strength": "machine_artifact_and_checksum_reconstruction",
            "artifacts": artifacts,
            "duplicate_of": prefix_duplicate(source.get("duplicate_of", UNKNOWN), "team_a_current"),
            "relation_type": "canonical_machine_only_experiment",
            "compatible_tags": tags,
            "raw_compatible_tags": raw_tags or [UNKNOWN],
            "notes": UNKNOWN,
            "reproducibility_status": "machine_results_preserved; original blend manifest missing where stated",
            "facts": {"machine_audit": facts},
            "interpretation": {"machine_audit": clean_value(source.get("interpretation", UNKNOWN))},
            "confounders": UNKNOWN,
            "conflicts": as_list(source.get("conflicts", UNKNOWN)) or [UNKNOWN],
            "source_report": f"evidence/strategy_results_records.jsonl#{raw_id}",
            "source_origin": "machine-only strategy audit",
            "report_sha256": UNKNOWN,
            "machine_audit_id": raw_id,
        }
        row["comparison_class"] = comparison_class(row, source)
        row["outcome_bucket"] = outcome_bucket(text_value(row["status"]), row["delta_cv"])
        result.append(row)
    return result


def build_dedup(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {row["experiment_id"]: row for row in rows}
    clusters = [
        ("D001", "team_a_current:EXP-032", ["team_a_current:EXP-032-MANIFEST"], "duplicate_document_for_same_experiment", "yes", "two reports describe one experiment unit"),
        ("D002", "team_a_current:EXP-030", ["team_a_current:EXP-030B", "team_a_current:EXP-030C"], "seed_and_multiseed_rerun_series", "yes", "runs are retained; novelty is the same depth-curriculum hypothesis"),
        ("D003", "team_b_core:EXP-005", ["team_b_core:EXP-031"], "exact_semantic_replay", "yes", "same decision mean and delta under the same protocol"),
        ("D004", "team_a_current:EXP-020", ["SAMPLE-BASELINE-B-AVG3", "SAMPLE-TB1-AVG3"], "array_identical_artifact_aliases", "no", "NPZ package hashes differ; aligned arrays are identical"),
        ("D005", "team_a_current:EXP-029", ["team_a_current:EXP-030"], "shared_array_identical_V1016_baseline", "no", "baseline artifact is shared; hypotheses differ"),
        ("D006", "team_a_current:EXP-032", ["team_a_current:EXP-032B"], "bitwise_reuse_of_conditional_predictions", "no", "EXP032B changes only the extensive activity composition"),
        ("D007", "team_a_current:EXP-038", ["FNL-BASE-R2-S42"], "stochastic_same_seed_control_rerun", "no", "observed execution-noise control, not a new hypothesis"),
        ("D008", "team_a_current:EXP-043", ["DET-PAIR-RUN2"], "exact_internal_replay", "no", "predictions, optimizer/model/RNG snapshots and hashes match run1"),
        ("D009", "team_a_current:EXP-047", ["team_a_current:EXP-051"], "numerical_oof_replay_with_distinct_production", "no", "production optimizer and test artifacts changed"),
        ("D010", "team_a_current:EXP-067", ["AUTHORITATIVE-LATEST-V1"], "partial_duplicate_directory", "no", "V2 is canonical; some files identical and some audit outputs changed"),
        ("D011", "team_a_current:EXP-061", ["team_a_current:EXP-062", "team_a_current:EXP-064", "team_a_current:EXP-052", "team_a_current:EXP-054", "team_a_current:EXP-055"], "baseline_identical_zero_correction_outputs", "no", "distinct hypotheses terminate at a zero correction on different scopes"),
    ]
    result: list[dict[str, Any]] = []
    for cluster_id, canonical, related, relation, collapse, notes in clusters:
        evidence: list[str] = []
        for experiment_id in [canonical] + related:
            if experiment_id in by_id:
                evidence.append(text_value(by_id[experiment_id]["source_report"]))
        result.append({
            "cluster_id": cluster_id,
            "canonical_experiment": canonical,
            "related_experiment_or_artifact": related,
            "relation_type": relation,
            "collapse_for_unique_hypothesis_count": collapse,
            "evidence": sorted(set(evidence)) or ["evidence/strategy_results_audit.md"],
            "information_preserved": "yes",
            "notes": notes,
        })
    return result


def build_id_collisions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["local_id"]].append(row)
    result: list[dict[str, Any]] = []
    for lid, members in sorted(groups.items()):
        namespaces = sorted({row["namespace"] for row in members})
        if len(namespaces) < 2:
            continue
        result.append({
            "local_id": lid,
            "namespaces": namespaces,
            "global_ids": [row["experiment_id"] for row in members],
            "canonical_names": [row["canonical_name"] for row in members],
            "resolution": "globally namespaced; no cross-branch identity inferred",
        })
    return result


def build_baselines(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        parent = text_value(row["parent_baseline"])
        if not is_unknown(parent) and parent.lower() not in {"none", "not_applicable", "not applicable"}:
            grouped[(row["namespace"], parent)].append(row)
        if "baseline" in text_value(row["status"]).lower() or "baseline" in row["canonical_name"].lower():
            grouped[(row["namespace"], row["experiment_id"])].append(row)

    result: list[dict[str, Any]] = []
    for (namespace, baseline), refs in grouped.items():
        refs.sort(key=lambda item: (text_value(item["date"]), item["experiment_id"]))
        first = refs[0]
        baseline_score: Any = UNKNOWN
        if baseline == first["experiment_id"]:
            baseline_score = first["cv_score"]
        result.append({
            "date_first_seen": first["date"],
            "research_line": namespace,
            "baseline_id_or_recipe": baseline,
            "introduced_or_first_referenced_by": first["experiment_id"],
            "validation_protocol": first["validation_protocol"],
            "comparison_class": first["comparison_class"],
            "score_if_directly_measured": baseline_score,
            "referenced_by": sorted({row["experiment_id"] for row in refs}),
            "evidence": sorted({row["source_report"] for row in refs}),
            "notes": "Chronology is within this research line; cross-line baselines are not equated by name.",
        })
    result.sort(key=lambda row: (text_value(row["date_first_seen"]), row["research_line"], row["baseline_id_or_recipe"]))
    return result


def build_ancestry(rows: list[dict[str, Any]], lb_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    by_ns_local = {(row["namespace"], row["local_id"]): row["experiment_id"] for row in rows}
    for row in rows:
        parent = text_value(row["parent_baseline"])
        if not is_unknown(parent) and parent.lower() not in {"none", "not_applicable", "not applicable"}:
            matches = re.findall(r"EXP[-_ ]?(\d+)([A-Za-z]*)", parent, flags=re.I)
            resolved: list[str] = []
            for number, suffix in matches:
                lid = f"EXP-{int(number):03d}{suffix.upper()}"
                if (row["namespace"], lid) in by_ns_local:
                    resolved.append(by_ns_local[(row["namespace"], lid)])
            if not resolved:
                resolved = [f"baseline_recipe:{row['namespace']}:{slug(parent)[:80]}"]
            for source in sorted(set(resolved)):
                edges.append({
                    "source": source,
                    "target": row["experiment_id"],
                    "edge_type": "baseline_to_change",
                    "recipe_or_change": row["change"],
                    "evidence": row["source_report"],
                    "confidence": row["evidence_strength"],
                })
        if not is_unknown(row["duplicate_of"]):
            edges.append({
                "source": row["duplicate_of"],
                "target": row["experiment_id"],
                "edge_type": row["relation_type"],
                "recipe_or_change": row["change"],
                "evidence": row["source_report"],
                "confidence": row["evidence_strength"],
            })
        if not is_unknown(row["submission"]):
            for submission in as_list(row["submission"]):
                if is_unknown(submission) or text_value(submission).lower() == "none":
                    continue
                edges.append({
                    "source": row["experiment_id"],
                    "target": f"submission:{Path(text_value(submission)).name}",
                    "edge_type": "produced_or_described_submission",
                    "recipe_or_change": UNKNOWN,
                    "evidence": row["source_report"],
                    "confidence": row["evidence_strength"],
                })

    for lb in lb_rows:
        target = f"submission:{lb.get('filename', Path(text_value(lb.get('submission_path', UNKNOWN))).name)}"
        lineage = as_list(lb.get("lineage", UNKNOWN))
        for item in lineage:
            source_text = text_value(item)
            match = re.fullmatch(r"exp[_-]?(\d+)([a-z]*)", source_text, flags=re.I)
            if match:
                lid = f"EXP-{int(match.group(1)):03d}{match.group(2).upper()}"
                source = by_ns_local.get(("team_a_current", lid), f"lineage_label:{source_text}")
            else:
                source = f"lineage_label:{source_text}"
            edges.append({
                "source": source,
                "target": target,
                "edge_type": "component_or_lineage_to_confirmed_submission",
                "recipe_or_change": clean_value(lb.get("recipe", UNKNOWN)),
                "evidence": clean_value(lb.get("evidence", UNKNOWN)),
                "confidence": clean_value(lb.get("evidence_strength", UNKNOWN)),
            })

    unique: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for edge in edges:
        key = (edge["source"], edge["target"], edge["edge_type"], text_value(edge["recipe_or_change"]))
        unique[key] = edge
    return sorted(unique.values(), key=lambda row: (row["source"], row["target"], row["edge_type"]))


def build_leaderboard(root: Path, lb_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    audit_by_name = {row.get("filename", ""): row for row in load_csv(root / "submissions" / "audit.csv")}
    submission_registry = {row.get("filename", ""): row for row in load_csv(root / "submissions" / "registry.csv")}
    result: list[dict[str, Any]] = []
    for item in lb_rows:
        filename = text_value(item.get("filename", Path(text_value(item.get("submission_path", UNKNOWN))).name))
        audit = audit_by_name.get(filename, {})
        registered = submission_registry.get(filename, {})
        recipe = clean_value(item.get("recipe", audit.get("recipe", registered.get("recipe", UNKNOWN))))
        if is_unknown(recipe):
            recipe = clean_value(registered.get("recipe", UNKNOWN))
        source_predictions = clean_value(registered.get("source_predictions", audit.get("source_predictions", UNKNOWN)))
        result.append({
            "date": clean_value(item.get("date", UNKNOWN)),
            "filename": filename,
            "score": clean_value(item.get("score_public", UNKNOWN)),
            "recipe": recipe,
            "source_predictions": source_predictions,
            "experiment_lineage": clean_value(item.get("lineage", UNKNOWN)),
            "artifact_path": clean_value(item.get("submission_path", UNKNOWN)),
            "artifact_sha256": clean_value(item.get("sha256", UNKNOWN)),
            "artifact_exists": "yes",
            "evidence_strength": clean_value(item.get("evidence_strength", UNKNOWN)),
            "external_platform_export_present": bool(item.get("external_platform_export_present", False)),
            "verification_scope": "repository-internal score-to-existing-file link; not independent platform confirmation",
            "notes": clean_value(item.get("note", UNKNOWN)),
        })
    result.sort(key=lambda row: (text_value(row["date"]), text_value(row["filename"])))
    return result


def build_unverified_lb_claims(root: Path, rows: list[dict[str, Any]], leaderboard: list[dict[str, Any]]) -> list[dict[str, Any]]:
    verified_names = {text_value(row["filename"]).lower() for row in leaderboard}
    audit = load_csv(root / "submissions" / "audit.csv")
    existing_names = {row.get("filename", "").lower() for row in audit if row.get("exists") == "yes"}
    result: list[dict[str, Any]] = []
    for row in rows:
        if is_unknown(row["lb_score"]):
            continue
        submission_text = text_value(row["submission"])
        csv_names = sorted({Path(name.replace("\\", "/")).name for name in re.findall(r"[^\s;,\[\]\"]+\.csv", submission_text, flags=re.I)})
        if any(name.lower() in verified_names for name in csv_names):
            continue
        existing = [name for name in csv_names if name.lower() in existing_names]
        missing = [name for name in csv_names if name.lower() not in existing_names]
        if len(existing) > 1:
            reason = "score is ambiguous across multiple existing artifacts"
        elif existing:
            reason = "artifact exists, but the score event is not SHA-bound to an upload/result record"
        elif csv_names:
            reason = "named submission artifact is absent from the available repositories/worktrees"
        else:
            reason = "no exact submission artifact is bound to the score claim"
        result.append({
            "experiment_id": row["experiment_id"],
            "score_claim": row["lb_score"],
            "submission_claim": row["submission"],
            "existing_candidate_artifacts": existing or [UNKNOWN],
            "missing_candidate_artifacts": missing or [UNKNOWN],
            "reason_not_confirmed": reason,
            "source": row["source_report"],
            "status": "unverified_repository_claim",
            "external_platform_export_present": "no",
        })

    teammate = load_jsonl(root / "evidence" / "teammate_records.jsonl")
    lb_summary = next((row for row in teammate if row.get("record_type") == "leaderboard_evidence_summary"), None)
    if lb_summary:
        existing_keys = {(text_value(row["score_claim"]), text_value(row["experiment_id"])) for row in result}
        for claim in as_list(lb_summary.get("unverified_report_or_script_claims", UNKNOWN)):
            if not isinstance(claim, dict):
                continue
            label = text_value(claim.get("label", UNKNOWN))
            score = clean_value(claim.get("score", UNKNOWN))
            # latest and known_ridge are already represented by EXP067/EXP068.
            if label in {"latest", "known_ridge_submission_public"}:
                continue
            candidate = {
                "experiment_id": f"teammate_context:{label}",
                "score_claim": score,
                "submission_claim": UNKNOWN,
                "existing_candidate_artifacts": [UNKNOWN],
                "missing_candidate_artifacts": [label],
                "reason_not_confirmed": clean_value(claim.get("reason", UNKNOWN)),
                "source": "evidence/teammate_records.jsonl#TM-LB-EVIDENCE",
                "status": "unverified_context_constant_not_promoted_to_experiment",
                "external_platform_export_present": "no",
            }
            key = (text_value(candidate["score_claim"]), text_value(candidate["experiment_id"]))
            if key not in existing_keys:
                result.append(candidate)
    result.sort(key=lambda row: (text_value(row["experiment_id"]), text_value(row["score_claim"])))
    return result


def build_families(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["family"]].append(row)
    result: list[dict[str, Any]] = []
    for family, members in sorted(grouped.items()):
        by_class: dict[str, list[tuple[float, str]]] = defaultdict(list)
        runtimes: list[str] = []
        for row in members:
            score = pure_float(row["cv_score_numeric"])
            if score is not None and "no_comparable_cv" not in row["comparison_class"] and "comparability_unconfirmed" not in row["comparison_class"]:
                by_class[row["comparison_class"]].append((score, row["experiment_id"]))
            if not is_unknown(row["runtime"]):
                runtimes.append(text_value(row["runtime"]))
        best: dict[str, Any] = {}
        median: dict[str, Any] = {}
        for cls, values in sorted(by_class.items()):
            best_score, best_id = min(values)
            best[cls] = {"score": best_score, "experiment_id": best_id, "n": len(values)}
            median[cls] = statistics.median(score for score, _ in values)
        buckets = Counter(row["outcome_bucket"] for row in members)
        negative = buckets["negative"]
        positive = buckets["positive"]
        if negative >= 5 and positive == 0:
            saturation = "multiple_negative_implementations; does_not_exclude_untried_family_variants"
        elif negative >= 3 and positive <= 1:
            saturation = "several_negative_results; evidence_is_implementation_and_protocol_specific"
        else:
            saturation = "mixed_or_insufficient_evidence"
        result.append({
            "family": family,
            "experiment_count": len(members),
            "positive": positive,
            "negative": negative,
            "inconclusive": buckets["inconclusive"],
            "scored_comparable_units": sum(len(values) for values in by_class.values()),
            "best_results_by_comparison_class": best or UNKNOWN,
            "median_results_by_comparison_class": median or UNKNOWN,
            "runtime_evidence": runtimes or [UNKNOWN],
            "saturation_evidence": saturation,
            "experiment_ids": [row["experiment_id"] for row in members],
        })
    return result


def conflict_category(statement: str) -> str:
    lower = statement.lower()
    if any(token in lower for token in ("no sha-bound", "missing", "absent", "unavailable", "not recorded", "not preserved", "could not be", "cannot be byte", "blocked")):
        return "missing_evidence_or_provenance_gap"
    if any(token in lower for token in ("not directly comparable", "must not be conflated", "distinct metrics", "different metrics", "would be invalid", "cannot causally isolate", "inference diagnostics")):
        return "comparability_warning"
    if any(token in lower for token in ("stale", "supersed", "mismatch", "contradict", "ambiguous", "cannot be bound", "coexist", "versus", " vs ", "differs", "different result", "override")):
        return "contradiction_or_documented_tension"
    return "audit_caveat"


def build_contradictions(rows: list[dict[str, Any]], id_collisions: list[dict[str, Any]], root: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    counter = 1
    for row in rows:
        for conflict in as_list(row["conflicts"]):
            if is_unknown(conflict):
                continue
            if text_value(conflict).lower().startswith(("none;", "none numeric", "none found", "none:")):
                continue
            result.append({
                "contradiction_id": f"C{counter:04d}",
                "experiment_id": row["experiment_id"],
                "category": conflict_category(text_value(conflict)),
                "statement": conflict,
                "source_a": row["source_report"],
                "source_b": row["artifacts"],
                "resolution": "retained without guessing; see experiment FACTS and INTERPRETATION",
                "impact": "metric, verdict, provenance, or reproducibility must be read with the stated caveat",
            })
            counter += 1

    for collision in id_collisions:
        if collision["local_id"] not in {"EXP-057", "EXP-058"}:
            continue
        result.append({
            "contradiction_id": f"C{counter:04d}",
            "experiment_id": collision["global_ids"],
            "category": "unrelated_experiments_share_local_id",
            "statement": f"{collision['local_id']} names unrelated experiments in parallel namespaces: " + "; ".join(collision["canonical_names"]),
            "source_a": collision["namespaces"][0],
            "source_b": collision["namespaces"][1:],
            "resolution": "global namespace prefix is mandatory",
            "impact": "bare local ID is ambiguous",
        })
        counter += 1

    missing = load_csv(root / "contradictions" / "manifested_submissions_missing.csv")
    for row in missing:
        result.append({
            "contradiction_id": f"C{counter:04d}",
            "experiment_id": row.get("experiment_id", UNKNOWN),
            "category": "manifested_submission_artifact_missing",
            "statement": row.get("filename", row.get("original_path", UNKNOWN)),
            "source_a": row.get("source", row.get("manifest_source", UNKNOWN)),
            "source_b": "filesystem inventory",
            "resolution": "file remains marked missing; no substitute inferred",
            "impact": "submission recipe cannot be byte-verified from this checkout",
        })
        counter += 1

    teammate = load_jsonl(root / "evidence" / "teammate_records.jsonl")
    for row in teammate:
        if row.get("record_type") not in {"contradiction", "orphan"}:
            continue
        result.append({
            "contradiction_id": f"C{counter:04d}",
            "experiment_id": clean_value(row.get("experiment_id", UNKNOWN)),
            "category": clean_value(row.get("kind", row.get("record_type", UNKNOWN))),
            "statement": clean_value(row.get("facts", UNKNOWN)),
            "source_a": "evidence/teammate_records.jsonl",
            "source_b": "artifact/manifests inventory",
            "resolution": clean_value(row.get("status", "retained without guessing")),
            "impact": "see teammate evidence audit",
        })
        counter += 1

    secondary = load_jsonl(root / "evidence" / "secondary_summary_conflicts.jsonl")
    for row in secondary:
        result.append({
            "contradiction_id": f"C{counter:04d}",
            "experiment_id": clean_value(row.get("canonical_experiment_id", UNKNOWN)),
            "category": clean_value(row.get("conflict_type", "secondary_summary_conflict")),
            "statement": clean_value(row.get("claim_as_secondary", UNKNOWN)),
            "source_a": clean_value(row.get("summary_path", UNKNOWN)),
            "source_b": clean_value(row.get("primary_or_machine_evidence", UNKNOWN)),
            "resolution": clean_value(row.get("neutral_resolution", "secondary claim not promoted to fact")),
            "impact": f"secondary-only; severity={clean_value(row.get('severity', UNKNOWN))}; used_for_facts=no",
        })
        counter += 1
    return result


def slug(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9а-яё]+", "-", value, flags=re.I)
    return value.strip("-") or "unknown"


def md_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return "```json\n" + json.dumps(value, ensure_ascii=False, indent=2) + "\n```"
    return text_value(value)


def write_experiment_cards(root: Path, rows: list[dict[str, Any]]) -> None:
    out = root / "experiments" / "normalized"
    out.mkdir(parents=True, exist_ok=True)
    expected: set[str] = set()
    for row in rows:
        filename = slug(row["experiment_id"]) + ".md"
        expected.add(filename)
        content = [
            f"# {row['experiment_id']} — {row['canonical_name']}",
            "",
            f"- Family: `{row['family']}`",
            f"- Date: `{text_value(row['date'])}`",
            f"- Parent baseline: {text_value(row['parent_baseline'])}",
            f"- Change: {text_value(row['change'])}",
            f"- Model: {text_value(row['model_family'])}",
            f"- Validation: {text_value(row['validation_protocol'])}",
            f"- Comparison class: `{row['comparison_class']}`",
            f"- CV: {text_value(row['cv_score'])}",
            f"- Delta CV: {text_value(row['delta_cv'])}",
            f"- Public LB: {text_value(row['lb_score'])}",
            f"- Status: `{text_value(row['status'])}`",
            f"- Evidence strength: `{text_value(row['evidence_strength'])}`",
            f"- Duplicate/rerun of: {text_value(row['duplicate_of'])}",
            "",
            "## FACTS",
            "",
            md_value(row["facts"]),
            "",
            "## INTERPRETATION",
            "",
            md_value(row["interpretation"]),
            "",
            "## Confounders and conflicts",
            "",
            f"Confounders: {text_value(row['confounders'])}",
            "",
            md_value(row["conflicts"]),
            "",
            "## Reproducibility and evidence",
            "",
            f"- Reproducibility: {text_value(row['reproducibility_status'])}",
            f"- Source report snapshot: `{row['source_report']}`",
            f"- Report SHA-256: `{row['report_sha256']}`",
            f"- Artifacts: {md_value(row['artifacts'])}",
            f"- Compatibility tags: {', '.join(row['compatible_tags'])}",
            "",
        ]
        (out / filename).write_text("\n".join(content), encoding="utf-8")
    # Only remove stale generated cards, never source-report snapshots.
    for path in out.glob("*.md"):
        if path.name not in expected:
            path.unlink()


def write_family_docs(root: Path, families: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    out = root / "families"
    out.mkdir(parents=True, exist_ok=True)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)
    expected = {slug(item["family"]) + ".md" for item in families}
    for summary in families:
        family = summary["family"]
        lines = [
            f"# {family}",
            "",
            "This page is descriptive research memory, not a recommendation for future work.",
            "",
            f"- Experiments: {summary['experiment_count']}",
            f"- Positive / negative / inconclusive: {summary['positive']} / {summary['negative']} / {summary['inconclusive']}",
            f"- Saturation evidence: {summary['saturation_evidence']}",
            "",
            "## Comparable score groups",
            "",
            md_value({
                "best": summary["best_results_by_comparison_class"],
                "median": summary["median_results_by_comparison_class"],
            }),
            "",
            "## Experiments",
            "",
            "| Experiment | Date | Status | CV | Comparison class |",
            "|---|---|---|---:|---|",
        ]
        for row in sorted(by_family[family], key=lambda item: (text_value(item["date"]), item["experiment_id"])):
            lines.append(f"| {row['experiment_id']} | {text_value(row['date'])} | {text_value(row['status'])} | {text_value(row['cv_score'])} | `{row['comparison_class']}` |")
        lines.append("")
        (out / (slug(family) + ".md")).write_text("\n".join(lines), encoding="utf-8")
    for path in out.glob("*.md"):
        if path.name not in expected and path.name.lower() != "readme.md":
            path.unlink()


def write_baseline_doc(root: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Baseline chronology",
        "",
        "Chronology is separated by research line. Repeated textual baselines are retained because a name alone does not prove identical folds, train coverage, or prediction sources.",
        "",
        "| First seen | Research line | Baseline or recipe | First reference | Comparison class |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(f"| {text_value(row['date_first_seen'])} | {row['research_line']} | {text_value(row['baseline_id_or_recipe']).replace('|', '\\|')} | {row['introduced_or_first_referenced_by']} | `{row['comparison_class']}` |")
    lines.append("")
    (root / "baselines" / "chronology.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
    report_rows, strategy = merge_primary_records(root)
    machine_only_rows = machine_only_strategy_units(strategy, report_rows)
    teammate_rows, package_links = teammate_run_units(root)
    rows = report_rows + machine_only_rows + teammate_rows
    rows.sort(key=lambda row: (text_value(row["date"]), row["namespace"], row["local_id"], row["source_report"]))
    catalogs = load_csv(root / "registry" / "report_catalog.csv")
    if len(report_rows) != len(catalogs):
        raise RuntimeError(f"Normalized report rows ({len(report_rows)}) != report catalog rows ({len(catalogs)}). Wait for audits or fix mapping.")
    if len({(row['source_report'], row['report_sha256']) for row in report_rows}) != len(report_rows):
        raise RuntimeError("Duplicate normalized source report mapping detected")

    lb_rows = load_jsonl(root / "evidence" / "leaderboard_verified.jsonl")
    leaderboard = build_leaderboard(root, lb_rows)
    unverified_lb = build_unverified_lb_claims(root, rows, leaderboard)
    dedup = build_dedup(rows)
    collisions = build_id_collisions(rows)
    baselines = build_baselines(rows)
    ancestry = build_ancestry(rows, lb_rows)
    families = build_families(rows)
    contradictions = build_contradictions(rows, collisions, root)

    fields = [
        "experiment_id", "canonical_name", "family", "date", "parent_baseline", "change",
        "model_family", "validation_protocol", "comparison_class", "cv_score", "delta_cv",
        "folds_positive", "folds_total", "lb_score", "runtime", "status", "evidence_strength",
        "artifacts", "duplicate_of", "compatible_tags", "notes", "namespace", "local_id",
        "source_family", "train_construction", "features", "target", "folds", "seeds",
        "hyperparameters", "cv_score_numeric", "cv_score_report", "per_fold_cv", "submission",
        "outcome_bucket", "relation_type", "raw_compatible_tags", "reproducibility_status",
        "facts", "interpretation", "confounders", "conflicts", "source_report", "source_origin",
        "report_sha256", "machine_audit_id",
    ]
    write_csv(root / "registry" / "experiments.csv", rows, fields)
    write_jsonl(root / "registry" / "experiments.jsonl", rows)
    write_csv(root / "registry" / "deduplication.csv", dedup)
    write_jsonl(root / "registry" / "deduplication.jsonl", dedup)
    write_csv(root / "registry" / "teammate_package_links.csv", package_links)
    write_jsonl(root / "registry" / "teammate_package_links.jsonl", package_links)
    write_csv(root / "registry" / "id_collisions.csv", collisions)
    write_jsonl(root / "registry" / "id_collisions.jsonl", collisions)
    write_csv(root / "registry" / "family_summary.csv", families)
    write_jsonl(root / "registry" / "family_summary.jsonl", families)
    write_csv(root / "baselines" / "chronology.csv", baselines)
    write_jsonl(root / "baselines" / "chronology.jsonl", baselines)
    write_csv(root / "ensembles" / "ancestry_edges.csv", ancestry)
    write_jsonl(root / "ensembles" / "ancestry_edges.jsonl", ancestry)
    write_csv(root / "leaderboard" / "chronology.csv", leaderboard)
    write_jsonl(root / "leaderboard" / "chronology.jsonl", leaderboard)
    write_csv(root / "leaderboard" / "report_only_claims.csv", unverified_lb)
    write_jsonl(root / "leaderboard" / "report_only_claims.jsonl", unverified_lb)
    write_csv(root / "contradictions" / "registry.csv", contradictions)
    write_jsonl(root / "contradictions" / "registry.jsonl", contradictions)
    write_experiment_cards(root, rows)
    write_family_docs(root, families, rows)
    write_baseline_doc(root, baselines)

    summary = {
        "primary_report_rows": len(report_rows),
        "machine_only_experiment_units": len(machine_only_rows),
        "artifact_derived_teammate_run_units": len(teammate_rows),
        "central_registry_rows": len(rows),
        "registry_rows_with_duplicate_or_rerun_link": sum(not is_unknown(row["duplicate_of"]) for row in rows),
        "duplicate_rerun_or_reuse_clusters": len(dedup),
        "clusters_collapsed_for_unique_hypothesis_count": sum(row["collapse_for_unique_hypothesis_count"] == "yes" for row in dedup),
        "novelty_level_units_after_collapsing_exact_doc_and_rerun_rows": len(rows) - sum(
            len([item for item in as_list(row["related_experiment_or_artifact"]) if text_value(item) in {unit["experiment_id"] for unit in rows}])
            for row in dedup if row["collapse_for_unique_hypothesis_count"] == "yes"
        ),
        "teammate_package_copies_linked_not_duplicated": len(package_links),
        "families": len(families),
        "baseline_references": len(baselines),
        "ancestry_edges": len(ancestry),
        "contradictions": len(contradictions),
        "cross_namespace_local_id_collisions": len(collisions),
        "machine_strategy_supplements": len(strategy),
        "repository_confirmed_lb_rows": len(leaderboard),
        "externally_platform_verified_lb_rows": sum(bool(row.get("external_platform_export_present")) for row in leaderboard),
        "unverified_lb_claim_records": len(unverified_lb),
    }
    (root / "registry" / "registry_build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
