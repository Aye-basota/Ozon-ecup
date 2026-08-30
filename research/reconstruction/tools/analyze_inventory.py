from __future__ import annotations

import csv
import hashlib
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def dataset_family(path: str) -> str:
    name = Path(path).stem
    replacements = [
        (r"\d{4}-\d{2}-\d{2}.*$", "{date}"),
        (r"\d{8}.*$", "{date}"),
        (r"_(?:2025|2026)\d{4}.*$", "_{date}"),
    ]
    for pattern, replacement in replacements:
        name = re.sub(pattern, replacement, name)
    parent = Path(path).parent.as_posix()
    return f"{parent}/{name}"


def component_key(path: str) -> str:
    p = Path(path)
    name = p.stem
    name = re.sub(r"(?i)^(?:oof|test|ztest|uid|pred|prediction|model|curve|report|fold)[_-]", "", name)
    name = re.sub(r"(?i)_(?:oof|test|pred|prediction|uid|model)$", "", name)
    if p.parent.name.upper().startswith(("BTYD", "BURST", "BUYCTRL", "CHANNEL", "DET", "EVENT", "FINGER", "FRESH", "LANDMARK", "LATE", "OPEN", "PLATFORM", "STATE", "TBR")):
        return p.parent.name + ":" + name
    return name


def main() -> None:
    files = read_csv(DEST / "inventory" / "files.csv")
    worktree_unique_path = DEST / "artifacts" / "worktree_unique_files.csv"
    if worktree_unique_path.exists():
        association_map = {
            "independent_anniversary": "independent_anniversary:exp_058",
            "independent_calendar": "independent_calendar:exp_029",
            "independent_domain": "independent_domain:exp_028",
            "independent_global_regime": "independent_global_regime:exp_057",
            "independent_renewal": "independent_renewal:exp_027",
            "team_a_s2": "team_a_s2:exp_012",
        }
        for row in read_csv(worktree_unique_path):
            wt_type = row["type"]
            wt_name = Path(row["relative_path"]).name.lower()
            if (wt_name.startswith(("model_", "checkpoint_")) or Path(row["relative_path"]).suffix.lower() in {".pt", ".pth", ".ckpt"}) and "checkpoint_or_model" not in wt_type:
                wt_type += ";checkpoint_or_model"
            files.append({
                "original_path": f"linked_worktree/{row['namespace']}/{row['relative_path']}",
                "filename": Path(row["relative_path"]).name,
                "extension": Path(row["relative_path"]).suffix.lower() or "[none]",
                "type": wt_type,
                "purpose": "linked_worktree_evidence",
                "size_bytes": row["size_bytes"],
                "sha256": row["sha256"],
                "experiment_association": association_map.get(row["namespace"], "unknown"),
                "schema_or_arrays": row["schema_or_arrays"],
                "last_write_utc": "unknown",
                "copied": "see_worktree_artifacts_manifest",
                "evidence_excluded": "no",
            })
    by_sha: dict[str, list[dict]] = defaultdict(list)
    for row in files:
        if row["sha256"] != "unknown":
            by_sha[row["sha256"]].append(row)
    duplicate_rows: list[dict] = []
    for sha, group in by_sha.items():
        if len(group) < 2:
            continue
        canonical = sorted(group, key=lambda r: (len(r["original_path"]), r["original_path"]))[0]["original_path"]
        total_bytes = sum(int(r["size_bytes"]) for r in group)
        for row in sorted(group, key=lambda r: r["original_path"]):
            duplicate_rows.append({
                "sha256": sha,
                "size_bytes_each": row["size_bytes"],
                "copies": len(group),
                "aggregate_bytes": total_bytes,
                "canonical_path": canonical,
                "path": row["original_path"],
                "duplicate_of": "" if row["original_path"] == canonical else canonical,
                "type": row["type"],
            })
    write_csv(
        DEST / "artifacts" / "exact_duplicates.csv",
        duplicate_rows,
        ["sha256", "size_bytes_each", "copies", "aggregate_bytes", "canonical_path", "path", "duplicate_of", "type"],
    )

    datasets = [row for row in files if "dataset" in row["type"]]
    dataset_groups: dict[str, list[dict]] = defaultdict(list)
    for row in datasets:
        dataset_groups[dataset_family(row["original_path"])].append(row)
    dataset_rows: list[dict] = []
    for family, group in sorted(dataset_groups.items()):
        digest = hashlib.sha256()
        for row in sorted(group, key=lambda r: r["original_path"]):
            digest.update(row["original_path"].encode("utf-8"))
            digest.update(b"\0")
            digest.update(row["sha256"].encode("ascii", errors="replace"))
            digest.update(b"\n")
        dataset_rows.append({
            "dataset_family": family,
            "files": len(group),
            "total_bytes": sum(int(row["size_bytes"]) for row in group),
            "combined_path_content_sha256": digest.hexdigest(),
            "schemas": ";".join(sorted(set(row["schema_or_arrays"] for row in group)))[:8000],
            "paths": ";".join(row["original_path"] for row in sorted(group, key=lambda r: r["original_path"])),
            "individual_sha256": ";".join(row["sha256"] for row in sorted(group, key=lambda r: r["original_path"])),
        })
    write_csv(
        DEST / "inventory" / "dataset_fingerprints.csv",
        dataset_rows,
        ["dataset_family", "files", "total_bytes", "combined_path_content_sha256", "schemas", "paths", "individual_sha256"],
    )

    component_artifacts = [
        row for row in files
        if any(token in row["type"] for token in ("oof_prediction", "test_prediction", "checkpoint_or_model"))
        and "dataset" not in row["type"]
    ]
    components: dict[str, list[dict]] = defaultdict(list)
    for row in component_artifacts:
        components[component_key(row["original_path"])].append(row)
    component_rows: list[dict] = []
    for key, group in sorted(components.items()):
        oof = [r for r in group if "oof_prediction" in r["type"]]
        tests = [r for r in group if "test_prediction" in r["type"]]
        models = [r for r in group if "checkpoint_or_model" in r["type"]]
        assocs = sorted({a for r in group for a in r["experiment_association"].split(";") if a})
        shas = [r["sha256"] for r in group]
        duplicate_of = "unknown"
        if len(set(shas)) < len(shas):
            duplicate_of = "contains_exact_duplicate_artifacts"
        component_rows.append({
            "component_id": key,
            "experiment_association": ";".join(assocs) or "unknown",
            "model_family": next((a.split(":", 1)[1] for a in assocs if a.startswith("component_family:")), "unknown"),
            "oof_artifacts": ";".join(r["original_path"] for r in oof) or "unknown",
            "test_artifacts": ";".join(r["original_path"] for r in tests) or "unknown",
            "model_artifacts": ";".join(r["original_path"] for r in models) or "unknown",
            "artifact_count": len(group),
            "schema_or_arrays": ";".join(sorted({r["schema_or_arrays"] for r in group}))[:12000],
            "sha256": ";".join(shas),
            "pairing_status": "oof_and_test" if oof and tests else ("oof_only" if oof else ("test_only" if tests else "model_only")),
            "duplicate_note": duplicate_of,
            "validation_protocol": "unknown",
            "ensemble_readiness": "potential_component_only_not_a_recommendation" if oof or tests else "model_state_only",
        })
    write_csv(
        DEST / "registry" / "components.csv",
        component_rows,
        ["component_id", "experiment_association", "model_family", "oof_artifacts", "test_artifacts", "model_artifacts", "artifact_count", "schema_or_arrays", "sha256", "pairing_status", "duplicate_note", "validation_protocol", "ensemble_readiness"],
    )

    submissions = [row for row in files if "submission" in row["type"] and row["extension"] == ".csv"]
    submission_rows = [{
        "filename": row["filename"],
        "original_path": row["original_path"],
        "size_bytes": row["size_bytes"],
        "sha256": row["sha256"],
        "schema": row["schema_or_arrays"],
        "experiment_association": row["experiment_association"],
        "recipe_status": "unknown_pending_lineage_audit",
        "lb_score": "unknown",
        "lb_evidence_status": "unknown",
    } for row in submissions]
    write_csv(
        DEST / "submissions" / "inventory.csv",
        submission_rows,
        ["filename", "original_path", "size_bytes", "sha256", "schema", "experiment_association", "recipe_status", "lb_score", "lb_evidence_status"],
    )

    summary = {
        "source_files": len(files),
        "exact_duplicate_sha_groups": sum(1 for group in by_sha.values() if len(group) > 1),
        "files_in_duplicate_groups": len(duplicate_rows),
        "dataset_families": len(dataset_rows),
        "component_groups": len(component_rows),
        "submission_csv_candidates": len(submission_rows),
    }
    (DEST / "inventory" / "inventory_analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
