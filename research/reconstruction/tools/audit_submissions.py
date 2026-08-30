from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE = Path(sys.argv[1]).resolve()
DEST = Path(sys.argv[2]).resolve()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalize(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def resolve_manifest_path(path: str) -> str:
    value = normalize(path)
    if not value:
        return value
    if value.startswith(("submissions/", "пайплайн сокомандника/")):
        return value
    return "submissions/" + value


def order_hash(values: pd.Series) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def explicit_submission_path(text: str) -> str:
    matches = re.findall(r"(?i)(?:submissions?/)?[A-Za-z0-9_.\-/]+\.csv", text)
    for item in matches:
        if "submission" in item.lower() or item.lower().startswith("submissions/"):
            return normalize(item.strip("`*.,;:()"))
    return "unknown"


def main() -> None:
    inventory = read_csv(DEST / "inventory" / "files.csv")
    by_rel = {normalize(row["original_path"]): row for row in inventory}
    worktree_file_path = DEST / "inventory" / "worktree_files.csv"
    worktree_rows = read_csv(worktree_file_path) if worktree_file_path.exists() else []
    worktree_submission_by_rel = {
        normalize(row["relative_path"]): row
        for row in worktree_rows
        if row.get("namespace") == "team_a_s2" and "submission" in row.get("type", "")
    }
    manifest_path = DEST / "evidence" / "machine_manifests" / "team_a_current__submissions.csv"
    manifest_rows = read_csv(manifest_path)
    manifest_by_file = {resolve_manifest_path(row.get("file", "")): row for row in manifest_rows}

    sample = pd.read_csv(SOURCE / "data" / "raw" / "sample_submit.csv")
    sample_order_sha = order_hash(sample["user_id"])
    candidates: set[str] = set()
    for relative, row in by_rel.items():
        if row["extension"] == ".csv" and "submission" in row["type"]:
            candidates.add(relative)
    candidates.update(manifest_by_file)

    rows: list[dict] = []
    for relative in sorted(candidates):
        source_path = SOURCE / Path(relative)
        worktree_artifact = None
        if not source_path.exists() and relative in worktree_submission_by_rel:
            worktree_artifact = worktree_submission_by_rel[relative]
            source_path = Path(worktree_artifact["worktree_root"]) / Path(relative)
        manifest = manifest_by_file.get(relative, {})
        exists = source_path.exists()
        record = {
            "filename": Path(relative).name,
            "original_path": relative,
            "exists": "yes" if exists else "no",
            "sha256": by_rel.get(relative, {}).get("sha256", "unknown") if worktree_artifact is None else worktree_artifact["sha256"],
            "artifact_location": str(source_path) if exists else "missing",
            "artifact_root": "linked_worktree" if worktree_artifact is not None else "main_worktree",
            "rows": "unknown",
            "columns": "unknown",
            "valid_submission_schema": "no",
            "user_order_matches_sample": "unknown",
            "unique_users": "unknown",
            "duplicate_users": "unknown",
            "nonfinite_predictions": "unknown",
            "negative_predictions": "unknown",
            "mean_prediction": "unknown",
            "mean_log1p_prediction": "unknown",
            "experiment_id": manifest.get("exp_id") or "unknown",
            "date": manifest.get("date") or "unknown",
            "level_manifest": manifest.get("level") or "unknown",
            "lb_score": manifest.get("lb_public") or "unknown",
            "lb_evidence_status": "artifact_plus_experiment_manifest_not_platform_independently_verified" if manifest.get("lb_public") and exists else ("manifest_score_but_artifact_missing" if manifest.get("lb_public") else "no_lb_score"),
            "recipe": manifest.get("oof_source") or "unknown",
            "source_predictions": manifest.get("oof_source") or "unknown",
            "lineage": manifest.get("exp_id") or "unknown",
            "recipe_status": "manifest_recipe_present" if manifest.get("oof_source") else "unknown",
        }
        if exists:
            try:
                df = pd.read_csv(source_path)
                record["rows"] = len(df)
                record["columns"] = "|".join(str(c) for c in df.columns)
                if list(df.columns) == ["user_id", "predict"]:
                    record["valid_submission_schema"] = "yes"
                    record["user_order_matches_sample"] = "yes" if len(df) == len(sample) and order_hash(df["user_id"]) == sample_order_sha else "no"
                    record["unique_users"] = int(df["user_id"].nunique(dropna=False))
                    record["duplicate_users"] = int(df["user_id"].duplicated().sum())
                    pred = pd.to_numeric(df["predict"], errors="coerce").to_numpy(dtype=float)
                    record["nonfinite_predictions"] = int((~np.isfinite(pred)).sum())
                    record["negative_predictions"] = int((pred < 0).sum())
                    finite = pred[np.isfinite(pred)]
                    if len(finite):
                        record["mean_prediction"] = f"{finite.mean():.12g}"
                        if (finite >= -1).all():
                            record["mean_log1p_prediction"] = f"{np.log1p(finite).mean():.12g}"
            except Exception as exc:
                record["columns"] = f"parse_error:{type(exc).__name__}"
        rows.append(record)

    # Report-only LB claims remain separate and are not promoted into the strict chronology.
    report_catalog = read_csv(DEST / "registry" / "report_catalog.csv")
    claims: list[dict] = []
    for report in report_catalog:
        if report["lb_candidate"] == "unknown":
            continue
        submission = explicit_submission_path(report.get("facts_reported", "") + "\n" + report.get("config_reported", ""))
        file_exists = submission != "unknown" and (SOURCE / Path(submission)).exists()
        claims.append({
            "experiment_id": report["experiment_id"],
            "date": report["date_reported"],
            "filename": submission,
            "score": report["lb_candidate"],
            "claim_source": report["source_ref"] + ":" + report["source_path"],
            "claim_evidence_tier": report["evidence_tier"],
            "artifact_exists": "yes" if file_exists else "no",
            "strictly_confirmed": "no",
            "reason": "report_claim_only_or_parser_candidate",
        })

    strict = [
        {
            "date": row["date"],
            "filename": row["filename"],
            "score": row["lb_score"],
            "recipe": row["recipe"],
            "source_predictions": row["source_predictions"],
            "experiment_lineage": row["lineage"],
            "artifact_sha256": row["sha256"],
            "evidence_status": row["lb_evidence_status"],
        }
        for row in rows
        if row["lb_score"] != "unknown" and row["exists"] == "yes" and row["valid_submission_schema"] == "yes"
    ]
    fields = [
        "filename", "original_path", "exists", "sha256", "artifact_location", "artifact_root", "rows", "columns", "valid_submission_schema",
        "user_order_matches_sample", "unique_users", "duplicate_users", "nonfinite_predictions", "negative_predictions",
        "mean_prediction", "mean_log1p_prediction", "experiment_id", "date", "level_manifest", "lb_score",
        "lb_evidence_status", "recipe", "source_predictions", "lineage", "recipe_status",
    ]
    write_csv(DEST / "submissions" / "audit.csv", rows, fields)
    write_csv(
        DEST / "leaderboard" / "chronology.csv",
        sorted(strict, key=lambda r: (r["date"], r["filename"])),
        ["date", "filename", "score", "recipe", "source_predictions", "experiment_lineage", "artifact_sha256", "evidence_status"],
    )
    write_csv(
        DEST / "leaderboard" / "report_only_claims.csv",
        claims,
        ["experiment_id", "date", "filename", "score", "claim_source", "claim_evidence_tier", "artifact_exists", "strictly_confirmed", "reason"],
    )
    missing = [row for row in rows if row["exists"] == "no"]
    write_csv(DEST / "contradictions" / "manifested_submissions_missing.csv", missing, fields)
    print(json.dumps({
        "submission_candidates_or_manifest_rows": len(rows),
        "valid_submission_artifacts": sum(r["valid_submission_schema"] == "yes" for r in rows),
        "missing_manifested_artifacts": len(missing),
        "strict_lb_chronology_rows": len(strict),
        "report_only_lb_claims": len(claims),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
