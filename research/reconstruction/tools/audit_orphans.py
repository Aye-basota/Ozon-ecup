from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path


DEST = Path(sys.argv[1]).resolve()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    components = read_csv(DEST / "registry" / "components.csv")
    run_metrics = read_csv(DEST / "registry" / "run_metrics.csv")
    report_catalog = read_csv(DEST / "registry" / "report_catalog.csv")
    run_ids = {norm(row["run_id"]): row["run_id"] for row in run_metrics if len(norm(row["run_id"])) >= 5}
    report_text = "\n".join(
        (DEST / row["clean_evidence_path"]).read_text(encoding="utf-8", errors="replace")
        for row in report_catalog
        if (DEST / row["clean_evidence_path"]).exists()
    ).lower()
    rows: list[dict] = []
    for component in components:
        component_norm = norm(component["component_id"])
        matched_run = "unknown"
        if component_norm in run_ids:
            matched_run = run_ids[component_norm]
        else:
            candidates = [original for key, original in run_ids.items() if len(key) >= 8 and (key in component_norm or component_norm in key)]
            if candidates:
                matched_run = sorted(candidates, key=len, reverse=True)[0]
        explicit_association = any(
            token.startswith(("team_a:exp_", "independent_", "team_b_", "team_a_s2:"))
            for token in component["experiment_association"].split(";")
        )
        referenced = component["component_id"].lower() in report_text or any(
            Path(path).name.lower() in report_text
            for field in ("oof_artifacts", "test_artifacts", "model_artifacts")
            for path in component[field].split(";")
            if path != "unknown"
        )
        if explicit_association:
            status = "linked_by_artifact_path_association"
        elif matched_run != "unknown":
            status = "linked_to_run_metric_id"
        elif referenced:
            status = "referenced_by_primary_report"
        else:
            status = "orphan_candidate"
        rows.append({
            "component_id": component["component_id"],
            "pairing_status": component["pairing_status"],
            "experiment_association": component["experiment_association"],
            "matched_run_id": matched_run,
            "primary_report_reference": "yes" if referenced else "no",
            "link_status": status,
            "oof_artifacts": component["oof_artifacts"],
            "test_artifacts": component["test_artifacts"],
            "model_artifacts": component["model_artifacts"],
            "notes": "Orphan candidate means no mechanical link was recovered; it does not assert the artifact was unused.",
        })
    write_csv(
        DEST / "artifacts" / "component_linkage_audit.csv",
        rows,
        ["component_id", "pairing_status", "experiment_association", "matched_run_id", "primary_report_reference", "link_status", "oof_artifacts", "test_artifacts", "model_artifacts", "notes"],
    )
    write_csv(
        DEST / "artifacts" / "orphan_prediction_candidates.csv",
        [r for r in rows if r["link_status"] == "orphan_candidate" and r["pairing_status"] in {"oof_only", "test_only", "oof_and_test"}],
        ["component_id", "pairing_status", "experiment_association", "matched_run_id", "primary_report_reference", "link_status", "oof_artifacts", "test_artifacts", "model_artifacts", "notes"],
    )
    print(json.dumps({
        "components": len(rows),
        "linked_by_association": sum(r["link_status"] == "linked_by_artifact_path_association" for r in rows),
        "linked_to_run": sum(r["link_status"] == "linked_to_run_metric_id" for r in rows),
        "linked_by_report": sum(r["link_status"] == "referenced_by_primary_report" for r in rows),
        "orphan_component_candidates": sum(r["link_status"] == "orphan_candidate" for r in rows),
        "orphan_prediction_candidates": sum(r["link_status"] == "orphan_candidate" and r["pairing_status"] in {"oof_only", "test_only", "oof_and_test"} for r in rows),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
