"""Finalize EXP081 validity labels, revised headroom, and immutable inventory."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    summary = pd.read_csv(HERE / "candidate_summary.csv")
    rows = []
    for row in summary.to_dict("records"):
        name = row["candidate"]
        primary = "full_span" in name
        if "user_crossfit" in name or "tail_routed" in name:
            validity = "SAME_CUTOFF_USER_CROSSFIT_NOT_TEMPORAL"
        elif name == "lgbm_A_ordered_previous_folds_full_span":
            validity = "ORDERED_BUT_TARGET_LABELS_NOT_AVAILABLE_AT_ADJACENT_CUTOFFS"
        elif name == "behavioral_prototype_k128_full_span":
            validity = "EXP080_COMPARABLE_FORWARD_NOT_FULLY_PURGED"
        elif name.endswith("current_only"):
            validity = "NONPRIMARY_CURRENT_ONLY_PROJECTION"
        elif "deployable_span" in name:
            validity = "SENSITIVITY_16_COMPONENT_PROJECTION"
        else:
            validity = "EXP080_COMPARABLE_FORWARD_NOT_FULLY_PURGED"
        rows.append({
            **row,
            "numeric_gate_under_stated_protocol": bool(row["passes_gate"]),
            "primary_full_40_span": primary,
            "strict_target_availability_purged_4fold": False,
            "validity": validity,
            "AUTHORIZED": False,
        })
    verdicts = pd.DataFrame(rows)
    verdicts.to_csv(HERE / "final_candidate_verdicts.csv", index=False)

    exp080 = json.loads((HERE / "exp080_reproduction.json").read_text(encoding="utf-8"))
    reported = exp080["observable"]["reported"]
    full = summary[summary.candidate.str.contains("full_span")]
    purged_fixed = pd.read_csv(HERE / "purged_fixed_basis_metrics.csv")
    purged_tail = pd.read_csv(HERE / "purged_tail_metrics.csv")
    purged_audit = json.loads((HERE / "purged_tail_audit.json").read_text(encoding="utf-8"))
    purged_points = [
        *purged_fixed.purged_latest_Delta_MSE.tolist(),
        *purged_tail.purged_latest_Delta_MSE.tolist(),
        purged_audit["ungated"]["Delta_MSE"],
    ]
    revised = {
        "required_Delta_MSE": exp080["gap"]["required_Delta_MSE"],
        "EXP080_robust_headroom": reported["robust_forward_headroom_95pct_lower_bound"],
        "EXP080_optimistic_headroom": reported["observable_joint_optimal_headroom"],
        "EXP080_comparable_forward_point": reported["joint_nested_forward_headroom_point"],
        "new_cross_sectional_full_span_optimistic_max": float(full.optimistic_headroom.max()),
        "revised_optimistic_headroom": float(max(
            reported["observable_joint_optimal_headroom"], full.optimistic_headroom.max()
        )),
        "new_nonpurged_full_span_forward_diagnostic_max": float(full.strict_forward_headroom.max()),
        "revised_EXP080_comparable_forward_headroom": reported["joint_nested_forward_headroom_point"],
        "purged_latest_best_point_headroom": max(0.0, -min(purged_points)),
        "purged_latest_robust_95pct_headroom": 0.0,
        "full_4fold_purged_bound_identifiable": False,
        "why_not_identifiable": (
            "Canonical folds are 14 days apart but targets last 30 days; only 2025-09-04 "
            "labels are available by the latest clean cutoff 2025-10-16."
        ),
        "authorized_candidates": [],
        "verdict": "NO_EVIDENCE_CONFIRMED",
    }
    (HERE / "revised_headroom.json").write_text(
        json.dumps(revised, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    exclude = {"artifact_manifest.csv", "checksums.sha256", "__pycache__"}
    files = [p for p in sorted(HERE.iterdir()) if p.is_file() and p.name not in exclude]
    manifest = pd.DataFrame([
        {"file": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)} for p in files
    ])
    manifest.to_csv(HERE / "artifact_manifest.csv", index=False, lineterminator="\n")
    (HERE / "checksums.sha256").write_text(
        "".join(f"{row.sha256}  {row.file}\n" for row in manifest.itertuples()), encoding="utf-8"
    )
    print(json.dumps(revised, indent=2))


if __name__ == "__main__":
    main()
