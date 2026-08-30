"""Normalize experiment outcomes without pooling incompatible score protocols."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


MANUAL_DELTA = {
    "team_a_current:EXP-004": 0.00605,
    "team_a_current:EXP-008": 0.01494,
    "team_a_current:EXP-013": 0.00076,
    "team_a_current:EXP-014": -0.00071,
    "team_a_s2:EXP-011": -0.00089334,
    "team_a_s2:EXP-012": 0.00945,
    "independent_calendar:EXP-029": None,
    "independent_domain:EXP-028": 0.0000855190,
    "independent_renewal:EXP-027": -0.0002664138,
    "team_b_core:EXP-001": None,
    "team_b_core:EXP-028": 0.007514,
    "team_b_core:EXP-029": 0.007424,
    "team_a_current:EXP-065": None,
    "independent_anniversary:EXP-058": 0.0003902915,
}

NO_NEW_RESULT = {
    "team_a_current:EXP-051",
    "team_a_current:EXP-065",
    "team_a_current:EXP-032-MANIFEST",
}

COLLAPSE_RELATIONS = {
    "exact_replay",
    "seed_rerun",
    "multiseed_rerun",
    "duplicate_document_for_same_experiment",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=Path("registry/experiments.csv"))
    parser.add_argument("--leaderboard", type=Path, default=Path("leaderboard/chronology.csv"))
    parser.add_argument("--output", type=Path, default=Path("analysis/intermediate/registry_audit.json"))
    return parser.parse_args()


def simple_delta(value: str) -> float | None:
    value = str(value).strip()
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?", value, flags=re.I):
        return float(value)
    if value.startswith("200 -0.00067"):
        return -0.00067
    if value.startswith("AVG3 -0.00062"):
        return -0.00071
    if value.startswith("head -0.00145"):
        return -0.00071
    if value.startswith("real minus linear"):
        return None
    return None


def bucket(delta: float | None) -> str:
    if delta is None:
        return "inconclusive_or_no_comparable_delta"
    if delta <= -0.003:
        return "gain_gt_0.003"
    if delta <= -0.001:
        return "gain_0.001_to_0.003"
    if delta <= -0.0005:
        return "gain_0.0005_to_0.001"
    if delta <= -0.0001:
        return "gain_0.0001_to_0.0005"
    if delta < 0.0001:
        return "neutral_abs_lt_0.0001"
    return "negative_delta_ge_0.0001"


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.registry, dtype=str).fillna("unknown")
    novelty = df[~df["relation_type"].isin(COLLAPSE_RELATIONS)].copy()
    assert len(df) == 138 and len(novelty) == 134

    records = []
    for row in novelty.to_dict("records"):
        exp_id = row["experiment_id"]
        delta = MANUAL_DELTA.get(exp_id, simple_delta(row["delta_cv"]))
        if exp_id in NO_NEW_RESULT:
            delta_for_progress = None
        else:
            delta_for_progress = delta
        records.append(
            {
                "experiment_id": exp_id,
                "canonical_name": row["canonical_name"],
                "family": row["family"],
                "namespace": row["namespace"],
                "comparison_class": row["comparison_class"],
                "status": row["status"],
                "outcome_bucket_registry": row["outcome_bucket"],
                "normalized_primary_delta": delta,
                "distribution_delta": delta_for_progress,
                "distribution_bucket": bucket(delta_for_progress),
                "folds_positive": row["folds_positive"],
                "folds_total": row["folds_total"],
                "evidence_strength": row["evidence_strength"],
            }
        )
    counts: dict[str, int] = {}
    for record in records:
        key = record["distribution_bucket"]
        counts[key] = counts.get(key, 0) + 1

    family_summary = []
    for family, group in pd.DataFrame(records).groupby("family"):
        numeric = pd.to_numeric(group["distribution_delta"], errors="coerce")
        family_summary.append(
            {
                "family": family,
                "experiments": int(len(group)),
                "comparable_deltas": int(numeric.notna().sum()),
                "best_delta": None if numeric.notna().sum() == 0 else float(numeric.min()),
                "median_delta": None if numeric.notna().sum() == 0 else float(numeric.median()),
                "positive_numeric_count": int((numeric < -0.0001).sum()),
                "last_five_ids": group.tail(5)["experiment_id"].tolist(),
                "last_five_deltas": [None if pd.isna(x) else float(x) for x in numeric.tail(5)],
            }
        )

    lb = pd.read_csv(args.leaderboard, dtype=str)
    lb["score_float"] = lb["score"].astype(float)
    strong_line = lb[
        lb["filename"].isin(
            [
                "submission_strategy_1.csv",
                "submission_dist_head.csv",
                "submission_SEQ01_mix.csv",
                "submission_STRONGEST_CURRENT.csv",
            ]
        )
    ].copy()
    strong_line = strong_line.sort_values(["date", "score_float"])
    lb_progress = []
    previous = None
    for row in strong_line.to_dict("records"):
        score = float(row["score_float"])
        lb_progress.append(
            {
                "date": row["date"],
                "filename": row["filename"],
                "score": score,
                "gain_from_previous": None if previous is None else score - previous,
            }
        )
        previous = score

    result = {
        "registry_rows": len(df),
        "novelty_rows": len(novelty),
        "distribution_method": "one primary parent-aligned RMSLE delta per novelty unit; revalidations and non-comparable diagnostics excluded",
        "distribution_counts": counts,
        "records": records,
        "family_summary": family_summary,
        "strong_line_lb_progress": lb_progress,
        "strong_line_post_s1_best_total_gain": float(strong_line.iloc[-1]["score_float"] - strong_line.iloc[0]["score_float"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"distribution_counts": counts, "strong_line_lb_progress": lb_progress}, indent=2))


if __name__ == "__main__":
    main()
