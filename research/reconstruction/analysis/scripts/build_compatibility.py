"""Build the all-positive-experiment compatibility matrix as JSON.

The final CSV is authored/validated by build_csvs.mjs.  This script keeps the
judgment rules explicit and reproducible instead of embedding them in prose.
"""

from __future__ import annotations

import json
from pathlib import Path


REGISTRY_AUDIT = Path("analysis/intermediate/registry_audit.json")
OUTPUT = Path("analysis/intermediate/compatibility.json")


ABSORBED_IN_CHAMPION = {
    "team_a_current:EXP-003",
    "team_a_current:EXP-005",
    "team_a_current:EXP-006",
    "team_a_current:EXP-014",
    "team_a_current:EXP-025",
    "team_a_current:EXP-026",
    "team_a_current:EXP-027",
    "team_a_current:EXP-035",
    "team_a_current:EXP-036",
    "team_a_current:EXP-037",
}

ROLE_OVERRIDES = {
    "team_a_current:EXP-003": "train_construction_dense_cutoffs",
    "team_a_current:EXP-005": "tabular_representation_normalized_long",
    "team_a_current:EXP-006": "tabular_ensemble",
    "team_a_current:EXP-014": "distribution_head",
    "team_a_current:EXP-016": "correlated_weight_search",
    "team_a_current:EXP-017": "tabular_rounds",
    "team_a_current:EXP-018": "tabular_seed_average",
    "team_a_current:EXP-025": "sequence_representation",
    "team_a_current:EXP-026": "sequence_seed_average",
    "team_a_current:EXP-027": "sequence_depth_policy",
    "team_a_current:RUN-S04-LGB": "conditional_fresh_tabular",
    "team_a_current:EXP-030": "sequence_depth_curriculum",
    "team_a_current:EXP-032": "conditional_fresh_sequence",
    "team_a_current:EXP-032B": "conditional_fresh_sequence",
    "team_a_current:EXP-035": "sequence_slot_composition",
    "team_a_current:EXP-036": "etx_representation",
    "team_a_current:EXP-037": "champion_composition",
    "team_a_current:EXP-038": "future_funnel_auxiliary_target",
    "team_a_current:EXP-040": "fresh_residual_correction",
    "team_a_current:EXP-047": "btyd_residual_correction",
    "team_a_current:EXP-049": "btyd_plus_fresh_compound",
    "team_a_current:EXP-051": "btyd_production_revalidation",
    "team_a_current:EXP-059": "sequence_slot_reweight",
    "independent_renewal:EXP-027": "renewal_residual_correction",
    "team_a_s2:EXP-010": "structural_count_hurdle",
    "team_a_s2:EXP-011": "structural_monetary_shrinkage",
}

MUTUALLY_EXCLUSIVE_ROLES = {
    "sequence_representation",
    "sequence_depth_curriculum",
    "conditional_fresh_sequence",
    "future_funnel_auxiliary_target",
}

RESIDUAL_OR_COMPOSITION = {
    "fresh_residual_correction",
    "btyd_residual_correction",
    "btyd_production_revalidation",
    "renewal_residual_correction",
    "sequence_slot_reweight",
    "btyd_plus_fresh_compound",
}


def inferred_role(record: dict[str, object]) -> str:
    exp_id = str(record["experiment_id"])
    if exp_id in ROLE_OVERRIDES:
        return ROLE_OVERRIDES[exp_id]
    namespace = str(record["namespace"])
    family = str(record["family"])
    if namespace == "team_b_core":
        return "team_b_core_" + family
    if namespace == "team_b_alt":
        return "team_b_alt_" + family
    if namespace == "team_a_s2":
        return "strategy2_" + family
    return family


def current_relationship(exp_id: str, role: str, namespace: str) -> str:
    if exp_id in ABSORBED_IN_CHAMPION:
        return "absorbed_or_nested_in_EXP-037"
    if exp_id == "team_a_current:EXP-051":
        return "not_in_champion; production-ready_revalidation_of_EXP-047"
    if exp_id in {"team_a_current:EXP-047", "team_a_current:EXP-059", "team_a_current:EXP-040"}:
        return "not_in_champion; pairable_OOF_candidate"
    if exp_id == "team_a_current:EXP-049":
        return "compound_OOF_only; missing_FRESH_test_support"
    if namespace != "team_a_current":
        return "parallel_pipeline_or_protocol; not_absorbed"
    if role in MUTUALLY_EXCLUSIVE_ROLES:
        return "alternative_to_current_component"
    return "not_absorbed_or_transfer_unproven"


def relation(a: dict[str, str], b: dict[str, str]) -> str:
    if a["experiment_id"] == b["experiment_id"]:
        return "SELF"
    ids = frozenset([a["experiment_id"], b["experiment_id"]])
    if ids in {
        frozenset(["team_a_current:EXP-035", "team_a_current:EXP-036"]),
        frozenset(["team_a_current:EXP-036", "team_a_current:EXP-037"]),
        frozenset(["team_a_current:EXP-035", "team_a_current:EXP-037"]),
    }:
        return "T+"
    if ids in {
        frozenset(["team_a_current:EXP-047", "team_a_current:EXP-059"]),
        frozenset(["team_a_current:EXP-051", "team_a_current:EXP-059"]),
    }:
        return "A+"
    if ids in {
        frozenset(["team_a_current:EXP-040", "team_a_current:EXP-047"]),
        frozenset(["team_a_current:EXP-040", "team_a_current:EXP-051"]),
        frozenset(["team_a_current:EXP-040", "team_a_current:EXP-049"]),
    }:
        return "A-"
    if "team_a_current:EXP-049" in ids and (
        "team_a_current:EXP-047" in ids or "team_a_current:EXP-040" in ids
    ):
        return "N"
    if a["namespace"] != b["namespace"]:
        return "U"
    if a["role"] == b["role"]:
        return "R"
    if a["role"] in MUTUALLY_EXCLUSIVE_ROLES and b["role"] in MUTUALLY_EXCLUSIVE_ROLES:
        return "X"
    if a["role"] in RESIDUAL_OR_COMPOSITION or b["role"] in RESIDUAL_OR_COMPOSITION:
        return "A?"
    if a["relationship"].startswith("absorbed") and b["relationship"].startswith("absorbed"):
        return "N"
    return "C"


def main() -> None:
    audit = json.loads(REGISTRY_AUDIT.read_text(encoding="utf-8"))
    gain_records = [
        r for r in audit["records"] if str(r["distribution_bucket"]).startswith("gain_")
    ]
    # EXP-051 is a positive 4/4 production revalidation but was intentionally
    # excluded from the progress distribution to avoid double-counting EXP-047.
    replay = next(r for r in audit["records"] if r["experiment_id"] == "team_a_current:EXP-051")
    replay = dict(replay)
    replay["normalized_primary_delta"] = -0.00026918201136146475
    gain_records.append(replay)

    nodes = []
    for record in gain_records:
        exp_id = str(record["experiment_id"])
        role = inferred_role(record)
        namespace = str(record["namespace"])
        nodes.append(
            {
                "experiment_id": exp_id,
                "canonical_name": str(record["canonical_name"]),
                "namespace": namespace,
                "role": role,
                "relationship": current_relationship(exp_id, role, namespace),
                "delta": record["normalized_primary_delta"],
                "folds_positive": str(record["folds_positive"]),
                "folds_total": str(record["folds_total"]),
                "comparison_class": str(record["comparison_class"]),
                "evidence_strength": str(record["evidence_strength"]),
            }
        )
    nodes.sort(key=lambda x: x["experiment_id"])
    matrix = []
    for a in nodes:
        row = dict(a)
        row["compatibility"] = {b["experiment_id"]: relation(a, b) for b in nodes}
        matrix.append(row)

    legend = {
        "SELF": "same experiment",
        "T+": "historically tested together with positive result",
        "A+": "new audit: fixed recipes combine near-additively on aligned OOF",
        "A-": "tested together; positive total but antagonistic interaction",
        "A?": "mechanistically potentially additive; exact pair not tested",
        "N": "nested/absorbed/dependent; do not add arithmetically",
        "R": "same signal family; likely redundant/highly correlated",
        "X": "alternative implementations or mutually exclusive component slots",
        "C": "correlated/uncertain compatibility",
        "U": "different protocol/pipeline; pairability unproven",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps({"legend": legend, "nodes": nodes, "matrix": matrix}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"positive_nodes": len(nodes), "legend": legend}, indent=2))


if __name__ == "__main__":
    main()
