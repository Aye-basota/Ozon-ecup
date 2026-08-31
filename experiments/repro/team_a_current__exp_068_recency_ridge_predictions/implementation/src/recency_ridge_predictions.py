from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FOLDS = ("2025-09-04", "2025-09-18", "2025-10-02", "2025-10-16")
FOLD_WEIGHTS = (1.0, 2.0, 4.0, 8.0)
HISTORICAL_NAME = "ridge_drop_recent_hurdle_stable18_s075"
LATE_RIDGE_ANCHOR = "blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85"
PREFIX_DEFAULT = "RECENCY_RIDGE_PRED_EXP068_A1"
REPLAY_FLOOR = 5e-7

HISTORICAL_MEMBERS = (
    "cap",
    "unc",
    "dist",
    "hurdle",
    "multiscale_direct",
    "recent_direct",
    "recent_dist",
    "recent_hurdle_fast12",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_submission(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["user_id", "predict"]:
        raise ValueError(f"Unexpected submission schema: {path}")
    uid = frame["user_id"].to_numpy(np.int64)
    prediction = frame["predict"].to_numpy(np.float64)
    if len(uid) != 250_000 or len(np.unique(uid)) != len(uid):
        raise ValueError(f"Invalid user rows in {path}")
    if not np.isfinite(prediction).all() or (prediction < 0).any():
        raise ValueError(f"Invalid prediction values in {path}")
    return uid, prediction, np.log1p(prediction)


def distance(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    delta = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    return {
        "corr": float(np.corrcoef(a, b)[0, 1]),
        "var_diff": float(np.var(delta)),
        "std_diff": float(np.std(delta)),
        "mean_abs_diff": float(np.mean(np.abs(delta))),
        "max_abs_diff": float(np.max(np.abs(delta))),
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def expected_checkpoint_names() -> tuple[list[str], list[str]]:
    oof = [f"{member}__{fold}.npz" for fold in FOLDS for member in HISTORICAL_MEMBERS]
    test = [
        "hurdle_test.npz",
        "meta_raw_test.npz",
        "multiscale_direct_test.npz",
        "recent_direct_test.npz",
        "recent_dist_test.npz",
        "recent_hurdle_fast12_test.npz",
    ]
    return oof, test


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default=PREFIX_DEFAULT)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    teammate = root / "пайплайн сокомандника"
    results = root / "research" / "strategies" / "results" / args.prefix
    artifacts = root / "artifacts" / args.prefix
    results.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    if any(results.iterdir()) or any(artifacts.iterdir()):
        raise FileExistsError(f"Prefix already contains artifacts: {args.prefix}")

    combo_script = teammate / "research_scripts" / "continue_fixedstack_combo_10h.py"
    fixed_script = teammate / "research_scripts" / "run_best_bas_fixedstack_14h_v2.py"
    previous_script = teammate / "research_scripts" / "run_best_bas_research_23h.py"
    final_script = teammate / "research_scripts" / "continue_best_bas_final6h.py"
    combo_bundle = teammate / "review_bundles" / "fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654.zip"
    combo_root = teammate / "review_bundles" / "fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted"
    manifest_path = combo_root / "results" / "RUN_MANIFEST.json"
    final_manifest_path = (
        teammate
        / "review_bundles"
        / "final6h_REVIEW_BUNDLE_20260823_204823_extracted"
        / "results"
        / "RUN_MANIFEST.json"
    )
    validation_path = combo_root / "results" / "SHORTLIST_VALIDATION.csv"
    historical_path = combo_root / "submissions" / (
        "submission_combo10h_candidate_1_ridge_drop_recent_hurdle_stable18_s075.csv"
    )
    latest_root = teammate / "latest"
    submission_paths = {
        "historical": historical_path,
        "friend": latest_root / "components" / "friend.csv",
        "occ_meta_B": latest_root / "components" / "occ_meta_B.csv",
        "occ_raw_X3": latest_root / "components" / "occ_raw_X3.csv",
        "latest": latest_root / "latest.csv",
        "late_ridge_anchor": teammate
        / "review_bundles"
        / "final6h_REVIEW_BUNDLE_20260823_204823_extracted"
        / "submissions"
        / f"submission_final6h_A_{LATE_RIDGE_ANCHOR}.csv",
    }

    code = combo_script.read_text(encoding="utf-8")
    required_literals = (
        "StandardScaler(copy=False)",
        'Ridge(alpha=float(alpha), solver="lsqr", tol=1e-4)',
        "ridge__sample_weight",
        "d = np.clip(m.predict(Xt), -2.0, 2.0)",
        "float(shrink) * d",
        "include_meta=True",
    )
    missing_literals = [literal for literal in required_literals if literal not in code]
    if missing_literals:
        raise RuntimeError(f"Historical code literals missing: {missing_literals}")

    validation = pd.read_csv(validation_path)
    row = validation.loc[validation["name"] == HISTORICAL_NAME]
    if len(row) != 1:
        raise RuntimeError("Historical validation row is missing or duplicated")
    row = row.iloc[0]
    historical_experts = ast.literal_eval(str(row["notes"]).split("=", 1)[1])
    if historical_experts != list(HISTORICAL_MEMBERS):
        raise RuntimeError("Historical expert order changed")

    submissions: dict[str, dict[str, Any]] = {}
    vectors: dict[str, np.ndarray] = {}
    reference_uid: np.ndarray | None = None
    for name, path in submission_paths.items():
        uid, prediction, z = load_submission(path)
        if reference_uid is None:
            reference_uid = uid
        elif not np.array_equal(reference_uid, uid):
            raise RuntimeError(f"Submission alignment failure: {name}")
        vectors[name] = z
        submissions[name] = {
            "path": str(path.relative_to(root)),
            "sha256": sha256(path),
            "rows": int(len(uid)),
            "mean_z": float(z.mean()),
            "min_z": float(z.min()),
            "max_z": float(z.max()),
            "zero_predictions": int(np.sum(prediction == 0)),
        }

    latest_rebuilt = np.maximum(
        0.12 * vectors["friend"]
        + 0.16 * vectors["occ_meta_B"]
        + 0.72 * vectors["occ_raw_X3"],
        0.0,
    )
    latest_rebuild_error = float(np.max(np.abs(latest_rebuilt - vectors["latest"])))
    if latest_rebuild_error > 1e-12:
        raise RuntimeError("Authoritative latest TEST recipe no longer replays")

    historical_distances = {
        name: distance(vectors["historical"], vectors[name])
        for name in ("friend", "occ_meta_B", "occ_raw_X3", "latest", "late_ridge_anchor")
    }
    late_anchor_distances = {
        name: distance(vectors["late_ridge_anchor"], vectors[name])
        for name in ("occ_meta_B", "occ_raw_X3", "latest")
    }

    all_npz = list(root.rglob("*.npz"))
    by_name: dict[str, list[str]] = {}
    for path in all_npz:
        by_name.setdefault(path.name, []).append(str(path.relative_to(root)))
    expected_oof, expected_test = expected_checkpoint_names()
    found_oof = {name: by_name.get(name, []) for name in expected_oof if name in by_name}
    found_test = {name: by_name.get(name, []) for name in expected_test if name in by_name}
    missing_oof = [name for name in expected_oof if name not in by_name]
    missing_test = [name for name in expected_test if name not in by_name]

    # This only measures the numerical float roundtrip of the already-supplied
    # reference CSV. It is deliberately not reported as a model replay.
    z_reference = vectors["historical"]
    reference_self_roundtrip_error = float(
        np.max(np.abs(z_reference - np.log1p(np.maximum(np.expm1(z_reference), 0.0))))
    )

    source_hashes = {
        str(path.relative_to(root)): sha256(path)
        for path in (combo_script, fixed_script, previous_script, final_script, manifest_path, combo_bundle)
    }
    final_manifest = json.loads(final_manifest_path.read_text(encoding="utf-8"))

    historical_replay = {
        "status": "BLOCKED_HISTORICAL_REPLAY",
        "reference_submission": submissions["historical"],
        "required_floor": REPLAY_FLOOR,
        "formula_reconstructed": False,
        "formula_replay_max_abs_log_error": None,
        "reference_self_roundtrip_max_abs_log_error": reference_self_roundtrip_error,
        "reference_self_roundtrip_is_not_replay": True,
        "blocking_reason": (
            "The supplied bundle contains the final CSV and aggregate validation rows, "
            "but omits the exact row-level OOF/test prediction bank and meta_raw matrices "
            "used to fit and apply Ridge. Reusing the final CSV or algebraically inverting "
            "it would be circular, so no reconstructed_z is claimed."
        ),
        "missing_original_oof_checkpoints": missing_oof,
        "missing_original_test_checkpoints": missing_test,
        "found_original_oof_checkpoints": found_oof,
        "found_original_test_checkpoints": found_test,
        "all_npz_scanned": len(all_npz),
        "historical_recipe": {
            "fold_scheme": "expanding walk-forward over four folds; fold 0 is table_core passthrough",
            "folds": list(FOLDS),
            "train_validation_alignment": "each member aligned by exact user_id set to cap checkpoint order",
            "target": "log1p(y) - table_core",
            "members_in_exact_order": list(HISTORICAL_MEMBERS),
            "meta_raw": "enabled; up to 72 raw activity/recency columns from cap checkpoint",
            "prediction_derived_columns": [
                "table_core",
                "p",
                "mu",
                "log1p(mu)",
                "p*(1-p)",
                "member_mean",
                "member_std",
                "member_min",
                "member_max",
                "member_range",
                "member_i-table_core for each of 8 members",
            ],
            "standardization": "StandardScaler(copy=False), unweighted fit on stacked donor rows",
            "ridge": {
                "alpha": 150.0,
                "solver": "lsqr",
                "tol": 0.0001,
                "fit_intercept": True,
                "sample_weight": "fold weight passed only to Ridge step",
            },
            "recency_weights": dict(zip(FOLDS, FOLD_WEIGHTS)),
            "correction_clip": [-2.0, 2.0],
            "historical_scale": 0.75,
            "table_clip": [0.0, 20.0],
            "friend_transform": "friend + 0.55*(candidate_table-table_core)",
            "global_level": 2.3293,
            "global_level_policy": "clip [0,20], add level-mean(z), clip [0,20] once",
            "csv_policy": "predict=max(expm1(clip(z,0,20)),0); pandas to_csv(index=False)",
        },
        "validation_record": {
            "wcv": float(row["wcv"]),
            "base_wcv": float(row["base_wcv"]),
            "delta": float(row["delta"]),
            "wins": int(row["wins"]),
            "wins_recent": int(row["wins_recent"]),
            "latest_delta": float(row["latest_delta"]),
            "fold_scores": ast.literal_eval(row["fold_scores"]),
            "fold_deltas": ast.literal_eval(row["fold_deltas"]),
            "warning": "This historical walk-forward record is not canonical outer LOFO against latest.",
        },
        "public_lb": {
            "value": float(final_manifest["known_ridge_submission_public"]),
            "status": "RECORDED_BUT_NOT_SHA_BOUND",
            "evidence": "final6h RUN_MANIFEST known_ridge_submission_public and provenance text",
            "assigned_to_exact_csv": False,
        },
        "source_hashes": source_hashes,
    }
    dump_json(results / "historical_replay.json", historical_replay)

    member_rows = []
    definitions = {
        "cap": "fixed capped direct component; also carries meta_raw on OOF",
        "unc": "fixed uncapped direct component",
        "dist": "fixed distribution-head component",
        "hurdle": "two-part p*mu helper; supplies p and mu",
        "multiscale_direct": "L=180 multiscale direct, 650 rounds",
        "recent_direct": "L=0 norm_long direct, tau=105, 650 rounds",
        "recent_dist": "L=0 norm_long dist, tau=120, 330 rounds",
        "recent_hurdle_fast12": "last 12 eligible cutoffs, tau=70, two-part, 430 rounds",
    }
    for position, name in enumerate(HISTORICAL_MEMBERS):
        member_rows.append(
            {
                "position": position,
                "name": name,
                "definition": definitions[name],
                "historical_oof_available": False,
                "historical_test_available": name in {"cap", "unc", "dist"},
                "included_in_historical_winner": True,
            }
        )
    member_manifest = {
        "historical_members": member_rows,
        "excluded_despite_name": {
            "recent_hurdle_stable18": "explicitly dropped by winning subset",
        },
        "raw_meta_columns": {
            "included": True,
            "maximum_count": 72,
            "exact_names_available": False,
            "implication": "Historical public-associated winner is not prediction-only.",
        },
        "lineage": {
            "friend": "does not include historical winner; byte-identical STRONGEST_CURRENT",
            "occ_meta_B": (
                f"does not directly include {HISTORICAL_NAME}; descends from distinct {LATE_RIDGE_ANCHOR}"
            ),
            "occ_raw_X3": (
                f"does not directly include {HISTORICAL_NAME}; descends from distinct {LATE_RIDGE_ANCHOR}"
            ),
            "latest": "no direct historical-winner input; indirect later Ridge/greedy ancestry through B and X3",
            "late_anchor_difference": (
                "later anchor uses all 9 finalizable members including stable18, fold weights^(1.7), "
                "then adaptive blend with greedy35 (prior .85, lambda .12)"
            ),
        },
        "test_only_redundancy": {
            "historical_vs": historical_distances,
            "late_anchor_vs": late_anchor_distances,
            "latest_rebuild_max_abs_log_error": latest_rebuild_error,
            "oof_residual_metrics_available": False,
        },
        "submission_hashes": submissions,
    }
    dump_json(results / "member_manifest.json", member_manifest)

    nested_rows = []
    for formulation in ("FULL_RECENCY_RIDGE", "PREDICTIONS_ONLY", "SHUFFLED_RECENT"):
        for fold in FOLDS:
            nested_rows.append(
                {
                    "formulation": formulation,
                    "fold": fold,
                    "status": "BLOCKED_NOT_RUN",
                    "base_rmsle_cal": "",
                    "candidate_rmsle_cal": "",
                    "delta": "",
                    "selected_scale": "",
                    "condition_number": "",
                    "reason": "historical replay failed before Phase C; canonical latest OOF also missing",
                }
            )
    write_csv(
        results / "nested_lofo.csv",
        [
            "formulation",
            "fold",
            "status",
            "base_rmsle_cal",
            "candidate_rmsle_cal",
            "delta",
            "selected_scale",
            "condition_number",
            "reason",
        ],
        nested_rows,
    )

    write_csv(
        results / "coefficients.csv",
        ["outer_fold", "member", "coefficient", "sign", "stability", "status", "reason"],
        [
            {
                "outer_fold": "",
                "member": "",
                "coefficient": "",
                "sign": "",
                "stability": "",
                "status": "BLOCKED_NOT_RUN",
                "reason": "missing exact historical matrices and canonical latest OOF",
            }
        ],
    )
    write_csv(
        results / "controls.csv",
        ["comparison", "status", "delta_wcv", "folds_improved", "latest_fold_delta", "reason"],
        [
            {
                "comparison": comparison,
                "status": "BLOCKED_NOT_RUN",
                "delta_wcv": "",
                "folds_improved": "",
                "latest_fold_delta": "",
                "reason": "Phase A exact replay gate failed",
            }
            for comparison in (
                "FULL_RECENCY_RIDGE-latest",
                "PREDICTIONS_ONLY-latest",
                "FULL_RECENCY_RIDGE-PREDICTIONS_ONLY",
                "FULL_RECENCY_RIDGE-SHUFFLED_RECENT",
            )
        ],
    )
    segment_names = [
        "zero",
        "positive",
        "AUC(y>0)",
        "rec_buy:never",
        "rec_buy:0-15",
        "rec_buy:15-60",
        "rec_buy:60+",
        "w180_days_buy:0-1",
        "w180_days_buy:2-15",
        "w180_days_buy:16+",
        "user_hash_half:0",
        "user_hash_half:1",
    ]
    write_csv(
        results / "segments.csv",
        ["segment", "status", "n", "base_metric", "candidate_metric", "delta", "reason"],
        [
            {
                "segment": segment,
                "status": "BLOCKED_NOT_RUN",
                "n": "",
                "base_metric": "",
                "candidate_metric": "",
                "delta": "",
                "reason": "Phase A exact replay gate failed; no canonical latest OOF",
            }
            for segment in segment_names
        ],
    )

    test_regime = {
        "status": "NOT_RUN",
        "historical_reference_csv_available": True,
        "historical_formula_replay_passed": False,
        "canonical_latest_oof_available": False,
        "production_parity": "UNRESOLVED",
        "CAP_LINEAGE": "UNKNOWN",
        "latest_level": submissions["latest"]["mean_z"],
        "fixed_level_reference": 2.3293,
        "test_only_redundancy": historical_distances,
        "blocked_diagnostics": [
            "Var(correction_test)/Var(correction_oof)",
            "quantile support",
            "activity-bin support",
            "max leverage",
            "outside-OOF-range share",
        ],
    }
    dump_json(results / "test_regime.json", test_regime)

    np.savez_compressed(
        artifacts / "oof_predictions.npz",
        user_id=np.asarray([], dtype=np.int64),
        fold=np.asarray([], dtype="U10"),
        target=np.asarray([], dtype=np.float64),
        z_historical=np.asarray([], dtype=np.float64),
        z_latest=np.asarray([], dtype=np.float64),
        status=np.asarray(["BLOCKED_HISTORICAL_REPLAY"], dtype="U40"),
    )

    summary = {
        "prefix": args.prefix,
        "verdict": "BLOCKED_HISTORICAL_REPLAY",
        "phase_A": "FAILED: exact input bank absent; no circular replay claimed",
        "phase_B": {
            "status": "TEST_ONLY_AUDIT_COMPLETE; OOF_RESIDUAL_AUDIT_BLOCKED",
            "historical_directly_in_friend": False,
            "historical_directly_in_occ_meta_B": False,
            "historical_directly_in_occ_raw_X3": False,
            "historical_directly_in_latest": False,
            "later_ridge_greedy_ancestry_in_occ_meta_B_occ_raw_X3_latest": True,
            "corr_historical_latest_test": historical_distances["latest"]["corr"],
            "var_historical_minus_latest_test": historical_distances["latest"]["var_diff"],
        },
        "phase_C": "NOT_RUN",
        "controls": "NOT_RUN",
        "test_regime": "NOT_RUN",
        "submission_created": False,
        "leaderboard_upload": False,
        "canonical_latest_oof_available": False,
        "CAP_LINEAGE": "UNKNOWN",
        "public_lb_exact_csv": None,
        "public_lb_family_recorded_unbound": 1.6492897556391737,
    }
    dump_json(results / "summary.json", summary)

    report = f"""# RECENCY-RIDGE-ON-PREDICTIONS — report

## Outcome

**Verdict: `BLOCKED_HISTORICAL_REPLAY`.** The historical reference CSV is present,
but the exact row-level OOF/test prediction bank and `meta_raw` matrices used to fit
the Ridge are absent. The required formula replay error therefore cannot be computed;
the `{REPLAY_FLOOR:g}` gate is not claimed from a circular read of the final CSV.

No new Ridge, lambda/scale search, canonical LOFO, controls, test candidate, submission,
or leaderboard upload was run.

## Phase A — what was recovered exactly

- Reference: `{submissions['historical']['path']}`
- SHA-256: `{submissions['historical']['sha256']}`; rows: 250,000; exact common user order.
- Historical table member order: `{', '.join(HISTORICAL_MEMBERS)}`.
- Target: `log1p(y) - table_core`; expanding prior-fold walk-forward, with the first
  fold equal to `table_core` (not four-fold outer LOFO).
- Ridge: `StandardScaler(copy=False)` then `Ridge(alpha=150, solver='lsqr',
  tol=1e-4, fit_intercept=True)`; fold weights `1:2:4:8` go only to Ridge;
  correction clipped to `[-2,2]`, fixed scale `0.75`.
- Finalization: `candidate_table=clip(table_core+0.75*d,0,20)`, then
  `friend+0.55*(candidate_table-table_core)`, fixed mean level `2.3293`, one final clip,
  `predict=expm1(z)`, pandas CSV serialization.
- `recent_dist`: norm-long dist, temporal tau 120, 330 rounds.
- `recent_hurdle_fast12`: two-part model on the last 12 eligible cutoffs, tau 70,
  430 rounds. Despite the winner name, `recent_hurdle_stable18` was dropped.

The external description “prediction-level Ridge” is not exact: the winning recipe
has `include_meta=True` and prepends up to 72 raw activity/recency columns. The separate
`ridge_predonly_finalizable` is a different candidate. Exact meta column names and fitted
coefficients are unavailable because the cap OOF checkpoints were omitted.

The source tree contains {len(all_npz)} NPZ files, but zero of the {len(expected_oof)}
expected historical OOF checkpoints and zero of the {len(expected_test)} missing helper
test checkpoints. The package does retain ready CAP/UNC/DIST TEST arrays, which are
insufficient to refit Ridge without donor OOF, targets, `p/mu`, and `meta_raw`.

The number `1.6492897556391737` is recorded as `known_ridge_submission_public` in the
later Final6h manifest, but no SHA-to-score row binds it to the exact reference CSV.
It is therefore recorded as family-level/unbound evidence, not as verified LB for SHA
`{submissions['historical']['sha256']}`.

## Phase B — redundancy audit

The exact historical winner is not a direct component of `friend`, `occ_meta_B`,
`occ_raw_X3`, or `latest`. `friend` is byte-identical to `STRONGEST_CURRENT` and predates
the Ridge. Both occurrence components instead descend from a later, distinct anchor:
`{LATE_RIDGE_ANCHOR}`. That anchor uses all nine finalizable members (including stable18),
weights folds as `(1:2:4:8)^1.7`, then adaptively blends Ridge with greedy35.

Nevertheless the historical function is nearly absorbed on TEST: against `latest`,
`corr={historical_distances['latest']['corr']:.12f}` and
`Var(z_historical-z_latest)={historical_distances['latest']['var_diff']:.12g}`.
Against the later Ridge/greedy anchor, `corr={historical_distances['late_ridge_anchor']['corr']:.12f}`
and variance difference is `{historical_distances['late_ridge_anchor']['var_diff']:.12g}`.
These are production-TEST geometry diagnostics only. Target residual correlations and
`corr(z_ridge-z_latest, log1p(y)-z_latest)` are unavailable without canonical aligned OOF.

The authoritative TEST `latest` recipe replays independently from friend/B/X3 to
`{latest_rebuild_error:.3g}` max log error, but canonical row-level OOF for B/X3 is still
missing, consistent with `exp_066/067`.

## Phases C and test regime

Not run. Phase A failed first, and the independent prerequisite—canonical four-fold OOF
`latest`—is also absent. `CAP_LINEAGE=UNKNOWN`; private safety and production parity
remain unresolved. All requested downstream CSV/JSON/NPZ artifacts are explicit blocked
markers rather than synthetic metrics.
"""
    (results / "REPORT.md").write_text(report, encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
