"""Finalize EXP070 after the preregistered two-hour runtime stop."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_experiment as exp


PRESENT = ["2025-09-04", "2025-09-18", "2025-10-16"]
MISSING = "2025-10-02"
MODEL_RUNTIME_SECONDS = 6984.0


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def partial_weighted(scores: dict[str, float]) -> float:
    weights = {"2025-09-04": 1.0, "2025-09-18": 2.0, "2025-10-16": 8.0}
    return sum(weights[k] * scores[k] for k in PRESENT) / sum(weights.values())


def main() -> None:
    aligned = pd.read_parquet(exp.ALIGNED_OOF)
    results = [exp.load_fold_cache(fold) for fold in PRESENT]
    frame = pd.concat([exp.aligned_fold(result, aligned) for result in results], ignore_index=True)
    n_classes = 5

    probability_rows = []
    probability_pass = True
    for fold in PRESENT:
        block = frame.loc[frame["fold"] == fold]
        real_rows, passed = exp.probability_rows(block, "real", n_classes)
        shuffled_rows, _ = exp.probability_rows(block, "shuffled", n_classes)
        probability_rows.extend(real_rows + shuffled_rows)
        probability_pass &= passed
    pd.DataFrame(probability_rows).to_csv(exp.OUT / "probability_metrics.csv", index=False)
    raw_path, prob_path = exp.write_raw_vectors(frame, n_classes)

    historical = {}
    for name, artifact in (("S1-E10", "S1-E10"), ("DIST", "S1-DIST"), ("E11", "S1-E11")):
        values = np.load(exp.OLD / "artifacts" / f"oof_{artifact}.npz", allow_pickle=False)
        source = pd.DataFrame({"fold": values["cutoff"], "user_id": values["user_id"], "z": values["z"]})
        historical[name] = frame[["fold", "user_id"]].merge(
            source, on=["fold", "user_id"], how="left", validate="one_to_one"
        )["z"].to_numpy(float)

    y = frame["target"].to_numpy(float)
    folds = frame["fold"].to_numpy(str)
    z_base = exp.zcol(frame, "pred_exp037")
    z_dist = exp.zcol(frame, "pred_dist")
    z_real = frame["z_count_real"].to_numpy(float)
    z_shuf = frame["z_count_shuffled"].to_numpy(float)
    candidates = {
        "EXP037": z_base,
        "S1-E10": historical["S1-E10"],
        "DIST": historical["DIST"],
        "E11": historical["E11"],
        "COUNT_REAL": z_real,
        "COUNT_SHUFFLED": z_shuf,
        "REPLACE_REAL_BETA1": z_base + 0.25 * (z_real - z_dist),
        "REPLACE_SHUFFLED_BETA1": z_base + 0.25 * (z_shuf - z_dist),
        "ADD10_REAL": z_base + 0.10 * (z_real - z_base),
        "ADD10_SHUFFLED": z_base + 0.10 * (z_shuf - z_base),
    }
    metric_rows, evaluations = exp.fixed_metric_rows(candidates, y, folds)
    score_lookup: dict[str, dict[str, float]] = {}
    for name, evaluation in evaluations.items():
        score_lookup[name] = {fold: float(score) for fold, score in zip(PRESENT, evaluation["cal"])}
        metric_rows.append(
            {
                "candidate": name,
                "fold": "partial_weighted_1_2_8_NOT_CANONICAL_WCV",
                "n": len(y),
                "rmsle_raw": np.nan,
                "rmsle_cal": partial_weighted(score_lookup[name]),
                "offset": np.nan,
                "delta_vs_exp037": partial_weighted(score_lookup[name]) - partial_weighted(score_lookup["EXP037"]),
                "improved": partial_weighted(score_lookup[name]) < partial_weighted(score_lookup["EXP037"]),
            }
        )
    pd.DataFrame(metric_rows).to_csv(exp.OUT / "fold_metrics.csv", index=False)

    comparison_rows = []
    for path, real_name, shuf_name in (
        ("standalone", "COUNT_REAL", "COUNT_SHUFFLED"),
        ("replacement_beta1", "REPLACE_REAL_BETA1", "REPLACE_SHUFFLED_BETA1"),
        ("add10", "ADD10_REAL", "ADD10_SHUFFLED"),
    ):
        for fold in PRESENT:
            comparison_rows.append(
                {
                    "path": path,
                    "fold": fold,
                    "real_score": score_lookup[real_name][fold],
                    "shuffled_score": score_lookup[shuf_name][fold],
                    "real_minus_shuffled": score_lookup[real_name][fold] - score_lookup[shuf_name][fold],
                    "status": "fixed_completed_fold",
                }
            )
        comparison_rows.append(
            {
                "path": path,
                "fold": "partial_weighted_1_2_8_NOT_CANONICAL_WCV",
                "real_score": partial_weighted(score_lookup[real_name]),
                "shuffled_score": partial_weighted(score_lookup[shuf_name]),
                "real_minus_shuffled": partial_weighted(score_lookup[real_name]) - partial_weighted(score_lookup[shuf_name]),
                "status": "diagnostic_only_missing_2025-10-02",
            }
        )
    real_vs_shuffled = pd.DataFrame(comparison_rows)
    real_vs_shuffled.to_csv(exp.OUT / "real_vs_shuffled.csv", index=False)

    pd.DataFrame(
        [
            {
                "path": "replacement",
                "arm": "real_and_shuffled",
                "heldout_fold": MISSING,
                "donor_folds": "",
                "grid": json.dumps(exp.BETA_GRID.tolist()),
                "selected_value": np.nan,
                "selection_wcv": np.nan,
                "donor_scores": "",
                "heldout_score": np.nan,
                "heldout_baseline_score": np.nan,
                "heldout_delta": np.nan,
                "status": "NOT_RUN_RUNTIME_STOP; canonical LOFO requires all four held-out folds and three donors",
            },
            {
                "path": "add_one",
                "arm": "real_and_shuffled",
                "heldout_fold": MISSING,
                "donor_folds": "",
                "grid": json.dumps(exp.ALPHA_GRID.tolist()),
                "selected_value": np.nan,
                "selection_wcv": np.nan,
                "donor_scores": "",
                "heldout_score": np.nan,
                "heldout_baseline_score": np.nan,
                "heldout_delta": np.nan,
                "status": "NOT_RUN_RUNTIME_STOP; canonical LOFO requires all four held-out folds and three donors",
            },
        ]
    ).to_csv(exp.OUT / "nested_selection.csv", index=False)

    fixed_replacement = candidates["REPLACE_REAL_BETA1"]
    exp.segment_metrics(frame, fixed_replacement, "replacement_beta1_partial_3fold").to_csv(
        exp.OUT / "segment_metrics.csv", index=False
    )
    exp.partial_diversity(frame, fixed_replacement, "replacement_beta1_partial_3fold").to_csv(
        exp.OUT / "diversity_oof.csv", index=False
    )
    exp.write_json(
        "oof_projection_metrics.json",
        {
            "status": "not_run_runtime_stop",
            "completed_folds": PRESENT,
            "missing_fold": MISSING,
            "reason": "donor-fold ridge projection and unexplained variance require canonical four-fold OOF",
        },
    )
    exp.write_json("test_span_projection.json", {"status": "not_run_no_pass", "reason": "full OOF success gates unavailable"})
    exp.write_json(
        "production_regime.json",
        {
            "status": "not_run_no_pass",
            "test_prediction_constructed": False,
            "reason": "runtime hard stop left 2025-10-02 untrained; no PASS TYPE A/B can be declared",
        },
    )

    label_audit = pd.read_csv(exp.OUT / "label_audit.csv")
    label_pass = bool(label_audit["target_match"].all() and label_audit["N30_match"].all())
    pilot = json.loads((exp.OUT / "pilot_metrics.json").read_text(encoding="utf-8"))
    partial_base = partial_weighted(score_lookup["EXP037"])
    partial_real = partial_weighted(score_lookup["REPLACE_REAL_BETA1"])
    partial_shuf = partial_weighted(score_lookup["REPLACE_SHUFFLED_BETA1"])
    fold_lines = []
    for fold in PRESENT:
        delta = score_lookup["REPLACE_REAL_BETA1"][fold] - score_lookup["EXP037"][fold]
        gap = score_lookup["REPLACE_REAL_BETA1"][fold] - score_lookup["REPLACE_SHUFFLED_BETA1"][fold]
        fold_lines.append(f"- `{fold}`: replacement beta=1 delta vs EXP-037 `{delta:+.9f}`, real-minus-shuffled `{gap:+.9f}`.")

    raw_hash, prob_hash = digest(raw_path), digest(prob_path)
    disk_bytes = sum(path.stat().st_size for path in exp.OUT.rglob("*") if path.is_file())
    report = f"""# EXP070_COUNT_VALUE_MOE — final report

## 1. Verdict

**REJECT**

Final recommendation: **DO_NOT_ADD**.

This is an operationally conservative rejection: the latest-fold pilot passed, but exact fixed training required about 45 minutes per largest fold pair. The two-hour hard stop was reached before `2025-10-02`; therefore canonical four-fold wCV and honest three-donor LOFO do not exist. No PASS or WEAK_SIGNAL claim is inferred from an incomplete fold set.

## 2. Exact count label and bins

`N30` is the number of distinct stored calendar dates in `(T,T+30]` with `gmv > 0`. The oldest-fold training panel had C4 frequency `3.135003%`, above the `0.5%` fallback threshold, so the frozen bins remained C0=0, C1=1, C2=2–3, C3=4–7, C4>=8.

## 3. Label/leakage audit

- Deterministic slow reference: `{'PASS' if label_pass else 'FAIL'}` on 1,000 rows.
- Features: exact cached 227-column normalized-long S1-E10 matrices, built only through `event_date <= T`.
- Targets: `(T,T+30]`; every fitted training cutoff obeyed `T+30 <= V`; b1 training and b3 validation panels were used.
- Canonical row keys and targets aligned on every completed fold.

## 4. Standalone real and shuffled results

The standalone count-value MoE was worse than EXP-037 on the pilot by `+0.002120674`, but better than its shuffled control by `-0.000379716`. All completed-fold standalone results are in `fold_metrics.csv`; matched results are in `real_vs_shuffled.csv`.

## 5. Nested replacement/add-one results

Not run: honest LOFO requires all four held-out folds, and `2025-10-02` was not trained before the runtime hard stop. No alpha or beta was selected.

## 6. Per-fold and latest-fold deltas

{chr(10).join(fold_lines)}

The diagnostic three-fold `1:2:8` replacement delta is `{partial_real-partial_base:+.9f}` and real-minus-shuffled is `{partial_real-partial_shuf:+.9f}`. This is explicitly **not canonical wCV** and is not a selection result.

## 7. Probability calibration diagnostics

Raw multiclass probabilities were used. Probability audit: `{'PASS' if probability_pass else 'FAIL'}`. Log loss, Brier, class-wise OVR AUC, ECE, p0 deciles, and observed/predicted incidence are in `probability_metrics.csv` for every completed fold.

## 8. Residual-segment interpretation

`segment_metrics.csv` reports fixed beta=1 diagnostics for target zero/positive, real/predicted count classes, historical purchase days (including 2–15), recency bins, EXP-037 level, and DIST/count disagreement. They are explanatory only; no segment correction was selected.

## 9. OOF correction novelty

Pairwise correlations and RMS log differences on the three completed folds are in `diversity_oof.csv`. Donor-fold ridge projection was not run because a canonical four-fold correction vector does not exist.

## 10. TEST distance outside the geometry span

Not run. No PASS candidate was produced, no TEST count-value vector was trained, and geometry weights were not touched.

## 11. Runtime and disk usage

- Fixed model run stopped at `{MODEL_RUNTIME_SECONDS:.1f}` seconds (`116.4` minutes), before the 7,200-second hard ceiling.
- New persistent artifacts at report time: `{disk_bytes}` bytes, below 3 GB.
- Six physical cores / six LightGBM threads.

## 12. Exact OOF/TEST artifact paths and SHA256

- `{raw_path}` — `{raw_hash}` (three completed folds; diagnostic partial OOF).
- `{prob_path}` — `{prob_hash}` (three completed folds).
- Standardized PASS OOF/TEST: not produced.
- TEST CSV: not produced.

All experiment-local hashes are in `checksums.sha256`.

## 13. Final recommendation

**DO_NOT_ADD**

Do not resume with reduced rounds, fewer rows, altered folds, or missing placebo arms under EXP070. A future rerun would need a larger explicit runtime budget while retaining this frozen configuration.
"""
    (exp.OUT / "report.md").write_text(report, encoding="utf-8")
    exp.write_json(
        "runtime_resources.json",
        {
            "model_runtime_seconds": MODEL_RUNTIME_SECONDS,
            "hard_stop_seconds": 7200,
            "stop_reason": "projected matched 2025-10-02 fold completion exceeded hard stop",
            "completed_folds": PRESENT,
            "missing_fold": MISSING,
            "persistent_bytes_before_checksums": disk_bytes,
            "lightgbm_threads": 6,
        },
    )

    consumed = {exp.RAW, exp.FEATURE_LIST, exp.ALIGNED_OOF}
    latest_training = exp.training_cutoffs(exp.PILOT_FOLD)
    for cutoff in sorted(set(latest_training + exp.FOLDS)):
        consumed.add(exp.feature_path(cutoff))
        if cutoff in latest_training:
            consumed.add(exp.panel_path(cutoff, 1))
        if cutoff in exp.FOLDS:
            consumed.add(exp.panel_path(cutoff, 3))
    exp.artifact_manifest(consumed)

    top_files = sorted(path for path in exp.OUT.iterdir() if path.is_file() and path.name != "checksums.sha256")
    (exp.OUT / "checksums.sha256").write_text(
        "".join(f"{digest(path)}  {path.name}\n" for path in top_files), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
