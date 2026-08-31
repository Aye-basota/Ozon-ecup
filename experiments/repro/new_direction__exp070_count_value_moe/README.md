# EXP070_COUNT_VALUE_MOE — final report

## Catalogue metadata

- **Catalogue ID:** `new_direction__exp070_count_value_moe`
- **Namespace:** `new_direction`
- **Experiment ID:** `EXP070_COUNT_VALUE_MOE`
- **Original source:** `research/new_directions/EXP070_COUNT_VALUE_MOE`
- **Source ref:** `origin/team-a late research package`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** late research direction / experiment package
- **Model:** LightGBM, Ridge
- **Features:** calendar features, recency, 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** This is an operationally conservative rejection: the latest-fold pilot passed, but exact fixed training required about 45 minutes per largest fold pair. The two-hour hard stop was reached before `2025-10-02`; therefore canonical four-fold wCV and honest three-donor LOFO do not exist. No PASS or WEAK_SIGNAL claim is inferred from an incomplete fold set.
- **Known score:** The diagnostic three-fold `1:2:8` replacement delta is `-0.000086647` and real-minus-shuffled is `-0.000052609`. This is explicitly **not canonical wCV** and is not a selection result.
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when the data/frozen artifacts named by the report are present
- **Notes:** Directory-level audit unit: 24 files, 2 launcher/helper scripts, 1 preserved report documents. Numeric claims are copied from those reports.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# EXP070_COUNT_VALUE_MOE — final report

## 1. Verdict

**REJECT**

Final recommendation: **DO_NOT_ADD**.

This is an operationally conservative rejection: the latest-fold pilot passed, but exact fixed training required about 45 minutes per largest fold pair. The two-hour hard stop was reached before `2025-10-02`; therefore canonical four-fold wCV and honest three-donor LOFO do not exist. No PASS or WEAK_SIGNAL claim is inferred from an incomplete fold set.

## 2. Exact count label and bins

`N30` is the number of distinct stored calendar dates in `(T,T+30]` with `gmv > 0`. The oldest-fold training panel had C4 frequency `3.135003%`, above the `0.5%` fallback threshold, so the frozen bins remained C0=0, C1=1, C2=2–3, C3=4–7, C4>=8.

## 3. Label/leakage audit

- Deterministic slow reference: `PASS` on 1,000 rows.
- Features: exact cached 227-column normalized-long S1-E10 matrices, built only through `event_date <= T`.
- Targets: `(T,T+30]`; every fitted training cutoff obeyed `T+30 <= V`; b1 training and b3 validation panels were used.
- Canonical row keys and targets aligned on every completed fold.

## 4. Standalone real and shuffled results

The standalone count-value MoE was worse than EXP-037 on the pilot by `+0.002120674`, but better than its shuffled control by `-0.000379716`. All completed-fold standalone results are in `fold_metrics.csv`; matched results are in `real_vs_shuffled.csv`.

## 5. Nested replacement/add-one results

Not run: honest LOFO requires all four held-out folds, and `2025-10-02` was not trained before the runtime hard stop. No alpha or beta was selected.

## 6. Per-fold and latest-fold deltas

- `2025-09-04`: replacement beta=1 delta vs EXP-037 `-0.000102453`, real-minus-shuffled `+0.000180104`.
- `2025-09-18`: replacement beta=1 delta vs EXP-037 `+0.000032696`, real-minus-shuffled `+0.000086296`.
- `2025-10-16`: replacement beta=1 delta vs EXP-037 `-0.000114507`, real-minus-shuffled `-0.000116425`.

The diagnostic three-fold `1:2:8` replacement delta is `-0.000086647` and real-minus-shuffled is `-0.000052609`. This is explicitly **not canonical wCV** and is not a selection result.

## 7. Probability calibration diagnostics

Raw multiclass probabilities were used. Probability audit: `PASS`. Log loss, Brier, class-wise OVR AUC, ECE, p0 deciles, and observed/predicted incidence are in `probability_metrics.csv` for every completed fold.

## 8. Residual-segment interpretation

`segment_metrics.csv` reports fixed beta=1 diagnostics for target zero/positive, real/predicted count classes, historical purchase days (including 2–15), recency bins, EXP-037 level, and DIST/count disagreement. They are explanatory only; no segment correction was selected.

## 9. OOF correction novelty

Pairwise correlations and RMS log differences on the three completed folds are in `diversity_oof.csv`. Donor-fold ridge projection was not run because a canonical four-fold correction vector does not exist.

## 10. TEST distance outside the geometry span

Not run. No PASS candidate was produced, no TEST count-value vector was trained, and geometry weights were not touched.

## 11. Runtime and disk usage

- Fixed model run stopped at `6984.0` seconds (`116.4` minutes), before the 7,200-second hard ceiling.
- New persistent artifacts at report time: `100937128` bytes, below 3 GB.
- Six physical cores / six LightGBM threads.

## 12. Exact OOF/TEST artifact paths and SHA256

- `C:\Users\Admin\Desktop\e-cup-research-clean\research\new_directions\EXP070_COUNT_VALUE_MOE\count_value_moe_raw_OOF.parquet` — `6e6289e9a4e92c804e43a346dd5d08844b7dd60011874ef078f3c107bd55fc50` (three completed folds; diagnostic partial OOF).
- `C:\Users\Admin\Desktop\e-cup-research-clean\research\new_directions\EXP070_COUNT_VALUE_MOE\count_probabilities_OOF.parquet` — `930a1ca915d6c81ce4122a793d4c8f330231a6b3898208d720c4d26958def605` (three completed folds).
- Standardized PASS OOF/TEST: not produced.
- TEST CSV: not produced.

All experiment-local hashes are in `checksums.sha256`.

## 13. Final recommendation

**DO_NOT_ADD**

Do not resume with reduced rounds, fewer rows, altered folds, or missing placebo arms under EXP070. A future rerun would need a larger explicit runtime budget while retaining this frozen configuration.
