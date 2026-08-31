# Logged run — OPEN_FUNNEL_EXP061

## Catalogue metadata

- **Catalogue ID:** `team_a_run__open_funnel_exp061`
- **Namespace:** `team_a_run`
- **Experiment ID:** `OPEN_FUNNEL_EXP061`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** logged run / arm
- **Model:** Unknown / not recoverable from repository history
- **Features:** funnel features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** conclusion:** Leakage/alignment/shuffle audits PASS; raw residual corr up to |0.0223| but REAL SHUFFLED and CONTROL_ONLY all selected scale 0 on every fold and both halves. Delta wCV 0; late 0; REAL-SHUFFLED 0. Full model/test/LB/submission not run. Details exp_061
- **Known score:** conclusion:** Leakage/alignment/shuffle audits PASS; raw residual corr up to |0.0223| but REAL SHUFFLED and CONTROL_ONLY all selected scale 0 on every fold and both halves. Delta wCV 0; late 0; REAL-SHUFFLED 0. Full model/test/LB/submission not run. Details exp_061
- **Seed:** params:** {"base":"STRONGEST_CURRENT","new_features":14,"rounds":120,"num_leaves":15,"min_data_in_leaf":1000,"lambda_l2":50,"scales":[0,0.25,0.5,0.75,1],"shuffle":"joint within 392-396 baseline-state strata","split":"splitmix64(user_id) 4-way + two-sided scale","seed":42}
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — OPEN_FUNNEL_EXP061

This run was recovered from `experiments/log.csv`.

- **exp_id:** OPEN_FUNNEL_EXP061
- **timestamp:** 2026-08-25T02:42:00+03:00
- **commit:** a28a71f
- **description:** exp_061 cutoff-safe unresolved Search/Cart state after last purchase vs joint within-state shuffled control
- **scenario:** S1-preflight
- **n_features:** 241
- **model:** artifact-only-cross-user-lgb-residual
- **params:** {"base":"STRONGEST_CURRENT","new_features":14,"rounds":120,"num_leaves":15,"min_data_in_leaf":1000,"lambda_l2":50,"scales":[0,0.25,0.5,0.75,1],"shuffle":"joint within 392-396 baseline-state strata","split":"splitmix64(user_id) 4-way + two-sided scale","seed":42}
- **cutoffs:** val 2025-09-04/09-18/10-02/10-16
- **L:** 90
- **panel_blocks:** 3
- **fold_scores:** [1.766883356,1.760509577,1.748629224,1.741278566]
- **cv_mean:** 1.754325181
- **cv_std:** 0.009995
- **bias_mean:** Unknown / not recoverable from repository history
- **best_offset:** Unknown / not recoverable from repository history
- **cv_mean_calib:** 1.754325181
- **delta_vs_b0:** Unknown / not recoverable from repository history
- **runtime_s:** 50.9
- **verdict:** REJECT
- **conclusion:** Leakage/alignment/shuffle audits PASS; raw residual corr up to |0.0223| but REAL SHUFFLED and CONTROL_ONLY all selected scale 0 on every fold and both halves. Delta wCV 0; late 0; REAL-SHUFFLED 0. Full model/test/LB/submission not run. Details exp_061
- **wcv:** 1.747509862
- **fold_cal:** [1.766883356,1.760509577,1.748629224,1.741278566]
- **mean_z:** Unknown / not recoverable from repository history
- **lb_public:** Unknown / not recoverable from repository history
