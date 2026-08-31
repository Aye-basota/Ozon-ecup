# Logged run — RESDISC_053

## Catalogue metadata

- **Catalogue ID:** `team_a_run__resdisc_053`
- **Namespace:** `team_a_run`
- **Experiment ID:** `RESDISC_053`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** logged run / arm
- **Model:** ensemble
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** wcv:** 1.747509863
- **Known score:** wcv:** 1.747509863
- **Seed:** conclusion:** Exact artifact audit PASS. ETX-vs-SEQ oracle 0.020172 is below seed-null 0.031483 (semantic excess -0.011311). Winner AUC 0.526759 but bounded gate late delta only -0.000006419; residual donor scales 0/0 and delta ~0; full LOFO/test/LB/submission not run. Details exp_053
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — RESDISC_053

This run was recovered from `experiments/log.csv`.

- **exp_id:** RESDISC_053
- **timestamp:** 2026-08-24T14:03:23
- **commit:** a28a71f
- **description:** exp_053 artifact-only residual signal discovery: exact ensemble oracles, semantic/seed controls, two-sided winner gate and signed-residual probe
- **scenario:** S1
- **n_features:** 261
- **model:** artifact-only-cpu-probes
- **params:** {"rounds":200,"num_leaves":31,"min_data_in_leaf":2000,"learning_rate":0.03,"lambda_l2":20,"max_bin":63,"feature_sets":["DISAGREEMENT_ONLY","STATE_ONLY","COMBINED"],"split":"splitmix64(user_id)&1","seed":42}
- **cutoffs:** donor 09-04/09-18/10-02; recipient 10-16
- **L:** existing cutoff-safe features
- **panel_blocks:** 3
- **fold_scores:** [1.766883357, 1.760509577, 1.748629224, 1.741278566]
- **cv_mean:** 1.754325181
- **cv_std:** 0.009995
- **bias_mean:** Unknown / not recoverable from repository history
- **best_offset:** -0.035387154
- **cv_mean_calib:** 1.754325181
- **delta_vs_b0:** -0.000006419
- **runtime_s:** 592.5
- **verdict:** NONE
- **conclusion:** Exact artifact audit PASS. ETX-vs-SEQ oracle 0.020172 is below seed-null 0.031483 (semantic excess -0.011311). Winner AUC 0.526759 but bounded gate late delta only -0.000006419; residual donor scales 0/0 and delta ~0; full LOFO/test/LB/submission not run. Details exp_053
- **wcv:** 1.747509863
- **fold_cal:** [1.766883357, 1.760509577, 1.748629224, 1.741278566]
- **mean_z:** Unknown / not recoverable from repository history
- **lb_public:** Unknown / not recoverable from repository history
