# Logged run — LATE_SSL_EXP056

## Catalogue metadata

- **Catalogue ID:** `team_a_run__late_ssl_exp056`
- **Namespace:** `team_a_run`
- **Experiment ID:** `LATE_SSL_EXP056`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** logged run / arm
- **Model:** Unknown / not recoverable from repository history
- **Features:** history-depth features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** wcv:** Unknown / not recoverable from repository history
- **Known score:** Unknown / not recoverable from repository history
- **Seed:** params:** {"checkpoint":"ETX-01-S42-V0904","n_tok":192,"depth_cap":212,"query_weekday":"Thursday","batch":512,"steps":4094,"mask_rate":0.15,"lambda_ssl":0.25,"seed":42,"deterministic":true}
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — LATE_SSL_EXP056

This run was recovered from `experiments/log.csv`.

- **exp_id:** LATE_SSL_EXP056
- **timestamp:** 2026-08-24T21:00:00+03:00
- **commit:** a28a71f
- **description:** exp_056 exact ETX V0904 input-only late masked reconstruction vs matched clean-corridor SSL control
- **scenario:** S1-pilot
- **n_features:** 22
- **model:** etx-input-only-ssl-adapt
- **params:** {"checkpoint":"ETX-01-S42-V0904","n_tok":192,"depth_cap":212,"query_weekday":"Thursday","batch":512,"steps":4094,"mask_rate":0.15,"lambda_ssl":0.25,"seed":42,"deterministic":true}
- **cutoffs:** CONTROL 2025-05-22..2025-07-31; LATE 2025-08-07..2025-10-16; val 2025-10-16
- **L:** 212
- **panel_blocks:** 3
- **fold_scores:** [1.749403397]
- **cv_mean:** 1.749403397
- **cv_std:** 0
- **bias_mean:** Unknown / not recoverable from repository history
- **best_offset:** -0.116298335
- **cv_mean_calib:** 1.749403397
- **delta_vs_b0:** 0.000031631
- **runtime_s:** 3301.954
- **verdict:** REJECT
- **conclusion:** LATE-CONTROL raw -0.000001490 but calibrated +0.000031631; fixed slot +0.000007228; both user halves wrong sign; clean direct holdout degrades; embedding MMD does not shrink although reconstruction improves. Deterministic 100-step replay exact. Full folds/test/LB/submission not run. Details exp_056
- **wcv:** Unknown / not recoverable from repository history
- **fold_cal:** [1.749403397]
- **mean_z:** 2.747817755
- **lb_public:** Unknown / not recoverable from repository history
