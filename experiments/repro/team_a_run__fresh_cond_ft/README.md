# Logged run — FRESH-COND-FT

## Catalogue metadata

- **Catalogue ID:** `team_a_run__fresh_cond_ft`
- **Namespace:** `team_a_run`
- **Experiment ID:** `FRESH-COND-FT`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** logged run / arm
- **Model:** dilated TCN, sequence model
- **Features:** freshness/conditional features, history-depth features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** wcv:** Unknown / not recoverable from repository history
- **Known score:** Unknown / not recoverable from repository history
- **Seed:** params:** {"baseline": "DETSEQ01 seeds 42/43/44", "depth_aug": 0.0, "direct_head": "frozen", "encoder_lr": 3e-05, "conditional_head_lr": 0.001, "lambda_cond": 0.25, "fine_tune_epochs": 1, "conditional_batch": "128 common + 128 added", "extra_depth_clip": 289, "deterministic": true}
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — FRESH-COND-FT

This run was recovered from `experiments/log.csv`.

- **exp_id:** FRESH-COND-FT
- **timestamp:** 2026-08-22T11:15:00
- **commit:** a28a71f
- **description:** exp_044 controlled paired encoder fine-tune on new deterministic plain SEQ-01 baselines; fresh positive-only conditional supervision versus equal-volume CLEAN VOL
- **scenario:** S1
- **n_features:** 17
- **model:** tcn+conditional-ft
- **params:** {"baseline": "DETSEQ01 seeds 42/43/44", "depth_aug": 0.0, "direct_head": "frozen", "encoder_lr": 3e-05, "conditional_head_lr": 0.001, "lambda_cond": 0.25, "fine_tune_epochs": 1, "conditional_batch": "128 common + 128 added", "extra_depth_clip": 289, "deterministic": true}
- **cutoffs:** 24 CLEAN @ step 7; 13 EXTRA
- **L:** 365
- **panel_blocks:** 3
- **fold_scores:** [1.739352194, 1.737774927, 1.738799379]
- **cv_mean:** 1.738642167
- **cv_std:** 0.000653442
- **bias_mean:** Unknown / not recoverable from repository history
- **best_offset:** Unknown / not recoverable from repository history
- **cv_mean_calib:** 1.738642167
- **delta_vs_b0:** 0.000052355
- **runtime_s:** 26869.1
- **verdict:** REJECT
- **conclusion:** FRESH-VOL mean -0.000088094, median -0.000107510, sd 0.000065396, 3/3; below -0.0003 gate and positive-only +0.000295. Technical replay exact; full folds/test/LOFO/LB/submission skipped. Details exp_044
- **wcv:** Unknown / not recoverable from repository history
- **fold_cal:** Unknown / not recoverable from repository history
- **mean_z:** Unknown / not recoverable from repository history
- **lb_public:** Unknown / not recoverable from repository history
