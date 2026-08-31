# Logged run — EXP-MIN

## Catalogue metadata

- **Catalogue ID:** `team_a_run__exp_min`
- **Namespace:** `team_a_run`
- **Experiment ID:** `EXP-MIN`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** logged run / arm
- **Model:** Unknown / not recoverable from repository history
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** wcv:** Unknown / not recoverable from repository history
- **Known score:** conclusion:** LB 1.6674246 против 1.6512803 у submission_strategy_1.csv (+0.01614); файл submissions/experimental_submission_1.csv
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** conclusion:** LB 1.6674246 против 1.6512803 у submission_strategy_1.csv (+0.01614); файл submissions/experimental_submission_1.csv
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — EXP-MIN

This run was recovered from `experiments/log.csv`.

- **exp_id:** EXP-MIN
- **timestamp:** 2026-08-10T23:27:00
- **commit:** e3b1cde
- **description:** EXP-MIN: 15 устойчивых признаков, extreme regularization (детали exp_007)
- **scenario:** S1
- **n_features:** 15
- **model:** direct
- **params:** {"L": null, "cutoffs": "all", "feature_fraction": 1.0, "lambda_l2": 20.0, "min_data_in_leaf": 800, "min_history": 90, "model": "direct", "num_leaves": 31, "panel_blocks": 3, "rounds": 400, "step": 7, "train_blocks": 1}
- **cutoffs:** 24 @ step 7
- **L:** Unknown / not recoverable from repository history
- **panel_blocks:** 3
- **fold_scores:** [1.78521, 1.7776, 1.76795, 1.76262]
- **cv_mean:** 1.77335
- **cv_std:** 0.00870
- **bias_mean:** -0.09545
- **best_offset:** -0.09500
- **cv_mean_calib:** 1.77065
- **delta_vs_b0:** Unknown / not recoverable from repository history
- **runtime_s:** Unknown / not recoverable from repository history
- **verdict:** REJECT
- **conclusion:** LB 1.6674246 против 1.6512803 у submission_strategy_1.csv (+0.01614); файл submissions/experimental_submission_1.csv
- **wcv:** Unknown / not recoverable from repository history
- **fold_cal:** Unknown / not recoverable from repository history
- **mean_z:** Unknown / not recoverable from repository history
- **lb_public:** 1.6674246
