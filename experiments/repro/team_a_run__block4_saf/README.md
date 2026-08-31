# Logged run — BLOCK4-SAF

## Catalogue metadata

- **Catalogue ID:** `team_a_run__block4_saf`
- **Namespace:** `team_a_run`
- **Experiment ID:** `BLOCK4-SAF`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** logged run / arm
- **Model:** Unknown / not recoverable from repository history
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** wcv:** 1.74775
- **Known score:** wcv:** 1.74775
- **Seed:** params:** {"L": 180, "alphas": [0.25, 0.5, 0.75, 1.0, 1.25], "bagging_fraction": 0.8, "base": "STRONGEST_CURRENT", "conditional_population": "opposite splitmix64 group from P_V", "feature_fraction": 0.8, "lambda_l2": 10, "learning_rate": 0.03, "max_bin": 63, "min_data_in_leaf": 500, "num_leaves": 63, "q_population": "active in previous 2 blocks; clean only", "rounds": 200, "seeds": [42, 43, 44], "winsor": [0.005, 0.995]}
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — BLOCK4-SAF

This run was recovered from `experiments/log.csv`.

- **exp_id:** BLOCK4-SAF
- **timestamp:** 2026-08-21T13:46:47
- **commit:** a28a71f
- **description:** exp_039 selection-aware q*(nu_F-nu_C) residual correction of STRONGEST_CURRENT; user cross-fit, shuffle and seed controls
- **scenario:** S1
- **n_features:** 288
- **model:** lgb-binary+crossfit-regression
- **params:** {"L": 180, "alphas": [0.25, 0.5, 0.75, 1.0, 1.25], "bagging_fraction": 0.8, "base": "STRONGEST_CURRENT", "conditional_population": "opposite splitmix64 group from P_V", "feature_fraction": 0.8, "lambda_l2": 10, "learning_rate": 0.03, "max_bin": 63, "min_data_in_leaf": 500, "num_leaves": 63, "q_population": "active in previous 2 blocks; clean only", "rounds": 200, "seeds": [42, 43, 44], "winsor": [0.005, 0.995]}
- **cutoffs:** q 18/20/22/24 @ step7; conditional C=V-60,F=V-30
- **L:** 180
- **panel_blocks:** 3
- **fold_scores:** [1.76757, 1.76079, 1.74879, 1.74149]
- **cv_mean:** Unknown / not recoverable from repository history
- **cv_std:** Unknown / not recoverable from repository history
- **bias_mean:** Unknown / not recoverable from repository history
- **best_offset:** Unknown / not recoverable from repository history
- **cv_mean_calib:** 1.75493
- **delta_vs_b0:** 0.00024
- **runtime_s:** 2048
- **verdict:** REJECT
- **conclusion:** Honest LOFO +0.000240, 0/4; 10-16 +0.000213; all held-out alpha=0.25; shuffle +0.007401; rec_buy15-60 +0.000203; test Var ratio 0.871 PASS; no submission. Details exp_039
- **wcv:** 1.74775
- **fold_cal:** [1.76757, 1.76079, 1.74879, 1.74149]
- **mean_z:** 2.68819
- **lb_public:** Unknown / not recoverable from repository history
