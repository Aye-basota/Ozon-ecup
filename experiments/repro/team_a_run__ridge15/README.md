# Logged run — RIDGE15

## Catalogue metadata

- **Catalogue ID:** `team_a_run__ridge15`
- **Namespace:** `team_a_run`
- **Experiment ID:** `RIDGE15`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** logged run / arm
- **Model:** sequence model, Ridge, blend
- **Features:** 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** params:** {"diversity_var_unc_dist_strongest": [0.06024684, 0.04869617, 0.04482285], "features": "S1-E10", "lambda_grid": [1e-05, 0.0001, 0.001, 0.01, 0.1, 1.0], "lofo_lambda": [0.001, 0.001, 0.001, 0.001], "main_weights": {"ETX-AVG3": 0.225, "RIDGE15": 0.15, "S1-DIST": 0.25, "S1-E02": 0.05, "S1-E03a": 0.1, "SEQ-AVG3": 0.225}, "ridge_target": "log1p(GMV30)", "standalone_auc": 0.8405019199, "standalone_wcv": 1.7601138327}
- **Known score:** params:** {"diversity_var_unc_dist_strongest": [0.06024684, 0.04869617, 0.04482285], "features": "S1-E10", "lambda_grid": [1e-05, 0.0001, 0.001, 0.01, 0.1, 1.0], "lofo_lambda": [0.001, 0.001, 0.001, 0.001], "main_weights": {"ETX-AVG3": 0.225, "RIDGE15": 0.15, "S1-DIST": 0.25, "S1-E02": 0.05, "S1-E03a": 0.1, "SEQ-AVG3": 0.225}, "ridge_target": "log1p(GMV30)", "standalone_auc": 0.8405019199, "standalone_wcv": 1.7601138327}
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — RIDGE15

This run was recovered from `experiments/log.csv`.

- **exp_id:** RIDGE15
- **timestamp:** 2026-08-21T21:55:46
- **commit:** a28a71f
- **description:** exp_041 Ridge on S1-E10 as fixed 15% member replacing UNC; one equal UNC/DIST control
- **scenario:** S1
- **n_features:** 227
- **model:** ridge+fixed-blend
- **params:** {"diversity_var_unc_dist_strongest": [0.06024684, 0.04869617, 0.04482285], "features": "S1-E10", "lambda_grid": [1e-05, 0.0001, 0.001, 0.01, 0.1, 1.0], "lofo_lambda": [0.001, 0.001, 0.001, 0.001], "main_weights": {"ETX-AVG3": 0.225, "RIDGE15": 0.15, "S1-DIST": 0.25, "S1-E02": 0.05, "S1-E03a": 0.1, "SEQ-AVG3": 0.225}, "ridge_target": "log1p(GMV30)", "standalone_auc": 0.8405019199, "standalone_wcv": 1.7601138327}
- **cutoffs:** 18/20/22/24 @ step 7
- **L:** Unknown / not recoverable from repository history
- **panel_blocks:** 3
- **fold_scores:** [1.76996, 1.76231, 1.74906, 1.74187]
- **cv_mean:** 1.75580
- **cv_std:** 0.01098
- **bias_mean:** -0.05640
- **best_offset:** -0.05641
- **cv_mean_calib:** 1.75474
- **delta_vs_b0:** Unknown / not recoverable from repository history
- **runtime_s:** 210.50000
- **verdict:** REJECT
- **conclusion:** Main +0.000278 wCV, 0/4, 10-16 +0.000243; control +0.000207, 0/4, 10-16 +0.000179. Ridge diverse but too weak; no production/submission.
- **wcv:** 1.74779
- **fold_cal:** [1.76708, 1.76097, 1.7489, 1.74152]
- **mean_z:** 2.68203
- **lb_public:** Unknown / not recoverable from repository history
