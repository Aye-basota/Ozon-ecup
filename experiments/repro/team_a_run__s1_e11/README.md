# Logged run — S1-E11

## Catalogue metadata

- **Catalogue ID:** `team_a_run__s1_e11`
- **Namespace:** `team_a_run`
- **Experiment ID:** `S1-E11`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** logged run / arm
- **Model:** calibration diagnostic
- **Features:** calendar features, 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** wcv:** 1.75070
- **Known score:** conclusion:** −0.00095 CV к S1-E10; калиброванный OOF 1.75792 против 1.75716 у S1-BEST; сабмита нет, на LB не отправлялся; детали exp_013
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** conclusion:** −0.00095 CV к S1-E10; калиброванный OOF 1.75792 против 1.75716 у S1-BEST; сабмита нет, на LB не отправлялся; детали exp_013
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — S1-E11

This run was recovered from `experiments/log.csv`.

- **exp_id:** S1-E11
- **timestamp:** 2026-08-10T23:52:14
- **commit:** e3b1cde
- **description:** E11: двухчастная модель на нормированных длинных окнах
- **scenario:** S1
- **n_features:** 227
- **model:** two_part
- **params:** {"L": null, "calendar": false, "cutoffs": "all", "drop_groups": [], "keep_only": null, "min_history": 90, "model": "two_part", "norm_long": true, "panel_blocks": 3, "params": {}, "rounds": 600, "step": 7, "train_blocks": 1, "weight_tau": null}
- **cutoffs:** 24 @ step 7
- **L:** Unknown / not recoverable from repository history
- **panel_blocks:** 3
- **fold_scores:** [1.77344, 1.76548, 1.75209, 1.74469]
- **cv_mean:** 1.75893
- **cv_std:** 0.01121
- **bias_mean:** -0.05520
- **best_offset:** -0.05500
- **cv_mean_calib:** 1.75792
- **delta_vs_b0:** Unknown / not recoverable from repository history
- **runtime_s:** 1985.10000
- **verdict:** OPEN
- **conclusion:** −0.00095 CV к S1-E10; калиброванный OOF 1.75792 против 1.75716 у S1-BEST; сабмита нет, на LB не отправлялся; детали exp_013
- **wcv:** 1.75070
- **fold_cal:** [1.77123, 1.76469, 1.75177, 1.7441]
- **mean_z:** 2.68057
- **lb_public:** Unknown / not recoverable from repository history
