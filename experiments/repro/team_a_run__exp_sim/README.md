# Logged run — EXP-SIM

## Catalogue metadata

- **Catalogue ID:** `team_a_run__exp_sim`
- **Namespace:** `team_a_run`
- **Experiment ID:** `EXP-SIM`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** logged run / arm
- **Model:** blend, calibration diagnostic
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** wcv:** Unknown / not recoverable from repository history
- **Known score:** conclusion:** строка описывает чистую similarity; сабмит = смесь 0.50 с S1-BEST: калиброванный OOF 1.77210 (+0.01494), LB 1.6682180 (+0.01694); файл submissions/experimental_submission_2.csv
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** conclusion:** строка описывает чистую similarity; сабмит = смесь 0.50 с S1-BEST: калиброванный OOF 1.77210 (+0.01494), LB 1.6682180 (+0.01694); файл submissions/experimental_submission_2.csv
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — EXP-SIM

This run was recovered from `experiments/log.csv`.

- **exp_id:** EXP-SIM
- **timestamp:** 2026-08-10T23:27:00
- **commit:** e3b1cde
- **description:** EXP-SIM: rank-cohort similarity (чистая модель); сабмит — смесь 0.50 с S1-BEST (детали exp_008)
- **scenario:** S1
- **n_features:** 4
- **model:** similarity
- **params:** {"bins": [8, 6, 8, 6], "blend_weight": 0.5, "cutoffs": "all", "feats": ["w180_days_buy", "w30_days_buy", "w30_lgmv", "rec_buy"], "panel_blocks": 3, "smoothing_coarse": 500, "smoothing_fine": 100, "step": 7, "train_blocks": 1}
- **cutoffs:** 24 @ step 7
- **L:** Unknown / not recoverable from repository history
- **panel_blocks:** 3
- **fold_scores:** [1.86578, 1.86318, 1.84892, 1.82889]
- **cv_mean:** 1.85169
- **cv_std:** 0.01465
- **bias_mean:** 0.33448
- **best_offset:** Unknown / not recoverable from repository history
- **cv_mean_calib:** 1.82105
- **delta_vs_b0:** Unknown / not recoverable from repository history
- **runtime_s:** Unknown / not recoverable from repository history
- **verdict:** REJECT
- **conclusion:** строка описывает чистую similarity; сабмит = смесь 0.50 с S1-BEST: калиброванный OOF 1.77210 (+0.01494), LB 1.6682180 (+0.01694); файл submissions/experimental_submission_2.csv
- **wcv:** Unknown / not recoverable from repository history
- **fold_cal:** Unknown / not recoverable from repository history
- **mean_z:** Unknown / not recoverable from repository history
- **lb_public:** 1.6682180
