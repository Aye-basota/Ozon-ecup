# Logged run — HOLIDAY-YOY

## Catalogue metadata

- **Catalogue ID:** `team_a_run__holiday_yoy`
- **Namespace:** `team_a_run`
- **Experiment ID:** `HOLIDAY-YOY`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** logged run / arm
- **Model:** Unknown / not recoverable from repository history
- **Features:** holiday/YoY features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** conclusion:** wCV +0.000093, 0/4, AUC -0.000003: ordinary CV neutral. YoY diagnostic purchase Pearson 0.03996 vs placebo 0.01281, OOS R2 0.001683 vs 0.000077; one high-risk submission allowed. File submissions/submission_HOLIDAY-YOY.csv; L*=2.3293; E03a kept.
- **Known score:** conclusion:** wCV +0.000093, 0/4, AUC -0.000003: ordinary CV neutral. YoY diagnostic purchase Pearson 0.03996 vs placebo 0.01281, OOS R2 0.001683 vs 0.000077; one high-risk submission allowed. File submissions/submission_HOLIDAY-YOY.csv; L*=2.3293; E03a kept.
- **Seed:** params:** {"base": "S1-DIST-MIX", "beta": "crossfit_yoy_slope-placebo_slope", "level": 2.3293, "seed": 42, "shrinkage": "support/(support+median_positive)", "weights": [0.15, 0.3, 0.1, 0.45]}
- **Postprocessing:** model:** postprocess
- **Submission:** conclusion:** wCV +0.000093, 0/4, AUC -0.000003: ordinary CV neutral. YoY diagnostic purchase Pearson 0.03996 vs placebo 0.01281, OOS R2 0.001683 vs 0.000077; one high-risk submission allowed. File submissions/submission_HOLIDAY-YOY.csv; L*=2.3293; E03a kept.
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — HOLIDAY-YOY

This run was recovered from `experiments/log.csv`.

- **exp_id:** HOLIDAY-YOY
- **timestamp:** 2026-08-12T23:26:32
- **commit:** 34a2335
- **description:** Персональная holiday-response 2025->2026; direct zero-mean correction, beta=YoY-placebo
- **scenario:** S1
- **n_features:** 6
- **model:** postprocess
- **params:** {"base": "S1-DIST-MIX", "beta": "crossfit_yoy_slope-placebo_slope", "level": 2.3293, "seed": 42, "shrinkage": "support/(support+median_positive)", "weights": [0.15, 0.3, 0.1, 0.45]}
- **cutoffs:** saved OOF; 4 folds
- **L:** Unknown / not recoverable from repository history
- **panel_blocks:** 3
- **fold_scores:** [1.77252, 1.76385, 1.75121, 1.74413]
- **cv_mean:** 1.75793
- **cv_std:** 0.01099
- **bias_mean:** -0.06640
- **best_offset:** -0.06640
- **cv_mean_calib:** 1.75652
- **delta_vs_b0:** Unknown / not recoverable from repository history
- **runtime_s:** Unknown / not recoverable from repository history
- **verdict:** SEND_HIGH_RISK
- **conclusion:** wCV +0.000093, 0/4, AUC -0.000003: ordinary CV neutral. YoY diagnostic purchase Pearson 0.03996 vs placebo 0.01281, OOS R2 0.001683 vs 0.000077; one high-risk submission allowed. File submissions/submission_HOLIDAY-YOY.csv; L*=2.3293; E03a kept.
- **wcv:** 1.74958
- **fold_cal:** [1.76917, 1.76263, 1.75082, 1.74324]
- **mean_z:** 2.69203
- **lb_public:** Unknown / not recoverable from repository history
