# Logged run — LEVEL_MINUS_006_EXP060

## Catalogue metadata

- **Catalogue ID:** `team_a_run__level_minus_006_exp060`
- **Namespace:** `team_a_run`
- **Experiment ID:** `LEVEL_MINUS_006_EXP060`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** logged run / arm
- **Model:** Unknown / not recoverable from repository history
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** wcv:** Unknown / not recoverable from repository history
- **Known score:** Unknown / not recoverable from repository history
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** description:** exp_060 fixed public production-level diagnostic: STRONGEST_CURRENT with the only change z - 0.06
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — LEVEL_MINUS_006_EXP060

This run was recovered from `experiments/log.csv`.

- **exp_id:** LEVEL_MINUS_006_EXP060
- **timestamp:** 2026-08-24T21:44:03+03:00
- **commit:** a28a71f
- **description:** exp_060 fixed public production-level diagnostic: STRONGEST_CURRENT with the only change z - 0.06
- **scenario:** production-level-diagnostic
- **n_features:** 0
- **model:** fixed-log-shift
- **params:** {"base":"STRONGEST_CURRENT","shift_log_space":-0.06,"normalization_after_shift":false,"training":"NONE"}
- **cutoffs:** test 2026-02-13
- **L:** Unknown / not recoverable from repository history
- **panel_blocks:** Unknown / not recoverable from repository history
- **fold_scores:** []
- **cv_mean:** Unknown / not recoverable from repository history
- **cv_std:** Unknown / not recoverable from repository history
- **bias_mean:** Unknown / not recoverable from repository history
- **best_offset:** Unknown / not recoverable from repository history
- **cv_mean_calib:** Unknown / not recoverable from repository history
- **delta_vs_b0:** Unknown / not recoverable from repository history
- **runtime_s:** 1.2
- **verdict:** PREPARED
- **conclusion:** Exact source SHA/schema/order verified. One submission only; 250000 rows; NaN/inf/negative/missing/duplicates 0; mean z 2.329321370 -> 2.269498656 after mandatory nonnegative floor; SHA256 1b40f67d119d0dcc4798a4da5612707b8d44f1dfe3fa20b28c28b836c2c8c0f1. Not uploaded. Details exp_060
- **wcv:** Unknown / not recoverable from repository history
- **fold_cal:** Unknown / not recoverable from repository history
- **mean_z:** 2.269498656
- **lb_public:** Unknown / not recoverable from repository history
