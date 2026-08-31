# Logged run — RECENCY_RIDGE_PRED_EXP068

## Catalogue metadata

- **Catalogue ID:** `team_a_run__recency_ridge_pred_exp068`
- **Namespace:** `team_a_run`
- **Experiment ID:** `RECENCY_RIDGE_PRED_EXP068`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** logged run / arm
- **Model:** Ridge, two-part / hurdle
- **Features:** recency
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** wcv:** Unknown / not recoverable from repository history
- **Known score:** Unknown / not recoverable from repository history
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — RECENCY_RIDGE_PRED_EXP068

This run was recovered from `experiments/log.csv`.

- **exp_id:** RECENCY_RIDGE_PRED_EXP068
- **timestamp:** 2026-08-25T17:00:00+03:00
- **commit:** a28a71f
- **description:** exp_068 exact teammate recency Ridge-stack replay and redundancy audit against latest
- **scenario:** artifact-only-historical-replay-audit
- **n_features:** 0
- **model:** none
- **params:** {"historical":"ridge_drop_recent_hurdle_stable18_s075","training":"NONE","required_replay_floor":5e-7,"prefix":"RECENCY_RIDGE_PRED_EXP068_A1"}
- **cutoffs:** historical folds 2025-09-04/09-18/10-02/10-16; canonical latest OOF required
- **L:** Unknown / not recoverable from repository history
- **panel_blocks:** Unknown / not recoverable from repository history
- **fold_scores:** Unknown / not recoverable from repository history
- **cv_mean:** Unknown / not recoverable from repository history
- **cv_std:** Unknown / not recoverable from repository history
- **bias_mean:** Unknown / not recoverable from repository history
- **best_offset:** Unknown / not recoverable from repository history
- **cv_mean_calib:** Unknown / not recoverable from repository history
- **delta_vs_b0:** Unknown / not recoverable from repository history
- **runtime_s:** 2.2
- **verdict:** BLOCKED_HISTORICAL_REPLAY
- **conclusion:** Reference CSV SHA 95965c33...7959 and exact code recipe found, but 32/32 historical OOF plus 6/6 helper TEST checkpoints absent; winner includes up to 72 raw meta columns. No circular replay, no new Ridge/controls/test/submission. TEST-only corr historical-latest 0.999968, Var(diff) 0.000159637; canonical latest OOF and CAP lineage also missing. Details exp_068
- **wcv:** Unknown / not recoverable from repository history
- **fold_cal:** Unknown / not recoverable from repository history
- **mean_z:** 2.3293
- **lb_public:** Unknown / not recoverable from repository history
