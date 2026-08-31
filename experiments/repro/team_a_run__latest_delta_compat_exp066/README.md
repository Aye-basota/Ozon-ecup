# Logged run — LATEST_DELTA_COMPAT_EXP066

## Catalogue metadata

- **Catalogue ID:** `team_a_run__latest_delta_compat_exp066`
- **Namespace:** `team_a_run`
- **Experiment ID:** `LATEST_DELTA_COMPAT_EXP066`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** logged run / arm
- **Model:** BTYD
- **Features:** freshness/conditional features
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
# Logged run — LATEST_DELTA_COMPAT_EXP066

This run was recovered from `experiments/log.csv`.

- **exp_id:** LATEST_DELTA_COMPAT_EXP066
- **timestamp:** 2026-08-25T16:02:22+03:00
- **commit:** a28a71f
- **description:** exp_066 frozen SAFE-ANCHOR BTYD05 FRESH SEQ65 compatibility against canonical latest
- **scenario:** artifact-only-prerequisite-audit
- **n_features:** 0
- **model:** none
- **params:** {"base":"latest","reference":"STRONGEST_CURRENT","families":["SAFE-ANCHOR","BTYD05","FRESH","SEQ65"],"training":"NONE","required_columns":["z_latest","z_STRONGEST_CURRENT","target","user_id","fold"]}
- **cutoffs:** required canonical val 2025-09-04/09-18/10-02/10-16
- **L:** Unknown / not recoverable from repository history
- **panel_blocks:** Unknown / not recoverable from repository history
- **fold_scores:** Unknown / not recoverable from repository history
- **cv_mean:** Unknown / not recoverable from repository history
- **cv_std:** Unknown / not recoverable from repository history
- **bias_mean:** Unknown / not recoverable from repository history
- **best_offset:** Unknown / not recoverable from repository history
- **cv_mean_calib:** Unknown / not recoverable from repository history
- **delta_vs_b0:** Unknown / not recoverable from repository history
- **runtime_s:** 12.0
- **verdict:** BLOCKED_NO_CANONICAL_LATEST_OOF
- **conclusion:** AUTHORITATIVE-LATEST-INTEGRATION and canonical z_latest OOF absent; 627 NPZ and 83 Parquet schemas checked with zero z_latest hits. Primary LOFO/control/test/CAP candidate audit and submission not run; no reconstruction from test predictions or public LB. Details exp_066
- **wcv:** Unknown / not recoverable from repository history
- **fold_cal:** Unknown / not recoverable from repository history
- **mean_z:** Unknown / not recoverable from repository history
- **lb_public:** Unknown / not recoverable from repository history
