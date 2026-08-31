# Logged run — FINAL_INTEGRATION_EXP065

## Catalogue metadata

- **Catalogue ID:** `team_a_run__final_integration_exp065`
- **Namespace:** `team_a_run`
- **Experiment ID:** `FINAL_INTEGRATION_EXP065`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** logged run / arm
- **Model:** BTYD, blend
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** wcv:** 1.747509862
- **Known score:** params:** {"candidate_A":"exact exp_037","candidate_B":"exp_051 BTYD05 hedge","level":2.3293,"training":"NONE","leaderboard_upload":false}
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — FINAL_INTEGRATION_EXP065

This run was recovered from `experiments/log.csv`.

- **exp_id:** FINAL_INTEGRATION_EXP065
- **timestamp:** 2026-08-25T03:49:00+03:00
- **commit:** a28a71f
- **description:** exp_065 independent exp_037 rebuild plus exactly two checked canonical production candidates
- **scenario:** final-integration
- **n_features:** 0
- **model:** fixed-log-blend-rebuild
- **params:** {"candidate_A":"exact exp_037","candidate_B":"exp_051 BTYD05 hedge","level":2.3293,"training":"NONE","leaderboard_upload":false}
- **cutoffs:** val 2025-09-04/09-18/10-02/10-16; test 2026-02-13
- **L:** 289
- **panel_blocks:** 3
- **fold_scores:** [1.766883356,1.760509577,1.748629224,1.741278566]
- **cv_mean:** 1.754325181
- **cv_std:** 0.009995
- **bias_mean:** Unknown / not recoverable from repository history
- **best_offset:** Unknown / not recoverable from repository history
- **cv_mean_calib:** 1.754325181
- **delta_vs_b0:** 0
- **runtime_s:** 5.4
- **verdict:** ACCEPT
- **conclusion:** A rebuilt byte-identical SHA abc2218b...e04bda; critical ETX-SEQ test/OOF variance ratio 0.775 PASS. B byte-identical SHA c3cfb4d...c2932; exp051 support PASS ratio 1.1734. Both exact sample/schema/order, 250000 finite nonnegative rows; no LB upload. Strongest remains exp037. Details exp_065
- **wcv:** 1.747509862
- **fold_cal:** [1.766883356,1.760509577,1.748629224,1.741278566]
- **mean_z:** 2.329321370
- **lb_public:** Unknown / not recoverable from repository history
