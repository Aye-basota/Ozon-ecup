# Logged run — BTYD05-PROD

## Catalogue metadata

- **Catalogue ID:** `team_a_run__btyd05_prod`
- **Namespace:** `team_a_run`
- **Experiment ID:** `BTYD05-PROD`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** logged run / arm
- **Model:** sequence model, BTYD
- **Features:** freshness/conditional features, 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** wcv:** Unknown / not recoverable from repository history
- **Known score:** Unknown / not recoverable from repository history
- **Seed:** conclusion:** FRESH parity FAIL: exact SEQ-D3A-BASE seed-42 TEST encoder absent. Exact BTYD donor-0 MLE fails unchanged stability gates: NLL spread 1.624e-5>1e-6; log-param spread 1.1888>0.10; max gradient 0.002274>0.001. Fail-fast before scoring/support composition; no submission. Details exp_050
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — BTYD05-PROD

This run was recovered from `experiments/log.csv`.

- **exp_id:** BTYD05-PROD
- **timestamp:** 2026-08-23T23:58:00
- **commit:** a28a71f
- **description:** exp_050 exact production resolution for fixed BTYD05_FRESH1 with BTYD05 fallback
- **scenario:** S1
- **n_features:** Unknown / not recoverable from repository history
- **model:** bgnbd-production-resolution
- **params:** {"candidate": "BTYD05_FRESH1", "fallback": "BTYD05", "cutoff": "2026-02-13", "split": "splitmix64(user_id)&1", "origin": "2024-12-31", "K": 3.0, "seed": 42}
- **cutoffs:** test 2026-02-13
- **L:** common-origin
- **panel_blocks:** 3
- **fold_scores:** Unknown / not recoverable from repository history
- **cv_mean:** Unknown / not recoverable from repository history
- **cv_std:** Unknown / not recoverable from repository history
- **bias_mean:** Unknown / not recoverable from repository history
- **best_offset:** Unknown / not recoverable from repository history
- **cv_mean_calib:** Unknown / not recoverable from repository history
- **delta_vs_b0:** Unknown / not recoverable from repository history
- **runtime_s:** 13.9
- **verdict:** TECHNICAL_BLOCK
- **conclusion:** FRESH parity FAIL: exact SEQ-D3A-BASE seed-42 TEST encoder absent. Exact BTYD donor-0 MLE fails unchanged stability gates: NLL spread 1.624e-5>1e-6; log-param spread 1.1888>0.10; max gradient 0.002274>0.001. Fail-fast before scoring/support composition; no submission. Details exp_050
- **wcv:** Unknown / not recoverable from repository history
- **fold_cal:** Unknown / not recoverable from repository history
- **mean_z:** Unknown / not recoverable from repository history
- **lb_public:** Unknown / not recoverable from repository history
