# Logged run — FINGERPRINT_EXP058

## Catalogue metadata

- **Catalogue ID:** `team_a_run__fingerprint_exp058`
- **Namespace:** `team_a_run`
- **Experiment ID:** `FINGERPRINT_EXP058`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** logged run / arm
- **Model:** Unknown / not recoverable from repository history
- **Features:** dataset/user fingerprint
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** wcv:** Unknown / not recoverable from repository history
- **Known score:** conclusion:** Integrity/BASE/marginal/test-metadata audits PASS. REAL-PERM standalone -0.000129771, but fixed UNC slot REAL-PERM +0.000071048 and REAL-STRONGEST -0.000022517; hash halves +0.0000688/+0.0000733 vs PERM; AUC +0.0001097; positive RMSLE worse. Full folds/test/LB not run. Details exp_058
- **Seed:** params:** {"base":"historical UNC exact replay","rounds":600,"seed":42,"fingerprints_before":30,"fingerprints_after":15,"permutation":"fixed user bijection within exact row-incidence signature","slot_weight":0.20}
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — FINGERPRINT_EXP058

This run was recovered from `experiments/log.csv`.

- **exp_id:** FINGERPRINT_EXP058
- **timestamp:** 2026-08-24T19:10:23+03:00
- **commit:** a28a71f
- **description:** exp_058 dataset/user identity and extraction fingerprints vs one fixed incidence-matched joint permutation
- **scenario:** S1
- **n_features:** 251
- **model:** lgb-direct-paired
- **params:** {"base":"historical UNC exact replay","rounds":600,"seed":42,"fingerprints_before":30,"fingerprints_after":15,"permutation":"fixed user bijection within exact row-incidence signature","slot_weight":0.20}
- **cutoffs:** 24 @ step 7
- **L:** Unknown / not recoverable from repository history
- **panel_blocks:** 3
- **fold_scores:** [1.741256049]
- **cv_mean:** 1.741256049
- **cv_std:** 0
- **bias_mean:** Unknown / not recoverable from repository history
- **best_offset:** -0.031549866
- **cv_mean_calib:** 1.741256049
- **delta_vs_b0:** -0.000022517
- **runtime_s:** 541.87
- **verdict:** REJECT
- **conclusion:** Integrity/BASE/marginal/test-metadata audits PASS. REAL-PERM standalone -0.000129771, but fixed UNC slot REAL-PERM +0.000071048 and REAL-STRONGEST -0.000022517; hash halves +0.0000688/+0.0000733 vs PERM; AUC +0.0001097; positive RMSLE worse. Full folds/test/LB not run. Details exp_058
- **wcv:** Unknown / not recoverable from repository history
- **fold_cal:** [1.741256049]
- **mean_z:** 2.663068995
- **lb_public:** Unknown / not recoverable from repository history
