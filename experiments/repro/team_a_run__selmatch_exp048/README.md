# Logged run — SELMATCH_EXP048

## Catalogue metadata

- **Catalogue ID:** `team_a_run__selmatch_exp048`
- **Namespace:** `team_a_run`
- **Experiment ID:** `SELMATCH_EXP048`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** logged run / arm
- **Model:** BTYD
- **Features:** freshness/conditional features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** wcv:** 1.747509863
- **Known score:** wcv:** 1.747509863
- **Seed:** params:** {"base": "STRONGEST_CURRENT", "eligible_folds": ["2025-09-04", "2025-09-18", "2025-10-02"], "match_variable": "future_blocks_active k=0/1/2/3", "reference_landmarks": 16, "bootstrap": 500, "shuffle": 100, "seed": 42}
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — SELMATCH_EXP048

This run was recovered from `experiments/log.csv`.

- **exp_id:** SELMATCH_EXP048
- **timestamp:** 2026-08-23T23:30:00
- **commit:** a28a71f
- **description:** exp_048 artifact-only selection-mismatch audit: future continuation k, natural reference, pseudo-matched CV, cluster bootstrap and shuffle placebo
- **scenario:** S1
- **n_features:** Unknown / not recoverable from repository history
- **model:** artifact-only-reweight
- **params:** {"base": "STRONGEST_CURRENT", "eligible_folds": ["2025-09-04", "2025-09-18", "2025-10-02"], "match_variable": "future_blocks_active k=0/1/2/3", "reference_landmarks": 16, "bootstrap": 500, "shuffle": 100, "seed": 42}
- **cutoffs:** OOF 09-04/09-18/10-02/10-16
- **L:** Unknown / not recoverable from repository history
- **panel_blocks:** 3
- **fold_scores:** [1.766883357, 1.760509577, 1.748629224, 1.741278566]
- **cv_mean:** 1.754325181
- **cv_std:** 0.009995
- **bias_mean:** Unknown / not recoverable from repository history
- **best_offset:** Unknown / not recoverable from repository history
- **cv_mean_calib:** 1.754325181
- **delta_vs_b0:** 0.000000
- **runtime_s:** 125.0
- **verdict:** TECHNICAL_INCONCLUSIVE
- **conclusion:** Exact baseline PASS. pi_ref(k=0)=0.004951 but k=0 absent in every eligible fold, so full-reference matched CV is unsupported. Conditional k>0 best fixed candidate BTYD05_FRESH1 delta -0.000551, but largest incremental selection penalty only 0.000094 and no >=0.001 candidate; no training/test/submission. Details exp_048
- **wcv:** 1.747509863
- **fold_cal:** [1.766883357, 1.760509577, 1.748629224, 1.741278566]
- **mean_z:** 2.688187850
- **lb_public:** Unknown / not recoverable from repository history
