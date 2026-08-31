# Logged run — SELMATCH_EXP049

## Catalogue metadata

- **Catalogue ID:** `team_a_run__selmatch_exp049`
- **Namespace:** `team_a_run`
- **Experiment ID:** `SELMATCH_EXP049`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** logged run / arm
- **Model:** BTYD
- **Features:** freshness/conditional features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** conclusion:** Corrected same-fold C delta -0.000551306, 3/3; A -0.000546668, 3/3; bootstrap P=1.0 with calibration refit; signal shuffle PASS. Selection-shuffle contradiction traced to mixed estimands. Validation PREFERRED, but exact BTYD/FRESH test artifacts and registered production recipe are absent; production audit FAIL, no submission. Details exp_049
- **Known score:** wcv:** 1.740877030
- **Seed:** params:** {"base": "STRONGEST_CURRENT", "candidate": "BTYD05_FRESH1", "folds": ["2025-09-04", "2025-09-18", "2025-10-02"], "fold_weights": [1, 2, 4], "match": "k>0 renormalized reference", "bootstrap": 500, "signal_shuffle": 200, "selection_shuffle": 100, "seed": 42}
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — SELMATCH_EXP049

This run was recovered from `experiments/log.csv`.

- **exp_id:** SELMATCH_EXP049
- **timestamp:** 2026-08-23T22:13:53
- **commit:** a28a71f
- **description:** exp_049 corrected same-fold EXP-048 analysis and production-support audit for fixed BTYD05_FRESH1
- **scenario:** S1
- **n_features:** Unknown / not recoverable from repository history
- **model:** artifact-only-analysis-production-audit
- **params:** {"base": "STRONGEST_CURRENT", "candidate": "BTYD05_FRESH1", "folds": ["2025-09-04", "2025-09-18", "2025-10-02"], "fold_weights": [1, 2, 4], "match": "k>0 renormalized reference", "bootstrap": 500, "signal_shuffle": 200, "selection_shuffle": 100, "seed": 42}
- **cutoffs:** OOF 09-04/09-18/10-02
- **L:** Unknown / not recoverable from repository history
- **panel_blocks:** 3
- **fold_scores:** [1.753725986, 1.744693242, 1.735756685]
- **cv_mean:** 1.740877030
- **cv_std:** 0.007355
- **bias_mean:** Unknown / not recoverable from repository history
- **best_offset:** Unknown / not recoverable from repository history
- **cv_mean_calib:** 1.740877030
- **delta_vs_b0:** -0.000551306
- **runtime_s:** 118.0
- **verdict:** REJECT
- **conclusion:** Corrected same-fold C delta -0.000551306, 3/3; A -0.000546668, 3/3; bootstrap P=1.0 with calibration refit; signal shuffle PASS. Selection-shuffle contradiction traced to mixed estimands. Validation PREFERRED, but exact BTYD/FRESH test artifacts and registered production recipe are absent; production audit FAIL, no submission. Details exp_049
- **wcv:** 1.740877030
- **fold_cal:** [1.753725986, 1.744693242, 1.735756685]
- **mean_z:** Unknown / not recoverable from repository history
- **lb_public:** Unknown / not recoverable from repository history
