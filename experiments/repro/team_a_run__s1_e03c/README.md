# Logged run — S1-E03c

## Catalogue metadata

- **Catalogue ID:** `team_a_run__s1_e03c`
- **Namespace:** `team_a_run`
- **Experiment ID:** `S1-E03c`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** logged run / arm
- **Model:** Unknown / not recoverable from repository history
- **Features:** calendar features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** wcv:** 1.79323
- **Known score:** wcv:** 1.79323
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — S1-E03c

This run was recovered from `experiments/log.csv`.

- **exp_id:** S1-E03c
- **timestamp:** 2026-08-10T21:48:34
- **commit:** b2f0287
- **description:** E03c: L=90 (min_hist 90, 29 cutoffs)
- **scenario:** S1
- **n_features:** 165
- **model:** direct
- **params:** {"L": 90, "calendar": false, "cutoffs": "all", "drop_groups": [], "keep_only": null, "min_history": 90, "model": "direct", "panel_blocks": 3, "params": {}, "rounds": 600, "step": 7, "train_blocks": 1, "weight_tau": null}
- **cutoffs:** 24 @ step 7
- **L:** 90
- **panel_blocks:** 3
- **fold_scores:** [1.8092, 1.80558, 1.79461, 1.78786]
- **cv_mean:** 1.79931
- **cv_std:** 0.00852
- **bias_mean:** -0.01311
- **best_offset:** -0.01500
- **cv_mean_calib:** 1.79914
- **delta_vs_b0:** Unknown / not recoverable from repository history
- **runtime_s:** 855.50000
- **verdict:** Unknown / not recoverable from repository history
- **conclusion:** Unknown / not recoverable from repository history
- **wcv:** 1.79323
- **fold_cal:** [1.80912, 1.80554, 1.79458, 1.78749]
- **mean_z:** 2.63890
- **lb_public:** Unknown / not recoverable from repository history
