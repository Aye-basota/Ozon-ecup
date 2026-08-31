# Logged run — S1-E04

## Catalogue metadata

- **Catalogue ID:** `team_a_run__s1_e04`
- **Namespace:** `team_a_run`
- **Experiment ID:** `S1-E04`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** logged run / arm
- **Model:** Unknown / not recoverable from repository history
- **Features:** calendar features
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
# Logged run — S1-E04

This run was recovered from `experiments/log.csv`.

- **exp_id:** S1-E04
- **timestamp:** 2026-08-10T21:47:14
- **commit:** b2f0287
- **description:** E04: uncapped минус длинные признаки (w365/all/tenure/lifetime)
- **scenario:** S1
- **n_features:** 195
- **model:** direct
- **params:** {"L": null, "calendar": false, "cutoffs": "all", "drop_groups": ["long"], "keep_only": null, "min_history": 90, "model": "direct", "panel_blocks": 3, "params": {}, "rounds": 600, "step": 7, "train_blocks": 1, "weight_tau": null}
- **cutoffs:** 24 @ step 7
- **L:** Unknown / not recoverable from repository history
- **panel_blocks:** 3
- **fold_scores:** [1.76029, 1.7538]
- **cv_mean:** 1.75705
- **cv_std:** 0.00324
- **bias_mean:** -0.07482
- **best_offset:** -0.07500
- **cv_mean_calib:** 1.75542
- **delta_vs_b0:** Unknown / not recoverable from repository history
- **runtime_s:** 877.60000
- **verdict:** Unknown / not recoverable from repository history
- **conclusion:** Unknown / not recoverable from repository history
- **wcv:** Unknown / not recoverable from repository history
- **fold_cal:** [1.75898, 1.75189]
- **mean_z:** 2.70941
- **lb_public:** Unknown / not recoverable from repository history
