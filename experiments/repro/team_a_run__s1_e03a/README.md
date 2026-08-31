# Logged run — S1-E03a

## Catalogue metadata

- **Catalogue ID:** `team_a_run__s1_e03a`
- **Namespace:** `team_a_run`
- **Experiment ID:** `S1-E03a`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** logged run / arm
- **Model:** Unknown / not recoverable from repository history
- **Features:** calendar features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** wcv:** 1.76064
- **Known score:** wcv:** 1.76064
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — S1-E03a

This run was recovered from `experiments/log.csv`.

- **exp_id:** S1-E03a
- **timestamp:** 2026-08-10T21:29:03
- **commit:** b2f0287
- **description:** E03a: L=180 relaxed (min_hist 90, 29 cutoffs)
- **scenario:** S1
- **n_features:** 195
- **model:** direct
- **params:** {"L": 180, "calendar": false, "cutoffs": "all", "drop_groups": [], "keep_only": null, "min_history": 90, "model": "direct", "panel_blocks": 3, "params": {}, "rounds": 600, "step": 7, "train_blocks": 1, "weight_tau": null}
- **cutoffs:** 24 @ step 7
- **L:** 180
- **panel_blocks:** 3
- **fold_scores:** [1.77887, 1.77267, 1.76256, 1.75737]
- **cv_mean:** 1.76787
- **cv_std:** 0.00840
- **bias_mean:** -0.06899
- **best_offset:** -0.07000
- **cv_mean_calib:** 1.76640
- **delta_vs_b0:** Unknown / not recoverable from repository history
- **runtime_s:** 656.30000
- **verdict:** Unknown / not recoverable from repository history
- **conclusion:** Unknown / not recoverable from repository history
- **wcv:** 1.76064
- **fold_cal:** [1.77699, 1.77203, 1.76145, 1.75534]
- **mean_z:** 2.69468
- **lb_public:** Unknown / not recoverable from repository history
