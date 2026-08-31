# Logged run — S1-E02

## Catalogue metadata

- **Catalogue ID:** `team_a_run__s1_e02`
- **Namespace:** `team_a_run`
- **Experiment ID:** `S1-E02`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** logged run / arm
- **Model:** Unknown / not recoverable from repository history
- **Features:** calendar features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** wcv:** 1.75151
- **Known score:** wcv:** 1.75151
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — S1-E02

This run was recovered from `experiments/log.csv`.

- **exp_id:** S1-E02
- **timestamp:** 2026-08-10T21:15:37
- **commit:** 37828cd
- **description:** E02: dense cutoff grid (all, step 7) vs 3 recent; uncapped, 1-block train panel
- **scenario:** S1
- **n_features:** 236
- **model:** direct
- **params:** {"L": null, "calendar": false, "cutoffs": "all", "drop_groups": [], "keep_only": null, "min_history": 90, "model": "direct", "panel_blocks": 3, "params": {}, "rounds": 600, "step": 7, "train_blocks": 1, "weight_tau": null}
- **cutoffs:** 24 @ step 7
- **L:** Unknown / not recoverable from repository history
- **panel_blocks:** 3
- **fold_scores:** [1.78022, 1.7676, 1.75316, 1.7463]
- **cv_mean:** 1.76182
- **cv_std:** 0.01311
- **bias_mean:** -0.09500
- **best_offset:** -0.09500
- **cv_mean_calib:** 1.75913
- **delta_vs_b0:** Unknown / not recoverable from repository history
- **runtime_s:** 733.70000
- **verdict:** Unknown / not recoverable from repository history
- **conclusion:** Unknown / not recoverable from repository history
- **wcv:** 1.75151
- **fold_cal:** [1.77152, 1.76427, 1.75287, 1.74513]
- **mean_z:** 2.71989
- **lb_public:** Unknown / not recoverable from repository history
