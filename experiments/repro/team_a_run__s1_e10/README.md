# Logged run — S1-E10

## Catalogue metadata

- **Catalogue ID:** `team_a_run__s1_e10`
- **Namespace:** `team_a_run`
- **Experiment ID:** `S1-E10`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** logged run / arm
- **Model:** Unknown / not recoverable from repository history
- **Features:** calendar features, 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** wcv:** 1.75170
- **Known score:** wcv:** 1.75170
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — S1-E10

This run was recovered from `experiments/log.csv`.

- **exp_id:** S1-E10
- **timestamp:** 2026-08-10T22:19:20
- **commit:** b2f0287
- **description:** E10: длинные окна, нормированные на доступную глубину истории (all_*/lifetime_* выброшены как дубликаты w365_*)
- **scenario:** S1
- **n_features:** 227
- **model:** direct
- **params:** {"L": null, "calendar": false, "cutoffs": "all", "drop_groups": [], "keep_only": null, "min_history": 90, "model": "direct", "norm_long": true, "panel_blocks": 3, "params": {}, "rounds": 600, "step": 7, "train_blocks": 1, "weight_tau": null}
- **cutoffs:** 24 @ step 7
- **L:** Unknown / not recoverable from repository history
- **panel_blocks:** 3
- **fold_scores:** [1.77429, 1.76617, 1.75356, 1.7455]
- **cv_mean:** 1.75988
- **cv_std:** 0.01111
- **bias_mean:** -0.05447
- **best_offset:** -0.05500
- **cv_mean_calib:** 1.75889
- **delta_vs_b0:** Unknown / not recoverable from repository history
- **runtime_s:** 1589.10000
- **verdict:** Unknown / not recoverable from repository history
- **conclusion:** Unknown / not recoverable from repository history
- **wcv:** 1.75170
- **fold_cal:** [1.77209, 1.76537, 1.75326, 1.74495]
- **mean_z:** 2.67982
- **lb_public:** Unknown / not recoverable from repository history
