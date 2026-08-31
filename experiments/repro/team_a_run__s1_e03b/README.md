# Logged run — S1-E03b

## Catalogue metadata

- **Catalogue ID:** `team_a_run__s1_e03b`
- **Namespace:** `team_a_run`
- **Experiment ID:** `S1-E03b`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** logged run / arm
- **Model:** Unknown / not recoverable from repository history
- **Features:** calendar features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** wcv:** 1.76208
- **Known score:** wcv:** 1.76208
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — S1-E03b

This run was recovered from `experiments/log.csv`.

- **exp_id:** S1-E03b
- **timestamp:** 2026-08-10T21:34:17
- **commit:** b2f0287
- **description:** E03b: L=180 strict (min_hist 180, 16 cutoffs)
- **scenario:** S1
- **n_features:** 195
- **model:** direct
- **params:** {"L": 180, "calendar": false, "cutoffs": "all", "drop_groups": [], "keep_only": null, "min_history": 180, "model": "direct", "panel_blocks": 3, "params": {}, "rounds": 600, "step": 7, "train_blocks": 1, "weight_tau": null}
- **cutoffs:** 11 @ step 7
- **L:** 180
- **panel_blocks:** 3
- **fold_scores:** [1.78326, 1.77491, 1.76406, 1.75775]
- **cv_mean:** 1.76999
- **cv_std:** 0.00981
- **bias_mean:** -0.06651
- **best_offset:** -0.06500
- **cv_mean_calib:** 1.76861
- **delta_vs_b0:** Unknown / not recoverable from repository history
- **runtime_s:** 312.70000
- **verdict:** Unknown / not recoverable from repository history
- **conclusion:** Unknown / not recoverable from repository history
- **wcv:** 1.76208
- **fold_cal:** [1.78138, 1.77418, 1.76309, 1.75614]
- **mean_z:** 2.69213
- **lb_public:** Unknown / not recoverable from repository history
