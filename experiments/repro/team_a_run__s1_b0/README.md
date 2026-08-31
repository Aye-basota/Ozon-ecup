# Logged run — S1-B0

## Catalogue metadata

- **Catalogue ID:** `team_a_run__s1_b0`
- **Namespace:** `team_a_run`
- **Experiment ID:** `S1-B0`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** logged run / arm
- **Model:** Unknown / not recoverable from repository history
- **Features:** calendar features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** wcv:** 1.75743
- **Known score:** wcv:** 1.75743
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — S1-B0

This run was recovered from `experiments/log.csv`.

- **exp_id:** S1-B0
- **timestamp:** 2026-08-10T20:59:45
- **commit:** 37828cd
- **description:** B0 naive: uncapped feats, 1-block train panel, 3 recent cutoffs
- **scenario:** S1
- **n_features:** 236
- **model:** direct
- **params:** {"L": null, "calendar": false, "cutoffs": "recent3", "drop_groups": [], "keep_only": null, "min_history": 90, "model": "direct", "panel_blocks": 3, "params": {}, "rounds": 600, "step": 7, "train_blocks": 1, "weight_tau": null}
- **cutoffs:** 3 @ step 7
- **L:** Unknown / not recoverable from repository history
- **panel_blocks:** 3
- **fold_scores:** [1.78975, 1.77428, 1.75873, 1.7524]
- **cv_mean:** 1.76879
- **cv_std:** 0.01448
- **bias_mean:** -0.10222
- **best_offset:** -0.10000
- **cv_mean_calib:** 1.76570
- **delta_vs_b0:** Unknown / not recoverable from repository history
- **runtime_s:** 158.10000
- **verdict:** Unknown / not recoverable from repository history
- **conclusion:** Unknown / not recoverable from repository history
- **wcv:** 1.75743
- **fold_cal:** [1.77868, 1.77096, 1.7585, 1.75086]
- **mean_z:** 2.72704
- **lb_public:** Unknown / not recoverable from repository history
