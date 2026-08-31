# Logged run — FNL-BASER2-S42

## Catalogue metadata

- **Catalogue ID:** `team_a_run__fnl_baser2_s42`
- **Namespace:** `team_a_run`
- **Experiment ID:** `FNL-BASER2-S42`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** logged run / arm
- **Model:** dilated TCN, sequence model
- **Features:** funnel features, history-depth features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** description:** EXP-038 FNL: future funnel supervision (Search/Cart) поверх SEQ-D3A, фолд 10-16, сид 42 | арка BASE-R2, lambda 0.0
- **Known score:** Unknown / not recoverable from repository history
- **Seed:** verdict:** КОНТРОЛЬ ШУМА: повтор BASE тем же сидом; цена прогона 0.00033
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — FNL-BASER2-S42

This run was recovered from `experiments/log.csv`.

- **exp_id:** FNL-BASER2-S42
- **timestamp:** 2026-08-21T10:36:16
- **commit:** a28a71f
- **description:** EXP-038 FNL: future funnel supervision (Search/Cart) поверх SEQ-D3A, фолд 10-16, сид 42 | арка BASE-R2, lambda 0.0
- **scenario:** S1
- **n_features:** 17
- **model:** tcn+aux
- **params:** {"arm": "BASE-R2", "depth_aug": 0.5, "epochs": 4, "heads": null, "lam": 0.0, "loss": "MSE(z30)+lam*s_z*mean(L_m/s_m)", "s_z": 5.4899, "seed": 42}
- **cutoffs:** 24 @ step 7
- **L:** Unknown / not recoverable from repository history
- **panel_blocks:** 3
- **fold_scores:** [1.74751]
- **cv_mean:** 1.74751
- **cv_std:** 0.00000
- **bias_mean:** 0.01261
- **best_offset:** 0.01261
- **cv_mean_calib:** 1.74746
- **delta_vs_b0:** Unknown / not recoverable from repository history
- **runtime_s:** Unknown / not recoverable from repository history
- **verdict:** КОНТРОЛЬ ШУМА: повтор BASE тем же сидом; цена прогона 0.00033
- **conclusion:** Δ к BASE +0.00033 (+1.0 цены прогона); Δ к BUYCTRL н/д; AUC(y>0) 0.84229
- **wcv:** Unknown / not recoverable from repository history
- **fold_cal:** [1.74746]
- **mean_z:** 2.61890
- **lb_public:** Unknown / not recoverable from repository history
