# Logged run — FNL-BUYCTRL-L10-S42

## Catalogue metadata

- **Catalogue ID:** `team_a_run__fnl_buyctrl_l10_s42`
- **Namespace:** `team_a_run`
- **Experiment ID:** `FNL-BUYCTRL-L10-S42`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** logged run / arm
- **Model:** dilated TCN, sequence model
- **Features:** funnel features, history-depth features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** description:** EXP-038 FNL: future funnel supervision (Search/Cart) поверх SEQ-D3A, фолд 10-16, сид 42 | арка BUYCTRL, lambda 0.1
- **Known score:** Unknown / not recoverable from repository history
- **Seed:** params:** {"arm": "BUYCTRL", "depth_aug": 0.5, "epochs": 4, "heads": null, "lam": 0.1, "loss": "MSE(z30)+lam*s_z*mean(L_m/s_m)", "s_z": 5.4899, "seed": 42}
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — FNL-BUYCTRL-L10-S42

This run was recovered from `experiments/log.csv`.

- **exp_id:** FNL-BUYCTRL-L10-S42
- **timestamp:** 2026-08-21T10:36:16
- **commit:** a28a71f
- **description:** EXP-038 FNL: future funnel supervision (Search/Cart) поверх SEQ-D3A, фолд 10-16, сид 42 | арка BUYCTRL, lambda 0.1
- **scenario:** S1
- **n_features:** 17
- **model:** tcn+aux
- **params:** {"arm": "BUYCTRL", "depth_aug": 0.5, "epochs": 4, "heads": null, "lam": 0.1, "loss": "MSE(z30)+lam*s_z*mean(L_m/s_m)", "s_z": 5.4899, "seed": 42}
- **cutoffs:** 24 @ step 7
- **L:** Unknown / not recoverable from repository history
- **panel_blocks:** 3
- **fold_scores:** [1.74683]
- **cv_mean:** 1.74683
- **cv_std:** 0.00000
- **bias_mean:** 0.05271
- **best_offset:** 0.05271
- **cv_mean_calib:** 1.74603
- **delta_vs_b0:** Unknown / not recoverable from repository history
- **runtime_s:** Unknown / not recoverable from repository history
- **verdict:** REJECT-контроль: вторая голова без нового источника
- **conclusion:** Δ к BASE -0.00110 (-3.3 цены прогона); Δ к BUYCTRL н/д; AUC(y>0) 0.84292
- **wcv:** Unknown / not recoverable from repository history
- **fold_cal:** [1.74603]
- **mean_z:** 2.57880
- **lb_public:** Unknown / not recoverable from repository history
