# Logged run — SEQ-D3A-G2-S44

## Catalogue metadata

- **Catalogue ID:** `team_a_run__seq_d3a_g2_s44`
- **Namespace:** `team_a_run`
- **Experiment ID:** `SEQ-D3A-G2-S44`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** logged run / arm
- **Model:** dilated TCN, sequence model
- **Features:** history-depth features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** description:** exp_030c SEQ-D3A на сиде 44: 4 фолда парно BASE/D3A, арендованная A10 #2, --compile
- **Known score:** wcv:** 1.75163
- **Seed:** params:** {"arch": "dilated causal TCN", "blocks": 8, "hidden": 64, "kernel": 3, "dropout": 0.1, "batch": 1024, "lr": 0.003, "wd": 0.01, "epochs": 4, "precision": "bf16", "seq_len": 365, "channels": 17, "depth_aug": 0.5, "depth_grid": [90, 120, 150, 180, 220, 254, 289], "compile": true, "seed": 44, "base": "SEQ-D3A-G2-BASE-S44 (same machine, same regime)"}
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — SEQ-D3A-G2-S44

This run was recovered from `experiments/log.csv`.

- **exp_id:** SEQ-D3A-G2-S44
- **timestamp:** 2026-08-19T19:40:00
- **commit:** a28a71f
- **description:** exp_030c SEQ-D3A на сиде 44: 4 фолда парно BASE/D3A, арендованная A10 #2, --compile
- **scenario:** S1
- **n_features:** 17
- **model:** tcn
- **params:** {"arch": "dilated causal TCN", "blocks": 8, "hidden": 64, "kernel": 3, "dropout": 0.1, "batch": 1024, "lr": 0.003, "wd": 0.01, "epochs": 4, "precision": "bf16", "seq_len": 365, "channels": 17, "depth_aug": 0.5, "depth_grid": [90, 120, 150, 180, 220, 254, 289], "compile": true, "seed": 44, "base": "SEQ-D3A-G2-BASE-S44 (same machine, same regime)"}
- **cutoffs:** all @ step 7
- **L:** Unknown / not recoverable from repository history
- **panel_blocks:** 3
- **fold_scores:** [1.76938, 1.76308, 1.75332, 1.74571]
- **cv_mean:** Unknown / not recoverable from repository history
- **cv_std:** Unknown / not recoverable from repository history
- **bias_mean:** Unknown / not recoverable from repository history
- **best_offset:** Unknown / not recoverable from repository history
- **cv_mean_calib:** Unknown / not recoverable from repository history
- **delta_vs_b0:** -0.00107
- **runtime_s:** Unknown / not recoverable from repository history
- **verdict:** CONTINUE
- **conclusion:** лучше пофолдового BASE 1.75270 на 4/4 фолдах
- **wcv:** 1.75163
- **fold_cal:** [1.76938, 1.76308, 1.75332, 1.74571]
- **mean_z:** Unknown / not recoverable from repository history
- **lb_public:** Unknown / not recoverable from repository history
