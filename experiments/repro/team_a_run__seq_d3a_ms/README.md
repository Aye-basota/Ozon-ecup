# Logged run — SEQ-D3A-MS

## Catalogue metadata

- **Catalogue ID:** `team_a_run__seq_d3a_ms`
- **Namespace:** `team_a_run`
- **Experiment ID:** `SEQ-D3A-MS`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** logged run / arm
- **Model:** dilated TCN, sequence model
- **Features:** history-depth features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** description:** exp_030c SEQ-D3A-MS: сводка 3 сида x 4 фолда, единица анализа - пофолдовая парная дельта
- **Known score:** params:** {"arch": "dilated causal TCN", "blocks": 8, "hidden": 64, "kernel": 3, "dropout": 0.1, "batch": 1024, "lr": 0.003, "wd": 0.01, "epochs": 4, "precision": "bf16", "seq_len": 365, "channels": 17, "depth_aug": 0.5, "depth_grid": [90, 120, 150, 180, 220, 254, 289], "compile": true, "seeds": [42, 43, 44], "envs": "42 локально eager, 43/44 A10 compile", "note": "wCV между средами не усредняется, усредняются только дельты"}
- **Seed:** conclusion:** ΔwCV -0.00095 (sd 0.00016, se 0.00009) по 3 сидам; гейт >=3/4 включая 10-16 пройден (3/4); порог отправки -0.0020 не взят; sd(D3A)/sd(BASE) 11.4x только на 09-18
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — SEQ-D3A-MS

This run was recovered from `experiments/log.csv`.

- **exp_id:** SEQ-D3A-MS
- **timestamp:** 2026-08-19T19:40:00
- **commit:** a28a71f
- **description:** exp_030c SEQ-D3A-MS: сводка 3 сида x 4 фолда, единица анализа - пофолдовая парная дельта
- **scenario:** S1
- **n_features:** 17
- **model:** tcn
- **params:** {"arch": "dilated causal TCN", "blocks": 8, "hidden": 64, "kernel": 3, "dropout": 0.1, "batch": 1024, "lr": 0.003, "wd": 0.01, "epochs": 4, "precision": "bf16", "seq_len": 365, "channels": 17, "depth_aug": 0.5, "depth_grid": [90, 120, 150, 180, 220, 254, 289], "compile": true, "seeds": [42, 43, 44], "envs": "42 локально eager, 43/44 A10 compile", "note": "wCV между средами не усредняется, усредняются только дельты"}
- **cutoffs:** all @ step 7
- **L:** Unknown / not recoverable from repository history
- **panel_blocks:** 3
- **fold_scores:** [-0.00049, 0.00047, -0.00105, -0.00131]
- **cv_mean:** Unknown / not recoverable from repository history
- **cv_std:** Unknown / not recoverable from repository history
- **bias_mean:** Unknown / not recoverable from repository history
- **best_offset:** Unknown / not recoverable from repository history
- **cv_mean_calib:** Unknown / not recoverable from repository history
- **delta_vs_b0:** -0.00095
- **runtime_s:** Unknown / not recoverable from repository history
- **verdict:** KEEP
- **conclusion:** ΔwCV -0.00095 (sd 0.00016, se 0.00009) по 3 сидам; гейт >=3/4 включая 10-16 пройден (3/4); порог отправки -0.0020 не взят; sd(D3A)/sd(BASE) 11.4x только на 09-18
- **wcv:** Unknown / not recoverable from repository history
- **fold_cal:** Unknown / not recoverable from repository history
- **mean_z:** Unknown / not recoverable from repository history
- **lb_public:** Unknown / not recoverable from repository history
