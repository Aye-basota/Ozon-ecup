# Logged run — ETX-01-S42

## Catalogue metadata

- **Catalogue ID:** `team_a_run__etx_01_s42`
- **Namespace:** `team_a_run`
- **Experiment ID:** `ETX-01-S42`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** logged run / arm
- **Model:** event Transformer / ETX, sequence model, calibration diagnostic
- **Features:** calendar features, history-depth features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** description:** exp_036 ETX-01: sparse event transformer (токен = реальный день лога, <=192 события + query, causal SDPA, ALiBi в календарном времени); быстрый gate на фолде 2025-10-16, seed 42
- **Known score:** conclusion:** exp_036 гейт STRATEGY_13 вариант B на фолде 2025-10-16, сид 42, 1,115,433 параметров. Калибр. RMSLE 1.74306 против 1.74637 у SEQ-D3A-S42 (Δ -0.00331) и 1.74808 у SEQ-D3A-BASE-S42 без depth-curriculum. AUC(y>0) 0.84360 (Δ +0.00065). Var(z-z_SEQ)=0.03286, Var(z-z_DIST_MIX)=0.02373, corr остатков с SEQ 0.99460, с DIST-MIX 0.99609. tau_h разошлись 1.0..4552.2 д. 116 мин, пик VRAM 4.12 ГБ, 2,901 примеров/с на RTX 4060 Ti.
- **Seed:** conclusion:** exp_036 гейт STRATEGY_13 вариант B на фолде 2025-10-16, сид 42, 1,115,433 параметров. Калибр. RMSLE 1.74306 против 1.74637 у SEQ-D3A-S42 (Δ -0.00331) и 1.74808 у SEQ-D3A-BASE-S42 без depth-curriculum. AUC(y>0) 0.84360 (Δ +0.00065). Var(z-z_SEQ)=0.03286, Var(z-z_DIST_MIX)=0.02373, corr остатков с SEQ 0.99460, с DIST-MIX 0.99609. tau_h разошлись 1.0..4552.2 д. 116 мин, пик VRAM 4.12 ГБ, 2,901 примеров/с на RTX 4060 Ti.
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — ETX-01-S42

This run was recovered from `experiments/log.csv`.

- **exp_id:** ETX-01-S42
- **timestamp:** 2026-08-20T02:18:21
- **commit:** a28a71f
- **description:** exp_036 ETX-01: sparse event transformer (токен = реальный день лога, <=192 события + query, causal SDPA, ALiBi в календарном времени); быстрый gate на фолде 2025-10-16, seed 42
- **scenario:** S1
- **n_features:** 22
- **model:** event-transformer
- **params:** {"arch": "sparse event transformer (causal SDPA + time-ALiBi)", "batch": 512, "blocks": 5, "d_model": 128, "dropout": 0.1, "epoch_cal": [1.751, 1.74446, 1.74299, 1.74306], "epochs": 4, "ffn": 384, "fold": "2025-10-16", "head_dim": 16, "heads": 8, "lr": 0.0015, "n_tok": 192, "opt": "AdamW cosine, lr x10 \u0434\u043b\u044f log_m", "params_n": 1115433, "peak_vram_gb": 4.12, "precision": "bf16", "rows_s": 2901, "seed": 42, "static_features": 6, "supervision": "\u043e\u0434\u0438\u043d forward = \u043e\u0434\u0438\u043d \u043f\u0440\u0438\u043c\u0435\u0440 (\u0438\u0441\u0442\u043e\u0440\u0438\u044f<=T -> y30(T))", "tau_final_max": 4552.225, "tau_final_min": 0.966, "tok_features": 22, "warmup": 500, "wd": 0.01}
- **cutoffs:** 24 @ step 7
- **L:** Unknown / not recoverable from repository history
- **panel_blocks:** 3
- **fold_scores:** [1.74308]
- **cv_mean:** 1.74308
- **cv_std:** 0.00000
- **bias_mean:** -0.00723
- **best_offset:** -0.00723
- **cv_mean_calib:** 1.74306
- **delta_vs_b0:** -0.00331
- **runtime_s:** 6960.80000
- **verdict:** CONTINUE
- **conclusion:** exp_036 гейт STRATEGY_13 вариант B на фолде 2025-10-16, сид 42, 1,115,433 параметров. Калибр. RMSLE 1.74306 против 1.74637 у SEQ-D3A-S42 (Δ -0.00331) и 1.74808 у SEQ-D3A-BASE-S42 без depth-curriculum. AUC(y>0) 0.84360 (Δ +0.00065). Var(z-z_SEQ)=0.03286, Var(z-z_DIST_MIX)=0.02373, corr остатков с SEQ 0.99460, с DIST-MIX 0.99609. tau_h разошлись 1.0..4552.2 д. 116 мин, пик VRAM 4.12 ГБ, 2,901 примеров/с на RTX 4060 Ti.
- **wcv:** Unknown / not recoverable from repository history
- **fold_cal:** [1.74306]
- **mean_z:** 2.63875
- **lb_public:** Unknown / not recoverable from repository history
