# Logged run — ETX-AVG3

## Catalogue metadata

- **Catalogue ID:** `team_a_run__etx_avg3`
- **Namespace:** `team_a_run`
- **Experiment ID:** `ETX-AVG3`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** logged run / arm
- **Model:** event Transformer / ETX, dilated TCN, sequence model
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** description:** exp_037 EXP-037: ETX-AVG3 = лог-среднее сидов 42/43/44 ETX-01, 4 фолда каждый; архитектура не менялась, отличие только --seed
- **Known score:** params:** {"arch": "sparse event transformer (causal SDPA + time-ALiBi)", "seeds": [42, 43, 44], "batch": 512, "blocks": 5, "d_model": 128, "dropout": 0.1, "epochs": 4, "ffn": 384, "head_dim": 16, "heads": 8, "lr": 0.0015, "n_tok": 192, "params_n": 1115433, "precision": "bf16", "warmup": 500, "wd": 0.01, "wcv_per_seed": [1.74953, 1.74935, 1.74943], "seed_sd_wcv": 9e-05, "env": "сиды 43/44 A10 --compile, сид 42 локально eager"}
- **Seed:** conclusion:** exp_037 ЭТАП 1. ETX-AVG3 wCV 1.74861 — лучший одиночный sequence-член проекта (SEQ-D3A-AVG3 1.74941, SEQ-AVG3 1.74963, ETX-01-S42 1.74953). AUC(y>0) 0.84135 — НИЖЕ, чем у SEQ-AVG3 0.84159: ETX выигрывает величиной, не ранжированием. Сид-шум ETX по wCV sd=0.00009 (1.74953/1.74935/1.74943) против 0.00250 у TCN — на порядок стабильнее. Попарное расхождение сидов на OOF 0.0086..0.0097 (у TCN 0.008). УСРЕДНЕНИЕ СИДОВ НА OOF НИЧЕГО НЕ КУПИЛО В СМЕСИ: LOFO -0.00091 (1 сид) -> -0.00094 (2)
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — ETX-AVG3

This run was recovered from `experiments/log.csv`.

- **exp_id:** ETX-AVG3
- **timestamp:** 2026-08-20T12:20:00
- **commit:** a28a71f
- **description:** exp_037 EXP-037: ETX-AVG3 = лог-среднее сидов 42/43/44 ETX-01, 4 фолда каждый; архитектура не менялась, отличие только --seed
- **scenario:** S1
- **n_features:** 22
- **model:** event-transformer
- **params:** {"arch": "sparse event transformer (causal SDPA + time-ALiBi)", "seeds": [42, 43, 44], "batch": 512, "blocks": 5, "d_model": 128, "dropout": 0.1, "epochs": 4, "ffn": 384, "head_dim": 16, "heads": 8, "lr": 0.0015, "n_tok": 192, "params_n": 1115433, "precision": "bf16", "warmup": 500, "wd": 0.01, "wcv_per_seed": [1.74953, 1.74935, 1.74943], "seed_sd_wcv": 9e-05, "env": "сиды 43/44 A10 --compile, сид 42 локально eager"}
- **cutoffs:** all @ step 7
- **L:** Unknown / not recoverable from repository history
- **panel_blocks:** 3
- **fold_scores:** [1.77819, 1.76788, 1.75087, 1.74219]
- **cv_mean:** 1.75979
- **cv_std:** 0.01409
- **bias_mean:** -0.10030
- **best_offset:** -0.10030
- **cv_mean_calib:** 1.75588
- **delta_vs_b0:** -0.00102
- **runtime_s:** 9714
- **verdict:** ACCEPT как СОАВТОР (LOFO -0.00092 4/4), REJECT как замена (2/4)
- **conclusion:** exp_037 ЭТАП 1. ETX-AVG3 wCV 1.74861 — лучший одиночный sequence-член проекта (SEQ-D3A-AVG3 1.74941, SEQ-AVG3 1.74963, ETX-01-S42 1.74953). AUC(y>0) 0.84135 — НИЖЕ, чем у SEQ-AVG3 0.84159: ETX выигрывает величиной, не ранжированием. Сид-шум ETX по wCV sd=0.00009 (1.74953/1.74935/1.74943) против 0.00250 у TCN — на порядок стабильнее. Попарное расхождение сидов на OOF 0.0086..0.0097 (у TCN 0.008). УСРЕДНЕНИЕ СИДОВ НА OOF НИЧЕГО НЕ КУПИЛО В СМЕСИ: LOFO -0.00091 (1 сид) -> -0.00094 (2) -> -0.00092 (3), всё внутри шума; ценность AVG3 — тестовая сторона и снижение риска розыгрыша рецепта.
- **wcv:** 1.74861
- **fold_cal:** [1.76927, 1.76252, 1.74977, 1.74197]
- **mean_z:** 2.72600
- **lb_public:** Unknown / not recoverable from repository history
