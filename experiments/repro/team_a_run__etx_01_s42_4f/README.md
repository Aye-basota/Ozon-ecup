# Logged run — ETX-01-S42-4F

## Catalogue metadata

- **Catalogue ID:** `team_a_run__etx_01_s42_4f`
- **Namespace:** `team_a_run`
- **Experiment ID:** `ETX-01-S42-4F`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** logged run / arm
- **Model:** event Transformer / ETX, dilated TCN, sequence model
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** description:** exp_036 ETX-01 полная схема S1: sparse event transformer, 4 фолда, seed 42; + честный LOFO слота SEQ
- **Known score:** conclusion:** exp_036 ПОЛНАЯ схема S1, 4 фолда, seed 42. wCV 1.74953 против 1.75284 у SEQ-D3A-S42 того же сида (ΔwCV -0.00331, 3/4 включая 10-16) и паритет с трёхсидовыми SEQ-D3A-AVG3 1.74941 / SEQ-AVG3 1.74963 ОДНОЙ моделью. Пофолдовые дельты к SEQ-D3A-S42: +0.00005 / -0.00463 / -0.00350 / -0.00331 — ранний фолд 09-04 ноль, что согласуется с механизмом (выигрыш ETX в ДЛИННОЙ истории: обрезка до 180д стоит +0.01259 против +0.00841 у TCN). ЧЕСТНЫЙ LOFO слота SEQ: как ЗАМЕНА -0.00035 на 2/4 (ГЕЙТ
- **Seed:** conclusion:** exp_036 ПОЛНАЯ схема S1, 4 фолда, seed 42. wCV 1.74953 против 1.75284 у SEQ-D3A-S42 того же сида (ΔwCV -0.00331, 3/4 включая 10-16) и паритет с трёхсидовыми SEQ-D3A-AVG3 1.74941 / SEQ-AVG3 1.74963 ОДНОЙ моделью. Пофолдовые дельты к SEQ-D3A-S42: +0.00005 / -0.00463 / -0.00350 / -0.00331 — ранний фолд 09-04 ноль, что согласуется с механизмом (выигрыш ETX в ДЛИННОЙ истории: обрезка до 180д стоит +0.01259 против +0.00841 у TCN). ЧЕСТНЫЙ LOFO слота SEQ: как ЗАМЕНА -0.00035 на 2/4 (ГЕЙТ
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — ETX-01-S42-4F

This run was recovered from `experiments/log.csv`.

- **exp_id:** ETX-01-S42-4F
- **timestamp:** 2026-08-20T06:52:30
- **commit:** a28a71f
- **description:** exp_036 ETX-01 полная схема S1: sparse event transformer, 4 фолда, seed 42; + честный LOFO слота SEQ
- **scenario:** S1
- **n_features:** 22
- **model:** event-transformer
- **params:** {"arch": "sparse event transformer (causal SDPA + time-ALiBi)", "batch": 512, "blocks": 5, "d_model": 128, "dropout": 0.1, "epochs": 4, "ffn": 384, "head_dim": 16, "heads": 8, "lofo_coexist_50": -0.0009055896704030619, "lofo_coexist_50_folds": 4, "lofo_replace": -0.0003507863549831865, "lofo_replace_folds": 2, "lr": 0.0015, "n_tok": 192, "params_n": 1115433, "precision": "bf16", "runtime_min_per_fold": [81.8, 91.4, 98.8, 116.0], "seed": 42, "warmup": 500, "wd": 0.01}
- **cutoffs:** all @ step 7
- **L:** Unknown / not recoverable from repository history
- **panel_blocks:** 3
- **fold_scores:** [1.7782, 1.77007, 1.75185, 1.74308]
- **cv_mean:** 1.76080
- **cv_std:** 0.01399
- **bias_mean:** -0.09961
- **best_offset:** -0.09961
- **cv_mean_calib:** 1.75780
- **delta_vs_b0:** -0.00331
- **runtime_s:** 23880.00000
- **verdict:** CONTINUE: как замена REJECT 2/4, как соавтор 0.5+0.5 LOFO -0.00091 4/4
- **conclusion:** exp_036 ПОЛНАЯ схема S1, 4 фолда, seed 42. wCV 1.74953 против 1.75284 у SEQ-D3A-S42 того же сида (ΔwCV -0.00331, 3/4 включая 10-16) и паритет с трёхсидовыми SEQ-D3A-AVG3 1.74941 / SEQ-AVG3 1.74963 ОДНОЙ моделью. Пофолдовые дельты к SEQ-D3A-S42: +0.00005 / -0.00463 / -0.00350 / -0.00331 — ранний фолд 09-04 ноль, что согласуется с механизмом (выигрыш ETX в ДЛИННОЙ истории: обрезка до 180д стоит +0.01259 против +0.00841 у TCN). ЧЕСТНЫЙ LOFO слота SEQ: как ЗАМЕНА -0.00035 на 2/4 (ГЕЙТ ПРОВАЛЕН), как СОАВТОР 0.5*ETX+0.5*SEQ-AVG3 = -0.00091 на 4/4 — лучший член слота за проект (было -0.00061 у SEQ-D3A-AVG3, -0.00055 у выбранного SEQ-AVG3). Инкремент к уже собранному SEQAVG3-CLIP-MIX -0.00036, ниже пола 0.0005. AUC(y>0) 0.84109. Разнообразие против табличной части: Var(z-z_tab) 0.02629 у ETX против 0.02909 у SEQ-AVG3 — ETX БЛИЖЕ к GBDT, и польза идёт не от расстояния до таблицы, а от расстояния до TCN.
- **wcv:** 1.74953
- **fold_cal:** [1.76994, 1.7639, 1.75017, 1.74306]
- **mean_z:** 2.72523
- **lb_public:** Unknown / not recoverable from repository history
