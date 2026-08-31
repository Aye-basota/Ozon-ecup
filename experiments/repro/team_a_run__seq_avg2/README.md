# Logged run — SEQ-AVG2

## Catalogue metadata

- **Catalogue ID:** `team_a_run__seq_avg2`
- **Namespace:** `team_a_run`
- **Experiment ID:** `SEQ-AVG2`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** logged run / arm
- **Model:** dilated TCN, sequence model
- **Features:** history-depth features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** conclusion:** exp_026 этап 2: SEQ-AVG2 = mean(сид42, сид43) в лог-пространстве. Одиночная wCV 1.75016 (+0.00068 к S1-DIST-MIX против +0.00322 у одного сида), впервые лучше лучшей одиночной табличной S1-ROUNDS на -0.00092, 4/4. AUC(y>0) 0.84265 всё ещё ниже смеси (-0.00069) — ранжирование не улучшилось. Var(z-z_mix) 0.03054 = 4.29x пола сидов. Честный LOFO при S1-E03a=0.10: -0.00160 (в выборке -0.00164), 4/4 включая 10-16 (-0.00134). Гейт этапа 2 пройден, запущен сид 44. Детали exp_026
- **Known score:** conclusion:** exp_026 этап 2: SEQ-AVG2 = mean(сид42, сид43) в лог-пространстве. Одиночная wCV 1.75016 (+0.00068 к S1-DIST-MIX против +0.00322 у одного сида), впервые лучше лучшей одиночной табличной S1-ROUNDS на -0.00092, 4/4. AUC(y>0) 0.84265 всё ещё ниже смеси (-0.00069) — ранжирование не улучшилось. Var(z-z_mix) 0.03054 = 4.29x пола сидов. Честный LOFO при S1-E03a=0.10: -0.00160 (в выборке -0.00164), 4/4 включая 10-16 (-0.00134). Гейт этапа 2 пройден, запущен сид 44. Детали exp_026
- **Seed:** conclusion:** exp_026 этап 2: SEQ-AVG2 = mean(сид42, сид43) в лог-пространстве. Одиночная wCV 1.75016 (+0.00068 к S1-DIST-MIX против +0.00322 у одного сида), впервые лучше лучшей одиночной табличной S1-ROUNDS на -0.00092, 4/4. AUC(y>0) 0.84265 всё ещё ниже смеси (-0.00069) — ранжирование не улучшилось. Var(z-z_mix) 0.03054 = 4.29x пола сидов. Честный LOFO при S1-E03a=0.10: -0.00160 (в выборке -0.00164), 4/4 включая 10-16 (-0.00134). Гейт этапа 2 пройден, запущен сид 44. Детали exp_026
- **Postprocessing:** conclusion:** exp_026 этап 2: SEQ-AVG2 = mean(сид42, сид43) в лог-пространстве. Одиночная wCV 1.75016 (+0.00068 к S1-DIST-MIX против +0.00322 у одного сида), впервые лучше лучшей одиночной табличной S1-ROUNDS на -0.00092, 4/4. AUC(y>0) 0.84265 всё ещё ниже смеси (-0.00069) — ранжирование не улучшилось. Var(z-z_mix) 0.03054 = 4.29x пола сидов. Честный LOFO при S1-E03a=0.10: -0.00160 (в выборке -0.00164), 4/4 включая 10-16 (-0.00134). Гейт этапа 2 пройден, запущен сид 44. Детали exp_026
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — SEQ-AVG2

This run was recovered from `experiments/log.csv`.

- **exp_id:** SEQ-AVG2
- **timestamp:** 2026-08-13T19:05:00
- **commit:** 34a2335
- **description:** exp_026 SEQ-02: усреднение сидов dilated TCN, [42, 43]
- **scenario:** S1
- **n_features:** 17
- **model:** tcn-avg
- **params:** {"arch": "dilated causal TCN", "batch": 1024, "blocks": 8, "channels": 17, "compile": true, "depth_policy": "full", "epochs": 4, "hidden": 64, "kernel": 3, "precision": "bf16", "seeds": [42, 43], "seq_len": 365}
- **cutoffs:** all @ step 7
- **L:** Unknown / not recoverable from repository history
- **panel_blocks:** 3
- **fold_scores:** [1.77026, 1.76342, 1.75205, 1.74423]
- **cv_mean:** 1.75749
- **cv_std:** 0.01005
- **bias_mean:** -0.01021
- **best_offset:** -0.01025
- **cv_mean_calib:** 1.75674
- **delta_vs_b0:** Unknown / not recoverable from repository history
- **runtime_s:** Unknown / not recoverable from repository history
- **verdict:** CONTINUE
- **conclusion:** exp_026 этап 2: SEQ-AVG2 = mean(сид42, сид43) в лог-пространстве. Одиночная wCV 1.75016 (+0.00068 к S1-DIST-MIX против +0.00322 у одного сида), впервые лучше лучшей одиночной табличной S1-ROUNDS на -0.00092, 4/4. AUC(y>0) 0.84265 всё ещё ниже смеси (-0.00069) — ранжирование не улучшилось. Var(z-z_mix) 0.03054 = 4.29x пола сидов. Честный LOFO при S1-E03a=0.10: -0.00160 (в выборке -0.00164), 4/4 включая 10-16 (-0.00134). Гейт этапа 2 пройден, запущен сид 44. Детали exp_026
- **wcv:** 1.75016
- **fold_cal:** [1.76878, 1.7629, 1.75112, 1.74417]
- **mean_z:** 2.63584
- **lb_public:** Unknown / not recoverable from repository history
