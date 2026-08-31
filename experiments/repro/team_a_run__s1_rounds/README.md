# Logged run — S1-ROUNDS

## Catalogue metadata

- **Catalogue ID:** `team_a_run__s1_rounds`
- **Namespace:** `team_a_run`
- **Experiment ID:** `S1-ROUNDS`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** logged run / arm
- **Model:** Unknown / not recoverable from repository history
- **Features:** 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** params:** {"rounds": 300, "curve": "25..1600, 16 точек", "argmin_wcv": 200, "argmin_fold_10-16": 300}
- **Known score:** conclusion:** S_05 вариант A. Кривая RMSLE_cal(rounds) для direct на 4 фолдах, 16 точек 25..1600, снята срезами по префиксу деревьев (--snap-save) за 28 мин. 600 раундов оказались ПЕРЕОБУЧЕНИЕМ, а не недобором: wCV 1.75170 (600) против 1.75103 (200, argmin), 1.75108 (300) и 1.75457 (1600). Правый край сетки стратегии стоит +0.00287 wCV — больше суммы всех выигрышей E0 и смеси с головой. Рекомендация strategy_1_results.md §9 п.2 «поднять раунды» опровергнута. Дельта -0.00062 к S1-E10 на 4 фолдах
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — S1-ROUNDS

This run was recovered from `experiments/log.csv`.

- **exp_id:** S1-ROUNDS
- **timestamp:** 2026-08-12T12:11:20
- **commit:** e3032ca
- **description:** S_05 A: кривая по раундам direct; боевая точка 300 раундов вместо 600
- **scenario:** S1
- **n_features:** 227
- **model:** direct
- **params:** {"rounds": 300, "curve": "25..1600, 16 точек", "argmin_wcv": 200, "argmin_fold_10-16": 300}
- **cutoffs:** Unknown / not recoverable from repository history
- **L:** Unknown / not recoverable from repository history
- **panel_blocks:** Unknown / not recoverable from repository history
- **fold_scores:** [1.7729, 1.76545, 1.75271, 1.74502]
- **cv_mean:** 1.75902
- **cv_std:** 0.01084
- **bias_mean:** -0.05077
- **best_offset:** -0.05077
- **cv_mean_calib:** 1.75814
- **delta_vs_b0:** Unknown / not recoverable from repository history
- **runtime_s:** Unknown / not recoverable from repository history
- **verdict:** ПРИНЯТО в разработку
- **conclusion:** S_05 вариант A. Кривая RMSLE_cal(rounds) для direct на 4 фолдах, 16 точек 25..1600, снята срезами по префиксу деревьев (--snap-save) за 28 мин. 600 раундов оказались ПЕРЕОБУЧЕНИЕМ, а не недобором: wCV 1.75170 (600) против 1.75103 (200, argmin), 1.75108 (300) и 1.75457 (1600). Правый край сетки стратегии стоит +0.00287 wCV — больше суммы всех выигрышей E0 и смеси с головой. Рекомендация strategy_1_results.md §9 п.2 «поднять раунды» опровергнута. Дельта -0.00062 к S1-E10 на 4 фолдах из 4 (диапазон exp_016 §6 «в разработку»). argmin по фолдам 150/200/250/300 при 18/20/22/24 обучающих cutoff ах — оптимум растёт с объёмом выборки, +50 раундов на 2 cutoff а; дно плоское (0.00003 между 200 и 300 на фолде 10-16 при парной SE 0.00012), поэтому позиция argmin внутри 150..300 не разрешена. Боевая точка 300 выбрана по фолду 10-16 (правило §Risks стратегии). Контроль методики: срез 600 воспроизвёл S1-E10 на 4 фолдах из 4 до 5-го знака, полный прогон 300 раундов побитово равен срезу 300 из прогона 1600 (Var=0.00000). Детали exp_017
- **wcv:** 1.75108
- **fold_cal:** [1.77106, 1.7648, 1.75243, 1.74447]
- **mean_z:** 2.67639
- **lb_public:** Unknown / not recoverable from repository history
