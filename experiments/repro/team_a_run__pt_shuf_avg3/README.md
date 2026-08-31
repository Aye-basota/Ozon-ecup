# Logged run — PT-SHUF-AVG3

## Catalogue metadata

- **Catalogue ID:** `team_a_run__pt_shuf_avg3`
- **Namespace:** `team_a_run`
- **Experiment ID:** `PT-SHUF-AVG3`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** logged run / arm
- **Model:** Unknown / not recoverable from repository history
- **Features:** personal-time features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** conclusion:** Контрольный вариант C. rho и весь профиль интервалов переставлены между пользователями фиксированной перестановкой при сохранении маргинального распределения (mean(rho) совпал до 4-го знака, rho совпала лишь у 0.19%). wCV 1.750428 — НЕ ХУЖЕ настоящего личного времени (1.750451) и лучше базы на 3 фолдах из 4 против 2 у B. По отдельным сидам среднее -0.000025 против +0.000021 у B. Сработал критерий закрытия направления «B ~ C»: выигрыша нет, а то, что есть, не связано с тем, ЧЕЙ это
- **Known score:** conclusion:** Контрольный вариант C. rho и весь профиль интервалов переставлены между пользователями фиксированной перестановкой при сохранении маргинального распределения (mean(rho) совпал до 4-го знака, rho совпала лишь у 0.19%). wCV 1.750428 — НЕ ХУЖЕ настоящего личного времени (1.750451) и лучше базы на 3 фолдах из 4 против 2 у B. По отдельным сидам среднее -0.000025 против +0.000021 у B. Сработал критерий закрытия направления «B ~ C»: выигрыша нет, а то, что есть, не связано с тем, ЧЕЙ это
- **Seed:** conclusion:** Контрольный вариант C. rho и весь профиль интервалов переставлены между пользователями фиксированной перестановкой при сохранении маргинального распределения (mean(rho) совпал до 4-го знака, rho совпала лишь у 0.19%). wCV 1.750428 — НЕ ХУЖЕ настоящего личного времени (1.750451) и лучше базы на 3 фолдах из 4 против 2 у B. По отдельным сидам среднее -0.000025 против +0.000021 у B. Сработал критерий закрытия направления «B ~ C»: выигрыша нет, а то, что есть, не связано с тем, ЧЕЙ это
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — PT-SHUF-AVG3

This run was recovered from `experiments/log.csv`.

- **exp_id:** PT-SHUF-AVG3
- **timestamp:** 2026-08-12T19:16:03
- **commit:** 34a2335
- **description:** STRATEGY_08 C: контроль честности — профиль личного времени переставлен между пользователями, avg3
- **scenario:** S1
- **n_features:** 257
- **model:** direct
- **params:** {"base": "S1-SEEDAVG3", "only_change": "feature set", "ptime": "full", "ptime_source": "shuf", "rounds": 300, "seeds": [42, 43, 44]}
- **cutoffs:** all @ step 7
- **L:** Unknown / not recoverable from repository history
- **panel_blocks:** 3
- **fold_scores:** [1.77223, 1.76464, 1.75203, 1.74446]
- **cv_mean:** 1.75834
- **cv_std:** 0.01078
- **bias_mean:** -0.05134
- **best_offset:** -0.05134
- **cv_mean_calib:** 1.75744
- **delta_vs_b0:** Unknown / not recoverable from repository history
- **runtime_s:** Unknown / not recoverable from repository history
- **verdict:** REJECT (контроль)
- **conclusion:** Контрольный вариант C. rho и весь профиль интервалов переставлены между пользователями фиксированной перестановкой при сохранении маргинального распределения (mean(rho) совпал до 4-го знака, rho совпала лишь у 0.19%). wCV 1.750428 — НЕ ХУЖЕ настоящего личного времени (1.750451) и лучше базы на 3 фолдах из 4 против 2 у B. По отдельным сидам среднее -0.000025 против +0.000021 у B. Сработал критерий закрытия направления «B ~ C»: выигрыша нет, а то, что есть, не связано с тем, ЧЕЙ это ритм. Детали exp_021
- **wcv:** 1.75043
- **fold_cal:** [1.77037, 1.76396, 1.75175, 1.74389]
- **mean_z:** 2.67697
- **lb_public:** Unknown / not recoverable from repository history
