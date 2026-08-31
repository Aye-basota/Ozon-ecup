# Logged run — S1-SEEDAVG5

## Catalogue metadata

- **Catalogue ID:** `team_a_run__s1_seedavg5`
- **Namespace:** `team_a_run`
- **Experiment ID:** `S1-SEEDAVG5`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** logged run / arm
- **Model:** Unknown / not recoverable from repository history
- **Features:** 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** conclusion:** S_05 вариант B. Пять сидов (42..46) боевой конфигурации direct при 300 раундах. Разброс wCV между сидами: std 0.00010, range 0.00023 — МЕНЬШЕ порога закрытия направления 0.0003 из стратегии, но само правило оказалось мисспецифицировано: усреднение всё равно дало -0.00071 к сиду 42 и -0.00077 к среднему сиду, на 4 фолдах из 4. Разброс скора по сидам мал потому, что скор уже усредняет 190k строк; убирает же усреднение дисперсию ПРЕДСКАЗАНИЙ: Var(z_i - z_j) = 0.00712 в среднем по 10 п
- **Known score:** conclusion:** S_05 вариант B. Пять сидов (42..46) боевой конфигурации direct при 300 раундах. Разброс wCV между сидами: std 0.00010, range 0.00023 — МЕНЬШЕ порога закрытия направления 0.0003 из стратегии, но само правило оказалось мисспецифицировано: усреднение всё равно дало -0.00071 к сиду 42 и -0.00077 к среднему сиду, на 4 фолдах из 4. Разброс скора по сидам мал потому, что скор уже усредняет 190k строк; убирает же усреднение дисперсию ПРЕДСКАЗАНИЙ: Var(z_i - z_j) = 0.00712 в среднем по 10 п
- **Seed:** conclusion:** S_05 вариант B. Пять сидов (42..46) боевой конфигурации direct при 300 раундах. Разброс wCV между сидами: std 0.00010, range 0.00023 — МЕНЬШЕ порога закрытия направления 0.0003 из стратегии, но само правило оказалось мисспецифицировано: усреднение всё равно дало -0.00071 к сиду 42 и -0.00077 к среднему сиду, на 4 фолдах из 4. Разброс скора по сидам мал потому, что скор уже усредняет 190k строк; убирает же усреднение дисперсию ПРЕДСКАЗАНИЙ: Var(z_i - z_j) = 0.00712 в среднем по 10 п
- **Postprocessing:** params:** {"rounds": 300, "seeds": [42,43,44,45,46], "avg": "log-space equal weights"}
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — S1-SEEDAVG5

This run was recovered from `experiments/log.csv`.

- **exp_id:** S1-SEEDAVG5
- **timestamp:** 2026-08-12T12:10:04
- **commit:** e3032ca
- **description:** S_05 B: среднее 5 сидов direct при 300 раундах
- **scenario:** S1
- **n_features:** 227
- **model:** direct
- **params:** {"rounds": 300, "seeds": [42,43,44,45,46], "avg": "log-space equal weights"}
- **cutoffs:** Unknown / not recoverable from repository history
- **L:** Unknown / not recoverable from repository history
- **panel_blocks:** Unknown / not recoverable from repository history
- **fold_scores:** [1.77214, 1.76457, 1.75202, 1.74437]
- **cv_mean:** 1.75827
- **cv_std:** 0.01077
- **bias_mean:** -0.05107
- **best_offset:** -0.05108
- **cv_mean_calib:** 1.75738
- **delta_vs_b0:** Unknown / not recoverable from repository history
- **runtime_s:** Unknown / not recoverable from repository history
- **verdict:** ПРИНЯТО в разработку
- **conclusion:** S_05 вариант B. Пять сидов (42..46) боевой конфигурации direct при 300 раундах. Разброс wCV между сидами: std 0.00010, range 0.00023 — МЕНЬШЕ порога закрытия направления 0.0003 из стратегии, но само правило оказалось мисспецифицировано: усреднение всё равно дало -0.00071 к сиду 42 и -0.00077 к среднему сиду, на 4 фолдах из 4. Разброс скора по сидам мал потому, что скор уже усредняет 190k строк; убирает же усреднение дисперсию ПРЕДСКАЗАНИЙ: Var(z_i - z_j) = 0.00712 в среднем по 10 парам, то есть sigma^2 = 0.00356 на сид. Кривая усреднения монотонна и совпала с теорией MSE_k = M + sigma^2/k с точностью 0.00005: k=1 1.75114, k=2 1.75066, k=3 1.75050, k=4 1.75042, k=5 1.75037. Плато после 3 сидов (3->5 даёт лишь 0.00013). ГЛАВНЫЙ побочный результат: 0.00712 — это пол разнообразия проекта. Var(z_DIST - z_E10)=0.01320 и Var(z_E11 - z_DIST)=0.0107 всего в 1.5-1.8 раза выше пола, то есть половина «разнообразия» смеси — переобучение сидов, а не разные функции. A+B вместе: S1-E10 1.75170 -> 1.75037 (-0.00133) на 4 фолдах из 4. Детали exp_018
- **wcv:** 1.75037
- **fold_cal:** [1.77026, 1.76391, 1.75175, 1.74381]
- **mean_z:** 2.67670
- **lb_public:** Unknown / not recoverable from repository history
