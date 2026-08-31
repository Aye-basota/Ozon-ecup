# Logged run — PT-OD-AVG3

## Catalogue metadata

- **Catalogue ID:** `team_a_run__pt_od_avg3`
- **Namespace:** `team_a_run`
- **Experiment ID:** `PT-OD-AVG3`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** logged run / arm
- **Model:** Unknown / not recoverable from repository history
- **Features:** gap/burst features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** conclusion:** Вариант A. wCV 1.750566 против 1.750456 у S1-SEEDAVG3: +0.000110, 0 фолдов из 4. AUC(y>0) 0.84323 против 0.84326. Обобщение rec_over_buygap на весь личный распределительный профиль (od_rank, квантили, отклонение от ожидаемого момента) не добавляет ничего. Детали exp_021
- **Known score:** conclusion:** Вариант A. wCV 1.750566 против 1.750456 у S1-SEEDAVG3: +0.000110, 0 фолдов из 4. AUC(y>0) 0.84323 против 0.84326. Обобщение rec_over_buygap на весь личный распределительный профиль (od_rank, квантили, отклонение от ожидаемого момента) не добавляет ничего. Детали exp_021
- **Seed:** conclusion:** Вариант A. wCV 1.750566 против 1.750456 у S1-SEEDAVG3: +0.000110, 0 фолдов из 4. AUC(y>0) 0.84323 против 0.84326. Обобщение rec_over_buygap на весь личный распределительный профиль (od_rank, квантили, отклонение от ожидаемого момента) не добавляет ничего. Детали exp_021
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — PT-OD-AVG3

This run was recovered from `experiments/log.csv`.

- **exp_id:** PT-OD-AVG3
- **timestamp:** 2026-08-12T19:16:03
- **commit:** 34a2335
- **description:** STRATEGY_08 A: просрочка относительно собственного распределения интервалов (9 колонок), avg3
- **scenario:** S1
- **n_features:** 236
- **model:** direct
- **params:** {"base": "S1-SEEDAVG3", "only_change": "feature set", "ptime": "od", "ptime_source": "real", "rounds": 300, "seeds": [42, 43, 44]}
- **cutoffs:** all @ step 7
- **L:** Unknown / not recoverable from repository history
- **panel_blocks:** 3
- **fold_scores:** [1.77246, 1.76484, 1.75216, 1.74464]
- **cv_mean:** 1.75853
- **cv_std:** 0.01081
- **bias_mean:** -0.05287
- **best_offset:** -0.05288
- **cv_mean_calib:** 1.75758
- **delta_vs_b0:** Unknown / not recoverable from repository history
- **runtime_s:** Unknown / not recoverable from repository history
- **verdict:** REJECT
- **conclusion:** Вариант A. wCV 1.750566 против 1.750456 у S1-SEEDAVG3: +0.000110, 0 фолдов из 4. AUC(y>0) 0.84323 против 0.84326. Обобщение rec_over_buygap на весь личный распределительный профиль (od_rank, квантили, отклонение от ожидаемого момента) не добавляет ничего. Детали exp_021
- **wcv:** 1.75056
- **fold_cal:** [1.77053, 1.7641, 1.75186, 1.74404]
- **mean_z:** 2.67850
- **lb_public:** Unknown / not recoverable from repository history
