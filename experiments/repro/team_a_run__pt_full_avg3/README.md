# Logged run — PT-FULL-AVG3

## Catalogue metadata

- **Catalogue ID:** `team_a_run__pt_full_avg3`
- **Namespace:** `team_a_run`
- **Experiment ID:** `PT-FULL-AVG3`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** logged run / arm
- **Model:** Unknown / not recoverable from repository history
- **Features:** personal-time features, gap/burst features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** conclusion:** КЛЮЧЕВОЙ вариант. wCV 1.750451 против 1.750456: -0.000006 при пороге exp_016 §6 -0.0005 (промах в 83 раза), 2 фолда из 4, последний фолд ХУЖЕ базы (1.74398 против 1.74391). AUC(y>0) 0.84326 -> 0.84326: -0.0000004 при требуемых по обменному курсу диагностики +0.003..+0.008. Целевые сегменты не улучшились: rec_buy 15-60 RMSLE +0.00012/AUC -0.00002, полоса 2-15 покупательных дней +0.00004/-0.00004, их пересечение +0.00025/-0.00012. Разнообразия нет: Var(z-z_база)=0.00270 против пола с
- **Known score:** conclusion:** КЛЮЧЕВОЙ вариант. wCV 1.750451 против 1.750456: -0.000006 при пороге exp_016 §6 -0.0005 (промах в 83 раза), 2 фолда из 4, последний фолд ХУЖЕ базы (1.74398 против 1.74391). AUC(y>0) 0.84326 -> 0.84326: -0.0000004 при требуемых по обменному курсу диагностики +0.003..+0.008. Целевые сегменты не улучшились: rec_buy 15-60 RMSLE +0.00012/AUC -0.00002, полоса 2-15 покупательных дней +0.00004/-0.00004, их пересечение +0.00025/-0.00012. Разнообразия нет: Var(z-z_база)=0.00270 против пола с
- **Seed:** conclusion:** КЛЮЧЕВОЙ вариант. wCV 1.750451 против 1.750456: -0.000006 при пороге exp_016 §6 -0.0005 (промах в 83 раза), 2 фолда из 4, последний фолд ХУЖЕ базы (1.74398 против 1.74391). AUC(y>0) 0.84326 -> 0.84326: -0.0000004 при требуемых по обменному курсу диагностики +0.003..+0.008. Целевые сегменты не улучшились: rec_buy 15-60 RMSLE +0.00012/AUC -0.00002, полоса 2-15 покупательных дней +0.00004/-0.00004, их пересечение +0.00025/-0.00012. Разнообразия нет: Var(z-z_база)=0.00270 против пола с
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — PT-FULL-AVG3

This run was recovered from `experiments/log.csv`.

- **exp_id:** PT-FULL-AVG3
- **timestamp:** 2026-08-12T19:16:03
- **commit:** 34a2335
- **description:** STRATEGY_08 B: полное представление в личном времени (30 колонок), avg3
- **scenario:** S1
- **n_features:** 257
- **model:** direct
- **params:** {"base": "S1-SEEDAVG3", "only_change": "feature set", "ptime": "full", "ptime_source": "real", "rounds": 300, "seeds": [42, 43, 44]}
- **cutoffs:** all @ step 7
- **L:** Unknown / not recoverable from repository history
- **panel_blocks:** 3
- **fold_scores:** [1.77217, 1.76474, 1.752, 1.74459]
- **cv_mean:** 1.75838
- **cv_std:** 0.01074
- **bias_mean:** -0.05336
- **best_offset:** -0.05336
- **cv_mean_calib:** 1.75742
- **delta_vs_b0:** Unknown / not recoverable from repository history
- **runtime_s:** Unknown / not recoverable from repository history
- **verdict:** REJECT
- **conclusion:** КЛЮЧЕВОЙ вариант. wCV 1.750451 против 1.750456: -0.000006 при пороге exp_016 §6 -0.0005 (промах в 83 раза), 2 фолда из 4, последний фолд ХУЖЕ базы (1.74398 против 1.74391). AUC(y>0) 0.84326 -> 0.84326: -0.0000004 при требуемых по обменному курсу диагностики +0.003..+0.008. Целевые сегменты не улучшились: rec_buy 15-60 RMSLE +0.00012/AUC -0.00002, полоса 2-15 покупательных дней +0.00004/-0.00004, их пересечение +0.00025/-0.00012. Разнообразия нет: Var(z-z_база)=0.00270 против пола сидов 0.00712, corr остатков 0.99956 ВЫШЕ потолка похожести двух сидов 0.99885; corr остатков со смесью 0.99904. Подстановка в слот S1-E10 при фиксированных весах: 1.749484 -> 1.749451, из них -0.000018 уже давало усреднение сидов. Признаки при этом не игнорируются (pt_od_z_buy 5-й по gain на контрольном коротком прогоне) — они избыточны с rec_over_buygap. Гипотеза личного времени FAIL, STRATEGY_13 DEPRIORITIZED. Детали exp_021
- **wcv:** 1.75045
- **fold_cal:** [1.77023, 1.76399, 1.75168, 1.74398]
- **mean_z:** 2.67898
- **lb_public:** Unknown / not recoverable from repository history
