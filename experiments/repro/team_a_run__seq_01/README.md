# Logged run — SEQ-01

## Catalogue metadata

- **Catalogue ID:** `team_a_run__seq_01`
- **Namespace:** `team_a_run`
- **Experiment ID:** `SEQ-01`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** logged run / arm
- **Model:** dilated TCN, sequence model
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** conclusion:** Одиночная wCV 1.75270 (+0.00322 к S1-DIST-MIX, +0.00162 к лучшему табличному S1-ROUNDS), AUC(y>0) 0.84205 (-0.00129) — ранжирование НЕ улучшилось, ни один целевой сегмент не выигран. Главный результат — разнообразие: Var(z-z_mix)=0.03749 = 5.27x пола сидов 0.00712, corr остатков 0.99393 против потолка 0.99885. КОНТРОЛЬ: лучшая табличная модель S1-ROUNDS против той же смеси даёт Var 0.00685 (0.96x пола) и corr 0.99889, то есть ещё один GBDT неотличим от переcида. Подстановка в слот
- **Known score:** conclusion:** Одиночная wCV 1.75270 (+0.00322 к S1-DIST-MIX, +0.00162 к лучшему табличному S1-ROUNDS), AUC(y>0) 0.84205 (-0.00129) — ранжирование НЕ улучшилось, ни один целевой сегмент не выигран. Главный результат — разнообразие: Var(z-z_mix)=0.03749 = 5.27x пола сидов 0.00712, corr остатков 0.99393 против потолка 0.99885. КОНТРОЛЬ: лучшая табличная модель S1-ROUNDS против той же смеси даёт Var 0.00685 (0.96x пола) и corr 0.99889, то есть ещё один GBDT неотличим от переcида. Подстановка в слот
- **Seed:** conclusion:** Одиночная wCV 1.75270 (+0.00322 к S1-DIST-MIX, +0.00162 к лучшему табличному S1-ROUNDS), AUC(y>0) 0.84205 (-0.00129) — ранжирование НЕ улучшилось, ни один целевой сегмент не выигран. Главный результат — разнообразие: Var(z-z_mix)=0.03749 = 5.27x пола сидов 0.00712, corr остатков 0.99393 против потолка 0.99885. КОНТРОЛЬ: лучшая табличная модель S1-ROUNDS против той же смеси даёт Var 0.00685 (0.96x пола) и corr 0.99889, то есть ещё один GBDT неотличим от переcида. Подстановка в слот
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — SEQ-01

This run was recovered from `experiments/log.csv`.

- **exp_id:** SEQ-01
- **timestamp:** 2026-08-13T14:35:24
- **commit:** 34a2335
- **description:** exp_025 SEQ-01: dilated TCN на сырой дневной последовательности 365 дней (17 каналов), голова только z30
- **scenario:** S1
- **n_features:** 17
- **model:** tcn
- **params:** {"arch": "dilated causal TCN", "batch": 1024, "blocks": 8, "channels": 17, "dropout": 0.1, "epochs": 4, "hidden": 64, "kernel": 3, "lr": 0.003, "opt": "AdamW cosine", "params_n": 245633, "precision": "bf16", "rf_days": 511, "seed": 42, "seq_len": 365, "wd": 0.01}
- **cutoffs:** all @ step 7
- **L:** Unknown / not recoverable from repository history
- **panel_blocks:** 3
- **fold_scores:** [1.77124, 1.76544, 1.75358, 1.74757]
- **cv_mean:** 1.75946
- **cv_std:** 0.00936
- **bias_mean:** -0.00964
- **best_offset:** -0.00969
- **cv_mean_calib:** 1.75930
- **delta_vs_b0:** Unknown / not recoverable from repository history
- **runtime_s:** Unknown / not recoverable from repository history
- **verdict:** CONTINUE
- **conclusion:** Одиночная wCV 1.75270 (+0.00322 к S1-DIST-MIX, +0.00162 к лучшему табличному S1-ROUNDS), AUC(y>0) 0.84205 (-0.00129) — ранжирование НЕ улучшилось, ни один целевой сегмент не выигран. Главный результат — разнообразие: Var(z-z_mix)=0.03749 = 5.27x пола сидов 0.00712, corr остатков 0.99393 против потолка 0.99885. КОНТРОЛЬ: лучшая табличная модель S1-ROUNDS против той же смеси даёт Var 0.00685 (0.96x пола) и corr 0.99889, то есть ещё один GBDT неотличим от переcида. Подстановка в слот S1-E10 при неизменных весах: S1-ROUNDS -0.000005 (2/4), SEQ-01 -0.000767 (4/4). LOFO при фиксированной страховке S1-E03a=0.10: честный выигрыш -0.00106 (в выборке -0.00115), 4/4 фолда включая 10-16; со свободными весами -0.00131, но они обнуляют E03a (запрещено MIX-E11). Плато 28 комбинаций из 1330. Вывод: в сырой последовательности есть новый сигнал, но это сигнал РАЗНООБРАЗИЯ, а не ранжирования. Детали exp_025
- **wcv:** 1.75270
- **fold_cal:** [1.77057, 1.76475, 1.75352, 1.74704]
- **mean_z:** 2.63527
- **lb_public:** Unknown / not recoverable from repository history
