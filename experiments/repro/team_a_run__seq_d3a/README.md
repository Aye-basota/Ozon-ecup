# Logged run — SEQ-D3A

## Catalogue metadata

- **Catalogue ID:** `team_a_run__seq_d3a`
- **Namespace:** `team_a_run`
- **Experiment ID:** `SEQ-D3A`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** logged run / arm
- **Model:** dilated TCN, sequence model
- **Features:** history-depth features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** description:** exp_030 SEQ-D3A: depth curriculum (случайная обрезка РЕАЛЬНЫХ дней), 4 фолда, seed 42
- **Known score:** conclusion:** exp_030 EXP-030. Один флаг --depth-aug 0.5: с p=0.5 вход обрезается до D~U{90,120,150,180,220,254,289} РЕАЛЬНЫХ дней; вход curriculum побитово равен боевому --depth-clip D (тест). wCV 1.75361 -> 1.75284, ΔwCV -0.00077, но лучше только на 2 фолдах из 4: 09-04 +0.00013, 09-18 +0.00355, 10-02 -0.00129, 10-16 -0.00171. AUC(y>0) склеенного 0.84052 -> 0.84042 (плоско; +0.00083/+0.00047 на двух поздних, -0.00137 на 09-18). Var(z-z_BASE)=0.02776 = 3.90x пола сидов, corr остатков 0.9944..0.
- **Seed:** conclusion:** exp_030 EXP-030. Один флаг --depth-aug 0.5: с p=0.5 вход обрезается до D~U{90,120,150,180,220,254,289} РЕАЛЬНЫХ дней; вход curriculum побитово равен боевому --depth-clip D (тест). wCV 1.75361 -> 1.75284, ΔwCV -0.00077, но лучше только на 2 фолдах из 4: 09-04 +0.00013, 09-18 +0.00355, 10-02 -0.00129, 10-16 -0.00171. AUC(y>0) склеенного 0.84052 -> 0.84042 (плоско; +0.00083/+0.00047 на двух поздних, -0.00137 на 09-18). Var(z-z_BASE)=0.02776 = 3.90x пола сидов, corr остатков 0.9944..0.
- **Postprocessing:** None documented
- **Submission:** conclusion:** exp_030 EXP-030. Один флаг --depth-aug 0.5: с p=0.5 вход обрезается до D~U{90,120,150,180,220,254,289} РЕАЛЬНЫХ дней; вход curriculum побитово равен боевому --depth-clip D (тест). wCV 1.75361 -> 1.75284, ΔwCV -0.00077, но лучше только на 2 фолдах из 4: 09-04 +0.00013, 09-18 +0.00355, 10-02 -0.00129, 10-16 -0.00171. AUC(y>0) склеенного 0.84052 -> 0.84042 (плоско; +0.00083/+0.00047 на двух поздних, -0.00137 на 09-18). Var(z-z_BASE)=0.02776 = 3.90x пола сидов, corr остатков 0.9944..0.
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — SEQ-D3A

This run was recovered from `experiments/log.csv`.

- **exp_id:** SEQ-D3A
- **timestamp:** 2026-08-18T09:44:08
- **commit:** a28a71f
- **description:** exp_030 SEQ-D3A: depth curriculum (случайная обрезка РЕАЛЬНЫХ дней), 4 фолда, seed 42
- **scenario:** S1
- **n_features:** 17
- **model:** tcn
- **params:** {"arch": "dilated causal TCN", "blocks": 8, "hidden": 64, "kernel": 3, "dropout": 0.1, "batch": 1024, "lr": 0.003, "wd": 0.01, "epochs": 4, "precision": "bf16", "seq_len": 365, "channels": 17, "seed": 42, "depth_aug": 0.5, "depth_grid": [90, 120, 150, 180, 220, 254, 289], "base": "SEQ-D3A-BASE-S42 (same regime)"}
- **cutoffs:** all @ step 7
- **L:** Unknown / not recoverable from repository history
- **panel_blocks:** 3
- **fold_scores:** [1.77124, 1.77078, 1.75415, 1.74869]
- **cv_mean:** 1.76122
- **cv_std:** 0.00999
- **bias_mean:** Unknown / not recoverable from repository history
- **best_offset:** Unknown / not recoverable from repository history
- **cv_mean_calib:** 1.75961
- **delta_vs_b0:** Unknown / not recoverable from repository history
- **runtime_s:** Unknown / not recoverable from repository history
- **verdict:** CONTINUE (2/4; собственный гейт пройден, сессионный ≥3/4 — нет)
- **conclusion:** exp_030 EXP-030. Один флаг --depth-aug 0.5: с p=0.5 вход обрезается до D~U{90,120,150,180,220,254,289} РЕАЛЬНЫХ дней; вход curriculum побитово равен боевому --depth-clip D (тест). wCV 1.75361 -> 1.75284, ΔwCV -0.00077, но лучше только на 2 фолдах из 4: 09-04 +0.00013, 09-18 +0.00355, 10-02 -0.00129, 10-16 -0.00171. AUC(y>0) склеенного 0.84052 -> 0.84042 (плоско; +0.00083/+0.00047 на двух поздних, -0.00137 на 09-18). Var(z-z_BASE)=0.02776 = 3.90x пола сидов, corr остатков 0.9944..0.9969. Оба целевых сегмента лучше на ВСЕХ фолдах (rec_buy 15-60 -0.0005..-0.0025; полоса 2-15 -0.0009..-0.0012); весь проигрыш 09-18 сидит в строках без покупательной истории (никогда не покупал +0.04677 при доле 0.103, w180_days_buy 0-1 +0.02015). ГЛАВНЫЙ РЕЗУЛЬТАТ - ГЛУБИНА, и она противоположна exp_029: gain от 212 к полной вырос на 4 фолдах из 4 (-0.00262/-0.00374/-0.00270/-0.00308 -> -0.00267/-0.00403/-0.00290/-0.00424), штраф full-оптимум упал везде (+0.00087 -> +0.00033 на 10-16, на 09-18 оптимум стал полной глубиной). CROSSDEPTH (модель 09-04 на панели 10-16, +77 дней = точный аналог теста): gain -0.00155 -> -0.00368 (в 2.4 раза), внутренний оптимум 261 ИСЧЕЗ, лучшая глубина стала максимальной 289, full-оптимум +0.00143 -> 0. availprobe остался +0.00651 -> +0.00591: режим avail=1 по-прежнему непрожит, приём его НЕ вводит, политика теста остаётся --depth-clip 289. Дельта на 09-18 постоянна по глубине (+0.0035..+0.0039 на всех D) => проигрыш не глубинный. ГЕЙТЫ РАСХОДЯТСЯ: собственный гейт EXP-030 (PROBABLY_EXP) пройден по обоим условиям (10-16 -0.00171 при >= -0.0005; gain +77 не сжался при >= -0.0030), сессионный гейт >=3/4 фолдов НЕ пройден (2/4). Один сид неразрешим в принципе: seed std TCN по wCV 0.00250 = 3.2x наблюдённой дельты. Рекомендация: сперва дешёвый разделяющий замер 09-18 на сиде 43 (~66 мин), 3 сида только после него. LOFO/смесь/сабмит не считались. Детали exp_030
- **wcv:** 1.75284
- **fold_cal:** [1.76989, 1.76853, 1.75367, 1.74637]
- **mean_z:** 2.63145
- **lb_public:** Unknown / not recoverable from repository history
