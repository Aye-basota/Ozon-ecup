# Logged run — SEQ-AVG3

## Catalogue metadata

- **Catalogue ID:** `team_a_run__seq_avg3`
- **Namespace:** `team_a_run`
- **Experiment ID:** `SEQ-AVG3`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** logged run / arm
- **Model:** dilated TCN, sequence model, calibration diagnostic
- **Features:** history-depth features, 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** conclusion:** exp_026 этап 3: SEQ-AVG3 = mean(42,43,44). Одиночная wCV 1.74963 — ПАРИТЕТ со всей боевой смесью 1.74948 (+0.00015 при поле разрешения 0.0005) без единого из 227 признаков; против S1-ROUNDS -0.00144, 4/4, оба проблемных сегмента в минусе (rec_buy 15-60 -0.00231, полоса 2-15 -0.00172). AUC(y>0) 0.84277 по-прежнему ниже смеси (-0.00057): усреднение снимало шум, а не добавляло сигнал. РАЗЛОЖЕНИЕ РАЗНООБРАЗИЯ: Var(z-z_mix) 0.0371(k=1) -> 0.03054(k=2) -> 0.02844(k=3), предел C=Cov между
- **Known score:** conclusion:** exp_026 этап 3: SEQ-AVG3 = mean(42,43,44). Одиночная wCV 1.74963 — ПАРИТЕТ со всей боевой смесью 1.74948 (+0.00015 при поле разрешения 0.0005) без единого из 227 признаков; против S1-ROUNDS -0.00144, 4/4, оба проблемных сегмента в минусе (rec_buy 15-60 -0.00231, полоса 2-15 -0.00172). AUC(y>0) 0.84277 по-прежнему ниже смеси (-0.00057): усреднение снимало шум, а не добавляло сигнал. РАЗЛОЖЕНИЕ РАЗНООБРАЗИЯ: Var(z-z_mix) 0.0371(k=1) -> 0.03054(k=2) -> 0.02844(k=3), предел C=Cov между
- **Seed:** conclusion:** exp_026 этап 3: SEQ-AVG3 = mean(42,43,44). Одиночная wCV 1.74963 — ПАРИТЕТ со всей боевой смесью 1.74948 (+0.00015 при поле разрешения 0.0005) без единого из 227 признаков; против S1-ROUNDS -0.00144, 4/4, оба проблемных сегмента в минусе (rec_buy 15-60 -0.00231, полоса 2-15 -0.00172). AUC(y>0) 0.84277 по-прежнему ниже смеси (-0.00057): усреднение снимало шум, а не добавляло сигнал. РАЗЛОЖЕНИЕ РАЗНООБРАЗИЯ: Var(z-z_mix) 0.0371(k=1) -> 0.03054(k=2) -> 0.02844(k=3), предел C=Cov между
- **Postprocessing:** None documented
- **Submission:** conclusion:** exp_026 этап 3: SEQ-AVG3 = mean(42,43,44). Одиночная wCV 1.74963 — ПАРИТЕТ со всей боевой смесью 1.74948 (+0.00015 при поле разрешения 0.0005) без единого из 227 признаков; против S1-ROUNDS -0.00144, 4/4, оба проблемных сегмента в минусе (rec_buy 15-60 -0.00231, полоса 2-15 -0.00172). AUC(y>0) 0.84277 по-прежнему ниже смеси (-0.00057): усреднение снимало шум, а не добавляло сигнал. РАЗЛОЖЕНИЕ РАЗНООБРАЗИЯ: Var(z-z_mix) 0.0371(k=1) -> 0.03054(k=2) -> 0.02844(k=3), предел C=Cov между
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — SEQ-AVG3

This run was recovered from `experiments/log.csv`.

- **exp_id:** SEQ-AVG3
- **timestamp:** 2026-08-13T19:05:00
- **commit:** 34a2335
- **description:** exp_026 SEQ-02: усреднение сидов dilated TCN, [42, 43, 44]
- **scenario:** S1
- **n_features:** 17
- **model:** tcn-avg
- **params:** {"arch": "dilated causal TCN", "batch": 1024, "blocks": 8, "channels": 17, "compile": true, "depth_policy": "full", "epochs": 4, "hidden": 64, "kernel": 3, "precision": "bf16", "seeds": [42, 43, 44], "seq_len": 365}
- **cutoffs:** all @ step 7
- **L:** Unknown / not recoverable from repository history
- **panel_blocks:** 3
- **fold_scores:** [1.76936, 1.76282, 1.75173, 1.74381]
- **cv_mean:** 1.75693
- **cv_std:** 0.00986
- **bias_mean:** -0.00623
- **best_offset:** -0.00625
- **cv_mean_calib:** 1.75616
- **delta_vs_b0:** Unknown / not recoverable from repository history
- **runtime_s:** Unknown / not recoverable from repository history
- **verdict:** ACCEPT
- **conclusion:** exp_026 этап 3: SEQ-AVG3 = mean(42,43,44). Одиночная wCV 1.74963 — ПАРИТЕТ со всей боевой смесью 1.74948 (+0.00015 при поле разрешения 0.0005) без единого из 227 признаков; против S1-ROUNDS -0.00144, 4/4, оба проблемных сегмента в минусе (rec_buy 15-60 -0.00231, полоса 2-15 -0.00172). AUC(y>0) 0.84277 по-прежнему ниже смеси (-0.00057): усреднение снимало шум, а не добавляло сигнал. РАЗЛОЖЕНИЕ РАЗНООБРАЗИЯ: Var(z-z_mix) 0.0371(k=1) -> 0.03054(k=2) -> 0.02844(k=3), предел C=Cov между сидами = 0.02402 = 3.37x пола сидов GBDT; 35.2% разнообразия SEQ-01 было шумом обучения, 64.8% — устойчивая новая функция (контроль: другой GBDT 0.96x пола). Разложение V_k=D+s2/k при двух сидах — ТОЖДЕСТВО, а не проверяемая модель. Честный LOFO при S1-E03a=0.10: -0.00169 (в выборке -0.00174), 4/4 включая 10-16 (-0.00139). Рычаг исчерпан: 1->2 сида дали -0.00054, 2->3 только -0.00009. ОТНОСИТЕЛЬНО УЖЕ ОТПРАВЛЕННОГО SEQ-01-MIX прирост всего -0.00060. ЭТАП 1 (глубина, inference-only на 4 фолдах и 3 сидах): минимум калиброванного RMSLE ВСЕГДА на максимальной доступной глубине, кривая проходит границу обученной глубины без излома, предельная польза дня падает до 13e-6 к 289 — --depth-clip 289 в сабмите SEQ-01 был ошибкой ценой -0.0009..-0.0026 на фолд. Полный сид теперь 1.33 ч на A10 против 4.5 ч. Детали exp_026
- **wcv:** 1.74963
- **fold_cal:** [1.76816, 1.76225, 1.75052, 1.74372]
- **mean_z:** 2.63186
- **lb_public:** Unknown / not recoverable from repository history
