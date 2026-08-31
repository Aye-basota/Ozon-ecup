# Logged run — SEQ-D3A-S43

## Catalogue metadata

- **Catalogue ID:** `team_a_run__seq_d3a_s43`
- **Namespace:** `team_a_run`
- **Experiment ID:** `SEQ-D3A-S43`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** logged run / arm
- **Model:** dilated TCN, sequence model, blend
- **Features:** history-depth features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** description:** exp_030b SEQ-D3A-S43: разделяющий замер провала фолда 09-18, seed 43, 1 фолд, BASE + D3A
- **Known score:** conclusion:** exp_030b Разделяющий замер пункта 1 плана exp_030: фолд 2025-09-18, seed 43, BASE и D3A, всё остальное как в exp_030 (только --seed 43). ПРОВАЛ +0.00355 НЕ ВОСПРОИЗВЁЛСЯ: приём дал -0.00035 (BASE 1.76402 -> D3A 1.76367), ΔAUC -0.00137 -> -0.00007. Обе history-poor группы вернулись к нулю: 'никогда не покупал' +0.04677 (AUC -0.06367) -> -0.00115 (+0.01032); w180_days_buy 0-1 +0.02015 (-0.02466) -> +0.00024 (+0.00156). Целевые сегменты лучше на ОБОИХ сидах: rec_buy 15-60 -0.00050/-0.
- **Seed:** conclusion:** exp_030b Разделяющий замер пункта 1 плана exp_030: фолд 2025-09-18, seed 43, BASE и D3A, всё остальное как в exp_030 (только --seed 43). ПРОВАЛ +0.00355 НЕ ВОСПРОИЗВЁЛСЯ: приём дал -0.00035 (BASE 1.76402 -> D3A 1.76367), ΔAUC -0.00137 -> -0.00007. Обе history-poor группы вернулись к нулю: 'никогда не покупал' +0.04677 (AUC -0.06367) -> -0.00115 (+0.01032); w180_days_buy 0-1 +0.02015 (-0.02466) -> +0.00024 (+0.00156). Целевые сегменты лучше на ОБОИХ сидах: rec_buy 15-60 -0.00050/-0.
- **Postprocessing:** conclusion:** exp_030b Разделяющий замер пункта 1 плана exp_030: фолд 2025-09-18, seed 43, BASE и D3A, всё остальное как в exp_030 (только --seed 43). ПРОВАЛ +0.00355 НЕ ВОСПРОИЗВЁЛСЯ: приём дал -0.00035 (BASE 1.76402 -> D3A 1.76367), ΔAUC -0.00137 -> -0.00007. Обе history-poor группы вернулись к нулю: 'никогда не покупал' +0.04677 (AUC -0.06367) -> -0.00115 (+0.01032); w180_days_buy 0-1 +0.02015 (-0.02466) -> +0.00024 (+0.00156). Целевые сегменты лучше на ОБОИХ сидах: rec_buy 15-60 -0.00050/-0.
- **Submission:** conclusion:** exp_030b Разделяющий замер пункта 1 плана exp_030: фолд 2025-09-18, seed 43, BASE и D3A, всё остальное как в exp_030 (только --seed 43). ПРОВАЛ +0.00355 НЕ ВОСПРОИЗВЁЛСЯ: приём дал -0.00035 (BASE 1.76402 -> D3A 1.76367), ΔAUC -0.00137 -> -0.00007. Обе history-poor группы вернулись к нулю: 'никогда не покупал' +0.04677 (AUC -0.06367) -> -0.00115 (+0.01032); w180_days_buy 0-1 +0.02015 (-0.02466) -> +0.00024 (+0.00156). Целевые сегменты лучше на ОБОИХ сидах: rec_buy 15-60 -0.00050/-0.
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — SEQ-D3A-S43

This run was recovered from `experiments/log.csv`.

- **exp_id:** SEQ-D3A-S43
- **timestamp:** 2026-08-19T14:44:00
- **commit:** a28a71f
- **description:** exp_030b SEQ-D3A-S43: разделяющий замер провала фолда 09-18, seed 43, 1 фолд, BASE + D3A
- **scenario:** S1
- **n_features:** 17
- **model:** tcn
- **params:** {"arch": "dilated causal TCN", "blocks": 8, "hidden": 64, "kernel": 3, "dropout": 0.1, "batch": 1024, "lr": 0.003, "wd": 0.01, "epochs": 4, "precision": "bf16", "seq_len": 365, "channels": 17, "seed": 43, "depth_aug": 0.5, "depth_grid": [90, 120, 150, 180, 220, 254, 289], "folds": ["2025-09-18"], "base": "SEQ-D3A-BASE-S43 (same regime, retrained)"}
- **cutoffs:** all @ step 7
- **L:** Unknown / not recoverable from repository history
- **panel_blocks:** 3
- **fold_scores:** [1.76368]
- **cv_mean:** 1.76368
- **cv_std:** 0.00000
- **bias_mean:** -0.0048
- **best_offset:** -0.005
- **cv_mean_calib:** 1.76367
- **delta_vs_b0:** -0.00035
- **runtime_s:** 3647
- **verdict:** SCALE TO 3 SEEDS (провал 09-18 не воспроизвёлся: +0.00355 -> -0.00035)
- **conclusion:** exp_030b Разделяющий замер пункта 1 плана exp_030: фолд 2025-09-18, seed 43, BASE и D3A, всё остальное как в exp_030 (только --seed 43). ПРОВАЛ +0.00355 НЕ ВОСПРОИЗВЁЛСЯ: приём дал -0.00035 (BASE 1.76402 -> D3A 1.76367), ΔAUC -0.00137 -> -0.00007. Обе history-poor группы вернулись к нулю: 'никогда не покупал' +0.04677 (AUC -0.06367) -> -0.00115 (+0.01032); w180_days_buy 0-1 +0.02015 (-0.02466) -> +0.00024 (+0.00156). Целевые сегменты лучше на ОБОИХ сидах: rec_buy 15-60 -0.00050/-0.00048, полоса 2-15 -0.00085/-0.00059. ГЛАВНОЕ, ЧЕГО НЕ БЫЛО В exp_030 - шумовая полоса на ЭТОМ ЖЕ фолде тем же кодом: Var(Δz) приёма на сиде 43 = 0.01858 МЕНЬШЕ, чем у чистой смены сида в BASE (0.02234), то есть эффект приёма меньше эффекта сида; и нестабилен именно вариант, а не база - по сидам BASE разошёлся на -0.00095, D3A на -0.00485 (впятеро), на 'никогда не покупал' -0.01119 против -0.05911. Разложение уровень/ранжирование: на сиде 42 D3A ушёл в грубое перепрогнозирование history-poor (bias +0.078 -> -0.186) и со своим сдвигом сегмент оставался хуже на +0.0357 => ломался не только уровень; на сиде 43 bias +0.0290 -> +0.0317. ГЛУБИНА: дельта приёма по глубине на сиде 43 около нуля на всех D (-0.00128..-0.00007) против константы +0.0015..+0.0039 на сиде 42 - провал не глубинный ни на одном сиде. ОГОВОРКА К exp_030: пофолдовый рост gain на 09-18 (-0.00374 -> -0.00403) приёму НЕ принадлежит, BASE сида 43 сам даёт -0.00409 и оптимум 261; это НЕ касается crossdepth 10-16 (-0.00155 -> -0.00368), но и его надо перепроверить на 3 сидах. availprobe на 09-18 BASE +0.00287 / D3A +0.00410 - режим avail=1 непрожит, политика теста остаётся --depth-clip 289. МЕХАНИЗМ (замер по данным, без модели): обрезка до D обнуляет все поведенческие каналы старше D и ставит avail=0, поэтому пользователь с rec_buy > D неотличим на входе от непокупателя, а таргет у него свой и выше. На 20 обучающих cutoff'ах фолда при D=90 20.5% области 'покупок не видно' - спрятанные покупатели с E z 1.061 против 0.490 и P(y>0) 0.285 против 0.135; взвешенная по сетке Δz +0.52, ожидаемая доля отравленных предъявлений при p=0.5 около 0.85% строк на эпоху. Объём шума мал, но подмешан ровно в history-poor - это объясняет 5-кратный разброс варианта по сидам. wCV НЕ СЧИТАЕТСЯ: определён только на полной схеме S1 из 4 фолдов (ассерт в validation.wcv), 09-18 весит 2 из 15; оценка 'если бы 09-18 был как на сиде 43' даёт ΔwCV -0.00077 -> -0.00129 и 2/4 -> 3/4, но это вклад ОДНОГО фолда при смешанных сидах, не метрика, и порога отправки -0.0020 не берёт. Blend/LOFO/сабмит не считались. Рекомендация: полный 3 сида x 4 фолда, и смотреть не только среднее ΔwCV, но и разброс варианта против базы; при подтверждённой нестабильности - сетка глубин с 150, без 90/120. Детали exp_030b
- **wcv:** Unknown / not recoverable from repository history
- **fold_cal:** [1.76367]
- **mean_z:** 2.6456
- **lb_public:** Unknown / not recoverable from repository history
