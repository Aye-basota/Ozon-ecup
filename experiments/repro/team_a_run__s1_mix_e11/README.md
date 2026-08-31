# Logged run — S1-MIX-E11

## Catalogue metadata

- **Catalogue ID:** `team_a_run__s1_mix_e11`
- **Namespace:** `team_a_run`
- **Experiment ID:** `S1-MIX-E11`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** logged run / arm
- **Model:** blend
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** params:** {"level": 2.3293, "selection": "\u043f\u0435\u0440\u0435\u0431\u043e\u0440 \u043f\u043e wCV, \u0448\u0430\u0433 0.05, 10626 \u043a\u043e\u043c\u0431\u0438\u043d\u0430\u0446\u0438\u0439", "weights": {"S1-DIST": 0.3, "S1-E02": 0.35, "S1-E03a": 0.0, "S1-E10": 0.1, "S1-E11": 0.25}}
- **Known score:** params:** {"level": 2.3293, "selection": "\u043f\u0435\u0440\u0435\u0431\u043e\u0440 \u043f\u043e wCV, \u0448\u0430\u0433 0.05, 10626 \u043a\u043e\u043c\u0431\u0438\u043d\u0430\u0446\u0438\u0439", "weights": {"S1-DIST": 0.3, "S1-E02": 0.35, "S1-E03a": 0.0, "S1-E10": 0.1, "S1-E11": 0.25}}
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** description:** Смесь 0.10 S1-E10 + 0.35 S1-E02 + 0.30 S1-DIST + 0.25 S1-E11 (S1-E03a обнулён), --level 2.3293
- **Submission:** conclusion:** Последний сабмит проекта, отправлен вопреки вердикту exp_016 §7. wCV 1.74911 против 1.74948 у S1-DIST-MIX (-0.00038; LOFO-честно -0.00036), лучше на 4 фолдах из 4. LB 1.6510029 против 1.6507774 — НА +0.00023 ХУЖЕ. Промах прогноза +0.00060 = 2.4 парных SE, прогноз отвергается; сама дельта LB t=+0.9, то есть кандидат не «хуже», а «не лучше». Коэффициент переноса -0.60 (к S1-DIST-MIX) и 0.288 (к S1-BEST) против ~1.0 на крупных дельтах. Первая инверсия порядка в проекте: Spearman(wCV,
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — S1-MIX-E11

This run was recovered from `experiments/log.csv`.

- **exp_id:** S1-MIX-E11
- **timestamp:** 2026-08-11T23:55:49
- **commit:** e3032ca
- **description:** Смесь 0.10 S1-E10 + 0.35 S1-E02 + 0.30 S1-DIST + 0.25 S1-E11 (S1-E03a обнулён), --level 2.3293
- **scenario:** S1
- **n_features:** Unknown / not recoverable from repository history
- **model:** blend
- **params:** {"level": 2.3293, "selection": "\u043f\u0435\u0440\u0435\u0431\u043e\u0440 \u043f\u043e wCV, \u0448\u0430\u0433 0.05, 10626 \u043a\u043e\u043c\u0431\u0438\u043d\u0430\u0446\u0438\u0439", "weights": {"S1-DIST": 0.3, "S1-E02": 0.35, "S1-E03a": 0.0, "S1-E10": 0.1, "S1-E11": 0.25}}
- **cutoffs:** Unknown / not recoverable from repository history
- **L:** Unknown / not recoverable from repository history
- **panel_blocks:** 3
- **fold_scores:** [1.76913, 1.76241, 1.75032, 1.74268]
- **cv_mean:** 1.75770
- **cv_std:** Unknown / not recoverable from repository history
- **bias_mean:** -0.06754
- **best_offset:** Unknown / not recoverable from repository history
- **cv_mean_calib:** 1.75613
- **delta_vs_b0:** Unknown / not recoverable from repository history
- **runtime_s:** Unknown / not recoverable from repository history
- **verdict:** REJECT
- **conclusion:** Последний сабмит проекта, отправлен вопреки вердикту exp_016 §7. wCV 1.74911 против 1.74948 у S1-DIST-MIX (-0.00038; LOFO-честно -0.00036), лучше на 4 фолдах из 4. LB 1.6510029 против 1.6507774 — НА +0.00023 ХУЖЕ. Промах прогноза +0.00060 = 2.4 парных SE, прогноз отвергается; сама дельта LB t=+0.9, то есть кандидат не «хуже», а «не лучше». Коэффициент переноса -0.60 (к S1-DIST-MIX) и 0.288 (к S1-BEST) против ~1.0 на крупных дельтах. Первая инверсия порядка в проекте: Spearman(wCV, LB) упал с 1.00 до 0.90. Механизм: обнуление S1-E03a снимает страховку против 120-дневной экстраполяции, её ценность (~0.0005) лежит в неизмеримой оси exp_016 §5, а прирост от переподбора весов — в измеримой. Порог -0.0020 подтверждён. Детали exp_016 §8
- **wcv:** 1.74911
- **fold_cal:** [1.76913, 1.76241, 1.75032, 1.74268]
- **mean_z:** 2.69317
- **lb_public:** 1.6510029
