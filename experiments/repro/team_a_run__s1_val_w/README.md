# Logged run — S1-VAL-W

## Catalogue metadata

- **Catalogue ID:** `team_a_run__s1_val_w`
- **Namespace:** `team_a_run`
- **Experiment ID:** `S1-VAL-W`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** logged run / arm
- **Model:** calibration diagnostic
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** description:** exp_016: калибровка валидатора по LB — единая схема wCV (веса фолдов 1:2:4:8, калиброванные пофолдовые RMSLE)
- **Known score:** conclusion:** Не модель, а схема валидации. По 5 сабмитам с известным LB: Spearman = 1.0 у ВСЕХ локальных метрик (порядок ничего не различает), различает устойчивость коэффициента переноса dLB/dлокальная. wCV 1:2:4:8: перенос 1.000 +- 0.057 против 1.007 +- 0.151 у равных весов и 0.999 +- 0.240 у OOF cal. Асимметрия переноса 0.64x из STATE.md оказалась артефактом OOF cal: на wCV тот же замер даёт 0.975. Парный шум public LB = 0.00025, а не 0.0058 (непарный) — прежний порог 0.01 завышен в 20 раз.
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** conclusion:** Не модель, а схема валидации. По 5 сабмитам с известным LB: Spearman = 1.0 у ВСЕХ локальных метрик (порядок ничего не различает), различает устойчивость коэффициента переноса dLB/dлокальная. wCV 1:2:4:8: перенос 1.000 +- 0.057 против 1.007 +- 0.151 у равных весов и 0.999 +- 0.240 у OOF cal. Асимметрия переноса 0.64x из STATE.md оказалась артефактом OOF cal: на wCV тот же замер даёт 0.975. Парный шум public LB = 0.00025, а не 0.0058 (непарный) — прежний порог 0.01 завышен в 20 раз.
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — S1-VAL-W

This run was recovered from `experiments/log.csv`.

- **exp_id:** S1-VAL-W
- **timestamp:** 2026-08-11T23:30:00
- **commit:** e3032ca
- **description:** exp_016: калибровка валидатора по LB — единая схема wCV (веса фолдов 1:2:4:8, калиброванные пофолдовые RMSLE)
- **scenario:** S1
- **n_features:** Unknown / not recoverable from repository history
- **model:** none
- **params:** {"fold_weights": [1, 2, 4, 8], "calibrated": true, "folds": "VAL_FOLDS_S1"}
- **cutoffs:** Unknown / not recoverable from repository history
- **L:** Unknown / not recoverable from repository history
- **panel_blocks:** 3
- **fold_scores:** Unknown / not recoverable from repository history
- **cv_mean:** Unknown / not recoverable from repository history
- **cv_std:** Unknown / not recoverable from repository history
- **bias_mean:** Unknown / not recoverable from repository history
- **best_offset:** Unknown / not recoverable from repository history
- **cv_mean_calib:** Unknown / not recoverable from repository history
- **delta_vs_b0:** Unknown / not recoverable from repository history
- **runtime_s:** Unknown / not recoverable from repository history
- **verdict:** ПРИНЯТО
- **conclusion:** Не модель, а схема валидации. По 5 сабмитам с известным LB: Spearman = 1.0 у ВСЕХ локальных метрик (порядок ничего не различает), различает устойчивость коэффициента переноса dLB/dлокальная. wCV 1:2:4:8: перенос 1.000 +- 0.057 против 1.007 +- 0.151 у равных весов и 0.999 +- 0.240 у OOF cal. Асимметрия переноса 0.64x из STATE.md оказалась артефактом OOF cal: на wCV тот же замер даёт 0.975. Парный шум public LB = 0.00025, а не 0.0058 (непарный) — прежний порог 0.01 завышен в 20 раз. Реализация: src/report.py, src/validation.py (calibrate, wcv), FOLD_WEIGHTS_S1; 10 проверок в src/test_validation.py; пересчёт python -m src.cv_lb
- **wcv:** Unknown / not recoverable from repository history
- **fold_cal:** Unknown / not recoverable from repository history
- **mean_z:** Unknown / not recoverable from repository history
- **lb_public:** Unknown / not recoverable from repository history
