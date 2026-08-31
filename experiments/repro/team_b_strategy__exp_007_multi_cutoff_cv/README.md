# exp_007 — multi-cutoff CV

## Catalogue metadata

- **Catalogue ID:** `team_b_strategy__exp_007_multi_cutoff_cv`
- **Namespace:** `team_b_strategy`
- **Experiment ID:** `exp_007_multi_cutoff_cv`
- **Original source:** `git:824f41575bc2:experiments/exp_007_multi_cutoff_cv.md`
- **Source ref:** `824f41575bc2fa4ae11b8f6f9dfd907571276d37`
- **Source commit:** `68b543d7b654257812ba4bbb58547fd8a832fb00`
- **Kind:** git-history experiment card
- **Model:** Unknown / not recoverable from repository history
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Один validation cutoff может быть случайно оптимистичным или пессимистичным. Multi-cutoff CV нужен не для сабмита напрямую, а чтобы оценивать устойчивость идей во времени.
- **Known score:** CV по фолдам: 2025-10-01 RMSLE 1.724390; 2025-11-01 RMSLE 1.726790; 2025-12-01 RMSLE 1.754997; 2026-01-15 RMSLE 1.711195
- **Seed:** Модель: HGBR baseline. Feature set: `baseline`. Фолды: train 2025-07/08/09 → val 2025-10-01; train 2025-08/09/10 → val 2025-11-01; train 2025-09/10/11 → val 2025-12-01; train 2025-10/11/12 → val 2026-01-15. Seed: 42 из `config.py`.
- **Postprocessing:** None documented
- **Submission:** Один validation cutoff может быть случайно оптимистичным или пессимистичным. Multi-cutoff CV нужен не для сабмита напрямую, а чтобы оценивать устойчивость идей во времени.
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL from the frozen commit when the card contains a command; PARTIAL when the card explicitly says the experimental code was rolled back
- **Notes:** Frozen implementation is copied from the commit that introduced this card.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_007 — multi-cutoff CV

- **Дата:** 2026-08-11
- **Автор:** Codex
- **Коммит:** 9d0e5b7

## Гипотеза

Один validation cutoff может быть случайно оптимистичным или пессимистичным. Multi-cutoff CV нужен не для сабмита напрямую, а чтобы оценивать устойчивость идей во времени.

## Что изменено относительно базы

Baseline HGBR оценён на четырёх out-of-time фолдах вместо одного validation cutoff.

## Результат

- CV по фолдам: 2025-10-01 RMSLE 1.724390; 2025-11-01 RMSLE 1.726790; 2025-12-01 RMSLE 1.754997; 2026-01-15 RMSLE 1.711195
- CV mean: RMSE 268.652144, MAE 75.880914, RMSLE 1.729343; RMSLE std 0.015957
- LB: не отправляли

## Вердикт и вывод

Нейтрально: это validation-аудит, а не улучшающая фича. Последний single-cutoff выглядит чуть оптимистичнее среднего по времени, поэтому будущие крупные идеи стоит проверять хотя бы на нескольких срезах.

## Конфиг прогона

Модель: HGBR baseline. Feature set: `baseline`. Фолды: train 2025-07/08/09 → val 2025-10-01; train 2025-08/09/10 → val 2025-11-01; train 2025-09/10/11 → val 2025-12-01; train 2025-10/11/12 → val 2026-01-15. Seed: 42 из `config.py`.
