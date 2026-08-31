# exp_005 — scale calibration

## Catalogue metadata

- **Catalogue ID:** `team_b_strategy__exp_005_scale_calibration`
- **Namespace:** `team_b_strategy`
- **Experiment ID:** `exp_005_scale_calibration`
- **Original source:** `git:824f41575bc2:experiments/exp_005_scale_calibration.md`
- **Source ref:** `824f41575bc2fa4ae11b8f6f9dfd907571276d37`
- **Source commit:** `68b543d7b654257812ba4bbb58547fd8a832fb00`
- **Kind:** git-history experiment card
- **Model:** calibration diagnostic
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** CV по фолдам: validation cutoff 2026-01-15
- **Known score:** CV mean: лучший scale 0.65, RMSLE 1.672748
- **Seed:** Модель: HGBR baseline. Feature set: `baseline`. Проверенные scale: 0.45–0.90 и грубая сетка 0.7–2.0. Train cutoffs: 2025-10-01, 2025-11-01, 2025-12-01. Validation cutoff: 2026-01-15. Seed: 42 из `config.py`.
- **Postprocessing:** Так как метрика RMSLE штрафует относительные ошибки в log-space, модель с `log1p`-таргетом может выигрывать от простой пост-калибровки масштаба. Проверяем множители прогноза на чистом baseline.
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL from the frozen commit when the card contains a command; PARTIAL when the card explicitly says the experimental code was rolled back
- **Notes:** Frozen implementation is copied from the commit that introduced this card.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_005 — scale calibration

- **Дата:** 2026-08-11
- **Автор:** Codex
- **Коммит:** 9d0e5b7

## Гипотеза

Так как метрика RMSLE штрафует относительные ошибки в log-space, модель с `log1p`-таргетом может выигрывать от простой пост-калибровки масштаба. Проверяем множители прогноза на чистом baseline.

## Что изменено относительно базы

После baseline HGBR-прогноза применён множитель `predict * scale`; модель и фичи не менялись.

## Результат

- CV по фолдам: validation cutoff 2026-01-15
- CV mean: лучший scale 0.65, RMSLE 1.672748
- LB: не отправляли

## Вердикт и вывод

Сильный успех: RMSLE улучшился с 1.711195 до 1.672748. Для текущего validation модель лучше слегка занижать, чем оставлять сырые `expm1`-прогнозы.

## Конфиг прогона

Модель: HGBR baseline. Feature set: `baseline`. Проверенные scale: 0.45–0.90 и грубая сетка 0.7–2.0. Train cutoffs: 2025-10-01, 2025-11-01, 2025-12-01. Validation cutoff: 2026-01-15. Seed: 42 из `config.py`.
