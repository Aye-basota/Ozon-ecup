# exp_006 — lightgbm

## Catalogue metadata

- **Catalogue ID:** `team_b_strategy__exp_006_lightgbm`
- **Namespace:** `team_b_strategy`
- **Experiment ID:** `exp_006_lightgbm`
- **Original source:** `git:824f41575bc2:experiments/exp_006_lightgbm.md`
- **Source ref:** `824f41575bc2fa4ae11b8f6f9dfd907571276d37`
- **Source commit:** `68b543d7b654257812ba4bbb58547fd8a832fb00`
- **Kind:** git-history experiment card
- **Model:** LightGBM
- **Features:** window aggregates
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** CV по фолдам: validation cutoff 2026-01-15
- **Known score:** CV mean: RMSE 243.895440, MAE 65.516312, RMSLE 1.710143
- **Seed:** Модель: `LGBMRegressor(objective="regression", n_estimators=600, learning_rate=0.03, num_leaves=31, subsample=0.8, colsample_bytree=0.8, reg_lambda=0.05)`. Feature set: `baseline`. Train cutoffs: 2025-10-01, 2025-11-01, 2025-12-01. Validation cutoff: 2026-01-15. Seed: 42 из `config.py`.
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL from the frozen commit when the card contains a command; PARTIAL when the card explicitly says the experimental code was rolled back
- **Notes:** Frozen implementation is copied from the commit that introduced this card.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_006 — lightgbm

- **Дата:** 2026-08-11
- **Автор:** Codex
- **Коммит:** 9d0e5b7

## Гипотеза

LightGBM часто сильнее sklearn HGBR на табличных агрегатах и быстрее обучается на больших данных. Проверяем замену модели без изменения baseline-фичей.

## Что изменено относительно базы

Модель заменена с `HistGradientBoostingRegressor` на `LGBMRegressor`; feature set остался `baseline`.

## Результат

- CV по фолдам: validation cutoff 2026-01-15
- CV mean: RMSE 243.895440, MAE 65.516312, RMSLE 1.710143
- LB: не отправляли

## Вердикт и вывод

Успех: RMSLE улучшился с baseline 1.711195 до 1.710143. LightGBM стоит использовать в champion stack.

## Конфиг прогона

Модель: `LGBMRegressor(objective="regression", n_estimators=600, learning_rate=0.03, num_leaves=31, subsample=0.8, colsample_bytree=0.8, reg_lambda=0.05)`. Feature set: `baseline`. Train cutoffs: 2025-10-01, 2025-11-01, 2025-12-01. Validation cutoff: 2026-01-15. Seed: 42 из `config.py`.
