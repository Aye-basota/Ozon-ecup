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
