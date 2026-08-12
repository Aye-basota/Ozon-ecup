# exp_002 — conversions

- **Дата:** 2026-08-11
- **Автор:** Codex
- **Коммит:** 9d0e5b7

## Гипотеза

Пользователи с одинаковым объёмом активности могут сильно отличаться по качеству этой активности. Конверсии search→cart, search→order, cart→order, средний чек и доли GMV должны помочь модели лучше отличать простое browsing-поведение от покупательского.

## Что изменено относительно базы

К baseline-агрегатам добавлены ratio-фичи по всем данным до cutoff и окнам 7/14/30/60/120 дней.

## Результат

- CV по фолдам: validation cutoff 2026-01-15
- CV mean: RMSE 243.974514, MAE 65.601427, RMSLE 1.710919
- LB: не отправляли

## Вердикт и вывод

Успех: RMSLE улучшился с 1.711195 до 1.710919. Улучшение маленькое, но направление полезное; ratio-фичи стоит оставить как новый baseline.

## Конфиг прогона

Модель: `HistGradientBoostingRegressor(loss="squared_error", learning_rate=0.05, max_leaf_nodes=31, max_iter=250, l2_regularization=0.05)`. Train cutoffs: 2025-10-01, 2025-11-01, 2025-12-01. Validation cutoff: 2026-01-15. Test cutoff для сабмита: 2026-02-14. Seed: 42 из `config.py`. Сабмит: `submissions/exp_002_conversions_hgbr.csv`.
