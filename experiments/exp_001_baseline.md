# exp_001 — baseline

- **Дата:** 2026-08-11
- **Автор:** Codex
- **Коммит:** 9d0e5b7

## Гипотеза

Простые агрегаты пользовательской активности до cutoff уже должны дать рабочую точку отсчёта. Для baseline используем суммы, средние, максимумы и оконные агрегаты за 7/14/30/60/120 дней, а target считаем как будущий `gmv` за 30 дней.

## Что изменено относительно базы

Реализованы `build_features(cutoff_date)` и первый train pipeline на `HistGradientBoostingRegressor` с `log1p` таргетом.

## Результат

- CV по фолдам: validation cutoff 2026-01-15
- CV mean: RMSE 244.596747, MAE 65.626454, RMSLE 1.711195
- LB: не отправляли

## Вердикт и вывод

Нейтрально: baseline успешно обучается и даёт воспроизводимую отправную точку. После уточнения условий соревнования главным ориентиром считаем RMSLE.

## Конфиг прогона

Модель: `HistGradientBoostingRegressor(loss="squared_error", learning_rate=0.05, max_leaf_nodes=31, max_iter=250, l2_regularization=0.05)`. Train cutoffs: 2025-10-01, 2025-11-01, 2025-12-01. Validation cutoff: 2026-01-15. Test cutoff для сабмита: 2026-02-14. Seed: 42 из `config.py`.
