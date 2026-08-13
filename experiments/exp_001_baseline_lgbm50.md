# exp_001 — baseline LightGBM 50 features

- **Дата:** 2026-08-12
- **Автор:** Codex
- **Коммит:** a6807ea

## Гипотеза

User-based агрегаты по прошлой активности должны дать честный первый baseline для прогноза GMV за следующие 30 дней. LightGBM на `log1p(target)` должен быть лучше простого правила `predict = GMV за последние 30 дней`.

## Что изменено относительно базы

Реализован baseline-пайплайн: 50 признаков из `PLAN.md`, 2-fold out-of-time validation, LightGBM и 3 сабмита.

## Результат

- CV по фолдам: fold1 LGBM `1.690608`, fold2 LGBM `1.743426`.
- Baselines: fold1 naive30 `2.196829`, naive90 `2.094162`; fold2 naive30 `2.216295`, naive90 `2.063102`.
- CV mean: `1.717017` (лучший на момент: `exp_001`, `1.717017`).
- LB: не отправляли.

## Вердикт и вывод

Нейтрально/accept как стартовый baseline: LGBM заметно лучше naive30 на обоих фолдах. Sanity-диапазон naive30 из `PLAN.md` не совпал с фактическим (`2.196829` на fold1), продолжение выполнено после ручной проверки.

## Конфиг прогона

LightGBM: `objective=regression`, `learning_rate=0.05`, `num_leaves=63`, `feature_fraction=0.8`, `bagging_fraction=0.8`, `bagging_freq=1`, `min_data_in_leaf=100`, `lambda_l2=1.0`, `seed=42`. Cutoff: fold1 train `2025-12-15`, val `2026-01-14`; fold2 train `2025-11-15`, val `2025-12-15`; target window 30 дней; train target = `log1p(y)`, prediction = `expm1`, clip below zero.
