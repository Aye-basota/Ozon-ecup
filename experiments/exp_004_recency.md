# exp_004 — recency

- **Дата:** 2026-08-11
- **Автор:** Codex
- **Коммит:** 9d0e5b7

## Гипотеза

Для LTV важна не только суммарная активность, но и свежесть конкретных действий. Пользователь, который недавно покупал или добавлял в корзину, может иметь другой будущий GMV, чем пользователь с такой же исторической суммой, но давно не активный.

## Что изменено относительно базы

К baseline-агрегатам добавлены recency-фичи: дни с последнего `search`, `cat`, `to_cart`, `to_ord`, `gmv`, `search_to_cart`, `cat_to_cart`.

## Результат

- CV по фолдам: validation cutoff 2026-01-15
- CV mean: RMSE 244.749083, MAE 65.600258, RMSLE 1.710617
- LB: не отправляли

## Вердикт и вывод

Успех: RMSLE улучшился с baseline 1.711195 до 1.710617. Recency-фичи полезнее conversions на текущей HGBR-модели.

## Конфиг прогона

Модель: `HistGradientBoostingRegressor(loss="squared_error", learning_rate=0.05, max_leaf_nodes=31, max_iter=250, l2_regularization=0.05)`. Feature set: `recency`. Train cutoffs: 2025-10-01, 2025-11-01, 2025-12-01. Validation cutoff: 2026-01-15. Seed: 42 из `config.py`.
