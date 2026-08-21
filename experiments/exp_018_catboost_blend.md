# exp_018 — CatBoost blend поверх dist-head

- **Дата:** 2026-08-16
- **Автор:** Codex
- **Коммит:** pending

## Гипотеза

CatBoost может дать ошибку, отличающуюся от LightGBM-компонентов. Пробуем добавить его как третий небольшой голос поверх текущего exp_017: recency LightGBM + post-order dist-head.

## Что изменено относительно базы

Добавлен CPU CatBoostRegressor на `long_buy_post_order` и log-space blend с exp_017-компонентами.

## Результат

- CV по фолдам: fold1 `1.672166 -> 1.671958`, fold2 `1.744423 -> 1.743885` при `cat_weight=0.20`
- CV mean: `1.707921` (лучший на момент: exp_017, `1.708295`)
- LB: не отправляли

## Вердикт и вывод

Успех локально: оба main-fold улучшаются монотонно при росте веса CatBoost до `0.20`. Улучшение маленькое, но стабильное; подготовлен submit-кандидат.

## Конфиг прогона

Команда CV: `python src/catboost_blend.py cv`. Команда submit: `python src/catboost_blend.py submit --cat-weight 0.20 --output exp_018_catboost_blend_wcat020_scale120.csv`. Веса: recency `0.50`, dist-head `0.30`, CatBoost `0.20`; global scale `1.20`; component scales `0.64/0.62/0.62`; CatBoost CPU `iterations=800`, `depth=6`, `learning_rate=0.05`, `l2_leaf_reg=3.0`, seed из `config.py`.
