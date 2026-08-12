# exp_009 — long buy ensemble

- **Дата:** 2026-08-12
- **Автор:** Codex
- **Коммит:** 9d0e5b7

## Гипотеза

Importance-файлы S1-BEST показывают, что самые сильные признаки — длинная история покупок: `w180_days_buy`, `w180_orders`, `w365_days_buy`, `w365_orders`, `w90_days_buy`. Добавим похожие long-history buy-фичи и проверим, дают ли они полезную ошибку для ансамбля с текущим champion.

## Что изменено относительно базы

Добавлен feature set `long_buy`: recency + окна 90/180/365 для покупочных дней, заказов, GMV, log-GMV mean/std, tenure, first_buy_age и buy-rate/aov признаки. Сабмит строится как log-space ансамбль `recency LightGBM scale 0.64` и `long_buy LightGBM scale 0.62` с весами 0.5/0.5.

## Результат

- CV по фолдам: validation cutoff 2026-01-15
- CV mean: `long_buy + LightGBM + scale 0.62` RMSLE 1.671832; log-ensemble 50/50 RMSLE 1.670716
- LB: 1.6568530995317488

## Вердикт и вывод

Успех локально: long_buy сам по себе чуть хуже exp_008, но в log-space ансамбле улучшает CV с 1.671639 до 1.670716. Файл для отправки: `submissions/exp_009_recency_long_buy_lgbm_logens.csv`.

## Конфиг прогона

Модели: два `LGBMRegressor` из exp_006. Train cutoffs: 2025-10-01, 2025-11-01, 2025-12-01. Validation cutoff: 2026-01-15. Test cutoff: 2026-02-14. Seed: 42 из `config.py`. Веса ансамбля: 0.5 recency, 0.5 long_buy; смешивание в `log1p(pred)` пространстве.
