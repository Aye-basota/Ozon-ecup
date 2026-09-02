# exp_010 — ensemble tuning

- **Дата:** 2026-08-12
- **Автор:** B1
- **Коммит:** 9d0e5b7

## Гипотеза

После LB 1.656853 на exp_009 стоит не менять модель резко, а подобрать близкие веса ансамбля и общий scale. Локальный CV вокруг optimum плоский, поэтому небольшое смещение может улучшить public LB.

## Что изменено относительно базы

Используется тот же ансамбль exp_009: `recency LightGBM scale 0.64` + `long_buy LightGBM scale 0.62` в log-space. Проверены веса `w_recency` 0.30..0.70 и global scale 0.94..1.06.

## Результат

- CV по фолдам: validation cutoff 2026-01-15
- CV mean: лучший локально остаётся около `w_recency=0.50, global_scale=1.00`, RMSLE 1.670716; `w_recency=0.55, global_scale=1.00` даёт RMSLE 1.670717; `w_recency=0.55, global_scale=1.01` даёт RMSLE 1.670731
- LB: не отправляли

## Вердикт и вывод

Нейтрально: локально текущий exp_009 почти оптимален, но варианты 0.55/1.00 и 0.55/1.01 достаточно близки, чтобы проверить на LB. Файлы: `submissions/exp_010_logens_wrec055_scale100.csv`, `submissions/exp_010_logens_wrec055_scale101.csv`.

## Конфиг прогона

Модели: два `LGBMRegressor` из exp_009. Train cutoffs: 2025-10-01, 2025-11-01, 2025-12-01. Validation cutoff: 2026-01-15. Seed: 42 из `config.py`.
