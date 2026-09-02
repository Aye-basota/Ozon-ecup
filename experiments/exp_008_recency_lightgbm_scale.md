# exp_008 — recency lightgbm scale

- **Дата:** 2026-08-11
- **Автор:** B1
- **Коммит:** 9d0e5b7

## Гипотеза

Хорошие изолированные идеи можно объединить для сабмита: recency-фичи дают сигнал свежести действий, LightGBM лучше аппроксимирует табличные зависимости, а scale-калибровка оптимизирует RMSLE.

## Что изменено относительно базы

Champion stack: feature set `recency`, модель `LightGBM`, post-scale `0.64`.

## Результат

- CV по фолдам: validation cutoff 2026-01-15
- CV mean: RMSE 254.269612, MAE 65.909651, RMSLE 1.671639
- LB: 1.657

## Вердикт и вывод

Успех: лучший текущий CV, заметно лучше baseline 1.711195 и exp_002 1.710919. Сабмит собран в `submissions/exp_008_recency_lightgbm_scale064.csv`.

## Конфиг прогона

Модель: LightGBM из exp_006. Feature set: `recency`. Scale grid для кандидатов показал лучший вариант `recency + lightgbm + 0.64`. Train cutoffs: 2025-10-01, 2025-11-01, 2025-12-01. Validation cutoff: 2026-01-15. Test cutoff: 2026-02-14. Seed: 42 из `config.py`.
