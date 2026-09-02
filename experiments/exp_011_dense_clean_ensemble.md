# exp_011 — dense clean-cutoff ensemble

- **Дата:** 2026-08-13
- **Автор:** B1
- **Коммит:** 68b543d

## Гипотеза

Берём практику Strategy 1: не учиться только на трёх месячных cutoff, а сделать плотную weekly-нарезку на более раннем чистом периоде. Это должно дать модели больше out-of-time примеров и меньше зависеть от странного последнего месяца с просадкой покупок.

## Что изменено относительно базы

Добавлен отдельный скрипт `src/dense_ensemble.py`: weekly train-cutoff grid, clean validation cutoff, log-space ансамбль recency + long_buy, multi-scale submit за один fit.

## Результат

- CV по фолдам: clean val `2025-10-16`, train cutoffs `2025-07-24..2025-09-11` для CV.
- CV mean: `1.733432` при `global_scale=1.0`; лучший из сетки `1.712473` при `global_scale=1.4`. Сравнивать напрямую с exp_009 нельзя: это другой validation regime.
- Компоненты при `global_scale=1.4`: recency-only `1.714832`, long_buy-only `1.713702`, смесь `1.712473`.
- LB: scale `1.0` — `1.661242797071839`; scale `1.2` — `1.6549097093483665`; scale `1.4` — `1.6598855449125254`.

## Вердикт и вывод

Успех по LB: лучший вариант `scale=1.2` дал `1.6549097093483665` и стал новым лучшим сабмитом. Clean validation из этого эксперимента переоценивала `scale=1.4`, поэтому для дальнейшего scale-тюнинга лучше использовать validation-схему из `main` — см. exp_012.

## Конфиг прогона

LightGBM `n_estimators=600`, `learning_rate=0.03`, `num_leaves=31`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=0.05`, seed из `config.py`. CV: `python src/dense_ensemble.py cv --folds 1 --recent-train-cutoffs 8 --components both --scale-grid 0.9,1.0,1.1,1.2,1.3,1.4`. Submit: `python src/dense_ensemble.py submit --recent-train-cutoffs 8 --components both --scale-grid 1.0,1.2,1.4 --output exp_011_dense8_logens.csv`.
