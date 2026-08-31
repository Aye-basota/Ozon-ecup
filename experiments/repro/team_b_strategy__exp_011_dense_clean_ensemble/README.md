# exp_011 — dense clean-cutoff ensemble

## Catalogue metadata

- **Catalogue ID:** `team_b_strategy__exp_011_dense_clean_ensemble`
- **Namespace:** `team_b_strategy`
- **Experiment ID:** `exp_011_dense_clean_ensemble`
- **Original source:** `git:824f41575bc2:experiments/exp_011_dense_clean_ensemble.md`
- **Source ref:** `824f41575bc2fa4ae11b8f6f9dfd907571276d37`
- **Source commit:** `0366bd9b40b8ca784dbe3e2ccdb744dcc22f5b4a`
- **Kind:** git-history experiment card
- **Model:** LightGBM, ensemble
- **Features:** recency
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Добавлен отдельный скрипт `src/dense_ensemble.py`: weekly train-cutoff grid, clean validation cutoff, log-space ансамбль recency + long_buy, multi-scale submit за один fit.
- **Known score:** CV mean: `1.733432` при `global_scale=1.0`; лучший из сетки `1.712473` при `global_scale=1.4`. Сравнивать напрямую с exp_009 нельзя: это другой validation regime.
- **Seed:** LightGBM `n_estimators=600`, `learning_rate=0.03`, `num_leaves=31`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=0.05`, seed из `config.py`. CV: `python src/dense_ensemble.py cv --folds 1 --recent-train-cutoffs 8 --components both --scale-grid 0.9,1.0,1.1,1.2,1.3,1.4`. Submit: `python src/dense_ensemble.py submit --recent-train-cutoffs 8 --components both --scale-grid 1.0,1.2,1.4 --output exp_011_dense8_logens.csv`.
- **Postprocessing:** Добавлен отдельный скрипт `src/dense_ensemble.py`: weekly train-cutoff grid, clean validation cutoff, log-space ансамбль recency + long_buy, multi-scale submit за один fit.
- **Submission:** Успех по LB: лучший вариант `scale=1.2` дал `1.6549097093483665` и стал новым лучшим сабмитом. Clean validation из этого эксперимента переоценивала `scale=1.4`, поэтому для дальнейшего scale-тюнинга лучше использовать validation-схему из `main` — см. exp_012.
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL from the frozen commit when the card contains a command; PARTIAL when the card explicitly says the experimental code was rolled back
- **Notes:** Frozen implementation is copied from the commit that introduced this card.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_011 — dense clean-cutoff ensemble

- **Дата:** 2026-08-13
- **Автор:** Codex
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
