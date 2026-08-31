# exp_010 — ensemble tuning

## Catalogue metadata

- **Catalogue ID:** `team_b_strategy__exp_010_ensemble_tuning`
- **Namespace:** `team_b_strategy`
- **Experiment ID:** `exp_010_ensemble_tuning`
- **Original source:** `git:824f41575bc2:experiments/exp_010_ensemble_tuning.md`
- **Source ref:** `824f41575bc2fa4ae11b8f6f9dfd907571276d37`
- **Source commit:** `68b543d7b654257812ba4bbb58547fd8a832fb00`
- **Kind:** git-history experiment card
- **Model:** LightGBM, ensemble
- **Features:** recency
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** CV по фолдам: validation cutoff 2026-01-15
- **Known score:** После LB 1.656853 на exp_009 стоит не менять модель резко, а подобрать близкие веса ансамбля и общий scale. Локальный CV вокруг optimum плоский, поэтому небольшое смещение может улучшить public LB.
- **Seed:** Модели: два `LGBMRegressor` из exp_009. Train cutoffs: 2025-10-01, 2025-11-01, 2025-12-01. Validation cutoff: 2026-01-15. Seed: 42 из `config.py`.
- **Postprocessing:** Используется тот же ансамбль exp_009: `recency LightGBM scale 0.64` + `long_buy LightGBM scale 0.62` в log-space. Проверены веса `w_recency` 0.30..0.70 и global scale 0.94..1.06.
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL from the frozen commit when the card contains a command; PARTIAL when the card explicitly says the experimental code was rolled back
- **Notes:** Frozen implementation is copied from the commit that introduced this card.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_010 — ensemble tuning

- **Дата:** 2026-08-12
- **Автор:** Codex
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
