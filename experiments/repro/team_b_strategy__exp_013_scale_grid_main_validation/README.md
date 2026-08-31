# exp_013 — scale grid on main validation

## Catalogue metadata

- **Catalogue ID:** `team_b_strategy__exp_013_scale_grid_main_validation`
- **Namespace:** `team_b_strategy`
- **Experiment ID:** `exp_013_scale_grid_main_validation`
- **Original source:** `git:824f41575bc2:experiments/exp_013_scale_grid_main_validation.md`
- **Source ref:** `824f41575bc2fa4ae11b8f6f9dfd907571276d37`
- **Source commit:** `18c3bdd2a8a04de8ed0650a749659448c98ece5d`
- **Kind:** git-history experiment card
- **Model:** ensemble
- **Features:** recency
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** # exp_013 — scale grid on main validation
- **Known score:** После LB exp_011 лучший scale был `1.2`, но validation из `main` показала, что минимум может быть чуть выше. Проверяем плотную сетку `1.10..1.35` на двух single-cutoff фолдах из `main`.
- **Seed:** Модель: exp_011 log-space ensemble `recency + long_buy`, component scales `0.64/0.62`, `recency_weight=0.5`, dense8 train cutoffs для submit `2025-08-28..2025-10-16`, seed из `config.py`.
- **Postprocessing:** Модель: exp_011 log-space ensemble `recency + long_buy`, component scales `0.64/0.62`, `recency_weight=0.5`, dense8 train cutoffs для submit `2025-08-28..2025-10-16`, seed из `config.py`.
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL from the frozen commit when the card contains a command; PARTIAL when the card explicitly says the experimental code was rolled back
- **Notes:** Frozen implementation is copied from the commit that introduced this card.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_013 — scale grid on main validation

- **Дата:** 2026-08-13
- **Автор:** Codex
- **Коммит:** 0366bd9

## Гипотеза

После LB exp_011 лучший scale был `1.2`, но validation из `main` показала, что минимум может быть чуть выше. Проверяем плотную сетку `1.10..1.35` на двух single-cutoff фолдах из `main`.

## Что изменено относительно базы

Код модели не менялся; для exp_011 dense8 log-ensemble прогнаны global scale `1.10, 1.125, ..., 1.35`.

## Результат

- CV по фолдам: `2025-12-15 -> 2026-01-14`, `2025-11-15 -> 2025-12-15`.
- CV mean top:
  - scale `1.300`: mean `1.707984`, fold scores `1.674495`, `1.741473`
  - scale `1.275`: mean `1.708010`, fold scores `1.673925`, `1.742095`
  - scale `1.325`: mean `1.708098`, fold scores `1.675198`, `1.740999`
  - scale `1.250`: mean `1.708184`
  - scale `1.200`: mean `1.709007`
- LB:
  - scale `1.275`: `1.6557083162306878`
  - scale `1.300`: `1.656279619331512`
  - scale `1.325`: не отправляли из-за лимита
  - baseline для сравнения: exp_011 scale `1.200` — `1.6549097093483665`

## Вердикт и вывод

Провал относительно текущего LB-чемпиона. Validation из `main` полезна для грубого выбора scale, но в мелкой сетке переоценила движение выше `1.20`: на LB чем ближе к `1.20`, тем лучше. Текущий лучший остаётся exp_011 `scale=1.20`.

## Конфиг прогона

Модель: exp_011 log-space ensemble `recency + long_buy`, component scales `0.64/0.62`, `recency_weight=0.5`, dense8 train cutoffs для submit `2025-08-28..2025-10-16`, seed из `config.py`.
