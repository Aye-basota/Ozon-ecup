# exp_012 — LB validation alignment

## Catalogue metadata

- **Catalogue ID:** `team_b_strategy__exp_012_lb_validation_alignment`
- **Namespace:** `team_b_strategy`
- **Experiment ID:** `exp_012_lb_validation_alignment`
- **Original source:** `git:824f41575bc2:experiments/exp_012_lb_validation_alignment.md`
- **Source ref:** `824f41575bc2fa4ae11b8f6f9dfd907571276d37`
- **Source commit:** `18c3bdd2a8a04de8ed0650a749659448c98ece5d`
- **Kind:** git-history experiment card
- **Model:** LightGBM, ensemble
- **Features:** recency
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** # exp_012 — LB validation alignment
- **Known score:** Успех: validation правильно выбрала `scale=1.2`, как и LB. Абсолютные значения CV выше LB примерно на `0.05`, но локальное ранжирование scale-кандидатов совпало, поэтому для подбора scale лучше использовать эту схему, а не dense clean-cutoff CV из exp_011.
- **Seed:** Модель: exp_011 log-space ensemble `recency + long_buy`, component scales `0.64/0.62`, `recency_weight=0.5`, LightGBM с seed из `config.py`. Для каждого fold обучение на одном `train_cutoff` из `main:src/validation.py`, прогноз на соответствующий `val_cutoff`, затем global scale `1.0/1.2/1.4`.
- **Postprocessing:** Модель: exp_011 log-space ensemble `recency + long_buy`, component scales `0.64/0.62`, `recency_weight=0.5`, LightGBM с seed из `config.py`. Для каждого fold обучение на одном `train_cutoff` из `main:src/validation.py`, прогноз на соответствующий `val_cutoff`, затем global scale `1.0/1.2/1.4`.
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL from the frozen commit when the card contains a command; PARTIAL when the card explicitly says the experimental code was rolled back
- **Notes:** Frozen implementation is copied from the commit that introduced this card.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_012 — LB validation alignment

- **Дата:** 2026-08-13
- **Автор:** Codex
- **Коммит:** 0366bd9

## Гипотеза

Проверяем validation-схему из `main`, предложенную коллегой: два single-cutoff фолда `2025-12-15 -> 2026-01-14` и `2025-11-15 -> 2025-12-15`. Если схема близка к LB, она должна правильно ранжировать три scale-варианта exp_011.

## Что изменено относительно базы

Код модели не менялся; прогнаны exp_011 dense ensemble scale `1.0/1.2/1.4` на двух фолдах из `main:src/validation.py`.

## Результат

- CV по фолдам:
  - scale `1.0`: `1.678854`, `1.761295`; mean `1.720075`
  - scale `1.2`: `1.673083`, `1.744931`; mean `1.709007`
  - scale `1.4`: `1.678030`, `1.740401`; mean `1.709215`
- LB:
  - scale `1.0`: `1.661242797071839`
  - scale `1.2`: `1.6549097093483665`
  - scale `1.4`: `1.6598855449125254`

## Вердикт и вывод

Успех: validation правильно выбрала `scale=1.2`, как и LB. Абсолютные значения CV выше LB примерно на `0.05`, но локальное ранжирование scale-кандидатов совпало, поэтому для подбора scale лучше использовать эту схему, а не dense clean-cutoff CV из exp_011.

## Конфиг прогона

Модель: exp_011 log-space ensemble `recency + long_buy`, component scales `0.64/0.62`, `recency_weight=0.5`, LightGBM с seed из `config.py`. Для каждого fold обучение на одном `train_cutoff` из `main:src/validation.py`, прогноз на соответствующий `val_cutoff`, затем global scale `1.0/1.2/1.4`.
