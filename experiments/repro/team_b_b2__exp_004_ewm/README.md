# exp_004 — G3 EWM aggregates

## Catalogue metadata

- **Catalogue ID:** `team_b_b2__exp_004_ewm`
- **Namespace:** `team_b_b2`
- **Experiment ID:** `exp_004_ewm`
- **Original source:** `git:88dc69163b1f:experiments/exp_004_ewm.md`
- **Source ref:** `88dc69163b1f39aaac55ddfbfc9986e2203cfbdf`
- **Source commit:** `30a276391f860890b6236058e0cfc76b9b503d83`
- **Kind:** git-history experiment card
- **Model:** LightGBM
- **Features:** EWM aggregates
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** CV по фолдам: fold1 `1.689885`, fold2 `1.743566`, fold3 `1.732749`
- **Known score:** CV mean: `1.716725` (лучший на момент: exp_001, `1.717017`)
- **Seed:** LightGBM baseline, параметры из `src/train.py`, folds 1–2 для решения, fold3 справочно, seed из `src/config.py`.
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL from the frozen commit when the card contains a command; PARTIAL when the card explicitly says the experimental code was rolled back
- **Notes:** Frozen implementation is copied from the commit that introduced this card.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_004 — G3 EWM aggregates

- **Дата:** 2026-08-13
- **Автор:** Codex
- **Коммит:** не коммитили (reject; код откатан)

## Гипотеза

Добавить экспоненциально-взвешенные суммы с halflife 7 и 14 дней: `ewm7_gmv`, `ewm14_gmv`, `ewm7_orders`. Свежие действия пользователя должны быть информативнее старых плоских окон.

## Что изменено относительно базы

Добавлялись только новые `ewm_*` колонки в конец `FEATURES` через `build_features`; после reject код возвращён к базе без `ewm_*`.

## Результат

- CV по фолдам: fold1 `1.689885`, fold2 `1.743566`, fold3 `1.732749`
- CV mean: `1.716725` (лучший на момент: exp_001, `1.717017`)
- LB: не отправляли

## Вердикт и вывод

Reject: mean улучшился только на `0.000292`, ниже порога. Fold1 стал немного лучше, fold2 немного хуже; структура ошибки почти не изменилась.

## Конфиг прогона

LightGBM baseline, параметры из `src/train.py`, folds 1–2 для решения, fold3 справочно, seed из `src/config.py`.
