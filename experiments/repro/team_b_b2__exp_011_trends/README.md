# exp_011 — G10 simple trend ratios

## Catalogue metadata

- **Catalogue ID:** `team_b_b2__exp_011_trends`
- **Namespace:** `team_b_b2`
- **Experiment ID:** `exp_011_trends`
- **Original source:** `git:88dc69163b1f:experiments/exp_011_trends.md`
- **Source ref:** `88dc69163b1f39aaac55ddfbfc9986e2203cfbdf`
- **Source commit:** `30a276391f860890b6236058e0cfc76b9b503d83`
- **Kind:** git-history experiment card
- **Model:** LightGBM
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** CV по фолдам: fold1 `1.690822`, fold2 `1.743369`, fold3 `1.732566`
- **Known score:** CV mean: `1.717095` (лучший на момент: exp_001, `1.717017`)
- **Seed:** LightGBM baseline, параметры из `src/train.py`, folds 1–2 для решения, fold3 справочно, seed из `src/config.py`.
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL from the frozen commit when the card contains a command; PARTIAL when the card explicitly says the experimental code was rolled back
- **Notes:** Frozen implementation is copied from the commit that introduced this card.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_011 — G10 simple trend ratios

- **Дата:** 2026-08-13
- **Автор:** Codex
- **Коммит:** не коммитили (reject; код откатан)

## Гипотеза

Добавить простые ratio-тренды между свежим и предыдущим окном: `w30_gmv/(w60-w30+1)`, `w14_searches/(w30_searches+1)`, аналоги для orders и days_buy.

## Что изменено относительно базы

Добавлялись только новые `trend_*` признаки в конец `FEATURES` через `build_features`; после reject код возвращён к базе.

## Результат

- CV по фолдам: fold1 `1.690822`, fold2 `1.743369`, fold3 `1.732566`
- CV mean: `1.717095` (лучший на момент: exp_001, `1.717017`)
- LB: не отправляли

## Вердикт и вывод

Reject: mean хуже baseline на `0.000078`. Fold2 слегка улучшился, но fold1 деградировал; структура ошибки не даёт причины принять признак.

## Конфиг прогона

LightGBM baseline, параметры из `src/train.py`, folds 1–2 для решения, fold3 справочно, seed из `src/config.py`.
