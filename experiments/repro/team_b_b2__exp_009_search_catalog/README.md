# exp_009 — G8 search/catalog split features

## Catalogue metadata

- **Catalogue ID:** `team_b_b2__exp_009_search_catalog`
- **Namespace:** `team_b_b2`
- **Experiment ID:** `exp_009_search_catalog`
- **Original source:** `git:88dc69163b1f:experiments/exp_009_search_catalog.md`
- **Source ref:** `88dc69163b1f39aaac55ddfbfc9986e2203cfbdf`
- **Source commit:** `30a276391f860890b6236058e0cfc76b9b503d83`
- **Kind:** git-history experiment card
- **Model:** LightGBM
- **Features:** Search/Catalog decomposition
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** CV по фолдам: fold1 `1.690353`, fold2 `1.743251`, fold3 `1.732910`
- **Known score:** CV mean: `1.716802` (лучший на момент: exp_001, `1.717017`)
- **Seed:** LightGBM baseline, параметры из `src/train.py`, folds 1–2 для решения, fold3 справочно, seed из `src/config.py`.
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL from the frozen commit when the card contains a command; PARTIAL when the card explicitly says the experimental code was rolled back
- **Notes:** Frozen implementation is copied from the commit that introduced this card.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_009 — G8 search/catalog split features

- **Дата:** 2026-08-13
- **Автор:** Codex
- **Коммит:** не коммитили (reject; код откатан)

## Гипотеза

Разделить поисковый и каталожный сигнал по окнам 30/90/180/all: `gmv_search`, `gmv_cat`, долю search GMV, дни search/cat и конверсию `search_to_ord/searches`.

## Что изменено относительно базы

Добавлялись только новые `sc_*` признаки в конец `FEATURES` через `build_features`; после reject код возвращён к базе без `sc_*`.

## Результат

- CV по фолдам: fold1 `1.690353`, fold2 `1.743251`, fold3 `1.732910`
- CV mean: `1.716802` (лучший на момент: exp_001, `1.717017`)
- LB: не отправляли

## Вердикт и вывод

Reject: улучшение mean `0.000215`, ниже порога. Сигнал есть на обоих основных фолдах, но слишком маленький для принятия отдельной гипотезы.

## Конфиг прогона

LightGBM baseline, параметры из `src/train.py`, folds 1–2 для решения, fold3 справочно, seed из `src/config.py`.
