# exp_003 — G2 calendar cutoff features

## Catalogue metadata

- **Catalogue ID:** `team_b_b2__exp_003_calendar`
- **Namespace:** `team_b_b2`
- **Experiment ID:** `exp_003_calendar`
- **Original source:** `git:88dc69163b1f:experiments/exp_003_calendar.md`
- **Source ref:** `88dc69163b1f39aaac55ddfbfc9986e2203cfbdf`
- **Source commit:** `30a276391f860890b6236058e0cfc76b9b503d83`
- **Kind:** git-history experiment card
- **Model:** LightGBM
- **Features:** holiday/YoY features, calendar features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** CV по фолдам: fold1 `1.690608`, fold2 `1.743426`, fold3 `1.732506`
- **Known score:** CV mean: `1.717017` (лучший на момент: exp_001, `1.717017`)
- **Seed:** LightGBM baseline, параметры из `src/train.py`, folds 1–2 для решения, fold3 справочно, seed из `src/config.py`.
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL from the frozen commit when the card contains a command; PARTIAL when the card explicitly says the experimental code was rolled back
- **Notes:** Frozen implementation is copied from the commit that introduced this card.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_003 — G2 calendar cutoff features

- **Дата:** 2026-08-13
- **Автор:** Codex
- **Коммит:** не коммитили (reject; код откатан)

## Гипотеза

Добавить календарные признаки от cutoff: день года, день недели старта таргет-окна, расстояние до 25 декабря и флаг январского post-holiday окна. Идея — дать модели явный сезонный контекст.

## Что изменено относительно базы

Добавлялись только новые `cal_*` колонки в конец `FEATURES` через `build_features`; после reject код возвращён к базе без `cal_*`.

## Результат

- CV по фолдам: fold1 `1.690608`, fold2 `1.743426`, fold3 `1.732506`
- CV mean: `1.717017` (лучший на момент: exp_001, `1.717017`)
- LB: не отправляли

## Вердикт и вывод

Reject: улучшения нет. В текущей single-cutoff схеме календарные признаки константны внутри train-фолда, поэтому LightGBM не извлекает из них сигнал; можно вернуться к ним только после multi-cutoff обучения.

## Конфиг прогона

LightGBM baseline, параметры из `src/train.py`, folds 1–2 для решения, fold3 справочно, seed из `src/config.py`.
