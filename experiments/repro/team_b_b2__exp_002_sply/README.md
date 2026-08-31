# exp_002 — G1 SPLY

## Catalogue metadata

- **Catalogue ID:** `team_b_b2__exp_002_sply`
- **Namespace:** `team_b_b2`
- **Experiment ID:** `exp_002_sply`
- **Original source:** `git:88dc69163b1f:experiments/exp_002_sply.md`
- **Source ref:** `88dc69163b1f39aaac55ddfbfc9986e2203cfbdf`
- **Source commit:** `30a276391f860890b6236058e0cfc76b9b503d83`
- **Kind:** git-history experiment card
- **Model:** LightGBM
- **Features:** calendar features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** CV по фолдам: fold1 `1.690596`, fold2 `1.743426`, fold3 `1.732506`
- **Known score:** CV mean: `1.717011` (лучший на момент: exp_001, `1.717017`)
- **Seed:** LightGBM baseline, параметры из `src/train.py`, folds 1–2 для решения, fold3 справочно, seed из `src/config.py`.
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL from the frozen commit when the card contains a command; PARTIAL when the card explicitly says the experimental code was rolled back
- **Notes:** Frozen implementation is copied from the commit that introduced this card.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_002 — G1 SPLY

- **Дата:** 2026-08-13
- **Автор:** Codex
- **Коммит:** не коммитили (reject; код откатан)

## Гипотеза

Добавить признаки same-period last year: суммы `gmv`, заказов, дней покупок, присутствия и поисков по окну `[cutoff-365, cutoff-365+30)`, плюс календарные пики вокруг 23 февраля и 8 марта. Ожидание из PLAN.md: сезонные сигналы должны уменьшить ошибку на fold1/fold2.

## Что изменено относительно базы

Добавлялись только новые `sply_*` колонки в конец `FEATURES` через `build_features`; после reject код возвращён к базе без `sply_*`.

## Результат

- CV по фолдам: fold1 `1.690596`, fold2 `1.743426`, fold3 `1.732506`
- CV mean: `1.717011` (лучший на момент: exp_001, `1.717017`)
- LB: не отправляли

## Вердикт и вывод

Reject: улучшение mean всего `0.000006`, ниже порогов PLAN.md. Error-analysis почти не изменился: `y=0` даёт 51.6% SLE, `y>0` 48.4%; значит локальная схема почти не извлекает пользу из этих SPLY-окон.

## Конфиг прогона

LightGBM baseline, параметры из `src/train.py`, folds 1–2 для решения, fold3 справочно, seed из `src/config.py`.
