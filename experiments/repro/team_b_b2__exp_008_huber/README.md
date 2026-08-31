# exp_008 — G7 huber log target loss

## Catalogue metadata

- **Catalogue ID:** `team_b_b2__exp_008_huber`
- **Namespace:** `team_b_b2`
- **Experiment ID:** `exp_008_huber`
- **Original source:** `git:88dc69163b1f:experiments/exp_008_huber.md`
- **Source ref:** `88dc69163b1f39aaac55ddfbfc9986e2203cfbdf`
- **Source commit:** `30a276391f860890b6236058e0cfc76b9b503d83`
- **Kind:** git-history experiment card
- **Model:** LightGBM
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** CV по фолдам: fold1 `1.703652`, fold2 `1.761199`, fold3 `1.758791`
- **Known score:** CV mean: `1.732425` (лучший на момент: exp_001, `1.717017`)
- **Seed:** LightGBM Huber, `alpha=0.9`, folds 1–2 для решения, fold3 справочно, seed из `src/config.py`.
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL from the frozen commit when the card contains a command; PARTIAL when the card explicitly says the experimental code was rolled back
- **Notes:** Frozen implementation is copied from the commit that introduced this card.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_008 — G7 huber log target loss

- **Дата:** 2026-08-13
- **Автор:** Codex
- **Коммит:** не коммитили (reject; код откатан)

## Гипотеза

Заменить обычную regression objective по `log1p(y)` на Huber objective по тому же `log1p(y)`. Ожидание: устойчивость к крупным покупателям может улучшить общий RMSLE.

## Что изменено относительно базы

Менялся только LightGBM objective: `huber`, `alpha=0.9`; таргет `log1p(y)` и прогноз `expm1` сохранялись. После reject код возвращён к `objective="regression"`.

## Результат

- CV по фолдам: fold1 `1.703652`, fold2 `1.761199`, fold3 `1.758791`
- CV mean: `1.732425` (лучший на момент: exp_001, `1.717017`)
- LB: не отправляли

## Вердикт и вывод

Reject: оба основных фолда хуже baseline. Huber немного снижает общий bias на fold1, но ухудшает RMSLE и сегменты покупателей.

## Конфиг прогона

LightGBM Huber, `alpha=0.9`, folds 1–2 для решения, fold3 справочно, seed из `src/config.py`.
