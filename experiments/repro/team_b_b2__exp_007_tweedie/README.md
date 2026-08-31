# exp_007 — G7 tweedie raw target loss

## Catalogue metadata

- **Catalogue ID:** `team_b_b2__exp_007_tweedie`
- **Namespace:** `team_b_b2`
- **Experiment ID:** `exp_007_tweedie`
- **Original source:** `git:88dc69163b1f:experiments/exp_007_tweedie.md`
- **Source ref:** `88dc69163b1f39aaac55ddfbfc9986e2203cfbdf`
- **Source commit:** `30a276391f860890b6236058e0cfc76b9b503d83`
- **Kind:** git-history experiment card
- **Model:** LightGBM
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** CV по фолдам: fold1 `2.394555`, fold2 `2.388086`, fold3 `2.331039`
- **Known score:** CV mean: `2.391320` (лучший на момент: exp_001, `1.717017`)
- **Seed:** LightGBM Tweedie, `tweedie_variance_power=1.5`, folds 1–2 для решения, fold3 справочно, seed из `src/config.py`.
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL from the frozen commit when the card contains a command; PARTIAL when the card explicitly says the experimental code was rolled back
- **Notes:** Frozen implementation is copied from the commit that introduced this card.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_007 — G7 tweedie raw target loss

- **Дата:** 2026-08-13
- **Автор:** Codex
- **Коммит:** не коммитили (reject; код откатан)

## Гипотеза

Заменить RMSE по `log1p(y)` на Tweedie loss по сырому `y` с `variance_power=1.5`. Tweedie может лучше учитывать нули и длинный хвост GMV.

## Что изменено относительно базы

Менялся только loss и таргетная трансформация: LightGBM учился на сыром `y`, прогноз использовался напрямую без `expm1`. После reject код возвращён к `log1p(y)` и `expm1`.

## Результат

- CV по фолдам: fold1 `2.394555`, fold2 `2.388086`, fold3 `2.331039`
- CV mean: `2.391320` (лучший на момент: exp_001, `1.717017`)
- LB: не отправляли

## Вердикт и вывод

Reject: Tweedie резко хуже baseline. Error-analysis: нулевые пользователи дают 85.1% SLE, то есть сырой loss слишком сильно перепрогнозирует нули для RMSLE-задачи.

## Конфиг прогона

LightGBM Tweedie, `tweedie_variance_power=1.5`, folds 1–2 для решения, fold3 справочно, seed из `src/config.py`.
