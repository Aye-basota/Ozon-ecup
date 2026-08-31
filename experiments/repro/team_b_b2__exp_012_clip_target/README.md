# exp_012 — G11 clipped train target p999

## Catalogue metadata

- **Catalogue ID:** `team_b_b2__exp_012_clip_target`
- **Namespace:** `team_b_b2`
- **Experiment ID:** `exp_012_clip_target`
- **Original source:** `git:88dc69163b1f:experiments/exp_012_clip_target.md`
- **Source ref:** `88dc69163b1f39aaac55ddfbfc9986e2203cfbdf`
- **Source commit:** `30a276391f860890b6236058e0cfc76b9b503d83`
- **Kind:** git-history experiment card
- **Model:** LightGBM
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Учить модель на таргете, клипнутом по 99.9 перцентилю train-фолда, чтобы уменьшить влияние китов. Предсказания при этом не клипуются сверх обычного `clip(min=0)`.
- **Known score:** CV mean: `1.716910` (лучший на момент: exp_001, `1.717017`)
- **Seed:** LightGBM baseline, target clip по train q0.999, folds 1–2 для решения, fold3 справочно, seed из `src/config.py`.
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL from the frozen commit when the card contains a command; PARTIAL when the card explicitly says the experimental code was rolled back
- **Notes:** Frozen implementation is copied from the commit that introduced this card.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_012 — G11 clipped train target p999

- **Дата:** 2026-08-13
- **Автор:** Codex
- **Коммит:** не коммитили (reject; код откатан)

## Гипотеза

Учить модель на таргете, клипнутом по 99.9 перцентилю train-фолда, чтобы уменьшить влияние китов. Предсказания при этом не клипуются сверх обычного `clip(min=0)`.

## Что изменено относительно базы

Менялся только таргет внутри `fit_lgbm`: перед `log1p` применялся `clip(upper=q0.999)`. После reject код возвращён к обучению на полном `y`.

## Результат

- CV по фолдам: fold1 `1.690289`, fold2 `1.743531`, fold3 `1.732735`
- CV mean: `1.716910` (лучший на момент: exp_001, `1.717017`)
- LB: не отправляли

## Вердикт и вывод

Reject: улучшение mean `0.000107`, ниже порога, fold2 немного хуже. Clipping не меняет основную структуру ошибки.

## Конфиг прогона

LightGBM baseline, target clip по train q0.999, folds 1–2 для решения, fold3 справочно, seed из `src/config.py`.
