# exp_006 — G6 hurdle

## Catalogue metadata

- **Catalogue ID:** `team_b_b2__exp_006_hurdle`
- **Namespace:** `team_b_b2`
- **Experiment ID:** `exp_006_hurdle`
- **Original source:** `git:88dc69163b1f:experiments/exp_006_hurdle.md`
- **Source ref:** `88dc69163b1f39aaac55ddfbfc9986e2203cfbdf`
- **Source commit:** `30a276391f860890b6236058e0cfc76b9b503d83`
- **Kind:** git-history experiment card
- **Model:** LightGBM, two-part / hurdle
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** CV по фолдам: fold1 `2.055990`, fold2 `2.035534`, fold3 `1.965917`
- **Known score:** CV mean: `2.045762` (лучший на момент: exp_001, `1.717017`)
- **Seed:** LightGBM classifier + LightGBM regressor, параметры из `src/train.py`, folds 1–2 для решения, fold3 справочно, seed из `src/config.py`.
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL from the frozen commit when the card contains a command; PARTIAL when the card explicitly says the experimental code was rolled back
- **Notes:** Frozen implementation is copied from the commit that introduced this card.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_006 — G6 hurdle

- **Дата:** 2026-08-13
- **Автор:** Codex
- **Коммит:** не коммитили (reject; код откатан)

## Гипотеза

Построить двухстадийную модель: classifier для `P(y>0)` на всех пользователях и regressor по `log1p(y)` только на пользователях с положительным таргетом. Итоговый прогноз: `P(y>0) * expm1(reg)`.

## Что изменено относительно базы

Менялась только модельная схема в `src/train.py`; признаки и параметры базового регрессора не менялись. После reject код возвращён к одиночной LightGBM-регрессии.

## Результат

- CV по фолдам: fold1 `2.055990`, fold2 `2.035534`, fold3 `1.965917`
- CV mean: `2.045762` (лучший на момент: exp_001, `1.717017`)
- LB: не отправляли

## Вердикт и вывод

Reject: резкая деградация на всех фолдах. Error-analysis показывает понятную причину: модель стала намного сильнее перепрогнозировать нулевых пользователей (`y=0` 81.8% SLE), хотя ошибка на `y>0` снизилась.

## Конфиг прогона

LightGBM classifier + LightGBM regressor, параметры из `src/train.py`, folds 1–2 для решения, fold3 справочно, seed из `src/config.py`.
