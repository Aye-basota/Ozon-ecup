# exp_014 — G13 second LGBM log blend

## Catalogue metadata

- **Catalogue ID:** `team_b_b2__exp_014_second_lgbm_blend`
- **Namespace:** `team_b_b2`
- **Experiment ID:** `exp_014_second_lgbm_blend`
- **Original source:** `git:88dc69163b1f:experiments/exp_014_second_lgbm_blend.md`
- **Source ref:** `88dc69163b1f39aaac55ddfbfc9986e2203cfbdf`
- **Source commit:** `30a276391f860890b6236058e0cfc76b9b503d83`
- **Kind:** git-history experiment card
- **Model:** LightGBM, blend
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** CV по фолдам: fold1 `1.690181`, fold2 `1.742846`, fold3 `1.731949`
- **Known score:** CV mean: `1.716513` (лучший на момент: exp_001, `1.717017`)
- **Seed:** Baseline LightGBM + second LightGBM (`num_leaves=31`, `feature_fraction=0.6`), log-space blend, folds 1–2 для решения, fold3 справочно, seed из `src/config.py`.
- **Postprocessing:** Baseline LightGBM + second LightGBM (`num_leaves=31`, `feature_fraction=0.6`), log-space blend, folds 1–2 для решения, fold3 справочно, seed из `src/config.py`.
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL from the frozen commit when the card contains a command; PARTIAL when the card explicitly says the experimental code was rolled back
- **Notes:** Frozen implementation is copied from the commit that introduced this card.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_014 — G13 second LGBM log blend

- **Дата:** 2026-08-13
- **Автор:** Codex
- **Коммит:** не коммитили (reject; код откатан)

## Гипотеза

Добавить вторую LightGBM-модель с `num_leaves=31` и `feature_fraction=0.6`, затем подобрать blend в log-пространстве сеткой 0.1. Разная ёмкость дерева может давать комплементарные ошибки.

## Что изменено относительно базы

Менялась только схема модели в CV: baseline LGBM + вторая LGBM, blend в log-пространстве. После reject код возвращён к одной baseline-модели.

## Результат

- CV по фолдам: fold1 `1.690181`, fold2 `1.742846`, fold3 `1.731949`
- CV mean: `1.716513` (лучший на момент: exp_001, `1.717017`)
- LB: не отправляли

## Вердикт и вывод

Reject по строгому правилу: улучшение mean `0.000504`, ниже порога. Вес `0.5` оказался лучшим на всех трёх фолдах, но отдельным accepted-изменением это недостаточно.

## Конфиг прогона

Baseline LightGBM + second LightGBM (`num_leaves=31`, `feature_fraction=0.6`), log-space blend, folds 1–2 для решения, fold3 справочно, seed из `src/config.py`.
