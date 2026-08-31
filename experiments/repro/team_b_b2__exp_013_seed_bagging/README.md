# exp_013 — G12 seed bagging

## Catalogue metadata

- **Catalogue ID:** `team_b_b2__exp_013_seed_bagging`
- **Namespace:** `team_b_b2`
- **Experiment ID:** `exp_013_seed_bagging`
- **Original source:** `git:88dc69163b1f:experiments/exp_013_seed_bagging.md`
- **Source ref:** `88dc69163b1f39aaac55ddfbfc9986e2203cfbdf`
- **Source commit:** `30a276391f860890b6236058e0cfc76b9b503d83`
- **Kind:** git-history experiment card
- **Model:** LightGBM
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** CV по фолдам: fold1 `1.689997`, fold2 `1.742736`, fold3 `1.731774`
- **Known score:** CV mean: `1.716366` (лучший на момент: exp_001, `1.717017`)
- **Seed:** LightGBM baseline, 5 сидов, log-average, folds 1–2 для решения, fold3 справочно, seed из `src/config.py` плюс соседние seed для bagging.
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL from the frozen commit when the card contains a command; PARTIAL when the card explicitly says the experimental code was rolled back
- **Notes:** Frozen implementation is copied from the commit that introduced this card.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_013 — G12 seed bagging

- **Дата:** 2026-08-13
- **Автор:** Codex
- **Коммит:** не коммитили (reject; код откатан)

## Гипотеза

Усреднить 5 LightGBM-моделей с разными seed в log-пространстве: среднее `log1p`-предсказаний, затем `expm1`. Ожидание из PLAN.md: уменьшение variance без изменения признаков.

## Что изменено относительно базы

Менялась только схема предсказания в CV: 5 сидов `[42, 43, 44, 45, 46]`, усреднение в log-пространстве. После reject код возвращён к одиночной seed-42 модели.

## Результат

- CV по фолдам: fold1 `1.689997`, fold2 `1.742736`, fold3 `1.731774`
- CV mean: `1.716366` (лучший на момент: exp_001, `1.717017`)
- LB: не отправляли

## Вердикт и вывод

Reject по строгому правилу: улучшение mean `0.000651`, ниже порога `0.005`, хотя все три фолда стали лучше. Это полезная финальная техника для ансамбля, но отдельной accepted-итерацией её принимать нельзя.

## Конфиг прогона

LightGBM baseline, 5 сидов, log-average, folds 1–2 для решения, fold3 справочно, seed из `src/config.py` плюс соседние seed для bagging.
