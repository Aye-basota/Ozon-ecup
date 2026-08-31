# exp_010 — G9 AOV and intensity features

## Catalogue metadata

- **Catalogue ID:** `team_b_b2__exp_010_aov_intensity`
- **Namespace:** `team_b_b2`
- **Experiment ID:** `exp_010_aov_intensity`
- **Original source:** `git:88dc69163b1f:experiments/exp_010_aov_intensity.md`
- **Source ref:** `88dc69163b1f39aaac55ddfbfc9986e2203cfbdf`
- **Source commit:** `30a276391f860890b6236058e0cfc76b9b503d83`
- **Kind:** git-history experiment card
- **Model:** LightGBM
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** CV по фолдам: fold1 `1.690339`, fold2 `1.743583`, fold3 `1.732744`
- **Known score:** CV mean: `1.716961` (лучший на момент: exp_001, `1.717017`)
- **Seed:** LightGBM baseline, параметры из `src/train.py`, folds 1–2 для решения, fold3 справочно, seed из `src/config.py`.
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL from the frozen commit when the card contains a command; PARTIAL when the card explicitly says the experimental code was rolled back
- **Notes:** Frozen implementation is copied from the commit that introduced this card.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_010 — G9 AOV and intensity features

- **Дата:** 2026-08-13
- **Автор:** Codex
- **Коммит:** не коммитили (reject; код откатан)

## Гипотеза

Добавить AOV и интенсивность покупок: `aov_30`, `aov_90`, `aov_all`, `gmv_per_search_all`, `orders_per_active_day_30`. Эти признаки могут помочь различать покупателей с похожей частотой, но разным средним чеком.

## Что изменено относительно базы

Добавлялись только новые ratio-признаки в конец `FEATURES` через `build_features`; после reject код возвращён к базе.

## Результат

- CV по фолдам: fold1 `1.690339`, fold2 `1.743583`, fold3 `1.732744`
- CV mean: `1.716961` (лучший на момент: exp_001, `1.717017`)
- LB: не отправляли

## Вердикт и вывод

Reject: улучшение mean всего `0.000056`, ниже порога, при небольшом ухудшении fold2. Error-analysis почти не отличается от baseline.

## Конфиг прогона

LightGBM baseline, параметры из `src/train.py`, folds 1–2 для решения, fold3 справочно, seed из `src/config.py`.
