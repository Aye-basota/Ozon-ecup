# exp_025 — backlog funnel conversion features

## Catalogue metadata

- **Catalogue ID:** `team_b_b2__exp_025_funnel`
- **Namespace:** `team_b_b2`
- **Experiment ID:** `exp_025_funnel`
- **Original source:** `git:88dc69163b1f:experiments/exp_025_funnel.md`
- **Source ref:** `88dc69163b1f39aaac55ddfbfc9986e2203cfbdf`
- **Source commit:** `427abcc36b2e94a1c58345458f90df36c122f2a4`
- **Kind:** git-history experiment card
- **Model:** LightGBM
- **Features:** funnel features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** CV по фолдам: fold1 `1.675062`, fold2 `1.742066`, fold3 `1.745565`
- **Known score:** CV mean: `1.708564` (лучший на момент: exp_015, `1.708737`)
- **Seed:** LightGBM exp_015, `CALIBRATION_DELTA=-0.17`, новые `funnel_*` признаки, folds 1–2 для решения, fold3 справочно, seed из `src/config.py`.
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: card metrics survive; runnable source uses a documented later branch-head fallback
- **Notes:** Frozen implementation is copied from the commit that introduced this card. The card-introduction commit contained syntactically incomplete WIP source; the frozen implementation uses the nearest surviving branch-head version for: src/train.py. Exact provenance is in implementation/SOURCE_PROVENANCE.json.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_025 — backlog funnel conversion features

- **Дата:** 2026-08-13
- **Автор:** Codex
- **Коммит:** не коммитили (reject; код откатан)

## Гипотеза

Добавить признаки воронки search→cart→order по окнам 30/90/all: conversion rates, cart→order rate, дни с successful search-to-cart/search-to-order.

## Что изменено относительно базы

Добавлялись только новые `funnel_*` признаки в конец `FEATURES` через `build_features`; после reject код возвращён к exp_015.

## Результат

- CV по фолдам: fold1 `1.675062`, fold2 `1.742066`, fold3 `1.745565`
- CV mean: `1.708564` (лучший на момент: exp_015, `1.708737`)
- LB: не отправляли

## Вердикт и вывод

Reject: улучшение mean `0.000173`, ниже порога. Фичи немного помогают fold1, но не дают достаточного сдвига к целевой зоне.

## Конфиг прогона

LightGBM exp_015, `CALIBRATION_DELTA=-0.17`, новые `funnel_*` признаки, folds 1–2 для решения, fold3 справочно, seed из `src/config.py`.
