# exp_005 — G4 multi-cutoff training 8x14d

## Catalogue metadata

- **Catalogue ID:** `team_b_b2__exp_005_multicutoff`
- **Namespace:** `team_b_b2`
- **Experiment ID:** `exp_005_multicutoff`
- **Original source:** `git:88dc69163b1f:experiments/exp_005_multicutoff.md`
- **Source ref:** `88dc69163b1f39aaac55ddfbfc9986e2203cfbdf`
- **Source commit:** `30a276391f860890b6236058e0cfc76b9b503d83`
- **Kind:** git-history experiment card
- **Model:** LightGBM
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Учить LightGBM не на одном cutoff, а на сетке из 8 cutoff'ов с шагом 14 дней, объединённых в один train. Последний обучающий cutoff равен прежнему train cutoff фолда, поэтому target последнего train окна заканчивается не позже val cutoff.
- **Known score:** CV mean: `1.717040` (лучший на момент: exp_001, `1.717017`)
- **Seed:** LightGBM baseline, параметры из `src/train.py`, 8 cutoff'ов с шагом 14 дней, folds 1–2 для решения, fold3 справочно, seed из `src/config.py`.
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL from the frozen commit when the card contains a command; PARTIAL when the card explicitly says the experimental code was rolled back
- **Notes:** Frozen implementation is copied from the commit that introduced this card.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_005 — G4 multi-cutoff training 8x14d

- **Дата:** 2026-08-13
- **Автор:** Codex
- **Коммит:** не коммитили (reject; код откатан)

## Гипотеза

Учить LightGBM не на одном cutoff, а на сетке из 8 cutoff'ов с шагом 14 дней, объединённых в один train. Последний обучающий cutoff равен прежнему train cutoff фолда, поэтому target последнего train окна заканчивается не позже val cutoff.

## Что изменено относительно базы

Менялась только схема обучения: `X_train/y_train` собирались конкатенацией 8 cached датасетов; признаки и параметры модели не менялись. После reject код возвращён к single-cutoff обучению.

## Результат

- CV по фолдам: fold1 `1.691941`, fold2 `1.742140`, fold3 `1.733246`
- CV mean: `1.717040` (лучший на момент: exp_001, `1.717017`)
- LB: не отправляли

## Вердикт и вывод

Reject: mean стал на `0.000023` хуже baseline. Multi-cutoff снизил bias на `y>0`, но усилил перепрогноз нулей (`y=0` вырос до 52.4% SLE), поэтому баланс метрики не улучшился.

## Конфиг прогона

LightGBM baseline, параметры из `src/train.py`, 8 cutoff'ов с шагом 14 дней, folds 1–2 для решения, fold3 справочно, seed из `src/config.py`.
