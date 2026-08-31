# exp_015 — G14 fixed calibration log shift

## Catalogue metadata

- **Catalogue ID:** `team_b_b2__exp_015_calibration_shift`
- **Namespace:** `team_b_b2`
- **Experiment ID:** `exp_015_calibration_shift`
- **Original source:** `git:88dc69163b1f:experiments/exp_015_calibration_shift.md`
- **Source ref:** `88dc69163b1f39aaac55ddfbfc9986e2203cfbdf`
- **Source commit:** `30a276391f860890b6236058e0cfc76b9b503d83`
- **Kind:** git-history experiment card
- **Model:** LightGBM, calibration diagnostic
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** В `predict_gmv` добавлен фиксированный `CALIBRATION_DELTA = -0.17`; признаки, параметры модели и validation cutoffs не менялись.
- **Known score:** CV mean: `1.708737` (лучший до этого: exp_001, `1.717017`)
- **Seed:** LightGBM baseline, `CALIBRATION_DELTA = -0.17`, folds 1–2 для решения, fold3 справочно, seed из `src/config.py`.
- **Postprocessing:** None documented
- **Submission:** Локально accept по fold1–2: улучшение mean `0.008279`, оба основных фолда лучше baseline. После LB-пробы submit-калибровка rejected: local +`0.008` → LB −`0.009`. Причина: нетрансферабельность глобального `δ` через сезонность. Новое правило: глобальные калибровочные константы подбираются per-fold; в сабмит константа идёт только если её знак и порядок подтверждены на фолде, имитирующем конфигурацию сабмита (сейчас fold2), иначе сабмит без калибровки.
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL from the frozen commit when the card contains a command; PARTIAL when the card explicitly says the experimental code was rolled back
- **Notes:** Frozen implementation is copied from the commit that introduced this card.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_015 — G14 fixed calibration log shift

- **Дата:** 2026-08-13
- **Автор:** Codex
- **Коммит:** 30a2763

## Гипотеза

Подобрать константный сдвиг `delta` в log-пространстве: `pred = expm1(raw_log_pred + delta)`. Baseline fold1 имеет общий отрицательный bias, поэтому отрицательный сдвиг должен уменьшить перепрогноз нулевых пользователей.

## Что изменено относительно базы

В `predict_gmv` добавлен фиксированный `CALIBRATION_DELTA = -0.17`; признаки, параметры модели и validation cutoffs не менялись.

## Результат

- CV по фолдам: fold1 `1.675509`, fold2 `1.741966`, fold3 `1.745607`
- CV mean: `1.708737` (лучший до этого: exp_001, `1.717017`)
- LB: `1.6700`, хуже baseline exp_001 без сдвига (`1.6615`) примерно на `0.009`

## Вердикт и вывод

Локально accept по fold1–2: улучшение mean `0.008279`, оба основных фолда лучше baseline. После LB-пробы submit-калибровка rejected: local +`0.008` → LB −`0.009`. Причина: нетрансферабельность глобального `δ` через сезонность. Новое правило: глобальные калибровочные константы подбираются per-fold; в сабмит константа идёт только если её знак и порядок подтверждены на фолде, имитирующем конфигурацию сабмита (сейчас fold2), иначе сабмит без калибровки.

## Конфиг прогона

LightGBM baseline, `CALIBRATION_DELTA = -0.17`, folds 1–2 для решения, fold3 справочно, seed из `src/config.py`.
