# exp_022 — G6 classifier probability feature boosting

## Catalogue metadata

- **Catalogue ID:** `team_b_b2__exp_022_classifier_feature`
- **Namespace:** `team_b_b2`
- **Experiment ID:** `exp_022_classifier_feature`
- **Original source:** `git:88dc69163b1f:experiments/exp_022_classifier_feature.md`
- **Source ref:** `88dc69163b1f39aaac55ddfbfc9986e2203cfbdf`
- **Source commit:** `427abcc36b2e94a1c58345458f90df36c122f2a4`
- **Kind:** git-history experiment card
- **Model:** LightGBM
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** CV по фолдам: fold1 `1.677786`, fold2 `1.745313`, fold3 `1.749852`
- **Known score:** CV mean: `1.711550` (лучший на момент: exp_015, `1.708737`)
- **Seed:** LightGBM classifier + baseline LightGBM regressor, `CALIBRATION_DELTA=-0.17`, folds 1–2 для решения, fold3 справочно, seed из `src/config.py`.
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: card metrics survive; runnable source uses a documented later branch-head fallback
- **Notes:** Frozen implementation is copied from the commit that introduced this card. The card-introduction commit contained syntactically incomplete WIP source; the frozen implementation uses the nearest surviving branch-head version for: src/train.py. Exact provenance is in implementation/SOURCE_PROVENANCE.json.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_022 — G6 classifier probability feature boosting

- **Дата:** 2026-08-13
- **Автор:** Codex
- **Коммит:** не коммитили (reject; код откатан)

## Гипотеза

Проверить второй вариант G6 из PLAN.md: обучить classifier `P(y>0)` и добавить вероятность как дополнительный признак в обычную LGBM-регрессию.

## Что изменено относительно базы

Менялась только модельная схема в `src/train.py`: classifier → `cls_positive_prob` → регрессор. После reject код возвращён к exp_015.

## Результат

- CV по фолдам: fold1 `1.677786`, fold2 `1.745313`, fold3 `1.749852`
- CV mean: `1.711550` (лучший на момент: exp_015, `1.708737`)
- LB: не отправляли

## Вердикт и вывод

Reject: оба основных фолда хуже exp_015. Вероятность `P(y>0)` как in-sample feature не решает конфликт между перепрогнозом нулей и недопрогнозом покупателей.

## Конфиг прогона

LightGBM classifier + baseline LightGBM regressor, `CALIBRATION_DELTA=-0.17`, folds 1–2 для решения, fold3 справочно, seed из `src/config.py`.
