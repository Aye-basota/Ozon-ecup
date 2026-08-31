# exp_026 — backlog activity trend last7/prev7

## Catalogue metadata

- **Catalogue ID:** `team_b_b2__exp_026_activity_trend`
- **Namespace:** `team_b_b2`
- **Experiment ID:** `exp_026_activity_trend`
- **Original source:** `git:88dc69163b1f:experiments/exp_026_activity_trend.md`
- **Source ref:** `88dc69163b1f39aaac55ddfbfc9986e2203cfbdf`
- **Source commit:** `427abcc36b2e94a1c58345458f90df36c122f2a4`
- **Kind:** git-history experiment card
- **Model:** LightGBM
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** CV по фолдам: fold1 `1.675181`, fold2 `1.741882`, fold3 `1.745876`
- **Known score:** CV mean: `1.708532` (лучший на момент: exp_015, `1.708737`)
- **Seed:** LightGBM exp_015, `CALIBRATION_DELTA=-0.17`, новые `act_*` признаки, folds 1–2 для решения, fold3 справочно, seed из `src/config.py`.
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: card metrics survive; runnable source uses a documented later branch-head fallback
- **Notes:** Frozen implementation is copied from the commit that introduced this card. The card-introduction commit contained syntactically incomplete WIP source; the frozen implementation uses the nearest surviving branch-head version for: src/train.py. Exact provenance is in implementation/SOURCE_PROVENANCE.json.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_026 — backlog activity trend last7/prev7

- **Дата:** 2026-08-13
- **Автор:** Codex
- **Коммит:** не коммитили (reject; код откатан)

## Гипотеза

Добавить признаки тренда активности: последние 7 дней против предыдущих 7 по GMV, orders, carts, searches и presence.

## Что изменено относительно базы

Добавлялись только новые `act_*` признаки в конец `FEATURES` через `build_features`; после reject код возвращён к exp_015.

## Результат

- CV по фолдам: fold1 `1.675181`, fold2 `1.741882`, fold3 `1.745876`
- CV mean: `1.708532` (лучший на момент: exp_015, `1.708737`)
- LB: не отправляли

## Вердикт и вывод

Reject: улучшение mean `0.000205`, ниже порога. Fold3 дополнительно ухудшился.

## Конфиг прогона

LightGBM exp_015, `CALIBRATION_DELTA=-0.17`, новые `act_*` признаки, folds 1–2 для решения, fold3 справочно, seed из `src/config.py`.
