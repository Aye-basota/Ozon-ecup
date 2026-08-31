# exp_024 — post-G14 extra diagnostics

## Catalogue metadata

- **Catalogue ID:** `team_b_b2__exp_024_post_g14_diagnostics`
- **Namespace:** `team_b_b2`
- **Experiment ID:** `exp_024_post_g14_diagnostics`
- **Original source:** `git:88dc69163b1f:experiments/exp_024_post_g14_diagnostics.md`
- **Source ref:** `88dc69163b1f39aaac55ddfbfc9986e2203cfbdf`
- **Source commit:** `427abcc36b2e94a1c58345458f90df36c122f2a4`
- **Kind:** git-history experiment card
- **Model:** LightGBM, XGBoost, blend
- **Features:** history-depth features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Unknown / not recoverable from repository history
- **Known score:** Unknown / not recoverable from repository history
- **Seed:** Accepted LightGBM exp_015; XGBoost `max_depth=6`, `learning_rate=0.05`, `n_estimators=2000`, hist tree method; folds 1–2, seed из `src/config.py`.
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: card metrics survive; runnable source uses a documented later branch-head fallback
- **Notes:** Frozen implementation is copied from the commit that introduced this card. The card-introduction commit contained syntactically incomplete WIP source; the frozen implementation uses the nearest surviving branch-head version for: src/train.py. Exact provenance is in implementation/SOURCE_PROVENANCE.json.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_024 — post-G14 extra diagnostics

- **Дата:** 2026-08-13
- **Автор:** Codex
- **Коммит:** не коммитили (reject; код не менялся)

## Гипотеза

После accepted calibration проверить дешёвые постобработки и разнородные blends: segmented calibration, naive blends, XGBoost blend и thresholding малых предсказаний.

## Что изменено относительно базы

Код не менялся; расчёты запускались inline на cached datasets.

## Результат

- `rec_buy` segmented calibration: fold1 `1.675442`, fold2 `1.741799`, mean `1.708621`
- `w365_gmv` decile calibration: fold1 `1.675252`, fold2 `1.741765`, mean `1.708509`
- naive30/naive90 log blends: best weight LGBM `1.0`, mean `1.708737`
- XGBoost blend: fold1 `1.674912`, fold2 `1.741379`, mean `1.708145`
- thresholding low predictions: best threshold `0`, mean `1.708737`
- LB: не отправляли

## Вердикт и вывод

Reject: лучшая extra diagnostic (`XGBoost` blend) улучшила exp_015 только на `0.000592`, ниже порога. Пост-G14 локальный прогресс упёрся в шумовые улучшения.

## Конфиг прогона

Accepted LightGBM exp_015; XGBoost `max_depth=6`, `learning_rate=0.05`, `n_estimators=2000`, hist tree method; folds 1–2, seed из `src/config.py`.
