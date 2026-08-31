# exp_023 — G13 CatBoost blend diagnostic

## Catalogue metadata

- **Catalogue ID:** `team_b_b2__exp_023_catboost_blend`
- **Namespace:** `team_b_b2`
- **Experiment ID:** `exp_023_catboost_blend`
- **Original source:** `git:88dc69163b1f:experiments/exp_023_catboost_blend.md`
- **Source ref:** `88dc69163b1f39aaac55ddfbfc9986e2203cfbdf`
- **Source commit:** `427abcc36b2e94a1c58345458f90df36c122f2a4`
- **Kind:** git-history experiment card
- **Model:** LightGBM, CatBoost, blend
- **Features:** history-depth features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** CV по фолдам для best blend: fold1 `1.674957`, fold2 `1.740600`, fold3 не прогоняли
- **Known score:** CV mean: `1.707779` (лучший на момент: exp_015, `1.708737`)
- **Seed:** CatBoostRegressor `depth=6`, `learning_rate=0.05`, `iterations=2000`, early stopping 200; LightGBM exp_015; folds 1–2, seed из `src/config.py`.
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: card metrics survive; runnable source uses a documented later branch-head fallback
- **Notes:** Frozen implementation is copied from the commit that introduced this card. The card-introduction commit contained syntactically incomplete WIP source; the frozen implementation uses the nearest surviving branch-head version for: src/train.py. Exact provenance is in implementation/SOURCE_PROVENANCE.json.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_023 — G13 CatBoost blend diagnostic

- **Дата:** 2026-08-13
- **Автор:** Codex
- **Коммит:** не коммитили (reject; код не менялся)

## Гипотеза

Проверить второй вариант G13: CatBoostRegressor на `log1p(y)` и blend с accepted calibrated LightGBM в log-пространстве. CatBoost может дать ошибки, отличающиеся от LightGBM.

## Что изменено относительно базы

Код не менялся; диагностический расчёт запускался inline на cached datasets. CatBoost уже есть в `requirements.txt` и `.venv`.

## Результат

- CV по фолдам для best blend: fold1 `1.674957`, fold2 `1.740600`, fold3 не прогоняли
- CV mean: `1.707779` (лучший на момент: exp_015, `1.708737`)
- LB: не отправляли

## Вердикт и вывод

Reject: improvement `0.000959`, ниже порога. CatBoost alone с delta `-0.16` дал mean `1.708933`; best blend weight LGBM `0.5`.

## Конфиг прогона

CatBoostRegressor `depth=6`, `learning_rate=0.05`, `iterations=2000`, early stopping 200; LightGBM exp_015; folds 1–2, seed из `src/config.py`.
