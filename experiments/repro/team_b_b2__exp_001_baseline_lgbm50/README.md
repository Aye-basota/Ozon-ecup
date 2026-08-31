# exp_001 — baseline LightGBM 50 features

## Catalogue metadata

- **Catalogue ID:** `team_b_b2__exp_001_baseline_lgbm50`
- **Namespace:** `team_b_b2`
- **Experiment ID:** `exp_001_baseline_lgbm50`
- **Original source:** `git:88dc69163b1f:experiments/exp_001_baseline_lgbm50.md`
- **Source ref:** `88dc69163b1f39aaac55ddfbfc9986e2203cfbdf`
- **Source commit:** `30a276391f860890b6236058e0cfc76b9b503d83`
- **Kind:** git-history experiment card
- **Model:** LightGBM
- **Features:** window aggregates
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Реализован baseline-пайплайн: 50 признаков из `PLAN.md`, 2-fold out-of-time validation, LightGBM и 3 сабмита.
- **Known score:** CV mean: `1.717017` (лучший на момент: `exp_001`, `1.717017`).
- **Seed:** LightGBM: `objective=regression`, `learning_rate=0.05`, `num_leaves=63`, `feature_fraction=0.8`, `bagging_fraction=0.8`, `bagging_freq=1`, `min_data_in_leaf=100`, `lambda_l2=1.0`, `seed=42`. Cutoff: fold1 train `2025-12-15`, val `2026-01-14`; fold2 train `2025-11-15`, val `2025-12-15`; target window 30 дней; train target = `log1p(y)`, prediction = `expm1`, clip below zero.
- **Postprocessing:** None documented
- **Submission:** Реализован baseline-пайплайн: 50 признаков из `PLAN.md`, 2-fold out-of-time validation, LightGBM и 3 сабмита.
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL from the frozen commit when the card contains a command; PARTIAL when the card explicitly says the experimental code was rolled back
- **Notes:** Frozen implementation is copied from the commit that introduced this card.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_001 — baseline LightGBM 50 features

- **Дата:** 2026-08-12
- **Автор:** Codex
- **Коммит:** a6807ea

## Гипотеза

User-based агрегаты по прошлой активности должны дать честный первый baseline для прогноза GMV за следующие 30 дней. LightGBM на `log1p(target)` должен быть лучше простого правила `predict = GMV за последние 30 дней`.

## Что изменено относительно базы

Реализован baseline-пайплайн: 50 признаков из `PLAN.md`, 2-fold out-of-time validation, LightGBM и 3 сабмита.

## Результат

- CV по фолдам: fold1 LGBM `1.690608`, fold2 LGBM `1.743426`.
- Baselines: fold1 naive30 `2.196829`, naive90 `2.094162`; fold2 naive30 `2.216295`, naive90 `2.063102`.
- CV mean: `1.717017` (лучший на момент: `exp_001`, `1.717017`).
- LB: не отправляли.

## Вердикт и вывод

Нейтрально/accept как стартовый baseline: LGBM заметно лучше naive30 на обоих фолдах. Sanity-диапазон naive30 из `PLAN.md` не совпал с фактическим (`2.196829` на fold1), продолжение выполнено после ручной проверки.

## Конфиг прогона

LightGBM: `objective=regression`, `learning_rate=0.05`, `num_leaves=63`, `feature_fraction=0.8`, `bagging_fraction=0.8`, `bagging_freq=1`, `min_data_in_leaf=100`, `lambda_l2=1.0`, `seed=42`. Cutoff: fold1 train `2025-12-15`, val `2026-01-14`; fold2 train `2025-11-15`, val `2025-12-15`; target window 30 дней; train target = `log1p(y)`, prediction = `expm1`, clip below zero.
