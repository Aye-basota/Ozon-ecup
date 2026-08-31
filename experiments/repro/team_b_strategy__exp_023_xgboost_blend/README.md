# exp_023 — XGBoost component in model-level blend

## Catalogue metadata

- **Catalogue ID:** `team_b_strategy__exp_023_xgboost_blend`
- **Namespace:** `team_b_strategy`
- **Experiment ID:** `exp_023_xgboost_blend`
- **Original source:** `git:824f41575bc2:experiments/exp_023_xgboost_blend.md`
- **Source ref:** `824f41575bc2fa4ae11b8f6f9dfd907571276d37`
- **Source commit:** `824f41575bc2fa4ae11b8f6f9dfd907571276d37`
- **Kind:** git-history experiment card
- **Model:** LightGBM, XGBoost, blend
- **Features:** recency, history-depth features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** CV по фолдам:
- **Known score:** fold 1 best: `(0.15, 0.20, 0.35, 0.30)`, RMSLE `1.670792`, bias `-0.0034`
- **Seed:** seed=config.SEED
- **Postprocessing:** проверить модельное разнообразие без просадки уровня.
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL from the frozen commit when the card contains a command; PARTIAL when the card explicitly says the experimental code was rolled back
- **Notes:** Frozen implementation is copied from the commit that introduced this card.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_023 — XGBoost component in model-level blend

- **Дата:** 2026-08-26
- **Автор:** Codex
- **Коммит:** рабочее дерево

## Гипотеза

XGBoost строит деревья иначе, чем LightGBM, поэтому может дать более
декоррелированный компонент на тех же `behavior_v1` признаках. Если добавить его
маленьким/средним весом к текущему model-level ансамблю, средний RMSLE должен
улучшиться за счёт усреднения ошибок.

## Что изменено относительно базы

К компонентам `recency`, `post_order_dist`, `behavior_dist` добавлен
`xgb_behavior`: XGBoost regressor на `behavior_v1`, таргет `log1p(y)`.

## Результат

- CV по фолдам:
  - fold 1 best: `(0.15, 0.20, 0.35, 0.30)`, RMSLE `1.670792`, bias `-0.0034`
  - fold 2 best: `(0.40, 0.05, 0.25, 0.30)`, RMSLE `1.743296`, bias `+0.1236`
- CV mean:
  - `exp019`: `1.707699`
  - `exp022`: `1.707575`
  - `exp023`: `1.707119`
- LB: не отправляли.

Лучший средний grid:

| recency | post_order_dist | behavior_dist | xgb_behavior | mean RMSLE | bias mean |
|---------|-----------------|---------------|--------------|------------|-----------|
| 0.30 | 0.10 | 0.30 | 0.30 | 1.707119 | +0.0619 |
| 0.25 | 0.15 | 0.30 | 0.30 | 1.707120 | +0.0626 |
| 0.30 | 0.15 | 0.25 | 0.30 | 1.707123 | +0.0617 |

Сгенерированные submit-кандидаты:

- `submissions/exp_023_xgb_blend_rec030_post010_beh030_xgb030_scale120.csv`
  - mean log1p `2.346320`
- `submissions/exp_023_xgb_blend_rec030_post010_beh030_xgb030_level_e19.csv`
  - mean log1p `2.370966`, уровень как у exp019

## Вердикт и вывод

**SUBMIT-CANDIDATE.** XGBoost дал лучший локальный прирост среди последних
ансамблей: `-0.000580` к exp019 и `-0.000456` к exp022. Первым на LB стоит
отправлять `exp_023_xgb_blend_rec030_post010_beh030_xgb030_level_e19.csv`, чтобы
проверить модельное разнообразие без просадки уровня.

## Конфиг прогона

```text
validation: two single-cutoff folds from main
fold 1: train 2025-12-15 -> val 2026-01-14
fold 2: train 2025-11-15 -> val 2025-12-15

components:
recency: LightGBM regression, scale 0.64
post_order_dist: LightGBM multiclass dist-head, long_buy_post_order, scale 0.62
behavior_dist: LightGBM multiclass dist-head, behavior_v1, scale 0.62
xgb_behavior: XGBRegressor, behavior_v1, log1p target, scale 0.62

XGBoost:
n_estimators=450, learning_rate=0.03, max_depth=6, min_child_weight=20,
subsample=0.8, colsample_bytree=0.8, reg_lambda=5, tree_method=hist,
seed=config.SEED

grid: weights step 0.05, behavior_dist >= 0.25, xgb_behavior <= 0.30,
global_scale 1.20
submit train_cutoffs: last 8 clean weekly cutoffs, 2025-08-28..2025-10-16
```
