# exp_024 — CatBoost + XGBoost blend

## Catalogue metadata

- **Catalogue ID:** `team_b_strategy__exp_024_cat_xgb_blend`
- **Namespace:** `team_b_strategy`
- **Experiment ID:** `exp_024_cat_xgb_blend`
- **Original source:** `git:824f41575bc2:experiments/exp_024_cat_xgb_blend.md`
- **Source ref:** `824f41575bc2fa4ae11b8f6f9dfd907571276d37`
- **Source commit:** `824f41575bc2fa4ae11b8f6f9dfd907571276d37`
- **Kind:** git-history experiment card
- **Model:** LightGBM, CatBoost, XGBoost, blend
- **Features:** recency, history-depth features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** таргет `log1p(y)`. Веса подбирались по validation grid.
- **Known score:** fold 1 best: `(0.15, 0.15, 0.30, 0.20, 0.20)`, RMSLE `1.670680`, bias `-0.0034`
- **Seed:** task_type=CPU, seed=config.SEED
- **Postprocessing:** потому что обычный submit сильно ниже по уровню.
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL from the frozen commit when the card contains a command; PARTIAL when the card explicitly says the experimental code was rolled back
- **Notes:** Frozen implementation is copied from the commit that introduced this card.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_024 — CatBoost + XGBoost blend

- **Дата:** 2026-08-26
- **Автор:** Codex
- **Коммит:** рабочее дерево

## Гипотеза

CatBoost может быть сильным отдельным бустингом и давать ошибки, отличающиеся от
LightGBM/XGBoost. Если добавить CatBoost на `behavior_v1` фичах поверх exp023, то
ансамбль должен улучшиться за счёт model-family diversity.

## Что изменено относительно базы

К exp023 добавлен пятый компонент `cat_behavior`: CatBoostRegressor на `behavior_v1`,
таргет `log1p(y)`. Веса подбирались по validation grid.

## Результат

- CV по фолдам:
  - fold 1 best: `(0.15, 0.15, 0.30, 0.20, 0.20)`, RMSLE `1.670680`, bias `-0.0034`
  - fold 2 best: `(0.30, 0.00, 0.20, 0.30, 0.20)`, RMSLE `1.743066`, bias `+0.1262`
- CV mean:
  - `exp019`: `1.707699`
  - `exp023`: `1.707119`
  - `exp024`: `1.706955`
- LB: не отправляли.

Лучший средний grid:

| recency | post_order_dist | behavior_dist | xgb_behavior | cat_behavior | mean RMSLE | bias mean |
|---------|-----------------|---------------|--------------|--------------|------------|-----------|
| 0.25 | 0.10 | 0.20 | 0.25 | 0.20 | 1.706955 | +0.0621 |
| 0.20 | 0.10 | 0.25 | 0.25 | 0.20 | 1.706955 | +0.0629 |
| 0.20 | 0.10 | 0.20 | 0.30 | 0.20 | 1.706957 | +0.0626 |

Сгенерированные submit-кандидаты:

- `submissions/exp_024_cat_xgb_blend_rec025_post010_beh020_xgb025_cat020_scale120.csv`
  - mean log1p `2.335396`
- `submissions/exp_024_cat_xgb_blend_rec025_post010_beh020_xgb025_cat020_level_e19.csv`
  - mean log1p `2.370966`, уровень как у exp019

## Вердикт и вывод

**SUBMIT-CANDIDATE.** CatBoost улучшил локальный CV поверх XGBoost blend:
`1.707119 → 1.706955`. Основной файл для LB — level-matched
`exp_024_cat_xgb_blend_rec025_post010_beh020_xgb025_cat020_level_e19.csv`,
потому что обычный submit сильно ниже по уровню.

Важное продолжение: текущий grid ограничивал `cat_behavior_weight <= 0.20`, и
лучшие решения упёрлись в этот лимит. Значит следующий эксперимент — CatBoost-heavy
grid с верхним лимитом 0.60–0.80.

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
cat_behavior: CatBoostRegressor, behavior_v1, log1p target, scale 0.62

CatBoost:
iterations=500, learning_rate=0.05, depth=6, l2_leaf_reg=3,
task_type=CPU, seed=config.SEED

grid: weights step 0.05, behavior_dist >= 0.20,
xgb_behavior <= 0.30, cat_behavior <= 0.20, global_scale 1.20
submit train_cutoffs: last 8 clean weekly cutoffs, 2025-08-28..2025-10-16
```
