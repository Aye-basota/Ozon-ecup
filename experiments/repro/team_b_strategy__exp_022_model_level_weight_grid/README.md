# exp_022 — model-level weight grid

## Catalogue metadata

- **Catalogue ID:** `team_b_strategy__exp_022_model_level_weight_grid`
- **Namespace:** `team_b_strategy`
- **Experiment ID:** `exp_022_model_level_weight_grid`
- **Original source:** `git:824f41575bc2:experiments/exp_022_model_level_weight_grid.md`
- **Source ref:** `824f41575bc2fa4ae11b8f6f9dfd907571276d37`
- **Source commit:** `824f41575bc2fa4ae11b8f6f9dfd907571276d37`
- **Kind:** git-history experiment card
- **Model:** LightGBM, blend
- **Features:** recency
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Если обучить сильные компоненты на validation folds и подобрать веса в `log1p`
- **Known score:** fold 1 best: `(0.25, 0.25, 0.50)`, RMSLE `1.671141`, bias `-0.0008`
- **Seed:** seed: config.SEED
- **Postprocessing:** веса ансамбля, а не случайный сдвиг уровня.
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL from the frozen commit when the card contains a command; PARTIAL when the card explicitly says the experimental code was rolled back
- **Notes:** Frozen implementation is copied from the commit that introduced this card.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_022 — model-level weight grid

- **Дата:** 2026-08-26
- **Автор:** Codex
- **Коммит:** рабочее дерево

## Гипотеза

Post-hoc blend готовых CSV слишком слабый, потому что не видит локальную метрику.
Если обучить сильные компоненты на validation folds и подобрать веса в `log1p`
пространстве, можно улучшить `exp019`: оставить behavior_v1 как главный сигнал,
но добавить recency и post-order dist как регуляризующие компоненты.

## Что изменено относительно базы

Вместо фиксированной смеси `recency 0.50 + behavior_dist 0.50` подбираются веса
трёх model-level компонентов: `recency`, `long_buy_post_order dist-head`,
`behavior_v1 dist-head`.

## Результат

- CV по фолдам:
  - fold 1 best: `(0.25, 0.25, 0.50)`, RMSLE `1.671141`, bias `-0.0008`
  - fold 2 best: `(0.50, 0.10, 0.40)`, RMSLE `1.743831`, bias `+0.1216`
- CV mean: best grid `(0.40, 0.15, 0.45)` -> `1.707575`; лучший на момент
  `exp019` был `1.707699`.
- LB: не отправляли.

Top grid:

| recency | post_order_dist | behavior_dist | mean RMSLE | bias mean |
|---------|-----------------|---------------|------------|-----------|
| 0.40 | 0.15 | 0.45 | 1.707575 | +0.0622 |
| 0.35 | 0.20 | 0.45 | 1.707576 | +0.0629 |
| 0.40 | 0.20 | 0.40 | 1.707579 | +0.0620 |
| 0.35 | 0.15 | 0.50 | 1.707585 | +0.0631 |

Сгенерированные submit-кандидаты:

- `submissions/exp_022_model_blend_rec040_post015_beh045_scale120.csv`
  - mean log1p `2.363113`
- `submissions/exp_022_model_blend_rec040_post015_beh045_level_e19.csv`
  - mean log1p `2.370966`, уровень как у exp019

## Вердикт и вывод

**SUBMIT-CANDIDATE.** Локальный прирост небольшой (`-0.000124` к exp019), но знак
правильный. Первым на LB стоит отправлять level-matched файл
`exp_022_model_blend_rec040_post015_beh045_level_e19.csv`, чтобы проверить именно
веса ансамбля, а не случайный сдвиг уровня.

## Конфиг прогона

```text
validation: two single-cutoff folds from main
fold 1: train 2025-12-15 -> val 2026-01-14
fold 2: train 2025-11-15 -> val 2025-12-15

components:
recency: LightGBM regression, scale 0.64
post_order_dist: LightGBM multiclass dist-head, long_buy_post_order, scale 0.62
behavior_dist: LightGBM multiclass dist-head, behavior_v1, scale 0.62

grid: weights step 0.05, behavior_dist >= 0.30, global_scale 1.20
submit train_cutoffs: last 8 clean weekly cutoffs, 2025-08-28..2025-10-16
seed: config.SEED
```
