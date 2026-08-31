# exp_020 — behavior_v1_slim

## Catalogue metadata

- **Catalogue ID:** `team_b_strategy__exp_020_behavior_v1_slim`
- **Namespace:** `team_b_strategy`
- **Experiment ID:** `exp_020_behavior_v1_slim`
- **Original source:** `git:824f41575bc2:experiments/exp_020_behavior_v1_slim.md`
- **Source ref:** `824f41575bc2fa4ae11b8f6f9dfd907571276d37`
- **Source commit:** `adc03ee1ea639bcde0b4e10cfca61936b1aaac1d`
- **Kind:** git-history experiment card
- **Model:** ensemble
- **Features:** recency
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** CV по фолдам: fold1 `1.671527 -> 1.671674` против full behavior; fold2 `1.743872 -> 1.743918`
- **Known score:** CV mean: `1.707796` (full exp019 `1.707699`, exp017 `1.708295`)
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL from the frozen commit when the card contains a command; PARTIAL when the card explicitly says the experimental code was rolled back
- **Notes:** Frozen implementation is copied from the commit that introduced this card.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_020 — behavior_v1_slim

- **Дата:** 2026-08-17
- **Автор:** Codex
- **Коммит:** pending

## Гипотеза

Полный `behavior_v1` дал новый лучший LB, но стал слишком тяжелым. Проверяем slim-версию: оставить все старые `long_buy_post_order` фичи и только top-50 новых `b1_` по gain/permutation.

## Что изменено относительно базы

Добавлен feature set `behavior_v1_slim`: 215 старых фичей + 50 отобранных `b1_` фичей вместо 114.

## Результат

- CV по фолдам: fold1 `1.671527 -> 1.671674` против full behavior; fold2 `1.743872 -> 1.743918`
- CV mean: `1.707796` (full exp019 `1.707699`, exp017 `1.708295`)
- LB: `1.6547357224929944`

## Вердикт и вывод

Провал как submit: локально slim оставался лучше exp017, но на LB оказался хуже exp017 и full exp019. Использовать как диагностический отбор, но не как боевую модель и не как точное объяснение champion full-модели.

## Конфиг прогона

CV: `python src/dist_head_ensemble.py cv --dist-feature-set behavior_v1_slim --weight-grid 0.5`. Submit: `python src/dist_head_ensemble.py submit --dist-feature-set behavior_v1_slim --recency-weight 0.5 --output exp_020_behavior_v1_slim_dist_wrec050_scale120.csv`. Веса и dist-head как в exp019; feature count `265`, из них `50` новых `b1_`.
