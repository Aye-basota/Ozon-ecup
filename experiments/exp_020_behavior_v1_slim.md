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
