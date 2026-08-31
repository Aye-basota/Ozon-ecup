# exp_019 — behavior_v1 фичи в dist-head

## Catalogue metadata

- **Catalogue ID:** `team_b_strategy__exp_019_behavior_v1_dist`
- **Namespace:** `team_b_strategy`
- **Experiment ID:** `exp_019_behavior_v1_dist`
- **Original source:** `git:824f41575bc2:experiments/exp_019_behavior_v1_dist.md`
- **Source ref:** `824f41575bc2fa4ae11b8f6f9dfd907571276d37`
- **Source commit:** `adc03ee1ea639bcde0b4e10cfca61936b1aaac1d`
- **Kind:** git-history experiment card
- **Model:** ensemble
- **Features:** calendar features, recency
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** CV по фолдам: fold1 `1.672166 -> 1.671527`, fold2 `1.744423 -> 1.743872`
- **Known score:** CV mean: `1.707699` (лучший на момент: exp_017, `1.708295`; exp_018 local `1.707921`)
- **Seed:** CV: `python src/dist_head_ensemble.py cv --dist-feature-set behavior_v1 --weight-grid 0.5`. Submit: `python src/dist_head_ensemble.py submit --dist-feature-set behavior_v1 --recency-weight 0.5 --output exp_019_behavior_v1_dist_wrec050_scale120.csv`. Веса: recency `0.50`, dist-head `0.50`, global scale `1.20`; component scales `0.64/0.62`; dist-head `16` bins, `250` rounds, `learning_rate=0.05`, `num_leaves=31`, seed из `config.py`.
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL from the frozen commit when the card contains a command; PARTIAL when the card explicitly says the experimental code was rolled back
- **Notes:** Frozen implementation is copied from the commit that introduced this card.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_019 — behavior_v1 фичи в dist-head

- **Дата:** 2026-08-17
- **Автор:** Codex
- **Коммит:** pending

## Гипотеза

Первый большой батч поведенческих фичей должен уловить не только объем покупок, но и паттерны: регулярность покупок, просрочку собственного цикла, cart friction, post-order и pre-order активность, устойчивость чека и календарные привычки.

## Что изменено относительно базы

Добавлен feature set `behavior_v1`: `long_buy_post_order` + 114 новых `b1_` фичей; в champion-схеме exp_017 второй компонент dist-head переключен с `long_buy_post_order` на `behavior_v1`.

## Результат

- CV по фолдам: fold1 `1.672166 -> 1.671527`, fold2 `1.744423 -> 1.743872`
- CV mean: `1.707699` (лучший на момент: exp_017, `1.708295`; exp_018 local `1.707921`)
- LB: `1.6545023535300867`

## Вердикт и вывод

Успех: это новый лучший LB. Полный `behavior_v1` тяжелый (`329` фичей всего, `114` новых), поэтому следующий шаг — сделать `behavior_v1_slim` по gain/permutation и затем считать SHAP уже на slim-версии.

## Конфиг прогона

CV: `python src/dist_head_ensemble.py cv --dist-feature-set behavior_v1 --weight-grid 0.5`. Submit: `python src/dist_head_ensemble.py submit --dist-feature-set behavior_v1 --recency-weight 0.5 --output exp_019_behavior_v1_dist_wrec050_scale120.csv`. Веса: recency `0.50`, dist-head `0.50`, global scale `1.20`; component scales `0.64/0.62`; dist-head `16` bins, `250` rounds, `learning_rate=0.05`, `num_leaves=31`, seed из `config.py`.
