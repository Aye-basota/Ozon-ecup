# exp_014 — classifier zero gate

## Catalogue metadata

- **Catalogue ID:** `team_b_strategy__exp_014_classifier_gate`
- **Namespace:** `team_b_strategy`
- **Experiment ID:** `exp_014_classifier_gate`
- **Original source:** `git:824f41575bc2:experiments/exp_014_classifier_gate.md`
- **Source ref:** `824f41575bc2fa4ae11b8f6f9dfd907571276d37`
- **Source commit:** `fbf1dc24eadaaa0d2a1e2bdc425270015b272123`
- **Kind:** git-history experiment card
- **Model:** LightGBM, ensemble
- **Features:** recency
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** CV по фолдам: `2025-12-15 -> 2026-01-14`, `2025-11-15 -> 2025-12-15`.
- **Known score:** База без gate: mean RMSLE `1.709007`.
- **Seed:** Base model: exp_011 log-space ensemble `recency + long_buy`, component scales `0.64/0.62`, `global_scale=1.20`, `recency_weight=0.5`. Classifier: LightGBM binary, `n_estimators=400`, `learning_rate=0.03`, `num_leaves=31`, seed из `config.py`, feature set `long_buy`.
- **Postprocessing:** Base model: exp_011 log-space ensemble `recency + long_buy`, component scales `0.64/0.62`, `global_scale=1.20`, `recency_weight=0.5`. Classifier: LightGBM binary, `n_estimators=400`, `learning_rate=0.03`, `num_leaves=31`, seed из `config.py`, feature set `long_buy`.
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL from the frozen commit when the card contains a command; PARTIAL when the card explicitly says the experimental code was rolled back
- **Notes:** Frozen implementation is copied from the commit that introduced this card.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_014 — classifier zero gate

- **Дата:** 2026-08-14
- **Автор:** Codex
- **Коммит:** 0366bd9

## Гипотеза

Добавить отдельный классификатор `P(target > 0)` перед финальным прогнозом и занулять пользователей с низкой вероятностью покупки. Это может помочь отличать активных пользователей от тех, кто после прошлой покупки перестал пользоваться сервисом.

## Что изменено относительно базы

Добавлен `src/classifier_gate.py`: поверх exp_011 dense8 log-ensemble обучается `LightGBMClassifier` на бинарный таргет `target > 0`, затем проверяются hard-threshold пороги зануления.

## Результат

- CV по фолдам: `2025-12-15 -> 2026-01-14`, `2025-11-15 -> 2025-12-15`.
- База без gate: mean RMSLE `1.709007`.
- Hard gate:
  - threshold `0.05`: mean RMSLE `1.709007`, zero share `0.0003`
  - threshold `0.08`: mean RMSLE `1.709078`, zero share `0.0102`
  - threshold `0.10`: mean RMSLE `1.709356`, zero share `0.0317`
  - threshold `0.12`: mean RMSLE `1.710203`, zero share `0.0634`
  - threshold `0.15`: mean RMSLE `1.712695`, zero share `0.1134`
  - threshold `0.20`: mean RMSLE `1.716274`, zero share `0.1582`
- LB: не отправляли.

## Вердикт и вывод

Провал. Hard zero gate быстро ухудшает RMSLE: даже малый false-zero rate дорогой, потому что занулённые будущие покупатели получают большой log-error. Ситуация “купил и продолжил искать / купил и ушёл” частично уже покрыта recency-фичами (`recency_search_days`, `recency_to_ord_days`) в текущем ансамбле.

## Конфиг прогона

Base model: exp_011 log-space ensemble `recency + long_buy`, component scales `0.64/0.62`, `global_scale=1.20`, `recency_weight=0.5`. Classifier: LightGBM binary, `n_estimators=400`, `learning_rate=0.03`, `num_leaves=31`, seed из `config.py`, feature set `long_buy`.
