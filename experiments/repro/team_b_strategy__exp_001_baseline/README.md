# exp_001 — baseline

## Catalogue metadata

- **Catalogue ID:** `team_b_strategy__exp_001_baseline`
- **Namespace:** `team_b_strategy`
- **Experiment ID:** `exp_001_baseline`
- **Original source:** `git:824f41575bc2:experiments/exp_001_baseline.md`
- **Source ref:** `824f41575bc2fa4ae11b8f6f9dfd907571276d37`
- **Source commit:** `68b543d7b654257812ba4bbb58547fd8a832fb00`
- **Kind:** git-history experiment card
- **Model:** Unknown / not recoverable from repository history
- **Features:** window aggregates
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** CV по фолдам: validation cutoff 2026-01-15
- **Known score:** CV mean: RMSE 244.596747, MAE 65.626454, RMSLE 1.711195
- **Seed:** Модель: `HistGradientBoostingRegressor(loss="squared_error", learning_rate=0.05, max_leaf_nodes=31, max_iter=250, l2_regularization=0.05)`. Train cutoffs: 2025-10-01, 2025-11-01, 2025-12-01. Validation cutoff: 2026-01-15. Test cutoff для сабмита: 2026-02-14. Seed: 42 из `config.py`.
- **Postprocessing:** None documented
- **Submission:** Модель: `HistGradientBoostingRegressor(loss="squared_error", learning_rate=0.05, max_leaf_nodes=31, max_iter=250, l2_regularization=0.05)`. Train cutoffs: 2025-10-01, 2025-11-01, 2025-12-01. Validation cutoff: 2026-01-15. Test cutoff для сабмита: 2026-02-14. Seed: 42 из `config.py`.
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL from the frozen commit when the card contains a command; PARTIAL when the card explicitly says the experimental code was rolled back
- **Notes:** Frozen implementation is copied from the commit that introduced this card.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_001 — baseline

- **Дата:** 2026-08-11
- **Автор:** Codex
- **Коммит:** 9d0e5b7

## Гипотеза

Простые агрегаты пользовательской активности до cutoff уже должны дать рабочую точку отсчёта. Для baseline используем суммы, средние, максимумы и оконные агрегаты за 7/14/30/60/120 дней, а target считаем как будущий `gmv` за 30 дней.

## Что изменено относительно базы

Реализованы `build_features(cutoff_date)` и первый train pipeline на `HistGradientBoostingRegressor` с `log1p` таргетом.

## Результат

- CV по фолдам: validation cutoff 2026-01-15
- CV mean: RMSE 244.596747, MAE 65.626454, RMSLE 1.711195
- LB: не отправляли

## Вердикт и вывод

Нейтрально: baseline успешно обучается и даёт воспроизводимую отправную точку. После уточнения условий соревнования главным ориентиром считаем RMSLE.

## Конфиг прогона

Модель: `HistGradientBoostingRegressor(loss="squared_error", learning_rate=0.05, max_leaf_nodes=31, max_iter=250, l2_regularization=0.05)`. Train cutoffs: 2025-10-01, 2025-11-01, 2025-12-01. Validation cutoff: 2026-01-15. Test cutoff для сабмита: 2026-02-14. Seed: 42 из `config.py`.
