# exp_003 — trends

## Catalogue metadata

- **Catalogue ID:** `team_b_strategy__exp_003_trends`
- **Namespace:** `team_b_strategy`
- **Experiment ID:** `exp_003_trends`
- **Original source:** `git:824f41575bc2:experiments/exp_003_trends.md`
- **Source ref:** `824f41575bc2fa4ae11b8f6f9dfd907571276d37`
- **Source commit:** `68b543d7b654257812ba4bbb58547fd8a832fb00`
- **Kind:** git-history experiment card
- **Model:** Unknown / not recoverable from repository history
- **Features:** window aggregates
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** CV по фолдам: validation cutoff 2026-01-15
- **Known score:** CV mean: RMSE 244.595270, MAE 65.603401, RMSLE 1.711856
- **Seed:** Модель: `HistGradientBoostingRegressor(loss="squared_error", learning_rate=0.05, max_leaf_nodes=31, max_iter=250, l2_regularization=0.05)`. Feature set: `trends`. Train cutoffs: 2025-10-01, 2025-11-01, 2025-12-01. Validation cutoff: 2026-01-15. Seed: 42 из `config.py`.
- **Postprocessing:** None documented
- **Submission:** Провал: RMSLE ухудшился с baseline 1.711195 до 1.711856. В текущей модели такие trend-фичи добавляют шум сильнее, чем полезный сигнал; в default-сабмит их отправлять не стоит.
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL from the frozen commit when the card contains a command; PARTIAL when the card explicitly says the experimental code was rolled back
- **Notes:** Frozen implementation is copied from the commit that introduced this card.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_003 — trends

- **Дата:** 2026-08-11
- **Автор:** Codex
- **Коммит:** 9d0e5b7

## Гипотеза

Будущая ценность пользователя должна зависеть не только от накопленной активности, но и от её направления. Если последние 7/30/60 дней заметно сильнее или слабее предыдущего сопоставимого периода, модель может лучше поймать рост или затухание спроса.

## Что изменено относительно базы

К baseline-агрегатам добавлены trend-фичи для `active_days`, `searches`, `to_cart`, `to_ord`, `gmv`: recent vs previous для 7/14, 30/60, 60/120 и daily-ratio для 7/30, 14/60, 30/120.

## Результат

- CV по фолдам: validation cutoff 2026-01-15
- CV mean: RMSE 244.595270, MAE 65.603401, RMSLE 1.711856
- LB: не отправляли

## Вердикт и вывод

Провал: RMSLE ухудшился с baseline 1.711195 до 1.711856. В текущей модели такие trend-фичи добавляют шум сильнее, чем полезный сигнал; в default-сабмит их отправлять не стоит.

## Конфиг прогона

Модель: `HistGradientBoostingRegressor(loss="squared_error", learning_rate=0.05, max_leaf_nodes=31, max_iter=250, l2_regularization=0.05)`. Feature set: `trends`. Train cutoffs: 2025-10-01, 2025-11-01, 2025-12-01. Validation cutoff: 2026-01-15. Seed: 42 из `config.py`.
