# exp_002 — conversions

## Catalogue metadata

- **Catalogue ID:** `team_b_strategy__exp_002_conversions`
- **Namespace:** `team_b_strategy`
- **Experiment ID:** `exp_002_conversions`
- **Original source:** `git:824f41575bc2:experiments/exp_002_conversions.md`
- **Source ref:** `824f41575bc2fa4ae11b8f6f9dfd907571276d37`
- **Source commit:** `68b543d7b654257812ba4bbb58547fd8a832fb00`
- **Kind:** git-history experiment card
- **Model:** Unknown / not recoverable from repository history
- **Features:** window aggregates
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** CV по фолдам: validation cutoff 2026-01-15
- **Known score:** CV mean: RMSE 243.974514, MAE 65.601427, RMSLE 1.710919
- **Seed:** Модель: `HistGradientBoostingRegressor(loss="squared_error", learning_rate=0.05, max_leaf_nodes=31, max_iter=250, l2_regularization=0.05)`. Train cutoffs: 2025-10-01, 2025-11-01, 2025-12-01. Validation cutoff: 2026-01-15. Test cutoff для сабмита: 2026-02-14. Seed: 42 из `config.py`. Сабмит: `submissions/exp_002_conversions_hgbr.csv`.
- **Postprocessing:** None documented
- **Submission:** Модель: `HistGradientBoostingRegressor(loss="squared_error", learning_rate=0.05, max_leaf_nodes=31, max_iter=250, l2_regularization=0.05)`. Train cutoffs: 2025-10-01, 2025-11-01, 2025-12-01. Validation cutoff: 2026-01-15. Test cutoff для сабмита: 2026-02-14. Seed: 42 из `config.py`. Сабмит: `submissions/exp_002_conversions_hgbr.csv`.
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL from the frozen commit when the card contains a command; PARTIAL when the card explicitly says the experimental code was rolled back
- **Notes:** Frozen implementation is copied from the commit that introduced this card.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_002 — conversions

- **Дата:** 2026-08-11
- **Автор:** Codex
- **Коммит:** 9d0e5b7

## Гипотеза

Пользователи с одинаковым объёмом активности могут сильно отличаться по качеству этой активности. Конверсии search→cart, search→order, cart→order, средний чек и доли GMV должны помочь модели лучше отличать простое browsing-поведение от покупательского.

## Что изменено относительно базы

К baseline-агрегатам добавлены ratio-фичи по всем данным до cutoff и окнам 7/14/30/60/120 дней.

## Результат

- CV по фолдам: validation cutoff 2026-01-15
- CV mean: RMSE 243.974514, MAE 65.601427, RMSLE 1.710919
- LB: не отправляли

## Вердикт и вывод

Успех: RMSLE улучшился с 1.711195 до 1.710919. Улучшение маленькое, но направление полезное; ratio-фичи стоит оставить как новый baseline.

## Конфиг прогона

Модель: `HistGradientBoostingRegressor(loss="squared_error", learning_rate=0.05, max_leaf_nodes=31, max_iter=250, l2_regularization=0.05)`. Train cutoffs: 2025-10-01, 2025-11-01, 2025-12-01. Validation cutoff: 2026-01-15. Test cutoff для сабмита: 2026-02-14. Seed: 42 из `config.py`. Сабмит: `submissions/exp_002_conversions_hgbr.csv`.
