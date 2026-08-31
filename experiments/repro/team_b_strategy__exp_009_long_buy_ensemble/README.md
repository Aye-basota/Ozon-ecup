# exp_009 — long buy ensemble

## Catalogue metadata

- **Catalogue ID:** `team_b_strategy__exp_009_long_buy_ensemble`
- **Namespace:** `team_b_strategy`
- **Experiment ID:** `exp_009_long_buy_ensemble`
- **Original source:** `git:824f41575bc2:experiments/exp_009_long_buy_ensemble.md`
- **Source ref:** `824f41575bc2fa4ae11b8f6f9dfd907571276d37`
- **Source commit:** `68b543d7b654257812ba4bbb58547fd8a832fb00`
- **Kind:** git-history experiment card
- **Model:** LightGBM, ensemble
- **Features:** recency
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** CV по фолдам: validation cutoff 2026-01-15
- **Known score:** CV mean: `long_buy + LightGBM + scale 0.62` RMSLE 1.671832; log-ensemble 50/50 RMSLE 1.670716
- **Seed:** Модели: два `LGBMRegressor` из exp_006. Train cutoffs: 2025-10-01, 2025-11-01, 2025-12-01. Validation cutoff: 2026-01-15. Test cutoff: 2026-02-14. Seed: 42 из `config.py`. Веса ансамбля: 0.5 recency, 0.5 long_buy; смешивание в `log1p(pred)` пространстве.
- **Postprocessing:** Успех локально: long_buy сам по себе чуть хуже exp_008, но в log-space ансамбле улучшает CV с 1.671639 до 1.670716. Файл для отправки: `submissions/exp_009_recency_long_buy_lgbm_logens.csv`.
- **Submission:** Добавлен feature set `long_buy`: recency + окна 90/180/365 для покупочных дней, заказов, GMV, log-GMV mean/std, tenure, first_buy_age и buy-rate/aov признаки. Сабмит строится как log-space ансамбль `recency LightGBM scale 0.64` и `long_buy LightGBM scale 0.62` с весами 0.5/0.5.
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL from the frozen commit when the card contains a command; PARTIAL when the card explicitly says the experimental code was rolled back
- **Notes:** Frozen implementation is copied from the commit that introduced this card.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_009 — long buy ensemble

- **Дата:** 2026-08-12
- **Автор:** Codex
- **Коммит:** 9d0e5b7

## Гипотеза

Importance-файлы S1-BEST показывают, что самые сильные признаки — длинная история покупок: `w180_days_buy`, `w180_orders`, `w365_days_buy`, `w365_orders`, `w90_days_buy`. Добавим похожие long-history buy-фичи и проверим, дают ли они полезную ошибку для ансамбля с текущим champion.

## Что изменено относительно базы

Добавлен feature set `long_buy`: recency + окна 90/180/365 для покупочных дней, заказов, GMV, log-GMV mean/std, tenure, first_buy_age и buy-rate/aov признаки. Сабмит строится как log-space ансамбль `recency LightGBM scale 0.64` и `long_buy LightGBM scale 0.62` с весами 0.5/0.5.

## Результат

- CV по фолдам: validation cutoff 2026-01-15
- CV mean: `long_buy + LightGBM + scale 0.62` RMSLE 1.671832; log-ensemble 50/50 RMSLE 1.670716
- LB: 1.6568530995317488

## Вердикт и вывод

Успех локально: long_buy сам по себе чуть хуже exp_008, но в log-space ансамбле улучшает CV с 1.671639 до 1.670716. Файл для отправки: `submissions/exp_009_recency_long_buy_lgbm_logens.csv`.

## Конфиг прогона

Модели: два `LGBMRegressor` из exp_006. Train cutoffs: 2025-10-01, 2025-11-01, 2025-12-01. Validation cutoff: 2026-01-15. Test cutoff: 2026-02-14. Seed: 42 из `config.py`. Веса ансамбля: 0.5 recency, 0.5 long_buy; смешивание в `log1p(pred)` пространстве.
