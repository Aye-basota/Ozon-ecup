# exp_008 — recency lightgbm scale

## Catalogue metadata

- **Catalogue ID:** `team_b_strategy__exp_008_recency_lightgbm_scale`
- **Namespace:** `team_b_strategy`
- **Experiment ID:** `exp_008_recency_lightgbm_scale`
- **Original source:** `git:824f41575bc2:experiments/exp_008_recency_lightgbm_scale.md`
- **Source ref:** `824f41575bc2fa4ae11b8f6f9dfd907571276d37`
- **Source commit:** `68b543d7b654257812ba4bbb58547fd8a832fb00`
- **Kind:** git-history experiment card
- **Model:** LightGBM, calibration diagnostic
- **Features:** recency
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** CV по фолдам: validation cutoff 2026-01-15
- **Known score:** CV mean: RMSE 254.269612, MAE 65.909651, RMSLE 1.671639
- **Seed:** Модель: LightGBM из exp_006. Feature set: `recency`. Scale grid для кандидатов показал лучший вариант `recency + lightgbm + 0.64`. Train cutoffs: 2025-10-01, 2025-11-01, 2025-12-01. Validation cutoff: 2026-01-15. Test cutoff: 2026-02-14. Seed: 42 из `config.py`.
- **Postprocessing:** None documented
- **Submission:** Успех: лучший текущий CV, заметно лучше baseline 1.711195 и exp_002 1.710919. Сабмит собран в `submissions/exp_008_recency_lightgbm_scale064.csv`.
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL from the frozen commit when the card contains a command; PARTIAL when the card explicitly says the experimental code was rolled back
- **Notes:** Frozen implementation is copied from the commit that introduced this card.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_008 — recency lightgbm scale

- **Дата:** 2026-08-11
- **Автор:** Codex
- **Коммит:** 9d0e5b7

## Гипотеза

Хорошие изолированные идеи можно объединить для сабмита: recency-фичи дают сигнал свежести действий, LightGBM лучше аппроксимирует табличные зависимости, а scale-калибровка оптимизирует RMSLE.

## Что изменено относительно базы

Champion stack: feature set `recency`, модель `LightGBM`, post-scale `0.64`.

## Результат

- CV по фолдам: validation cutoff 2026-01-15
- CV mean: RMSE 254.269612, MAE 65.909651, RMSLE 1.671639
- LB: 1.657

## Вердикт и вывод

Успех: лучший текущий CV, заметно лучше baseline 1.711195 и exp_002 1.710919. Сабмит собран в `submissions/exp_008_recency_lightgbm_scale064.csv`.

## Конфиг прогона

Модель: LightGBM из exp_006. Feature set: `recency`. Scale grid для кандидатов показал лучший вариант `recency + lightgbm + 0.64`. Train cutoffs: 2025-10-01, 2025-11-01, 2025-12-01. Validation cutoff: 2026-01-15. Test cutoff: 2026-02-14. Seed: 42 из `config.py`.
