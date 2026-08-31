# exp_034 — A1 temporal ensemble

## Catalogue metadata

- **Catalogue ID:** `team_b_b2__exp_034_temporal_ensemble`
- **Namespace:** `team_b_b2`
- **Experiment ID:** `exp_034_temporal_ensemble`
- **Original source:** `git:88dc69163b1f:experiments/exp_034_temporal_ensemble.md`
- **Source ref:** `88dc69163b1f39aaac55ddfbfc9986e2203cfbdf`
- **Source commit:** `427abcc36b2e94a1c58345458f90df36c122f2a4`
- **Kind:** git-history experiment card
- **Model:** LightGBM, ensemble
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** CV temporal по фолдам: fold1 `1.700107`, fold2 `1.742105`, fold3 `1.731667`.
- **Known score:** CV mean(1,2): temporal `1.721106`, central `1.716842`, разница `+0.004264`.
- **Seed:** LightGBM из `src/train.py`, seed из `src/config.py` (`42`), `CALIBRATION_DELTA=0.0`. Cutoff'ы: fold1 `2025-11-17,2025-12-01,2025-12-15`; fold2 `2025-10-18,2025-11-01,2025-11-15`; fold3 `2025-09-18,2025-10-02,2025-10-16`.
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: card metrics survive; runnable source uses a documented later branch-head fallback
- **Notes:** Frozen implementation is copied from the commit that introduced this card. The card-introduction commit contained syntactically incomplete WIP source; the frozen implementation uses the nearest surviving branch-head version for: src/train.py. Exact provenance is in implementation/SOURCE_PROVENANCE.json.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_034 — A1 temporal ensemble

- **Дата:** 2026-08-13
- **Автор:** Codex
- **Коммит:** 30a2763 + dirty worktree

## Гипотеза

Усреднение в log-пространстве single-cutoff LightGBM-моделей, обученных на соседних cutoff'ах, может снизить дисперсию предсказаний без смешивания сезонов в одном трейне. Это проверяет A1 из `PLAN.md` и отличается от отвергнутого multi-cutoff обучения.

## Что изменено относительно базы

Добавлен `src/temporal_ensemble.py`: 3 соседних train cutoff'а на fold, log-average предсказаний.

## Результат

- CV temporal по фолдам: fold1 `1.700107`, fold2 `1.742105`, fold3 `1.731667`.
- CV central в том же прогоне: fold1 `1.690241`, fold2 `1.743442`, fold3 `1.732848`.
- CV mean(1,2): temporal `1.721106`, central `1.716842`, разница `+0.004264`.
- CV mean(1,2,3): temporal `1.724626`, central `1.722177`, разница `+0.002449`.
- LB: не отправляли.

## Вердикт и вывод

Reject. Fold2/fold3 слегка выигрывают, но fold1 деградирует на `0.009866`; по протоколу fold1 важнее как более похожий на тест. Error-analysis fold1 показывает усиление перепрогноза нулей: `y=0` даёт `54.5%` SLE, rec_buy `15_60` остаётся `41.9%` SLE.

## Конфиг прогона

LightGBM из `src/train.py`, seed из `src/config.py` (`42`), `CALIBRATION_DELTA=0.0`. Cutoff'ы: fold1 `2025-11-17,2025-12-01,2025-12-15`; fold2 `2025-10-18,2025-11-01,2025-11-15`; fold3 `2025-09-18,2025-10-02,2025-10-16`.
