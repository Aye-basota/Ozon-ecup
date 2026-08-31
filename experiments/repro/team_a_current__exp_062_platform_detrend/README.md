# exp_062 — PLATFORM-DETREND

## Catalogue metadata

- **Catalogue ID:** `team_a_current__exp_062_platform_detrend`
- **Namespace:** `team_a_current`
- **Experiment ID:** `exp_062_platform_detrend`
- **Original source:** `experiments/exp_062_platform_detrend.md`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** experiment card
- **Model:** LightGBM
- **Features:** window aggregates
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** CV по фолдам REAL: 1.766883356 / 1.760509577 / 1.748629224 / 1.741278566; дельты к базе `0 / 0 / 0 / 0`.
- **Known score:** wCV REAL / PLACEBO / CONTROL_ONLY: 1.747509862 / 1.747509862 / 1.747509862; `Delta wCV = 0`, REAL−PLACEBO `0`.
- **Seed:** CPU preflight, seed 42 из `src/config.py`; exact aligned exp_037 OOF (SHA256 `40aa9719...0ed0a`); 10 новых признаков, 11 fixed controls, LightGBM residual 120 rounds, `num_leaves=15`, `min_data_in_leaf=1000`, `lambda_l2=50`, scales `0/0.25/0.5/0.75/1`, splitmix64 user 4-way + two-sided scale; cutoff 2025-09-04/09-18/10-02/10-16. Полный frozen protocol: `research/strategies/results/PLATFORM_DETREND_EXP062/HYPOTHESIS_CARD.md`; results SHA256 `15154f3f...ba7e`.
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when required data/artifacts are present; otherwise PARTIAL as stated in the card

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_062 — PLATFORM-DETREND

- **Дата:** 2026-08-25
- **Автор:** A1
- **Коммит:** a28a71f + working tree

## Гипотеза

Активность пользователя относительно наблюдаемого в тот же день platform-wide режима несёт новый сигнал, которого нет в персональных оконных агрегатах. Cutoff-safe 30/90-дневные Search/Cart/Order/GMV суммы, нормированные на интенсивность текущей панели, должны объяснить остаток `STRONGEST-CURRENT` лучше matched date-shuffled placebo.

## Что изменено относительно базы

Через opt-in `build_features(cutoff_date, platform_detrend_source=...)` добавлены 10 новых колонок `pd_*_rel`; существующие признаки и exact exp_037 baseline не менялись.

## Результат

- CV по фолдам REAL: 1.766883356 / 1.760509577 / 1.748629224 / 1.741278566; дельты к базе `0 / 0 / 0 / 0`.
- wCV REAL / PLACEBO / CONTROL_ONLY: 1.747509862 / 1.747509862 / 1.747509862; `Delta wCV = 0`, REAL−PLACEBO `0`.
- Все две-sided donor→recipient проверки на обеих user-halves и всех четырёх folds выбрали scale `0`; segment gains также `0`.
- LB: не отправляли.

## Вердикт и вывод

**REJECT.** Leakage, current-panel selection, alignment и factor-marginal audits прошли; placebo меняет 95.56% date alignments при совпадении marginal factors до `1.4e-15`. Однако learner не использует ни REAL, ни placebo признаки: platform-relative activity условно избыточна с чемпионом. Full model/test/LB не запускать и не спасать factor/window/bin/model tuning в этой exact family.

## Конфиг прогона

CPU preflight, seed 42 из `src/config.py`; exact aligned exp_037 OOF (SHA256 `40aa9719...0ed0a`); 10 новых признаков, 11 fixed controls, LightGBM residual 120 rounds, `num_leaves=15`, `min_data_in_leaf=1000`, `lambda_l2=50`, scales `0/0.25/0.5/0.75/1`, splitmix64 user 4-way + two-sided scale; cutoff 2025-09-04/09-18/10-02/10-16. Полный frozen protocol: `research/strategies/results/PLATFORM_DETREND_EXP062/HYPOTHESIS_CARD.md`; results SHA256 `15154f3f...ba7e`.
