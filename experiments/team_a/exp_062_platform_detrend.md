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
