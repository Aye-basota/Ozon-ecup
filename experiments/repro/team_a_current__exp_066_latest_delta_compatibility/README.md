# exp_066 — LATEST-DELTA-COMPATIBILITY

## Catalogue metadata

- **Catalogue ID:** `team_a_current__exp_066_latest_delta_compatibility`
- **Namespace:** `team_a_current`
- **Experiment ID:** `exp_066_latest_delta_compatibility`
- **Original source:** `experiments/exp_066_latest_delta_compatibility.md`
- **Source ref:** `working tree`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** experiment card
- **Model:** BTYD
- **Features:** freshness/conditional features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Unknown / not recoverable from repository history
- **Known score:** Unknown / not recoverable from repository history
- **Seed:** Training/model/seed отсутствуют. Выполнен только read-only inventory canonical
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when required data/artifacts are present; otherwise PARTIAL as stated in the card

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_066 — LATEST-DELTA-COMPATIBILITY

- **Дата:** 2026-08-25
- **Автор:** A1
- **Коммит:** `a28a71f` + working tree
- **Prefix:** `LATEST_DELTA_COMPAT_EXP066_A1`
- **Тип:** artifact-only prerequisite audit; training = **NONE**

## Гипотеза

Готовые направления SAFE-ANCHOR, BTYD05, FRESH и SEQ65 могли сохранить
incremental residual после смены базы `STRONGEST_CURRENT -> latest`. Проверка
разрешена только на exact canonical four-fold OOF `latest` с полями
`z_latest`, `z_STRONGEST_CURRENT`, `target`, `user_id`, `fold`.

## Что изменено относительно базы

Ничего: primary experiment остановлен до построения кандидатов, потому что
canonical OOF `latest` отсутствует.

## Результат

- Результат/карточка `AUTHORITATIVE-LATEST-INTEGRATION` в worktree не найден.
- Проверены имена и поля 627 NPZ и схемы 83 Parquet: поле `z_latest` не найдено.
- `exp_065` содержит exact OOF только для `STRONGEST_CURRENT` и production
  integration; он прямо считает teammate `latest` public-LB-calibrated и
  ineligible.
- Teammate bundle регистрирует только production/test рецепт
  `.12*STRONGEST + .16*occ_meta_B + .72*occ_raw_X3` и public LB. Это не
  canonical four-fold OOF и не использовано для реконструкции.
- Fixed curves, nested LOFO, residual/segment/hash diagnostics, negative control,
  test-regime и CAP candidate audit не вычислялись. Submission не создан.

## Вердикт и вывод

**BLOCKED_NO_CANONICAL_LATEST_OOF.** Нельзя честно ответить, поглотил ли `latest`
направления BTYD05/FRESH/SEQ65/private-safe anchor. Возобновлять этот exact
experiment можно только после появления authoritative artifact с exact
построчным alignment по `(fold,user_id)` и пятью обязательными полями; test
predictions и public LB не являются заменой.

## Конфиг прогона

Training/model/seed отсутствуют. Выполнен только read-only inventory canonical
artifacts и provenance audit. Отчёт и blocked placeholders:
`research/strategies/results/LATEST_DELTA_COMPAT_EXP066_A1/`; marker artifact:
`artifacts/LATEST_DELTA_COMPAT_EXP066_A1/oof_candidates.npz`.
