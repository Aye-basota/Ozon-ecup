# exp_068 — RECENCY-RIDGE-ON-PREDICTIONS

## Catalogue metadata

- **Catalogue ID:** `team_a_current__exp_068_recency_ridge_predictions`
- **Namespace:** `team_a_current`
- **Experiment ID:** `exp_068_recency_ridge_predictions`
- **Original source:** `experiments/exp_068_recency_ridge_predictions.md`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** experiment card
- **Model:** Ridge, two-part / hurdle, blend
- **Features:** recency, 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** fit_intercept=True)`, веса фолдов `1:2:4:8`, correction clip `[-2,2]`,
- **Known score:** Unknown / not recoverable from repository history
- **Seed:** `python src/recency_ridge_predictions.py`; seed/model training отсутствуют;
- **Postprocessing:** сохранены, но coefficients, standardizer state, row-level targets/members и
- **Submission:** Nested LOFO, coefficients, controls, segments, test-regime и submission:
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when required data/artifacts are present; otherwise PARTIAL as stated in the card

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_068 — RECENCY-RIDGE-ON-PREDICTIONS

- **Дата:** 2026-08-25
- **Автор:** A1
- **Коммит:** `a28a71f` + working tree
- **Prefix:** `RECENCY_RIDGE_PRED_EXP068_A1`
- **Тип:** artifact-only historical replay/provenance audit; training = **NONE**

## Гипотеза

Успешный teammate Ridge-stack можно точно воспроизвести из frozen prediction
members, после чего проверить его residual signal поверх canonical `latest`, не
повторяя `exp_041` и не обучая Ridge на 227 `S1-E10` признаках.

## Что изменено относительно базы

Модели и submissions не менялись. Добавлен isolated read-only audit runner и
blocked-артефакты под уникальным prefix.

## Результат

- Исторический CSV найден: 250 000 строк, SHA256
  `95965c33bfe32378227e39ec1e0a792fed19cc29dc73d84b8b83bc0cda447959`,
  mean `z=2.3293`.
- Код exact recipe восстановлен: residual target `log1p(y)-table_core`,
  `StandardScaler(copy=False)`, `Ridge(alpha=150, solver=lsqr, tol=1e-4,
  fit_intercept=True)`, веса фолдов `1:2:4:8`, correction clip `[-2,2]`,
  fixed scale `.75`, затем `friend + .55*(candidate_table-table_core)` и
  fixed level `2.3293`.
- Exact member order: `cap, unc, dist, hurdle, multiscale_direct,
  recent_direct, recent_dist, recent_hurdle_fast12`; `stable18` победитель
  действительно **исключал**, несмотря на имя.
- Внешнее описание «prediction-level» неточно: winner имел `include_meta=True`
  и добавлял до 72 raw `meta_raw` колонок. Отдельный prediction-only Ridge был
  другим кандидатом.
- Среди 629 NPZ отсутствуют все 32 exact historical OOF checkpoints и все 6
  недостающих helper TEST checkpoints. Финальный CSV и aggregate validation
  сохранены, но coefficients, standardizer state, row-level targets/members и
  exact `meta_raw` matrices отсутствуют. Circular replay из готового CSV не
  засчитан; требуемый `max_abs<=5e-7` не вычислим.
- Public `1.6492897556391737` записан в Final6h manifest как
  `known_ridge_submission_public`, но SHA-to-score binding отсутствует; для
  exact CSV LB оставлен UNKNOWN.
- Direct lineage: historical winner не входит в `friend`, `occ_meta_B`,
  `occ_raw_X3` или `latest`. B/X3 наследуют другой поздний anchor
  `blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85`.
- TEST geometry всё же показывает почти полное поглощение:
  `corr(z_hist,z_latest)=0.999968244`,
  `Var(z_hist-z_latest)=0.000159637`; к позднему anchor corr `0.999989818`,
  Var difference `0.0000553331`. Это не OOF/residual evidence.
- Canonical four-fold `z_latest` по-прежнему отсутствует (`exp_066/067`),
  `CAP_LINEAGE=UNKNOWN`.
- Nested LOFO, coefficients, controls, segments, test-regime и submission:
  **NOT RUN**. Все обязательные файлы созданы как явные blocked markers.

## Вердикт и вывод

**BLOCKED_HISTORICAL_REPLAY.** Exact historical prediction bank не передан, а
winner дополнительно оказался не чистым prediction-only рецептом. По условию
эксперимента новая модель, lambda/scale search и canonical Phase C запрещены до
exact replay; submission не создавался и LB upload не выполнялся.

## Конфиг прогона

`python src/recency_ridge_predictions.py`; seed/model training отсутствуют;
CPU-only read-only audit. Отчёт:
`research/strategies/results/RECENCY_RIDGE_PRED_EXP068_A1/REPORT.md`.
