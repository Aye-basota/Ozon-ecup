# exp_058 — DATASET-FINGERPRINT / USER-IDENTITY AUDIT

## Catalogue metadata

- **Catalogue ID:** `team_a_current__exp_058_dataset_fingerprint`
- **Namespace:** `team_a_current`
- **Experiment ID:** `exp_058_dataset_fingerprint`
- **Original source:** `experiments/exp_058_dataset_fingerprint.md`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** experiment card
- **Model:** LightGBM, CatBoost, calibration diagnostic
- **Features:** calendar features, gap/burst features, dataset/user fingerprint, 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** manual target против production target: max abs `5.46e-12`, в `log1p` `1.78e-15`; сохранённый OOF float32 совпал бит-в-бит на всех 4 фолдах;
- **Known score:** zero RMSLE `−0.000122291`, но positive RMSLE `+0.000206437` — механизм не согласован с общим улучшением;
- **Seed:** параметры и seed — точный `UNC` из `EXP-046`, seed **42** из `src/config.py`;
- **Postprocessing:** None documented
- **Submission:** Test inference / submission: **не запускались и не создавались**.
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when required data/artifacts are present; otherwise PARTIAL as stated in the card

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_058 — DATASET-FINGERPRINT / USER-IDENTITY AUDIT

- **Дата:** 2026-08-24
- **Автор:** A1
- **Коммит:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Префикс артефактов:** `FINGERPRINT_EXP058`

## Гипотеза

Стабильные свойства выгрузки и идентичности пользователя — ранг/биты `user_id`, календарное покрытие, квартальные числа строк, row-group membership и история прохождения production-панелей — могут содержать воспроизводимый сигнал, которого нет в 227 признаках `S1-E10`. Полезность должна проявиться не просто против базы, а против одного фиксированного совместного PERM-контроля, сохраняющего incidence и маргинальные распределения на каждой модельной панели.

## Что изменено относительно базы

К точному `UNC`-рецепту `EXP-046` добавлены 15 из 30 заранее объявленных cutoff-safe fingerprint-признаков; сравниваются две строго парные модели `REAL-FP` и `PERM-FP`, отличающиеся только соответствием этих 15 значений пользователю.

## Phase 0 — аудит данных и target

Полный raw-аудит прошёл до обучения моделей (`PASS`):

- parquet metadata и полный scan совпали: **30 631 006 строк**, **250 000 уникальных пользователей**;
- набор и порядок пользователей совпадают с `sample_submission.csv`; порядок sample уже совпадает с сортировкой `user_id`;
- дублей `user_id × date` — **0**, нарушений raw ordering — **0**, отрицательных `gmv` — **0**;
- `gmv_search + gmv_cat = gmv`: максимальная абсолютная погрешность `5.28e-11`, нарушений при `tol=1e-6` — **0**;
- **4 549 734** нулевых строк сохранены; пропуски календарных дней не заполнялись: у 249 995 пользователей есть внутренние gaps, всего 60 304 696 отсутствующих дней внутри observed spans;
- manual target против production target: max abs `5.46e-12`, в `log1p` `1.78e-15`; сохранённый OOF float32 совпал бит-в-бит на всех 4 фолдах;
- ручные проверки `all_days_present`, `tenure`, `rec_any` дали max diff **0** на всех фолдах.

Первый integrity-only запуск корректно остановился до моделей из-за ошибки самого audit-кода: availability ошибочно сравнивалась с full-year статистикой. После исправления аудит сравнивает только строки `date <= cutoff`; целевой pipeline не менялся.

## Fingerprint-признаки и novelty

Заранее объявлено 30 полей. Автоматический novelty-фильтр против 227 `S1-E10` удалял поле при `|Pearson| ≥ 0.99999` или `|Spearman| ≥ 0.9995`; осталось **15**:

`fp_uid_rank_frac`, `fp_uid_bits_low16`, `fp_uid_bits_high16`, `fp_active_months`, `fp_month_mask_lo7`, `fp_month_mask_hi7`, `fp_rows_q0..q3`, `fp_rowgroup_count`, `fp_panel1_pass_count`, `fp_panel3_pass_count`, `fp_panel1_first_day`, `fp_panel3_first_day`.

Удалено 15 дублирующих/константных полей: rank buckets, sample rank/neighborhood, first/last/observed day и gap/prefix, `rows_q4`, file/partition counts, rowgroup first/last/span. Test-cutoff metadata audit прошёл: 250 000 пользователей, одинаковая схема, все значения конечны, target не используется, физическая metadata доступна (`PASS`). Test inference при этом не запускался.

## PERM-контроль

Одна фиксированная bijection с seed **42** строится внутри точной подписи присутствия пользователя во всех train/validation-панелях. Маппинг неизменен между cutoff'ами, меняет **96.9404%** пользователей и точно сохраняет множество пользователей и числовые маргинальные распределения на каждой из 25 панелей. Страта используется только контролем и не подаётся модели как признак.

Ограничение контроля: incidence-matching намеренно консервативен для признаков, почти заданных самим incidence. На validation меняются только 4.63% `fp_panel1_pass_count`, 0.15% `fp_rowgroup_count`, а `fp_panel1_first_day` инвариантен. Поле всё равно оставлено симметрично в обеих руках; оно не может создать парную REAL−PERM дельту. Остальные identity/quarter поля меняются в основном у 91–99% строк, `panel3`-поля — у 22–36%.

## Конфиг прогона

- модель: LightGBM direct, `rounds=600`, без early stopping;
- параметры и seed — точный `UNC` из `EXP-046`, seed **42** из `src/config.py`;
- train cutoff'ы: 24 даты `2025-04-03..2025-09-11`, шаг 7; `train_blocks=1`, validation `2025-10-16`, `panel_blocks=3`, `L=None`, `norm_long=False`;
- 4 955 174 train-строк, 197 379 validation-строк; 236 base + 15 fingerprint = 251 колонка;
- BASE — исторический `UNC` exact replay: строки, target и prediction совпали; B/C имеют одинаковые row/target/feature-order hashes;
- сильнейший slot: фиксированная замена `UNC` с весом 0.20 внутри `STRONGEST_CURRENT`.

## Результат

| Вариант | RMSLE cal | AUC(y>0) | mean z | offset |
|---|---:|---:|---:|---:|
| BASE UNC, exact replay | 1.745131674 | 0.842916896 | 2.695462435 | −0.064018326 |
| PERM-FP | 1.745029349 | 0.842972933 | 2.680118658 | −0.048667622 |
| REAL-FP | 1.744899577 | 0.843082592 | 2.676275993 | −0.044786544 |
| STRONGEST | 1.741278566 | 0.844315263 | 2.666906284 | −0.035387154 |
| STRONGEST + PERM slot | **1.741185002** | **0.844346843** | 2.663837529 | −0.032318399 |
| STRONGEST + REAL slot | 1.741256049 | 0.844334398 | 2.663068995 | −0.031549866 |

Ключевые парные дельты:

- standalone `REAL−PERM = −0.000129771` — ниже рабочего порога и частично level-effect;
- в целевом слоте `REAL−PERM = +0.000071048`: настоящий fingerprint **хуже контроля**;
- `REAL slot−STRONGEST = −0.000022517`: практически execution noise;
- `ΔAUC REAL−PERM = +0.000109659`, residual alignment corr `0.003507`;
- zero RMSLE `−0.000122291`, но positive RMSLE `+0.000206437` — механизм не согласован с общим улучшением;
- hash-half 0/1: `REAL−PERM = +0.000068771 / +0.000073307`; обе половины проигрывают контролю;
- средняя standalone `Δz = −0.003843` снимается калибровкой, то есть отдельного полезного shape-сигнала не видно.

Когортные срезы не дали стабильного gate: по first-observed-month улучшились 5/8 когорт с диапазоном примерно `−0.00065..+0.00115`; по 256 rank buckets — 109/256. Fingerprint-поля получили только **0.6596%** суммарного gain; нет одного подозрительного поля. Top shares внутри fingerprint gain: `panel1_pass_count` 17.61%, `uid_bits_low16` 17.61%, `uid_rank_frac` 16.00%, `rows_q0` 13.15%. Spearman gain-importance REAL/PERM: 0.978 по всем и 0.879 только по fingerprints.

- CV по фолдам: полный 4-fold CV **не запускался** согласно stop-rule; дешёвый pilot только `2025-10-16`.
- LB: **не отправляли**, public LB не использовался.
- Test inference / submission: **не запускались и не создавались**.

## Вердикт и вывод

**REJECT.** REAL не проходит заранее заданный барьер `−0.0003`, проигрывает корректному PERM-контролю в сильнейшем слоте, проигрывает ему в обеих hash-halves и ухудшает positive-сегмент. Поэтому не запускать full folds, CatBoost, tuning, test inference или submission и не спасать эту exact fingerprint-ветку подбором subsets/buckets/hash-представлений.

## Воспроизведение и артефакты

```powershell
python src/dataset_fingerprint.py
python src/dataset_fingerprint.py --analysis-only
python -m pytest src/test_dataset_fingerprint.py src/test_pipeline.py src/test_validation.py -q
```

Финальная проверка: **36 passed**. Структурированные результаты лежат в `research/strategies/results/FINGERPRINT_EXP058/`, модели и промежуточные матрицы — в gitignored `artifacts/FINGERPRINT_EXP058/`.

Работа выполнена в текущем checkout: отдельный worktree от `BASE_HEAD` потерял бы требуемые незакоммиченные артефакты предыдущих экспериментов. Все новые файлы имеют уникальный префикс EXP058; чужие изменения не сбрасывались и не перезаписывались.
