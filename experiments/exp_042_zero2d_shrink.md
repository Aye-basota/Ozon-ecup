# exp_042 — ZERO2D-SHRINK: soft negative correction по amount × p0

- **Дата:** 2026-08-21
- **Автор:** A1
- **Коммит:** рабочее дерево поверх `a28a71f`
- **Код:** `src/zero2d_shrink.py`, `src/test_zero2d_shrink.py`
- **Результаты:** `research/strategies/results/ZERO2D_SHRINK/`
- **Запуск:** `python src/zero2d_shrink.py`

## Гипотеза

Заметная часть ошибки `STRONGEST_CURRENT` может приходить от пользователей с
небольшим/средним положительным прогнозом и высокой вероятностью нулевой покупки
`p0` из боевой головы `S1-DIST`. Проверялось, даёт ли `p0` безопасную мягкую
negative-only поправку сверх самой величины прогноза, без hard zero.

## Что изменено относительно базы

Ни одна модель не обучалась. К сохранённому OOF `STRONGEST_CURRENT` добавлялась
только honest-LOFO поправка в log-space по фиксированной решётке 7 amount bins ×
5 train-only квантилей `p0`; residual shrinkage `n/(n+20000)`, минимум 500 строк,
weighted decreasing isotonic по `p0`, `c<=0`, eta только `0.25/0.50/0.75/1.00`.

## Аудит базы и p0 — PASS

Фактическая смесь воспроизведена из пяти актуальных OOF-компонент:

```text
0.10 CAP + 0.20 UNC + 0.25 DIST + 0.225 ETX-AVG3 + 0.225 SEQ-AVG3
```

| fold | STRONGEST_CURRENT |
|---|---:|
| 2025-09-04 | 1.766883357 |
| 2025-09-18 | 1.760509577 |
| 2025-10-02 | 1.748629224 |
| 2025-10-16 | 1.741278566 |
| **wCV 1:2:4:8** | **1.747509863** |

На ЛИДЕРБОРДЕ ДАЛ 1,649547109893236

`PACT_dist_<fold>.npz` покрывают все **770 616** OOF-строк. На каждом fold
`0<=p0<=1`, порядок `user_id` совпадает с `STRONGEST_CURRENT`, targets совпадают,
`p_act` побитово равен `1-p0`, а восстановленный `z_DIST` отличается от боевого
не более чем на `2.384e-7`. Weighted AUC опубликованного activity signal
`1-p0` равен **0.846020**. Test labels и поздние contaminated targets не читались.

## Где будущие нули создают ошибку

Все числа ниже — aggregate с весами folds 1:2:4:8 после отдельного optimal
fold log-shift; amount определяется только через зарегистрированный `z_shape`.
Будущие нули дают **43.8474% всей squared log error**.

| amount bin | users | actual zero rate | доля всей zero-error | доля всей error |
|---|---:|---:|---:|---:|
| `[0,1)` | 17.33% | 84.07% | 3.973% | 10.147% |
| `[1,3)` | 14.92% | 62.36% | 12.558% | 16.750% |
| **`[3,10)`** | **22.09%** | 41.82% | **32.106%** | **29.577%** |
| `[10,30)` | 20.11% | 21.55% | **32.105%** | 25.242% |
| `[30,50)` | 8.18% | 9.90% | 9.469% | 7.838% |
| `[50,100)` | 8.79% | 5.18% | 6.919% | 6.560% |
| `[100,+inf)` | 8.59% | 1.61% | 2.869% | 3.884% |

Максимум формально в `[3,10)`, но `[10,30)` совпадает с ним до `8e-6` доли:
вместе эти два диапазона дают **64.21%** всей ошибки будущих нулей.

Порог 30/50 при этом **небезопасен**. Ниже 30 лежат 74.44% пользователей и
80.74% zero-error, но actual-zero rate там только 50.29%; ниже 50 — уже 82.62%
людей и 90.21% zero-error при zero rate 46.30%. Amount локализует абсолютный
вклад ошибки, но не отделяет нули от большого числа positives.

## Honest nested LOFO и controls — REJECT

Для каждого outer fold mapping, p0 edges и eta использовали только остальные
три folds. Fold weights в cell residual означали
`sample_weight=project_fold_weight/n_rows_in_fold`. Все четыре outer-запуска
выбрали eta **1.00**.

| fold | BASE | ZERO2D | delta | eta |
|---|---:|---:|---:|---:|
| 2025-09-04 | 1.766883357 | 1.766919174 | **+0.000035818** | 1.00 |
| 2025-09-18 | 1.760509577 | 1.760514513 | **+0.000004936** | 1.00 |
| 2025-10-02 | 1.748629224 | 1.748584459 | **−0.000044764** | 1.00 |
| 2025-10-16 | 1.741278566 | 1.741248820 | **−0.000029746** | 1.00 |
| **wCV** | **1.747509863** | **1.747485107** | **−0.000024756** | **2/4** |

| control | honest ΔwCV | folds | eta outer |
|---|---:|---:|---:|
| **ZERO2D** | **−0.000024756** | 2/4 | 1/1/1/1 |
| **AMOUNT-ONLY** | **−0.000034321** | **4/4** | 1/1/1/1 |
| **shuffled p0** внутри `fold × amount` | **−0.000026348** | 3/4 | 1/1/1/1 |

ZERO2D не просто не набрал `−0.0005`: AMOUNT-ONLY лучше на `0.0000096`, а
перемешанный `p0` повторяет ZERO2D с разницей `1.6e-6`. Дополнительного signal
`p0` сверх amount нет. После isotonic только 13 из 140 outer mappings cells имели
отрицательную поправку; на honest OOF correction ненулевая у 31.91% строк,
диапазон `[-0.020743, 0]` в log-space. Отдельная fold calibration исключает
объяснение через новый global level.

## Zero / positive decomposition

| metric (aggregate) | BASE | ZERO2D | изменение |
|---|---:|---:|---:|
| RMSLE all (wCV) | 1.747509863 | 1.747485107 | **−0.000024756** |
| MSE contribution `y=0` | 1.339036905 | 1.339060692 | **+0.000023787 хуже** |
| MSE contribution `y>0` | 1.714822412 | 1.714712385 | **−0.000110028 лучше** |
| RMSLE zero rows | 1.856633429 | 1.856646878 | **+0.000013449 хуже** |
| RMSLE positive rows | 1.674532911 | 1.674479375 | **−0.000053536 лучше** |
| AUC(y>0) | 0.843543334 | 0.843544732 | +0.000001399 |

Наблюдаемый микровыигрыш имеет **обратный мотивировавшему механизм**: actual
zeros стали чуть хуже, positives — чуть лучше. Поэтому вопрос «выигрыш zeros
больше ущерба positives?» получает ответ **нет: выигрыша zeros вообще нет**.

Сегменты также не дают основания спасать результат: `rec_buy 15–60`
`+0.000016`, `w180 2–15` `+0.000009`; небольшие минусы у `w180 0–1`
`−0.000101`, `w180 >=16` `−0.000070`, never purchased `−0.000146` намного ниже
пола и не разрешают ручные segment weights.

## Diagnostic hard zero

| rule | ΔwCV | изменение zero-error | изменение positive-error |
|---|---:|---:|---:|
| top 5% по p0 → `z=0` | **+0.001266** | −0.283% | **+0.479%** |
| top 10% по p0 → `z=0` | **+0.004970** | **+1.773%** | −0.370% |

Оба правила вредят общей метрике. Даже top 5%, где zero-error действительно
уменьшается, теряет на positives больше; top 10% после общей fold calibration
ломает уже и zero-error. Hard-zero submission не строился.

## Decision gate и прямые ответы

1. Доля всей ошибки от будущих нулей: **43.8474%**.
2. Максимальный amount bin: **`[3,10)` — 32.1063% zero-error**, практически
   точная ничья с `[10,30)` — 32.1055%.
3. Threshold 30/50 небезопасен: охватывает 74.44%/82.62% пользователей при
   zero rate лишь 50.29%/46.30%; hard zero по более сильному `p0` уже вреден.
4. Honest ZERO2D LOFO: **−0.000024756**, 2/4.
5. AMOUNT-ONLY: **−0.000034321**, 4/4 — лучше ZERO2D.
6. Shuffled-p0: **−0.000026348**, 3/4 — неотличим от ZERO2D.
7. Zero contribution **+0.000023787 хуже**, positive contribution
   **−0.000110028 лучше**; механизм про исправление zeros не сработал.
8. Улучшено **2/4** folds.
9. `2025-10-16`: **−0.000029746**.
10. Eta: **1.00 на всех четырёх outer folds**.
11. Test regime: **PASS** только для явно запрошенного post-REJECT LB probe:
    `Var(c_test)/Var(c_oof)=1.0438`, empty cells 0.0008%, `<500` support 0.308%,
    `p0` вне OOF min/max 0.004%.
12. Submission: **создан по последующему явному запросу владельца** —
    `submission_ZERO2D_SHRINK.csv`; это не меняет локальный REJECT.
13. Финальный статус: **REJECT**.

Провалены обязательные гейты `ΔwCV<=−0.0005`, минимум 3/4, отрыв от
AMOUNT-ONLY `0.0002` и улучшение actual-zero error. Последний fold и shuffle
gate прошли, но этого недостаточно. Первичный прогон корректно остановился до
production. Позже владелец явно запросил curiosity-сабмит вопреки REJECT;
новые thresholds, число bins, shrinkage, segment weights и hard zeros при этом
не подбирались.

## Post-REJECT LB probe по явному запросу

Full-OOF mapping на всех четырёх folds выбрал `eta=1.00`; кривая wCV для
`eta=0.25/0.50/0.75/1.00` равна
`1.747498378 / 1.747488795 / 1.747481113 / 1.747475333`. Это production-выбор
после уже завершённого nested LOFO, не замена честной оценки
`1.747485107` (`delta −0.000024756`).

Test regime прошёл:

- `Var(correction_test)/Var(correction_oof) = 1.043776`;
- max cell-share shift `0.042502`, empty-cell fraction `0.000008`, доля строк в
  cells с OOF `n<500` `0.003080`, `p0` вне OOF min/max `0.000040`;
- доля ненулевой correction `34.3032%` на test против `31.8748%` OOF;
- test correction p01/p05/p50/p95/p99 =
  `−0.019217 / −0.019217 / 0 / 0 / 0`;
- перед correction штатный уровень выставлен в `2.3293`, среднее после
  correction стало `2.324433833`, финальный relevel вернул `2.3293`.

Для извлечения test `p0` полная 29-cutoff DIST-голова переобучена тем же
production recipe (`6,065,972 × 227`, 250 rounds, seed 42). Она не воспроизвела
старый `ztest_S1-DIST.npy` побитово даже при буквальном исходном execution path:
`max|Δz|=0.733360`, `MAE=0.044530`, `corr=0.9991905`. Поэтому используется
same-recipe rebuild `p0` только в рамках явного curiosity override; это
дополнительная причина не считать файл production-safe ACCEPT.

Сохранён `submissions/submission_ZERO2D_SHRINK.csv`: 250,000 строк в порядке
sample submission, `mean(log1p(pred))=2.3293000000`, нулей `0.178%`, min/max
`0 / 2654.1543`, SHA-256
`9f1cf32671fb18291659b61da232244d370f7c9af2e0cf9d8aebf9eba406d461`.
Локальный финальный статус остаётся **REJECT; PREPARED FOR LB PROBE**.

## Тесты, outputs и воспроизведение

```text
python -m pytest src/test_zero2d_shrink.py -q
# 10 passed
python src/zero2d_shrink.py
```

Тесты закрепляют точные amount boundaries, отсутствие outer fold в p0 quantiles,
residual mapping и fold calibration, weighted isotonic monotonicity, `c<=0`,
eta=0, log-space application, порядок sample submission и финальный уровень
2.3293. Инварианты CSV проверены отдельно: доли пользователей/ошибки суммируются
в 1, `zero+positive=total`, sparse cells равны нулю, все 140 cell corrections
монотонны и неположительны.

Основные outputs: `amount_diagnostics.csv`, `zero_error_by_amount.csv`,
`zero2d_cells.csv`, `nested_lofo.csv`, `controls.csv`, `error_decomposition.csv`,
`segments.csv`, `test_regime.json`; дополнительно `audit.json`, `hard_zero.csv`,
`summary.json` и три зарегистрированных PNG.
