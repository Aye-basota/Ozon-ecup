# exp_023 — HOLIDAY-YOY: персональная сезонность 14.02–15.03

## Catalogue metadata

- **Catalogue ID:** `team_a_current__exp_023_holiday_yoy`
- **Namespace:** `team_a_current`
- **Experiment ID:** `exp_023_holiday_yoy`
- **Original source:** `experiments/exp_023_holiday_yoy.md`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** experiment card
- **Model:** Unknown / not recoverable from repository history
- **Features:** holiday/YoY features, Search/Catalog decomposition, window aggregates, 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** колонок были включены в дорогой DIST-retrain, остановленный после двух фолдов.
- **Known score:** | **wCV 1:2:4:8** | **1.749484** | **1.749577** | **+0.000093** | **−0.000003** |
- **Seed:** correction centered within holiday-history; no-history=0; seed=config.SEED
- **Postprocessing:** `41C551A62A663D29382D3D82274F075F223FFE8E0989ECEF4C71EFD53E9456AA`, уровень
- **Submission:** Создан `submissions/submission_HOLIDAY-YOY.csv`: 250 000 строк, SHA-256
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when required data/artifacts are present; otherwise PARTIAL as stated in the card

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_023 — HOLIDAY-YOY: персональная сезонность 14.02–15.03

- **Дата:** 2026-08-12
- **Автор:** A1 / HOLIDAY-YOY
- **Коммит:** 34a2335 + реализация в рабочем дереве

## Гипотеза

Индивидуальная реакция пользователя на окно 14.02–15.03.2025 относительно
соседних 30-дневных периодов может повториться в аналогичном test-окне 2026 и
дать cross-sectional signal сверх агрегатов S1-DIST-MIX. Обычный autumn CV этот
перенос не воспроизводит, поэтому verdict заранее разделён на CV и прямой
2025→2026 diagnostic с placebo.

## Что изменено относительно базы

База и веса не менялись: `0.15·E10 + 0.30·E02 + 0.10·E03a/S1-CAP + 0.45·DIST`.
Поверх неё добавлена zero-mean поправка из усаженных персональных holiday-response
по GMV, orders, days_buy, cart, searches и catalog; коэффициент каждой группы —
cross-fitted slope(YoY) минус slope(placebo). Уровень снова выставлен ровно
`L*=2.3293`.

Реализованы 40 cutoff-safe `hy_*` колонок, включая Search/Catalog-разложение для
GMV/orders/cart. Для cutoff 03.04/10.04 двухсторонние признаки равны `NaN`, потому
что post-окно заканчивается только 14.04; после этого читаются данные не позже
cutoff. Быстрый сабмит из-за дедлайна использует шесть прямых response; все 40
колонок были включены в дорогой DIST-retrain, остановленный после двух фолдов.

## Диагностика до обучения

Полностью наблюдаемый YoY-прокси до 13.02.2026: персональная реакция на
01–14 января относительно 15–28 января, 2025→2026. Placebo тем же кодом:
15–28 января относительно 29 января–11 февраля.

| сводка по GMV/orders/days_buy | YoY | placebo |
|---|---:|---:|
| median Pearson при истории в оба года | **0.03996** | 0.01281 |
| median Spearman | **0.03833** | 0.01225 |
| median cross-fitted OOS R² | **0.001683** | 0.000077 |
| median AUC знака response-2026 | **0.51822** | 0.50548 |

Все шесть групп дали положительный YoY slope, больший placebo. Сигнал мал, но
устойчив по направлению и примерно втрое сильнее placebo. Leakage audit прошёл:
до 14.04 post-признаки полностью отсутствуют, после — конечны; max source date
для test = 14.04.2025 < 13.02.2026.

## Результат: обычный 4-fold CV

Полный saved-OOF замер прямой поправки поверх боевой смеси:

| fold | база | HOLIDAY-YOY | Δ RMSLE | Δ AUC(y>0) |
|---|---:|---:|---:|---:|
| 2025-09-04 | 1.769134 | 1.769168 | +0.000035 | +0.000002 |
| 2025-09-18 | 1.762566 | 1.762625 | +0.000059 | −0.000005 |
| 2025-10-02 | 1.750727 | 1.750823 | +0.000096 | −0.000003 |
| **2025-10-16** | **1.743135** | **1.743242** | **+0.000107** | −0.000004 |
| **wCV 1:2:4:8** | **1.749484** | **1.749577** | **+0.000093** | **−0.000003** |

`Var(z_new-z_base)=0.00017194`, на test `0.00014016`; max |Δz| test = 0.1300.
Это ниже seed-floor `0.00712` в 41 раз: обычный CV — строго **нейтральный**, не
победа. Partial DIST-retrain со всеми 40 колонками согласуется: component-fold
09-04 −0.00005, 09-18 +0.00032; полный retrain остановлен по 20-минутному дедлайну
и не используется в verdict.

## Importance и сегменты

Deadline-safe вариант линейный, поэтому importance — коэффициенты
`(slope_yoy-slope_placebo)/6`: searches 0.009255, days_buy 0.008831, cart
0.008022, orders 0.006476, catalog 0.005711, GMV 0.003934.

| holiday-history | share OOF | Δ wCV | Δ AUC |
|---|---:|---:|---:|
| positive | 35.48% | **−0.000029** | −0.000052 |
| negative | 32.90% | +0.000311 | +0.000009 |
| no history | 31.63% | 0 | 0 |

На test no-history = 43.11%; им поправка строго не применяется. Осеннее ухудшение
целиком приходит из negative-history, положительный сегмент почти нейтрален.

## Два отдельных verdict

1. **Обычный CV: NEUTRAL / не прошёл стандартный gate.** ΔwCV +0.000093, 0/4
   улучшенных fold, 10-16 хуже, AUC без изменения.
2. **YoY diagnostic + placebo: PASS малого сигнала.** Pearson/Spearman/OOS R²/AUC
   одинаково выше placebo; это ровно зарегистрированное исключение для одного
   high-risk submission при CV≈0.

## Риск, submission и решение

Главный риск — перенос New-Year response на 23.02/08.03 и `n=1` holiday-year;
обычный CV его не подтверждает. Риск ограничен: средний сдвиг нулевой, E03a/S1-CAP
сохранён, L*=2.3293, Var(Δ) очень мала, no-history не тронут.

Создан `submissions/submission_HOLIDAY-YOY.csv`: 250 000 строк, SHA-256
`41C551A62A663D29382D3D82274F075F223FFE8E0989ECEF4C71EFD53E9456AA`, уровень
2.329300. **Вердикт: SEND_HIGH_RISK** — отправлять только как один осознанный
seasonal bet, не считать заменой текущего S1-DIST-MIX без LB-подтверждения.

## Конфиг прогона

```text
base=S1-DIST-MIX; weights=0.15/0.30/0.10/0.45; level=2.3293
feature source=2025-01-15..2025-04-14; test cutoff=2026-02-13
shrinkage reliability=support/(support+median_positive_support)
beta_m=max(slope_yoy_m-slope_placebo_m, 0); correction=mean_m(beta_m*hy_m)
correction centered within holiday-history; no-history=0; seed=config.SEED
python src/holiday_yoy.py --stage diagnostic
python src/holiday_yoy.py --stage fast
```

Полные таблицы: `research/strategies/results/HOLIDAY-YOY/`.
