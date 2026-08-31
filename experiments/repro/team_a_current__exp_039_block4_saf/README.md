# exp_039 — BLOCK4-SAF: selection-aware block-to-block residual correction

## Catalogue metadata

- **Catalogue ID:** `team_a_current__exp_039_block4_saf`
- **Namespace:** `team_a_current`
- **Experiment ID:** `exp_039_block4_saf`
- **Original source:** `experiments/exp_039_block4_saf.md`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** experiment card
- **Model:** LightGBM, calibration diagnostic
- **Features:** calendar features, freshness/conditional features, window aggregates
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Все числа — пофолдовый optimal log-shift после winsor 0.5/99.5% и вычитания
- **Known score:** | **wCV** | **1.747510** | **1.747749** | **+0.000240** | range 0.0 |
- **Seed:** Следовательно, отрицательный вывод не является одним неудачным seed.
- **Postprocessing:** результат не объясняется уровнем. `corr(delta, ly-z_STRONGEST)` по fold:
- **Submission:** validation,test_regime,summary}.json` и плоские CSV diagnostics. Сабмита нет.
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when required data/artifacts are present; otherwise PARTIAL as stated in the card

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_039 — BLOCK4-SAF: selection-aware block-to-block residual correction

- **Дата:** 2026-08-21
- **Автор:** A1
- **Коммит:** рабочее дерево поверх `a28a71f`
- **Код:** `src/block4_saf.py`, `src/test_block4_saf.py`; opt-in additions in
  `src/features.py`; результаты `research/strategies/results/BLOCK4_SAF/`
- **Запуск:** `python src/block4_saf.py`

## Гипотеза

Поздние полностью известные блоки нельзя отдавать обычной direct/activity модели:
панель гарантирует в них `A=1`. Но для `z=log1p(GMV)`, где `z=0` при `A=0`, верно
`E[z|X]=P(A=1|X)E[z|X,A=1]`. Проверялось, даёт ли разность двух одинаковых
conditional-heads, обученных на последних известных блоках, честную residual-поправку
к `STRONGEST_CURRENT`:

```text
delta = q(V) * (nu_F(V) - nu_C(V))
z_new = z_STRONGEST_CURRENT + alpha * centered_winsorized(delta)
```

Это direct correction итогового прогноза, не auxiliary loss и не повтор FNL.

## Аудит постановки — PASS

- activity в организаторском/panel code — **любая строка дневного лога**;
  target/activity используют ровно `(T,T+30]`, то есть даты `T+1..T+30`;
- `purchase => activity`: **0 нарушений на 40 проверенных cutoff'ах**;
- на `2025-11-15`, `2025-12-15`, `2026-01-14`: **250 000/250 000 активны**;
- все четыре `P_V` полностью активны в B2 и B3:
  `188518/188518`, `191025/191025`, `193694/193694`, `197379/197379`;
- features читают только `event_date<=cutoff`; end-to-end mutation-test строк после
  cutoff не меняет ни одну колонку;
- `user_id` и cutoff calendar features отсутствуют в 288 model features;
- `q` использует только clean cutoffs. Последние train labels по fold:
  `07-31 / 08-14 / 08-28 / 09-11`; production заканчивается `2025-10-16`;
- production geometry закреплена тестом: `V=2026-02-13`, `C=2025-12-15`,
  `F=2026-01-14`; cross-fit groups disjoint, production 124 787 / 125 213 users.

Если бы любой пункт нарушился, раннер остановился бы до обучения с
`REJECT_INVALID_ASSUMPTION`; стоп-критерий не сработал.

## Признаки и модели

База — сохранённые **195** колонок `S1-CAP/L=180`. Добавлены **93** opt-in block
features. Точные дубли `block0` не создавались: для `gmv/orders/days_buy/
days_present/searches/to_cart/gmv_cat` переиспользованы `w30_*`; block1 и block2
получены из `w60-w30` и `w90-w60`. Для отсутствовавших в pipeline
`gmv_search/search_to_ord/cat_to_ord` построены три прямых агрегата. Для всех
десяти величин добавлены d01/d12, acceleration, два ratio и mean/std.

`q` — binary LightGBM на пользователях, активных в двух предыдущих блоках;
conditional `nu_C/nu_F` — одинаковые regression LightGBM на противоположной
`splitmix64(user_id)&1` группе. Параметры дословно задания: lr 0.03, leaves 63,
min leaf 500, feature/bagging 0.8, L2 10, max_bin 63, 200 rounds, seeds 42/43/44.
Production `q`: 5 637 626 строк, 29 clean cutoff'ов. Conditional C/F в каждой
стороне имеют **ровно одинаковое число строк**.

## Главный результат: честный LOFO — REJECT

Все числа — пофолдовый optimal log-shift после winsor 0.5/99.5% и вычитания
среднего `delta` внутри fold. Веса 1:2:4:8.

| fold | `STRONGEST_CURRENT` | honest candidate | Δ | alpha без fold |
|---|---:|---:|---:|---:|
| 2025-09-04 | 1.766883 | 1.767574 | **+0.000691** | 0.25 |
| 2025-09-18 | 1.760510 | 1.760789 | **+0.000279** | 0.25 |
| 2025-10-02 | 1.748629 | 1.748790 | **+0.000161** | 0.25 |
| 2025-10-16 | 1.741279 | 1.741491 | **+0.000213** | 0.25 |
| **wCV** | **1.747510** | **1.747749** | **+0.000240** | range 0.0 |

Не просто не взят порог: ухудшены **4/4**, включая главный 10-16. Кривая
монотонно уходит вверх с alpha:

| alpha | ΔwCV (те же пофолдовые calibrated scores) |
|---:|---:|
| 0.25 | +0.000240 |
| 0.50 | +0.001598 |
| 0.75 | +0.004074 |
| 1.00 | +0.007660 |
| 1.25 | +0.012350 |

То есть оптимум лежит у `alpha=0`, которого в кандидатной сетке намеренно нет.

## Controls и diagnostics

### Shuffle

`z_F` перемешан внутри 9 совместных bins `w180_days_buy × rec_buy`, после чего
обучен тот же `nu_F_SHUF`. Его собственный честный LOFO: **+0.007401**, 0/4,
held-out alpha везде 0.25. Shuffle не имитирует полезный gain; control чистый.

### Seed/run floor

Conditional seed 42/43/44 по отдельности дают `ΔwCV` **+0.000311 / +0.000364 /
+0.000326**, каждый 0/4. Среднее трёх снижает вред до +0.000240, но знак не меняет.
Следовательно, отрицательный вывод не является одним неудачным seed.

### No-level и residual signal

`delta` после preprocessing имеет среднее < numerical epsilon в каждом fold, а
score всё равно считается после отдельной оптимальной log-калибровки. Значит
результат не объясняется уровнем. `corr(delta, ly-z_STRONGEST)` по fold:
**−0.0036 / +0.0062 / +0.0098 / +0.0075** — практически ноль; corr residuals
до/после 0.99965..0.99970. `Var(delta)` = 0.0299..0.0344; доля `|delta|>0.1`
51.2..55.2%, `>0.25` 14.5..17.6%, `>0.5` 0.46..1.04%.

`q` сам валиден: AUC activity 0.8974/0.8980/0.8979/0.9008, logloss
0.0898/0.0834/0.0769/0.0736, Brier 0.02325/0.02123/0.01910/0.01823.
Conditional `nu_F` имеет меньший RMSE, чем `nu_C`, на трёх поздних fold'ах
(на 10-16: 1.76476 против 1.76916), но **их разность не предсказывает следующий
residual**. Purchase AUC также не растёт систематически.

### Сегменты

| segment | honest Δ RMSLE |
|---|---:|
| `rec_buy 15–60` | **+0.000203** |
| `w180_days_buy 2–15` | +0.000384 |
| intersection | +0.000332 |
| `w180_days_buy 0–1` | +0.000279 |
| `w180_days_buy 16+` | **−0.000316** |
| never purchased | +0.000164 |

Единственный локальный минус — частые покупатели 16+, но отдельные segment weights
запрещены постановкой и глобального эффекта он не спасает. Главный bottleneck
`rec_buy 15–60` ухудшается.

## Test regime audit — PASS, но submission не создаётся

Production cross-fit полностью досчитан и test predictions сохранены.

```text
Var(delta_oof)  = 0.037932
Var(delta_test) = 0.033058
ratio           = 0.8715
std ratio       = 0.9335
test clipped    = 1.864%
```

Режим находится внутри требуемых 0.5..1.5 и OOF-support. Это важно: validation
провал нельзя списать на test extrapolation. По decision gate при `ΔwCV>−0.0005`
и отсутствии улучшения 10-16 verdict обязан быть REJECT. Поэтому
`submissions/BLOCK4_SAF_submission.csv` **не создан**.

## Вердикт и прямые ответы

**REJECT.** Honest LOFO `+0.000240`, 0/4; последний fold `+0.000213`.

1. **Даёт ли последний блок новый signal?** Для standalone conditional RMSE —
   слабый freshness-эффект есть на 3/4, но нового **residual signal** нет:
   корреляция с ошибкой около нуля и correction ухудшает 4/4.
2. **Работает ли block-to-block extrapolation?** Нет. `nu_F−nu_C` — крупное,
   стабильное изменение функции, но оно не экстраполируется в residual B4.
3. **Улучшается ли `rec_buy 15–60`?** Нет, `+0.000203`.
4. **Проходит ли correction в ансамбле?** Нет, `STRONGEST_CURRENT` ухудшен 4/4
   даже после center + winsor + optimal fold calibration.
5. **Достоин ли кандидат leaderboard submission?** Нет. Test regime безопасен,
   но validation gate провален однозначно; submission не строился.

## Воспроизведение и артефакты

```text
python -m pytest src/test_block4_saf.py src/test_pipeline.py src/test_validation.py -q
python src/block4_saf.py
```

Основные файлы: `artifacts/oof_BLOCK4_SAF.npz`, `artifacts/test_BLOCK4_SAF.npz`,
`artifacts/BLOCK4_SAF_{fold_*,test_raw}.npz`; `results/BLOCK4_SAF/{audit,config,
validation,test_regime,summary}.json` и плоские CSV diagnostics. Сабмита нет.
