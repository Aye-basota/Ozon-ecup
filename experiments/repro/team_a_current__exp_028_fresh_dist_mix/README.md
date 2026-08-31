# exp_028 — FRESH-DIST-MIX: аудит максимально свежей supervision

## Catalogue metadata

- **Catalogue ID:** `team_a_current__exp_028_fresh_dist_mix`
- **Namespace:** `team_a_current`
- **Experiment ID:** `exp_028_fresh_dist_mix`
- **Original source:** `experiments/exp_028_fresh_dist_mix.md`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** experiment card
- **Model:** distribution head, blend
- **Features:** calendar features, freshness/conditional features, gap/burst features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** ## Validation / diagnostics
- **Known score:** `1.77114 -> 1.82485` (`+0.054 RMSLE`) и получила bias около `+0.366`.
- **Seed:** train panel, данные строго до `2026-02-13`; seed/model params не применялись.
- **Postprocessing:** `0.15/0.30/0.10/0.45`, log-space blend и уровень `L*=2.3293`. Ожидание — убрать
- **Submission:** LB-сабмитов. Безопасное использование этих 13 cutoff'ов — отдельная гипотеза
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when required data/artifacts are present; otherwise PARTIAL as stated in the card

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_028 — FRESH-DIST-MIX: аудит максимально свежей supervision

- **Дата:** 2026-08-13
- **Автор:** A2
- **Коммит:** 560b24b

## Гипотеза

Дообучить текущий production `S1-DIST-MIX` на всех физически размеченных
cutoff'ах до `2026-01-14`, не меняя признаки, таргет, модели, параметры, веса
`0.15/0.30/0.10/0.45`, log-space blend и уровень `L*=2.3293`. Ожидание — убрать
искусственный temporal gap и получить сигнал от самой свежей supervision.

## Что изменено относительно базы

Планировалось изменить только production train range: 29 чистых cutoff'ов
`2025-04-03..2025-10-16` → их объединение с 13 поздними cutoff'ами
`2025-10-22..2026-01-14`. Эксперимент остановлен на обязательном аудите до
обучения: поздние метки загрязнены правилом отбора 250 000 пользователей.

## Аудит production pipeline

Текущий `src/predict.py` получает полный train grid из `Setup.grid()`, а
`src/config.py::cutoff_grid()` заканчивает его на
`CORRIDOR_END = 2025-10-16`. Production-компоненты исходного `S1-DIST-MIX`:

| компонент | artifact | постановка | rounds |
|---|---|---|---:|
| E10 | `S1-NORM` | direct, uncapped + `norm_long` | 600 |
| E02 | `S1-UNC` | direct, uncapped | 600 |
| E03a / CAP | `S1-CAP` | direct, `L=180` | 600 |
| DIST | `S1-DIST` | distribution head, 16 бинов | 250 |

То есть принятые позднее capacity/seedAVG-изменения `exp_017/018` в этих
production artifacts не интегрированы; чистый A/B обязан был бы сохранить
исходные 600/250 rounds и seed из `config.py`.

Причина `CORRIDOR_END` зафиксирована прямо в коде, `README.md`, `STATE.md` и
раннем `e08`: target `(T,T+30]` не должен пересекать гарантированное окно
активности панели `2025-11-16..2026-02-13`. Все 250 000 пользователей исходного
набора отобраны условием «хотя бы один активный день в каждом из трёх 30-дневных
блоков» этого окна. Поэтому target позднего cutoff'а обусловлен будущим отбором,
тогда как target теста `2026-02-14..2026-03-15` такого условия не имеет.

Осознанный запрет найден в нескольких независимых местах:

- `STATE.md`, «Не повторять»: любой `T>=2025-10-17` и отдельно `2026-01-14`;
- `README.md`, чистый коридор `2025-04-03..2025-10-16`;
- `research/eda/e08_baseline.py`, явный `DIRTY_CUT = 2026-01-14`;
- `research/strategy_1.md`, Experiment 2: dirty-only модель хуже на чистом holdout;
- `STRATEGY_04_intensive_full_calendar.md`: поздние даты разрешены только для
  условной интенсивной головы на `y>0`, не для полной `direct`/`dist` модели.

## Cutoff'ы, observability и объём

Максимальная дата в parquet проверена кодом: `2026-02-13`. При семантике
`(T,T+30]` математически последний полностью наблюдаемый cutoff действительно
`2026-01-14`, его target — `2026-01-15..2026-02-13`.

Предложенный EXTRA grid отсчитан назад от этой границы с шагом 7 дней, как в
`STRATEGY_04`: `2025-10-22, 10-29, 11-05, 11-12, 11-19, 11-26, 12-03, 12-10,
12-17, 12-24, 12-31, 2026-01-07, 01-14`. Все 13 окон физически наблюдаемы.

| диапазон | cutoff'ов | train rows, 1-block | изменение |
|---|---:|---:|---:|
| BASE clean | 29 | 6 065 972 | — |
| EXTRA dirty | 13 | 3 118 996 | +51.4% к BASE |
| FRESH planned | 42 | 9 184 968 | +3 118 996 |

Физическая полнота не делает метки честными. Прямой пересчёт на фактической
1-block train panel дал:

| cutoff | target end | rows | P(any activity next 30d) | P(y>0) |
|---|---|---:|---:|---:|
| 2025-10-22 | 2025-11-21 | 226 851 | 0.9729 | 0.5759 |
| 2025-10-29 | 2025-11-28 | 228 672 | 0.9783 | 0.5780 |
| 2025-11-05 | 2025-12-05 | 230 480 | 0.9850 | 0.5818 |
| 2025-11-12 | 2025-12-12 | 232 358 | 0.9944 | 0.5843 |
| 2025-11-19 | 2025-12-19 | 234 931 | 0.9927 | 0.5876 |
| 2025-11-26 | 2025-12-26 | 238 157 | 0.9869 | 0.5934 |
| 2025-12-03 | 2026-01-02 | 241 772 | 0.9850 | 0.5856 |
| 2025-12-10 | 2026-01-09 | 246 086 | 0.9900 | 0.5742 |
| 2025-12-17 | 2026-01-16 | 249 063 | 0.9940 | 0.5609 |
| 2025-12-24 | 2026-01-23 | 247 160 | 0.9813 | 0.5493 |
| 2025-12-31 | 2026-01-30 | 246 413 | 0.9801 | 0.5400 |
| 2026-01-07 | 2026-02-06 | 247 053 | 0.9868 | 0.5421 |
| 2026-01-14 | 2026-02-13 | 250 000 | **1.0000** | 0.5407 |

Последняя строка особенно диагностична: и train panel, и будущая активность
равны всем 250 000 пользователям по конструкции. Это selection leakage в
экстенсивной части target, а не нехватка календарных строк и не lookup в фичах.

## Validation / diagnostics

Запрошенные late pseudo-validation cutoff'ы нельзя использовать как критерий:
у любого `V>2025-10-16` само validation target-окно пересекает гарантированное
окно и оптимистично вознаграждает тот же артефакт. Это не auxiliary holdout, а
загрязнённая метрика.

Нужная clean-проверка уже была выполнена в `e08`: модель, обученная только на
`2026-01-14`, на чистом validation cutoff ухудшилась примерно
`1.77114 -> 1.82485` (`+0.054 RMSLE`) и получила bias около `+0.366`.
Следовательно, документированный риск не теоретический и знак уже измерен.

После срабатывания стоп-критерия модели BASE/FRESH не обучались. Поэтому намеренно
не создавались test predictions, component diagnostics, prediction-shift
статистики и `submission_FRESH_DIST_MIX.csv`: любой такой файл выглядел бы
валидным технически, но реализовывал бы заведомо запрещённую supervision.

## Вердикт и вывод

**REJECT до обучения.** Последние 13 cutoff'ов полностью наблюдаемы по календарю,
но систематически загрязнены будущим условием включения пользователя в датасет.
Главная гипотеза в заявленном виде повторяет закрытый `e08`; рост строк на 51.4%
не компенсирует selection leakage, уже стоивший около `+0.054 RMSLE` на чистом
holdout. `S1-DIST-MIX` и текущий лучший pipeline не меняются.

`submission_FRESH_DIST_MIX.csv` не собран и не достоин одного из двух последних
LB-сабмитов. Безопасное использование этих 13 cutoff'ов — отдельная гипотеза
про интенсивную conditional-модель на `y>0` (`STRATEGY_04`), а не разрешение
добавить их в полный winning pipeline.

## Конфиг прогона

Обучение не запускалось по стоп-правилу задачи. Аудит воспроизвёл production
grid через `cutoff_grid(min_history=90, step=7)`, target `(T,T+30]`, 1-block
train panel, данные строго до `2026-02-13`; seed/model params не применялись.
