# exp_027 — RENEWAL-01 / Next-Purchase Clock

## Catalogue metadata

- **Catalogue ID:** `renewal_branch__exp_027_next_purchase_clock`
- **Namespace:** `renewal_branch`
- **Experiment ID:** `exp_027_next_purchase_clock`
- **Original source:** `git:bbae4b0c7a14:experiments/exp_027_next_purchase_clock.md`
- **Source ref:** `bbae4b0c7a14f3aa42aedff05d8d02c2c8fffdba`
- **Source commit:** `c9c6ddbe49f31916ba16854dfc6fd9423189a504`
- **Kind:** git-history experiment card
- **Model:** LightGBM, sequence model, Ridge, ensemble, calibration diagnostic
- **Features:** recency, Search/Catalog decomposition, gap/burst features, EWM aggregates, 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** val/test b3, `T+30≤V`, фолды 09-04/09-18/10-02/10-16, wCV 1:2:4:8.
- **Known score:** честный wCV **−0.000416**. Знак 4/4 устойчив, но проект требует ≤−0.0005 даже
- **Seed:** min leaf 500, L2 10, seeds 42/43/44 от `config.SEED`, 88 timing-only features.
- **Postprocessing:** существующей two-part головы; сырой GMV и log-space не перемножались.
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL from the frozen commit when the card contains a command; PARTIAL when the card explicitly says the experimental code was rolled back
- **Notes:** Frozen implementation is copied from the commit that introduced this card.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_027 — RENEWAL-01 / Next-Purchase Clock

- **Дата:** 2026-08-13
- **Автор:** Codex
- **Коммит реализации:** `c9c6ddb`
- **Ветка:** `codex/renewal-01`
- **Код:** `src/renewal.py`, `src/renewal_eval.py`, `src/test_renewal.py`
- **Полный отчёт:** `research/renewal_01/README.md`

## Гипотеза

Последовательность интервалов между покупками и положение пользователя внутри
текущего незавершённого интервала могут предсказывать попадание следующей покупки
в `(T, T+30]` лучше, чем действующая activity-head. Проверяется именно
recurrent-event представление, без 227 обычных табличных признаков и без
multi-horizon голов из `exp_024`.

## Что изменено относительно базы

Добавлены независимые R0 Kaplan–Meier/shrinkage и R1 LightGBM на 88 признаках
только purchase timing; production-база не менялась и оценивалась как актуальная
`SEQ-01-MIX` с фиксированной 10% долей safety-компонента CAP.

## Данные и anti-leak

- Purchase-day — строго строка `gmv > 0`, как в `features.target`.
- `gmv = gmv_search + gmv_cat` на всех 4 736 907 purchase-days; максимальная
  ошибка `5.28e-11`, несовпадений при `1e-8` нет. Поэтому Search/Catalog clocks
  определены как `gmv_search > 0` / `gmv_cat > 0`, а не придуманы отдельно.
- Все признаки строит только `build_features(cutoff_date)`, фильтр
  `event_date <= cutoff`; target остаётся `(T, T+30]`.
- S1 без изменений: 29 cutoff'ов `2025-04-03..2025-10-16`, train panel b1,
  val/test b3, `T+30≤V`, фолды 09-04/09-18/10-02/10-16, wCV 1:2:4:8.
- 9 тестов: будущие purchase rows не меняют ни одной колонки, проверены gaps,
  unfinished interval, channel clocks, cold start, KM-censoring и embargo.
- Полный regression suite: **93 passed, 1 pre-existing failed** —
  `test_calval.py::test_early_control_is_inside_corridor_and_earliest_first`
  на cutoff `2025-08-08`; `src/calval.py` и `src/test_calval.py` эксперимент не меняет.

## R0 — statistical renewal baseline

Дискретный Kaplan–Meier по завершённым gap с текущим интервалом как right-censored.
Индивидуальная вероятность `ends_next30 / risk_at_recency` shrink'ится к cohort KM
по median gap; для 0 покупок — сглаженный train prior. Primary shrinkage = 10;
чувствительность `2/5/10/20` измерена, максимум AUC действительно у 10, но вся
кривая остаётся далеко от R1.

## R1 — learned renewal clock

LightGBM binary, 180 раундов, 31 лист, min leaf 500, L2=10; среднее трёх сидов
`SEED/SEED+1/SEED+2`, где базовый seed только из `config.py`. Вход — 88 clock-only
колонок: gaps/quantiles/EWMA/trend/regularity/burstiness/normalized recency,
unfinished-interval risk set, cold-start flags и отдельные Search/Catalog clocks.
Первые по gain: `since_third_purchase`, `n_events`, `rec_over_ewma`, `n_intervals`.
Cross-fitted Platt меняет только calibration, не ранжирование.

## Классификация покупки в следующие 30 дней

Метрики ниже — fold-weighted 1:2:4:8; `CLOCK` = R1 avg3 + cross-fitted Platt.

| сигнал | ROC-AUC | PR-AUC | logloss | Brier | ECE-10 |
|---|---:|---:|---:|---:|---:|
| R0 | 0.81151 | 0.87330 | 0.54066 | 0.17309 | 0.02978 |
| R1 raw | 0.84106 | 0.89414 | **0.47589** | **0.15764** | 0.01497 |
| **CLOCK** | **0.84106** | **0.89414** | 0.47592 | 0.15770 | **0.00469** |
| existing `b30_p` | **0.84552** | **0.89758** | **0.47001** | **0.15545** | 0.00943 |

Clock лучше откалиброван по ECE после Platt, но хуже existing head по всем proper
scores и AUC: ΔAUC = **−0.00445**, Δlogloss = +0.00590, ΔBrier = +0.00224.
По фолдам AUC Clock = 0.83649 / 0.83882 / 0.84123 / 0.84211 против existing
0.84078 / 0.84319 / 0.84549 / 0.84670: проигрыш **4/4**.

Корреляция `p_clock_30` с existing `b30_p`: Pearson 0.96956, Spearman 0.97242;
threshold disagreement 7.52%. Clock прав при ошибке existing в 27 479 строках,
но existing прав при ошибке Clock в 30 451. В `clock-only high` доля покупок
0.4716, в `existing-only high` — 0.5241: разногласие направлено против Clock.

## Cold start и сегменты

Доли OOF: 0 / 1 / 2 / 3+ historical purchases =
10.68% / 6.86% / 6.28% / 76.18%.

| history | AUC Clock | AUC existing | Δ |
|---|---:|---:|---:|
| 0 | 0.50479 | 0.62305 | −0.11826 |
| 1 | 0.56582 | 0.61304 | −0.04723 |
| 2 | 0.57748 | 0.61889 | −0.04142 |
| 3+ | 0.78620 | 0.79259 | −0.00639 |

Ни один проверенный сегмент не выигрывает по AUC. Лучший относительный результат
Clock — normalized recency `<0.5`, но даже там ΔAUC = **−0.00524**. Regular,
irregular, dormant, high/low-frequency и все `recency/typical_gap` bands хуже
existing head. Следовательно, выигрыш не спрятан в малой страте.

Диагностические доли интервалов около 7/14/30/60/90 дней у 635 486 OOF-строк с
хотя бы одним gap: 0.1719 / 0.1280 / 0.1017 / 0.0695 / 0.0505; по фолдам
стабильны. Individual clock хорошо поддержан только у 76.2% строк с 3+ покупками;
на cold-start cohort prior не заменяет activity history.

## Связь с RMSLE

Two-part построен корректно в пространстве таргета:
`E[z] = p_clock_30 * E[log1p(y) | y>0]`, conditional magnitude взята из OOF
существующей two-part головы; сырой GMV и log-space не перемножались.

| модель | 09-04 | 09-18 | 10-02 | 10-16 | wCV | Δ к production |
|---|---:|---:|---:|---:|---:|---:|
| `SEQ-01-MIX` | 1.76749 | 1.76103 | 1.74946 | 1.74222 | **1.74834** | — |
| R0 two-part | 1.85077 | 1.84088 | 1.82556 | 1.81450 | 1.82338 | +0.07504 |
| CLOCK two-part | 1.77894 | 1.77173 | 1.76049 | 1.75388 | 1.75969 | +0.01135 |
| CLOCK replacement (DIST slot, w=0.15) | 1.76693 | 1.76042 | 1.74902 | 1.74188 | 1.74793 | −0.000416 |
| meta SELF-control | 1.76736 | 1.76086 | 1.74924 | 1.74210 | 1.74819 | −0.000155 |
| CLOCK meta | 1.76737 | 1.76071 | 1.74906 | 1.74201 | 1.74808 | −0.000266 |

### Ensemble specialist / LOFO

Grid replacement сохраняет CAP=0.10 и отдаёт Clock часть существующего слота,
не добавляя шестой вес сверх единицы. LOFO каждый раз выбрал 0.15 DIST→Clock;
held-out deltas = −0.000558 / −0.000609 / −0.000440 / −0.000338,
честный wCV **−0.000416**. Знак 4/4 устойчив, но проект требует ≤−0.0005 даже
для разработки: gate не пройден. К тому же эффект уменьшается к последнему fold.

Cross-fitted Ridge на `[base, p_clock, confidence, regularity,
recency/median_gap]`, alpha=1e5, clipping correction ±0.25 даёт −0.000266, 4/4.
Но SELF-control на одном `base` уже даёт −0.000155; инкремент именно Clock —
**−0.000111**. Sensitivity alpha `1e4/1e5/1e6` даёт общую дельту
−0.000309/−0.000266/−0.000217: ни одна точка не проходит gate.

## Ортогональность и residual diagnostics

- `Var(z_clock_two_part − z_base) = 0.06173` = **8.67×** актуального seed floor
  0.00712; corr остатков с production = **0.99007**.
- Corr остатков Clock с E10 / DIST / SEQ / E11 / MHZ / PTIME =
  0.98754 / 0.98822 / 0.98762 / 0.98779 / 0.98764 / 0.98850.
- То есть сигнал действительно ортогональнее очередного GBDT-пересида, но это
  в основном **слабая другая функция**, а не новая полезная информация:
  diversity не компенсирует худшее ранжирование.
- R1 seed sensitivity мала: AUC 0.841051/0.841029/0.841017; pairwise corr
  0.99974, `Var(p_i-p_j)≈3.95e-5`. Вывод не зависит от удачного seed.
- R0 shrinkage 2/5/10/20: AUC 0.80326/0.81020/0.81151/0.81094. Выбор 10 не
  создаёт ложный пик, а весь диапазон заведомо слаб.

## Артефакты

`artifacts/oof_RENEWAL-01.npz` (OOF p, labels, clocks, existing p, amount,
base/meta); `test_RENEWAL-01.npz`; 15 моделей LightGBM; `report_RENEWAL-01.json`;
`renewal_01_metrics.json`; fold/classification/calibration/segment/sensitivity/
correlation/replacement/LOFO/meta/profile CSV. Test `p_clock_30`: mean 0.59095,
q01/q50/q99 = 0.1170/0.6173/0.9977, порядок совпадает с `sample_submit`.

Submission не создавался: ни один ensemble gate не пройден.

## Вердикт и вывод

**STOP.** Renewal-clock не улучшает вероятность покупки: existing head сильнее
на 4/4 и во всех сегментах. Standalone/two-part сильно хуже. Ортогональность
реальна (8.67× seed floor), но LOFO −0.000416 ниже project gate, а честный вклад
Clock в meta всего −0.000111 и неотделим от уже закрытой постобработки.

Ровно один следующий эксперимент: **`SEQ-POS-01` — dense positional supervision
30-дневного target внутри причинного sequence-encoder после завершения AVGSEQ3**.
Renewal/MHZ auxiliaries не добавлять: этот эксперимент и `exp_024` показали, что
явная clock/hazard-разметка не улучшает activity ranking; оставшаяся живая ось —
представление сырой последовательности, уже давшее подтверждённое diversity.

## Конфиг прогона

Одна команда: `python src/renewal.py --baseline-artifacts artifacts`.
R0 KM + beta shrinkage 10; R1 LightGBM binary, 180 rounds, leaves 31,
min leaf 500, L2 10, seeds 42/43/44 от `config.SEED`, 88 timing-only features.
Production CV/weights/level/target semantics не менялись.
