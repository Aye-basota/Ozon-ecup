# exp_005 — S1-E10: длинные окна, нормированные на доступную глубину истории

## Catalogue metadata

- **Catalogue ID:** `team_a_current__exp_005_s1_e10_normalized_long`
- **Namespace:** `team_a_current`
- **Experiment ID:** `exp_005_s1_e10_normalized_long`
- **Original source:** `experiments/exp_005_s1_e10_normalized_long.md`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** experiment card
- **Model:** Unknown / not recoverable from repository history
- **Features:** gap/burst features, 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** | фолд | E02 (uncapped) | E10 (нормированные) | Δ | bias E02 → E10 |
- **Known score:** | **CV mean** | **1.76182** | **1.75988** | **−0.00194** | −0.0950 → **−0.0545** |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** до `L = 180`, где то же снижение смещения стоило +0.0061. Уровень прогноза на тесте
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when required data/artifacts are present; otherwise PARTIAL as stated in the card

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_005 — S1-E10: длинные окна, нормированные на доступную глубину истории

- **Дата:** 2026-08-10
- **Автор:** A1 (Strategy 1)
- **Ветка:** `team-a-strategy-1-impl`

## Гипотеза

`exp_004` показал развилку: длинные окна несут реальный сигнал (их удаление стоит
+0.0073, укорочение до 90 дней — +0.0305), но именно они дают adversarial AUC 0.99
против теста. Значит лечить надо не длину окна, а **несопоставимость** его между
cutoff'ами.

Наблюдение, из которого выросла правка: на cutoff'е доступно `T − 2025-01-01` дней,
то есть 92..289 в коридоре и 409 на тесте. Поэтому `w365_*` — это сумма за
`min(365, avail)` дней, а не за 365.

Проверено кодом: `all_*` **побитово совпадают** с `w365_*` на каждом обучающем
cutoff'е (`max|разница| = 0` для `days_present`, `days_buy`, `gmv`, `orders`,
`searches`) и расходятся **только на тесте**. То есть `all_*` и производные
`lifetime_*` не несут в обучении ни бита информации сверх `w365_*`, а на тесте
ведут себя иначе — чистый риск без выгоды.

## Что изменено относительно базы (S1-E02, 1.76182)

1. `all_*` и `lifetime_*` выброшены (9 признаков).
2. `w365_*` суммы и счётчики умножены на `365 / min(365, доступная история)`.
3. `tenure` → `tenure_frac = tenure / avail`, `first_buy_age` → `first_buy_frac`,
   `gap_max` → `gap_max_frac`.
4. Нормировка применяется **до** производных, чтобы отношения и логарифмы считались
   уже от сопоставимых величин.

236 признаков → 227.

## Результат

| фолд | E02 (uncapped) | E10 (нормированные) | Δ | bias E02 → E10 |
|------|----------------|---------------------|---|-----------------|
| 2025-09-04 | 1.78022 | 1.77429 | **−0.00593** | −0.1757 → −0.0882 |
| 2025-09-18 | 1.76760 | 1.76617 | −0.00143 | −0.1085 → −0.0532 |
| 2025-10-02 | 1.75316 | 1.75356 | +0.00040 | −0.0319 → −0.0326 |
| 2025-10-16 | 1.74630 | 1.74550 | −0.00080 | −0.0639 → −0.0439 |
| **CV mean** | **1.76182** | **1.75988** | **−0.00194** | −0.0950 → **−0.0545** |

OOF 1.76165 → 1.75973; после оптимального сдвига 1.75913 → 1.75889.
Уровень прогноза на тесте: 2.4958 → **2.4726**.

Adversarial AUC против теста: uncapped 0.9935/0.9912 → нормированные **1.0000/1.0000**.

## Вердикт и вывод

**KEEP**, но с явно зафиксированной оговоркой.

Что получилось: лучший одиночный конфиг Strategy 1. Смещение упало почти вдвое
(−0.095 → −0.055) при **улучшении**, а не ухудшении RMSLE — в отличие от усечения
до `L = 180`, где то же снижение смещения стоило +0.0061. Уровень прогноза на тесте
опустился на 0.023 ближе к якорю.

Что не получилось: adversarial AUC вырос до **1.0000**. Разбор показал, что это
не сдвиг распределения, а **квантование**: умножение на cutoff-специфичную константу
`365/avail` создаёт для каждого cutoff'а собственную сетку значений, по которой
классификатор безошибочно узнаёт cutoff. Диапазоны значений при этом как раз стали
сопоставимыми (в этом и была цель), и на самой метрике конфигурация выигрывает.

Так как отличить «безвредный отпечаток» от «вредной экстраполяции» локально нельзя,
в `S1-BEST` этот набор берётся с весом 0.45, ещё 0.45 отдано ненормированному
и 0.10 — усечённой модели с честным AUC 0.686.

## Конфиг прогона

```bash
python -m src.features --L 0 --min-history 90 --norm-long
python -m src.train --exp S1-E10 --L 0 --min-history 90 --train-blocks 1 --cutoffs all --norm-long
```
