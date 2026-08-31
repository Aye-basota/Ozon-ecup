# exp_003 — S1-E02: плотная сетка cutoff'ов

## Catalogue metadata

- **Catalogue ID:** `team_a_current__exp_003_s1_e02_dense_cutoffs`
- **Namespace:** `team_a_current`
- **Experiment ID:** `exp_003_s1_e02_dense_cutoffs`
- **Original source:** `experiments/exp_003_s1_e02_dense_cutoffs.md`
- **Source ref:** `working tree`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** experiment card
- **Model:** LightGBM
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** `cutoffs: recent3 → all` (18–24 cutoff'а в зависимости от фолда вместо 3).
- **Known score:** Ожидание стратегии: −0.005…−0.012 RMSLE.
- **Seed:** LightGBM как в B0, 600 раундов, seed 42
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when required data/artifacts are present; otherwise PARTIAL as stated in the card

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_003 — S1-E02: плотная сетка cutoff'ов

- **Дата:** 2026-08-10
- **Автор:** A1 (Strategy 1)
- **Ветка:** `team-a-strategy-1-impl`

## Гипотеза (из `strategy_1.md`, Эксперимент 4)

Каждый cutoff — это независимая реализация связки «признаки → таргет» для тех же
пользователей. Плотная сетка (шаг 7 дней по всему чистому коридору
2025-04-03 .. 2025-10-16) даёт и больше данных, и усреднение по фазам года,
то есть модель перестаёт заучивать сезонность конкретных трёх месяцев.
Ожидание стратегии: −0.005…−0.012 RMSLE.

## Что изменено относительно базы

`cutoffs: recent3 → all` (18–24 cutoff'а в зависимости от фолда вместо 3).
Всё остальное — как в B0.

## Результат

| фолд | B0 (3 cutoff'а) | E02 (все, шаг 7) | Δ | bias B0 → E02 |
|------|-----------------|------------------|---|----------------|
| 2025-09-04 | 1.78975 | 1.78022 | **−0.00953** | −0.1986 → −0.1757 |
| 2025-09-18 | 1.77428 | 1.76760 | **−0.00668** | −0.1084 → −0.1085 |
| 2025-10-02 | 1.75873 | 1.75316 | **−0.00557** | −0.0285 → −0.0319 |
| 2025-10-16 | 1.75240 | 1.74630 | **−0.00610** | −0.0734 → −0.0639 |
| **CV mean** | **1.76879** | **1.76182** | **−0.00697** | |

- CV std 0.01448 → 0.01311
- OOF 1.76861 → 1.76165, bias −0.1022 → −0.0950
- обучающая выборка: 0.63 млн → 3.7–5.0 млн строк
- runtime 157 с → 733 с

## Вердикт и вывод

**KEEP.** Улучшение −0.00697 при пороге значимости 0.005, причём на **4 фолдах из 4**
и с уменьшением разброса между фолдами. Ровно в предсказанном стратегией диапазоне.

Это самый крупный подтверждённый прирост в Strategy 1 из «структурных» изменений
и он же — самый дешёвый по риску: никаких новых признаков, только другое устройство
обучающей выборки.

Цена: обучение в 4.7 раза дольше.

## Конфиг прогона

```
L=None, min_history=90, cutoffs=all, step=7, train_blocks=1, panel_blocks=3
LightGBM как в B0, 600 раундов, seed 42
```

```bash
python -m src.train --exp S1-E02 --L 0 --min-history 90 --train-blocks 1 --cutoffs all
```
