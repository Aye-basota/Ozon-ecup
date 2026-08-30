:# research/

Исследовательские документы, EDA и стратегии.  
Правило чтения: **фактический код, карточки экспериментов и `log.csv` имеют приоритет** над старыми research-файлами. Статус каждого документа указан ниже.

## Статусы документов

- **authoritative** — актуален и отражает текущее состояние.
- **partially superseded** — часть выводов устарела, часть остаётся верной.
- **superseded** — ключевые выводы опровергнуты более поздними экспериментами.
- **background** — исторический/контекстный документ.

## EDA и общие находки

| Документ | Статус | Примечание |
|----------|--------|------------|
| `eda_findings.md` | partially superseded | Основные находки верны, но §3.2 (3-блочная train-панель) и §9.2 опровергнуты `S1-E01`; §7.4 — bias растёт с gap, но slope не воспроизвёлся как критерий (`exp_019`). |
| `compute_profile.md` | authoritative | Профили скорости, железо, ограничения. |
| `strategy_1.md` | partially superseded | План Strategy 1; часть пунктов выполнена, часть отвергнута (`S1-E04`, rounds≥600 и др.). |
| `strategy_1_results.md` | partially superseded | Результаты плана; смотреть вместе с `experiments/exp_017.md`, `exp_018.md`. |
| `strategy_2.md` | partially superseded | Структурная модель S2 проиграна (LB 1.6619); сезонная секция перенесена в `STRATEGY_12`. |
| `strategy_comparison.md` | partially superseded | Общие принципы верны, но §3 «фундамент L=180» неверен; оценки переноса `R10` отменены (`exp_016`). |
| `strategy_NN_1.md` | partially superseded | HDN-план; после `exp_014`/`exp_025`/`exp_026` ставка сузилась до энкодера; оценки LB-выигрыша завышены. |
| `strategy_NN_2.md` | partially superseded | ETX-план; `G1` (личное время) опровергнуто на табличном уровне (`exp_021`); оценки переноса завышены. |
| `strategy_NN_report.md` | partially superseded | Диагностика `N1`–`N16`; замеры не логировались как эксперименты. `N12` ослаб в 2.7× на боевой базе. Использовать как контекст, не как факты. |

## Стратегии

См. `research/strategies/STRATEGIES_INDEX.md` для полного shortlist и `research/strategies/STRATEGY_*.md`.

| Стратегия | Статус | Последний эксперимент |
|-----------|--------|----------------------|
| S01 gap-axis validation | REJECT | `exp_019` / `S1-GAPAXIS` |
| S02A train_blocks=0 | REJECT | `exp_020` / `S1-SAMPLE-A` |
| S02B dense step 3 matched | REJECT | `exp_022` / `S1-SAMPLE-B` |
| S03 count/value heads | REJECT | `exp_024` / `MHZ-FULL` |
| S04 intensive full calendar | OPEN | `exp_028` отменил полную версию; условная — не проверена |
| S05 capacity + seed average | CONTINUE | `exp_017`, `exp_018` (GBDT); `exp_026` (TCN) |
| S06 multi-target ensemble | OPEN | не проверена |
| S07 CatBoost diversity | OPEN | не проверена |
| S08 personal time features | REJECT | `exp_021` / `PT-*` |
| S09 propensity weighting | OPEN | зависит от S01, не запускалась |
| S10 HDN / TCN encoder | CONTINUE | `exp_025`–`exp_027`, `exp_029` |
| S11 orthogonal weak learners | OPEN | не проверена |
| S12 seasonal segment reallocation | OPEN | не проверена |
| S13 ETX event transformer | DEPRIORITIZED | гейты S08/S10 не пройдены |

## Результаты и артефакты

- `research/strategies/results/` — таблицы, графики и вспомогательные скрипты по стратегиям.
- `research/rmsle_diagnostics/` — разложение остаточной ошибки; актуально, но помнить, что любая монотонная/посегментная постобработка закрыта.
- `artifacts/` — OOF, отчёты, чекпойнты; не коммитятся.
