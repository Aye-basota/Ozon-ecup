# exp_020 — STRATEGY_02A: `train_blocks=0`

- **Дата:** 2026-08-12
- **Автор:** A1
- **Коммит:** `27098ea`

## Гипотеза

Снятие последнего фильтра train-панели добавит 10–20% строк и улучшит direct-модель.
После STRATEGY_05 capacity выбирается отдельно в каждом плече, а малая дельта
подтверждается средним трёх seed.

## Что изменено относительно базы

Только `train_blocks: 1 → 0`; features, target, folds, loss и остальные параметры
совпадают с `S1-E10`.

## Результат

- Capacity curve `150/200/250/300/450`: база выбрала 300, Variant A — 200 rounds.
- Seed 42: Δ = −0.000049 wCV, 3/4 fold, но последний хуже на +0.000134.
- Avg3 (42/43/44): база **1.750456**, Variant A **1.750569**,
  Δ = **+0.000113**, 2/4 fold, последний хуже на +0.000194.
- Строк последнего fold: 4,955,174 → 5,412,520 (**+9.23%**).
- `Var(z_A-z_base)=0.00279` после avg3; adversarial AUC 0.9999935 → 0.9999967.
- LB: не отправляли.

Полные таблицы: `research/strategies/results/STRATEGY_02/`; OOF/reports:
`artifacts/SAMPLE-TB*`, сводка `artifacts/sample_design_S1-SAMPLE-A-FINAL.json`.

## Вердикт и вывод

**REJECT.** После честной capacity и avg3 вариант хуже, выигрывает лишь два fold
и проигрывает 10-16. Дополнительные пользователи не дают полезного сигнала;
`train_blocks=1` остаётся в winning pipeline.

## Конфиг прогона

```text
direct, L=None, norm_long=True, min_history=90, step=7, train_blocks={1,0}
panel_blocks=3, cutoffs=all, rounds curve={150,200,250,300,450}
VAL_FOLDS_S1, seeds={config.SEED=42,43,44}, log-space avg3
```
