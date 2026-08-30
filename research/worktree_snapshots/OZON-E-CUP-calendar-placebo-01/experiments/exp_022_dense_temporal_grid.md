# exp_022 — STRATEGY_02B: dense temporal grid при равном объёме

- **Дата:** 2026-08-12
- **Автор:** A1
- **Коммит:** `34a2335` + рабочая реализация Variant B

## Гипотеза

Более плотная temporal grid содержит дополнительную информацию сама по себе,
даже когда общее число train-строк не растёт. Это дешёвый gate для тезиса dense
supervision в `STRATEGY_10`.

## Что изменено относительно базы

Только распределение train-строк: cutoff step `7→3`, а пользователи отбираются
target-free hash-долей 0.422. Features, target, folds, panel, модель и параметры
совпадают; train volume отличается не более чем на 1% в каждом fold.

## Результат

- Capacity curve `{150,200,250,300,450}` выбрала baseline 300 и dense 200 rounds
  по минимуму последнего fold; dense minimum wCV на 150 также хуже baseline.
- wCV: **1.752339** против **1.751076**, Δ **+0.001263**.
- Fold Δ dense−base: `+0.000657 / +0.001021 / +0.001719 / +0.001170` — 0/4.
- AUC(y>0): `0.843101→0.842684`, Δ `−0.000416`.
- `rec_buy 15–60`: Δ RMSLE `+0.001557`, Δ AUC `−0.001208`.
- 2–15 purchase days: Δ RMSLE `+0.001512`, Δ AUC `−0.000935`.
- Train rows dense−base по fold: `+0.97% / −0.72% / −0.11% / +0.41%`.
- Seed 42 достаточен: Δ около 9σ seed-difference и один знак на 4/4; avg3 не нужен.
- LB: не отправляли.

Полный результат: `research/strategies/results/STRATEGY_02/variant_B.md`.

## Вердикт и вывод

**FAIL.** Плотность не дала signal ни по score, ни по активности, ни в проблемных
сегментах. Dense-supervision premise для HDN/TCN существенно ослаблен; модели
`STRATEGY_10` автоматически не запускались.

## Конфиг прогона

```text
direct, 227 features S1-E10, L=None, norm_long=True, min_history=90,
panel_blocks=3, train_blocks=1, VAL_FOLDS_S1, seed=config.SEED=42,
baseline: step=7,row_frac=1.0,rounds=300;
dense: step=3,row_frac=0.422,rounds=200;
capacity curve=150/200/250/300/450, hash=(user_id*2654435761)%1000<422
```
