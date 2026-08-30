# exp_059 — SEQ65_TEMPORAL_HEAVY

- **Дата:** 2026-08-24
- **Автор:** A1
- **Коммит:** `a28a71f` + рабочее дерево

## Гипотеза

Production/test regime может сильнее зависеть от temporal representation, чем clean CV,
а `STRONGEST_CURRENT` может недовзвешивать совместный ETX/SEQ slot. Проверяется один
заранее фиксированный крупный сдвиг representation balance без обучения и без подбора веса.

## Что изменено относительно базы

Только суммарный вес sequence-slot: `0.45 -> 0.65`; ETX:SEQ остаётся `50:50`, CAP — `0.10`.

```text
PURPOSE: large representation-balance LB probe
BASE: STRONGEST_CURRENT
ONLY STRUCTURAL CHANGE: sequence total weight 0.45 -> 0.65
LEVEL: held fixed at 2.3293
```

Фиксированный рецепт в log-space:

```text
0.10 CAP + 0.10 UNC + 0.15 DIST + 0.325 ETX-AVG3 + 0.325 SEQ-AVG3
```

Использованы ровно девять production arrays из immutable bundle `STRONGEST_CURRENT`:
CAP/UNC/DIST, три `SEQ-AVG3 @ clip289` и три `ETX-AVG3 @ DCW`. SHA256 всех `ztest`
совпали с manifest, все девять `uid` имеют общий SHA256 `50e5ba9…`; реконструкция
чемпиона из этих компонент совпала с его CSV с `max |delta z| = 4.97e-07`.

## Результат

- Информационный fixed-recipe OOF: wCV `1.747272` против `1.747510` у
  `STRONGEST_CURRENT`, delta **−0.000238**, лучше на **4/4** фолдах
  (`−0.000345 / −0.000230 / −0.000272 / −0.000209`). Recipe по OOF не менялся.
- Production delta после level policy: `Var=0.00116180`, Pearson `0.99978233`,
  mean до нормализации `−0.01063037`, mean после `+0.00000117`,
  `max |delta z|=0.43498365`; q01/q50/q99 `−0.094284 / +0.000901 / +0.082811`.
- Level policy в точности как у чемпиона: shift `−0.12473150` доводит среднее z
  до floor до `2.329300000`; после production floor `z=max(z,0)` и CSV-roundtrip
  фактический `mean(log1p)=2.329322536` (270 нулей; у чемпиона тот же эффект).
- Создан `submissions/submission_SEQ65_TEMPORAL_HEAVY.csv`, 250 000 строк,
  schema/order/unique user PASS, NaN/inf/negative `0/0/0`, SHA256
  `33c6a4e70dbc0d061508c8179e3b820ffa00829d134d802fc0767ac3f4b69248`.
- LB: не отправляли.

Полная машинно-читаемая проверка: `research/strategies/results/SEQ65_TEMPORAL_HEAVY/diagnostics.json`.

## Вердикт и вывод

**READY FOR MANUAL LB PROBE.** Gross implementation/local blocker отсутствует;
информационный OOF слегка лучше, но не использовался для изменения recipe.
Создан ровно один кандидат; соседние веса не строились, файл автоматически не отправлялся.

## Конфиг прогона

Без обучения и без seed. Воспроизведение одной командой на чистом наборе production
arrays: `python research/strategies/results/SEQ65_TEMPORAL_HEAVY/build_submission.py`.

