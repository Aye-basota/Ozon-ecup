# exp_060 — LEVEL_MINUS_006

## Catalogue metadata

- **Catalogue ID:** `team_a_current__exp_060_level_minus_006`
- **Namespace:** `team_a_current`
- **Experiment ID:** `exp_060_level_minus_006`
- **Original source:** `experiments/exp_060_level_minus_006.md`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** experiment card
- **Model:** BTYD, blend, calibration diagnostic
- **Features:** freshness/conditional features, 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** CV по фолдам: не запускался; диагностический artifact-only experiment.
- **Known score:** Unknown / not recoverable from repository history
- **Seed:** Обучение и seed не используются. Единственный параметр — заранее фиксированный
- **Postprocessing:** `SHIFT = 0.06` в log-space.
- **Submission:** Output: `submissions/submission_LEVEL_MINUS_006.csv`
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when required data/artifacts are present; otherwise PARTIAL as stated in the card

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_060 — LEVEL_MINUS_006

- **Дата:** 2026-08-24
- **Автор:** A1
- **Коммит:** `a28a71fb2d0194052014c542f36d180dfe74bcf9` + текущее рабочее дерево

## Фиксированный протокол

```text
BASE = STRONGEST_CURRENT
ONLY CHANGE = z - 0.06
TRAINING = NONE
PURPOSE = public production-level diagnostic
```

Offset `−0.06` был зафиксирован до любого результата LB. Никаких альтернативных
offset, ZERO2D, FRESH, BTYD, segment corrections, новых blend weights, иной
калибровки или normalization обратно к `mean(z)=2.3293` не применялось.

## Гипотеза

Production target начинается сразу после гарантированной activity-selection
области, поэтому test population может иметь больше lapse/zero behavior, чем
CV-панель. Один крупный фиксированный global shift проверяет, завышен ли текущий
production GMV level, без обучения и последовательного LB-tuning.

## Что изменено относительно базы

Единственное изменение: `pred = max(expm1(log1p(pred_STRONGEST_CURRENT) - 0.06), 0)`.

## Source verification

- Source: `submissions/submission_STRONGEST_CURRENT.csv`
- Source SHA256: `abc2218b1a3d55d41121b7b5a22db7e95ffd45283b42ff0006c5e6e731e04bda`
- SHA совпадает с `experiments/submissions.csv` и manifest production bundle
  `deliverables/submission_STRONGEST_CURRENT_artifacts_2026-08-20.zip`.
- Schema/order: exact match с `data/raw/sample_submit.csv`; 250,000 строк,
  missing/duplicates нет.

## Diagnostics

| metric | STRONGEST_CURRENT | LEVEL_MINUS_006 |
|---|---:|---:|
| mean `log1p(pred)` | 2.329321370 | 2.269498656 |
| mean `pred` | 36.727126165 | 34.530242707 |
| median | 7.390898 | 6.902250 |
| zero share | 0.1092% | 0.5404% |
| p01 | 0.09759695 | 0.03367795 |
| p05 | 0.25340670 | 0.18041375 |
| p10 | 0.44038910 | 0.35650730 |
| p50 | 7.39089800 | 6.90225000 |
| p90 | 96.18385140 | 90.52430430 |
| p95 | 165.87470900 | 156.15668250 |
| p99 | 423.51441561 | 398.79262028 |
| max | 2641.376346 | 2487.496327 |

До обязательного nonnegative floor средний requested level равен
`2.269321369901`, то есть сдвиг `−0.060000000000` с ошибкой `9.3e-15`.
У фактически сериализованного submission сдвиг `−0.059822713806`: 1,351 строка
(0.5404%) после shift попала под `pred >= 0`; оставшиеся `1.77e-4` — ожидаемый
эффект floor и шестизначной CSV-сериализации, а не normalization.

## Submission verification

- Output: `submissions/submission_LEVEL_MINUS_006.csv`
- SHA256: `1b40f67d119d0dcc4798a4da5612707b8d44f1dfe3fa20b28c28b836c2c8c0f1`
- 250,000 строк; exact source/sample schema и row order.
- NaN/inf/negative/missing/duplicates: 0; LF-only, final newline сохранён.
- Создан ровно один submission; на сайт не отправлялся.
- Полная машинно-читаемая проверка: `artifacts/LEVEL_MINUS_006_EXP060/diagnostics.json`.

## Заранее зафиксированная интерпретация будущего LB

- Если заметно лучше `STRONGEST_CURRENT`: evidence, что production global GMV
  level ниже текущего CV-calibrated level.
- Если хуже примерно на ожидаемую квадратичную величину: global calibration,
  вероятно, уже близка к правильной.
- Если улучшение очень большое, порядка `0.001+`: сильный сигнал фундаментального
  CV→test level mismatch; нужен отдельный анализ, не немедленный tuning offset.
- Если разница около public noise: ничего не подбирать по этому одному score.

## Результат

- CV по фолдам: не запускался; диагностический artifact-only experiment.
- CV mean: не применимо.
- LB: не отправляли.

## Вердикт и вывод

**PREPARED, NOT UPLOADED.** Фиксированный production-level probe собран и
проверен; дальнейшие offset-файлы и подбор по LB запрещены протоколом.

## Конфиг прогона

```text
node research/strategies/results/LEVEL_MINUS_006_EXP060/build_level_minus_006.mjs
```

Обучение и seed не используются. Единственный параметр — заранее фиксированный
`SHIFT = 0.06` в log-space.
