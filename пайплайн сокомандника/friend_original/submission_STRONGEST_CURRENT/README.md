# `submission_STRONGEST_CURRENT` — production bundle

Чистый пакет лучшего сабмита проекта на 2026-08-20.

- Public LB: **1.6496571**
- wCV: **1.74751**
- Файл: `submission/submission_STRONGEST_CURRENT.csv`
- SHA256: `abc2218b1a3d55d41121b7b5a22db7e95ffd45283b42ff0006c5e6e731e04bda`
- Исходная ревизия: `a28a71fb2d0194052014c542f36d180dfe74bcf9` + зафиксированное рабочее дерево EXP-037

## Рецепт

Смешивание выполняется в пространстве `z = log1p(predict)`:

```text
0.10  S1-CAP
0.20  S1-UNC
0.25  S1-DIST
0.075 SEQ-01 (seed 42, depth clip 289)
0.075 SEQ-C289-S43
0.075 SEQ-C289-S44
0.075 ETX-01-S42-DCW
0.075 ETX-01-S43-DCW
0.075 ETX-01-S44-DCW
```

То же укрупнённо:

```text
0.10 CAP + 0.20 UNC + 0.25 DIST
+ 0.225 SEQ-AVG3(clip289) + 0.225 ETX-AVG3(DCW)
```

После смеси применяется один общий сдвиг до целевого `mean(log1p)=2.3293`, затем
`predict = expm1(max(z + delta, 0))`. Для сохранённых компонент фактический
`delta` равен примерно `-0.13536`.

## Быстрая проверка и пересборка

Из корня распакованного пакета:

```powershell
python -m pip install -r requirements-rebuild.txt
python pipeline/verify_package.py
python pipeline/build_submission.py
```

Пересобранный файл появится как
`submission/submission_STRONGEST_CURRENT_rebuilt.csv`. Скрипт сверит его с
оригиналом по строкам, значениям и SHA256. Raw data для этого пути не нужны.

## Состав

- `submission/` — отправленный CSV.
- `artifacts/predictions/` — ровно девять компонент `ztest_*.npy` и соответствующие
  `uid_*.npy`; это минимальный полный набор для независимой сверки выравнивания.
- `artifacts/models/` — все сохранённые production-checkpoint'ы, которые существуют:
  три ETX и два TCN.
- `artifacts/logs/` — только пять production-логов обучения сохранённых checkpoint'ов.
- `pipeline/` — автономная пересборка, исходники обучения/инференса и оригинальные
  production shell-скрипты.
- `docs/` — финальный отчёт, карточки происхождения компонент и компактные таблицы
  гейтов/LOFO.
- `MANIFEST.sha256` — контрольные суммы каждого файла пакета, кроме самого manifest.

## Какие веса отсутствуют и почему

Веса `S1-CAP`, `S1-UNC`, `S1-DIST` и TCN `SEQ-01` seed 42 исторически не
сохранялись соответствующими версиями pipeline. Их финальные тестовые прогнозы
включены полностью, поэтому итоговый CSV воспроизводится точно. Сохранённые веса
включены без исключений:

- `model_ETX-01-S42-TEST.pt`
- `model_ETX-01-S43-TEST.pt`
- `model_ETX-01-S44-TEST.pt`
- `model_SEQ-C289-S43-TEST.pt`
- `model_SEQ-C289-S44-TEST.pt`

Команды полного переобучения и реальная история запуска находятся в
`pipeline/TRAINING_HISTORY.md`.

## Что намеренно не включено

Raw data, кэши панелей, OOF всех исследовательских фолдов, провалившиеся модели,
полные/сырые альтернативы ETX и TCN, графики, временные файлы и несвязанные
эксперименты. Они не нужны ни для проверки отправленного файла, ни для его точной
пересборки из production-компонент.
