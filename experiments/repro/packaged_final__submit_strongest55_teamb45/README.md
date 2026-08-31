# Воспроизведение `SUBMIT_STRONGEST55_TEAMB45.csv`

## Catalogue metadata

- **Catalogue ID:** `packaged_final__submit_strongest55_teamb45`
- **Namespace:** `packaged_final`
- **Experiment ID:** `submit_strongest55_teamb45`
- **Original source:** `reproducibility/SUBMIT_STRONGEST55_TEAMB45/README.md`
- **Source ref:** `origin/team-a final/research package`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** packaged final candidate; exact outer blend, frozen upstream inputs
- **Model:** LightGBM, CatBoost, XGBoost, sequence model
- **Features:** freshness/conditional features, history-depth features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Unknown / not recoverable from repository history
- **Known score:** Unknown / not recoverable from repository history
- **Seed:** моделей и checkpoint `SEQ-01` seed 42; FP16 ETX зависит от CUDA execution path.
- **Postprocessing:** финальный retrained CSV, затем сравнивает каждый уровень с production:
- **Submission:** delivery/submission_STRONGEST_CURRENT_training_bundle_v2/pipeline/data/raw
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL from frozen inputs; raw retraining has the limitations documented by the package
- **Notes:** Reported leaderboard results and forecasts are kept distinct exactly as in the preserved source.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Воспроизведение `SUBMIT_STRONGEST55_TEAMB45.csv`

Единая команда из корня ветки:

```powershell
python scripts/reproduce_final.py --solution SUBMIT_STRONGEST55_TEAMB45 --from-precomputed
```

Подтверждённого фактического LB именно для этого 55/45 файла в найденных
материалах нет. `1.64823` с диапазоном `1.64818…1.64834` — сохранённый прогноз
geometry-отчёта, а не измеренный score. LB внутреннего `STRONGEST_CURRENT` равен
`1.6496571902356205`.

Эта папка содержит код обеих обучающих веток, точную сборку финального
submission, зафиксированные production-векторы для побитовой проверки,
checkpoint smoke-тесты и результаты фактического повторного обучения табличных
моделей.

## Что именно собирается

1. `submission_STRONGEST_CURRENT.csv`:
   `0.10 CAP + 0.20 UNC + 0.25 DIST + 0.225 SEQ-AVG3 + 0.225 ETX-AVG3`
   в `log1p`, затем общий уровень `mean(log1p)=2.3293`.
2. `team-b-final/final_classic_ml.csv`: исходный внутренний табличный ансамбль с
   `CURRENT_LOG_SCALE=1.12`.
3. Финал: `0.55 STRONGEST_CURRENT + 0.45 team-b-final` в `log1p`; перед смесью
   уровень team-b выравнивается по среднему `log1p` anchor-компонента.

Формулы, веса, seed и гиперпараметры моделей не менялись. Изменены только
runtime-пути: они задаются переменными окружения/CLI вместо прежних абсолютных
или соседних директорий.

## Быстрая побитовая пересборка

```powershell
python -m venv .venv-rebuild
.\.venv-rebuild\Scripts\python.exe -m pip install -r requirements-rebuild.txt
.\.venv-rebuild\Scripts\python.exe verify.py
.\.venv-rebuild\Scripts\python.exe build_submit.py
```

Результат: `outputs/SUBMIT_STRONGEST55_TEAMB45.csv`.

Ожидаемый SHA-256:

```text
1ce85203e3069363e3d2ba425078213d1a723a895e3c684573a6c1b998a14fb4
```

`build_submit.py` по умолчанию завершится ошибкой при любом отличии от этого
SHA. Для исследовательской смеси из заново обученных векторов предназначен
явный флаг `--allow-nonreference`.

## Данные

Обе ветки используют один и тот же `train.parquet`:

```text
SHA-256 5f3aa90992652b8a4f0f398e735a3ba11c2ea6ccf9e8fb1d236436e9a49167c0
```

Большие raw/cache-файлы не дублируются в этой папке. Их имена, размеры и SHA
зафиксированы в `EXTERNAL_DATA_MANIFEST.json`. В текущем репозитории canonical
raw лежит в:

```text
delivery/submission_STRONGEST_CURRENT_training_bundle_v2/pipeline/data/raw
```

Полная проверка raw, SEQ/ETX cache и team-b feature cache:

```powershell
python verify.py --full-data
```

Кэш team-b необязателен: `team_b/src/team_features.py` строит отсутствующие
файлы из raw. Production-кэши нужны только для быстрого повторения уже
проверенного запуска. SEQ/ETX-кэши также можно построить командами `build` ниже.

## Окружения обучения

Исторические ветки проверялись в разных зафиксированных окружениях, поэтому для
честной сверки нужны две venv:

```powershell
py -3.13 -m venv .venv-strongest
.\.venv-strongest\Scripts\python.exe -m pip install -r requirements-strongest.txt

py -3.11 -m venv .venv-team-b
.\.venv-team-b\Scripts\python.exe -m pip install -r requirements-team-b.txt
```

## Полный повторный запуск табличных моделей

Команда реально обучает `S1-UNC`, `S1-CAP`, `S1-DIST`, все current-модели и
пять team-b-компонентов (LightGBM/XGBoost/CatBoost), строит оба anchor CSV и
финальный retrained CSV, затем сравнивает каждый уровень с production:

```powershell
.\.venv-team-b\Scripts\python.exe scripts\run_tabular.py `
  --strongest-python .\.venv-strongest\Scripts\python.exe `
  --team-python .\.venv-team-b\Scripts\python.exe
```

По умолчанию заново обученные веса сохраняются в `work/tabular/*/models`.
Флаг `--skip-model-pickle` отключает только это сохранение и не пропускает
обучение. Проверенный запуск выполнялся с этим флагом для экономии диска; все
модели были обучены и использованы для inference.

Фактические результаты находятся в `audits/TABULAR_REPRO_AUDIT.json` и кратко
описаны в `AUDIT_RESULTS.md`.

## Ограниченный smoke-тест SEQ/ETX

Полный DL training намеренно не входит в обязательный аудит: historical ETX
campaign занимал около 4.7 часа. Следующая команда запускает 114 unit-тестов и
инференс пяти сохранённых checkpoint'ов на фиксированной подвыборке, контролируя
30-минутный бюджет:

```powershell
.\.venv-strongest\Scripts\python.exe scripts\smoke_dl.py `
  --rows 512 `
  --max-gpu-minutes 30
```

Проверенный прогон занял 21.42 секунды суммарного wall-time. Отчёт:
`audits/DL_SMOKE_AUDIT.json`.

## Команды полного DL training (не запускались в аудите)

Из `strongest/`, задав пути к raw, processed cache и выходным artifacts:

```powershell
$env:ECUP_RAW_DATA_DIR = "D:\ecup\raw"
$env:ECUP_PROCESSED_DIR = "D:\ecup\processed"
$env:ECUP_ARTIFACTS_DIR = "D:\ecup\artifacts"

python -m src.seq build
python -m src.etx build

python -m src.seq predict --exp SEQ-01 --seed 42 --epochs 4 --depth-clip 289
python -m src.seq predict --exp SEQ-C289-S43 --seed 43 --epochs 4 --depth-clip 289
python -m src.seq predict --exp SEQ-C289-S44 --seed 44 --epochs 4 --depth-clip 289

python -m src.etx predict --exp ETX-01-S42 --seed 42 --depth-clip 289
python -m src.etx predict --exp ETX-01-S43 --seed 43 --depth-clip 289
python -m src.etx predict --exp ETX-01-S44 --seed 44 --depth-clip 289

python original\ETX2\depth_fix.py --mode test --ckpt ETX-01-S42-TEST --depth-clip 289 --dow-shift -1 --exp ETX-01-S42-DCW
python original\ETX2\depth_fix.py --mode test --ckpt ETX-01-S43-TEST --depth-clip 289 --dow-shift -1 --exp ETX-01-S43-DCW
python original\ETX2\depth_fix.py --mode test --ckpt ETX-01-S44-TEST --depth-clip 289 --dow-shift -1 --exp ETX-01-S44-DCW
```

Linux/CUDA может добавить `--compile` для seed 43/44, как в сохранённых
production shell-скриптах `strongest/original/`. Полная фактическая история
команд находится в `strongest/TRAINING_HISTORY.md`.

## Структура

- `strongest/` — автономная копия TAB/SEQ/ETX training и inference-кода;
- `team_b/` — автономная копия полного `team-b-final` и trainer-обёртка;
- `frozen/` — девять production-векторов и два source CSV для exact rebuild;
- `reference/` — эталонный итоговый CSV и `sample_submit.csv`;
- `scripts/` — табличный аудит, STRONGEST builder, DL smoke и data manifest;
- `audits/` — сохранённые результаты выполненных прогонов;
- `work/`, `outputs/` — генерируемые файлы, исключённые из Git;
- `MANIFEST.sha256` — контрольные суммы всех поставляемых файлов.

## Честная граница точности

Байтовая пересборка финального CSV гарантируется из зафиксированных production-
векторов. Полное побитовое переобучение historical `STRONGEST_CURRENT` заявлять
нельзя: в исходном проекте не были сохранены booster weights трёх табличных
моделей и checkpoint `SEQ-01` seed 42; FP16 ETX зависит от CUDA execution path.
Код, raw, остальные checkpoint'ы и точные production-векторы включены, а
фактические расхождения fresh training измерены, а не скрыты.
