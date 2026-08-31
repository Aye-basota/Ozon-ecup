# Latest E-CUP submission — воспроизводимый pipeline bundle

Этот архив показывает, как был получен текущий `latest.csv` с public LB **1.64921756224069**.
Он предназначен прежде всего для аудита/воспроизведения товарищем, а не для повторного 20-часового обучения.

## 1. Что внутри

- `data/train.parquet` — исходные данные соревнования, которые использовались последующими табличными экспериментами.
- `data/sample_submit.csv` — формат submission.
- `friend_original/submission_STRONGEST_CURRENT/` — исходный самодостаточный пакет сильнейшего решения товарища: pipeline, TEST PyTorch weights, сохранённые TEST predictions, документация и исходный submission.
- `research_scripts/` — код наших последующих экспериментов поверх фиксированного решения товарища.
- `review_bundles/` — отчёты и submissions ключевых успешных стадий.
- `latest/components/` — три точных submission-компонента, из которых непосредственно собран текущий latest.
- `latest/latest.csv` — отправленный финальный submission.
- `latest/rebuild_latest.py` — полностью воспроизводит `latest.csv` из трёх компонентов без обучения.
- `provenance/PROJECT_STATE_AFTER_PHASE11.md` — общий аудит проекта и история Phase 12–14/best_bas.

## 2. Исходный STRONGEST_CURRENT

В `log1p` пространстве:

```
0.10 * S1-CAP
+ 0.20 * S1-UNC
+ 0.25 * S1-DIST
+ 0.225 * ETX-AVG3 @ DCW
+ 0.225 * SEQ-AVG3 @ depth-clip 289
```

Контрольные значения:

- public LB: `1.6496571`
- wCV: около `1.74751`
- mean TEST log1p: около `2.32932137`

Исходный пакет умеет точно пересобрать свой submission из уже сохранённых TEST predictions через `friend_original/submission_STRONGEST_CURRENT/pipeline/build_submission.py`.

## 3. Надстройки поверх STRONGEST_CURRENT

### Fixed-stack / Ridge

Основные файлы:

- `research_scripts/run_best_bas_fixedstack_14h_v2.py`
- `research_scripts/continue_fixedstack_combo_10h.py`
- `review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654.zip`

Здесь SEQ/ETX товарища фиксировались, а улучшалась в основном табличная часть через Ridge/meta, recency, greedy, p-band, hurdle и другие варианты.
Первый подтверждённый public gain: около `1.6492897556391737`.

### Final6h occurrence

Основной файл:

- `research_scripts/continue_best_bas_final6h.py`
- `review_bundles/final6h_REVIEW_BUNDLE_20260823_204823.zip`

Было обучено 8 occurrence-only LightGBM и построены meta/risk overlays поверх стабильного Ridge-stack.
Компонент `latest/components/occ_meta_B.csv` — именно Branch B из этой стадии.
Его известный public LB: `1.649261256742314`.

### Extra90 raw occurrence

Основной файл:

- `research_scripts/materialize_final6h_extra90m.py`
- `review_bundles/extra90_REVIEW_BUNDLE_20260823_222555.zip`

`latest/components/occ_raw_X3.csv` соответствует extra90 candidate 3: raw `occ_r10_fast` adaptive overlay поверх Ridge/greedy базы.
Известный public LB: `1.6492260257794873`.

## 4. Точный рецепт current latest

Финальный файл не обучает новую модель. Он смешивает **три уже обученных решения в `z = log1p(pred)`**:

```
z_latest = 0.12 * z_STRONGEST_CURRENT
         + 0.16 * z_final6h_B
         + 0.72 * z_extra90_3

predict = expm1(max(z_latest, 0))
```

Public LB полученного `latest.csv`: **1.64921756224069**.

Точные компоненты лежат:

- `latest/components/friend.csv`
- `latest/components/occ_meta_B.csv`
- `latest/components/occ_raw_X3.csv`

## 5. Как проверить latest

Из корня архива:

```bash
python latest/rebuild_latest.py
```

Скрипт создаст `latest/latest_rebuilt.csv` и проверит максимальное отличие в `log1p` от приложенного `latest/latest.csv`.
Ожидается отличие порядка машинной/CSV точности (`<=1e-10`).

## 6. Как посмотреть полный путь от raw data

1. Исходный pipeline товарища — `friend_original/submission_STRONGEST_CURRENT/pipeline/` и `docs/`.
2. Fixed-stack/Ridge experiments — соответствующие runners и combo10h review bundle.
3. Occurrence bank/meta overlay — `continue_best_bas_final6h.py` + final6h review bundle.
4. Extra90 raw-occurrence diversification — `materialize_final6h_extra90m.py` + extra90 review bundle.
5. Последнее смешивание — `latest/rebuild_latest.py`.

## 7. Что НЕ включено

В локальном проекте существовала рабочая папка `_best_bas_research` примерно с 80 fold NPZ, 15 TEST NPZ и ~9.7 GB feature cache. Эти 10+ GB у составителя данного архива недоступны и поэтому не включены.

Это означает:

- exact final `latest.csv` воспроизводится полностью из приложенных трёх компонентов;
- исходный `STRONGEST_CURRENT` воспроизводится из приложенного original bundle;
- код и отчёты всех успешных надстроек приложены;
- но повторить промежуточное обучение Ridge/occurrence **без пересчёта** только из этого архива нельзя, потому что полный `_best_bas_research` cache/checkpoints отсутствует.

Если нужен абсолютно полный forensic bundle со всеми 80+15 NPZ и локальными caches, его надо собрать непосредственно из локального `src/DL/best_bas/_best_bas_research/`.
