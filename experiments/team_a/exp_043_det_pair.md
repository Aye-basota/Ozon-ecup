# exp_043 — DET-PAIR: deterministic continuation SEQ-D3A

- **Дата:** 2026-08-22
- **Автор:** A1
- **Коммит:** `a28a71f` + рабочее дерево
- **Код:** `src/det_pair.py`, `src/test_det_pair.py`; `src/seq.py`,
  `src/validation.py`, `src/config.py` не изменялись
- **Вычисления:** локальная RTX 4060 Ti, два отдельных процесса, 46.0 мин
- **Leaderboard:** не отправляли; submission не создавался

## Гипотеза

`exp_038` измерил цену двух якобы одинаковых запусков TCN на fold 10-16:
`|ΔRMSLE_cal| = 0.00033`, `Var(Δz) = 0.02041`. Известный механизм — три CPU
worker'а кладут готовые батчи в общую очередь, поэтому фактический порядок шагов
SGD зависит от гонки и эффекты порядка `5e-4..1e-3` нельзя считать строго
парными.

Проверяем, можно ли воспроизвести **тот способ старта, который нужен
FT-FRESH-ENC** — загрузка готового model checkpoint плюс НОВЫЙ optimizer — и
получить одинаковую neural continuation при фиксированных batch/index plan и
execution policy. Сам FT-FRESH-ENC, conditional loss и EXTRA-данные в этом
эксперименте не реализуются.

## Точный starting artifact

`C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\model_SEQ-D3A-S42-V1016.pt`

- имя: `SEQ-D3A-S42-V1016`;
- fold: `2025-10-16`;
- file SHA256:
  `dc48c442a593139c1c3078dff66b66c84267062706b98552ac40944c6f6adc92`;
- canonical model-state SHA256:
  `cce84601c5df3048d98e14d85e88f111cfe3eb2cf91d43dbf70d5547631efb55`;
- checkpoint содержит model state + cfg, но не optimizer state;
- cfg checkpoint: TCN hidden 64, 8 блоков, kernel 3, dropout 0.10, 17 каналов,
  окно 365, `depth_aug=0.5`, grid `90/120/150/180/220/254/289`, batch 1024,
  chunk 256, AdamW `lr=3e-3`, `wd=1e-2`, betas `(0.9, 0.98)`, bf16, eager,
  seed 42 из `src/config.py`.

Выбран именно подтверждённый `SEQ-D3A`, потому что `exp_030c` подтвердил приём
на 3 сидах, а `exp_032b` предписывает будущему FT-FRESH перенести supervision
внутрь одноголового `SEQ-D3A`. Старт — `load_state_dict(checkpoint)` и новый
AdamW в каждом процессе. Отсутствие старого optimizer state не является
блокером: начальный canonical optimizer SHA256 в обоих repeats одинаков —
`4d18202a9a066a49b55298890d38de0281669a56fe214d8ff07004c0a3497f5e`.

## Что изменено относительно базы

Архитектура, loss `MSE(z30)`, признаки, depth curriculum, train/validation
панели, optimizer family, schedule formula, bf16 и TF32 не менялись. Изменена
только execution policy и способ старта: продолжение из checkpoint с новым
optimizer вместо обучения с нуля.

## Materialized plan и конфигурация продолжения

Plan сохранён **один раз**:

- `artifacts/DET_PAIR/materialized_plan.npz`;
- file SHA256:
  `24c37f06cb783854911b9ee65be07c1d1982261cd95b64e7de10a4b35dad0748`;
- plan ID:
  `a47dedac5cb002641c8d5ffc96b2a778561fba94dacfbc2c4d7a3996d5a43dbb`;
- 24 train cutoff'а `2025-04-03..2025-09-11`, правило `T+30<=V`;
- train-панель 1-блочная, 4 955 174 строки;
- один полный continuation epoch, 4 842 optimizer steps;
- каждый train index встречается ровно один раз;
- материализованы batch groups, row indices, per-batch augmentation seed и
  float64 LR каждого шага;
- validation: штатная 3-блочная панель fold 10-16, 197 379 строк; сохранённый
  порядок `user_id/row/y` побитово совпадает с `oof_SEQ-D3A-S42-V1016.npz`;
- snapshots: шаги `0, 1, 100, 1000, 4842`.

Продолжение использует прежний main MSE и production LR `3e-3` с тем же warmup
300 + cosine formula, но schedule начинается заново вместе с новым optimizer.
Это намеренно сильный stress-test детерминизма, а не выбор LR или loss будущего
FT-FRESH.

## Execution configuration

- два repeats — отдельные Python-процессы, последовательно;
- Python/NumPy/Torch/CUDA RNG = `SEED=42`; `PYTHONHASHSEED=42`;
- `workers=1`, один producer thread, строгая проверка номера каждого batch;
- `cudnn.benchmark=False`, `cudnn.deterministic=True`;
- `torch.use_deterministic_algorithms(True)`, debug mode `error`;
- `CUBLAS_WORKSPACE_CONFIG=:4096:8` до инициализации CUDA;
- AMP: `torch.autocast(cuda, bfloat16)` — как production;
- TF32 matmul/cuDNN: включён — как production;
- `torch.compile=False` — как у исходного локального seed-42 checkpoint;
- PyTorch `2.11.0+cu126`, CUDA runtime 12.6, cuDNN 91002;
- GPU: NVIDIA GeForce RTX 4060 Ti;
- model/optimizer/RNG snapshots и validation вызываются в одинаковых точках.

## Отличия от production execution

| политика | production `src/seq.py` | DET-PAIR | причина |
|---|---|---|---|
| CPU workers | 3 | **1** | убрать race общей очереди |
| batch order | новый `_plan()` на эпоху, realized order зависит от queue | **один сохранённый plan** | строгая парность |
| cuDNN benchmark | `True` | **`False`** | фиксировать алгоритм |
| deterministic algorithms | не включены | **включены** | запрет недетерминированных CUDA kernels |
| cuBLAS workspace | не зафиксирован | **`:4096:8`** | deterministic GEMM |
| snapshot barriers | только конец эпохи/curve | **0/1/100/1000/4842** | локализовать расхождение |
| старт | model с нуля + новый AdamW | **checkpoint + новый AdamW** | реальный старт будущего FT-FRESH |
| бюджет DET | 4 эпохи исходного train | **1 continuation epoch** | проверить весь полный batch plan без FT |

Eager/bf16/TF32, архитектура, MSE, признаки, панели и validation order не
ухудшались ради детерминизма. `compile=False` не является новой уступкой: так
был обучен starting artifact. FT-FRESH-ENC не начат.

## Результат

RMSLE в первой строке — проектный показатель fold: RMSLE после оптимального
лог-сдвига через неизменённый `src.validation.calibrate`. Для полноты сохранён и
raw RMSLE до сдвига.

| repeat | steps | train MSE | raw RMSLE | RMSLE_cal | offset | SHA256 raw float32 `z` |
|---|---:|---:|---:|---:|---:|---|
| run 1 | 4 842 | 3.151584586 | 1.746488314 | **1.745829867810** | +0.047954583 | `ec1a0773f46965176e2f2a3748db9604a3d15d58b6b02b062057a6a1b9963f89` |
| run 2 | 4 842 | 3.151584586 | 1.746488314 | **1.745829867810** | +0.047954583 | `ec1a0773f46965176e2f2a3748db9604a3d15d58b6b02b062057a6a1b9963f89` |

Raw validation arrays:

- `artifacts/DET_PAIR/run1/z_raw.npy`;
- `artifacts/DET_PAIR/run2/z_raw.npy`;
- `.npy` file SHA256 у обоих:
  `3dfc264c78cdc905253f6a322dc3cf778228021fcf5b6192703b54e387ae60d5`.

### Парные метрики

| метрика | факт | gate |
|---|---:|---:|
| RMSLE run 1 | **1.745829867810** | — |
| RMSLE run 2 | **1.745829867810** | — |
| `abs(delta RMSLE)` | **0.0** | `<=1e-4` |
| `Var(z1-z2)` | **0.0** | `<=1e-5` |
| `max abs(z1-z2)` | **0.0** | `<=1e-3` |
| Pearson `corr(z1,z2)` | **1.000000000** | `>=0.999999` |
| prediction hashes | **одинаковы** | предпочтительно одинаковы |

Третий repeat не запускался: условие для него — разные hashes при малом
расхождении — не наступило.

### Snapshots

На шагах `0/1/100/1000/4842` у run 1 и run 2 одновременно совпали:

- полные snapshot-file SHA256;
- canonical model-state SHA256;
- canonical optimizer-state SHA256;
- Python/NumPy/Torch/CUDA RNG SHA256.

Финальный model-state SHA256 обоих repeats:
`cdce6627ec476c6c8f9ae90402f9dc0a7344391bf6f06549523748cba897c81b`.
Финальный optimizer-state SHA256:
`6006fef91173661b19b0a7cdf70759bc7b21ef427448069490fe5eb08b1e6ab6`.
Полная таблица hashes: `artifacts/DET_PAIR/comparison.json`.

## Найденные источники nondeterminism

1. **Подтверждённый источник из `exp_038`:** `workers=3` делят plan блоками и
   складывают результаты в общую queue; первым потребляется тот batch, который
   раньше собрал CPU. Устранён `workers=1` плюс материализованным plan.
2. **cuDNN algorithm selection:** production держит `benchmark=True`.
   Зафиксировано `benchmark=False` + `deterministic=True`.
3. **CUDA/cuBLAS kernels:** production не требует deterministic algorithms.
   Включены `torch.use_deterministic_algorithms(True)` и фиксированный workspace;
   неподдерживаемых операций в полном forward/backward не найдено.
4. **RNG:** зафиксированы Python, NumPy, Torch CPU и все CUDA generators;
   per-batch seed depth curriculum хранится в plan.
5. **Optimizer state:** старого state в checkpoint нет, но это не источник
   неопределённости. Оба новых AdamW стартуют пустыми и совпадают на каждом
   snapshot до финального шага.

После этих мер ненулевого residual nondeterminism не обнаружено.

## Вердикт и разрешающая способность

### `PASS DET-PAIR`

Два полностью отдельных neural continuation дали побитово одинаковые model,
optimizer, RNG snapshots и raw validation predictions. FT-FRESH-ENC по условию
задачи **не начат**.

Наблюдаемый numerical floor этой фиксированной execution policy — **ровно 0**
на 197 379 float32 прогнозах после 4 842 optimizer steps. Для планирования
следующего эксперимента принимается консервативная разрешающая способность
**`1e-4 RMSLE_cal`**: эффекты `5e-4..1e-3` лежат в 5–10 раз выше execution floor
и теперь численно различимы в строгой паре.

Это не отменяет statistical/seed/fold uncertainty проекта: будущие neural
эффекты всё равно надо мерить парно на одном plan и подтверждать по фолдам/сидам.
PASS говорит только, что к ним больше не примешивается цена запуска `0.00033` и
`Var(Δz)=0.0204` из `exp_038`.

## Проверки и воспроизведение

```text
python -m pytest src/test_det_pair.py -q
# 5 passed

python -m pytest src/test_seq.py src/test_fnl.py src/test_validation.py -q -m "not slow"
# 130 passed, 1 deselected

python src/det_pair.py
# repeat artifacts уже завершены: plan/run1/run2 переиспользуются и compare даёт PASS
```

`artifacts/DET_PAIR/run{1,2}/result.json` содержат полные cfg/environment,
метрики, hashes и длительности; `comparison.json` — парное сравнение. Никаких
файлов в `submissions/` не создавалось.
