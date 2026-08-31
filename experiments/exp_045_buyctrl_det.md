# exp_045 — BUYCTRL-DET: настоящая `buy30` supervision против shuffle-control

- **Дата:** 2026-08-22/23
- **Автор:** A1
- **Коммит:** `a28a71f` + рабочее дерево
- **Код:** `src/buyctrl_det.py`, `src/test_buyctrl_det.py`; production-модули не менялись
- **Фолд:** `2025-10-16`, стандартная полная 3-блочная val-панель, 197 379 строк
- **Сиды:** `42/43/44`, только `src.config.SEED + {0,1,2}`
- **Вычисления:** локальная RTX 4060 Ti; 6 auxiliary arms = 9.99 GPU-ч, BASE переиспользован из `exp_044`
- **Leaderboard/submission:** не запускались и не создавались

## Гипотеза и causal contrast

Проверяется ровно одно утверждение: настоящая auxiliary supervision
`buy30 = 1[y30 > 0]` должна сформировать общий encoder plain `SEQ-01` полезнее,
чем loss того же веса и формы со случайной меткой той же prevalence.

Primary contrast заранее зафиксирован как **`BUYTRUE − BUYSHUF` по `RMSLE_cal`**.
Secondary — `BUYTRUE − BASE`. Endpoint — строго конец 4-й эпохи; validation не
использовалась ни для выбора эпохи, ни для остановки.

Decision rule до запуска:

- `PASS`: mean primary `<= -0.0007`, правильный знак минимум 2/3 и meaningful
  secondary (`<= -0.0003` в исполняемом правиле);
- `FAIL`: mean primary `> -0.0003`;
- иначе `INCONCLUSIVE`.

## Точная постановка

Все arms — plain `SEQ-01`: 17 каналов, история 365 дней, hidden 64, 8 TCN-блоков,
kernel 3, dropout 0.10, pooled-вектор 192, batch 1024/chunk 256, AdamW
`lr=0.003`, `wd=0.01`, betas `(0.9,0.98)`, warmup 300 + один общий cosine на
19 368 шагов, grad clip 1.0, bf16/TF32/eager. Обучение с нуля на 24 cutoff'ах
`2025-04-03..2025-09-11`, 4 955 174 строки на эпоху, 4 эпохи. `depth_aug=0`,
никаких D3A/FRESH/новых фичей.

Loss:

```text
BASE:     MSE(z30)
BUYTRUE:  MSE(z30) + 0.1 * BCEWithLogits(aux, 1[y30>0])
BUYSHUF:  MSE(z30) + 0.1 * BCEWithLogits(aux, cutoff_shuffle(1[y30>0]))
```

Aux-head — один `Linear(192,1)`, веса 0, bias = logit глобальной train prevalence
`0.534151`. Голова читает тот же pooled-вектор, участвует только в train; final
prediction всегда делает исходная direct head. Loss — именно raw BCE без
нормировки из `exp_038`, согласно preregistered формуле.

BASE — не новый stochastic прогон, а уже завершённый deterministic plain-SEQ
endpoint `DETSEQ01-S42/S43/S44-V1016` из `exp_044`. Его train recipe и
materialized plans в точности те же; checkpoint 4-й эпохи заново предсказан на
полной стандартной панели. Это сильнее нового повтора: `exp_043/044` уже показали
нулевой execution floor, а reuse исключает ещё один розыгрыш BASE.

## Строгая парность и аудит shuffle

- Для каждого seed обе auxiliary arms имеют одинаковые model/direct/aux states,
  joint-optimizer state и RNG state на step 0; model state совпадает с initial
  state соответствующего BASE.
- Создание aux-head выполняется после main model, затем RNG восстанавливается:
  dropout stream совпадает с BASE. RNG hashes BUYTRUE/BUYSHUF/BASE совпали на
  шагах `0/1/100/1000/4842/9684/14526/19368` для всех трёх seeds.
- Train users/rows, четыре index/batch plans, batch augmentation seeds, float64
  LR каждого шага, число primary optimizer steps и validation order совпадают.
- `workers=1`, `cudnn.benchmark=False`, `cudnn.deterministic=True`,
  `torch.use_deterministic_algorithms(True)`, debug `error`,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`, fixed Python/NumPy/Torch/CUDA RNG,
  отдельный процесс на arm.
- BUYTRUE и BUYSHUF различаются только массивом auxiliary labels. Shuffle сделан
  отдельно внутри каждого из 24 cutoff'ов; число 0/1 каждого cutoff совпадает
  точно. Доля изменившихся labels: `0.49661/0.49658/0.49693`.
- Anti-lookahead: labels — только уже существующий main target
  `1[y30>0]`; все train cutoff'ы удовлетворяют `T+30<=V`; новых входов нет.
- Artifact-backed regression suite: **166 passed, 1 slow deselected**.

Общий aux-init SHA256 всех seeds:
`30efa7f06575e194c65ef3c12c22f70a6afd83a6ee81cd60b8a5c2b8a1a92b39`.
Validation-order SHA256:
`848bb71de56c4ceecc8282254591547643c39622efdd4fc66af5563eeba70731`.

| seed | base plan ID | base plan file SHA256 | EXP-045 plan ID | EXP-045 plan file SHA256 |
|---:|---|---|---|---|
| 42 | `742cfb9e…08b3` | `3c840f28…fc2b` | `01d1c5b1…3174` | `74061f80…cde` |
| 43 | `e842c87c…a063` | `d90ab4c1…89d0` | `6c4111a8…94d` | `93d82b06…d541` |
| 44 | `19a4bd9a…f33d` | `61e576ed…8fbb` | `42b32c58…ce17` | `cc0a75e6…08ff` |

Полные hashes всех plan arrays, initial/final model/optimizer/RNG states и
snapshots сохранены в `artifacts/BUYCTRL_DET_EXP045/plans/` и `arms/*/result.json`.

## Primary endpoint

### RMSLE_cal, конец эпохи 4

| seed | BASE | BUYTRUE | BUYSHUF | **TRUE−SHUF** | TRUE−BASE |
|---:|---:|---:|---:|---:|---:|
| 42 | 1.746977431 | 1.746287310 | 1.746067317 | **+0.000219994** | −0.000690120 |
| 43 | 1.745872784 | 1.745377789 | 1.745553451 | **−0.000175662** | −0.000494995 |
| 44 | 1.746802958 | 1.748103917 | 1.746839909 | **+0.001264008** | +0.001300959 |

| contrast | mean | median | sample sd | правильный знак |
|---|---:|---:|---:|---:|
| **BUYTRUE−BUYSHUF** | **+0.000436113** | +0.000219994 | 0.000743770 | **1/3** |
| BUYTRUE−BASE | +0.000038615 | −0.000494995 | 0.001097567 | 2/3 |

Primary не просто не достиг `−0.0007`: средний знак противоположный и значение
выше FAIL-boundary `−0.0003`. Secondary в среднем равен нулю.

Prediction SHA256 (`BASE / BUYTRUE / BUYSHUF`):

- S42: `699af4ca…2c37` / `6ebfb69c…f0ca` / `be016501…8bf5`;
- S43: `e8872e7c…5a96` / `75568c65…13c2` / `7ad48514…4860`;
- S44: `79c7a964…072` / `63618625…22c` / `136c7a82…3a21`.

Raw float32 log-space predictions сохранены в `base/*/z_raw.npy` и
`arms/*/z_raw.npy`; raw auxiliary logits — в `arms/*/aux_logits.npy`.

## Mechanism diagnostics

### Auxiliary действительно выучила правильную задачу

| seed | aux BCE TRUE | aux BCE SHUF | aux AUC TRUE | aux AUC SHUF |
|---:|---:|---:|---:|---:|
| 42 | 0.469894 | 0.669763 | 0.846003 | 0.471897 |
| 43 | 0.469881 | 0.667009 | 0.845852 | 0.615930 |
| 44 | 0.470098 | 0.668681 | 0.845815 | 0.521686 |
| mean | **0.469958** | 0.668484 | **0.845890** | 0.536505 |

У BUYSHUF logits почти константны (`std=0.0197/0.0201/0.0228`); поэтому его AUC
нестабилен вокруг слабого случайного направления, несмотря на одинаковый BCE.
У BUYTRUE logits имеют `std=1.89..1.97`, validation AUC стабилен с sd `0.00010`.
Train BCE на epoch 4: TRUE `0.490018` mean против SHUF `0.689781`.

При этом direct activity AUC практически не изменился:
BASE/TRUE/SHUF mean = `0.842876/0.842920/0.842925`, то есть
`TRUE−SHUF = −0.000005`.

Это ровно сценарий 2 из постановки: **auxiliary task хорошо учится, но полезного
переноса в direct RMSLE нет**.

### Движение raw direct predictions

| seed | Var(Δz) TRUE−SHUF | corr | mean Δz | offset TRUE−SHUF | Var(Δz) TRUE−BASE | corr |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.022788 | 0.995369 | −0.035010 | +0.034236 | 0.023740 | 0.995199 |
| 43 | 0.021278 | 0.995765 | +0.049258 | −0.049263 | 0.023399 | 0.995537 |
| 44 | 0.025560 | 0.994723 | −0.022560 | +0.022641 | 0.022627 | 0.995434 |
| mean | 0.023209 | 0.995286 | −0.002771 | +0.002538 | 0.023255 | 0.995390 |

Encoder/direct forecast заметно перестраивается (`Var(Δz)≈0.023`, далеко выше
нулевого DET floor), но большая часть mean shift снимается calibration offset, а
остаточное перераспределение не уменьшает RMSLE.

### Сегмент `rec_buy 15–60`

TRUE−SHUF по seeds: `+0.000019 / −0.000072 / −0.000579`, mean
**`−0.000211`**. TRUE−BASE mean `−0.000264`. Направление 2/3, но масштаб ниже
даже FAIL-boundary полного fold и не спасает primary.

Другие сегменты не дают устойчивой компенсации: «никогда не покупал»
TRUE−SHUF mean `+0.002210`, `w180_days_buy 16+` `+0.001738`.

### Zero/positive decomposition

Относительно BUYSHUF у BUYTRUE positive RMSE в среднем меняется на
`−0.000364`, но zero RMSE ухудшается на `+0.001578`. В MSE contribution это
`−0.000737` у positives и `+0.002295` у zeros: слабый выигрыш величины покупок
перекрыт ошибкой на нулевых строках. Относительно BASE картина аналогична:
positive RMSE `−0.004780`, zero RMSE `+0.006881`.

Полные seed-level segment и zero/positive таблицы:
`segment_summary.csv` и `zero_positive_summary.csv`.

## Вердикт и вывод

### **FAIL — остальные folds не запускать**

Mean `BUYTRUE−BUYSHUF = +0.000436`, median `+0.000220`, sd `0.000744`, правильный
знак только 1/3. `BUYTRUE−BASE = +0.000039` mean: meaningful improvement нет.

Эксперимент доказывает для plain deterministic `SEQ-01`, fold `2025-10-16`,
seeds 42/43/44 и фиксированных `lambda=0.1`, 4 epochs: правильная buy-метка
выучивается auxiliary head намного лучше shuffled control и существенно двигает
encoder predictions, но **не делает direct RMSLE полезнее случайного auxiliary
loss**. Это сильный отрицательный результат для данной auxiliary-supervision
hypothesis, а не проблема оптимизации auxiliary head.

Эксперимент не доказывает ничего про другие folds, historical `SEQ-AVG3`,
`STRONGEST_CURRENT`, test/LB transfer, и другие lambda/epochs. По preregistered
правилу они не подбирались и не запускались.

## Артефакты и воспроизведение

- итог: `artifacts/BUYCTRL_DET_EXP045/analysis.json`;
- seed/segment/decomposition: `seed_summary.csv`, `segment_summary.csv`,
  `zero_positive_summary.csv`;
- plans и init: `artifacts/BUYCTRL_DET_EXP045/plans/`;
- predictions, logits, snapshots, hashes: `base/`, `arms/`;
- runner/tests: `src/buyctrl_det.py`, `src/test_buyctrl_det.py`.

```text
python -m pytest src/test_buyctrl_det.py src/test_seq.py src/test_fnl.py \
  src/test_det_pair.py src/test_fresh_cond_ft.py src/test_validation.py -q -m "not slow"
# 166 passed, 1 deselected

python src/buyctrl_det.py
# completed artifacts are verified/reused; analysis must reproduce exactly
```
