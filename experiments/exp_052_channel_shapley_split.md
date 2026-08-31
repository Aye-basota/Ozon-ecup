# exp_052 — CHANNEL-SHAPLEY-SPLIT

- **Дата:** 2026-08-24
- **Автор:** A1
- **Коммит:** a28a71f

## Гипотеза

Две единственные monetary heads по Shapley-вкладам Search/Catalog могут дать полезную residual-форму для `STRONGEST_CURRENT`. Architecture-matched control сохраняет две модели и полный target `z`, но перемешивает contribution share внутри `train cutoff × total-z decile`.

## Что изменено относительно базы

На неизменной матрице `S1-E10` (227 колонок) обучены только `E[phi_s|X]` и `E[phi_c|X]`; causal estimand — `d=z_real-z_shuf` внутри residual `STRONGEST_CURRENT`.

## Phase 0 — exact baseline и data audit

- Raw OOF reconstruction: `PASS_EXACT`; calibrated folds `1.766883357 / 1.760509577 / 1.748629224 / 1.741278566`, wCV `1.747509863`.
- OOF: 770,616 строк; sizes `188518 / 191025 / 193694 / 197379`; row/target/prediction SHA256: `3bfac84c... / e3a609fa... / ffa81e2c...`.
- Полный `train.parquet`: 30,631,006 строк, 250,000 users, 2025-01-01…2026-02-13. Negative/duplicate/identity failures: `0/0/0`; `max|gmv_search+gmv_cat-gmv|=5.2751e-11`, mean `5.2410e-16` при project tolerance `1e-6`.
- Все `(user, cutoff)` targets построены строго из `(T,T+30]`; текущий target совпал с `log1p(S+C)`, поздние даты и validation rows не использованы. `phi_s+phi_c=z` max error `8.88e-16`, min contributions `0`, symmetry error `0`, все 301,289 zero rows дали нулевые вклады.
- Aggregate future GMV share: Search `0.927322`, Catalog `0.072678`; zero rates Search/Catalog/total `0.402721 / 0.903288 / 0.388435`. Полные fold/regime/correlation/stability/drift tables сохранены в artifacts.
- Ранний roadmap `EXP-08 Channel-specific models` найден только как косвенная запись в `STRATEGY_06`/index; experiment card и artifacts выполнения отсутствуют, поэтому он не считался выполненным.

## Phase 1 — pre-flight

- Leakage-only composition oracle (future regime × fixed `u` bin × STRONGEST decile, honest LOFO mapping, shrink `n/(n+20000)`): wCV `1.264574304`, delta `-0.482935558`, лучше `4/4`, включая 10-16; centered shape-only delta тот же. Это только upper bound с фактической future composition, не достижимая production-оценка.
- Predictability gate: past90 → future contribution share weighted Spearman `0.278576`, signs `4/4` positive (`0.275623 / 0.278171 / 0.280581 / 0.278044`). Diagnostic classifier не запускался, поскольку correlation gate уже прошёл.
- **Pre-flight: GO.**

## Pilot 2025-10-16

- Train: 4,955,174 clean rows по существующему panel recipe; validation: 197,379 rows.
- CPU LightGBM direct regression, exact accepted `S1-E10 @ 300`, no early stopping, один shared `float32` feature matrix; REAL/SHUF row order, feature order и configs идентичны.
- Ровно четыре trajectories: REAL/SHUF Search seed 42, REAL/SHUF Catalog seed 43. Direct total head отсутствует.
- Shuffle детерминирован и materialized только внутри train cutoff × stable total-z decile; `u` multiset и `z` сохранены в каждой stratum, validation не участвовал в edges/permutations.

## Результат

Standalone calibrated RMSLE на 10-16:

| Predictor | RMSLE | Offset |
|---|---:|---:|
| `STRONGEST_CURRENT` | 1.741278566 | -0.035387154 |
| `z_real` | 1.745762114 | -0.049024383 |
| `z_shuf` | 1.744496418 | -0.045770824 |

`REAL-SHUF = +0.001265695`: настоящая composition хуже matched shuffled control. AUC(y>0) `0.843774 / 0.844040`; positive RMSLE `1.675410 / 1.673788`; zero RMSLE `1.851677 / 1.850917` (REAL/SHUF).

Two-sided residual:

- selected alpha A→B / B→A: `0 / 0`; recipient deltas: `0 / 0`; primary delta к `STRONGEST_CURRENT`: `0`.
- Fixed alpha deltas `[0,.25,.50,1]`: `0 / +0.000275456 / +0.000940531 / +0.003437483`.
- `corr(d,residual)` на halves: `-0.002727 / -0.003458`; `Var(d)=0.010861`, `max|d|=1.711200`, mean `d=0.003246`.
- Pearson/Spearman `z_real` vs STRONGEST: `0.997053 / 0.997099`; residual correlation `0.997654`. После final fold calibration выигрыша нет; level-shift explanation не проходит.

## Вердикт и вывод

**REJECT.** Pre-flight подтвердил наличие composition signal, но pilot causal control его не перенёс: REAL хуже SHUF, обе честные directions выбрали `alpha=0`, residual alignment имеет отрицательный знак, а вся positive alpha curve ухудшает метрику. `PROMOTE_TO_FULL_FOLDS = NO`; full folds, test inference, public LB и submission не запускались. Не спасать bins, seeds, rounds, alpha grid, targets или segment gates.

## Конфиг и воспроизведение

```text
python src/channel_shapley_split.py
python src/channel_shapley_split.py --analysis-only
python -m pytest src/test_channel_shapley_split.py src/test_pipeline.py src/test_validation.py -q
```

- Main runtime: `709.605 s`; tests: `40 passed`.
- Analysis-only replay: `PASS`, SHA256 `efd8659f2c93dff2720a644059d4c659d2a20912e0aa2c870af546721515a218`.
- Canonical result SHA256: `72321423b8783765809e2d92b7b979d0211bcc5e35a2c03bb351a5500b2ec249`.
- Results: `research/strategies/results/CHANNEL_SHAPLEY_SPLIT/CHANNEL_SHAPLEY_EXP052_*`.
- Boosters/raw arrays: `artifacts/CHANNEL_SHAPLEY_EXP052/CHANNEL_SHAPLEY_EXP052_*`.
