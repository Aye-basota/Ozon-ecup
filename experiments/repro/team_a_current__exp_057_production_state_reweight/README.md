# EXP-057 — PRODUCTION-STATE-REWEIGHT

## Catalogue metadata

- **Catalogue ID:** `team_a_current__exp_057_production_state_reweight`
- **Namespace:** `team_a_current`
- **Experiment ID:** `exp_057_production_state_reweight`
- **Original source:** `experiments/exp_057_production_state_reweight.md`
- **Source ref:** `working tree`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** experiment card
- **Model:** LightGBM, ensemble
- **Features:** calendar features, recency, history-depth features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: **4,955,174** legal UNC rows, 24 cutoff'а `2025-04-03…2025-09-11`; target: **197,379** validation states на `2025-10-16`, без target.
- **Known score:** A `UNIFORM`: exact old UNC, RMSLE_cal **1.745131674**.
- **Seed:** Domain model: LightGBM binary, 31 leaves, min leaf 2000, lr .03, 200 rounds, feature/bagging .8, L2 20, seed 42; equal source/target class mass; two-sided user cross-fit.
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when required data/artifacts are present; otherwise PARTIAL as stated in the card

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# EXP-057 — PRODUCTION-STATE-REWEIGHT

- **Дата:** 2026-08-24
- **Автор:** A1
- **Коммит:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`

## Гипотеза

Production UNC обучается на миллионах перекрывающихся исторических состояний, тогда как production представляет один поздний state каждого пользователя. Проверяем, улучшает ли target-free covariate-shift weighting фиксированный UNC-слот; causal contrast — настоящие STATE-MATCH weights против тех же весов, совместно перемешанных внутри заранее заданных state-страт.

## Что изменено относительно базы

Только веса train-строк исторического `S1-E02`; строки, порядок, 236 признаков, direct `log1p`, LightGBM, seed 42, 600 rounds и вес UNC-слота 0.20 зафиксированы.

## Phase 0 — exact baseline

- Historical UNC для `2025-10-16` воспроизведён bitwise из текущих 236 cutoff-safe признаков и сохранённой модели `TBR_EXP046_UNC_S42_V1016`.
- SHA256 UNC prediction: `76193afe9c48768184bbb617777237a465d5c5d35a36cab266677cf452ab9c7f`.
- Exact `STRONGEST_CURRENT = .10 CAP + .20 UNC + .25 DIST + .225 ETX + .225 SEQ`: **1.741278566448**, offset **−0.035387154112**; зарегистрированное ожидание 1.741278566 совпало.
- BASE_HEAD: `a28a71fb2d0194052014c542f36d180dfe74bcf9`. Исходный dirty workspace зафиксирован и не очищался/reset/checkout; чужие изменения и artifacts не перезаписывались.

## Domain weights и support

- Source: **4,955,174** legal UNC rows, 24 cutoff'а `2025-04-03…2025-09-11`; target: **197,379** validation states на `2025-10-16`, без target.
- Fixed semantic subset: **101** признак, только окна ≤180d, recency, days/GMV/orders/search/cart, conversions и trends. Запрещённых ID/date/weekday/depth/avail/w365/target/prediction признаков нет.
- Two-sided user cross-fit: fit side 0 → score side 1 и наоборот; user overlap в обеих парах **0**. Adversarial AUC **0.721328**.
- Raw odds clipped to `[0.25, 4.0]`, затем cutoff-normalization, user-total cap и global renormalization. Финальные веса: mean **1.000000**, min **0.183312**, max **4.914551**; превышение 4 появляется только после обязательных нормировок. Capped users **0.1961%**; max/median final user total **2.000000**.
- ESS **3,929,577 = 79.30%** source rows; минимальный cutoff ESS **73.41%**. Tiny-stratum dominance: **false**.
- STATE-MATCH/SHUFFLED weight multiset и total weight совпадают точно; total **4,955,174.002880**. Shuffle выполнен внутри `cutoff × rec_buy{0–14,15–60,>60} × w180_days_buy{0–1,2–15,16+}`.

## Additional audit

- У **235,076** source users число повторных rows: mean **21.079**, median/max **24/24**; **62.19%** пользователей встречаются во всех 24 cutoff'ах.
- 24 перекрывающихся 30d target windows покрывают 191 уникальный календарный день; adjacent overlap **23/30 = 0.766667**.
- Kish-effective target days **176.33**, то есть **5.88** эффективных независимых 30d windows (простая unique-day оценка **6.37**).
- Mean absolute semantic SMD к validation монотонно падает примерно с **0.2067** (`04-03`) до **0.0675** (`08-28`), затем слегка растёт до **0.0732** (`09-11`). Значит domain shift реален, но ближайшие cutoff'ы уже заметно ближе.

## Arms

- A `UNIFORM`: exact old UNC, RMSLE_cal **1.745131674**.
- B `SHUFFLED`: RMSLE_cal **1.745441380**, delta к UNIFORM **+0.000309706**.
- C `STATE_MATCH`: RMSLE_cal **1.745695449**, delta к UNIFORM **+0.000563775**, к SHUFFLED **+0.000254069**.
- B/C имеют одинаковые 4,955,174 rows, row order/hash, float32 feature matrix/order, train target, params, seed, 600 rounds и 4 CPU threads. GPU/early stopping не использовались.

## Primary ensemble endpoint

После сборки каждого fixed ensemble отдельно выполнена его calibration.

| Predictor | RMSLE_cal | AUC(y>0) | Δ к STRONGEST |
|---|---:|---:|---:|
| `STRONGEST_CURRENT` | 1.741278566 | 0.844315263 | — |
| `SHUFFLED_SLOT` | 1.741292411 | 0.844308862 | +0.000013844 |
| `MATCHED_SLOT` | 1.741294735 | 0.844322462 | **+0.000016169** |

Главный causal contrast `MATCHED_SLOT−SHUFFLED_SLOT` равен **+0.000002324** RMSLE: REAL не лучше control. AUC contrast **+0.000013600**, `Var(Δz)=0.000567018`, residual alignment **+0.006740**, но это не переносится в RMSLE.

Zero/positive decomposition для `MATCHED−SHUFFLED`: positive RMSLE улучшен на **−0.000284825**, zero RMSLE ухудшен на **+0.000412695**. Итоговая потеря поэтому не является простым level-сдвигом, а calibration после полной сборки не меняет знак primary endpoint.

## Сегменты и halves

| Segment | N | MATCHED−SHUFFLED | MATCHED−STRONGEST |
|---|---:|---:|---:|
| rec_buy 15–60 | 56,145 | −0.000092620 | −0.000073535 |
| w180 days_buy 2–15 | 111,441 | −0.000003918 | +0.000044541 |
| history-poor, w180 0–1 | 42,698 | +0.000052140 | −0.000039406 |
| frequent, w180 16+ | 43,240 | −0.000026574 | −0.000026946 |
| user half A | 98,707 | **−0.000082098** | −0.000026429 |
| user half B | 98,672 | **+0.000086032** | +0.000058405 |

Halves расходятся по знаку, поэтому обязательный two-sided gate провален.

## Вердикт и вывод

**REJECT; full folds/test/LB/submission NO.** Primary causal effect **+0.000002324**, `MATCHED_SLOT−STRONGEST = +0.000016169`, одна user half ухудшается, а standalone STATE_MATCH хуже и UNIFORM, и SHUFFLED. Support хороший (ESS 79.3%, tiny-stratum dominance false), поэтому отрицательный вывод объясняется отсутствием utility exact reweighting-схемы, а не collapse весов.

Не продолжать exact production-state reweighting через новые semantic subsets, domain-model tuning, clipping, shuffle bins, rounds/seeds или segment gates. Результат не переносить на supervised weighting или другую validation design.

## Конфиг прогона

- UNC: historical `S1-E02`, `L=None`, `min_history=90`, `train_blocks=1`, 24 train cutoff'а, 3-block validation, 236 features, direct `log1p`, LightGBM 600 rounds, seed 42 из `config.py`.
- Domain model: LightGBM binary, 31 leaves, min leaf 2000, lr .03, 200 rounds, feature/bagging .8, L2 20, seed 42; equal source/target class mass; two-sided user cross-fit.
- Execution: CPU only. B/C обучались изолированными subprocesses и собирали matrix по одному cutoff'у для снижения peak RAM; causal recipe и hashes строк/матрицы/config не менялись.
- Methodological deviation: отдельный worktree не создавался, потому что experiment зависит от workspace-local ignored data/cache и production artifacts. Изоляция обеспечена уникальными EXP057 paths; shared `STATE.md`, `HISTORY.md` и `experiments/log.csv` изменены только после завершения анализа.
- Не запускались full folds, CAP/DIST weighting, test/public inference, LB или submission.

## Проверки и артефакты

- `python -m pytest src/test_production_state_reweight.py src/test_pipeline.py src/test_validation.py -q` → **28 passed**.
- Analysis-only replay SHA256 дважды: `6386c3352e170993637437cfb5dba8fbfe44a6163ab225116b46023c6c0545c1`.
- Combined prediction artifact SHA256: `20bf1a001e98e931f6be5ef6fe614ef103afb6ad24bf9162a7ac1f3f0554d02a`.
- Weights artifact SHA256: `eec66d5d66a8cb41c886b821177e1ba953d93759605833556198e9fbe9573398`.
- Полные manifests/CSV: `research/strategies/results/STATE_REWEIGHT_EXP057/`; модели и NPZ: `artifacts/STATE_REWEIGHT_EXP057/`.
