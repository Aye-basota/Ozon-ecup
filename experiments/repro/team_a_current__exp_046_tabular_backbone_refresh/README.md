# exp_046 — TABULAR-BACKBONE-REFRESH: rounds × AVG3 для production UNC/CAP

## Catalogue metadata

- **Catalogue ID:** `team_a_current__exp_046_tabular_backbone_refresh`
- **Namespace:** `team_a_current`
- **Experiment ID:** `exp_046_tabular_backbone_refresh`
- **Original source:** `experiments/exp_046_tabular_backbone_refresh.md`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** experiment card
- **Model:** LightGBM, sequence model, ensemble, blend, calibration diagnostic
- **Features:** freshness/conditional features, 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** | **wCV 1:2:4:8** | **1.747509863** | **1.747509862520** |
- **Known score:** | **wCV 1:2:4:8** | **1.747509863** | **1.747509862520** |
- **Seed:** `seed_variance_diagnostics.csv`, `ensemble_folds.csv`,
- **Postprocessing:** Исторические raw OOF выровнены по `(cutoff,user_id)` и смешаны в log-space:
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when required data/artifacts are present; otherwise PARTIAL as stated in the card

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# exp_046 — TABULAR-BACKBONE-REFRESH: rounds × AVG3 для production UNC/CAP

- **Дата:** 2026-08-23
- **Автор:** A1
- **Коммит:** `a28a71f` + рабочее дерево
- **Код:** `src/tabular_backbone_refresh.py`, `src/test_tabular_backbone_refresh.py`
- **Результаты:** `research/strategies/results/TABULAR_BACKBONE_REFRESH/`
- **Artifacts:** `artifacts/TBR_EXP046/` — 24 trajectories, 109 файлов, 485 MB
- **Вычисления:** CPU, 24 LightGBM trajectory, суммарно 4 741 s; test/full-train/submission не запускались

## Гипотеза

Проверить controlled production debt: переносится ли ранее измеренное на `S1-E10`
улучшение от 300 вместо 600 rounds и усреднения seed 42/43/44 на реальные
production-компоненты `UNC/CAP`, и остаётся ли выигрыш внутри фиксированного
`STRONGEST_CURRENT`. Primary endpoint заранее зафиксирован как `D = AVG3@300`
для обоих компонентов; веса и остальные три компонента не меняются.

## Что изменено относительно базы

Переобучены только `UNC/CAP`; единственные изменяемые оси — master seed и число
использованных деревьев. `DIST/ETX-AVG3/SEQ-AVG3`, признаки, target, cutoff grid,
панели, калибровка и blend weights `0.10/0.20/0.25/0.225/0.225` заморожены.

## Phase 0 — exact historical baseline

Исторические raw OOF выровнены по `(cutoff,user_id)` и смешаны в log-space:

```text
0.10 S1-E03a (CAP) + 0.20 S1-E02 (UNC) + 0.25 S1-DIST
+ 0.225 ETX-AVG3 + 0.225 SEQ-AVG3
```

| fold | ожидаемый | воспроизведённый |
|---|---:|---:|
| 2025-09-04 | 1.766883357 | **1.766883356800** |
| 2025-09-18 | 1.760509577 | **1.760509576780** |
| 2025-10-02 | 1.748629224 | **1.748629223965** |
| 2025-10-16 | 1.741278566 | **1.741278566448** |
| **wCV 1:2:4:8** | **1.747509863** | **1.747509862520** |

Строк OOF 770 616, fold sizes `188518/191025/193694/197379`. Hash canonical
row keys `3bfac84c…b91`, target `e3a609fa…34b`. Полный manifest содержит absolute
paths и SHA-256 пяти входов; после эксперимента они повторно проверены и не
изменились. Summary rebuild дважды дал один SHA-256:
`8da527c4bce23e4f673c2e5b0b9694fde0d872b4398761db0af513babb83a899`.

## Точные recipes UNC/CAP

Общее: LightGBM direct regression по `log1p(y30)`, `float32` features/raw OOF,
train panel 1 block, validation panel 3 blocks, clean grid step 7, train cutoff
только при `T+30<=V`, 18/20/22/24 cutoff'а по fold, early stopping off.

```text
objective=regression, metric=rmse, learning_rate=0.05, num_leaves=127,
min_data_in_leaf=200, feature_fraction=0.7, bagging_fraction=0.8,
bagging_freq=1, lambda_l2=5, max_bin=63, force_row_wise=True, threads=12
```

| компонент | artifact | L | min_history | norm_long | features | feature-order SHA-256 |
|---|---|---:|---:|---|---:|---|
| UNC | `S1-E02` | none | 90 | false | 236 | `47d91fba…5d44` |
| CAP | `S1-E03a` | 180 | 90 | false | 195 | `3f2f8d3e…48dc` |

Фактический historical policy задаёт один master `seed`, а LightGBM
детерминированно выводит child streams. Для seed 42 resolved seeds равны
`bagging=400, feature_fraction=30056, data_random=175`; для 43 —
`11149/21351/179`, для 44 — `21897/12646/182`. Явно присвоить всем child keys
значение 42 означало бы изменить historical estimator и сломать replay; поэтому
runner сохраняет literal recipe и пишет resolved values в каждый manifest.

## Replay gate — PASS_BITWISE

До матрицы заново обучены `UNC/CAP seed42@600` только на fold 2025-10-16.

| component | max abs Δz | MAE | Var(Δz) | Pearson | ΔRMSLE_cal | prediction hash |
|---|---:|---:|---:|---:|---:|---|
| UNC | **0** | **0** | **0** | 1.0 | **0** | identical |
| CAP | **0** | **0** | **0** | 1.0 | **0** | identical |

`A` затем совпал с `H` на всех четырёх folds, не только на replay fold.
Prefix semantics переиспользует artifact-backed smoke `exp_017`: independent
300-round run побитово равен 300-tree prefix.

## Standalone component metrics

В ячейках — `wCV`; последняя колонка каждой группы — raw-log `AVG3`. Полные
fold scores, offsets, AUC, mean z и prediction hashes сохранены в
`standalone_metrics.csv`.

| component | rounds | S42 | S43 | S44 | AVG3 |
|---|---:|---:|---:|---:|---:|
| UNC | 200 | 1.750699744 | 1.750762488 | 1.750753103 | **1.750226051** |
| UNC | 250 | 1.750726246 | 1.750697910 | 1.750752508 | **1.750133552** |
| UNC | 300 | 1.750774885 | 1.750724135 | 1.750829625 | **1.750100978** |
| UNC | 600 | 1.751505993 | 1.751397885 | 1.751537454 | **1.750366137** |
| CAP | 200 | 1.760215603 | 1.760203734 | 1.760164920 | **1.759789427** |
| CAP | 250 | 1.760222178 | 1.760157572 | 1.760126203 | **1.759709467** |
| CAP | 300 | 1.760256888 | 1.760152641 | 1.760178927 | **1.759685692** |
| CAP | 600 | 1.760642088 | 1.760646456 | 1.760632447 | **1.759839569** |

Standalone `AVG3@300` действительно лучше `AVG3@600`: UNC −0.000265 и CAP
−0.000154. AUC `AVG3@300/600`: UNC `0.841414/0.841164`, CAP
`0.839653/0.839547`. Но это не primary endpoint: внутри ансамбля перенос не
состоялся.

Seed pair `Var(z_i-z_j)` при 300 rounds: UNC `0.007381..0.007440`, CAP
`0.005673..0.005727`; prediction corr `0.99843..0.99845` и
`0.99880..0.99882`. Standalone wCV range по seed около `0.000105` у обоих.
Полная таблица есть по каждому component/fold/round.

## H/A/B/C/D — primary factorial endpoint

`H` — сохранённый historical чемпион; `A` — fresh replay seed42@600;
`B` — AVG3@600; `C` — seed42@300; `D` — AVG3@300.

| variant | 09-04 | 09-18 | 10-02 | 10-16 | wCV | Δ к H | лучше folds | mean raw z |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H | 1.766883357 | 1.760509577 | 1.748629224 | 1.741278566 | **1.747509863** | 0 | — | 2.688187850 |
| A | 1.766883357 | 1.760509577 | 1.748629224 | 1.741278566 | **1.747509863** | **0** | 0/4 | 2.688187850 |
| B | 1.766852187 | 1.760413328 | 1.748590317 | 1.741227429 | **1.747457303** | **−0.000052560** | **4/4** | 2.688332363 |
| C | 1.766852031 | 1.760525693 | 1.748711672 | 1.741299359 | **1.747542999** | **+0.000033136** | 1/4 | 2.687689173 |
| D | 1.766828342 | 1.760479849 | 1.748658229 | 1.741273786 | **1.747507416** | **−0.000002447** | 3/4 | 2.687775865 |

Optimal offsets H/A: `−0.117345/−0.078736/−0.020978/−0.035387`; B:
`−0.117930/−0.078694/−0.020925/−0.035485`; C:
`−0.116114/−0.079118/−0.020975/−0.034249`; D:
`−0.116437/−0.078770/−0.021016/−0.034575`.

## Causal contrasts и interaction

| contrast | ΔwCV | fold deltas 09-04 / 09-18 / 10-02 / 10-16 | folds |
|---|---:|---|---:|
| `B-A`: AVG3 only @600 | **−0.000052560** | −0.000031 / −0.000096 / −0.000039 / **−0.000051** | 4/4 |
| `C-A`: rounds only @seed42 | **+0.000033136** | −0.000031 / +0.000016 / +0.000082 / **+0.000021** | 1/4 |
| `D-B`: rounds after AVG3 | **+0.000050113** | −0.000024 / +0.000067 / +0.000068 / **+0.000046** | 1/4 |
| `D-C`: AVG3 @300 | **−0.000035583** | −0.000024 / −0.000046 / −0.000053 / **−0.000026** | 4/4 |
| `D-A`: combined | **−0.000002447** | −0.000055 / −0.000030 / +0.000029 / **−0.000005** | 3/4 |
| `D-B-C+A` interaction | **+0.000016977** | +0.000007 / +0.000050 / −0.000015 / +0.000026 | — |

Seed averaging имеет правильный знак на 4/4 при обоих rounds, но после веса
UNC+CAP=0.30 его ensemble effect лишь `−0.000036..−0.000053`. Снижение rounds
имеет противоположный знак внутри смеси; positive interaction ещё на 0.000017
ухудшает combined endpoint.

## Component attribution

Все дельты ниже — отдельная замена относительно exact `H`, без обучения.

| variant | UNC-only | CAP-only | BOTH | replacement interaction |
|---|---:|---:|---:|---:|
| B AVG3@600 | −0.000044379 | −0.000008596 | **−0.000052560** | +0.000000415 |
| C S42@300 | +0.000016627 | +0.000011338 | **+0.000033136** | +0.000005172 |
| D AVG3@300 | −0.000007630 | −0.000000388 | **−0.000002447** | +0.000005571 |

Combined-провал не создаётся одним вредным компонентом: у primary `D` обе
одиночные замены практически нулевые, а не сильный плюс у одного и минус у
другого. Поэтому оснований для `PROMOTE_SINGLE_FACTOR` нет.

## Prediction/residual diagnostics

| variant vs H | Var(Δz) | max abs Δz | mean Δz | Pearson | Spearman | corr residuals | corr(Δz, residual_H) |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 0 | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | undefined (constant Δ) |
| B | 0.000197779 | 0.158670 | +0.000145 | 0.999958 | 0.999953 | 0.999968 | +0.007566 |
| C | 0.000178855 | 0.141587 | −0.000499 | 0.999962 | 0.999957 | 0.999971 | +0.002437 |
| D | 0.000295487 | 0.214798 | −0.000412 | 0.999937 | 0.999930 | 0.999952 | +0.005986 |

Prediction movement реально, но очень мало после весов 0.20/0.10 и почти не
согласовано с residual чемпиона. Raw diagnostics для каждого causal contrast,
включая interaction, сохранены в `factorial_contrasts.csv`.

## Secondary rounds diagnostics

Fixed ensemble curve использует AVG3 UNC/CAP при каждом snapshot; дельты к
primary 300:

| rounds | wCV | Δ к 300 | fold deltas к 300 |
|---:|---:|---:|---|
| 200 | 1.747572472 | +0.000065056 | +0.000029 / +0.000070 / +0.000074 / +0.000064 |
| 250 | 1.747533269 | +0.000025853 | +0.000017 / +0.000026 / +0.000029 / +0.000025 |
| **300** | **1.747507416** | **0** | 0 / 0 / 0 / 0 |
| 600 | 1.747457303 | **−0.000050113** | +0.000024 / −0.000067 / −0.000068 / −0.000046 |

Ответ протокола: **300 находится в плоском безопасном basin**, но история E10
про выигрыш rounds не переносится внутрь UNC/CAP fixed ensemble. Ни 200, ни 250
не лучше 300; 600 лучше всего лишь на 0.000050, то есть далеко ниже project floor.
Новый tuning из этих diagnostics не следует.

## FACT

- Exact historical reconstruction и fresh `A=H` побитово исключают configuration drift.
- `B-A = −0.000052560`, 4/4; `C-A = +0.000033136`, 1/4;
  primary `D-H = D-A = −0.000002447`, 3/4, 10-16 `−0.000004781`.
- Primary gain в 204 раза меньше promotion gate `−0.0005` и практически ноль.
- Shared fixed artifacts не изменились; test labels/predictions и submissions не читались/не создавались.

## INFERENCE

Standalone estimator-quality gain реален у обоих компонентов: AVG3@300 лучше
AVG3@600. Но ансамбль уже содержит сильно коррелированные `DIST/ETX/SEQ`, веса
обновляемых компонентов суммарно только 0.30, и уменьшение rounds меняет ошибку
в неполезном для фиксированной смеси направлении. Seed averaging переносит знак,
но не масштаб; rounds не переносит даже знак. Это и есть production-relevant
ответ, а не отрицание результатов `exp_017/018` на `S1-E10`.

## LIMITATION

Предыдущее evidence было только на `S1-E10` (`L=None`, `norm_long=True`, 227
features), а здесь проверены production `UNC/CAP` и только фиксированные points
300/600 для verdict. 200/250 — diagnostics, не новый sweep. Эксперимент ничего
не утверждает о `DIST-AVG3`, новых признаках, других blend weights, test transfer
или leaderboard; они намеренно не запускались.

## Вердикт и вывод

### **REJECT**

Primary `D` даёт `−0.000002447`, а лучший fixed single-factor `B` — лишь
`−0.000052560`; оба слабее rejection boundary `−0.0003`, несмотря на 3/4 и 4/4
fold signs. `C` и rounds после averaging вредят. Закрыть
`TABULAR-BACKBONE-REFRESH`; не начинать новый rounds/seed tuning.

К отдельному production/full-train/test experiment переходить **нельзя**.
Основания отдельно запускать `DIST-AVG3` **нет**: результат не BORDERLINE, а
уверенный REJECT на два порядка ниже gate. Submission не создан.

## Тесты, артефакты и воспроизведение

```text
python src/tabular_backbone_refresh.py
python src/tabular_backbone_refresh.py --analysis-only
python -m pytest src/test_tabular_backbone_refresh.py src/test_pipeline.py src/test_validation.py -q
# 30 passed
```

Два последовательных `--analysis-only` дали одинаковый summary SHA-256
`8da527c4…a899`. Главные outputs: `baseline_manifest.json`,
`replay_audit.{json,csv}`, `standalone_metrics.csv`,
`seed_variance_diagnostics.csv`, `ensemble_folds.csv`,
`factorial_contrasts.csv`, `component_attribution.csv`,
`rounds_diagnostics.csv`, `prediction_diagnostics.csv`, `summary.json`.
