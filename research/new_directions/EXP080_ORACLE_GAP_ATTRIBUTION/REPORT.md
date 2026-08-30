# EXP080 — Mathematical gap attribution

## Verdict

**NO_EVIDENCE.** В raw history есть огромный *oracle* signal о будущих исходах, но его
инкрементальная часть поверх production-like span практически не предсказывается из доступных
до cutoff признаков. Ни один механизм не прошёл одновременно заданные gates
`oracle ΔMSE >= 0.0010` и (`>=25%` oracle headroom forward-predictable или latest clean
`rho >= 0.020`). Новая production-модель, TEST inference и submission поэтому не запускались.

Все headline-числа пересчитаны из primary artifacts. Старые `REPORT.md` использовались только как
карта путей. Raw target совпал с canonical OOF до `7.28e-12`; EXP076 foldwise `b/G` воспроизведены
с maximum absolute error `1.38e-16`. Запрещённый `oof_BLOCK4_SAF.activity` не загружался.

К накопленному research применена следующая reconciliation: EXP069 FRESH и EXP075 temporal
directions включены только через их сохранённые clean predictions; EXP070 count-value,
EXP071 ETX-FRESH, EXP072 LWA и EXP073 control-variate не переоткрывались после их отрицательных
primary metrics; EXP074 остался technical block; EXP077 подтвердил in-span forward-stack NO-GO;
EXP078 закрыл global level; EXP079 закрыл EXP075 scale-axis. Для EXP075–EXP079 заново проверены
prediction hashes, fold keys, targets, projection identities и/или сохранённые fold-level моменты,
а не только текст их отчётов.

## A. Current gap mathematics

Фактические scores заданы условием задачи; локальный `scores/submissions.csv` ещё не содержит две
последние строки. Идентичность файлов проверена по SHA256:

- `SUBMIT_EXP075_JOINT_A1_365_A2.csv`:
  `d567d91d66e4d80e28998de6139c48c59f7a607b3f8165c88a1d05259c66c901`;
- `SUBMIT_ORTH_ALPHA.csv`:
  `9a8adb83e7b34bb6c12b7eb51584d1bf9a93825945d285258d4e1dd991f4b838`.

Пусть `R=1.646143314225527`, `R*=1.6446514942`. Тогда:

```text
current RMSLE                 = 1.646143314225527
target RMSLE                  = 1.644651494200000
required Delta RMSLE          = 0.001491820025527 improvement
current MSE                   = 2.709787810969402
target MSE                    = 2.704878537374293
required Delta MSE            = 0.004909273595110 gain
```

Эквивалентные представления разрыва:

```text
required independent rho      = sqrt(DeltaMSE) / R = 0.0425638570
required rho^2                = 0.0018116819
equivalent global intercept   = sqrt(DeltaMSE) = 0.0700662087 log units
MSE reduction per user        = 0.0049092736 log^2
SSE reduction for 250k users  = 1227.3184 log^2
fraction of current LB MSE    = 0.1811682%
```

На production-like historical proxy средний residual равен `+0.0590`, но по фолдам меняет знак:
`-0.1242, +0.1605, +0.0998, +0.0362`. Следовательно, даже offline global intercept не является
стабильным механизмом, а требуемый эквивалентный bias `0.0701` больше latest-fold bias.

## B. Residual MSE attribution

Production-like baseline здесь — composition-matched реконструкция `SUBMIT_ORTH_ALPHA` плюс exact
EXP075 joint correction при отправленной амплитуде `1.0`, с foldwise projection из
`[1, z_match, 40 clean OOF components]` и `z>=0` clip. Его абсолютный historical RMSLE
`1.750104` не переносится на LB и используется только для residual geometry.

| Component | MSE share | Oracle removable MSE | After-span headroom | Observable headroom |
| --------- | --------: | -------------------: | ------------------: | ------------------: |
| Zero/positive class | 100% | 1.482104 | 2.198211 | 0.000257 |
| Positive purchase-day count | 58.828% | 0.661256 | 0.279920 | 0.000000 |
| Positive purchased-item count | 58.828% | 0.672707 | 0.321150 | 0.000000 |
| Positive event-day count | 58.828% | 0.681722 | 0.015180 | 0.000000 |
| Conditional monetary value after count | 58.828% | 0.324451 | 0.439444 | 0.000042 |
| Pure horizon timing after count+value | 58.828% | — | 0.000289 | 0.000000 |
| Joint predefined user-state specialist | 100% | 0.001299 | 0.001299 | 0.000246 |

`After-span headroom` — Frisch–Waugh–Lovell incremental gain после projection. Он не обязан быть
меньше unprojected group-intercept gain: residualization меняет направление correction. Поэтому
`already covered` нельзя честно считать простым вычитанием двух столбцов. Вместо этого напрямую
измерена доля oracle-label variance, линейно представленная production span:

- future positive indicator: `32.75%`; с explicit DIST/Block4/BTYD hurdle outputs — `34.93%`;
- purchase-day count: `35.22%`;
- purchased-item count: `29.80%`;
- event-day count: `28.17%`;
- conditional monetary value: только `10.12%`;
- horizon activity pattern: `14.67%`.

То есть oracle-классы действительно огромны. Проблема не в отсутствии будущей информации в
математическом смысле, а в отсутствии предсказуемого *incremental error* поверх production span.

### A. Zero / non-zero

- zero users: `38.8435%` population и `41.1722%` residual MSE;
- positive users: `61.1565%` population и `58.8278%` residual MSE;
- perfect knowledge `y>0` даёт after-span oracle `2.198211 MSE`;
- explicit existing hurdle signals увеличивают label `R²` лишь с `0.3275` до `0.3493`;
- target-free forward activity block оставляет только `0.000257 MSE` gain, latest
  `rho=0.0110`: **REJECT_UNOBSERVABLE**.

### B. Count / frequency

У positive users residual сильнее связан с oracle count buckets, чем с raw linear count:
`corr(log1p(count), residual)` всего `0.0715` для purchase days и `0.1372` для purchased items,
но nonlinear bucket oracle даёт `0.2799–0.3211 MSE` после span. Fixed forward count block на
target-free history добавляет `+0.000016 MSE` (ухудшение), latest `rho=0.00269`:
**REJECT_UNOBSERVABLE**.

`to_ord` в raw — purchased-item count, не доказанное число транзакций; поэтому он так и назван.

### C. Monetary / basket

При известном purchase-day count conditional amount quintiles добавляют `0.439444 MSE` oracle
headroom. Production span объясняет лишь `10.12%` variance `log1p(avg GMV/purchase day)`, но
target-free conditional-value ridge после projection имеет raw weighted `rho=-0.00738` и latest
nested incremental `rho=0.00494`. Forward gain `0.0000416 MSE`: **REJECT_UNOBSERVABLE**.

## C. Segment / tail attribution

Из семи заранее заданных target-free segmentations ни одна отдельно не достигает `0.001 MSE`:

| Segmentation | Whole-segmentation after-span gain |
| --- | ---: |
| Recency | 0.000659 |
| Recent channel mix | 0.000526 |
| Baseline prediction decile | 0.000372 |
| Purchase frequency | 0.000273 |
| Zero-probability quintile | 0.000269 |
| Activity intensity | 0.000148 |
| Tenure quartile | 0.000109 |

Совместный 29-dummy specialist имеет oracle `0.001299`, но forward переносит `0.000246`
(`18.90%`, ниже gate `25%`) и latest rho `0.0111`. Stable signed specialist не найден. Например,
mean residual сегмента `recency 0–7` по фолдам равен
`-0.0835, +0.1889, +0.1208, +0.0343`; знак не стабилен.

Tail concentration:

| Tail | Population | Residual MSE share | Oracle indicator gain |
| --- | ---: | ---: | ---: |
| Top target GMV 1% | 1.00% | 2.53% | 0.0562 |
| Top target GMV 10% | 10.00% | 17.26% | 0.4342 |
| Top baseline prediction 10% | 10.00% | 4.81% | 0.000128 |
| Largest absolute residual 1% | 1.00% | 8.37% | 0.0321 |
| Largest absolute residual 10% | 10.00% | 44.19% | 0.0281 |

Ошибка концентрируется в post-hoc absolute-residual tail, но не в observable high-prediction tail.
Top-prediction users даже легче среднего; значит high-GMV specialist, выбираемый по baseline
prediction, не локализует gap. Largest-residual membership target-derived и не является feature.

## D. Target decomposition

| Horizon | GMV share | Target-log allocation share | Corr with residual | After-span linear gain |
| --- | ---: | ---: | ---: | ---: |
| Days 1–7 | 23.14% | 23.39% | 0.3125 | 0.3338 |
| Days 8–14 | 23.26% | 23.04% | 0.3296 | 0.3672 |
| Days 15–30 | 53.60% | 53.57% | 0.5296 | 1.0745 |

Late window несёт больше total-target mass и потому сильнее коррелирует с residual. Но после
conditioning на zero/count/value чистый oracle gain распределения по горизонту равен только
`0.000289 MSE < 0.0010`. Это закрывает отдельный `GMV7 + GMV8-14 + GMV15-30` experiment до
обучения: disproportionate raw attribution не является independent timing mechanism.

## E. Observable predictability

Fixed Ridge использовал 108 cutoff-safe RFM/channel features. Для каждого validation fold train
cutoffs были `T-[77,63,49,35]`; максимальный train target end был не позже `T-5 days`. Standalone
labels предсказываются хорошо: activity AUC `0.838–0.844`, count correlations `0.718–0.775`,
conditional-value correlation `0.391–0.398`. После full production projection сигнал исчезает.

| Block | rho raw | rho after strong | rho after projection | Latest nested rho | Forward gain MSE | Bootstrap 95% CI Delta MSE | Fraction required gap | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Activity | -0.00238 | 0.00410 | 0.00278 | 0.01101 | 0.000257 | [-0.000534, +0.000008] | 5.24% | REJECT_UNOBSERVABLE |
| Count | -0.00267 | 0.00503 | 0.00111 | 0.00269 | 0.000000 | [-0.000217, +0.000265] | 0.00% | REJECT_UNOBSERVABLE |
| Monetary | -0.00672 | -0.00545 | -0.00738 | 0.00494 | 0.000042 | [-0.000093, +0.000012] | 0.85% | REJECT_UNOBSERVABLE |
| State segments | 0.00163 | 0.00205 | 0.00336 | 0.01110 | 0.000246 | [-0.000559, +0.000099] | 5.00% | REJECT_UNOBSERVABLE |
| Joint covariance-aware | — | — | optimal 0.02804 | 0.01513 | 0.000529 | [-0.000996, -0.000035] | 10.77% point | REJECT_GATE |

`Forward gain` использует coefficients только из предыдущих clean validation folds. Для joint
block использовано `G=UᵀU/N`, `b=Uᵀr/N`, а не сумма raw rho. In-fold optimal observable headroom
равен `0.002412 MSE`, но это upper diagnostic с validation-fitted coefficients; strict-forward
point — `0.000529`, а статистически гарантированный 95% lower headroom — `0.0000352`.

## F. Mathematical attainability

```text
required MSE gain             = 0.004909274
THEORETICAL_ORACLE_HEADROOM   = 2.975002269
OBSERVABLE_HEADROOM           = 0.002412031   (optimistic held-out feature-space optimum)
ROBUST_FORWARD_HEADROOM       = 0.000035244   (95% lower bound; point = 0.000528690)
```

- optimistic observable bound covers `49.13%` of required gap;
- strict-forward point covers `10.77%` and эквивалентен примерно `0.0001606 RMSLE`;
- robust 95% bound covers `0.72%` и эквивалентен `0.0000107 RMSLE`.

На качестве joint forward point понадобилось бы примерно **10 mutually independent copies**;
на robust lower-bound качестве — около **140**. Даже optimistic in-fold optimum требует больше
двух полностью независимых блоков. Эти расчёты уже covariance-aware и не суммируют raw rho.

```text
Is ~1.64465 mathematically supported by available data?
NO_EVIDENCE
```

Oracle показывает, что будущий outcome в принципе устранил бы gap многократно. Но available
target-free history не даёт достаточного identifiable incremental direction. Следовательно,
имеющиеся historical данные и текущие observable features не дают статистического основания
ожидать score `~1.64465`.

## G. Ranked experiments

Ни один full experiment не авторизован. Ниже — ranking кандидатов по forward expected value после
уже выполненного дешёвого pilot.

### 1. Activity-error residual specialist — NO-GO

- **Mechanism:** предсказывать ошибку extensive component поверх production, не direct target.
- **Evidence:** oracle `2.1982`, forward observable `0.000257`, latest rho `0.0110`, expected
  `ΔRMSLE≈-0.0000781` при идеальном offline→LB transfer.
- **Why not contained:** output проектировался из 40-component span; оставшаяся корреляция мала.
- **Would-be experiment:** shallow calibrated classifier `y30>0`, те же clean folds/features,
  residual correction only after projection.
- **GO gate:** `>=25%` oracle или latest rho `>=0.020`; фактически `0.0117%` и `0.0110`.
- **Compute:** pilot `~2 min CPU`; full shallow LGBM `~20–40 min CPU`, но запуск запрещён gate.

### 2. Fixed user-state residual specialist — NO-GO

- **Mechanism:** один joint specialist по recency/frequency/intensity/tenure/prediction-decile/
  p-active/channel-mix.
- **Evidence:** oracle `0.001299`, forward `0.000246` (`18.9%`), latest rho `0.0111`, expected
  `ΔRMSLE≈-0.0000746`.
- **Why not contained:** 29 dummies residualized against production; optimal held-out gain реален,
  но coefficients нестабильны forward.
- **Would-be experiment:** ridge on production residual with fixed bins and no new representation.
- **GO gate:** `25%` или rho `0.020`; оба не выполнены.
- **Compute:** уже полностью проверено на CPU; дальнейший run не нужен.

### 3. Count × conditional value residual factorization — NO-GO

- **Mechanism:** errors of expected count and conditional amount, а не generic hurdle/direct GMV.
- **Evidence:** oracle count `0.3212`, monetary `0.4394`; forward count gain `0`, monetary
  `0.0000416`; latest rhos `0.00269/0.00494`.
- **Why not contained:** после projection target labels всё ещё имеют oracle headroom, но
  target-free predictions не align с production residual.
- **Would-be experiment:** shallow count regressor + positive-only amount regressor, joint
  covariance calibration.
- **GO gate:** оба блока fail; full model не разрешён.
- **Compute:** ridge pilot `~2 min CPU`; потенциальный LGBM `<1 h`, но не запускать.

## H. Executed experiment(s)

1. **Oracle-only phase:** 770,616 canonical rows, four clean folds, 40-component production span;
   моделей не обучено.
2. **Cheap structural predictability pilot:** fixed multi-output Ridge для activity/count и
   positive-only Ridge для conditional value; `136 s CPU`, GPU не использовался.
3. **Cross-fitted segment regression:** coefficients forward-fitted по предыдущим folds.

Все pilots завершились ниже pre-registered mathematical gate. Поэтому shallow LightGBM,
refinement, TEST inference и submission не выполнялись.

## I. Final conclusion

1. **Где residual MSE:** `41.17%` у zero users, `58.83%` у positive; post-hoc top 10% absolute
   residual users содержат `44.19%` MSE, но observable top-prediction 10% — только `4.81%`.
2. **Теоретически устранимо:** joint structural oracle после span — `2.9750 MSE`, намного больше
   required `0.004909`; физически gap находится прежде всего в неизвестном extensive outcome,
   count и conditional amount.
3. **Реально предсказуемо:** optimistic joint `0.002412`, strict-forward `0.000529`, robust 95%
   headroom `0.0000352 MSE`.
4. **Максимальный expected mechanism:** activity-error block (`0.000257 MSE` point), практически
   равен state specialist (`0.000246`), но оба fail formal gate.
5. **Путь к `~1.6446515`:** математически не обоснован текущими historical/observable данными.
6. **Следующий coding experiment:** **NONE_AUTHORIZED**. Не запускать новый model run. Следующее
   допустимое действие — добавить действительно новый target-free data channel или новый clean
   future cutoff и повторить этот же frozen attribution/gate; без нового источника данных запуск
   ещё одного encoder/LightGBM нарушит установленный критерий.

## Reproducibility

- `run_oracle.py` — baseline reconstruction, oracle decomposition, segments/tails/horizon;
- `run_observable.py` — clean historical structural Ridge pilots, projection, nested covariance,
  cluster bootstrap;
- `gap_math.json`, `oracle_components.csv`, `segment_attribution.csv`,
  `tail_attribution.csv`, `horizon_attribution.csv`, `joint_oracle.csv`;
- `mechanism_gates.csv`, `observable_fold_metrics.csv`, `observable_nested_marginal.csv`,
  `observable_bootstrap.json`, `attainability.json`;
- `oracle_audit.json`, `observable_audit.json`, `production_bank_audit.csv`.
