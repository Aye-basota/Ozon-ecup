# EXP085 — Competition-structure forensic audit

## Executive verdict

```text
NO_NEW_STRUCTURE_FOUND
```

Я воспроизвёл raw schema на всех `30,631,006` строках, точную test-панель,
исторические eligibility cohorts и production-like residual EXP080. Parity с
primary EXP080 arrays точная: `user_id`, `cutoff`, `target_log`, `z_current` и
residual совпадают до последнего бита; basis содержит 40 production-компонент
плюс константу, current/matched baseline и EXP075 post-span direction.

Главная структурная особенность задачи действительно есть, но это не новый
deployable signal: raw universe уже отобран по активности в финальных трёх
30-дневных блоках. Поэтому любой исторический pseudo-test внутри файла
дополнительно conditionally selected по далёкой будущей активности. На cutoff
`2025-10-16` это особенно жёстко: target заканчивается `2025-11-15`, а все
пользователи гарантированно активны в каждом из трёх следующих блоков
`2025-11-16..2026-02-13`. У настоящего test после `2026-03-15` такой гарантии
нет. Это объясняет optimistic historical transfer и является вероятным
источником провала масштаба EXP075, но не даёт target-free correction для test.

Ни один новый доступный structural transform не прошёл заданный gate:

- лучший cohort-relative oracle, совместный rank по `w90_gmv × recency`, имеет
  nominal headroom `0.001221 MSE`, но после поправки на 24 степени свободы —
  `0.000844`; purged `rho=0.00057`, `Delta MSE=+0.004090`;
- eligibility-boundary geometry: debiased oracle `0.000243`, purged
  `rho=0.00217`, `Delta MSE=+0.000243`;
- target-free calendar shift proxy: weighted `rho=0.00290`, debiased headroom
  `0.000026`; purged `rho=0.00447`, `Delta MSE=-0.0000586`;
- общий DOW + day-of-month/payday profile: debiased oracle `0.000031`, purged
  `rho=-0.00509`, `Delta MSE=+0.000045`;
- `user_id`, sample-order, local ID density, KNN cohort density и availability
  ranks также не проходят ни oracle, ни observable gate.

Поэтому обучение новой модели, TEST inference и новый submission не разрешены.

## A. Competition data-generation reconstruction

### A1. Что физически лежит в данных

| Факт | Воспроизведённое значение |
| --- | ---: |
| Raw rows | 30,631,006 |
| Users in raw | 250,000 |
| Dates | 2025-01-01 .. 2026-02-13 |
| Calendar days | 409 |
| Dense user-days | 102,250,000 |
| Observed row fraction | 29.95697% |
| Duplicate `(user_id,event_date)` | 0 |
| Nulls in any raw field | 0 |
| Sample rows / unique users | 250,000 / 250,000 |
| Sample order | ascending `user_id` |
| Sample IDs | exactly sorted raw user universe |

Raw содержит не общую платформенную популяцию, а ровно ту же 250k-панель, для
которой нужен submission. Внепанельных пользователей нет, поэтому невозможно
построить unbiased counterfactual для пользователей, не прошедших финальный
отбор.

### A2. Точное правило отбора 250k

Воспроизведённый intersection:

```text
G1: хотя бы одна raw-строка 2025-11-16 .. 2025-12-15
G2: хотя бы одна raw-строка 2025-12-16 .. 2026-01-14
G3: хотя бы одна raw-строка 2026-01-15 .. 2026-02-13
TEST cohort = G1 ∩ G2 ∩ G3
```

Результат — ровно `250,000/250,000`. Минимум observed days в каждом блоке равен
1; медианы по блокам `10 / 9 / 10` дней.

Это selection on observed pre-cutoff activity. Нет локального свидетельства,
что организатор использовал скрытый target `2026-02-14..2026-03-15` при выборе
250k. Следовательно, факт попадания в TEST сообщает о последних 90 днях, но не
является скрытым условием на activity после cutoff.

### A3. Cutoff и target

```text
raw history
  2025-01-01 .. 2026-02-13, только sparse observed rows
        ↓
cohort selection
  active in each of G1/G2/G3, all conditions observable by 2026-02-13
        ↓
cutoff
  2026-02-13
        ↓
hidden target
  GMV30 = Σ gmv on 2026-02-14 .. 2026-03-15
        ↓
leaderboard split
  fixed public 50k / private 200k, exact membership and construction not supplied
```

Sample submission не содержит ни cohort flag, ни public/private marker.
Локальное описание фиксирует только доли `20%/80%`. Submission geometry
подтверждает, что public — один фиксированный 20%-subset, но 73 scalar scores не
идентифицируют одновременно 50k membership и неизвестный target. Random,
stratified или hash-based способ split остаётся `UNIDENTIFIABLE`.

### A4. Чем задача отличается от iid forecasting

Для настоящего test наблюдается распределение

```text
P(Y_future | history <= 2026-02-13, E_test=1),
```

где `E_test` полностью является функцией последних 90 дней history.

Для исторического fold `V` внутри поставленного файла фактически наблюдается

```text
P(Y_(V,V+30] | history <= V, E_V=1, G_final=1).
```

Лишнее условие `G_final=1` — survivorship selection. Оно находится далеко в
будущем относительно ранних folds и становится deterministic continuation для
позднего fold. Это не iid drift и не обычная recency feature.

### A5. Panel growth, maturation, censoring

Same-rule eligibility внутри уже отобранной universe растёт:

| Cutoff | Eligible | Share of final 250k |
| --- | ---: | ---: |
| 2025-04-03 | 170,705 | 68.28% |
| 2025-07-01 | 181,239 | 72.50% |
| 2025-10-01 | 193,522 | 77.41% |
| 2025-12-01 | 211,273 | 84.51% |
| 2026-02-13 | 250,000 | 100.00% |

Это смесь panel growth и cohort maturation внутри final-survivor universe, а не
рост всей платформы.

- first observed date median: `2025-01-06`; p90: `2025-06-17`; p99:
  `2025-12-01`; максимум `2025-12-15`;
- 17.4064% имеют первую строку ровно `2025-01-01`, поэтому их истинный tenure
  left-censored;
- observed days/user: min `3`, median `102`, p90 `254`, max `409`;
- 36.5172% имеют строку в сам cutoff; остальные right-gap до cutoff observable;
- early validation имеет только 247 дней потенциальной истории против 409 у
  TEST. Long-window normalization, `CAP180`, sequence clipping и availability
  mask уже были введены именно из-за этой асимметрии.

Пропущенный день и существующая нулевая строка не эквивалентны. Найдено
`4,549,734` raw rows, где все информационные count/GMV поля равны нулю. Сам факт
наличия такой строки является состоянием `present/presence-only` и уже доступен
TABULAR/SEQ/ETX.

## B. Mathematical gap

```text
current RMSLE                         1.646143314225527
target RMSLE                          1.644651494200000
Delta RMSLE                          -0.001491820025527

current MSE                           2.709787810969402
target MSE                            2.704878537374293
Delta MSE                            -0.004909273595110

required independent rho             0.042563857015516
required explained residual variance 0.001811681924037 = 0.181168%
```

Для correction `d` в log-space:

```text
Delta MSE = E[d²] - 2 E[r d].
```

При оптимальном масштабе одна совокупная correction должна иметь
`RMS(d)=0.07006621`, covariance `E[r d]=0.00490927`. Это маленькая доля всей
residual variance, но достаточно большая относительно уже найденных
out-of-span directions.

| Сценарий | rho каждого независимого signal | optimal RMS каждого | required covariance каждого |
| --- | ---: | ---: | ---: |
| 1 direction | 0.042564 | 0.070066 | 0.0049093 |
| 2 equal independent | 0.030097 | 0.049544 | 0.0024546 |
| 3 equal independent | 0.024574 | 0.040453 | 0.0016364 |

Global intercept математически мог бы закрыть gap только при оставшейся ошибке
уровня `|mean residual|=0.07007`. Existing level probes и LB-calibrated level не
дают такого свидетельства; это эквивалент масштаба, не найденная калибровочная
ошибка.

## C. Selection / survivorship effects

### C1. Past eligibility boundary

Eligibility strength имеет огромную связь с raw target. На четырёх folds:

| Minimum observed days in any of 3 blocks | `P(Y>0)` range | `E[Y]` range |
| --- | ---: | ---: |
| 1 | 0.301–0.314 | 32.6–34.4 |
| 2 | 0.374–0.388 | 40.9–43.9 |
| 3–4 | 0.478–0.484 | 58.4–59.5 |
| 5–7 | 0.598–0.608 | 83.1–86.8 |
| 8+ | 0.795–0.802 | 175.5–183.7 |

Это сильный обычный predictive signal, но он практически весь уже содержится в
production через activity levels, recency, trends и sequence path. После
production span:

| Boundary transform | Nominal oracle MSE | Debiased oracle MSE | Purged rho | Purged Delta MSE |
| --- | ---: | ---: | ---: | ---: |
| minimum block days | 0.000251 | 0.000188 | 0.00385 | -0.000009 |
| max end-gap | 0.000137 | 0.000075 | 0.00408 | -0.000050 |
| block imbalance | 0.000172 | 0.000048 | 0.00326 | +0.000081 |
| newest/oldest block ratio | 0.000295 | 0.000154 | -0.00111 | +0.000231 |
| joint min-days × end-gap | 0.000620 | 0.000243 | 0.00217 | +0.000243 |

`Purged` означает frozen mapping из fold `2025-09-04` с target end
`2025-10-04`, применённый к `2025-10-16`. Ни один mechanism не проходит oracle
`0.001`, observable `rho=0.020` или `Delta MSE=-0.001`.

### C2. Future continuation / survivorship oracle

Если использовать число активных 30-дневных блоков в `(V+30,V+120]`, после
production span появляется большая oracle information:

| Fold | Share `k=3` | Oracle headroom | Oracle rho |
| --- | ---: | ---: | ---: |
| 2025-09-04 | 96.16% | 0.008984 | 0.05360 |
| 2025-09-18 | 97.52% | 0.006138 | 0.04450 |
| 2025-10-02 | 96.89% | 0.004511 | 0.03840 |
| 2025-10-16 | 100.00% | 0.000006 | 0.00143 |

Этот oracle по масштабу может закрыть gap, но он не является разрешённым
signal: это post-target future activity. На `2025-10-16` три блока буквально
совпадают с G1/G2/G3, поэтому переменная константна. Для TEST аналогичные блоки
лежат после `2026-03-15`, не входят в schema и не участвовали в test selection.

EXP048/049 ранее показали тот же forensic mechanism с другой стороны:
selection-reweight shift исправленного estimand равен примерно `-4.64e-6`, а
максимальный incremental selection penalty для существующих candidates —
`0.000094 RMSLE`. Следовательно, survivorship важен как validation-bias, но не
найден как deployable correction.

## D. Calendar / boundary effects

### D1. Exact target-window composition

Все четыре canonical cutoff — четверги. Их target начинается в пятницу и имеет
по пять пятниц и суббот, всего 9 weekend days. TEST cutoff — пятница; target
начинается в субботу и имеет пять суббот и пять воскресений, всего 10 weekend
days. То есть относительно canonical composition пятая пятница заменена пятой
воскресной датой.

| Window | Weekend days | Days 1–5 | Day 15 occurrences | Days 25–31 | Month starts/ends |
| --- | ---: | ---: | ---: | ---: | ---: |
| canonical folds | 9 | 4–5 | 1 | 6–7 | 1 / 1 |
| 2026-02-14..2026-03-15 | 10 | 5 | 2 | 4 | 1 / 1 |

`2026-02-23` — Monday, `2026-03-08` — Sunday.

### D2. Oracle versus observable proxy

При ретаргетинге каждого исторического fold на окно, сдвинутое на один день,
фактическая будущая разность `log1p(Y_shift)-log1p(Y)` имеет weighted oracle
headroom `0.05697 MSE`. Это не usable result: разность использует выпавший и
добавленный будущий daily GMV и почти буквально раскрывает target.

Target-free proxy из индивидуальной прошлой Sunday-minus-Friday rate:

```text
weighted observable rho                 0.002900
debiased observable headroom            0.0000256 MSE
purged rho                               0.004466
purged Delta MSE                        -0.0000586
```

Более общий transform, который на всей cohort использует только прошлый
пользовательский DOW profile и shrunk day-of-month/payday profile:

```text
debiased after-span headroom             0.000031 MSE
purged rho                              -0.005087
purged Delta MSE                        +0.000045
```

Оба ниже gates на порядок и более.

### D3. Holidays

HOLIDAY-YOY уже существует как OOF/test component, присутствует в 40-vector
production bank и в submission geometry span. Его standalone public score
`1.650734` хуже поздних champions, хотя LOO geometry обнаружила неожиданное
positive alignment `-2.28 sd`; то есть небольшая независимая seasonal direction
уже используется.

Для конкретных `23 февраля / 8 марта 2026` отсутствуют pre-cutoff promotion,
exposure, inventory и business-intensity variables. История содержит один
предыдущий сезон, но не clean repeated treatment с сопоставимым weekday/context.
Поэтому exact holiday shock — `UNIDENTIFIABLE`, а не неиспользованный
deterministic correction.

## E. Cohort-relative / transductive signal

Все transforms строились на полной unlabeled cohort конкретного cutoff до
просмотра target: ranks, joint ranks, local ID density и 11-NN density в
rank-normalized state space. После этого они projected out of full production
span. Label mapping для purged test frozen на `2025-09-04` и применён к
`2025-10-16`.

| Candidate | Nominal oracle MSE | Debiased oracle MSE | Oracle rho | Purged rho | Purged Delta MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| GMV × recency joint cohort rank | 0.001221 | **0.000844** | 0.01998 | 0.00057 | +0.004090 |
| recency percentile | 0.000902 | 0.000760 | 0.01711 | -0.00725 | +0.001449 |
| activity percentile | 0.000486 | 0.000345 | 0.01158 | 0.00164 | +0.000408 |
| GMV percentile | 0.000393 | 0.000254 | 0.01094 | -0.00730 | +0.000695 |
| baseline percentile | 0.000373 | 0.000231 | 0.01094 | -0.00816 | +0.001321 |
| KNN density percentile | 0.000357 | 0.000215 | 0.01075 | -0.00459 | +0.000426 |
| tenure/availability percentile | 0.000251 | 0.000110 | 0.00880 | 0.00453 | +0.000133 |
| channel-mix percentile | 0.000104 | 0.000000 | 0.00580 | 0.00316 | +0.000001 |
| `user_id` rank | 0.000217 | 0.000000 | 0.00840 | 0.00065 | +0.000252 |
| local ID gap density | 0.000154 | 0.000040 | 0.00692 | 0.00456 | -0.000016 |

Nominal joint-rank value превышает `0.001`, но 25-cell oracle расходует 24
эффективные степени свободы. Его null expectation `0.000377`; conservative
debiased headroom `0.000844`. Даже если игнорировать поправку, observable
transfer полностью проваливается.

Это воспроизводит вывод EXP081/082: transductive rank-density в прежнем
EXP081-comparable backtest имел optimistic `0.000635`, но fully purged
`rho=-0.00339`, `Delta MSE=+0.001225`; behavioral prototype K128 также ухудшал
MSE. Новая реализация не нашла скрытую формулировку, меняющую вывод.

Empirical-Bayes population shrinkage и cohort calibration также фактически
закрыты: geometry использует EB/shrinkage на prediction population, а EXP080
segment intercepts после span дают максимум порядка `0.00066 MSE` для целой
segmentation и не переносятся стабильно.

## F. Raw-schema forensic audit

### F1. Exact identities

На всех raw rows:

```text
gmv = gmv_search + gmv_cat                  mismatches 0
to_ord = search_to_ord + cat_to_ord         mismatches 0
to_cart = search_to_cart + cat_to_cart      mismatches 0
search = 1[searches > 0]                    mismatches 0
has_* = 1[corresponding count > 0]          mismatches 0
gmv > 0  iff  to_ord > 0                    mismatches 0 in both directions
negative counts / negative GMV              0 / 0 rows
max floating gmv identity error             5.28e-11
```

Следствия:

- `GMV30 = sum daily GMV = sum gmv_search + sum gmv_cat` — exact conservation;
- `Y30>0` эквивалентен хотя бы одному будущему order day;
- total cart/order/GMV и четыре `has_*` не добавляют независимых raw degrees of
  freedom;
- reconciliation не может улучшить уже выданный единственный scalar forecast,
  если нет двух independently useful component predictors с различной residual
  geometry;
- hurdle/count/value coupling уже использовали DIST, E11, S04, BTYD, BLOCK4,
  EXP070/080/083. Fine future-shape oracle остаётся `0.001361 MSE`, но его
  observable purged proxy имеет `rho=0.000805` и ухудшает MSE на `+0.001051`.

### F2. Field-by-field table

| Raw field / structure | Already used? | How | Potential missed information | Headroom |
| --- | --- | --- | --- | ---: |
| `event_date` | YES | cutoffs, windows, DOW sequences, Holiday-YoY | exact target calendar composition | `0.000031` observable; fails purged |
| `user_id` | alignment only | key, user split, bootstrap | rank/order/density if assignment structured | `0.000000–0.000040` debiased |
| `search` | YES | day mask / sequence | none; exact `searches>0` | 0 identity |
| `cat` | YES | catalog-day activity | browsing presence not reducible to downstream count | already in TAB/SEQ/ETX |
| `has_search_to_cart` | YES, redundant | raw/sequence variants | none; exact count flag | 0 identity |
| `has_search_to_ord` | YES, redundant | raw/sequence variants | none; exact count flag | 0 identity |
| `has_cat_to_cart` | YES, redundant | raw/sequence variants | none; exact count flag | 0 identity |
| `has_cat_to_ord` | YES, redundant | raw/sequence variants | none; exact count flag | 0 identity |
| `search_to_cart` | YES | multiscale sums, sequence | conditional channel conversion | `0.000644` conditioned oracle; purged harm |
| `search_to_ord` | YES | multiscale sums, sequence | search funnel geometry | same EXP083 gate failure |
| `cat_to_cart` | YES | multiscale sums, sequence | catalog funnel geometry | same EXP083 gate failure |
| `cat_to_ord` | YES | multiscale sums, sequence | catalog funnel geometry | same EXP083 gate failure |
| `gmv_search` | YES | channel windows/sequences | separate-component reconciliation | no independent constraint gain shown |
| `gmv_cat` | YES | channel windows/sequences | separate-component reconciliation | no independent constraint gain shown |
| `to_cart` | YES, redundant | total cart features | exact channel sum | 0 identity |
| `to_ord` | YES, redundant | count/hurdle/sequence | exact channel sum; couples to positivity | oracle large, observable exhausted |
| `gmv` | YES, target identity | windows and `Y30` | exact channel sum | 0 deterministic gain |
| `searches` | YES | sums/sequences | within-day dispersion/Jensen geometry | `0.000212` conditioned; purged harm |
| absent day vs present-zero row | YES | `present`, `ponly`, availability mask | missingness pattern | boundary/density audit below gate |
| three-block eligibility | YES | exact panel construction | distance to selection boundary | `0.000243` debiased joint |
| full-cohort ranks/density | TESTED | EXP081 + EXP085 | relational position without graph | `0.000844` best oracle, no transfer |

Физически отсутствуют product/category identity (поле `cat` — activity flag, не
category ID), item/order/session/seller IDs, price/promotion/exposure/inventory
и любые межпользовательские связи. Поэтому настоящий graph, nearest-product
cohort или campaign-neighbor signal невозможно восстановить из raw.

## G. Submission-geometry forensic evidence

### G1. Observed score path

| Artifact/class | Public RMSLE |
| --- | ---: |
| best pre-geometry single | 1.6492175622 |
| geometry v2 | 1.6467120249 |
| geometry next best | 1.6466079084 |
| public EB | 1.6463246740 |
| ORTH_ALPHA anchor | 1.6461597403 |
| EXP075 joint current | **1.6461433142** |

Наибольшая реальная прибавка пришла не от новой user model, а от
submission-level geometry и shrinkage. Post-submit analysis оценивает перенос
public geometry gain на full test примерно в `0.47`; noise отдельной public
direction `sd≈0.0066` в correlation coordinates не уменьшается числом probes.

Broad/intermediate directions содержали положительный net signal
`0.001670/0.004854 MSE`; восемь узких seed/scale twin directions дали
`-0.002167 MSE` net signal. Следовательно, больше микровариаций существующих
моделей — неправильный профиль missing information.

LOO неожиданно положительно выровнял HOLIDAY-YOY, baseline HGBR, S04-A,
E11-mix и submission-v2. Общая закономерность — разные objectives/inductive
bias и широкие family directions, а не cohort rank или boundary trick.

### G2. EXP075 как самый важный transfer experiment

EXP075 clean-forward дал joint `rho=0.03794`, nested `Delta MSE=-0.004392` и
теоретически почти закрывал gap. На LB:

```text
anchor -> current observed Delta RMSLE       -0.00001643
observed MSE gain                             0.00005408
fraction of current required gap              1.10%
fraction of OOF-predicted nested MSE gain      1.23%
LB-implied rho from EXP076                     0.01793
```

Направление не было полностью пустым; оно оказалось примерно вдвое слабее по
rho и слишком большим по deployed norm. Даже при оптимальном масштабе
`rho=0.01793` его mathematical ceiling около `0.000871 MSE`, только 17.75%
оставшегося gap, или приблизительно `0.000265 RMSLE`.

Это сильное свидетельство против гипотезы «ещё одна capacity-модель на том же
history автоматически даст 1.64465». Математически возможно получить
`rho≈0.043` теми же полями, но лучший новый temporal representation не показал
нужного real-window transfer.

### G3. Что submission bank не говорит

Он измеряет публичные projections residual на span существующих файлов. Он не
раскрывает 50k mask, private residual или causal source direction. Unshrunk
public optimization может улучшать public и ухудшать full test. Поэтому score
`1.64465`, если это public score конкурента, может включать больше public-sample
luck/overfit, чем истинного generalizable signal.

## H. What a 1.64465 solution must contain

### H1. Геометрические варианты

1. **Одна сильная direction:** `rho≈0.04256`, correction RMS `0.07007`, covariance
   `0.004909`. Это сильнее joint EXP075 clean rho и в 2.37 раза сильнее его
   LB-implied rho.
2. **Две независимые:** каждая `rho≈0.03010`, RMS `0.04954`, covariance
   `0.002455`. Такой уровень по отдельности показывали A1/A2 offline, но их
   реальный transfer не подтвердился.
3. **Три независимые:** каждая `rho≈0.02457`, RMS `0.04045`, covariance
   `0.001636`. Это наиболее правдоподобная форма: несколько mid-width directions
   из разных information families.
4. **Global correction:** нужен оставшийся log-level error `0.07007`; level/LB
   evidence этого не поддерживает.
5. **Segment correction:** должен совместно дать тот же norm/covariance;
   существующие segmentations дают менее `0.00066 MSE` oracle после span.
6. **Selection structure:** future-continuation oracle имеет достаточную норму,
   но физически отсутствует для TEST и не участвовал в его selection.

### H2. Может ли это быть просто лучшая модель на тех же features?

Не невозможно, но эмпирически маловероятно как единственное объяснение.

- Чистая capacity/temporal line уже достигала offline rho близкого к требуемому,
  но реальный rho сократился до `0.01793`.
- EXP080 observable joint features имели optimistic headroom `0.002412`, но
  strict-forward point всего `0.000529`, robust lower bound `0.000035`.
- EXP081 same-period nonlinear residual models показывали `0.0013–0.00147 MSE`,
  но были same-cutoff user-crossfit, не temporal. EXP082 fully purged core дал
  `rho=-0.00173`, positive `Delta MSE`.
- EXP083 нашёл distinct fine-future-shape oracle `0.001361`, но purged proxy
  `rho=0.000805`, `P(gain)=0`.

Чтобы обычная model-capacity гипотеза была достаточной, нужна не просто меньшая
OOF ошибка, а representation, чей residual mapping устойчив к смене calendar,
panel survivorship и 120-day train→test gap. Это фактически новый mechanism,
даже если он реализован LightGBM/Transformer.

### H3. Что вероятнее есть у команды с 1.64465

В порядке правдоподобия:

1. несколько широких prediction families с различными inductive biases и более
   удачной full-test/public shrinkage, а не один secret scalar feature;
2. более сильный occurrence/activity-intensity signal, устойчивый на реальном
   Feb–Mar окне; late teammate occurrence давал реальный LB progress, но его
   canonical OOF lineage отсутствует;
3. rules-permitted causal context вне нашего schema: promotions/exposure,
   inventory/availability, item/category intent, session/order state или более
   длинная независимая history;
4. если речь именно о public leaderboard — больший вклад public-subset luck и
   более агрессивная, возможно переобученная, geometry;
5. менее вероятно — чисто более мощная модель на тех же агрегатах без нового
   temporal/selection mechanism.

## I. Ranked missing mechanisms

| Rank | Mechanism | Oracle headroom | Observable evidence | Expected gain | Verdict |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Pre-cutoff causal indicator of near-future activity shock | fine-shape oracle `0.001361` after full conditioning | current proxy purged `rho=0.000805` | unknown; could cover 27.7% if observable | `NOT_IN_SCHEMA` |
| 2 | Exact future survivorship / continuation state | `0.0045–0.0090` on early folds | none; post-target and constant on 10-16 | sufficient in oracle | `UNIDENTIFIABLE / FORBIDDEN` |
| 3 | New broad occurrence family with exact lineage | same-period EXP081 `0.0013–0.00147` | fully purged core fails; late teammate LB positive | uncertain | `AVAILABLE_BUT_NOT_IDENTIFIED` |
| 4 | Cohort GMV × recency rank | nominal `0.001221`, debiased `0.000844` | purged `rho=0.00057`, `Delta=+0.00409` | harmful | `REJECT` |
| 5 | Eligibility-boundary geometry | `0.000243` debiased | purged `rho=0.00217` | harmful | `REJECT` |
| 6 | Specific Feb23/Mar8/payday composition | exact future-day oracle large but target-derived | calendar profile `0.000031`, purged negative | no supported gain | `UNIDENTIFIABLE` |
| 7 | Funnel/channel consistency constraints | `0.000644` after span+levels | purged `rho=0.00128`, `Delta=+0.01895` | harmful | `REJECT` |
| 8 | ID/sample-order/density | `0–0.000040` debiased | purged near zero | none | `REJECT` |
| 9 | Exact algebraic reconciliation | zero new degree for scalar target | identities already exact | zero | `ALREADY_USED` |

## J. Authorized experiment(s)

```text
NONE
```

Mathematical gate status:

- no honest after-production structural oracle reaches `0.001 MSE` and also
  supplies observable `rho>=0.020` or strict-forward `Delta MSE<=-0.001`;
- cohort joint rank only crosses `0.001` before degrees-of-freedom correction
  and fails purged transfer catastrophically;
- calendar oracle is future-target-derived; both legal proxies fail;
- survivorship oracle is post-target and absent for TEST.

Следовательно, новая supervised model, nonlinear refinement, TEST correction и
submission запрещены. Выполнены только fixed mathematical transforms и
target-free cohort operations.

Следующий конкретный шаг — не новый learner:

1. получить/восстановить historical universe, включающую пользователей,
   **не прошедших** G1/G2/G3, и replay frozen production predictions на spaced
   cutoffs; это единственный способ идентифицировать survivorship correction;
2. либо получить новый pre-cutoff causal field (exposure/promo/inventory,
   product/category/session/order state) и сначала повторить oracle/purged gate;
3. для точного закрытия EXP081 ambiguity можно материализовать frozen 40-model
   bank на `35d` folds, но EXP082 core evidence делает это низкоприоритетным и
   само по себе не авторизует model search.

## K. Final verdict

```text
NO_NEW_STRUCTURE_FOUND
```

### Что исчерпано

#### ALREADY_USED

- exact 3-block eligibility и стандартные historical cohort-as-test panels;
- multiscale level/recency/frequency/value, missing-day/present-zero distinction;
- raw daily sequence/path shape, burst/gap/event-order/open-funnel;
- hurdle, count/value, BTYD, activity probability, distribution head;
- search/catalog composition, Holiday-YoY, platform detrending;
- global/segment calibration, submission geometry и EB shrinkage;
- cohort ranks, density, prototypes и ID fingerprints — повторно проверены здесь.

#### AVAILABLE_BUT_NOT_TESTED

- exact 40-model feature bank на полностью spaced/purged 35-day cutoffs; это
  дорогой artifact reconstruction, а не новый information source;
- late teammate occurrence source с canonical OOF lineage: TEST vectors есть,
  точных clean-fold predictions нет.

#### NOT_IN_SCHEMA

- пользователи, не прошедшие final selection;
- post-2026-02-13 activity;
- product/item/true-category/order/session/seller graph;
- exposures, ads, promotions, prices, inventory/availability, campaigns;
- organizer/business calendar beyond public date labels;
- независимый второй год для clean Feb23/Mar8 causal estimation.

#### UNIDENTIFIABLE

- точный способ выбора public 50k и их membership;
- private target/residual и full-test effect public-optimized geometry;
- target-level/platform shock февраля–марта 2026;
- counterfactual performance на non-survivors;
- знак и масштаб future-continuation correction для настоящего TEST.

### Итог в семи пунктах

1. Наиболее важная structural особенность — **final-survivor-conditioned raw
   universe** и вызванный ею mismatch между историческим validation и test.
2. Новый разрешённый signal, не являющийся обычной ML-feature, **не найден**.
3. Лучший честный новый headroom — `0.000844 MSE` у cohort joint rank, ниже gate;
   boundary `0.000243`, calendar `0.000031`.
4. Эти механизмы не объясняют gap `0.004909 MSE`; их purged transfer нулевой или
   вредный.
5. Команда с `1.64465`, вероятнее всего, имеет несколько новых broad prediction
   directions, дополнительный causal data channel или более агрессивный
   public-specific blend; одна простая selection/rank trick не согласуется с
   измерениями.
6. Следующий experiment на текущих данных: **NO-GO**. Сначала нужен unbiased
   non-survivor universe или новый causal raw field; затем тот же oracle/purged
   gate до обучения.
7. Минимально необходимое новое знание — organizer clarification о construction
   public/test panel либо данные о пользователях вне final 250k; наиболее ценные
   новые поля — future-exposure/promo/inventory/item/session context, известные
   до cutoff.

## Reproducibility artifacts

- `run_forensic_audit.py` — full raw/schema/panel/selection/cohort/calendar scan;
- `run_calendar_profile.py` — DOW + day-of-month target-free supplement;
- `gap_math.json`, `dataset_reconstruction.json`, `raw_identity_audit.json`;
- `raw_schema_audit.csv`, `panel_growth.csv`, `selection_boundary_outcomes.csv`;
- `future_survivorship_oracle.csv`, `structural_candidate_metrics.csv`;
- `calendar_audit.json`, `calendar_profile_metrics.csv`;
- `production_reconstruction.json`, `submission_geometry_evidence.json`;
- `audit_summary.json`.

Ни одна модель не обучена, TEST predictions и submission не созданы, новые LB
probes не использованы.
