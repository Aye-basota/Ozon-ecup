# Independent Stage-1 Data Audit

Дата фиксации: 2026-08-24. История предыдущих экспериментов не читалась. Это immutable baseline independent reasoning для последующего сравнения на этапе 2.

## DATASET DIAGNOSIS

- Raw panel: 30,631,006 строк, 250,000 пользователей, 409 непрерывных календарных дат `2025-01-01…2026-02-13`, 18 полей. Ключ `(user_id, event_date)` уникален; null и отрицательного GMV нет; физический порядок строго `user_id, event_date`.
- Target для cutoff `T`: `sum(gmv)` на `(T, T+30 days]`, то есть ровно 30 дат. `gmv = gmv_search + gmv_cat` с максимальной численной погрешностью `5.3e-11`; `to_cart` и `to_ord` также точные суммы Search/Catalog.
- `sample_submit.predict` не dummy: он побитово равен фактическому GMV raw-окна `2026-01-15…2026-02-13` для каждого пользователя. Это подтверждает границы и показывает, что sample — lag-30 baseline для production `2026-02-14…2026-03-15`.
- Test universe имеет точный construction fingerprint: у всех 250,000 есть хотя бы одна raw-строка в каждом из трёх 30-дневных блоков `2025-11-16…12-15`, `2025-12-16…2026-01-14`, `2026-01-15…02-13`. В четвёртом блоке это верно только для 233,152. `first_date <= 2025-12-15`, `last_date >= 2026-01-15` у всех.
- 4,549,734 строк (14.85%) полностью нулевые по всем видимым activity/GMV полям; они встречаются у 239,395 пользователей. Это отдельное состояние raw-row presence, не эквивалентное отсутствующему дню. В последнем 30-дневном блоке 5,299 пользователей удовлетворяют panel membership без какой-либо видимой activity, только через такие строки.
- `has_*` полностью избыточны относительно соответствующих count > 0; `search == (searches > 0)` точно. Но `cat` не выводится из downstream actions: в 4,038,859 строках он несёт самостоятельную activity-информацию.
- `user_id` не показал structure signal: Spearman с history depth/recency/activity/GMV находится около нуля (максимум по проверенным осям по модулю < 0.0033). Row order детерминирован ID и датой, отдельного сигнала не добавляет.

## INFORMATION LOSS

1. Exact anniversary window: для production полностью доступно то же календарное окно год назад `2025-02-14…03-15`; обычные `180d/all-history` агрегаты растворяют его.
2. Raw-row presence против absent day: densify с нулями смешивает две разные сущности и ломает точное panel-membership rule.
3. Three-block panel membership: historical examples по всем 250k не соответствуют production population.
4. Relative trajectory: level percentile почти бесполезен, но изменение percentile между соседними 30d regimes содержит incremental signal сверх raw trend.
5. Renewal geometry: `order_recency / median inter-purchase gap`, overdue phase, dispersion и last-gap теряются в sums/rolling ratios.
6. Discrete funnel transitions: одинаковый current state имеет разный target в зависимости от previous state.
7. Calendar-aligned global monetization state: stable search/active volume при падении conversion/AOV не выражается одной user-local историей.

## TRAIN/TEST GAP

- Production vs closest supervised cutoff (`2026-01-14`): recent-raw adversarial AUC `0.660`; главные оси — 60/90d row coverage, change in visible/zero-row days, order/catalog recency, recent GMV.
- Добавление history depth поднимает AUC до `0.965`; 93.6% gain принадлежит `history_age_days`. Это главным образом absolute-time/left-censoring fingerprint, а не behavioral feature для прямого использования.
- Между соседними 30d regimes searches/user почти стабильны (`35.50 -> 35.43`) и active days растут (`10.83 -> 11.18`, +3.2%), но orders падают `3.044 -> 2.800` (-8.0%), GMV/order `33.32 -> 30.01` (-9.9%), GMV/user `101.43 -> 84.03` (-17.1%). Это чистый intent × monetization regime shift.
- Последний observed 30d platform GMV равен 21.01m против 25.36m в предыдущем блоке (`0.8285x`), хотя active-user-days выросли `1.032x`.
- Exact three-block selection меняет supervised distribution. На cutoff `2026-01-14`: all-users zero mass `45.93%`, mean z `2.242`; three-block population `43.81%`, mean z `2.338`. На более старых cutoffs разница ещё больше. После удаления cutoff intercept three-block training лучше воспроизводит conditional shape closest validation (`1.8545 -> 1.8153` в грубом lookup), но без calendar calibration простое фильтрование может ухудшить общий intercept.
- Generic level percentiles не стабилизируют cutoffs и не являются рекомендуемым направлением. Raw + normalized feature pairs способны идеально восстановить cutoff-global denominator (domain AUC 1.0), но predictive value надо искать в interactions/changes, не в самом факте идентификации cutoff.

## TARGET ANALYSIS

- На последнем размеченном окне `2026-01-15…02-13`: zero mass `45.934%`; positive GMV q10/median/q90/q99 = `9.66 / 61.36 / 361.67 / 1396.29`; mean/std `log1p(target)` = `2.242 / 2.288`.
- По historical cutoffs all-user zero mass меняется `56.98% -> 43.69% -> 45.93%`; mean z `1.831 -> 2.417 -> 2.242`. Даже в matched three-block population mean z меняется `2.400…2.713…2.338`: calendar/global intercept материален.
- Current funnel state: buyer — future positive 74.4%, mean z 3.207; cart-no-order — 32.7%, z 1.176; search-no-cart — 23.1%, z 0.824; explicit-zero-only — 17.7%, z 0.632.
- Transition effect: среди current `cart_no_order` mean target z = `1.920`, если previous block был `buyer`, против `0.814` после `cart_no_order`, `0.696` после `search_only`, `0.644` после other.
- Renewal phase: среди users с >=3 purchase days и без orders в recent30 mean z растёт `1.032 -> 1.837`, а zero mass падает `70.5% -> 51.0%` от phase <=0.5 до phase >2. Coarse 5-fold lookup gain на сопоставимой подвыборке: `-0.0072` RMSLE.

## RESIDUAL ERROR MAP

OOF `STRONGEST_CURRENT` не предоставлены. `sample_submit` не OOF, а exact previous-30d GMV, поэтому подписывать его ошибку как residual current solution нельзя.

Диагностический proxy на последнем cutoff (остаток от coarse RFM-cell mean, не OOF сильной модели):

- recent buyers: 56.3% users, 64.5% squared-error mass;
- cart-no-order: 23.1% users, 20.8% error mass;
- search-no-cart: 17.9% users, 13.0% error mass;
- anniversary buyers: 36.7% users, 40.3% error mass.

Главные conditional axes внутри привычного RFM: anniversary behavior, previous→current funnel transition, renewal phase, и percentile trajectory.

## NEW SIGNAL SOURCES

### 1. Exact annual target-window lag

Для честного closest-cutoff falsification использовано `2025-01-15…02-13` как feature для target `2026-01-15…02-13`.

- Annual buyers: 36.7% пользователей; target-positive 76.1% против 41.3%.
- Spearman annual GMV vs current target z = `0.407`; Pearson `log1p(annual GMV)` vs target z = `0.414`.
- После coarse matching recent GMV/orders/searches/active days/order recency signed conditional difference mean target z = `+0.608` на 205,053 users.
- 5-fold conditional lookup: `1.8035 -> 1.7896` (`-0.0139`). Это не оценка сильной модели, но очень крупный falsification margin.
- Production anniversary feature существует для всех: окно `2025-02-14…03-15`, buyer share 40.3%, active share 78.3%.

### 2. Cross-sectional rank trajectory

- Level percentile сам по себе ухудшил closest-cutoff lookup и был менее stable across cutoffs, чем absolute level.
- `percentile(b0 GMV) - percentile(b1 GMV)` дал `-0.0211` против base lookup и `-0.034…-0.037` out-of-time.
- Обычный raw log-GMV trend объясняет основную часть (`-0.0286…-0.0308`), но percentile shift добавляет ещё `-0.00236…-0.00389` на том же out-of-time test.
- Daily-global-normalized trend не добавил сверх raw trend; его отдельно не рекомендуется продвигать.

### 3. Three-block panel construction

Production population — exact intersection трёх raw-row monthly panels. Historical states должны строиться тем же cutoff-safe rule, причём по row presence, а не по ненулевым aggregates. Selection существенно меняет zero mass и conditional target shape; её надо сочетать с cutoff/calendar calibration.

## GLOBAL TEMPORAL CONTEXT

- Daily shocks относительно centered 28d median: `2025-12-31` searches `0.527x`, orders `0.560x`, GMV `0.711x`; `2025-11-11` GMV `1.194x`, orders `1.177x`.
- DOW: Friday searches `0.939x`, orders `0.950x`; Sunday searches `1.063x`, orders `1.045x`.
- Exact prior-year production calendar window имеет mean GMV/user `66.93` и buyer share `40.29%`; preceding anniversary validation window — `57.56` и `36.73%`.
- Simple per-day global normalization не выдержал incremental control against raw trend; использовать стоит global regime constants/calendar interactions и rank shifts, а не десятки механических ratios.

## TOP 10 HYPOTHESES

| # | Hypothesis | Observed evidence | Why it may be missing | Expected upside | Cheap test | Compute |
|---|---|---|---|---|---|---|
| 1 | Exact anniversary target-window features | Conditional annual effect `+0.608 z`; lookup `-0.0139`; corr `~0.41` | `180d/all` windows destroy exact calendar alignment | **Potentially transformative: 0.003–0.008** | Add user GMV/orders/search/cart/row-days for target window minus 365d; one honest Jan-14 fold | CPU 10–30 min + 1 existing-model fold |
| 2 | Reconstruct three-block panel membership in training | 250k/250k satisfy b0,b1,b2; target z shifts `2.242 -> 2.338` at closest cutoff; centered conditional lookup improves `0.039` | Fixed-user historical snapshots include states impossible at production | **Potentially meaningful: 0.001–0.003** | Train/evaluate only users with raw row in each of 3 preceding 30d blocks; retain calendar calibration | CPU <1 h |
| 3 | Percentile/rank shift, not rank level | Incremental `-0.00236…-0.00389` beyond raw trend OOT | User-local model cannot know movement relative to population regime | **Potentially meaningful: 0.001–0.003** | Add b0/b1 percentiles and differences for GMV/orders/cart/search/row-days to one fold | CPU <30 min |
| 4 | Purchase renewal phase / overdue geometry | Same recent-order bins show large phase gradient; matched lookup `-0.0072` on 184k | Rolling counts omit personal cadence and phase | **Potentially meaningful: 0.001–0.003** | Median/robust gap, phase, gap CV, time-to-next-expected; one fold | CPU 15–30 min |
| 5 | Discrete previous→current funnel state | Current cart-no-order z `1.920` after buyer vs `0.644–0.814` otherwise | Independent rolling columns require high-order interaction to recover transition | **Medium to meaningful: 0.0007–0.002** | Encode 4×4 Search/Cart/Order state transition and duration in state | CPU <15 min |
| 6 | Global monetization regime × user state / intercept calibration | Search stable, active +3.2%, orders -8%, AOV -9.9%, GMV -17.1%; cutoff target mean z swings 0.33 | User-only histories miss common conversion/AOV shocks | **Potentially meaningful: 0.001–0.003** | Add cutoff global conversion/AOV/trends and per-cutoff bias; validate last two cutoffs | CPU <30 min |
| 7 | Preserve explicit-zero row geometry | 14.85% rows; 95.8% users; panel rule uses them; conditional effect ~0.11 z after coarse RFM | Densify or nonzero-day filtering collapses row-present and absent | **Medium: 0.0003–0.001; >=0.001 if currently discarded** | Count/recency/bursts of explicit-zero rows and use them in eligibility | CPU <15 min |
| 8 | Channel-specific anniversary and transitions | Search dominates GMV; Catalog GMV has closest-cutoff PSI `0.257`; raw cat flag contains standalone activity | Total GMV/orders merges different intent paths | **Medium to meaningful: 0.0005–0.0015** | Split annual/phase/state features by Search vs Catalog and test jointly | CPU 15–30 min |
| 9 | User calendar affinity × forecast-day composition | Strong DOW/holiday shocks; exact production calendar available one year back | Rolling sums erase which weekdays/calendar events drive the user | **Medium: 0.0003–0.001** | User DOW profile, holiday-window affinity, anniversary day matching | CPU <30 min |
| 10 | Rebuild empirical training weights/effective sample | Same-user non-overlapping 30d targets correlate `0.557–0.581`; weekly targets would overlap 76.7% mechanically | Many cutoffs overcount users and old regimes | **Medium to meaningful: 0.0005–0.0015** | Non-overlapping/recent cutoffs, per-user cluster weight, three-block eligibility; compare one fold | CPU <1 h |

## TOP 3 BETS

1. **Exact anniversary window.** Это единственный найденный user-level source, который одновременно новый, полностью inference-safe и имеет честную closest-cutoff проверку. Даже если strong model заберёт только 10–20% от lookup margin `0.0139`, improvement остаётся порядка `0.0014–0.0028`; при weak long-seasonality representation возможно >=0.003.
2. **Three-block training population + calendar calibration.** Production states лежат на явном selection manifold, которого нет у all-user historical examples. Zero mass и mean z меняются на несколько процентных пунктов/`~0.10 z` даже на closest cutoff. Это достаточно крупно для >=0.001, но фильтр нельзя применять без отдельной cutoff calibration.
3. **Cross-sectional percentile trajectory.** После контроля raw trend остаётся OOT gain `0.0024–0.0039` в грубом lookup. Это ровно тот population-relative signal, которого user-independent representation не содержит. Level ranks и daily normalization отвергнуты; ставка только на rank movement и его interactions.

## NEGATIVE / DEPRIORITIZED FINDINGS

- `user_id`, row order и row-group position — не самостоятельный signal.
- Generic percentile levels — хуже absolute levels по cross-cutoff stability и не дали closest-cutoff gain.
- Daily-global-normalized trend — не добавил к raw trend.
- `history_age` отлично различает domains, но standalone lookup ухудшился; использовать как construction/domain control, не как главную predictive feature.
- Explicit-zero features не дали standalone lookup gain при уже присутствующем row-count/RFM; их ценность главным образом в корректной semantics и panel selection.

## LIMITATION

OOF predictions `STRONGEST_CURRENT` отсутствуют, поэтому настоящая residual map текущего решения не построена. Все RMSLE deltas выше — дешёвые nonparametric diagnostic lookups, не обещание такого же gain поверх текущего LB model.
