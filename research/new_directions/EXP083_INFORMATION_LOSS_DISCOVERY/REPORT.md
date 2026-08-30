# EXP083 — Information-loss / target-discovery audit

## Verdict

`PROMISING_BUT_INSUFFICIENT`

Fine-grained future activity shape (`Y1/Y3/Y7/Y14` and five increments) is the only genuinely new mechanism that passes the mathematical oracle gate. After removing the full historical production span, future count/value/activity structure and the coarse `7/14/30` timing block from EXP080, it retains `0.001360758 MSE` of oracle headroom. Thus the information exists and is not just the old three-window decomposition.

The target-free proxy does not transfer. Four clean, embargoed Ridge heads for `Y1/Y3/Y7/Y14` leave an optimistic same-fold `0.001191805 MSE` after span plus activity-level controls, but the only fully label-available purged residual transfer `2025-09-04 -> 2025-10-16` has `rho=0.000805`, `Delta MSE=+0.001051`, bootstrap `P(gain)=0`. The observable gate fails, so no full model, focused refinement, TEST inference or submission is authorized.

The production-like residual and basis are exactly the audited EXP080 construction:

```text
B_T = [1, z_current, z_match, 40 clean OOF production components, EXP075_postspan]
r_T = target_log - z_current
```

For a candidate design `D`, `U=(I-P_B)D`, `G=U'U/N`, `b=U'r/N`, and the same-fold oracle gain is `b'G^+b`. Purged coefficients are fitted only on `2025-09-04` (`target_end=2025-10-04`) and applied unchanged on `2025-10-16`.

## A. Information/feature audit

Raw data contains 30,631,006 daily user rows, 250,000 users and 18 columns. Only ten fields are informational: date, user, `cat`, `searches`, four channel-specific cart/order counts and two channel GMVs. The identities `gmv=gmv_search+gmv_cat`, `to_ord=search_to_ord+cat_to_ord`, `to_cart=search_to_cart+cat_to_cart`, all `has_*` flags, and `search=1[searches>0]` are exact. There are no product, item, true category, order, session, seller, price, promotion, exposure or inventory identifiers.

| Raw information | Existing transformations/features | Model families using it | Potentially lost information |
| --- | --- | --- | --- |
| Presence versus absent day | `present/buy/ponly`, days-present and presence-only rates, sequence mask | TABULAR, SEQ, ETX, DIST/CAP | No material loss; zero row and absent day are distinct |
| Total activity and GMV levels | 7/14/30/60/90/180/365 sums, rates, recencies, trends | TABULAR, DIST/CAP/UNC, SEQ/ETX | Fine within-window distribution is lost in tabular but available to sequence models |
| Search/catalog funnel counts | Canonical tabular keeps total `cart2ord` and `srch2cart`; raw channels go to SEQ/ETX | TABULAR, SEQ, ETX | Channel-specific conditional rates and search-vs-catalog disagreement are not explicit in the 227 features |
| Search/catalog monetary composition | Per-window `gmv_cat_share`, raw channel sequences, EXP052 Shapley heads | TABULAR, SEQ/ETX, EXP052 | Channel AOV and conditional conversion geometry are not explicit; future-composition heads were already rejected |
| Temporal event order | A1/A2, SEQ/ETX, OPEN-FUNNEL and EVENT-ORDER | SEQ, ETX, A1/A2 | Requested transition/lag descriptors are prior duplicates |
| Bursts and gaps | Recency, gap mean/std/CV and threshold-3 episode summaries | TABULAR/BTYD, EXP054 | Burst duration/density/open-gap representation was already tested |
| Future horizon state | MHZ hazard/count at 7/14/21/30/45/60; EXP080 coarse `7/14/30` oracle | MHZ, DIST, EXP080 | `Y1/Y3` and fine early-share shape had not been isolated after count/value/coarse-timing conditioning |
| Calendar/platform phase | DOW channels, Holiday-YoY, platform detrending | SEQ/ETX/HOLIDAY-YOY | No external causal exposure channel exists |
| Relational structure | User alignment and behavioral prototypes only | EXP081 | Real graph/neighbor information is absent from schema |
| Within-day funnel distribution | Raw daily counts; only aggregate ratios/binary transitions downstream | SEQ/ETX implicitly | Mean-of-ratios, dispersion and Jensen gaps are not explicit |

The exact audit and prior mapping are in `feature_information_audit.csv` and `novelty_audit.csv`. Funnel features were decoded from the same float32 `log1p` panel consumed by production sequence models. Against exact `w30_searches`, RMSE is about `0.0185` count and relative RMS `2.8e-4`; max error is `1.21` only at totals near two thousand. This precision is immaterial to the negative funnel gates, but it is recorded rather than called bitwise parity.

## B. Multi-horizon future-intensity analysis

### Oracle-only results

Cumulative targets and disjoint increments were rebuilt from exact GMV arrays:

```text
Y1, Y3, Y7, Y14, Y30
I1, I2_3, I4_7, I8_14, I15_30
```

Target parity against canonical `Y30` is at most `2.18e-11`. Individual weighted oracle diagnostics are:

| Candidate | Raw rho | After-span rho | After-span MSE | After count/value/activity MSE |
| --- | ---: | ---: | ---: | ---: |
| Y1 | 0.1385 | 0.1448 | 0.06404 | 0.000211 |
| Y3 | 0.2443 | 0.2680 | 0.21950 | 0.000372 |
| Y7 | 0.3759 | 0.4412 | 0.59467 | 0.000669 |
| Y14 | 0.5255 | 0.6587 | 1.32578 | 0.001388 |
| I2_3 | 0.2053 | 0.2198 | 0.14764 | 0.000285 |
| I4_7 | 0.2958 | 0.3302 | 0.33323 | 0.000499 |
| I8_14 | 0.3950 | 0.4593 | 0.64513 | 0.000670 |
| I15_30 | 0.5846 | 0.7287 | 1.62232 | 0.001945 |

Large value/increment gains are not treated as independent timing evidence: future amounts partly reveal the target itself. The novelty test uses only eight future ratios/decay descriptors plus the five-increment activity pattern (72 fixed columns), and conditions on activity, three count definitions, conditional value, and EXP080's coarse `7/14/30` activity/early/late-share design.

| Joint oracle | Raw MSE | After production span | After count/value/activity | After EXP080 coarse timing |
| --- | ---: | ---: | ---: | ---: |
| Cumulative `Y1..Y14` | 0.570553 | 1.327495 | 0.001480 | 0.010153 |
| All increments | 0.689122 | 2.255791 | 0.011335 | 0.022939 |
| Fine future shape only | 1.587915 | 2.586056 | 0.001695 | **0.001361** |

The fine-shape value is stable by fold: `0.001302 / 0.001208 / 0.001441 / 0.001366`. It clears the `0.001` oracle gate on every fold and is not the `0.000289` coarse timing headroom already closed by EXP080.

### Observable short-horizon heads

Gate A authorized one cheap pilot. For every validation cutoff, one multi-output Ridge was trained on four historical snapshots at lags `63/49/35/21` days. The longest label is `Y14`, and its latest training target ends seven days before validation. `Y30` and production residual were never training targets.

Head quality is real but modest and stable:

| Head | Target correlation range | Latest log-RMSE |
| --- | ---: | ---: |
| p1 | 0.2760–0.2857 | 0.8139 |
| p3 | 0.4068–0.4206 | 1.2376 |
| p7 | 0.5167–0.5294 | 1.5265 |
| p14 | 0.5904–0.5999 | 1.6839 |

The 11 observable descriptors are `[p1,p3,p7,p14]`, three nonnegative predicted increments, three early shares, and an early/late intensity log-ratio.

| Evaluation stage | Joint rho | Optimistic headroom |
| --- | ---: | ---: |
| Raw binned design | 0.02134 | 0.009237 |
| After strong level conditioning | 0.02021 | 0.001258 |
| After full production span | 0.02038 | 0.001279 |
| After span + strong levels | 0.01967 | 0.001192 |

This same-fold mapping is not temporal evidence. On the fully purged residual transfer:

```text
rho                         = 0.000804687
Delta MSE                   = +0.001050872  (worse)
Delta RMSLE                 = +0.000301621
95% bootstrap CI Delta MSE  = [+0.000526, +0.001605]
P(gain)                     = 0.000
```

The best individual descriptor, predicted early/late intensity, rises from post-span-plus-level rho `0.00036` to `0.01489` across the four periods, but remains below the `0.015` pilot floor and its source-period residual map does not transfer. Therefore Branch A stops before a nonlinear head or refinement.

## C. Funnel/channel geometry

The fixed 16-candidate set includes search/catalog cart and order rates, channel order/cart shares, channel AOV, log conversion disagreement, stage-efficiency disagreement, entropy, and 7-vs-90-day elasticities. Exact duplicates such as total `to_ord/to_cart` and `gmv_cat/gmv` were not reintroduced.

```text
raw joint oracle headroom                  0.009946761   rho 0.02613
after strong activity/value levels        0.000946383   rho 0.01756
after full production span                0.001408402   rho 0.02144
after span + strong level controls        0.000643578   rho 0.01447
```

The apparent span-only result is level-mediated: once the requested strong baseline conditioning and the production span are both enforced, oracle headroom falls below `0.001`. The purged binned pilot has `rho=0.001280`, `Delta MSE=+0.018947`, CI `[+0.016725,+0.021216]`, `P(gain)=0`. Individual post-span-plus-level correlations are tiny; the strongest weighted ones are about `0.00269` for search cart rate and `0.00250` for search order share.

Verdict: `REJECT_GATE`. Geometry exists in raw data and is partially absent from canonical tabular features, but does not predict a stable error beyond production and activity levels.

## D. Temporal-phase analysis

No new model or reparameterization was run because the requested descriptors map directly to completed experiments:

| Requested descriptor | Closest completed mechanism | Verdict/evidence |
| --- | --- | --- |
| Time since last burst, duration, density | EXP054 BURST-GAP | `SKIP_DUPLICATE`; raw probe rho only `0.00699/0.01164`, donor scales both zero |
| Inter-event gap compression/expansion | EXP054 plus canonical gap CV and EXP075 | `SKIP_DUPLICATE` |
| Channel transition imbalance | EXP064 EVENT-ORDER | `SKIP_DUPLICATE`; all REAL/SHUFFLED/control scales zero |
| Purchase-after-search lag/open intent | EXP061 OPEN-FUNNEL | `SKIP_DUPLICATE`; all correction scales zero |
| Recent-versus-long phase | Canonical trends, BLOCK4, EXP080 states | `SKIP_DUPLICATE`; generic trend/acceleration explicitly closed |

This is not a claim that every phase representation is impossible. It is a novelty decision: the specific descriptors requested do not define a new mechanism relative to EXP054/061/064 and EXP075/080.

## E. Other discovered mechanism

One additional raw-information candidate was allowed: within-day funnel coherence. Eight fixed descriptors measure completion-day rates, Search/Cart-to-order same-day completion, mean-of-ratios minus ratio-of-sums (Jensen gap), daily conversion dispersion, catalog order burst CV, and dual-channel order-day share.

```text
after production span oracle              0.000715165
after span + strong levels                0.000212039
purged rho                                -0.000676751
purged Delta MSE                          +0.001444211
95% CI                                    [+0.000866,+0.002010]
P(gain)                                    0.000
```

It fails the oracle gate before any model. The information exists in raw daily counts but is also implicitly available to SEQ/ETX; explicit invariants expose no qualifying incremental direction.

## F. Mathematical ranking

`Observable MSE` below is the optimistic same-fold feature-space headroom. `Expected gain` is the actual purged `Delta MSE` (negative would be good). Strict-forward headroom is zero for all observable branches.

| Mechanism | Oracle MSE | Observable MSE | Purged rho | Expected gain | Verdict |
| --------- | ---------: | -------------: | ---------: | ------------: | ------- |
| Fine multi-horizon future shape | **0.001360758** | 0.001191805 | 0.000805 | +0.001050872 | `PROMISING_BUT_INSUFFICIENT` |
| Funnel/channel geometry | 0.001408402 span-only; 0.000643578 conditioned | 0.000643578 | 0.001280 | +0.018947497 | `REJECT_GATE` |
| Temporal phase descriptors | N/A | N/A | N/A | N/A | `SKIP_DUPLICATE` |
| Within-day funnel coherence | 0.000715165 | 0.000212039 | -0.000677 | +0.001444211 | `REJECT_ORACLE` |

Purged `rho^2` is `6.48e-7`, `1.64e-6`, and `4.58e-7` respectively—orders of magnitude below the required `0.00181168`. On the latest-fold oracle maps, fine future shape is nearly orthogonal to funnel geometry (`-0.00839`) and within-day coherence (`+0.01165`); funnel/coherence correlation is `0.11475`. These directions are not summed: joint deployable covariance is unsupported because none transfers.

## G. Executed experiment

Only the Gate-A-authorized pilot was executed:

- four fold-specific multi-output Ridge fits;
- 108 existing cutoff-safe history features;
- targets only `Y1/Y3/Y7/Y14`;
- 7-day embargo after the longest training target;
- 733,030–755,228 training rows per fold;
- CPU only, about 155 seconds; no GPU;
- predictions saved in `multi_horizon_head_predictions.parquet`.

No full experiment was permitted because observable `rho < 0.015`. There was no nonlinear refinement, TEST prediction, leaderboard use or submission.

## H. Remaining attainable gap

```text
required MSE gain                       0.004909274
new distinct oracle-only headroom       0.001360758   (27.72%)
optimistic observable same-fold         0.001191805   (24.28%, invalid temporally)
strict-forward proven headroom          0.000000000   (0.00%)
remaining proven gap                    0.004909274
```

If the oracle signal were perfectly observable, it would move `1.6461433142` only to approximately `1.6457299454`, still short of `1.6446514942`. The current observable proxy supports no improvement at all; its purged application is harmful.

The missing raw channel is now specific: a pre-cutoff causal indicator of upcoming short-horizon activity shocks—such as exposure, campaign/promotion, inventory/availability, item/category intent, order/session state, or an independent non-survivorship cohort in which the future-state map can be learned. Daily aggregate history predicts the short targets themselves, but not the stable sign of their incremental production error.

## I. Verdict

```text
PROMISING_BUT_INSUFFICIENT
```

1. A genuinely distinct fine future-shape oracle was found.
2. Production does not contain all of it: after full span plus structural conditioning, oracle headroom is `0.001360758 MSE`.
3. The available target-free short-horizon predictions retain same-fold headroom but no purged signal (`rho=0.000805`, gain probability zero).
4. Multi-horizon passes the oracle gate and fails the observable gate.
5. Funnel geometry, phase descriptors and within-day coherence do not authorize models.
6. No next coding/model experiment is authorized on current data. The next valid action is to add a genuinely new causal raw channel or independent spaced clean cutoffs and replay the frozen short-horizon gate; another LightGBM/encoder on the same history would violate the result.

## Reproducibility

- `run_discovery.py` — raw/feature audit, oracle analysis, funnel/coherence diagnostics and purged bootstrap;
- `run_multihorizon_heads.py` — the only gate-authorized observable pilot;
- `multi_horizon_*`, `observable_*`, `mathematical_ranking.csv`, `candidate_direction_correlations.csv` and `audit.json` — primary numeric artifacts.
