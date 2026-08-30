# EXP081 — Adversarial falsification of EXP080 `NO_EVIDENCE`

## Verdict

**NO_EVIDENCE_CONFIRMED.** Математическая формулировка EXP080 была слишком сильной:
`0.000529 MSE` — не universal upper bound на всю sigma-algebra разрешённых признаков, а point
estimate конкретного 38-колоночного линейного forward-блока. Linear projection также не доказывает
conditional sufficiency. Эти предпосылки атакованы явно.

Однако ни nonlinear interaction, ни model disagreement, ни latent-state/tail routing, ни
relational prototype, ни transductive cohort transform не дали переносимого signal после полного
40-component production-like span. Единственный apparent nonlinear success возник при обучении на
residual labels того же cutoff у других пользователей. Он исчез уже на более оптимистичном
ordered-time переносе и резко ухудшил MSE в единственном полностью target-availability-purged
переходе `2025-09-04 -> 2025-10-16`.

Ни один production experiment, TEST correction или submission не авторизован.

## A. EXP080 reproduction

Primary hashes совпали с EXP080:

- raw `train.parquet`: `5f3aa909...67c0`;
- canonical EXP075 OOF: `d3e8893b...6123`;
- 770,616 canonical rows на четырёх clean cutoffs: 188,518 / 191,025 / 193,694 / 197,379;
- запрещённый target-derived `oof_BLOCK4_SAF.activity` не загружался;
- `user_id` использован только для alignment, user-disjoint cross-fitting и clustered bootstrap.

Gap пересчитан непосредственно из заданных scores:

```text
current RMSLE                  1.646143314225527
target RMSLE                   1.644651494200000
current MSE                    2.709787810969402
target MSE                     2.704878537374293
required Delta MSE gain        0.004909273595110
required independent rho       0.042563857016
```

38-column EXP080 basis восстановлен из primary `observable_predictions.parquet`, seed arrays,
state dummies и полного foldwise basis
`[1, z_current, z_match, 40 clean OOF components, EXP075_postspan]`.

| Quantity | EXP080 | EXP081 reproduction | Absolute difference |
| --- | ---: | ---: | ---: |
| optimistic observable headroom | 0.002412030689 | 0.002412030689 | 4.34e-19 |
| forward point headroom | 0.000528689945 | 0.000528689945 | 0 |
| latest forward rho | 0.0151272735 | 0.0151272735 | 0 |
| robust 95% lower headroom | 0.0000352440 | 0.0000340351 | 1.21e-6 |
| `P(Delta MSE < 0)` | 0.983 / 1,000 reps | 0.9835 / 2,000 reps | bootstrap MC only |

TEST/submission geometry также пересобрана из primary `Z.npz` и 11 local submissions:
78 vectors, centered rank 67. Canonical historical bank содержит 40 clean OOF components; exact
TEST counterpart имеется для 16. Все 16 были реально загружены для TEST-cohort shift audit.

### Temporal audit of the word “forward”

Фолды разнесены на 14 дней, а label horizon равен 30 дням. Поэтому residual label предыдущего
canonical fold не доступен на следующем cutoff:

| Validation cutoff | Latest earlier canonical target end | Available at cutoff? |
| --- | --- | --- |
| 2025-09-18 | 2025-10-04 | no |
| 2025-10-02 | 2025-10-18 | no |
| 2025-10-16 | 2025-11-01 if all earlier folds used | no |

Это не завышает доказанную доступную информацию; наоборот, делает EXP080 `0.000529` оптимистичным.
Единственный canonical residual fold, полностью известный к `2025-10-16`, — `2025-09-04`, target
end `2025-10-04`. Все decisive mechanisms поэтому дополнительно проверены этим purged переходом.

## B. Weak assumptions attacked

| Assumption | Test | Result | Loophole? |
| ---------- | ---- | ------ | --------- |
| Ridge blocks exhaust observable structure | two fixed shallow LightGBM configs, full 191-feature space | same-cutoff OOF passes numerical gate, ordered/purged transfer fails | No; regime-conditioned mirage |
| linear projection implies conditional sufficiency | 18 fixed products/rank interactions, projected after full span | optimistic 0.001037; forward 0.000161; latest rho 0.01117; purged Delta +0.001303 | No |
| full 40-bank projection is harmless | compare current-only, 16-deployable, full-40 spans | current-only passes, 16/full fail; evidence is contained in known bank directions | No primary loophole |
| aggregation loses disagreement information | 14 fixed disagreement features | forward 0.000130; latest rho 0.01041; purged Delta +0.000653 | No |
| fixed segments miss mixtures | target-free K=4 state, fixed from first fold | forward 0.000208; latest rho 0.00904; purged Delta +0.000563 | No |
| large-error tail is observable and correctable | top 5/10/20% classifiers plus signed routing | AUC transfers, signed correction does not | No |
| behavioral neighbours add relational information | K=128 target-free prototypes | forward Delta +0.001730; purged Delta +0.005189 | No |
| historical validation misses transductive TEST information | cohort ranks, density, cluster share fitted on whole unlabeled validation cohort | forward Delta +0.000141; latest rho 0.00327; purged Delta +0.001225 | No |
| global/model-family mixture is stable | cluster family winner and signed residual by fold | state residual means change sign; family winner drifts | No |

## C. Nonlinear conditional signal

### Why EXP080’s projection is not a theorem

Let `S = span(1, z_1, ..., z_k)` and `P_S` be the sample least-squares projection. Even if
`z_i in S`, generally `h = z_i z_j` is not in `S`. The post-projection candidate

```text
u = (I - P_S) h
```

is non-zero except in a degenerate finite-sample geometry. Its best scalar residual correction has

```text
beta* = <u,r> / <u,u>
Delta MSE gain = <u,r>^2 / <u,u> >= 0.
```

For a matrix `U` of interactions the exact FWL headroom is `b' G^+ b`, where
`G = U'U/N` and `b = U'r/N`. Thus a zero marginal linear result cannot prove
`E[r | sigma(Z,X)] = 0`. EXP080 bounded its chosen linear representation, not all measurable
functions. Это конкретная математическая слабость исходного wording.

### Fixed interaction basis

Fixed before scoring: `z*p_active`, `z*count_hat`, `z*value_hat`, family products,
disagreement by activity/count, variance interactions, rank products and squared family gaps —
18 columns total.

| Projection | Optimistic headroom | Forward headroom | Latest rho | Positive folds | P(gain) | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| current prediction only | 0.003529 | 0.001187 | 0.02357 | 3/4 | 0.998 | numeric PASS, non-primary |
| 16 exact deployable components | 0.001445 | 0.000375 | 0.01449 | 2/4 | 0.9645 | FAIL |
| full 40-component span | 0.001037 | 0.000161 | 0.01117 | 2/4 | 0.824 | FAIL |

Current-only result proves that full-bank projection materially changes the estimand, but it does
not prove new information outside the production component bank. Under the requested full
post-span test the fixed interactions fail. In the fully purged first-to-latest transfer they have
`rho=0.00234`, `Delta MSE=+0.001303`, 95% CI `[+0.000648,+0.001918]`.

### Nonlinear pilots and diagnostic upper-bound learner

Inputs: 40 production predictions, current proxy, nine structural predictions, all 108 cutoff-safe
RFM/channel features, 14 disagreement features, 18 interactions and ten cohort-relative/density
features — 200 columns. Models were frozen before running:

- LightGBM A: depth 3, 8 leaves, 180 trees, `min_child=1500`, strong L1/L2;
- LightGBM B: depth 5, 20 leaves, 240 trees, `min_child=1200`, stronger L1/L2.

Target is the production-like residual. Predictions are constant-removed and projected through the
full 40-component basis twice.

| Protocol | Candidate | Delta MSE | Latest rho | Positive folds | P(gain) |
| --- | --- | ---: | ---: | ---: | ---: |
| same-cutoff user-disjoint OOF + forward scalar | A | -0.001334 | 0.02063 | 3/4 | 1.000 |
| same-cutoff user-disjoint OOF + forward scalar | B | -0.001440 | 0.02176 | 3/4 | 1.000 |
| same-cutoff user-disjoint OOF + forward scalar | A/B mean | -0.001469 | 0.02183 | 3/4 | 1.000 |
| ordered earlier canonical folds, unit prediction | A | **+0.000584** | 0.01309 | 1/4 | 0.0385 |
| purged `09-04 -> 10-16`, OOF-calibrated | A | **+0.000752** | -0.00102 | single latest | 0.0005 |

The same-cutoff learner is a useful falsification diagnostic: it proves nonlinear conditional
structure exists cross-sectionally. It is not strict temporal evidence because its functional map
was fitted using other users’ outcomes from the same future period. The effect does not transfer
even under the less strict ordered protocol, and the purged test reverses sign. Therefore loophole
A is closed for deployment.

The best cross-sectional full-span optimistic headroom is `0.001564`, below EXP080’s already larger
linear optimistic headroom `0.002412`. Nonlinearity changes forward-looking appearance, not the
optimistic total bound.

## D. Model disagreement

Tested target-free per-user mean, std, q90-q10, MAD, SEQ-vs-TAB, ETX-vs-TAB,
DIST-vs-TAB, OTHER-vs-TAB, rank gaps, unusually-high/low fractions and asymmetric spreads.

Primary full-span joint result:

```text
optimistic headroom        0.000450
forward headroom           0.000130
latest rho                 0.010409
positive folds             1/4
P(Delta MSE < 0)           0.951
95% CI Delta MSE          [-0.000283, +0.000023]
purged latest rho          -0.002296
purged latest Delta MSE    +0.000653
```

Largest latest individual post-span absolute correlation is only `0.00822`
(`fraction_unusually_high`). Raw SEQ/TAB and ETX/TAB disagreement correlations near `0.017`
collapse to about `0.001` after the span, so aggregation did not hide a qualifying conditional
direction.

The exact 16-component TEST disagreement distribution is shifted relative to latest clean history:

| Metric | Historical mean | TEST mean | KS statistic |
| --- | ---: | ---: | ---: |
| model mean | 2.6510 | 2.4427 | 0.0672 |
| model std | 0.1539 | 0.1706 | 0.0935 |
| q90-q10 | 0.2482 | 0.2947 | 0.1792 |

This makes historical calibration less trustworthy, not more. Cohort-rank normalization was
therefore tested explicitly in section F and failed.

## E. Mixture / tail / relational signal

### Latent mixture

K=4 was fixed a priori and fit from 12 target-free state features on the first fold. Cluster shares
remain roughly stable, but mean signed residual changes from negative on `2025-09-04` to positive
on later folds in every state. The best production family also drifts between OTHER, SEQ and ETX.

```text
optimistic headroom        0.000521
forward headroom           0.000208
latest rho                 0.009038
positive folds             2/4
purged latest rho          -0.007341
purged latest Delta MSE    +0.000563
```

### Tail precursor

Tail membership is observable as variance:

| Top absolute residual | Cross-fitted AUC range | Purged latest AUC |
| --- | ---: | ---: |
| 5% | 0.700–0.739 | 0.706 |
| 10% | 0.684–0.724 | 0.694 |
| 20% | 0.714–0.742 | 0.727 |

But the signed correction is not transferable. Same-cutoff routed forward headroom is 0.000700 /
0.000735 / 0.000883 and latest rho 0.01842 / 0.01709 / 0.01845: all below the gate. In the purged
latest test:

| Tail | rho | Delta MSE | P(gain) |
| --- | ---: | ---: | ---: |
| 5% | 0.00277 | -0.000005 | 0.5365 |
| 10% | -0.00063 | +0.000137 | 0.0840 |
| 20% | -0.00369 | +0.000975 | 0.0000 |

Conclusion: the features can identify high conditional variance, not the sign of the production
error. Per instruction this is **REJECT**, not a specialist candidate.

### Relational information

Raw schema has `event_date`, `user_id` and 16 aggregate behavior/count/GMV columns. There is no
product, item, category, seller, order, session, price or other entity identifier. `cat` is a
channel/activity flag, not an entity relation. Therefore legitimate graph/co-purchase neighbours
cannot be formed, and `user_id` memorization was prohibited.

The available behavioral-fingerprint alternative was tested with 128 target-free cohort
prototypes. It worsens forward MSE by `+0.001730`; the purged latest prototype has `rho=0.00011`
and `Delta MSE=+0.005189`. This closes the relational channel present in the current schema.

## F. Transductive signal

For each validation fold the entire cohort was treated as an unlabeled TEST cohort. Ranks,
32-cluster density distance and cluster population share were fit on that whole cohort without
targets, then targets were evaluated only afterward.

```text
optimistic headroom        0.000635
forward Delta MSE         +0.000141
latest rho                 0.003266
positive folds             1/4
P(gain)                    0.1565
purged latest rho         -0.003390
purged latest Delta MSE   +0.001225
```

The backtest is honest for cohort-level transforms and gives no evidence that TEST ranks/density
make the historical residual signal observable.

## G. Strong observable upper-bound learner

The strongest same-cutoff diagnostic produces post-span `rho=0.02183` and point gain `0.001469`,
but its training protocol is cross-sectional, not temporal. The properly ordered learner gives
latest `rho=0.01309`; the only label-availability-purged learner gives approximately zero rho and
positive loss. Consequently the requested strong claim must be stated carefully:

```text
Cross-sectional nonlinear structure exists:                 YES
Stable strict-temporal nonlinear residual direction exists: NO_EVIDENCE
```

This result explains why a generic model search would be dangerous: a clean user split can pass
all numerical gates while learning a period-specific residual map that reverses out of time.

## H. Revised attainable headroom

| Quantity | MSE headroom | Fraction of required gap |
| --- | ---: | ---: |
| required Delta MSE | 0.004909274 | 100.0% |
| EXP080 robust 95% headroom | 0.000035244 | 0.72% |
| revised optimistic full-span headroom | 0.002412031 | 49.13% |
| EXP080-comparable forward point | 0.000528690 | 10.77% |
| new cross-sectional nonlinear diagnostic | 0.001468849 | 29.92% — invalid temporally |
| new strict-forward purged latest point | 0.000004894 | 0.10% — `P=0.5365` |
| new strict-forward purged robust headroom | 0 | 0% |

The revised optimistic number remains EXP080’s `0.002412`: the new full-span nonlinear optimistic
maximum is only `0.001564`. The current-only interaction diagnostic reaches `0.003529`, but loses
the gate after projection through 16 deployable and 40 full components and therefore is not
counted as new information.

A full four-fold target-availability-purged bound is not identifiable from the canonical schedule:
14-day spacing is shorter than the 30-day target. The single available purged latest point rejects
every tested mechanism, so it cannot justify raising the attainable headroom.

## I. Verdict

```text
NO_EVIDENCE_CONFIRMED
```

EXP080 is **not** validated as a theorem that `0.000529` upper-bounds every possible model. It is
validated as a practical decision: no allowed, full-span, strict-temporal mechanism approaches the
required `0.004909`. The largest apparent exception is caused by same-period cross-sectional
training and fails out of time. No candidate satisfies simultaneously:

```text
latest clean post-projection rho >= 0.020
    OR strict-forward Delta MSE <= -0.0010
P(Delta MSE < 0) >= 0.95
positive sign on >= 3/4 folds
target-availability-purged temporal training
full production-like span projection
```

## J. Next action

**NONE_AUTHORIZED_WITH_CURRENT_INFORMATION.** Не запускать ещё один generic LightGBM/CatBoost,
encoder, blend или TEST correction.

### Information inventory

#### AVAILABLE_AND_EXHAUSTED

- 40 clean canonical OOF predictions and their family disagreements;
- 16 exact OOF/TEST prediction counterparts;
- 108 cutoff-safe RFM/recency/frequency/channel features;
- activity, count and conditional-value predictions;
- fixed nonlinear/rank interactions;
- target-free K=4 states and 128 behavioral prototypes;
- cohort ranks, density and unsupervised cluster share;
- observable large-error membership for 5/10/20% tails;
- daily sequence/history already exercised by EXP075 and the production bank;
- calendar-derived model family already represented by `HOLIDAY-YOY-FAST`.

#### POTENTIALLY_AVAILABLE_BUT_UNUSED

- 24 clean OOF bank components lack exact TEST counterparts, so they cannot become a deployable
  relational/disagreement channel without their frozen TEST inference artifacts;
- a truly purged sequence of later validation cohorts could be collected, but the current raw
  cohort after `2025-11-16` is survivorship-selected and cannot supply it cleanly.

Neither item is current evidence of target signal.

#### NOT_PRESENT_IN_DATA

- product/item/category/seller relations;
- order/session identifiers and basket composition;
- price, discount, promotion, inventory and exposure metadata;
- richer event semantics beyond daily aggregate funnel counts;
- external campaign/calendar covariates beyond the date itself;
- an independent user cohort not selected by future three-block activity.

#### UNIDENTIFIABLE_WITH_CURRENT_HISTORY

- whether the same-cutoff nonlinear residual map would recur in an independent future regime;
- signed error inside the observable high-variance tail;
- future extensive activity shocks, count shocks and conditional monetary shocks after the cutoff;
- a four-fold fully purged estimate using the existing 14-day canonical schedule.

### Concrete missing channel

The minimum information needed to make the present oracle gap empirically identifiable is an
**independent, non-survivorship-selected cohort with a frozen cutoff and at least 30 subsequent days
of labels**, repeated with spacing of at least 30 days. This would permit honest temporal residual
learning and decide whether the cross-sectional nonlinear signal recurs.

To make the oracle signal itself more observable before cutoff, the schema would need a genuinely
new causal/relational channel currently absent: **item/category/order/price/promotion/exposure
metadata**. Those fields could connect similar users and reveal upcoming activity/count/value
shocks. More capacity over the same daily aggregates is not supported by this audit.

## Reproducibility

- `run_falsification.py` — main reproduction, interaction/disagreement/mixture/transductive and
  nonlinear diagnostics;
- `run_purged_tail.py` — fully target-availability-purged latest-fold nonlinear, tail and fixed-basis
  transfer tests;
- `candidate_summary.csv` and `candidate_details.json` — all fold metrics and bootstraps;
- `purged_fixed_basis_metrics.csv`, `purged_tail_metrics.csv`, `temporal_audit.json` — decisive
  temporal checks;
- `interaction_feature_metrics.csv`, `disagreement_feature_metrics.csv`,
  `mixture_state_metrics.csv`, `test_disagreement_shift.csv` — mechanism audits;
- `final_candidate_verdicts.csv`, `revised_headroom.json`, `artifact_manifest.csv`,
  `checksums.sha256` — final decision and inventory.
