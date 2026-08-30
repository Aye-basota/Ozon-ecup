# CALENDAR-PLACEBO-01 — fixed-L180 temporal/calendar drift

## Verdict

**STOP-CALENDAR.** Domain separability grows smoothly with chronological gap:
mean ROC-AUC is 0.5222 at 7 days, 0.5529 at 28 days, 0.5819 at 56 days and
0.6234 at 105 days. A linear placebo gap curve predicts AUC 0.6455 at the real
120-day gap; observed historical `2025-10-16 -> test 2026-02-13` AUC is 0.6443
(residual -0.0011). Thus AUC magnitude is ordinary temporal drift, not a
test-specific anomaly.

The signed real-test direction is unusual: cosine/rank similarity is
**-0.603/-0.397** against the 105-day historical placebo and **-0.501/-0.314**
against the mean of all nine placebo directions. This is consistent with a
winter reversal of summer/autumn activity growth, but is insufficient for a
Calendar specialist: the exact `2025-02-13` YoY state has only 44 calendar days
of source data and a 1-block panel, so it is not a legal fixed-L180/3-block
comparator. Among eligible historical states, calendar-nearest and
chronologically-nearest are the same cutoff (`2025-10-16`); annual alignment
cannot be separated from recency without inventing support.

The real-direction score has only weak error association and no actionable model
switch: Spearman with squared production error is `0.029/0.050/0.057/0.050`
over the four folds, RMSLE is not monotone over score quintiles, and
`SEQ-01-MIX` remains best in the high-score quintile and in every other quintile.

## Protocol and leakage contract

- Features are built only by `make_xy(T, L=180, n_blocks=3)`, hence by
  `build_features(T)` with source rows `event_date <= T`.
- All task endpoints have a full 180-day source window. The exact YoY endpoint is
  reported as ineligible rather than silently using 44 days.
- Every task is balanced to 120,000 rows per class and uses the same compact
  LightGBM recipe: 31 leaves, 80 rounds, 5 deterministic user-hash OOF folds,
  seed `config.SEED`.
- Every state of one user is assigned to one fold. User ID, cutoff/date, fold,
  source and domain markers never enter the feature matrix.
- GBDT gain is importance only. Direction comes from B-A standardized mean and
  median shifts plus standardized logistic coefficients.

The automated tests cover cutoff legality/future leakage, equal L180 and panel
support, grouped user splitting, identifier rejection and B-A signed-shift math.

## Historical placebo pairs and classifier results

| task | A -> B | gap | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|
| p07_early | 2025-07-03 -> 2025-07-10 | 7d | 0.51538 | 0.51425 |
| p07_mid | 2025-09-18 -> 2025-09-25 | 7d | 0.52761 | 0.52341 |
| p07_late | 2025-10-09 -> 2025-10-16 | 7d | 0.52358 | 0.52025 |
| p28_early | 2025-07-03 -> 2025-07-31 | 28d | 0.54818 | 0.54358 |
| p28_mid | 2025-08-21 -> 2025-09-18 | 28d | 0.55226 | 0.54594 |
| p28_late | 2025-09-18 -> 2025-10-16 | 28d | 0.55839 | 0.55246 |
| p56_early | 2025-07-03 -> 2025-08-28 | 56d | 0.58331 | 0.57619 |
| p56_late | 2025-08-21 -> 2025-10-16 | 56d | 0.58042 | 0.57277 |
| p105_max | 2025-07-03 -> 2025-10-16 | 105d | 0.62339 | 0.61615 |
| **real comparator** | **2025-10-16 -> 2026-02-13** | **120d** | **0.64435** | **0.63177** |

Same-gap repetitions matter: even the three 7-day AUCs range 0.5154–0.5276 and
the three 28-day AUCs range 0.5482–0.5584. A single month transition would be a
weak seasonality claim; the repeated placebo curve is the appropriate baseline.

## Direction of drift

The largest absolute real-test B-A SMDs are negative:
`trend_pres_30_90 -0.187`, `trend_srch_30_90 -0.181`, `w90_cart2ord -0.175`,
`w60_cart2ord -0.169`, `w30_lgmv -0.163`, `w30_cart2ord -0.159`, and
`trend_gmv_30_90 -0.157`. Recent activity/value is lower at the February test
cutoff than at the October historical cutoff after equalizing support.

| historical direction | cosine with real | signed-rank rho | top-20 overlap |
|---|---:|---:|---:|
| 105d Jul -> Oct | **-0.603** | **-0.397** | 3/20, 0/3 same sign |
| 56d Jul -> Aug | -0.760 | -0.629 | 8/20, 0/8 same sign |
| mean of all 9 placebos | **-0.501** | **-0.314** | — |

So the *magnitude* of separability follows ordinary time distance while the
*direction* reverses. That is evidence for a winter/calendar fingerprint in X,
not evidence that predicting Y should change.

## Calendar/YoY eligibility

`2025-02-13` is the exact day-of-year analog of test, but the dataset begins on
`2025-01-01`: only 44 inclusive calendar days are available and a 3-block panel
cannot exist. Using it would reintroduce exactly the technical support/source
fingerprint that DOMAIN-01 removed. The strict comparison therefore records it
as ineligible. All legal fixed-L180 cutoffs lie in July–October 2025; among them
`2025-10-16` is both calendar-nearest (120 days on the circular day-of-year
axis) and chronologically nearest. There is no independent YoY-vs-recency
contrast in the observed data.

This does not contradict `HOLIDAY-YOY`: that experiment found a weak personal
holiday-response signal but ordinary CV was neutral. It also cannot supply the
missing fixed-support population state required by this experiment.

## Relation to production error

The score is the fixed-L180 OOF state projected on the signed
`2025-10-16 -> test` SMD vector, standardized on the latest historical panel.

| fold | rho(score, squared error) | rho(score, signed residual) |
|---|---:|---:|
| 2025-09-04 | +0.0291 | -0.0745 |
| 2025-09-18 | +0.0498 | -0.0781 |
| 2025-10-02 | +0.0567 | -0.0885 |
| 2025-10-16 | +0.0501 | -0.0784 |

The association is stable but small and non-monotone in RMSLE: production
quintile RMSLE is `1.539/1.832/1.927/1.837/1.573`, so the score mostly orders
ordinary activity/difficulty regimes rather than a growing test-like failure.
In quintile 5, `SEQ-01-MIX` scores 1.57255 versus 1.57377 DIST-MIX, 1.57478
SEEDAVG3 and 1.57526 ROUNDS; no existing component wins in any of four folds.

## Artifacts and reproduction

Compact tracked outputs are in `research/calendar_placebo_01/results/`:

- `placebo_pairs.csv`, `domain_task_metrics.csv`, `domain_task_folds.csv`;
- `signed_shift_vectors.csv`, `drift_vector_similarity.csv`,
  `top_shifted_features.csv`, `feature_importance.csv`;
- `calendar_alignment.csv` and its JSON summary;
- `calendar_score_direction.csv`, per-fold/quantile error tables and summary;
- `leakage_support_audit.json` with the fixed-support/source contract;
- `calendar_placebo_01_summary.json`.

Regenerable model/OOF checkpoints remain gitignored under
`artifacts/calendar_placebo_01/`.

```powershell
python src/calendar_placebo.py run `
  --baseline-artifacts C:\path\to\production\artifacts --resume
python -m pytest src/test_calendar_placebo.py -q
```

## Exactly one next experiment

**SEQ-DEPTH-AUG-01:** train the existing sequence encoder with random input-depth
cropping so that the test-support regime is represented during training. This is
the live non-calendar branch already motivated by `exp_027`; no Calendar/YoY
specialist should be trained from the present diagnostics.
