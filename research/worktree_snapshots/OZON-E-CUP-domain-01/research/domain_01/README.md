# DOMAIN-01 — Test-Like Validation / Domain Shift

## Verdict

**STOP.** Real test and historical CV states are almost perfectly separable on the
227 production features, but the separation is overwhelmingly a technical
history-depth/support artifact.  After removing that axis, a modest residual
behavioral/calendar shift remains.  It does not change the ranking of the strong
models and conservative behavioral importance weighting does not improve either
ordinary or test-like weighted CV.

No leaderboard submission was created.

## Leakage and evaluation contract

- Positive class: the real test panel at `2026-02-13`; negative class: the four
  production validation panels `2025-09-04/09-18/10-02/10-16`.
- Primary input is exactly `make_xy(T, L=None, norm_long=True, panel_blocks=3)`:
  227 columns built by `build_features(cutoff_date)` from rows `event_date <= T`.
- `user_id`, cutoff date, fold id, source id and calendar columns are metadata and
  never enter a classifier.
- Five deterministic user-hash folds: every state of one user is held out together.
  There are 1,020,616 observations and 250,000 test observations (prior 0.24495).
- Historical `p_test_like` is strictly OOF for the domain task.  The production
  adaptation uses a separate full behavioral domain model only to score earlier
  train cutoff states, which were not rows of the domain training dataset.
- D0 is regularized linear logistic SGD; D1 is compact LightGBM (31 leaves, 120
  rounds; 80 rounds for ablations).  Seed is `config.SEED`.

Leakage audit: `artifacts/domain_01/leakage_audit.json`.  Group-fold metrics and
calibration tables are in `results/domain_classifier_folds.csv` and
`results/calibration_*.csv`.

## Domain classification

| representation | OOF ROC-AUC | PR-AUC | Brier | ECE-10 | interpretation |
|---|---:|---:|---:|---:|---|
| D0 linear, 227 production | 0.935485 | 0.846949 | 0.082862 | 0.035413 | shift is visible linearly |
| **D1, 227 production** | **0.998567** | **0.996046** | 0.017225 | 0.041512 | primary `p_test_like`; near separation |
| raw depth-only, 20 columns | 0.986102 | 0.974947 | 0.025252 | 0.009570 | depth alone explains almost all separation |
| fixed `L=180`, 195 columns | 0.647465 | 0.371352 | 0.176106 | 0.019748 | residual behavior/calendar shift |
| behavioral-only, 150 columns | 0.639283 | 0.360089 | 0.177017 | 0.015945 | residual behavior/calendar shift |
| missingness-only | 0.559672 | 0.298618 | 0.182777 | 0.003386 | missingness is not the main artifact |

D1-production is stable across user folds: ROC-AUC
`0.998627/0.998641/0.998642/0.998280/0.998641`.  Behavioral D1 is also stable at
`0.6376–0.6422`, so the residual is small but real rather than one user split.

The primary classifier is intentionally not used for production adaptation.  Its
near-disjoint support makes odds weights fragile: even with temperature 0.5 and a
strict final clip `[0.25, 4]`, effective sample size is only 57.6%.  Behavioral
temperature 0.5 keeps 96.8% effective sample size and is the sole adaptation recipe.

## What explains the shift

Top D1 gain/shift statistics (historical versus test):

| feature | gain share | SMD | KS | PSI | historical median | test median |
|---|---:|---:|---:|---:|---:|---:|
| `tenure_frac` | 0.5009 | -0.212 | 0.174 | 1.366 | 0.9847 | 0.9853 |
| `w365_days_cat` | 0.0899 | -0.086 | 0.100 | 0.065 | 9.79 | 8.00 |
| `gap_max_frac` | 0.0854 | -0.045 | 0.105 | 0.078 | 0.0729 | 0.0636 |
| `trend_gmv_90_365` | 0.0599 | +0.024 | 0.098 | 0.041 | 1.003 | 0.852 |
| `gap_cv` | 0.0564 | +0.277 | 0.117 | 0.080 | 1.014 | 1.094 |
| `gap_std` | 0.0424 | +0.268 | 0.093 | 0.061 | 3.575 | 4.336 |
| `w365_days_buy` | 0.0296 | -0.050 | 0.087 | 0.045 | 11.19 | 9.00 |
| `w365_days_presence_only` | 0.0225 | +0.023 | 0.050 | 0.015 | 11.19 | 11.00 |
| `rec_cat` | 0.0164 | +0.155 | 0.049 | 0.012 | 21 | 25 |
| `first_buy_frac` | 0.0119 | -0.053 | 0.040 | 0.022 | 0.883 | 0.885 |

Permutation is more decisive than gain: permuting `tenure_frac` alone drops AUC
by **0.2504**; the next largest drops are `gap_max_frac` 0.0342,
`gap_std` 0.0127 and `gap_cv` 0.0126.  `tenure_frac` has a cutoff-specific discrete
grid because it divides integer tenure by the available dataset depth.  It is a
technical source fingerprint even though its marginal SMD is only 0.21.

Numerical decomposition: raw depth-only AUC **0.9861**, while fixed-L180 AUC is
only **0.6475** and behavioral-only is **0.6393**.  Therefore the shift is mainly
technical history depth/support, plus a modest inseparable behavioral/calendar
remainder.  Missing patterns contribute only AUC 0.5597.  Explicit calendar date
was correctly forbidden, so this experiment does not pretend to split the
remaining 0.64 perfectly between seasonality and population behavior.

Full statistics for all features are in `results/feature_shift_report.csv`; gain
and permutation reports are separate CSVs.

## Which historical cutoffs look like test

| cutoff | n | mean primary `p_test_like` | mean behavioral p | production RMSLE cal | zero rate | mean active days, 30d |
|---|---:|---:|---:|---:|---:|---:|
| 2025-09-04 | 188,518 | 0.03491 | 0.23559 | 1.76749 | 0.3998 | 11.218 |
| 2025-09-18 | 191,025 | 0.03569 | 0.23582 | 1.76103 | 0.3896 | 11.347 |
| 2025-10-02 | 193,694 | 0.03870 | 0.23689 | 1.74946 | 0.3876 | 11.487 |
| **2025-10-16** | **197,379** | **0.04854** | **0.23847** | **1.74222** | **0.3871** | **11.720** |

The latest fold is the most test-like on both classifiers, agreeing with the
existing 1:2:4:8 validation direction.  Primary test-likeness and production
squared error have only Spearman **0.0133**.  Correlations between test-likeness
and model loss deltas are all `|rho| <= 0.0157`; decile diagnostics show no hidden
segment in which a competitor changes the conclusion.

## Importance-weighted CV

Weights use the covariate-shift ratio
`p(test|x)/(1-p(test|x)) * prior(hist)/prior(test)`, temperature, strict clipping,
and mean-one normalization within each validation cutoff.  Five pre-specified
schemes were checked.  `n_eff/n` ranges from 0.242 (primary T=1, clip 10) to 0.968
(behavioral T=0.5, clip 4).

The strong ranking does not change:

1. `SEQ-01-MIX`
2. `S1-DIST-MIX`
3. `S1-SEEDAVG3`
4. `S1-DIST`
5. `S1-ROUNDS`

The explicit `ROUNDS-slot-control` differs from `SEQ-01-MIX` by only 0.000004
ordinary wCV and swaps with it under some technical schemes by 1–2 millionths;
this is far below the project floor and not a ranking change.  E10/E02 can swap
inside the weaker standalone tail, with no effect on candidate selection.

Absolute weighted RMSLE changes because the weighted population is easier or
harder; only within-scheme model differences are interpreted.  Complete tables:
`results/weighted_cv_comparison.csv`, `weighted_cv_folds.csv`, and
`importance_weight_diagnostics.csv`.

## Minimal production adaptation

One isolated change was run: same `S1-ROUNDS` direct recipe, 227 features, 300
rounds and identical train cutoffs, but with behavioral D1 weights at temperature
0.5 and clip `[0.25,4]`.  Effective train sample size is 0.964 on every fold.

| model | ordinary wCV | behavioral-weighted wCV |
|---|---:|---:|
| same-recipe `S1-ROUNDS` | **1.751076** | **1.736725** |
| `DOMAIN-01-DIRECT` | 1.751164 | 1.736810 |
| delta | **+0.000087** | **+0.000086** |

Ordinary fold deltas are `+0.000010/-0.000088/-0.000285/+0.000327`: only 2/4
improve and the mandatory latest fold worsens.  The result misses every project
gate and has the same sign under the test-like criterion.

In the production slot, the full 0.15 replacement changes ordinary/weighted wCV
by `+0.000007/+0.000005` versus the same-recipe ROUNDS control.  LOFO selects
weights `0.05/0.05/0.025/0.125` and gives **+0.000010 wCV**, including a worse
latest fold.  Diversity is not incremental:

- `Var(pred_domain - pred_base) = 0.006578 = 0.924x` seed floor;
- residual correlation with same-recipe base = 0.998936;
- residual correlation with production mixture = 0.998287.

No candidate submission was prepared: standalone, weighted, diversity and LOFO
gates all fail.

## Artifacts and reproduction

Tracked compact results are under `research/domain_01/results/`.  Regenerable,
gitignored row/model artifacts are under `artifacts/domain_01/`:

- `domain_oof_probabilities.parquet` — all strict group-OOF domain probabilities;
- `historical_test_likeness.parquet` — historical OOF `p_test_like` and weights;
- `test_domain_probabilities.parquet`;
- `domain_01_metrics.json` and domain model/checkpoint files;
- `oof_DOMAIN-01-DIRECT.npz` and four production adaptation models.

Commands:

```powershell
python src/domain01.py diagnose --baseline-artifacts artifacts/source_main
python src/domain01.py adapt --resume --baseline-artifacts artifacts/source_main
python -m pytest src/test_domain01.py -q
```

## Exactly one next experiment

**CALENDAR-PLACEBO-01:** on fixed-L180 production states, fit the same grouped
domain classifier across historical cutoff pairs at several time gaps and compare
its signed feature drift with the fixed-L180 real-test discriminator.  This tests
whether the residual AUC 0.647 is ordinary calendar drift or a test-specific
population change, without repeating importance-weighted training that already
failed here.
