# EXP078 — Forward Global Level Forecast

## Catalogue metadata

- **Catalogue ID:** `new_direction__exp078_level_forecast`
- **Namespace:** `new_direction`
- **Experiment ID:** `EXP078_LEVEL_FORECAST`
- **Original source:** `research/new_directions/EXP078_LEVEL_FORECAST`
- **Source ref:** `origin/team-a late research package`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** late research direction / experiment package
- **Model:** Ridge
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** ## Rolling validation
- **Known score:** remaining `Delta RMSLE = -0.001508246136`
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** is **NO_GO_LEVEL**.  No residual/temporal search, EXP075 correction, user-level correction, public
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when the data/frozen artifacts named by the report are present
- **Notes:** Directory-level audit unit: 8 files, 1 launcher/helper scripts, 1 preserved report documents. Numeric claims are copied from those reports.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# EXP078 — Forward Global Level Forecast

## Verdict

**NO_GO_LEVEL**

All fixed gates were evaluated without a leaderboard probe or leaderboard-based sign/scale choice.

## Headroom math

- remaining `Delta RMSLE = -0.001508246136`
- remaining `Delta MSE = -0.004963353330` (magnitude `0.004963353330`)
- equivalent optimally removable level error `|c| = sqrt(-Delta MSE) = 0.070451070`

This is scale arithmetic, not evidence that the current submission is wrong by that amount.

## Leakage / cohort audit

The raw table is the fixed final cohort of 250,000 users.  Primary selection and all fitted
coefficients use only rows with `target_end <= 2025-11-15`.  The assertion is executable in
`run_exp078.py`; later labelled rows overlap `2025-11-16..2026-02-13`, are marked diagnostics,
and are excluded from method choice, coefficient fitting, shrinkage, and GO/NO-GO.

The canonical production proxy exists only on the historical 3-block-eligible users.  For the
production-like test, the same scalar method is therefore refit strict-forward on matched
eligible panel states; target/proxy row counts and mean target levels are asserted identical.

## Panel-level dataset

`panel_level_dataset.parquet` contains the 14-day fixed-cohort sequence, scalar target-free
state, and `target_level`.  It contains mean/median/q25/q75/q90 recent GMV for 30/60/90 days,
purchase fractions, mean purchase-day/order/search/event-day counts, recencies, and 30-vs-30 /
60-vs-60 changes.  The 11 real raw channels are `cat, searches, search_to_cart, search_to_ord, cat_to_cart, cat_to_ord, to_cart, to_ord, gmv_search, gmv_cat, gmv`.  No full-cohort,
cutoff-safe historical ORTH_ALPHA prediction exists, so no synthetic baseline panel mean was
fabricated.  `panel_level_dataset_eligible.parquet` is the matched production diagnostic.

## Rolling validation

Each validation prediction uses labels only when their 30-day target has already ended at the
validation origin.  L0 is freeze, L1 is the fixed 30-day panel step with historical beta, and L2
is Ridge (`alpha=10.0`) with at most 8 training-history-pruned predictors.

| method | n_validation | RMSE_level | MAE_level | bias | last3_RMSE | sign_accuracy | improvement_vs_L0_pct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L0 | 7 | 0.131013 | 0.120986 | -0.120986 | 0.097786 | 0.000 | +0.0 |
| L1 | 7 | 0.096740 | 0.075929 | +0.045749 | 0.090077 | 1.000 | +26.2 |
| L2 | 7 | 0.098244 | 0.073574 | +0.047446 | 0.078058 | 1.000 | +25.0 |

Primary gates: `{"L1": {"RMSE_le_0.070": false, "at_least_20pct_better_L0": true, "last3_not_worse_L0": true, "abs_bias_le_0.025": false, "no_obvious_monotonic_drift": true}, "L2": {"RMSE_le_0.070": false, "at_least_20pct_better_L0": true, "last3_not_worse_L0": true, "abs_bias_le_0.025": false, "no_obvious_monotonic_drift": true}}`.  Passing methods: `[]`.

Moving-block uncertainty (`block_length=2`, 20,000 draws): L0 RMSE 95% CI
`[0.080193, 0.169008]`; L1 `[0.044304, 0.140202]`; L2 `[0.032292, 0.147535]`.

## Production-like validation

Production-like stage was not run because neither L1 nor L2 passed the primary level gate.

## Shrinkage

- final historical forward estimate `lambda = N/A`; hard-clipped to `[0, 0.5]`
- stability: N/A
- forward-prefix lambdas: `N/A`

## Sensitivity / falsification

The preregistered 28-day spacing diagnostic has only three validation origins:
L0 `RMSE=0.149383, bias=-0.146393`; L1 `RMSE=0.108806, bias=+0.096305`;
L2 `RMSE=0.059563, bias=+0.048226`.  It is diagnostic only and cannot rescue
the failed 14-day primary gate.  Leave-one-clean-cutoff-out TEST-sign checks,
earliest/latest exclusions, and clipping checks were not run because the primary
gate explicitly forbids producing a TEST correction after `NO_GO_LEVEL`.

## TEST forecast

Primary gate failed; TEST target level and correction were not produced.

## Expected robust effect

N/A: the primary level gate stopped the experiment before production-like calibration.
The strong-result reference is `<= -0.0005`; it is descriptive, not a post-result
gate change.  Clipping diagnostics: N/A.

## Output

- No submission created.
- SHA256: `N/A`
- anchor SHA256: `9a8adb83e7b34bb6c12b7eb51584d1bf9a93825945d285258d4e1dd991f4b838`
- no file was uploaded

## Final conclusion

The best clean level RMSE is `0.096740` for `L1`, an
improvement of `26.2%` versus freeze (`0.131013`).  The preregistered verdict
is **NO_GO_LEVEL**.  No residual/temporal search, EXP075 correction, user-level correction, public
calibration, probe, or automatic submission was used.
