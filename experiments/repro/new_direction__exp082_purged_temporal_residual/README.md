# EXP082 — Fully Purged Temporal Residual Validation

## Catalogue metadata

- **Catalogue ID:** `new_direction__exp082_purged_temporal_residual`
- **Namespace:** `new_direction`
- **Experiment ID:** `EXP082_PURGED_TEMPORAL_RESIDUAL`
- **Original source:** `research/new_directions/EXP082_PURGED_TEMPORAL_RESIDUAL`
- **Source ref:** `origin/team-a late research package`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** late research direction / experiment package
- **Model:** LightGBM, sequence model
- **Features:** recency, gap/burst features, history-depth features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** # EXP082 — Fully Purged Temporal Residual Validation
- **Known score:** Nested ΔMSE: **0.00149662**; nested ΔRMSLE: **0.00042651**. Cluster-bootstrap 95% CI for ΔMSE: **[0.00107916, 0.00192116]**, `P(ΔMSE < 0)=0.0000`.
- **Seed:** Frozen composition: `0.10 CAP + 0.20 UNC + 0.25 DIST + 0.225 SEQ-S42 + 0.225 ETX-S42`. SEQ/ETX use the allowed frozen single-seed approximation; no weights or model settings were tuned.
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when the data/frozen artifacts named by the report are present
- **Notes:** Directory-level audit unit: 44 files, 4 launcher/helper scripts, 1 preserved report documents. Numeric claims are copied from those reports.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# EXP082 — Fully Purged Temporal Residual Validation

## Verdict

**BLOCKED**. Statistical verdict of the conservative core-162 diagnostic: **FINAL_NO_EVIDENCE**.

The four-fold temporal protocol itself is valid, and the production-like baseline passes its fidelity gate. However, the exact frozen EXP081 nonlinear learner is not reproducible on the requested dates: its full 40-model prediction bank and three auxiliary structural channels have saved OOF only on the old 14-day canonical folds. The new-fold diagnostic uses 162 cutoff-reproducible columns rather than the frozen 200-column feature matrix. No future prediction, imputation, distillation, leaderboard signal, or target-derived activity was used.

## Purged fold construction

| Cutoff | Target window | N | Spacing | Previous labels known | Outside survivorship interval |
|---|---|---:|---:|:---:|:---:|
| 2025-07-03 | (2025-07-03, 2025-08-02] | 181,338 | — | PASS | PASS |
| 2025-08-07 | (2025-08-07, 2025-09-06] | 184,617 | 35 | PASS | PASS |
| 2025-09-11 | (2025-09-11, 2025-10-11] | 189,815 | 35 | PASS | PASS |
| 2025-10-16 | (2025-10-16, 2025-11-15] | 197,379 | 35 | PASS | PASS |

Result: 4 folds / 3 genuine purged transitions. For every transition `target_end(previous) <= cutoff(current)`; the last target end is 2025-11-15.

## Production baseline fidelity

Frozen composition: `0.10 CAP + 0.20 UNC + 0.25 DIST + 0.225 SEQ-S42 + 0.225 ETX-S42`. SEQ/ETX use the allowed frozen single-seed approximation; no weights or model settings were tuned.

- correlation rebuilt S42 baseline vs exact AVG3 at 2025-10-16: `0.999886499` (gate >= 0.995: PASS);
- correlation vs composition-matched EXP076 proxy: `0.999447876`;
- RMS(rebuilt − exact AVG3): `0.025554`;
- candidate-rho geometry gate <= 0.003: PASS.

| Diagnostic correction | rho rebuilt | rho EXP076 matched | absolute difference | gate |
|---|---:|---:|---:|:---:|
| EXP075_joint | 0.001553 | 0.001551 | 0.000002 | PASS |
| EXP081_A | 0.020359 | 0.020640 | 0.000281 | PASS |
| EXP081_B | 0.021521 | 0.021763 | 0.000242 | PASS |
| EXP081_AB_mean | 0.021567 | 0.021833 | 0.000266 | PASS |

Every component artifact is separately audited for SHA256 parity, cutoff-safe last training target, unchanged config, and runtime below six hours in `production_component_audit.csv`.

## Residual learner reproduction

LightGBM A and B use the exact frozen depth/leaves/regularization/tree-count recipes from EXP081; A/B mean is their arithmetic mean. Preprocessing on common state/RFM and cohort columns is frozen. The feature fidelity gate nevertheless fails because bank-wide disagreement and interactions cannot be reconstructed exactly without the missing historical 41-model predictions.

- EXP081 feature count: `200`;
- cutoff-reproducible diagnostic feature count: `162`;
- exact feature/preprocessing fidelity: `FAIL`;
- derived disagreement formula match: `FAIL`.

Therefore the core-162 metrics below are conservative diagnostic evidence, not an exact reproduction capable of authorizing STRONG_GO or a TEST submission.

## Purged results

Primary diagnostic candidate: LightGBM A/B mean. Deployable amplitude for each row was fitted only from user-disjoint cross-fitted predictions on fully available earlier folds.

| Validation cutoff | Train folds | rho raw | rho vs strong residual | rho post-projection | b | G | oracle amp | deployed amp | ΔMSE | ΔRMSLE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2025-08-07 | 2025-07-03 | -0.184106 | 0.026988 | 0.016327 | 0.00185671 | 0.00397605 | 0.46697 | 0.67327 | -0.000697829 | -0.000193256 |
| 2025-09-11 | 2025-07-03, 2025-08-07 | -0.081183 | -0.012553 | -0.021303 | -0.00185649 | 0.00244088 | -0.76058 | 0.63632 | 0.00335095 | 0.000947534 |
| 2025-10-16 | 2025-07-03, 2025-08-07, 2025-09-11 | 0.057555 | 0.011753 | 0.003538 | 0.000339882 | 0.00304321 | 0.11169 | 0.72802 | 0.00111806 | 0.000320937 |

Recency-weighted post-projection rho: **-0.001733**. Latest rho: **0.003538**. Positive transitions: **2/3**.

Nested ΔMSE: **0.00149662**; nested ΔRMSLE: **0.00042651**. Cluster-bootstrap 95% CI for ΔMSE: **[0.00107916, 0.00192116]**, `P(ΔMSE < 0)=0.0000`.

Leave-one-transition-out metrics are saved in `bootstrap.json`; they are used to assess dependence on one fold.

## Same-period vs ordered vs fully-purged

| Protocol | Candidate fidelity | rho | latest rho | ΔMSE | P(gain) |
|---|---|---:|---:|---:|---:|
| A. same-period user-disjoint (EXP081) | LightGBM A / full-200 | 0.021485 | 0.020633 | -0.00133407 | 1.0000 |
| B. old ordered canonical (EXP081) | LightGBM A / full-200 | 0.013899 | 0.013092 | 0.000583986 | 0.0385 |
| C. fully purged 35-day (EXP082) | LightGBM A / core-162 | -0.002721 | 0.002743 | 0.00149906 | 0.0000 |

Because Protocol C lacks exact full-200 feature fidelity, this table is directional rather than a strictly identical-candidate causal comparison. It does not use same-period targets as temporal evidence.

## Projection / novelty

| Cutoff | RMS(u_raw) | RMS(u_perp) | perp fraction | second-pass RMS | relative error |
|---|---:|---:|---:|---:|---:|
| 2025-08-07 | 0.078557 | 0.063056 | 0.812080 | 5.720e-13 | 9.072e-12 |
| 2025-09-11 | 0.099875 | 0.049405 | 0.729070 | 1.873e-13 | 3.791e-12 |
| 2025-10-16 | 0.071351 | 0.055165 | 0.913047 | 4.667e-16 | 8.460e-15 |

Only `u_perp` is used for predictive-signal claims and deployed-gain arithmetic.

## Mathematical headroom

Required gap ΔMSE: **0.0049092736**. Weighted purged rho²: **0.00000300**.

| Headroom definition | MSE gain | Fraction of required gap |
|---|---:|---:|
| Correlation-only mathematical ceiling | 0.00000928 | 0.19% |
| Nested deployed point gain | 0.00000000 | 0.00% |
| Robust 95% gain | 0.00000000 | 0.00% |

Maximum individual purged post-projection rho across A/B/mean diagnostics: **0.017737**.

## Output

No `SUBMIT_EXP082_PURGED_RESIDUAL.csv` was created. Exact residual-feature fidelity and STRONG_GO are both required before TEST inference; leaderboard fitting or automatic submission was never used.

## Final conclusion

The requested four-fold purged clock is technically valid and the rebuilt production baseline is faithful. The exact EXP081 nonlinear mechanism cannot be adjudicated as requested because its full historical feature bank does not exist on the 35-day folds, and rebuilding that bank would require replaying dozens of separate legacy model pipelines rather than the authorized production recipe. The core-162 result quantifies the available temporal evidence but cannot upgrade to STRONG_GO.

To continue scientifically, the next useful information channel is not another learner on the same schema: use additional future labels on an independently frozen cohort, a genuinely new raw field, entity relations, or rules-permitted external data. For an exact rerun of this particular hypothesis, first materialize the frozen 40-model bank predictions at all four purged cutoffs; do not impute them from future canonical folds.

No leaderboard data was read or used.
