# EXP077 — Forward Production Stack

## Catalogue metadata

- **Catalogue ID:** `new_direction__exp077_forward_stack`
- **Namespace:** `new_direction`
- **Experiment ID:** `EXP077_FORWARD_STACK`
- **Original source:** `research/new_directions/EXP077_FORWARD_STACK`
- **Source ref:** `origin/team-a late research package`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** late research direction / experiment package
- **Model:** sequence model, BTYD, Ridge, ensemble, blend
- **Features:** holiday/YoY features, recency, freshness/conditional features, gap/burst features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** | S1-E02 | TABULAR | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\oof_S1-E02.npz` | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\ztest_S1-UNC.npy` | 4/4: 2025-09-04,2025-09-18,2025-10-02,2025-10-16 | `C:\Users\Admin\Desktop\e-cup-research-clean\research\new_directions\EXP076_STRONG_BASELINE_VALIDATION_CHANNEL\code\s3_build_matrix.py` |
- **Known score:** | 1_fold4_Delta_RMSLE_le_minus_0.0005 | PASS |
- **Seed:** | S1-SEEDAVG5 | TABULAR | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\oof_S1-SEEDAVG5.npz` | `— (excluded from deployable fit)` | 4/4: 2025-09-04,2025-09-18,2025-10-02,2025-10-16 | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\report_S1-SEEDAVG5.json` |
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when the data/frozen artifacts named by the report are present
- **Notes:** Directory-level audit unit: 11 files, 1 launcher/helper scripts, 1 preserved report documents. Numeric claims are copied from those reports.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# EXP077 — Forward Production Stack

## Verdict

**NO_GO**. The fixed G.1 gates were applied without leaderboard fitting and without training any model.

| Gate | Result |
| --- | --- |
| 1_fold4_Delta_RMSLE_le_minus_0.0005 | PASS |
| 2_at_least_2_of_3_heldout_better_composition_proxy | PASS |
| 3_latest_SBVC_rho_positive | FAIL |
| 4_weighted_SBVC_rho_ge_0.010 | FAIL |
| 5_nested_SBVC_Delta_MSE_negative | FAIL |
| 6_bootstrap_probability_ge_0.95 | FAIL |
| 7_improvement_not_one_fold | FAIL |
| 8_no_material_ORTH_ALPHA_residual_conflict | FAIL |
| 9_TEST_vector_format_projection_checks | PASS |

## Artifact audit

The EXP076 40-column clean OOF bank is reproduced exactly. The deployable stack is the strict
intersection of that bank with exact frozen TEST predictions: **16/40** components.
OOF-only components remain in the reference reproduction but are not silently replaced for TEST.
`oof_BLOCK4_SAF['activity']` and every target-derived activity field were excluded; only
`z_new_honest` was read from that artifact. No contaminated teammate OOF was loaded.

| component | family | OOF path | TEST path | fold coverage | provenance |
| --- | --- | --- | --- | --- | --- |
| S1-E02 | TABULAR | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\oof_S1-E02.npz` | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\ztest_S1-UNC.npy` | 4/4: 2025-09-04,2025-09-18,2025-10-02,2025-10-16 | `C:\Users\Admin\Desktop\e-cup-research-clean\research\new_directions\EXP076_STRONG_BASELINE_VALIDATION_CHANNEL\code\s3_build_matrix.py` |
| S1-E03a | TABULAR | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\oof_S1-E03a.npz` | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\ztest_S1-CAP.npy` | 4/4: 2025-09-04,2025-09-18,2025-10-02,2025-10-16 | `C:\Users\Admin\Desktop\e-cup-research-clean\research\new_directions\EXP076_STRONG_BASELINE_VALIDATION_CHANNEL\code\s3_build_matrix.py` |
| S1-DIST | TABULAR | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\oof_S1-DIST.npz` | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\ztest_S1-DIST.npy` | 4/4: 2025-09-04,2025-09-18,2025-10-02,2025-10-16 | `C:\Users\Admin\Desktop\e-cup-research-clean\research\new_directions\EXP076_STRONG_BASELINE_VALIDATION_CHANNEL\code\s3_build_matrix.py` |
| S1-E10 | TABULAR | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\oof_S1-E10.npz` | `— (excluded from deployable fit)` | 4/4: 2025-09-04,2025-09-18,2025-10-02,2025-10-16 | `C:\Users\Admin\Desktop\e-cup-research-clean\research\new_directions\EXP076_STRONG_BASELINE_VALIDATION_CHANNEL\code\s3_build_matrix.py` |
| S1-E11 | TABULAR | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\oof_S1-E11.npz` | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\ztest_S1-E11.npy` | 4/4: 2025-09-04,2025-09-18,2025-10-02,2025-10-16 | `C:\Users\Admin\Desktop\e-cup-research-clean\research\new_directions\EXP076_STRONG_BASELINE_VALIDATION_CHANNEL\code\s3_build_matrix.py` |
| S1-SEEDAVG5 | TABULAR | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\oof_S1-SEEDAVG5.npz` | `— (excluded from deployable fit)` | 4/4: 2025-09-04,2025-09-18,2025-10-02,2025-10-16 | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\report_S1-SEEDAVG5.json` |
| S1-B0 | TABULAR | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\oof_S1-B0.npz` | `— (excluded from deployable fit)` | 4/4: 2025-09-04,2025-09-18,2025-10-02,2025-10-16 | `C:\Users\Admin\Desktop\e-cup-research-clean\research\new_directions\EXP076_STRONG_BASELINE_VALIDATION_CHANNEL\code\s3_build_matrix.py` |
| S1-E01 | TABULAR | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\oof_S1-E01.npz` | `— (excluded from deployable fit)` | 4/4: 2025-09-04,2025-09-18,2025-10-02,2025-10-16 | `C:\Users\Admin\Desktop\e-cup-research-clean\research\new_directions\EXP076_STRONG_BASELINE_VALIDATION_CHANNEL\code\s3_build_matrix.py` |
| S1-E03b | TABULAR | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\oof_S1-E03b.npz` | `— (excluded from deployable fit)` | 4/4: 2025-09-04,2025-09-18,2025-10-02,2025-10-16 | `C:\Users\Admin\Desktop\e-cup-research-clean\research\new_directions\EXP076_STRONG_BASELINE_VALIDATION_CHANNEL\code\s3_build_matrix.py` |
| SEQ-AVG3 | SEQ | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\oof_SEQ-AVG3.npz` | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\ztest_SEQ-01.npy;C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\ztest_SEQ-C289-S43.npy;C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\ztest_SEQ-C289-S44.npy` | 4/4: 2025-09-04,2025-09-18,2025-10-02,2025-10-16 | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\report_SEQ-AVG3.json` |
| SEQ-D3A-AVG3 | SEQ | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\oof_SEQ-D3A-AVG3.npz` | `— (excluded from deployable fit)` | 4/4: 2025-09-04,2025-09-18,2025-10-02,2025-10-16 | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\report_SEQ-D3A-AVG3.json` |
| SEQ-D3A-BASE-AVG3 | SEQ | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\oof_SEQ-D3A-BASE-AVG3.npz` | `— (excluded from deployable fit)` | 4/4: 2025-09-04,2025-09-18,2025-10-02,2025-10-16 | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\report_SEQ-D3A-BASE-AVG3.json` |
| ETX-AVG3 | ETX | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\oof_ETX-AVG3.npz` | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\ztest_ETX-01-S42-DCW.npy;C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\ztest_ETX-01-S43-DCW.npy;C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\ztest_ETX-01-S44-DCW.npy` | 4/4: 2025-09-04,2025-09-18,2025-10-02,2025-10-16 | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\report_ETX-AVG3.json` |
| ETX-AVG2 | ETX | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\oof_ETX-AVG2.npz` | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\ztest_ETX-01-S42-DCW.npy;C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\ztest_ETX-01-S43-DCW.npy` | 4/4: 2025-09-04,2025-09-18,2025-10-02,2025-10-16 | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\report_ETX-AVG2.json` |
| ETX-01-S42 | ETX | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\oof_ETX-01-S42.npz` | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\ztest_ETX-01-S42-DCW.npy` | 4/4: 2025-09-04,2025-09-18,2025-10-02,2025-10-16 | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\report_ETX-01-S42.json` |
| PT-FULL-AVG3 | BTYD_OTHER | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\oof_PT-FULL-AVG3.npz` | `— (excluded from deployable fit)` | 4/4: 2025-09-04,2025-09-18,2025-10-02,2025-10-16 | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\report_PT-FULL-AVG3.json` |
| PT-OD-AVG3 | BTYD_OTHER | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\oof_PT-OD-AVG3.npz` | `— (excluded from deployable fit)` | 4/4: 2025-09-04,2025-09-18,2025-10-02,2025-10-16 | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\report_PT-OD-AVG3.json` |
| PT-SHUF-AVG3 | BTYD_OTHER | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\oof_PT-SHUF-AVG3.npz` | `— (excluded from deployable fit)` | 4/4: 2025-09-04,2025-09-18,2025-10-02,2025-10-16 | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\report_PT-SHUF-AVG3.json` |
| RIDGE15 | TABULAR | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\oof_RIDGE15.npz` | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\ztest_RIDGE15.npy` | 4/4: 2025-09-04,2025-09-18,2025-10-02,2025-10-16 | `C:\Users\Admin\Desktop\OZON-E-CUP\research\strategies\results\RIDGE15\summary.json` |
| HOLIDAY-YOY-FAST | TABULAR | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\oof_HOLIDAY-YOY-FAST.npz` | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\ztest_HOLIDAY-YOY-FAST.npy` | 4/4: 2025-09-04,2025-09-18,2025-10-02,2025-10-16 | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\report_HOLIDAY-YOY-FAST.json` |
| MHZ-FULL | TABULAR | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\oof_MHZ-FULL.npz` | `— (excluded from deployable fit)` | 4/4: 2025-09-04,2025-09-18,2025-10-02,2025-10-16 | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\report_MHZ-FULL.json` |
| MHZ-BASE | TABULAR | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\oof_MHZ-BASE.npz` | `— (excluded from deployable fit)` | 4/4: 2025-09-04,2025-09-18,2025-10-02,2025-10-16 | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\report_MHZ-BASE.json` |
| MHZ-P30 | TABULAR | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\oof_MHZ-P30.npz` | `— (excluded from deployable fit)` | 4/4: 2025-09-04,2025-09-18,2025-10-02,2025-10-16 | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\report_MHZ-P30.json` |
| MHZ-SELF | TABULAR | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\oof_MHZ-SELF.npz` | `— (excluded from deployable fit)` | 4/4: 2025-09-04,2025-09-18,2025-10-02,2025-10-16 | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\report_MHZ-SELF.json` |
| S04-A | TABULAR | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\oof_S04-A.npz` | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\ztest_S04-A.npy` | 4/4: 2025-09-04,2025-09-18,2025-10-02,2025-10-16 | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\report_S04-A.json` |
| S04-B | TABULAR | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\oof_S04-B.npz` | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\ztest_S04-B.npy` | 4/4: 2025-09-04,2025-09-18,2025-10-02,2025-10-16 | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\report_S04-B.json` |
| S04-C | TABULAR | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\oof_S04-C.npz` | `— (excluded from deployable fit)` | 4/4: 2025-09-04,2025-09-18,2025-10-02,2025-10-16 | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\report_S04-C.json` |
| GAP-E02-K5-G090-S42 | TABULAR | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\oof_GAP-E02-K5-G090-S42.npz` | `— (excluded from deployable fit)` | 4/4: 2025-09-04,2025-09-18,2025-10-02,2025-10-16 | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\report_GAP-E02-K5-G090-S42.json` |
| GAP-E10-K5-G090-S42 | TABULAR | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\oof_GAP-E10-K5-G090-S42.npz` | `— (excluded from deployable fit)` | 4/4: 2025-09-04,2025-09-18,2025-10-02,2025-10-16 | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\report_GAP-E10-K5-G090-S42.json` |
| GAP-DIST-K5-G060-S42 | TABULAR | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\oof_GAP-DIST-K5-G060-S42.npz` | `— (excluded from deployable fit)` | 4/4: 2025-09-04,2025-09-18,2025-10-02,2025-10-16 | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\report_GAP-DIST-K5-G060-S42.json` |
| SAMPLE-TB1-AVG3-R300 | TABULAR | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\oof_SAMPLE-TB1-AVG3-R300.npz` | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\ztest_TIER-A-DIRECT-AVG3-R300.npy` | 4/4: 2025-09-04,2025-09-18,2025-10-02,2025-10-16 | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\report_SAMPLE-TB1-AVG3-R300.json` |
| SAMPLE-DENSE-S3-F422-S42-R300 | TABULAR | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\oof_SAMPLE-DENSE-S3-F422-S42-R300.npz` | `— (excluded from deployable fit)` | 4/4: 2025-09-04,2025-09-18,2025-10-02,2025-10-16 | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\report_SAMPLE-DENSE-S3-F422-S42-R300.json` |
| S1-ROUNDS-R600 | TABULAR | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\oof_S1-ROUNDS-R600.npz` | `— (excluded from deployable fit)` | 4/4: 2025-09-04,2025-09-18,2025-10-02,2025-10-16 | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\report_S1-ROUNDS-R600.json` |
| S1-ROUNDS-R300 | TABULAR | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\oof_S1-ROUNDS-R300.npz` | `— (excluded from deployable fit)` | 4/4: 2025-09-04,2025-09-18,2025-10-02,2025-10-16 | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\report_S1-ROUNDS-R300.json` |
| BTYD:z_btyd | BTYD_OTHER | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\BTYD_STABLE_EXP051\oof_raw.npz` | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\BTYD_STABLE_EXP051\test_raw.npz` | 4/4: 2025-09-04,2025-09-18,2025-10-02,2025-10-16 | `C:\Users\Admin\Desktop\OZON-E-CUP\research\strategies\results\BTYD_STABLE_EXP051\artifact_manifest.json` |
| BTYD:z_strongest | BTYD_OTHER | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\BTYD_STABLE_EXP051\oof_raw.npz` | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\BTYD_STABLE_EXP051\test_raw.npz` | 4/4: 2025-09-04,2025-09-18,2025-10-02,2025-10-16 | `C:\Users\Admin\Desktop\OZON-E-CUP\research\strategies\results\BTYD_STABLE_EXP051\artifact_manifest.json` |
| BLOCK4:z_new_honest | BTYD_OTHER | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\oof_BLOCK4_SAF.npz` | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\test_BLOCK4_SAF.npz` | 4/4: 2025-09-04,2025-09-18,2025-10-02,2025-10-16 | `C:\Users\Admin\Desktop\OZON-E-CUP\research\strategies\results\BLOCK4_SAF\audit.json` |
| FRESH:z_fresh | BTYD_OTHER | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\oof_FRESH_CONTRAST_MOE.npz` | `— (excluded from deployable fit)` | 4/4: 2025-09-04,2025-09-18,2025-10-02,2025-10-16 | `C:\Users\Admin\Desktop\OZON-E-CUP\research\strategies\results\FRESH_CONTRAST\validation.json` |
| FRESH:z_vol | BTYD_OTHER | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\oof_FRESH_CONTRAST_MOE.npz` | `— (excluded from deployable fit)` | 4/4: 2025-09-04,2025-09-18,2025-10-02,2025-10-16 | `C:\Users\Admin\Desktop\OZON-E-CUP\research\strategies\results\FRESH_CONTRAST\validation.json` |
| FRESH:z_clean | BTYD_OTHER | `C:\Users\Admin\Desktop\OZON-E-CUP\artifacts\oof_FRESH_CONTRAST_MOE.npz` | `— (excluded from deployable fit)` | 4/4: 2025-09-04,2025-09-18,2025-10-02,2025-10-16 | `C:\Users\Admin\Desktop\OZON-E-CUP\research\strategies\results\FRESH_CONTRAST\validation.json` |

Full hashes, alignment mode, target parity, and exclusion reasons: `artifact_audit.csv`.
The compact canonical EXP037 parquet reproduces `BTYD:z_strongest` with max absolute error
`1.776e-15`.

## Forward results

Ridge regularization was frozen at `3e-05` from EXP076. There was no EXP077 sweep.
Fold 1 uses the frozen EXP037 recipe; each later fold uses weights fitted only on earlier folds.

| cutoff | candidate | RMSLE | Delta_vs_EXP037 | Delta_vs_composition_proxy |
| --- | --- | ---: | ---: | ---: |
| 2025-09-04 | EXP076 reference (40 OOF) | 1.770775680 | +0.000000000 | -0.001232787 |
| 2025-09-18 | EXP076 reference (40 OOF) | 1.759047473 | -0.003221905 | -0.008431388 |
| 2025-10-02 | EXP076 reference (40 OOF) | 1.748869001 | +0.000113952 | -0.002419844 |
| 2025-10-16 | EXP076 reference (40 OOF) | 1.741091071 | -0.000547036 | -0.000237833 |
| 2025-09-04 | new forward stack (16 deployable) | 1.770775680 | +0.000000000 | -0.001232787 |
| 2025-09-18 | new forward stack (16 deployable) | 1.759407695 | -0.002861683 | -0.008071166 |
| 2025-10-02 | new forward stack (16 deployable) | 1.749237887 | +0.000482838 | -0.002050958 |
| 2025-10-16 | new forward stack (16 deployable) | 1.741052261 | -0.000585847 | -0.000276644 |

| stack | weighted wCV 1:2:4:8 |
| --- | ---: |
| EXP037 frozen | 1.748229299 |
| composition-matched ORTH_ALPHA proxy | 1.749516854 |
| EXP076 reference, 40 OOF | 1.747538347 |
| new forward stack, 16 deployable | 1.747664047 |

Held-out sign consistency for the deployable stack is `3/3` versus the
composition proxy and `2/3` versus EXP037. Fold-4 Delta versus EXP037 is
`-0.000585847`. The worst held-out Delta versus EXP037 is
`+0.000482838`.

Final all-fold weight stability and family shares (signed shares use total component slope;
absolute shares use L1 mass):

| bank | L1_norm | effective_components | signed_share_SEQ | signed_share_ETX | signed_share_TABULAR | signed_share_BTYD_OTHER | absolute_share_SEQ | absolute_share_ETX | absolute_share_TABULAR | absolute_share_BTYD_OTHER |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| reference_40 | 3.704950 | 14.030803 | 0.272918 | 0.209061 | 0.207219 | 0.310802 | 0.074616 | 0.076519 | 0.408103 | 0.440762 |
| forward_deployable | 1.900854 | 8.022948 | 0.217993 | 0.219428 | 0.300207 | 0.262373 | 0.116333 | 0.169542 | 0.574108 | 0.140017 |

ORTH_ALPHA reconstructed family shares are SEQ `0.128282`, ETX
`0.196194`, TABULAR `0.687397`, and BTYD/OTHER
`-0.011873`. The deployable forward optimum moves signed SEQ share to `0.217993`
(`+0.089711`) and ETX to `0.219428` (`+0.023234`). Combined SEQ+ETX rises from
`0.324476` to `0.437421`: Claude's directional composition hypothesis is reproduced.
Signed ridge weights are not mixture probabilities; the absolute shares above expose
cancelling/unstable weights. Full per-component weights are in `weights.csv`.

## SBVC

`D_fold = z_forward - z_composition_proxy` was centered, projected outside the required
historical ensemble span (constant + strong baseline + all 40 canonical components), and the
projection was repeated. Because both terms are linear combinations of that same component
bank, the correction is algebraically in-span and is annihilated (numerical remnants below
`1e-10` are set to exact zero before inference).

| cutoff | rho_min | rho_post | nested amplitude min | nested amplitude post | rms_D_post_projection | nested_Delta_MSE | nested_Delta_RMSLE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2025-09-04 | +0.040203998 | +0.000000000 | 1.000000 | 1.000000 | 0.000e+00 | +0.000e+00 | +0.000e+00 |
| 2025-09-18 | +0.033596395 | +0.000000000 | 3.419868 | 0.000000 | 0.000e+00 | +0.000e+00 | +0.000e+00 |
| 2025-10-02 | +0.025019276 | +0.000000000 | 1.077400 | 0.000000 | 0.000e+00 | +0.000e+00 | +0.000e+00 |
| 2025-10-16 | +0.021933303 | +0.000000000 | 0.899649 | 0.000000 | 0.000e+00 | +0.000e+00 | +0.000e+00 |

Recency-weighted post-projection rho is `+0.000000000`;
latest rho is `+0.000000000`. Nested weighted Delta MSE is
`+0.000e+00` and nested weighted Delta RMSLE is
`+0.000e+00`. Cluster bootstrap 95% CI is
`[+0.000e+00, +0.000e+00]`, with
`P(Delta MSE < 0) = 0.000000`.

The min-projection rho is reported only as a diagnostic; it is not used for GO because EXP076
defines SBVC with the full historical component-span projection. Even that weaker diagnostic
does not support deployment once amplitudes are strict-forward: weighted nested Delta MSE is
`+0.002460845`, bootstrap 95% CI `[+0.001628431, +0.003338754]`, and
`P(Delta MSE < 0) = 0.000000`.

## ORTH_ALPHA residual interaction

The primary TEST reconstruction reproduces the EXP076 decomposition: `R² =
0.997317449`, unexplained RMS `0.084031307` versus
centered alpha RMS `1.622434423`.

For the actual standalone difference `D_test = z_forward_stack - z_alpha`,
`corr(D_test, r_alpha_unexplained) = -0.708759305`.
Its projection coefficient on `r_alpha_unexplained` is `-1.001170485` and the projection
RMS is `0.084129664`. This is a material conflict:
the standalone reweighting subtracts the unexplained ORTH_ALPHA residual essentially one-for-one.

## TEST geometry

The current submission span contains 78 vectors and has centered rank
67 at eigenvalue tolerance `1e-12`.

| metric | value |
| --- | ---: |
| RMS(D_test) | 0.149659913 |
| RMS(centered D_test) | 0.118699908 |
| RMS(D_perp) | 0.005331123 |
| perp_fraction (centered energy) | 0.002017143 |
| second-pass max projection | 6.430e-12 |
| corr(D_perp, current ORTH correction) | -0.000000018 |

The vector has 250,000 finite rows in unique sample order. Orthogonality is a format check only;
the historical SBVC gate remains decisive.

## Expected effect

No positive robust uplift is supported: the decision-relevant post-projection historical correction is algebraically zero, so its nested Delta RMSLE is 0. The standalone wCV change cannot be translated into an incremental production gain.

## Output

No submission was created under **NO_GO**. The audited TEST vectors are stored in
`forward_test_vectors.npz`; the reserved candidate path remains
`C:\Users\Admin\Desktop\e-cup-research-clean\submissions\SUBMIT_EXP077_FORWARD_STACK.csv` and was not written.

## Final conclusion

The advertised 40-OOF forward wCV is reproduced (`1.747538347`), but it is not a
deployable 40-component stack because exact TEST counterparts exist for only 16 components.
The clean deployable rebuild has fold-4 Delta `-0.000585847` versus EXP037. More importantly,
its required post-projection SBVC direction is identically zero and the unprojected standalone
difference cancels the unexplained ORTH_ALPHA residual. Therefore G.1 is **NO_GO** and is not
rescued with a different ridge, blend coefficient, or leaderboard fit.
