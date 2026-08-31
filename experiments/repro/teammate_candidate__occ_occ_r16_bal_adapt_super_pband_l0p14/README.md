# Teammate candidate — occ_occ_r16_bal_adapt__super_pband_l0p14

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__occ_occ_r16_bal_adapt_super_pband_l0p14`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `occ_occ_r16_bal_adapt__super_pband_l0p14`
- **Original source:** `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Unknown / not recoverable from repository history
- **Features:** occurrence features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`
- **Known score:** | wcv | 1.7482980725359212 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 2 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — occ_occ_r16_bal_adapt__super_pband_l0p14

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`

| Field | Value |
|---|---|
| name | occ_occ_r16_bal_adapt__super_pband_l0p14 |
| family | occurrence_overlay |
| wcv | 1.7482980725359212 |
| base_wcv | 1.749803702558191 |
| delta | -0.001505630022269629 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0017927825479342463 |
| worst_delta | 6.966106818939721e-05 |
| raw_delta | -0.0018018047243945734 |
| offset_mean | 0.005693316538695784 |
| offset_std | 0.07321697844262824 |
| fold_scores | [1.7693134296601989, 1.7614758325122002, 1.7495363813996674, 1.741757558469444] |
| fold_deltas | [6.966106818939721e-05, -0.0012491877831266596, -0.0014533688631266362, -0.0017927825479342463] |
| notes | raw=occ_r16_bal;base=super_pband_l0p14;adaptive=True |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | occ_occ_r16_bal_adapt__super_pband_l0p14 |
| family | occurrence_overlay |
| wcv | 1.7482980725359212 |
| base_wcv | 1.749803702558191 |
| delta | -0.001505630022269629 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0017927825479342463 |
| worst_delta | 6.966106818939721e-05 |
| raw_delta | -0.0018018047243945734 |
| offset_mean | 0.005693316538695784 |
| offset_std | 0.07321697844262824 |
| fold_scores | [1.7693134296601989, 1.7614758325122002, 1.7495363813996674, 1.741757558469444] |
| fold_deltas | [6.966106818939721e-05, -0.0012491877831266596, -0.0014533688631266362, -0.0017927825479342463] |
| notes | raw=occ_r16_bal;base=super_pband_l0p14;adaptive=True |
