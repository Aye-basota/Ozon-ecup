# Teammate candidate — occ_occ_r18_wide_fixed__table_core

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__occ_occ_r18_wide_fixed_table_core`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `occ_occ_r18_wide_fixed__table_core`
- **Original source:** `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Unknown / not recoverable from repository history
- **Features:** occurrence features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`
- **Known score:** | wcv | 1.7500974476336741 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 2 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — occ_occ_r18_wide_fixed__table_core

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`

| Field | Value |
|---|---|
| name | occ_occ_r18_wide_fixed__table_core |
| family | occurrence_overlay |
| wcv | 1.7500974476336741 |
| base_wcv | 1.749803702558191 |
| delta | 0.00029374507548300953 |
| wins | 0 |
| wins_recent | 0 |
| latest_delta | 0.0003606375819609209 |
| worst_delta | 0.0003606375819609209 |
| raw_delta | -0.00021426363996616496 |
| offset_mean | -0.05355287125960746 |
| offset_std | 0.028926996021396423 |
| fold_scores | [1.7693819706596416, 1.7628685692260697, 1.751263694149654, 1.7439109785993392] |
| fold_deltas | [0.0001382020676321538, 0.000143548930742865, 0.00027394388685997306, 0.0003606375819609209] |
| notes | raw=occ_r18_wide;base=table_core;adaptive=False |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | occ_occ_r18_wide_fixed__table_core |
| family | occurrence_overlay |
| wcv | 1.7500974476336741 |
| base_wcv | 1.749803702558191 |
| delta | 0.00029374507548300953 |
| wins | 0 |
| wins_recent | 0 |
| latest_delta | 0.0003606375819609209 |
| worst_delta | 0.0003606375819609209 |
| raw_delta | -0.00021426363996616496 |
| offset_mean | -0.05355287125960746 |
| offset_std | 0.028926996021396423 |
| fold_scores | [1.7693819706596416, 1.7628685692260697, 1.751263694149654, 1.7439109785993392] |
| fold_deltas | [0.0001382020676321538, 0.000143548930742865, 0.00027394388685997306, 0.0003606375819609209] |
| notes | raw=occ_r18_wide;base=table_core;adaptive=False |
