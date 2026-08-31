# Teammate candidate — occ_occ_r14_multiscale_fixed__table_core

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__occ_occ_r14_multiscale_fixed_table_core`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `occ_occ_r14_multiscale_fixed__table_core`
- **Original source:** `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Unknown / not recoverable from repository history
- **Features:** occurrence features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`
- **Known score:** | wcv | 1.7506208239371757 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 2 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — occ_occ_r14_multiscale_fixed__table_core

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`

| Field | Value |
|---|---|
| name | occ_occ_r14_multiscale_fixed__table_core |
| family | occurrence_overlay |
| wcv | 1.7506208239371757 |
| base_wcv | 1.749803702558191 |
| delta | 0.0008171213789846939 |
| wins | 0 |
| wins_recent | 0 |
| latest_delta | 0.0006765515846520742 |
| worst_delta | 0.00100588980305516 |
| raw_delta | 0.00022017071366747427 |
| offset_mean | -0.049332016413838685 |
| offset_std | 0.029633817241174157 |
| fold_scores | [1.7702496583950647, 1.7636794985501134, 1.7519721406865254, 1.7442268926020303] |
| fold_deltas | [0.00100588980305516, 0.0009544782547865616, 0.000982390423731383, 0.0006765515846520742] |
| notes | raw=occ_r14_multiscale;base=table_core;adaptive=False |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | occ_occ_r14_multiscale_fixed__table_core |
| family | occurrence_overlay |
| wcv | 1.7506208239371757 |
| base_wcv | 1.749803702558191 |
| delta | 0.0008171213789846939 |
| wins | 0 |
| wins_recent | 0 |
| latest_delta | 0.0006765515846520742 |
| worst_delta | 0.00100588980305516 |
| raw_delta | 0.00022017071366747427 |
| offset_mean | -0.049332016413838685 |
| offset_std | 0.029633817241174157 |
| fold_scores | [1.7702496583950647, 1.7636794985501134, 1.7519721406865254, 1.7442268926020303] |
| fold_deltas | [0.00100588980305516, 0.0009544782547865616, 0.000982390423731383, 0.0006765515846520742] |
| notes | raw=occ_r14_multiscale;base=table_core;adaptive=False |
