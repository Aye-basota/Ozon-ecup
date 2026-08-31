# Teammate candidate — occ_occ_r24_multiscale_adapt__table_core

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__occ_occ_r24_multiscale_adapt_table_core`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `occ_occ_r24_multiscale_adapt__table_core`
- **Original source:** `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Unknown / not recoverable from repository history
- **Features:** occurrence features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`
- **Known score:** | wcv | 1.750310548676825 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 2 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — occ_occ_r24_multiscale_adapt__table_core

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`

| Field | Value |
|---|---|
| name | occ_occ_r24_multiscale_adapt__table_core |
| family | occurrence_overlay |
| wcv | 1.750310548676825 |
| base_wcv | 1.749803702558191 |
| delta | 0.0005068461186337488 |
| wins | 0 |
| wins_recent | 0 |
| latest_delta | 0.0004718297314023623 |
| worst_delta | 0.0008209556030795984 |
| raw_delta | -0.0003674048076119559 |
| offset_mean | -0.013819423364409025 |
| offset_std | 0.04742546510053828 |
| fold_scores | [1.770064724195089, 1.763473876092235, 1.7513670969456419, 1.7440221707487806] |
| fold_deltas | [0.0008209556030795984, 0.0007488557969081988, 0.00037734668284783446, 0.0004718297314023623] |
| notes | raw=occ_r24_multiscale;base=table_core;adaptive=True |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | occ_occ_r24_multiscale_adapt__table_core |
| family | occurrence_overlay |
| wcv | 1.750310548676825 |
| base_wcv | 1.749803702558191 |
| delta | 0.0005068461186337488 |
| wins | 0 |
| wins_recent | 0 |
| latest_delta | 0.0004718297314023623 |
| worst_delta | 0.0008209556030795984 |
| raw_delta | -0.0003674048076119559 |
| offset_mean | -0.013819423364409025 |
| offset_std | 0.04742546510053828 |
| fold_scores | [1.770064724195089, 1.763473876092235, 1.7513670969456419, 1.7440221707487806] |
| fold_deltas | [0.0008209556030795984, 0.0007488557969081988, 0.00037734668284783446, 0.0004718297314023623] |
| notes | raw=occ_r24_multiscale;base=table_core;adaptive=True |
