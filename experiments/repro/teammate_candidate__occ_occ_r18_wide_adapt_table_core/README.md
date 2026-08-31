# Teammate candidate — occ_occ_r18_wide_adapt__table_core

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__occ_occ_r18_wide_adapt_table_core`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `occ_occ_r18_wide_adapt__table_core`
- **Original source:** `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Unknown / not recoverable from repository history
- **Features:** occurrence features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`
- **Known score:** | wcv | 1.7498433765409114 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 2 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — occ_occ_r18_wide_adapt__table_core

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`

| Field | Value |
|---|---|
| name | occ_occ_r18_wide_adapt__table_core |
| family | occurrence_overlay |
| wcv | 1.7498433765409114 |
| base_wcv | 1.749803702558191 |
| delta | 3.967398272015643e-05 |
| wins | 0 |
| wins_recent | 0 |
| latest_delta | 3.7820947255795545e-05 |
| worst_delta | 0.0001382020676321538 |
| raw_delta | -0.000812365333871649 |
| offset_mean | -0.013604353324608388 |
| offset_std | 0.04973619605945209 |
| fold_scores | [1.7693819706596416, 1.7627773716380095, 1.7510021596152336, 1.743588161964634] |
| fold_deltas | [0.0001382020676321538, 5.235134268266606e-05, 1.2409352439624044e-05, 3.7820947255795545e-05] |
| notes | raw=occ_r18_wide;base=table_core;adaptive=True |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | occ_occ_r18_wide_adapt__table_core |
| family | occurrence_overlay |
| wcv | 1.7498433765409114 |
| base_wcv | 1.749803702558191 |
| delta | 3.967398272015643e-05 |
| wins | 0 |
| wins_recent | 0 |
| latest_delta | 3.7820947255795545e-05 |
| worst_delta | 0.0001382020676321538 |
| raw_delta | -0.000812365333871649 |
| offset_mean | -0.013604353324608388 |
| offset_std | 0.04973619605945209 |
| fold_scores | [1.7693819706596416, 1.7627773716380095, 1.7510021596152336, 1.743588161964634] |
| fold_deltas | [0.0001382020676321538, 5.235134268266606e-05, 1.2409352439624044e-05, 3.7820947255795545e-05] |
| notes | raw=occ_r18_wide;base=table_core;adaptive=True |
