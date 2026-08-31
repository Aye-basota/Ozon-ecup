# Teammate candidate — occ_occ_r20_shallow_adapt__table_core

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__occ_occ_r20_shallow_adapt_table_core`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `occ_occ_r20_shallow_adapt__table_core`
- **Original source:** `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Unknown / not recoverable from repository history
- **Features:** occurrence features, 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`
- **Known score:** | wcv | 1.7498169576113456 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 2 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — occ_occ_r20_shallow_adapt__table_core

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`

| Field | Value |
|---|---|
| name | occ_occ_r20_shallow_adapt__table_core |
| family | occurrence_overlay |
| wcv | 1.7498169576113456 |
| base_wcv | 1.749803702558191 |
| delta | 1.3255053154711168e-05 |
| wins | 1 |
| wins_recent | 1 |
| latest_delta | 9.913699331232095e-06 |
| worst_delta | 0.00015482652728193713 |
| raw_delta | -0.0008372906929622855 |
| offset_mean | -0.01227654776155164 |
| offset_std | 0.04977334458901612 |
| fold_scores | [1.7692773785529503, 1.7628798468226088, 1.7509338135595856, 1.7435602547167095] |
| fold_deltas | [3.360996094081692e-05, 0.00015482652728193713, -5.5936703208470107e-05, 9.913699331232095e-06] |
| notes | raw=occ_r20_shallow;base=table_core;adaptive=True |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | occ_occ_r20_shallow_adapt__table_core |
| family | occurrence_overlay |
| wcv | 1.7498169576113456 |
| base_wcv | 1.749803702558191 |
| delta | 1.3255053154711168e-05 |
| wins | 1 |
| wins_recent | 1 |
| latest_delta | 9.913699331232095e-06 |
| worst_delta | 0.00015482652728193713 |
| raw_delta | -0.0008372906929622855 |
| offset_mean | -0.01227654776155164 |
| offset_std | 0.04977334458901612 |
| fold_scores | [1.7692773785529503, 1.7628798468226088, 1.7509338135595856, 1.7435602547167095] |
| fold_deltas | [3.360996094081692e-05, 0.00015482652728193713, -5.5936703208470107e-05, 9.913699331232095e-06] |
| notes | raw=occ_r20_shallow;base=table_core;adaptive=True |
