# Teammate candidate — metaocc_l31_plain__table_core

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__metaocc_l31_plain_table_core`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `metaocc_l31_plain__table_core`
- **Original source:** `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Unknown / not recoverable from repository history
- **Features:** occurrence features, 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`
- **Known score:** | wcv | 1.749055879854158 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 2 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — metaocc_l31_plain__table_core

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`

| Field | Value |
|---|---|
| name | metaocc_l31_plain__table_core |
| family | occurrence_meta |
| wcv | 1.749055879854158 |
| base_wcv | 1.749803702558191 |
| delta | -0.000747822704033189 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0008168927966789585 |
| worst_delta | 0.0 |
| raw_delta | -0.0012101044083093094 |
| offset_mean | -0.003577329204588765 |
| offset_std | 0.07256805334311052 |
| fold_scores | [1.7692437685920095, 1.7621285799464184, 1.7501174208904817, 1.7427334482206993] |
| fold_deltas | [0.0, -0.0005964403489084891, -0.0008723293723122971, -0.0008168927966789585] |
| notes | base=table_core;occ=['occ_r10_fast', 'occ_r16_bal', 'occ_r22_stable', 'occ_r14_multiscale', 'occ_r18_wide', 'occ_r24_multiscale', 'occ_r12_wide', 'occ_r20_shallow'] |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | metaocc_l31_plain__table_core |
| family | occurrence_meta |
| wcv | 1.749055879854158 |
| base_wcv | 1.749803702558191 |
| delta | -0.000747822704033189 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0008168927966789585 |
| worst_delta | 0.0 |
| raw_delta | -0.0012101044083093094 |
| offset_mean | -0.003577329204588765 |
| offset_std | 0.07256805334311052 |
| fold_scores | [1.7692437685920095, 1.7621285799464184, 1.7501174208904817, 1.7427334482206993] |
| fold_deltas | [0.0, -0.0005964403489084891, -0.0008723293723122971, -0.0008168927966789585] |
| notes | base=table_core;occ=['occ_r10_fast', 'occ_r16_bal', 'occ_r22_stable', 'occ_r14_multiscale', 'occ_r18_wide', 'occ_r24_multiscale', 'occ_r12_wide', 'occ_r20_shallow'] |
