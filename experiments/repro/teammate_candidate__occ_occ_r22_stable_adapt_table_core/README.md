# Teammate candidate — occ_occ_r22_stable_adapt__table_core

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__occ_occ_r22_stable_adapt_table_core`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `occ_occ_r22_stable_adapt__table_core`
- **Original source:** `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Unknown / not recoverable from repository history
- **Features:** occurrence features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`
- **Known score:** | wcv | 1.7498327698523852 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 2 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — occ_occ_r22_stable_adapt__table_core

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`

| Field | Value |
|---|---|
| name | occ_occ_r22_stable_adapt__table_core |
| family | occurrence_overlay |
| wcv | 1.7498327698523852 |
| base_wcv | 1.749803702558191 |
| delta | 2.9067294194170166e-05 |
| wins | 1 |
| wins_recent | 1 |
| latest_delta | 6.807517723728118e-05 |
| worst_delta | 6.807517723728118e-05 |
| raw_delta | -0.0008165135736793339 |
| offset_mean | -0.012350130031233262 |
| offset_std | 0.0502366220425094 |
| fold_scores | [1.7692589866960031, 1.762772329571607, 1.7509351430974092, 1.7436184161946156] |
| fold_deltas | [1.521810399363055e-05, 4.730927628004977e-05, -5.460716538485677e-05, 6.807517723728118e-05] |
| notes | raw=occ_r22_stable;base=table_core;adaptive=True |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | occ_occ_r22_stable_adapt__table_core |
| family | occurrence_overlay |
| wcv | 1.7498327698523852 |
| base_wcv | 1.749803702558191 |
| delta | 2.9067294194170166e-05 |
| wins | 1 |
| wins_recent | 1 |
| latest_delta | 6.807517723728118e-05 |
| worst_delta | 6.807517723728118e-05 |
| raw_delta | -0.0008165135736793339 |
| offset_mean | -0.012350130031233262 |
| offset_std | 0.0502366220425094 |
| fold_scores | [1.7692589866960031, 1.762772329571607, 1.7509351430974092, 1.7436184161946156] |
| fold_deltas | [1.521810399363055e-05, 4.730927628004977e-05, -5.460716538485677e-05, 6.807517723728118e-05] |
| notes | raw=occ_r22_stable;base=table_core;adaptive=True |
