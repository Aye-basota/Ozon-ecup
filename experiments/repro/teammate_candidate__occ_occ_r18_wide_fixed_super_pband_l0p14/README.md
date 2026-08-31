# Teammate candidate — occ_occ_r18_wide_fixed__super_pband_l0p14

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__occ_occ_r18_wide_fixed_super_pband_l0p14`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `occ_occ_r18_wide_fixed__super_pband_l0p14`
- **Original source:** `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Unknown / not recoverable from repository history
- **Features:** occurrence features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`
- **Known score:** | wcv | 1.7486180309351746 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 2 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — occ_occ_r18_wide_fixed__super_pband_l0p14

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`

| Field | Value |
|---|---|
| name | occ_occ_r18_wide_fixed__super_pband_l0p14 |
| family | occurrence_overlay |
| wcv | 1.7486180309351746 |
| base_wcv | 1.749803702558191 |
| delta | -0.0011856716230163342 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0012465941968233807 |
| worst_delta | 0.0001382020676321538 |
| raw_delta | -0.002004683271214495 |
| offset_mean | -0.013496365146591015 |
| offset_std | 0.05075235060508408 |
| fold_scores | [1.7693819706596416, 1.7611344177743908, 1.7497974208136895, 1.7423037468205549] |
| fold_deltas | [0.0001382020676321538, -0.001590602520936013, -0.0011923294491045233, -0.0012465941968233807] |
| notes | raw=occ_r18_wide;base=super_pband_l0p14;adaptive=False |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | occ_occ_r18_wide_fixed__super_pband_l0p14 |
| family | occurrence_overlay |
| wcv | 1.7486180309351746 |
| base_wcv | 1.749803702558191 |
| delta | -0.0011856716230163342 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0012465941968233807 |
| worst_delta | 0.0001382020676321538 |
| raw_delta | -0.002004683271214495 |
| offset_mean | -0.013496365146591015 |
| offset_std | 0.05075235060508408 |
| fold_scores | [1.7693819706596416, 1.7611344177743908, 1.7497974208136895, 1.7423037468205549] |
| fold_deltas | [0.0001382020676321538, -0.001590602520936013, -0.0011923294491045233, -0.0012465941968233807] |
| notes | raw=occ_r18_wide;base=super_pband_l0p14;adaptive=False |
