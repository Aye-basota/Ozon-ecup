# Teammate candidate — occ_occ_r18_wide_fixed__super_pband_l0p28

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__occ_occ_r18_wide_fixed_super_pband_l0p28`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `occ_occ_r18_wide_fixed__super_pband_l0p28`
- **Original source:** `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Unknown / not recoverable from repository history
- **Features:** occurrence features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`
- **Known score:** | wcv | 1.7486180843837849 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 2 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — occ_occ_r18_wide_fixed__super_pband_l0p28

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`

| Field | Value |
|---|---|
| name | occ_occ_r18_wide_fixed__super_pband_l0p28 |
| family | occurrence_overlay |
| wcv | 1.7486180843837849 |
| base_wcv | 1.749803702558191 |
| delta | -0.0011856181744063518 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0012464889834675752 |
| worst_delta | 0.0001382020676321538 |
| raw_delta | -0.002004643515104506 |
| offset_mean | -0.013497438213662522 |
| offset_std | 0.05075159907955697 |
| fold_scores | [1.7693819706596416, 1.7611344177743908, 1.7497974108192653, 1.7423038520339107] |
| fold_deltas | [0.0001382020676321538, -0.001590602520936013, -0.0011923394435287005, -0.0012464889834675752] |
| notes | raw=occ_r18_wide;base=super_pband_l0p28;adaptive=False |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | occ_occ_r18_wide_fixed__super_pband_l0p28 |
| family | occurrence_overlay |
| wcv | 1.7486180843837849 |
| base_wcv | 1.749803702558191 |
| delta | -0.0011856181744063518 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0012464889834675752 |
| worst_delta | 0.0001382020676321538 |
| raw_delta | -0.002004643515104506 |
| offset_mean | -0.013497438213662522 |
| offset_std | 0.05075159907955697 |
| fold_scores | [1.7693819706596416, 1.7611344177743908, 1.7497974108192653, 1.7423038520339107] |
| fold_deltas | [0.0001382020676321538, -0.001590602520936013, -0.0011923394435287005, -0.0012464889834675752] |
| notes | raw=occ_r18_wide;base=super_pband_l0p28;adaptive=False |
