# Teammate candidate — occ_occ_r20_shallow_adapt__super_pband_l0p06

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__occ_occ_r20_shallow_adapt_super_pband_l0p06`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `occ_occ_r20_shallow_adapt__super_pband_l0p06`
- **Original source:** `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Unknown / not recoverable from repository history
- **Features:** occurrence features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`
- **Known score:** | wcv | 1.7482487947688479 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 2 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — occ_occ_r20_shallow_adapt__super_pband_l0p06

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`

| Field | Value |
|---|---|
| name | occ_occ_r20_shallow_adapt__super_pband_l0p06 |
| family | occurrence_overlay |
| wcv | 1.7482487947688479 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015549077893432627 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001826606974538958 |
| worst_delta | 3.360996094081692e-05 |
| raw_delta | -0.001880921492157454 |
| offset_mean | 0.008401809989769284 |
| offset_std | 0.07252454362497895 |
| fold_scores | [1.7692773785529503, 1.7615408577376788, 1.7493957387904235, 1.7417237340428393] |
| fold_deltas | [3.360996094081692e-05, -0.001184162557648083, -0.0015940114723704824, -0.001826606974538958] |
| notes | raw=occ_r20_shallow;base=super_pband_l0p06;adaptive=True |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | occ_occ_r20_shallow_adapt__super_pband_l0p06 |
| family | occurrence_overlay |
| wcv | 1.7482487947688479 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015549077893432627 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001826606974538958 |
| worst_delta | 3.360996094081692e-05 |
| raw_delta | -0.001880921492157454 |
| offset_mean | 0.008401809989769284 |
| offset_std | 0.07252454362497895 |
| fold_scores | [1.7692773785529503, 1.7615408577376788, 1.7493957387904235, 1.7417237340428393] |
| fold_deltas | [3.360996094081692e-05, -0.001184162557648083, -0.0015940114723704824, -0.001826606974538958] |
| notes | raw=occ_r20_shallow;base=super_pband_l0p06;adaptive=True |
