# Teammate candidate — occ_occ_r16_bal_adapt__super_pband_l0p28

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__occ_occ_r16_bal_adapt_super_pband_l0p28`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `occ_occ_r16_bal_adapt__super_pband_l0p28`
- **Original source:** `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Unknown / not recoverable from repository history
- **Features:** occurrence features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`
- **Known score:** | wcv | 1.7482981152304455 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 2 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — occ_occ_r16_bal_adapt__super_pband_l0p28

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`

| Field | Value |
|---|---|
| name | occ_occ_r16_bal_adapt__super_pband_l0p28 |
| family | occurrence_overlay |
| wcv | 1.7482981152304455 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015055873277452687 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0017926897010946963 |
| worst_delta | 6.966106818939721e-05 |
| raw_delta | -0.0018017791278095144 |
| offset_mean | 0.005692243578560092 |
| offset_std | 0.07321658668333057 |
| fold_scores | [1.7693134296601989, 1.7614758325122002, 1.7495363558104546, 1.7417576513162836] |
| fold_deltas | [6.966106818939721e-05, -0.0012491877831266596, -0.001453394452339385, -0.0017926897010946963] |
| notes | raw=occ_r16_bal;base=super_pband_l0p28;adaptive=True |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | occ_occ_r16_bal_adapt__super_pband_l0p28 |
| family | occurrence_overlay |
| wcv | 1.7482981152304455 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015055873277452687 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0017926897010946963 |
| worst_delta | 6.966106818939721e-05 |
| raw_delta | -0.0018017791278095144 |
| offset_mean | 0.005692243578560092 |
| offset_std | 0.07321658668333057 |
| fold_scores | [1.7693134296601989, 1.7614758325122002, 1.7495363558104546, 1.7417576513162836] |
| fold_deltas | [6.966106818939721e-05, -0.0012491877831266596, -0.001453394452339385, -0.0017926897010946963] |
| notes | raw=occ_r16_bal;base=super_pband_l0p28;adaptive=True |
