# Teammate candidate — occ_occ_r22_stable_fixed__super_pband_l0p14

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__occ_occ_r22_stable_fixed_super_pband_l0p14`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `occ_occ_r22_stable_fixed__super_pband_l0p14`
- **Original source:** `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Unknown / not recoverable from repository history
- **Features:** occurrence features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`
- **Known score:** | wcv | 1.7484836468318004 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 2 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — occ_occ_r22_stable_fixed__super_pband_l0p14

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`

| Field | Value |
|---|---|
| name | occ_occ_r22_stable_fixed__super_pband_l0p14 |
| family | occurrence_overlay |
| wcv | 1.7484836468318004 |
| base_wcv | 1.749803702558191 |
| delta | -0.0013200557263906899 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0014392292118001748 |
| worst_delta | 1.521810399363055e-05 |
| raw_delta | -0.0021392938889106244 |
| offset_mean | -0.013981501053357569 |
| offset_std | 0.05036918811619113 |
| fold_scores | [1.7692589866960031, 1.7612334958790914, 1.7496599573945486, 1.742111111805578] |
| fold_deltas | [1.521810399363055e-05, -0.0014915244162354568, -0.0013297928682454163, -0.0014392292118001748] |
| notes | raw=occ_r22_stable;base=super_pband_l0p14;adaptive=False |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | occ_occ_r22_stable_fixed__super_pband_l0p14 |
| family | occurrence_overlay |
| wcv | 1.7484836468318004 |
| base_wcv | 1.749803702558191 |
| delta | -0.0013200557263906899 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0014392292118001748 |
| worst_delta | 1.521810399363055e-05 |
| raw_delta | -0.0021392938889106244 |
| offset_mean | -0.013981501053357569 |
| offset_std | 0.05036918811619113 |
| fold_scores | [1.7692589866960031, 1.7612334958790914, 1.7496599573945486, 1.742111111805578] |
| fold_deltas | [1.521810399363055e-05, -0.0014915244162354568, -0.0013297928682454163, -0.0014392292118001748] |
| notes | raw=occ_r22_stable;base=super_pband_l0p14;adaptive=False |
