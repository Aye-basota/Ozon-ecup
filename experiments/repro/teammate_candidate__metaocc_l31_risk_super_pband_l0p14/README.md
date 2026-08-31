# Teammate candidate — metaocc_l31_risk__super_pband_l0p14

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__metaocc_l31_risk_super_pband_l0p14`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `metaocc_l31_risk__super_pband_l0p14`
- **Original source:** `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Unknown / not recoverable from repository history
- **Features:** occurrence features, 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`
- **Known score:** | wcv | 1.7480423911596736 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 2 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — metaocc_l31_risk__super_pband_l0p14

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`

| Field | Value |
|---|---|
| name | metaocc_l31_risk__super_pband_l0p14 |
| family | occurrence_meta_risk |
| wcv | 1.7480423911596736 |
| base_wcv | 1.749803702558191 |
| delta | -0.0017613113985176436 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0020227658304829976 |
| worst_delta | 0.0 |
| raw_delta | -0.0021936747708834283 |
| offset_mean | -0.0007334232883614935 |
| offset_std | 0.07334534746010907 |
| fold_scores | [1.7692437685920095, 1.7610799942918618, 1.7492528771810514, 1.7415275751868953] |
| fold_deltas | [0.0, -0.0016450260034650555, -0.0017368730817426403, -0.0020227658304829976] |
| notes | base=super_pband_l0p14;occ=['occ_r10_fast', 'occ_r16_bal', 'occ_r22_stable', 'occ_r14_multiscale', 'occ_r18_wide', 'occ_r24_multiscale', 'occ_r12_wide', 'occ_r20_shallow'] |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | metaocc_l31_risk__super_pband_l0p14 |
| family | occurrence_meta_risk |
| wcv | 1.7480423911596736 |
| base_wcv | 1.749803702558191 |
| delta | -0.0017613113985176436 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0020227658304829976 |
| worst_delta | 0.0 |
| raw_delta | -0.0021936747708834283 |
| offset_mean | -0.0007334232883614935 |
| offset_std | 0.07334534746010907 |
| fold_scores | [1.7692437685920095, 1.7610799942918618, 1.7492528771810514, 1.7415275751868953] |
| fold_deltas | [0.0, -0.0016450260034650555, -0.0017368730817426403, -0.0020227658304829976] |
| notes | base=super_pband_l0p14;occ=['occ_r10_fast', 'occ_r16_bal', 'occ_r22_stable', 'occ_r14_multiscale', 'occ_r18_wide', 'occ_r24_multiscale', 'occ_r12_wide', 'occ_r20_shallow'] |
