# Teammate candidate — metaocc_l31_plain__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__metaocc_l31_plain_blend_ridge_recentpow1p7_s075_greedy35_finalizable_pr85`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `metaocc_l31_plain__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85`
- **Original source:** `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/FAMILY_BEST.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Ridge, blend
- **Features:** occurrence features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`
- **Known score:** | wcv | 1.7480653509088204 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 3 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — metaocc_l31_plain__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`

| Field | Value |
|---|---|
| name | metaocc_l31_plain__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85 |
| family | occurrence_meta |
| wcv | 1.7480653509088204 |
| base_wcv | 1.749803702558191 |
| delta | -0.0017383516493707063 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.002131191792551501 |
| worst_delta | 0.0 |
| raw_delta | -0.0014256043542049271 |
| offset_mean | 0.016146919537840646 |
| offset_std | 0.09803366012856797 |
| fold_scores | [1.7692437685920095, 1.7619821063791496, 1.7491047721208455, 1.7414191492248268] |
| fold_deltas | [0.0, -0.0007429139161772014, -0.0018849781419485456, -0.002131191792551501] |
| notes | base=blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85;occ=['occ_r10_fast', 'occ_r16_bal', 'occ_r22_stable', 'occ_r14_multiscale', 'occ_r18_wide', 'occ_r24_multiscale', 'occ_r12_wide', 'occ_r20_shallow'] |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/FAMILY_BEST.csv`

| Field | Value |
|---|---|
| family | occurrence_meta |
| best_name | metaocc_l31_plain__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85 |
| delta | -0.0017383516493707063 |
| wins_recent | 3 |
| latest_delta | -0.002131191792551501 |

## Evidence row 3

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | metaocc_l31_plain__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85 |
| family | occurrence_meta |
| wcv | 1.7480653509088204 |
| base_wcv | 1.749803702558191 |
| delta | -0.0017383516493707063 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.002131191792551501 |
| worst_delta | 0.0 |
| raw_delta | -0.0014256043542049271 |
| offset_mean | 0.016146919537840646 |
| offset_std | 0.09803366012856797 |
| fold_scores | [1.7692437685920095, 1.7619821063791496, 1.7491047721208455, 1.7414191492248268] |
| fold_deltas | [0.0, -0.0007429139161772014, -0.0018849781419485456, -0.002131191792551501] |
| notes | base=blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85;occ=['occ_r10_fast', 'occ_r16_bal', 'occ_r22_stable', 'occ_r14_multiscale', 'occ_r18_wide', 'occ_r24_multiscale', 'occ_r12_wide', 'occ_r20_shallow'] |
