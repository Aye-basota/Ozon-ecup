# Teammate candidate — occ_occ_r24_multiscale_fixed__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__occ_occ_r24_multiscale_fixed_blend_ridge_recentpow1p7_s075_greedy35_finalizable_pr85`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `occ_occ_r24_multiscale_fixed__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85`
- **Original source:** `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Ridge, blend
- **Features:** occurrence features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`
- **Known score:** | wcv | 1.7488959816293146 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 2 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — occ_occ_r24_multiscale_fixed__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`

| Field | Value |
|---|---|
| name | occ_occ_r24_multiscale_fixed__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85 |
| family | occurrence_overlay |
| wcv | 1.7488959816293146 |
| base_wcv | 1.749803702558191 |
| delta | -0.0009077209288763803 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0011461280844440314 |
| worst_delta | 0.0008209556030795984 |
| raw_delta | -0.0017193858661657716 |
| offset_mean | -0.009856223981247712 |
| offset_std | 0.05072908991814258 |
| fold_scores | [1.770064724195089, 1.7616531972822005, 1.750208725554189, 1.7424042129329342] |
| fold_deltas | [0.0008209556030795984, -0.0010718230131263606, -0.0007810247086050826, -0.0011461280844440314] |
| notes | raw=occ_r24_multiscale;base=blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85;adaptive=False |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | occ_occ_r24_multiscale_fixed__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85 |
| family | occurrence_overlay |
| wcv | 1.7488959816293146 |
| base_wcv | 1.749803702558191 |
| delta | -0.0009077209288763803 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0011461280844440314 |
| worst_delta | 0.0008209556030795984 |
| raw_delta | -0.0017193858661657716 |
| offset_mean | -0.009856223981247712 |
| offset_std | 0.05072908991814258 |
| fold_scores | [1.770064724195089, 1.7616531972822005, 1.750208725554189, 1.7424042129329342] |
| fold_deltas | [0.0008209556030795984, -0.0010718230131263606, -0.0007810247086050826, -0.0011461280844440314] |
| notes | raw=occ_r24_multiscale;base=blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85;adaptive=False |
