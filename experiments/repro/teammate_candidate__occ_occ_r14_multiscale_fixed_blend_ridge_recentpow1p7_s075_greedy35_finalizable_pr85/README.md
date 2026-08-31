# Teammate candidate — occ_occ_r14_multiscale_fixed__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__occ_occ_r14_multiscale_fixed_blend_ridge_recentpow1p7_s075_greedy35_finalizable_pr85`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `occ_occ_r14_multiscale_fixed__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85`
- **Original source:** `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Ridge, blend
- **Features:** occurrence features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`
- **Known score:** | wcv | 1.7489700507624906 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 2 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — occ_occ_r14_multiscale_fixed__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`

| Field | Value |
|---|---|
| name | occ_occ_r14_multiscale_fixed__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85 |
| family | occurrence_overlay |
| wcv | 1.7489700507624906 |
| base_wcv | 1.749803702558191 |
| delta | -0.0008336517957005333 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0011272748206736516 |
| worst_delta | 0.00100588980305516 |
| raw_delta | -0.0016323974622074614 |
| offset_mean | -0.009341883504785554 |
| offset_std | 0.05163112775735017 |
| fold_scores | [1.7702496583950647, 1.7618080601188013, 1.7503251133077633, 1.7424230661967046] |
| fold_deltas | [0.00100588980305516, -0.0009169601765255386, -0.0006646369550307174, -0.0011272748206736516] |
| notes | raw=occ_r14_multiscale;base=blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85;adaptive=False |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | occ_occ_r14_multiscale_fixed__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85 |
| family | occurrence_overlay |
| wcv | 1.7489700507624906 |
| base_wcv | 1.749803702558191 |
| delta | -0.0008336517957005333 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0011272748206736516 |
| worst_delta | 0.00100588980305516 |
| raw_delta | -0.0016323974622074614 |
| offset_mean | -0.009341883504785554 |
| offset_std | 0.05163112775735017 |
| fold_scores | [1.7702496583950647, 1.7618080601188013, 1.7503251133077633, 1.7424230661967046] |
| fold_deltas | [0.00100588980305516, -0.0009169601765255386, -0.0006646369550307174, -0.0011272748206736516] |
| notes | raw=occ_r14_multiscale;base=blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85;adaptive=False |
