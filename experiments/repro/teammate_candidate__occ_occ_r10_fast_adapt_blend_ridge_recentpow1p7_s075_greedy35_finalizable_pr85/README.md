# Teammate candidate — occ_occ_r10_fast_adapt__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__occ_occ_r10_fast_adapt_blend_ridge_recentpow1p7_s075_greedy35_finalizable_pr85`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `occ_occ_r10_fast_adapt__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85`
- **Original source:** `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/FAMILY_BEST.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Ridge, blend
- **Features:** occurrence features, 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`
- **Known score:** | wcv | 1.7481791013308823 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 3 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — occ_occ_r10_fast_adapt__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`

| Field | Value |
|---|---|
| name | occ_occ_r10_fast_adapt__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85 |
| family | occurrence_overlay |
| wcv | 1.7481791013308823 |
| base_wcv | 1.749803702558191 |
| delta | -0.0016246012273087196 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0019148299253186618 |
| worst_delta | 0.00023607115140222845 |
| raw_delta | -0.0019993064472014323 |
| offset_mean | 0.008682768198637461 |
| offset_std | 0.07020110465764609 |
| fold_scores | [1.7694798397434117, 1.7614240927796827, 1.7493186014809952, 1.7416355110920596] |
| fold_deltas | [0.00023607115140222845, -0.0013009275156441458, -0.001671148781798859, -0.0019148299253186618] |
| notes | raw=occ_r10_fast;base=blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85;adaptive=True |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/FAMILY_BEST.csv`

| Field | Value |
|---|---|
| family | occurrence_overlay |
| best_name | occ_occ_r10_fast_adapt__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85 |
| delta | -0.0016246012273087196 |
| wins_recent | 3 |
| latest_delta | -0.0019148299253186618 |

## Evidence row 3

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | occ_occ_r10_fast_adapt__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85 |
| family | occurrence_overlay |
| wcv | 1.7481791013308823 |
| base_wcv | 1.749803702558191 |
| delta | -0.0016246012273087196 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0019148299253186618 |
| worst_delta | 0.00023607115140222845 |
| raw_delta | -0.0019993064472014323 |
| offset_mean | 0.008682768198637461 |
| offset_std | 0.07020110465764609 |
| fold_scores | [1.7694798397434117, 1.7614240927796827, 1.7493186014809952, 1.7416355110920596] |
| fold_deltas | [0.00023607115140222845, -0.0013009275156441458, -0.001671148781798859, -0.0019148299253186618] |
| notes | raw=occ_r10_fast;base=blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85;adaptive=True |
