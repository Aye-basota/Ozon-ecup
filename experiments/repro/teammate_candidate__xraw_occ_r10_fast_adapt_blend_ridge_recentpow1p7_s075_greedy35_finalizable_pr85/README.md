# Teammate candidate — xraw_occ_r10_fast_adapt__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__xraw_occ_r10_fast_adapt_blend_ridge_recentpow1p7_s075_greedy35_finalizable_pr85`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `xraw_occ_r10_fast_adapt__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85`
- **Original source:** `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/FINAL_FOUR.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/MATERIALIZED_CANDIDATES.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Ridge, blend
- **Features:** 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`
- **Known score:** | wcv | 1.7481791013308823 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** | file | C:\Users\Dimentiy\repoVScode\Ozon-ecup\src\DL\best_bas\_best_bas_extra90m\submissions\submission_extra90_3_xraw_occ_r10_fast_adapt__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85.csv |
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 3 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — xraw_occ_r10_fast_adapt__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`

| Field | Value |
|---|---|
| name | xraw_occ_r10_fast_adapt__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85 |
| family | raw_occ_extra |
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
| notes | raw=occ_r10_fast;base=blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85 |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/FINAL_FOUR.csv`

| Field | Value |
|---|---|
| rank | 3 |
| name | xraw_occ_r10_fast_adapt__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85 |
| family | raw_occ_extra |
| delta | -0.0016246012273087196 |
| wins_recent | 3 |
| latest_delta | -0.0019148299253186618 |
| file | C:\Users\Dimentiy\repoVScode\Ozon-ecup\src\DL\best_bas\_best_bas_extra90m\submissions\submission_extra90_3_xraw_occ_r10_fast_adapt__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85.csv |
| corr | 0.9996601138284242 |
| std | 0.042902163680878555 |
| mae | 0.032615393235133135 |
| pct02 | 0.596476 |
| pct05 | 0.21638 |
| pct10 | 0.026476 |

## Evidence row 3

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/MATERIALIZED_CANDIDATES.csv`

| Field | Value |
|---|---|
| name | xraw_occ_r10_fast_adapt__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85 |
| family | raw_occ_extra |
| delta | -0.0016246012273087196 |
| wins_recent | 3 |
| latest_delta | -0.0019148299253186618 |
| friend_corr | 0.9996601138284242 |
| friend_std | 0.042902163680878555 |
| friend_mae | 0.032615393235133135 |
| friend_pct02 | 0.596476 |
| friend_pct05 | 0.21638 |
| friend_pct10 | 0.026476 |
