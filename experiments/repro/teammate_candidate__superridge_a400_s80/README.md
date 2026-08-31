# Teammate candidate — superridge_a400_s80

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__superridge_a400_s80`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `superridge_a400_s80`
- **Original source:** `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/STABLE_BRANCH_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/STABLE_BRANCH_VALIDATION.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Ridge, blend
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`
- **Known score:** | wcv | 1.7485916639159782 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 4 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — superridge_a400_s80

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`

| Field | Value |
|---|---|
| name | superridge_a400_s80 |
| family | super_ridge |
| wcv | 1.7485916639159782 |
| base_wcv | 1.749803702558191 |
| delta | -0.001212038642212665 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016305717303823908 |
| worst_delta | 0.0 |
| raw_delta | -0.0019060975986625373 |
| offset_mean | -0.027905254889166918 |
| offset_std | 0.052815942319044205 |
| fold_scores | [1.7692437685920095, 1.7623444265899295, 1.74989604566796, 1.7419197692869959] |
| fold_deltas | [0.0, -0.0003805937053973363, -0.001093704594834044, -0.0016305717303823908] |
| notes | ['blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85', 'blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr70', 'bias_p_spread_cand_pband_stack', 'bias_p_cand_pband_stack', 'cand_pband_stack', 'cand_simplex_l0p08', 'cand_simplex_l0p2', 'ridge_recentpow1p7_s075'] |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/STABLE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | superridge_a400_s80 |
| family | super_ridge |
| wcv | 1.7485916639159782 |
| base_wcv | 1.749803702558191 |
| delta | -0.001212038642212665 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016305717303823908 |
| worst_delta | 0.0 |
| raw_delta | -0.0019060975986625373 |
| offset_mean | -0.027905254889166918 |
| offset_std | 0.052815942319044205 |
| fold_scores | [1.7692437685920095, 1.7623444265899295, 1.74989604566796, 1.7419197692869959] |
| fold_deltas | [0.0, -0.0003805937053973363, -0.001093704594834044, -0.0016305717303823908] |
| notes | ['blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85', 'blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr70', 'bias_p_spread_cand_pband_stack', 'bias_p_cand_pband_stack', 'cand_pband_stack', 'cand_simplex_l0p08', 'cand_simplex_l0p2', 'ridge_recentpow1p7_s075'] |

## Evidence row 3

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`

| Field | Value |
|---|---|
| name | superridge_a400_s80 |
| family | super_ridge |
| wcv | 1.7485916639159782 |
| base_wcv | 1.749803702558191 |
| delta | -0.001212038642212665 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016305717303823908 |
| worst_delta | 0.0 |
| raw_delta | -0.0019060975986625373 |
| offset_mean | -0.027905254889166918 |
| offset_std | 0.052815942319044205 |
| fold_scores | [1.7692437685920095, 1.7623444265899295, 1.74989604566796, 1.7419197692869959] |
| fold_deltas | [0.0, -0.0003805937053973363, -0.001093704594834044, -0.0016305717303823908] |
| notes | ['blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85', 'blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr70', 'bias_p_spread_cand_pband_stack', 'bias_p_cand_pband_stack', 'cand_pband_stack', 'cand_simplex_l0p08', 'cand_simplex_l0p2', 'ridge_recentpow1p7_s075'] |

## Evidence row 4

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/STABLE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | superridge_a400_s80 |
| family | super_ridge |
| wcv | 1.7485916639159782 |
| base_wcv | 1.749803702558191 |
| delta | -0.001212038642212665 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016305717303823908 |
| worst_delta | 0.0 |
| raw_delta | -0.0019060975986625373 |
| offset_mean | -0.027905254889166918 |
| offset_std | 0.052815942319044205 |
| fold_scores | [1.7692437685920095, 1.7623444265899295, 1.74989604566796, 1.7419197692869959] |
| fold_deltas | [0.0, -0.0003805937053973363, -0.001093704594834044, -0.0016305717303823908] |
| notes | ['blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85', 'blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr70', 'bias_p_spread_cand_pband_stack', 'bias_p_cand_pband_stack', 'cand_pband_stack', 'cand_simplex_l0p08', 'cand_simplex_l0p2', 'ridge_recentpow1p7_s075'] |
