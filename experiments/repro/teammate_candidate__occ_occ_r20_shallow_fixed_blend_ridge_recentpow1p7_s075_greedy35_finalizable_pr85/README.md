# Teammate candidate — occ_occ_r20_shallow_fixed__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__occ_occ_r20_shallow_fixed_blend_ridge_recentpow1p7_s075_greedy35_finalizable_pr85`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `occ_occ_r20_shallow_fixed__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85`
- **Original source:** `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Ridge, blend
- **Features:** occurrence features, 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`
- **Known score:** | wcv | 1.748500410861284 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 2 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — occ_occ_r20_shallow_fixed__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`

| Field | Value |
|---|---|
| name | occ_occ_r20_shallow_fixed__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85 |
| family | occurrence_overlay |
| wcv | 1.748500410861284 |
| base_wcv | 1.749803702558191 |
| delta | -0.001303291696906846 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0014370228946423147 |
| worst_delta | 3.360996094081692e-05 |
| raw_delta | -0.0021253326698886946 |
| offset_mean | -0.012971936546868183 |
| offset_std | 0.050325587558070387 |
| fold_scores | [1.7692773785529503, 1.7613180092053504, 1.749671555243431, 1.742113318122736] |
| fold_deltas | [3.360996094081692e-05, -0.0014070110899764288, -0.0013181950193630332, -0.0014370228946423147] |
| notes | raw=occ_r20_shallow;base=blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85;adaptive=False |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | occ_occ_r20_shallow_fixed__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85 |
| family | occurrence_overlay |
| wcv | 1.748500410861284 |
| base_wcv | 1.749803702558191 |
| delta | -0.001303291696906846 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0014370228946423147 |
| worst_delta | 3.360996094081692e-05 |
| raw_delta | -0.0021253326698886946 |
| offset_mean | -0.012971936546868183 |
| offset_std | 0.050325587558070387 |
| fold_scores | [1.7692773785529503, 1.7613180092053504, 1.749671555243431, 1.742113318122736] |
| fold_deltas | [3.360996094081692e-05, -0.0014070110899764288, -0.0013181950193630332, -0.0014370228946423147] |
| notes | raw=occ_r20_shallow;base=blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85;adaptive=False |
