# Teammate candidate — xmeta_stable4_p17_l31_risk__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__xmeta_stable4_p17_l31_risk_blend_ridge_recentpow1p7_s075_greedy35_finalizable_pr85`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `xmeta_stable4_p17_l31_risk__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85`
- **Original source:** `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/MATERIALIZED_CANDIDATES.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Ridge, blend
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`
- **Known score:** | wcv | 1.7480617668417342 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 2 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — xmeta_stable4_p17_l31_risk__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`

| Field | Value |
|---|---|
| name | xmeta_stable4_p17_l31_risk__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85 |
| family | xmeta_risk |
| wcv | 1.7480617668417342 |
| base_wcv | 1.749803702558191 |
| delta | -0.001741935716456755 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.002013303507088615 |
| worst_delta | 0.0 |
| raw_delta | -0.002171049648820785 |
| offset_mean | -0.0007000085351995503 |
| offset_std | 0.07348541860471569 |
| fold_scores | [1.7692437685920095, 1.7611417175488329, 1.7492757497135054, 1.7415370375102897] |
| fold_deltas | [0.0, -0.0015833027464939864, -0.001714000549288608, -0.002013303507088615] |
| notes | subset=['occ_r18_wide', 'occ_r20_shallow', 'occ_r22_stable', 'occ_r24_multiscale'];power=1.7;leaves=31;base=blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85 |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/MATERIALIZED_CANDIDATES.csv`

| Field | Value |
|---|---|
| name | xmeta_stable4_p17_l31_risk__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85 |
| family | xmeta_risk |
| delta | -0.001741935716456755 |
| wins_recent | 3 |
| latest_delta | -0.002013303507088615 |
| friend_corr | 0.9996427778883469 |
| friend_std | 0.04447107126781856 |
| friend_mae | 0.03368166666091005 |
| friend_pct02 | 0.606192 |
| friend_pct05 | 0.229184 |
| friend_pct10 | 0.031256 |
