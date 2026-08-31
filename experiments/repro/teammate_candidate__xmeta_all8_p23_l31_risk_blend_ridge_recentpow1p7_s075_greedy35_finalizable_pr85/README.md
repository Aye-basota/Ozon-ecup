# Teammate candidate — xmeta_all8_p23_l31_risk__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__xmeta_all8_p23_l31_risk_blend_ridge_recentpow1p7_s075_greedy35_finalizable_pr85`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `xmeta_all8_p23_l31_risk__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85`
- **Original source:** `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/MATERIALIZED_CANDIDATES.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Ridge, blend
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`
- **Known score:** | wcv | 1.748020498532981 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 2 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — xmeta_all8_p23_l31_risk__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`

| Field | Value |
|---|---|
| name | xmeta_all8_p23_l31_risk__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85 |
| family | xmeta_risk |
| wcv | 1.748020498532981 |
| base_wcv | 1.749803702558191 |
| delta | -0.0017832040252099132 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.002060603481424561 |
| worst_delta | 0.0 |
| raw_delta | -0.002217737770697692 |
| offset_mean | -0.001055088568348171 |
| offset_std | 0.07325332446096988 |
| fold_scores | [1.7692437685920095, 1.7610686477818753, 1.7492521283878317, 1.7414897375359537] |
| fold_deltas | [0.0, -0.0016563725134515206, -0.0017376218749622918, -0.002060603481424561] |
| notes | subset=['occ_r10_fast', 'occ_r16_bal', 'occ_r22_stable', 'occ_r14_multiscale', 'occ_r18_wide', 'occ_r24_multiscale', 'occ_r12_wide', 'occ_r20_shallow'];power=2.3;leaves=31;base=blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85 |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/MATERIALIZED_CANDIDATES.csv`

| Field | Value |
|---|---|
| name | xmeta_all8_p23_l31_risk__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85 |
| family | xmeta_risk |
| delta | -0.0017832040252099132 |
| wins_recent | 3 |
| latest_delta | -0.002060603481424561 |
| friend_corr | 0.9996256220459218 |
| friend_std | 0.045449690428658374 |
| friend_mae | 0.034415723557732196 |
| friend_pct02 | 0.6134 |
| friend_pct05 | 0.23702 |
| friend_pct10 | 0.034448 |
