# Teammate candidate — xmeta_fast4_p23_l23_risk__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__xmeta_fast4_p23_l23_risk_blend_ridge_recentpow1p7_s075_greedy35_finalizable_pr85`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `xmeta_fast4_p23_l23_risk__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85`
- **Original source:** `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/MATERIALIZED_CANDIDATES.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Ridge, blend
- **Features:** 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`
- **Known score:** | wcv | 1.7480270379086813 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 2 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — xmeta_fast4_p23_l23_risk__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`

| Field | Value |
|---|---|
| name | xmeta_fast4_p23_l23_risk__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85 |
| family | xmeta_risk |
| wcv | 1.7480270379086813 |
| base_wcv | 1.749803702558191 |
| delta | -0.0017766646495096813 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0020275650322754135 |
| worst_delta | 0.0 |
| raw_delta | -0.0022086209989054416 |
| offset_mean | -0.0008806741796752136 |
| offset_std | 0.07337811499331447 |
| fold_scores | [1.7692437685920095, 1.7609739370566908, 1.7492579295110016, 1.7415227759851029] |
| fold_deltas | [0.0, -0.001751083238636042, -0.0017318207517924566, -0.0020275650322754135] |
| notes | subset=['occ_r10_fast', 'occ_r12_wide', 'occ_r14_multiscale', 'occ_r16_bal'];power=2.3;leaves=23;base=blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85 |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/MATERIALIZED_CANDIDATES.csv`

| Field | Value |
|---|---|
| name | xmeta_fast4_p23_l23_risk__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85 |
| family | xmeta_risk |
| delta | -0.0017766646495096813 |
| wins_recent | 3 |
| latest_delta | -0.0020275650322754135 |
| friend_corr | 0.999633493801394 |
| friend_std | 0.044997363096116065 |
| friend_mae | 0.034084802376435755 |
| friend_pct02 | 0.609852 |
| friend_pct05 | 0.233188 |
| friend_pct10 | 0.032952 |
