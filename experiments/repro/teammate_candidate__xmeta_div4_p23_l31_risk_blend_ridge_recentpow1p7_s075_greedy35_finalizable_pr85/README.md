# Teammate candidate — xmeta_div4_p23_l31_risk__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__xmeta_div4_p23_l31_risk_blend_ridge_recentpow1p7_s075_greedy35_finalizable_pr85`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `xmeta_div4_p23_l31_risk__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85`
- **Original source:** `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/FINAL_FOUR.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/MATERIALIZED_CANDIDATES.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Ridge, blend
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`
- **Known score:** | wcv | 1.7479830389407023 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** | file | C:\Users\Dimentiy\repoVScode\Ozon-ecup\src\DL\best_bas\_best_bas_extra90m\submissions\submission_extra90_2_xmeta_div4_p23_l31_risk__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85.csv |
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 3 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — xmeta_div4_p23_l31_risk__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`

| Field | Value |
|---|---|
| name | xmeta_div4_p23_l31_risk__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85 |
| family | xmeta_risk |
| wcv | 1.7479830389407023 |
| base_wcv | 1.749803702558191 |
| delta | -0.0018206636174889232 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0020723716645272283 |
| worst_delta | 0.0 |
| raw_delta | -0.0022529321848135274 |
| offset_mean | -0.0009213790207031939 |
| offset_std | 0.07337046980634934 |
| fold_scores | [1.7692437685920095, 1.7609318654911044, 1.7492035824283763, 1.741477969352851] |
| fold_deltas | [0.0, -0.0017931548042224854, -0.0017861678344177623, -0.0020723716645272283] |
| notes | subset=['occ_r10_fast', 'occ_r14_multiscale', 'occ_r20_shallow', 'occ_r24_multiscale'];power=2.3;leaves=31;base=blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85 |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/FINAL_FOUR.csv`

| Field | Value |
|---|---|
| rank | 2 |
| name | xmeta_div4_p23_l31_risk__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85 |
| family | xmeta_risk |
| delta | -0.0018206636174889232 |
| wins_recent | 3 |
| latest_delta | -0.0020723716645272283 |
| file | C:\Users\Dimentiy\repoVScode\Ozon-ecup\src\DL\best_bas\_best_bas_extra90m\submissions\submission_extra90_2_xmeta_div4_p23_l31_risk__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85.csv |
| corr | 0.9996313803551532 |
| std | 0.04514010386400232 |
| mae | 0.03415787760187457 |
| pct02 | 0.611248 |
| pct05 | 0.234076 |
| pct10 | 0.0332 |

## Evidence row 3

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/MATERIALIZED_CANDIDATES.csv`

| Field | Value |
|---|---|
| name | xmeta_div4_p23_l31_risk__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85 |
| family | xmeta_risk |
| delta | -0.0018206636174889232 |
| wins_recent | 3 |
| latest_delta | -0.0020723716645272283 |
| friend_corr | 0.9996313803551532 |
| friend_std | 0.04514010386400232 |
| friend_mae | 0.03415787760187457 |
| friend_pct02 | 0.611248 |
| friend_pct05 | 0.234076 |
| friend_pct10 | 0.0332 |
