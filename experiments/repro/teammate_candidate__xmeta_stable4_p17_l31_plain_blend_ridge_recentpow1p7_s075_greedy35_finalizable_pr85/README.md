# Teammate candidate — xmeta_stable4_p17_l31_plain__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__xmeta_stable4_p17_l31_plain_blend_ridge_recentpow1p7_s075_greedy35_finalizable_pr85`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `xmeta_stable4_p17_l31_plain__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85`
- **Original source:** `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Ridge, blend
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`
- **Known score:** | wcv | 1.7481231677525337 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 1 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — xmeta_stable4_p17_l31_plain__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`

| Field | Value |
|---|---|
| name | xmeta_stable4_p17_l31_plain__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85 |
| family | xmeta_plain |
| wcv | 1.7481231677525337 |
| base_wcv | 1.749803702558191 |
| delta | -0.0016805348056573818 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.002067736962532596 |
| worst_delta | 0.0 |
| raw_delta | -0.0013537814102557869 |
| offset_mean | 0.016324962147863366 |
| offset_std | 0.09843084978861963 |
| fold_scores | [1.7692437685920095, 1.762093449193758, 1.7491390042174284, 1.7414826040548457] |
| fold_deltas | [0.0, -0.0006315711015687508, -0.0018507460453656144, -0.002067736962532596] |
| notes | subset=['occ_r18_wide', 'occ_r20_shallow', 'occ_r22_stable', 'occ_r24_multiscale'];power=1.7;leaves=31;base=blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85 |
