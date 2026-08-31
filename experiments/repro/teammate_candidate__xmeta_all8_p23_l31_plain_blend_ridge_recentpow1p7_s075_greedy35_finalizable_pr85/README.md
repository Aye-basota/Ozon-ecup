# Teammate candidate — xmeta_all8_p23_l31_plain__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__xmeta_all8_p23_l31_plain_blend_ridge_recentpow1p7_s075_greedy35_finalizable_pr85`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `xmeta_all8_p23_l31_plain__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85`
- **Original source:** `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Ridge, blend
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`
- **Known score:** | wcv | 1.7480665224621428 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 1 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — xmeta_all8_p23_l31_plain__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`

| Field | Value |
|---|---|
| name | xmeta_all8_p23_l31_plain__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85 |
| family | xmeta_plain |
| wcv | 1.7480665224621428 |
| base_wcv | 1.749803702558191 |
| delta | -0.001737180096048204 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0021288563151089512 |
| worst_delta | 0.0 |
| raw_delta | -0.0014265832751144754 |
| offset_mean | 0.01584159943359724 |
| offset_std | 0.0980190650008408 |
| fold_scores | [1.7692437685920095, 1.7619821063791496, 1.7491044944909198, 1.7414214847022693] |
| fold_deltas | [0.0, -0.0007429139161772014, -0.0018852557718742613, -0.0021288563151089512] |
| notes | subset=['occ_r10_fast', 'occ_r16_bal', 'occ_r22_stable', 'occ_r14_multiscale', 'occ_r18_wide', 'occ_r24_multiscale', 'occ_r12_wide', 'occ_r20_shallow'];power=2.3;leaves=31;base=blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85 |
