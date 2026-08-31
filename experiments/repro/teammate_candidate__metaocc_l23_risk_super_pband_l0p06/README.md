# Teammate candidate — metaocc_l23_risk__super_pband_l0p06

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__metaocc_l23_risk_super_pband_l0p06`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `metaocc_l23_risk__super_pband_l0p06`
- **Original source:** `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Unknown / not recoverable from repository history
- **Features:** occurrence features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`
- **Known score:** | wcv | 1.7480598031613326 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 2 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — metaocc_l23_risk__super_pband_l0p06

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`

| Field | Value |
|---|---|
| name | metaocc_l23_risk__super_pband_l0p06 |
| family | occurrence_meta_risk |
| wcv | 1.7480598031613326 |
| base_wcv | 1.749803702558191 |
| delta | -0.001743899396858565 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0019922465553217084 |
| worst_delta | 0.0 |
| raw_delta | -0.00217736674362774 |
| offset_mean | -0.0007735498363066863 |
| offset_std | 0.0733064759596332 |
| fold_scores | [1.7692437685920095, 1.7610804299603962, 1.7492569158026832, 1.7415580944620566] |
| fold_deltas | [0.0, -0.0016445903349306956, -0.0017328344601108547, -0.0019922465553217084] |
| notes | base=super_pband_l0p06;occ=['occ_r10_fast', 'occ_r16_bal', 'occ_r22_stable', 'occ_r14_multiscale', 'occ_r18_wide', 'occ_r24_multiscale', 'occ_r12_wide', 'occ_r20_shallow'] |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | metaocc_l23_risk__super_pband_l0p06 |
| family | occurrence_meta_risk |
| wcv | 1.7480598031613326 |
| base_wcv | 1.749803702558191 |
| delta | -0.001743899396858565 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0019922465553217084 |
| worst_delta | 0.0 |
| raw_delta | -0.00217736674362774 |
| offset_mean | -0.0007735498363066863 |
| offset_std | 0.0733064759596332 |
| fold_scores | [1.7692437685920095, 1.7610804299603962, 1.7492569158026832, 1.7415580944620566] |
| fold_deltas | [0.0, -0.0016445903349306956, -0.0017328344601108547, -0.0019922465553217084] |
| notes | base=super_pband_l0p06;occ=['occ_r10_fast', 'occ_r16_bal', 'occ_r22_stable', 'occ_r14_multiscale', 'occ_r18_wide', 'occ_r24_multiscale', 'occ_r12_wide', 'occ_r20_shallow'] |
