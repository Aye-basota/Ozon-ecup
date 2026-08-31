# Teammate candidate — occ_occ_r18_wide_fixed__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__occ_occ_r18_wide_fixed_blend_ridge_recentpow1p7_s075_greedy35_finalizable_pr85`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `occ_occ_r18_wide_fixed__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85`
- **Original source:** `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Ridge, blend
- **Features:** occurrence features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`
- **Known score:** | wcv | 1.7486145932027668 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 2 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — occ_occ_r18_wide_fixed__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`

| Field | Value |
|---|---|
| name | occ_occ_r18_wide_fixed__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85 |
| family | occurrence_overlay |
| wcv | 1.7486145932027668 |
| base_wcv | 1.749803702558191 |
| delta | -0.0011891093554243484 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001250861884628307 |
| worst_delta | 0.0001382020676321538 |
| raw_delta | -0.002008865960332577 |
| offset_mean | -0.01356284656265443 |
| offset_std | 0.05070606537411624 |
| fold_scores | [1.7693819706596416, 1.7611351082136149, 1.7497927194731573, 1.74229947913275] |
| fold_deltas | [0.0001382020676321538, -0.0015899120817119972, -0.0011970307896367327, -0.001250861884628307] |
| notes | raw=occ_r18_wide;base=blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85;adaptive=False |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | occ_occ_r18_wide_fixed__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85 |
| family | occurrence_overlay |
| wcv | 1.7486145932027668 |
| base_wcv | 1.749803702558191 |
| delta | -0.0011891093554243484 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001250861884628307 |
| worst_delta | 0.0001382020676321538 |
| raw_delta | -0.002008865960332577 |
| offset_mean | -0.01356284656265443 |
| offset_std | 0.05070606537411624 |
| fold_scores | [1.7693819706596416, 1.7611351082136149, 1.7497927194731573, 1.74229947913275] |
| fold_deltas | [0.0001382020676321538, -0.0015899120817119972, -0.0011970307896367327, -0.001250861884628307] |
| notes | raw=occ_r18_wide;base=blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85;adaptive=False |
