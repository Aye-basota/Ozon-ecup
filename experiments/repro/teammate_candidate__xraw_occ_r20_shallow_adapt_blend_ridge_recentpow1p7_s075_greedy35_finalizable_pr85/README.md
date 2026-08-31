# Teammate candidate — xraw_occ_r20_shallow_adapt__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__xraw_occ_r20_shallow_adapt_blend_ridge_recentpow1p7_s075_greedy35_finalizable_pr85`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `xraw_occ_r20_shallow_adapt__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85`
- **Original source:** `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/MATERIALIZED_CANDIDATES.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Ridge, blend
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`
- **Known score:** | wcv | 1.748244896111739 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 2 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — xraw_occ_r20_shallow_adapt__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`

| Field | Value |
|---|---|
| name | xraw_occ_r20_shallow_adapt__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85 |
| family | raw_occ_extra |
| wcv | 1.748244896111739 |
| base_wcv | 1.749803702558191 |
| delta | -0.001558806446452173 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0018293433419152016 |
| worst_delta | 3.360996094081692e-05 |
| raw_delta | -0.0018864530832325718 |
| offset_mean | 0.00833248452881422 |
| offset_std | 0.07247859601961495 |
| fold_scores | [1.7692773785529503, 1.7615328582692178, 1.749390591295248, 1.741720997675463] |
| fold_deltas | [3.360996094081692e-05, -0.001192162026109056, -0.0015991589675459217, -0.0018293433419152016] |
| notes | raw=occ_r20_shallow;base=blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85 |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/MATERIALIZED_CANDIDATES.csv`

| Field | Value |
|---|---|
| name | xraw_occ_r20_shallow_adapt__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85 |
| family | raw_occ_extra |
| delta | -0.001558806446452173 |
| wins_recent | 3 |
| latest_delta | -0.0018293433419152016 |
| friend_corr | 0.99968757243108 |
| friend_std | 0.04082738011161255 |
| friend_mae | 0.030804975659365987 |
| friend_pct02 | 0.571008 |
| friend_pct05 | 0.194596 |
| friend_pct10 | 0.021772 |
