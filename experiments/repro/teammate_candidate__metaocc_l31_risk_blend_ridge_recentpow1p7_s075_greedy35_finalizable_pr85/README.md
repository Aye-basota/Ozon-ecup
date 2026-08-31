# Teammate candidate — metaocc_l31_risk__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__metaocc_l31_risk_blend_ridge_recentpow1p7_s075_greedy35_finalizable_pr85`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `metaocc_l31_risk__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85`
- **Original source:** `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/FAMILY_BEST.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/FINAL_SUBMISSIONS.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/FINAL_TWO_METRICS.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Ridge, two-part / hurdle, blend
- **Features:** occurrence features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`
- **Known score:** | wcv | 1.748037158176102 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** | corr__submission_continue12h_3_diverse_class1_over_guard | 0.7683524738505613 |
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 5 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — metaocc_l31_risk__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`

| Field | Value |
|---|---|
| name | metaocc_l31_risk__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85 |
| family | occurrence_meta_risk |
| wcv | 1.748037158176102 |
| base_wcv | 1.749803702558191 |
| delta | -0.0017665443820891783 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0020268591063281605 |
| worst_delta | 0.0 |
| raw_delta | -0.0022002315115861332 |
| offset_mean | -0.0007999088121521283 |
| offset_std | 0.07330088524060718 |
| fold_scores | [1.7692437685920095, 1.7610686477818753, 1.7492471132993417, 1.74152348191105] |
| fold_deltas | [0.0, -0.0016563725134515206, -0.0017426369634523375, -0.0020268591063281605] |
| notes | base=blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85;occ=['occ_r10_fast', 'occ_r16_bal', 'occ_r22_stable', 'occ_r14_multiscale', 'occ_r18_wide', 'occ_r24_multiscale', 'occ_r12_wide', 'occ_r20_shallow'] |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/FAMILY_BEST.csv`

| Field | Value |
|---|---|
| family | occurrence_meta_risk |
| best_name | metaocc_l31_risk__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85 |
| delta | -0.0017665443820891783 |
| wins_recent | 3 |
| latest_delta | -0.0020268591063281605 |

## Evidence row 3

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/FINAL_SUBMISSIONS.csv`

| Field | Value |
|---|---|
| branch | B |
| name | metaocc_l31_risk__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85 |
| file | C:\Users\Dimentiy\repoVScode\Ozon-ecup\src\DL\best_bas\_best_bas_final6h\submissions\submission_final6h_B_metaocc_l31_risk__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85.csv |
| delta_table | -0.0017665443820891783 |
| latest_delta | -0.0020268591063281605 |
| family | occurrence_meta_risk |
| corr | 0.9996277554755861 |
| std | 0.04536330866017492 |
| mae | 0.03437858891234656 |
| pct02 | 0.613636 |
| pct05 | 0.2373 |
| pct10 | 0.034112 |

## Evidence row 4

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/FINAL_TWO_METRICS.csv`

| Field | Value |
|---|---|
| branch | B |
| name | metaocc_l31_risk__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85 |
| family | occurrence_meta_risk |
| delta_table | -0.0017665443820891783 |
| wins_recent | 3 |
| latest_delta | -0.0020268591063281605 |
| friend_corr | 0.9996277554755861 |
| friend_std | 0.04536330866017492 |
| friend_mae | 0.03437858891234656 |
| friend_pct02 | 0.613636 |
| friend_pct05 | 0.2373 |
| friend_pct10 | 0.034112 |
| corr__submission_combo10h_candidate_2_ridge_core_plus_recent_dist_s075 | 0.9999097416486097 |
| corr__submission_combo10h_candidate_3_ridge_drop_recent_hurdle_stable18_s075__slotbeta875 | 0.9999374315330708 |
| corr__submission_combo10h_candidate_4_ridge_core_plus_recent_dist_s075__slotbeta875 | 0.9999011304188253 |
| corr__submission_combo_candidate_1_ridge_drop_recent_hurdle_stable18_s075 | 0.9999480247996011 |
| corr__submission_continue12h_1_safe_ranker_safe | 0.9967071910908624 |
| corr__submission_continue12h_2_class1_class1_occ | 0.9712103498056363 |
| corr__submission_continue12h_3_diverse_class1_over_guard | 0.7683524738505613 |

## Evidence row 5

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | metaocc_l31_risk__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85 |
| family | occurrence_meta_risk |
| wcv | 1.748037158176102 |
| base_wcv | 1.749803702558191 |
| delta | -0.0017665443820891783 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0020268591063281605 |
| worst_delta | 0.0 |
| raw_delta | -0.0022002315115861332 |
| offset_mean | -0.0007999088121521283 |
| offset_std | 0.07330088524060718 |
| fold_scores | [1.7692437685920095, 1.7610686477818753, 1.7492471132993417, 1.74152348191105] |
| fold_deltas | [0.0, -0.0016563725134515206, -0.0017426369634523375, -0.0020268591063281605] |
| notes | base=blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85;occ=['occ_r10_fast', 'occ_r16_bal', 'occ_r22_stable', 'occ_r14_multiscale', 'occ_r18_wide', 'occ_r24_multiscale', 'occ_r12_wide', 'occ_r20_shallow'] |
