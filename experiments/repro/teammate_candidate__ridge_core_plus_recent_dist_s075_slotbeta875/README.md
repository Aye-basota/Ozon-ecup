# Teammate candidate — ridge_core_plus_recent_dist_s075__slotbeta875

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__ridge_core_plus_recent_dist_s075_slotbeta875`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `ridge_core_plus_recent_dist_s075__slotbeta875`
- **Original source:** `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/FINAL_CANDIDATE_METRICS.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/FINAL_SELECTION.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Ridge
- **Features:** 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** | wcv | 1.7483791828797743 |
- **Known score:** | wcv | 1.7483791828797743 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** | file | C:\Users\Dimentiy\repoVScode\Ozon-ecup\src\DL\best_bas\_best_bas_combo_10h\submissions\submission_combo10h_candidate_4_ridge_core_plus_recent_dist_s075__slotbeta875.csv |
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 2 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — ridge_core_plus_recent_dist_s075__slotbeta875

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/FINAL_CANDIDATE_METRICS.csv`

| Field | Value |
|---|---|
| name | ridge_core_plus_recent_dist_s075__slotbeta875 |
| family | ridge_subset_slotstrength |
| wcv | 1.7483791828797743 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014245196784167468 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0014841313416025237 |
| worst_delta | 0.0 |
| raw_delta | -0.002197433949343471 |
| offset_mean | -0.029620444108804385 |
| offset_std | 0.05012522068458307 |
| fold_scores | [1.7692437685920095, 1.7609508908238227, 1.7495031288876883, 1.7420662096757757] |
| fold_deltas | [0.0, -0.0017741294715041267, -0.00148662137510569, -0.0014841313416025237] |
| notes | beta=0.875 |
| oof_table_var | 0.0030147270374974297 |
| test_table_var | 0.003376798362528812 |
| var_ratio | 1.1201008650295394 |
| friend_corr | 0.9998132666772231 |
| friend_std_dz | 0.031938683912605896 |
| friend_mean_abs_dz | 0.02409671490166489 |
| friend_pct02 | 0.47872 |
| friend_pct05 | 0.109872 |
| friend_pct10 | 0.00592 |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/FINAL_SELECTION.csv`

| Field | Value |
|---|---|
| rank | 4 |
| name | ridge_core_plus_recent_dist_s075__slotbeta875 |
| file | C:\Users\Dimentiy\repoVScode\Ozon-ecup\src\DL\best_bas\_best_bas_combo_10h\submissions\submission_combo10h_candidate_4_ridge_core_plus_recent_dist_s075__slotbeta875.csv |
| delta_table | -0.0014245196784167468 |
| latest_delta | -0.0014841313416025237 |
| wins_recent | 3 |
| family | ridge_subset_slotstrength |
| friend_corr | 0.9998132666772231 |
| friend_std_dz | 0.031938683912605896 |
| friend_pct05 | 0.109872 |
| var_ratio | 1.1201008650295394 |
