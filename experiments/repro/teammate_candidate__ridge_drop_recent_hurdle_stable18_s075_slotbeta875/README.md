# Teammate candidate — ridge_drop_recent_hurdle_stable18_s075__slotbeta875

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__ridge_drop_recent_hurdle_stable18_s075_slotbeta875`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `ridge_drop_recent_hurdle_stable18_s075__slotbeta875`
- **Original source:** `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/FINAL_CANDIDATE_METRICS.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/FINAL_SELECTION.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Ridge, two-part / hurdle
- **Features:** 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** | wcv | 1.7482998991040963 |
- **Known score:** | wcv | 1.7482998991040963 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** | file | C:\Users\Dimentiy\repoVScode\Ozon-ecup\src\DL\best_bas\_best_bas_combo_10h\submissions\submission_combo10h_candidate_3_ridge_drop_recent_hurdle_stable18_s075__slotbeta875.csv |
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 2 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — ridge_drop_recent_hurdle_stable18_s075__slotbeta875

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/FINAL_CANDIDATE_METRICS.csv`

| Field | Value |
|---|---|
| name | ridge_drop_recent_hurdle_stable18_s075__slotbeta875 |
| family | ridge_subset_slotstrength |
| wcv | 1.7482998991040963 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015038034540948728 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016065329991197252 |
| worst_delta | 0.0 |
| raw_delta | -0.0022776877529314455 |
| offset_mean | -0.029386448350013524 |
| offset_std | 0.05025713006680975 |
| fold_scores | [1.7692437685920095, 1.7609381851377675, 1.7494569708869574, 1.7419438080182585] |
| fold_deltas | [0.0, -0.0017868351575593966, -0.0015327793758366237, -0.0016065329991197252] |
| notes | beta=0.875 |
| oof_table_var | 0.0030871594136354695 |
| test_table_var | 0.003732957365047147 |
| var_ratio | 1.2091884042525616 |
| friend_corr | 0.9997938138526201 |
| friend_std_dz | 0.033588089741865716 |
| friend_mean_abs_dz | 0.02531982255835442 |
| friend_pct02 | 0.49914 |
| friend_pct05 | 0.12396 |
| friend_pct10 | 0.007888 |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/FINAL_SELECTION.csv`

| Field | Value |
|---|---|
| rank | 3 |
| name | ridge_drop_recent_hurdle_stable18_s075__slotbeta875 |
| file | C:\Users\Dimentiy\repoVScode\Ozon-ecup\src\DL\best_bas\_best_bas_combo_10h\submissions\submission_combo10h_candidate_3_ridge_drop_recent_hurdle_stable18_s075__slotbeta875.csv |
| delta_table | -0.0015038034540948728 |
| latest_delta | -0.0016065329991197252 |
| wins_recent | 3 |
| family | ridge_subset_slotstrength |
| friend_corr | 0.9997938138526201 |
| friend_std_dz | 0.033588089741865716 |
| friend_pct05 | 0.12396 |
| var_ratio | 1.2091884042525616 |
