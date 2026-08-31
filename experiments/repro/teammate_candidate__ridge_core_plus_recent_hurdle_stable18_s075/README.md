# Teammate candidate — ridge_core_plus_recent_hurdle_stable18_s075

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__ridge_core_plus_recent_hurdle_stable18_s075`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `ridge_core_plus_recent_hurdle_stable18_s075`
- **Original source:** `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/combo_validation_extra90.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/primitive_validation_extra90.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/STABLE_BRANCH_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/combo_validation_final6h_base.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/primitive_validation_final6h_base.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/STABLE_BRANCH_VALIDATION.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/ALL_VALIDATION.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_stable18.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final_reload.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_after_recent_hurdle_stable18.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_final.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_final_reload.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Ridge, two-part / hurdle
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`
- **Known score:** | wcv | 1.7483263584949975 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 15 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — ridge_core_plus_recent_hurdle_stable18_s075

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_core_plus_recent_hurdle_stable18_s075 |
| family | ridge_subset |
| wcv | 1.7483263584949975 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014773440631935604 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0015606704479216305 |
| worst_delta | 0.0 |
| raw_delta | -0.002255247091278534 |
| offset_mean | -0.02356969379231415 |
| offset_std | 0.053731138198840504 |
| fold_scores | [1.7692437685920095, 1.760903514980463, 1.7494818035790933, 1.7419896705694566] |
| fold_deltas | [0.0, -0.0018215053148638027, -0.0015079466837006894, -0.0015606704479216305] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'recent_hurdle_stable18'] |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/combo_validation_extra90.csv`

| Field | Value |
|---|---|
| name | ridge_core_plus_recent_hurdle_stable18_s075 |
| family | ridge_subset |
| wcv | 1.7483263584949975 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014773440631935604 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0015606704479216305 |
| worst_delta | 0.0 |
| raw_delta | -0.002255247091278534 |
| offset_mean | -0.02356969379231415 |
| offset_std | 0.053731138198840504 |
| fold_scores | [1.7692437685920095, 1.760903514980463, 1.7494818035790933, 1.7419896705694566] |
| fold_deltas | [0.0, -0.0018215053148638027, -0.0015079466837006894, -0.0015606704479216305] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'recent_hurdle_stable18'] |

## Evidence row 3

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/primitive_validation_extra90.csv`

| Field | Value |
|---|---|
| name | ridge_core_plus_recent_hurdle_stable18_s075 |
| family | ridge_subset |
| wcv | 1.7483263584949975 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014773440631935604 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0015606704479216305 |
| worst_delta | 0.0 |
| raw_delta | -0.002255247091278534 |
| offset_mean | -0.02356969379231415 |
| offset_std | 0.053731138198840504 |
| fold_scores | [1.7692437685920095, 1.760903514980463, 1.7494818035790933, 1.7419896705694566] |
| fold_deltas | [0.0, -0.0018215053148638027, -0.0015079466837006894, -0.0015606704479216305] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'recent_hurdle_stable18'] |

## Evidence row 4

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/STABLE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_core_plus_recent_hurdle_stable18_s075 |
| family | ridge_subset |
| wcv | 1.7483263584949975 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014773440631935604 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0015606704479216305 |
| worst_delta | 0.0 |
| raw_delta | -0.002255247091278534 |
| offset_mean | -0.02356969379231415 |
| offset_std | 0.053731138198840504 |
| fold_scores | [1.7692437685920095, 1.760903514980463, 1.7494818035790933, 1.7419896705694566] |
| fold_deltas | [0.0, -0.0018215053148638027, -0.0015079466837006894, -0.0015606704479216305] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'recent_hurdle_stable18'] |

## Evidence row 5

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_core_plus_recent_hurdle_stable18_s075 |
| family | ridge_subset |
| wcv | 1.7483263584949975 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014773440631935604 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0015606704479216305 |
| worst_delta | 0.0 |
| raw_delta | -0.002255247091278534 |
| offset_mean | -0.02356969379231415 |
| offset_std | 0.053731138198840504 |
| fold_scores | [1.7692437685920095, 1.760903514980463, 1.7494818035790933, 1.7419896705694566] |
| fold_deltas | [0.0, -0.0018215053148638027, -0.0015079466837006894, -0.0015606704479216305] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'recent_hurdle_stable18'] |

## Evidence row 6

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/combo_validation_final6h_base.csv`

| Field | Value |
|---|---|
| name | ridge_core_plus_recent_hurdle_stable18_s075 |
| family | ridge_subset |
| wcv | 1.7483263584949975 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014773440631935604 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0015606704479216305 |
| worst_delta | 0.0 |
| raw_delta | -0.002255247091278534 |
| offset_mean | -0.02356969379231415 |
| offset_std | 0.053731138198840504 |
| fold_scores | [1.7692437685920095, 1.760903514980463, 1.7494818035790933, 1.7419896705694566] |
| fold_deltas | [0.0, -0.0018215053148638027, -0.0015079466837006894, -0.0015606704479216305] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'recent_hurdle_stable18'] |

## Evidence row 7

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/primitive_validation_final6h_base.csv`

| Field | Value |
|---|---|
| name | ridge_core_plus_recent_hurdle_stable18_s075 |
| family | ridge_subset |
| wcv | 1.7483263584949975 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014773440631935604 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0015606704479216305 |
| worst_delta | 0.0 |
| raw_delta | -0.002255247091278534 |
| offset_mean | -0.02356969379231415 |
| offset_std | 0.053731138198840504 |
| fold_scores | [1.7692437685920095, 1.760903514980463, 1.7494818035790933, 1.7419896705694566] |
| fold_deltas | [0.0, -0.0018215053148638027, -0.0015079466837006894, -0.0015606704479216305] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'recent_hurdle_stable18'] |

## Evidence row 8

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/STABLE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_core_plus_recent_hurdle_stable18_s075 |
| family | ridge_subset |
| wcv | 1.7483263584949975 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014773440631935604 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0015606704479216305 |
| worst_delta | 0.0 |
| raw_delta | -0.002255247091278534 |
| offset_mean | -0.02356969379231415 |
| offset_std | 0.053731138198840504 |
| fold_scores | [1.7692437685920095, 1.760903514980463, 1.7494818035790933, 1.7419896705694566] |
| fold_deltas | [0.0, -0.0018215053148638027, -0.0015079466837006894, -0.0015606704479216305] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'recent_hurdle_stable18'] |

## Evidence row 9

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/ALL_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_core_plus_recent_hurdle_stable18_s075 |
| family | ridge_subset |
| wcv | 1.7483263584949975 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014773440631935604 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0015606704479216305 |
| worst_delta | 0.0 |
| raw_delta | -0.002255247091278534 |
| offset_mean | -0.02356969379231415 |
| offset_std | 0.053731138198840504 |
| fold_scores | [1.7692437685920095, 1.760903514980463, 1.7494818035790933, 1.7419896705694566] |
| fold_deltas | [0.0, -0.0018215053148638027, -0.0015079466837006894, -0.0015606704479216305] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'recent_hurdle_stable18'] |

## Evidence row 10

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_stable18.csv`

| Field | Value |
|---|---|
| name | ridge_core_plus_recent_hurdle_stable18_s075 |
| family | ridge_subset |
| wcv | 1.7483263584949975 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014773440631935604 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0015606704479216305 |
| worst_delta | 0.0 |
| raw_delta | -0.002255247091278534 |
| offset_mean | -0.02356969379231415 |
| offset_std | 0.053731138198840504 |
| fold_scores | [1.7692437685920095, 1.760903514980463, 1.7494818035790933, 1.7419896705694566] |
| fold_deltas | [0.0, -0.0018215053148638027, -0.0015079466837006894, -0.0015606704479216305] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'recent_hurdle_stable18'] |

## Evidence row 11

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final.csv`

| Field | Value |
|---|---|
| name | ridge_core_plus_recent_hurdle_stable18_s075 |
| family | ridge_subset |
| wcv | 1.7483263584949975 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014773440631935604 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0015606704479216305 |
| worst_delta | 0.0 |
| raw_delta | -0.002255247091278534 |
| offset_mean | -0.02356969379231415 |
| offset_std | 0.053731138198840504 |
| fold_scores | [1.7692437685920095, 1.760903514980463, 1.7494818035790933, 1.7419896705694566] |
| fold_deltas | [0.0, -0.0018215053148638027, -0.0015079466837006894, -0.0015606704479216305] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'recent_hurdle_stable18'] |

## Evidence row 12

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final_reload.csv`

| Field | Value |
|---|---|
| name | ridge_core_plus_recent_hurdle_stable18_s075 |
| family | ridge_subset |
| wcv | 1.7483263584949975 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014773440631935604 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0015606704479216305 |
| worst_delta | 0.0 |
| raw_delta | -0.002255247091278534 |
| offset_mean | -0.02356969379231415 |
| offset_std | 0.053731138198840504 |
| fold_scores | [1.7692437685920095, 1.760903514980463, 1.7494818035790933, 1.7419896705694566] |
| fold_deltas | [0.0, -0.0018215053148638027, -0.0015079466837006894, -0.0015606704479216305] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'recent_hurdle_stable18'] |

## Evidence row 13

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_after_recent_hurdle_stable18.csv`

| Field | Value |
|---|---|
| name | ridge_core_plus_recent_hurdle_stable18_s075 |
| family | ridge_subset |
| wcv | 1.7483263584949975 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014773440631935604 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0015606704479216305 |
| worst_delta | 0.0 |
| raw_delta | -0.002255247091278534 |
| offset_mean | -0.02356969379231415 |
| offset_std | 0.053731138198840504 |
| fold_scores | [1.7692437685920095, 1.760903514980463, 1.7494818035790933, 1.7419896705694566] |
| fold_deltas | [0.0, -0.0018215053148638027, -0.0015079466837006894, -0.0015606704479216305] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'recent_hurdle_stable18'] |

## Evidence row 14

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_final.csv`

| Field | Value |
|---|---|
| name | ridge_core_plus_recent_hurdle_stable18_s075 |
| family | ridge_subset |
| wcv | 1.7483263584949975 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014773440631935604 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0015606704479216305 |
| worst_delta | 0.0 |
| raw_delta | -0.002255247091278534 |
| offset_mean | -0.02356969379231415 |
| offset_std | 0.053731138198840504 |
| fold_scores | [1.7692437685920095, 1.760903514980463, 1.7494818035790933, 1.7419896705694566] |
| fold_deltas | [0.0, -0.0018215053148638027, -0.0015079466837006894, -0.0015606704479216305] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'recent_hurdle_stable18'] |

## Evidence row 15

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_final_reload.csv`

| Field | Value |
|---|---|
| name | ridge_core_plus_recent_hurdle_stable18_s075 |
| family | ridge_subset |
| wcv | 1.7483263584949975 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014773440631935604 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0015606704479216305 |
| worst_delta | 0.0 |
| raw_delta | -0.002255247091278534 |
| offset_mean | -0.02356969379231415 |
| offset_std | 0.053731138198840504 |
| fold_scores | [1.7692437685920095, 1.760903514980463, 1.7494818035790933, 1.7419896705694566] |
| fold_deltas | [0.0, -0.0018215053148638027, -0.0015079466837006894, -0.0015606704479216305] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'recent_hurdle_stable18'] |
