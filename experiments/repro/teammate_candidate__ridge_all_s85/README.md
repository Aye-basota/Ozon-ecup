# Teammate candidate — ridge_all_s85

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__ridge_all_s85`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `ridge_all_s85`
- **Original source:** `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/combo_validation_extra90.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/primitive_validation_extra90.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/STABLE_BRANCH_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/combo_validation_final6h_base.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/primitive_validation_final6h_base.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/STABLE_BRANCH_VALIDATION.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/ALL_VALIDATION.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_fast12.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_stable18.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_existing.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final_reload.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_after_recent_hurdle_fast12.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_after_recent_hurdle_stable18.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_existing.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_final.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_final_reload.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Ridge, two-part / hurdle
- **Features:** 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`
- **Known score:** | wcv | 1.74825734286749 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 19 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — ridge_all_s85

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_all_s85 |
| family | ridge_shrink |
| wcv | 1.74825734286749 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015463596907010788 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016813882336310648 |
| worst_delta | 0.0 |
| raw_delta | -0.0022981856583534762 |
| offset_mean | -0.01730073328445179 |
| offset_std | 0.05751177038229481 |
| fold_scores | [1.7692437685920095, 1.760912455905426, 1.7494599600848775, 1.7418689527837472] |
| fold_deltas | [0.0, -0.0018125643899007393, -0.0015297901779165457, -0.0016813882336310648] |
| notes |  |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/combo_validation_extra90.csv`

| Field | Value |
|---|---|
| name | ridge_all_s85 |
| family | ridge_shrink |
| wcv | 1.74825734286749 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015463596907010788 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016813882336310648 |
| worst_delta | 0.0 |
| raw_delta | -0.0022981856583534762 |
| offset_mean | -0.01730073328445179 |
| offset_std | 0.05751177038229481 |
| fold_scores | [1.7692437685920095, 1.760912455905426, 1.7494599600848775, 1.7418689527837472] |
| fold_deltas | [0.0, -0.0018125643899007393, -0.0015297901779165457, -0.0016813882336310648] |
| notes |  |

## Evidence row 3

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/primitive_validation_extra90.csv`

| Field | Value |
|---|---|
| name | ridge_all_s85 |
| family | ridge_shrink |
| wcv | 1.74825734286749 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015463596907010788 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016813882336310648 |
| worst_delta | 0.0 |
| raw_delta | -0.0022981856583534762 |
| offset_mean | -0.01730073328445179 |
| offset_std | 0.05751177038229481 |
| fold_scores | [1.7692437685920095, 1.760912455905426, 1.7494599600848775, 1.7418689527837472] |
| fold_deltas | [0.0, -0.0018125643899007393, -0.0015297901779165457, -0.0016813882336310648] |
| notes |  |

## Evidence row 4

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/STABLE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_all_s85 |
| family | ridge_shrink |
| wcv | 1.74825734286749 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015463596907010788 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016813882336310648 |
| worst_delta | 0.0 |
| raw_delta | -0.0022981856583534762 |
| offset_mean | -0.01730073328445179 |
| offset_std | 0.05751177038229481 |
| fold_scores | [1.7692437685920095, 1.760912455905426, 1.7494599600848775, 1.7418689527837472] |
| fold_deltas | [0.0, -0.0018125643899007393, -0.0015297901779165457, -0.0016813882336310648] |
| notes |  |

## Evidence row 5

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_all_s85 |
| family | ridge_shrink |
| wcv | 1.74825734286749 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015463596907010788 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016813882336310648 |
| worst_delta | 0.0 |
| raw_delta | -0.0022981856583534762 |
| offset_mean | -0.01730073328445179 |
| offset_std | 0.05751177038229481 |
| fold_scores | [1.7692437685920095, 1.760912455905426, 1.7494599600848775, 1.7418689527837472] |
| fold_deltas | [0.0, -0.0018125643899007393, -0.0015297901779165457, -0.0016813882336310648] |
| notes |  |

## Evidence row 6

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/combo_validation_final6h_base.csv`

| Field | Value |
|---|---|
| name | ridge_all_s85 |
| family | ridge_shrink |
| wcv | 1.74825734286749 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015463596907010788 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016813882336310648 |
| worst_delta | 0.0 |
| raw_delta | -0.0022981856583534762 |
| offset_mean | -0.01730073328445179 |
| offset_std | 0.05751177038229481 |
| fold_scores | [1.7692437685920095, 1.760912455905426, 1.7494599600848775, 1.7418689527837472] |
| fold_deltas | [0.0, -0.0018125643899007393, -0.0015297901779165457, -0.0016813882336310648] |
| notes |  |

## Evidence row 7

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/primitive_validation_final6h_base.csv`

| Field | Value |
|---|---|
| name | ridge_all_s85 |
| family | ridge_shrink |
| wcv | 1.74825734286749 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015463596907010788 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016813882336310648 |
| worst_delta | 0.0 |
| raw_delta | -0.0022981856583534762 |
| offset_mean | -0.01730073328445179 |
| offset_std | 0.05751177038229481 |
| fold_scores | [1.7692437685920095, 1.760912455905426, 1.7494599600848775, 1.7418689527837472] |
| fold_deltas | [0.0, -0.0018125643899007393, -0.0015297901779165457, -0.0016813882336310648] |
| notes |  |

## Evidence row 8

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/STABLE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_all_s85 |
| family | ridge_shrink |
| wcv | 1.74825734286749 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015463596907010788 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016813882336310648 |
| worst_delta | 0.0 |
| raw_delta | -0.0022981856583534762 |
| offset_mean | -0.01730073328445179 |
| offset_std | 0.05751177038229481 |
| fold_scores | [1.7692437685920095, 1.760912455905426, 1.7494599600848775, 1.7418689527837472] |
| fold_deltas | [0.0, -0.0018125643899007393, -0.0015297901779165457, -0.0016813882336310648] |
| notes |  |

## Evidence row 9

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/ALL_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_all_s85 |
| family | ridge_shrink |
| wcv | 1.74825734286749 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015463596907010788 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016813882336310648 |
| worst_delta | 0.0 |
| raw_delta | -0.0022981856583534762 |
| offset_mean | -0.01730073328445179 |
| offset_std | 0.05751177038229481 |
| fold_scores | [1.7692437685920095, 1.760912455905426, 1.7494599600848775, 1.7418689527837472] |
| fold_deltas | [0.0, -0.0018125643899007393, -0.0015297901779165457, -0.0016813882336310648] |
| notes |  |

## Evidence row 10

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_fast12.csv`

| Field | Value |
|---|---|
| name | ridge_all_s85 |
| family | ridge_shrink |
| wcv | 1.7482579834667453 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015457190914457482 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001672958370842359 |
| worst_delta | 0.0 |
| raw_delta | -0.0022974354793788027 |
| offset_mean | -0.017263160101535825 |
| offset_std | 0.05754544898227012 |
| fold_scores | [1.7692437685920095, 1.760888244553126, 1.7494576082826576, 1.741877382646536] |
| fold_deltas | [0.0, -0.001836775742200869, -0.0015321419801364033, -0.001672958370842359] |
| notes |  |

## Evidence row 11

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_stable18.csv`

| Field | Value |
|---|---|
| name | ridge_all_s85 |
| family | ridge_shrink |
| wcv | 1.74825734286749 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015463596907010788 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016813882336310648 |
| worst_delta | 0.0 |
| raw_delta | -0.0022981856583534762 |
| offset_mean | -0.01730073328445179 |
| offset_std | 0.05751177038229481 |
| fold_scores | [1.7692437685920095, 1.760912455905426, 1.7494599600848775, 1.7418689527837472] |
| fold_deltas | [0.0, -0.0018125643899007393, -0.0015297901779165457, -0.0016813882336310648] |
| notes |  |

## Evidence row 12

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_existing.csv`

| Field | Value |
|---|---|
| name | ridge_all_s85 |
| family | ridge_shrink |
| wcv | 1.7483513992410444 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014523033171466378 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0015374706922577364 |
| worst_delta | 0.0 |
| raw_delta | -0.0022084325825981743 |
| offset_mean | -0.017617950193390537 |
| offset_std | 0.05729907715101637 |
| fold_scores | [1.7692437685920095, 1.7608908935514755, 1.7495356175799353, 1.7420128703251205] |
| fold_deltas | [0.0, -0.0018341267438513054, -0.0014541326828587664, -0.0015374706922577364] |
| notes |  |

## Evidence row 13

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final.csv`

| Field | Value |
|---|---|
| name | ridge_all_s85 |
| family | ridge_shrink |
| wcv | 1.74825734286749 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015463596907010788 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016813882336310648 |
| worst_delta | 0.0 |
| raw_delta | -0.0022981856583534762 |
| offset_mean | -0.01730073328445179 |
| offset_std | 0.05751177038229481 |
| fold_scores | [1.7692437685920095, 1.760912455905426, 1.7494599600848775, 1.7418689527837472] |
| fold_deltas | [0.0, -0.0018125643899007393, -0.0015297901779165457, -0.0016813882336310648] |
| notes |  |

## Evidence row 14

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final_reload.csv`

| Field | Value |
|---|---|
| name | ridge_all_s85 |
| family | ridge_shrink |
| wcv | 1.74825734286749 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015463596907010788 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016813882336310648 |
| worst_delta | 0.0 |
| raw_delta | -0.0022981856583534762 |
| offset_mean | -0.01730073328445179 |
| offset_std | 0.05751177038229481 |
| fold_scores | [1.7692437685920095, 1.760912455905426, 1.7494599600848775, 1.7418689527837472] |
| fold_deltas | [0.0, -0.0018125643899007393, -0.0015297901779165457, -0.0016813882336310648] |
| notes |  |

## Evidence row 15

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_after_recent_hurdle_fast12.csv`

| Field | Value |
|---|---|
| name | ridge_all_s85 |
| family | ridge_shrink |
| wcv | 1.7482579834667453 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015457190914457482 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001672958370842359 |
| worst_delta | 0.0 |
| raw_delta | -0.0022974354793788027 |
| offset_mean | -0.017263160101535825 |
| offset_std | 0.05754544898227012 |
| fold_scores | [1.7692437685920095, 1.760888244553126, 1.7494576082826576, 1.741877382646536] |
| fold_deltas | [0.0, -0.001836775742200869, -0.0015321419801364033, -0.001672958370842359] |
| notes |  |

## Evidence row 16

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_after_recent_hurdle_stable18.csv`

| Field | Value |
|---|---|
| name | ridge_all_s85 |
| family | ridge_shrink |
| wcv | 1.74825734286749 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015463596907010788 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016813882336310648 |
| worst_delta | 0.0 |
| raw_delta | -0.0022981856583534762 |
| offset_mean | -0.01730073328445179 |
| offset_std | 0.05751177038229481 |
| fold_scores | [1.7692437685920095, 1.760912455905426, 1.7494599600848775, 1.7418689527837472] |
| fold_deltas | [0.0, -0.0018125643899007393, -0.0015297901779165457, -0.0016813882336310648] |
| notes |  |

## Evidence row 17

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_existing.csv`

| Field | Value |
|---|---|
| name | ridge_all_s85 |
| family | ridge_shrink |
| wcv | 1.7483513992410444 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014523033171466378 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0015374706922577364 |
| worst_delta | 0.0 |
| raw_delta | -0.0022084325825981743 |
| offset_mean | -0.017617950193390537 |
| offset_std | 0.05729907715101637 |
| fold_scores | [1.7692437685920095, 1.7608908935514755, 1.7495356175799353, 1.7420128703251205] |
| fold_deltas | [0.0, -0.0018341267438513054, -0.0014541326828587664, -0.0015374706922577364] |
| notes |  |

## Evidence row 18

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_final.csv`

| Field | Value |
|---|---|
| name | ridge_all_s85 |
| family | ridge_shrink |
| wcv | 1.74825734286749 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015463596907010788 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016813882336310648 |
| worst_delta | 0.0 |
| raw_delta | -0.0022981856583534762 |
| offset_mean | -0.01730073328445179 |
| offset_std | 0.05751177038229481 |
| fold_scores | [1.7692437685920095, 1.760912455905426, 1.7494599600848775, 1.7418689527837472] |
| fold_deltas | [0.0, -0.0018125643899007393, -0.0015297901779165457, -0.0016813882336310648] |
| notes |  |

## Evidence row 19

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_final_reload.csv`

| Field | Value |
|---|---|
| name | ridge_all_s85 |
| family | ridge_shrink |
| wcv | 1.74825734286749 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015463596907010788 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016813882336310648 |
| worst_delta | 0.0 |
| raw_delta | -0.0022981856583534762 |
| offset_mean | -0.01730073328445179 |
| offset_std | 0.05751177038229481 |
| fold_scores | [1.7692437685920095, 1.760912455905426, 1.7494599600848775, 1.7418689527837472] |
| fold_deltas | [0.0, -0.0018125643899007393, -0.0015297901779165457, -0.0016813882336310648] |
| notes |  |
