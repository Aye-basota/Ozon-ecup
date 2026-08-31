# Teammate candidate — ridge_all_s55

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__ridge_all_s55`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `ridge_all_s55`
- **Original source:** `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/combo_validation_extra90.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/primitive_validation_extra90.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/STABLE_BRANCH_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/combo_validation_final6h_base.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/primitive_validation_final6h_base.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/STABLE_BRANCH_VALIDATION.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/ALL_VALIDATION.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_fast12.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_stable18.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_existing.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final_reload.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_after_recent_hurdle_fast12.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_after_recent_hurdle_stable18.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_existing.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_final.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_final_reload.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Ridge, two-part / hurdle
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`
- **Known score:** | wcv | 1.7483984994712467 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 19 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — ridge_all_s55

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_all_s55 |
| family | ridge_shrink |
| wcv | 1.7483984994712467 |
| base_wcv | 1.749803702558191 |
| delta | -0.001405203086944192 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0015007201329855224 |
| worst_delta | 0.0 |
| raw_delta | -0.0021427894393949505 |
| offset_mean | -0.03606842919655294 |
| offset_std | 0.046302165933491436 |
| fold_scores | [1.7692437685920095, 1.761076251973993, 1.7495460631133912, 1.7420496208843927] |
| fold_deltas | [0.0, -0.00164876832133376, -0.0014436871494027947, -0.0015007201329855224] |
| notes |  |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/combo_validation_extra90.csv`

| Field | Value |
|---|---|
| name | ridge_all_s55 |
| family | ridge_shrink |
| wcv | 1.7483984994712467 |
| base_wcv | 1.749803702558191 |
| delta | -0.001405203086944192 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0015007201329855224 |
| worst_delta | 0.0 |
| raw_delta | -0.0021427894393949505 |
| offset_mean | -0.03606842919655294 |
| offset_std | 0.046302165933491436 |
| fold_scores | [1.7692437685920095, 1.761076251973993, 1.7495460631133912, 1.7420496208843927] |
| fold_deltas | [0.0, -0.00164876832133376, -0.0014436871494027947, -0.0015007201329855224] |
| notes |  |

## Evidence row 3

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/primitive_validation_extra90.csv`

| Field | Value |
|---|---|
| name | ridge_all_s55 |
| family | ridge_shrink |
| wcv | 1.7483984994712467 |
| base_wcv | 1.749803702558191 |
| delta | -0.001405203086944192 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0015007201329855224 |
| worst_delta | 0.0 |
| raw_delta | -0.0021427894393949505 |
| offset_mean | -0.03606842919655294 |
| offset_std | 0.046302165933491436 |
| fold_scores | [1.7692437685920095, 1.761076251973993, 1.7495460631133912, 1.7420496208843927] |
| fold_deltas | [0.0, -0.00164876832133376, -0.0014436871494027947, -0.0015007201329855224] |
| notes |  |

## Evidence row 4

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/STABLE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_all_s55 |
| family | ridge_shrink |
| wcv | 1.7483984994712467 |
| base_wcv | 1.749803702558191 |
| delta | -0.001405203086944192 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0015007201329855224 |
| worst_delta | 0.0 |
| raw_delta | -0.0021427894393949505 |
| offset_mean | -0.03606842919655294 |
| offset_std | 0.046302165933491436 |
| fold_scores | [1.7692437685920095, 1.761076251973993, 1.7495460631133912, 1.7420496208843927] |
| fold_deltas | [0.0, -0.00164876832133376, -0.0014436871494027947, -0.0015007201329855224] |
| notes |  |

## Evidence row 5

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_all_s55 |
| family | ridge_shrink |
| wcv | 1.7483984994712467 |
| base_wcv | 1.749803702558191 |
| delta | -0.001405203086944192 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0015007201329855224 |
| worst_delta | 0.0 |
| raw_delta | -0.0021427894393949505 |
| offset_mean | -0.03606842919655294 |
| offset_std | 0.046302165933491436 |
| fold_scores | [1.7692437685920095, 1.761076251973993, 1.7495460631133912, 1.7420496208843927] |
| fold_deltas | [0.0, -0.00164876832133376, -0.0014436871494027947, -0.0015007201329855224] |
| notes |  |

## Evidence row 6

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/combo_validation_final6h_base.csv`

| Field | Value |
|---|---|
| name | ridge_all_s55 |
| family | ridge_shrink |
| wcv | 1.7483984994712467 |
| base_wcv | 1.749803702558191 |
| delta | -0.001405203086944192 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0015007201329855224 |
| worst_delta | 0.0 |
| raw_delta | -0.0021427894393949505 |
| offset_mean | -0.03606842919655294 |
| offset_std | 0.046302165933491436 |
| fold_scores | [1.7692437685920095, 1.761076251973993, 1.7495460631133912, 1.7420496208843927] |
| fold_deltas | [0.0, -0.00164876832133376, -0.0014436871494027947, -0.0015007201329855224] |
| notes |  |

## Evidence row 7

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/primitive_validation_final6h_base.csv`

| Field | Value |
|---|---|
| name | ridge_all_s55 |
| family | ridge_shrink |
| wcv | 1.7483984994712467 |
| base_wcv | 1.749803702558191 |
| delta | -0.001405203086944192 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0015007201329855224 |
| worst_delta | 0.0 |
| raw_delta | -0.0021427894393949505 |
| offset_mean | -0.03606842919655294 |
| offset_std | 0.046302165933491436 |
| fold_scores | [1.7692437685920095, 1.761076251973993, 1.7495460631133912, 1.7420496208843927] |
| fold_deltas | [0.0, -0.00164876832133376, -0.0014436871494027947, -0.0015007201329855224] |
| notes |  |

## Evidence row 8

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/STABLE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_all_s55 |
| family | ridge_shrink |
| wcv | 1.7483984994712467 |
| base_wcv | 1.749803702558191 |
| delta | -0.001405203086944192 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0015007201329855224 |
| worst_delta | 0.0 |
| raw_delta | -0.0021427894393949505 |
| offset_mean | -0.03606842919655294 |
| offset_std | 0.046302165933491436 |
| fold_scores | [1.7692437685920095, 1.761076251973993, 1.7495460631133912, 1.7420496208843927] |
| fold_deltas | [0.0, -0.00164876832133376, -0.0014436871494027947, -0.0015007201329855224] |
| notes |  |

## Evidence row 9

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/ALL_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_all_s55 |
| family | ridge_shrink |
| wcv | 1.7483984994712467 |
| base_wcv | 1.749803702558191 |
| delta | -0.001405203086944192 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0015007201329855224 |
| worst_delta | 0.0 |
| raw_delta | -0.0021427894393949505 |
| offset_mean | -0.03606842919655294 |
| offset_std | 0.046302165933491436 |
| fold_scores | [1.7692437685920095, 1.761076251973993, 1.7495460631133912, 1.7420496208843927] |
| fold_deltas | [0.0, -0.00164876832133376, -0.0014436871494027947, -0.0015007201329855224] |
| notes |  |

## Evidence row 10

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_fast12.csv`

| Field | Value |
|---|---|
| name | ridge_all_s55 |
| family | ridge_shrink |
| wcv | 1.7484007122457839 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014029903124071967 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001491243900914263 |
| worst_delta | 0.0 |
| raw_delta | -0.0021404779765411774 |
| offset_mean | -0.03604012000166664 |
| offset_std | 0.04631346453702288 |
| fold_scores | [1.7692437685920095, 1.7610580507960878, 1.749544509142715, 1.742059097116464] |
| fold_deltas | [0.0, -0.0016669694992390571, -0.0014452411200789328, -0.001491243900914263] |
| notes |  |

## Evidence row 11

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_stable18.csv`

| Field | Value |
|---|---|
| name | ridge_all_s55 |
| family | ridge_shrink |
| wcv | 1.7483984994712467 |
| base_wcv | 1.749803702558191 |
| delta | -0.001405203086944192 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0015007201329855224 |
| worst_delta | 0.0 |
| raw_delta | -0.0021427894393949505 |
| offset_mean | -0.03606842919655294 |
| offset_std | 0.046302165933491436 |
| fold_scores | [1.7692437685920095, 1.761076251973993, 1.7495460631133912, 1.7420496208843927] |
| fold_deltas | [0.0, -0.00164876832133376, -0.0014436871494027947, -0.0015007201329855224] |
| notes |  |

## Evidence row 12

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_existing.csv`

| Field | Value |
|---|---|
| name | ridge_all_s55 |
| family | ridge_shrink |
| wcv | 1.7484709782269368 |
| base_wcv | 1.749803702558191 |
| delta | -0.001332724331254198 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0013898501851568046 |
| worst_delta | 0.0 |
| raw_delta | -0.002069491730613541 |
| offset_mean | -0.03626787115680739 |
| offset_std | 0.046139754824536736 |
| fold_scores | [1.7692437685920095, 1.7610554040910826, 1.7496065424930265, 1.7421604908322215] |
| fold_deltas | [0.0, -0.0016696162042442175, -0.001383207769767525, -0.0013898501851568046] |
| notes |  |

## Evidence row 13

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final.csv`

| Field | Value |
|---|---|
| name | ridge_all_s55 |
| family | ridge_shrink |
| wcv | 1.7483984994712467 |
| base_wcv | 1.749803702558191 |
| delta | -0.001405203086944192 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0015007201329855224 |
| worst_delta | 0.0 |
| raw_delta | -0.0021427894393949505 |
| offset_mean | -0.03606842919655294 |
| offset_std | 0.046302165933491436 |
| fold_scores | [1.7692437685920095, 1.761076251973993, 1.7495460631133912, 1.7420496208843927] |
| fold_deltas | [0.0, -0.00164876832133376, -0.0014436871494027947, -0.0015007201329855224] |
| notes |  |

## Evidence row 14

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final_reload.csv`

| Field | Value |
|---|---|
| name | ridge_all_s55 |
| family | ridge_shrink |
| wcv | 1.7483984994712467 |
| base_wcv | 1.749803702558191 |
| delta | -0.001405203086944192 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0015007201329855224 |
| worst_delta | 0.0 |
| raw_delta | -0.0021427894393949505 |
| offset_mean | -0.03606842919655294 |
| offset_std | 0.046302165933491436 |
| fold_scores | [1.7692437685920095, 1.761076251973993, 1.7495460631133912, 1.7420496208843927] |
| fold_deltas | [0.0, -0.00164876832133376, -0.0014436871494027947, -0.0015007201329855224] |
| notes |  |

## Evidence row 15

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_after_recent_hurdle_fast12.csv`

| Field | Value |
|---|---|
| name | ridge_all_s55 |
| family | ridge_shrink |
| wcv | 1.7484007122457839 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014029903124071967 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001491243900914263 |
| worst_delta | 0.0 |
| raw_delta | -0.0021404779765411774 |
| offset_mean | -0.03604012000166664 |
| offset_std | 0.04631346453702288 |
| fold_scores | [1.7692437685920095, 1.7610580507960878, 1.749544509142715, 1.742059097116464] |
| fold_deltas | [0.0, -0.0016669694992390571, -0.0014452411200789328, -0.001491243900914263] |
| notes |  |

## Evidence row 16

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_after_recent_hurdle_stable18.csv`

| Field | Value |
|---|---|
| name | ridge_all_s55 |
| family | ridge_shrink |
| wcv | 1.7483984994712467 |
| base_wcv | 1.749803702558191 |
| delta | -0.001405203086944192 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0015007201329855224 |
| worst_delta | 0.0 |
| raw_delta | -0.0021427894393949505 |
| offset_mean | -0.03606842919655294 |
| offset_std | 0.046302165933491436 |
| fold_scores | [1.7692437685920095, 1.761076251973993, 1.7495460631133912, 1.7420496208843927] |
| fold_deltas | [0.0, -0.00164876832133376, -0.0014436871494027947, -0.0015007201329855224] |
| notes |  |

## Evidence row 17

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_existing.csv`

| Field | Value |
|---|---|
| name | ridge_all_s55 |
| family | ridge_shrink |
| wcv | 1.7484709782269368 |
| base_wcv | 1.749803702558191 |
| delta | -0.001332724331254198 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0013898501851568046 |
| worst_delta | 0.0 |
| raw_delta | -0.002069491730613541 |
| offset_mean | -0.03626787115680739 |
| offset_std | 0.046139754824536736 |
| fold_scores | [1.7692437685920095, 1.7610554040910826, 1.7496065424930265, 1.7421604908322215] |
| fold_deltas | [0.0, -0.0016696162042442175, -0.001383207769767525, -0.0013898501851568046] |
| notes |  |

## Evidence row 18

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_final.csv`

| Field | Value |
|---|---|
| name | ridge_all_s55 |
| family | ridge_shrink |
| wcv | 1.7483984994712467 |
| base_wcv | 1.749803702558191 |
| delta | -0.001405203086944192 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0015007201329855224 |
| worst_delta | 0.0 |
| raw_delta | -0.0021427894393949505 |
| offset_mean | -0.03606842919655294 |
| offset_std | 0.046302165933491436 |
| fold_scores | [1.7692437685920095, 1.761076251973993, 1.7495460631133912, 1.7420496208843927] |
| fold_deltas | [0.0, -0.00164876832133376, -0.0014436871494027947, -0.0015007201329855224] |
| notes |  |

## Evidence row 19

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_final_reload.csv`

| Field | Value |
|---|---|
| name | ridge_all_s55 |
| family | ridge_shrink |
| wcv | 1.7483984994712467 |
| base_wcv | 1.749803702558191 |
| delta | -0.001405203086944192 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0015007201329855224 |
| worst_delta | 0.0 |
| raw_delta | -0.0021427894393949505 |
| offset_mean | -0.03606842919655294 |
| offset_std | 0.046302165933491436 |
| fold_scores | [1.7692437685920095, 1.761076251973993, 1.7495460631133912, 1.7420496208843927] |
| fold_deltas | [0.0, -0.00164876832133376, -0.0014436871494027947, -0.0015007201329855224] |
| notes |  |
