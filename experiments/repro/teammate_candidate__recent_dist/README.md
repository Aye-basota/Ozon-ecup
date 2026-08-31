# Teammate candidate — recent_dist

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__recent_dist`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `recent_dist`
- **Original source:** `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/combo_validation_extra90.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/primitive_validation_extra90.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/STABLE_BRANCH_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/combo_validation_final6h_base.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/primitive_validation_final6h_base.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/STABLE_BRANCH_VALIDATION.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/ALL_VALIDATION.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_fast12.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_stable18.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_existing.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final_reload.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_after_recent_hurdle_fast12.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_after_recent_hurdle_stable18.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_existing.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_final.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_final_reload.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** two-part / hurdle
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`
- **Known score:** | wcv | 1.7500901905888837 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 19 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — recent_dist

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`

| Field | Value |
|---|---|
| name | recent_dist |
| family | raw_new |
| wcv | 1.7500901905888837 |
| base_wcv | 1.749803702558191 |
| delta | 0.00028648803069259246 |
| wins | 1 |
| wins_recent | 1 |
| latest_delta | -2.2109715596574375e-05 |
| worst_delta | 0.000989371281357343 |
| raw_delta | -0.00020476412946329365 |
| offset_mean | -0.05205968675663031 |
| offset_std | 0.018638026063843946 |
| fold_scores | [1.7699731159722891, 1.7637143915766842, 1.7514312773233358, 1.7435282313017817] |
| fold_deltas | [0.0007293473802796324, 0.000989371281357343, 0.0004415270605417909, -2.2109715596574375e-05] |
| notes |  |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/combo_validation_extra90.csv`

| Field | Value |
|---|---|
| name | recent_dist |
| family | raw_new |
| wcv | 1.7500901905888837 |
| base_wcv | 1.749803702558191 |
| delta | 0.00028648803069259246 |
| wins | 1 |
| wins_recent | 1 |
| latest_delta | -2.2109715596574375e-05 |
| worst_delta | 0.000989371281357343 |
| raw_delta | -0.00020476412946329365 |
| offset_mean | -0.05205968675663031 |
| offset_std | 0.018638026063843946 |
| fold_scores | [1.7699731159722891, 1.7637143915766842, 1.7514312773233358, 1.7435282313017817] |
| fold_deltas | [0.0007293473802796324, 0.000989371281357343, 0.0004415270605417909, -2.2109715596574375e-05] |
| notes |  |

## Evidence row 3

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/primitive_validation_extra90.csv`

| Field | Value |
|---|---|
| name | recent_dist |
| family | raw_new |
| wcv | 1.7500901905888837 |
| base_wcv | 1.749803702558191 |
| delta | 0.00028648803069259246 |
| wins | 1 |
| wins_recent | 1 |
| latest_delta | -2.2109715596574375e-05 |
| worst_delta | 0.000989371281357343 |
| raw_delta | -0.00020476412946329365 |
| offset_mean | -0.05205968675663031 |
| offset_std | 0.018638026063843946 |
| fold_scores | [1.7699731159722891, 1.7637143915766842, 1.7514312773233358, 1.7435282313017817] |
| fold_deltas | [0.0007293473802796324, 0.000989371281357343, 0.0004415270605417909, -2.2109715596574375e-05] |
| notes |  |

## Evidence row 4

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/STABLE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | recent_dist |
| family | raw_new |
| wcv | 1.7500901905888837 |
| base_wcv | 1.749803702558191 |
| delta | 0.00028648803069259246 |
| wins | 1 |
| wins_recent | 1 |
| latest_delta | -2.2109715596574375e-05 |
| worst_delta | 0.000989371281357343 |
| raw_delta | -0.00020476412946329365 |
| offset_mean | -0.05205968675663031 |
| offset_std | 0.018638026063843946 |
| fold_scores | [1.7699731159722891, 1.7637143915766842, 1.7514312773233358, 1.7435282313017817] |
| fold_deltas | [0.0007293473802796324, 0.000989371281357343, 0.0004415270605417909, -2.2109715596574375e-05] |
| notes |  |

## Evidence row 5

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`

| Field | Value |
|---|---|
| name | recent_dist |
| family | raw_new |
| wcv | 1.7500901905888837 |
| base_wcv | 1.749803702558191 |
| delta | 0.00028648803069259246 |
| wins | 1 |
| wins_recent | 1 |
| latest_delta | -2.2109715596574375e-05 |
| worst_delta | 0.000989371281357343 |
| raw_delta | -0.00020476412946329365 |
| offset_mean | -0.05205968675663031 |
| offset_std | 0.018638026063843946 |
| fold_scores | [1.7699731159722891, 1.7637143915766842, 1.7514312773233358, 1.7435282313017817] |
| fold_deltas | [0.0007293473802796324, 0.000989371281357343, 0.0004415270605417909, -2.2109715596574375e-05] |
| notes |  |

## Evidence row 6

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/combo_validation_final6h_base.csv`

| Field | Value |
|---|---|
| name | recent_dist |
| family | raw_new |
| wcv | 1.7500901905888837 |
| base_wcv | 1.749803702558191 |
| delta | 0.00028648803069259246 |
| wins | 1 |
| wins_recent | 1 |
| latest_delta | -2.2109715596574375e-05 |
| worst_delta | 0.000989371281357343 |
| raw_delta | -0.00020476412946329365 |
| offset_mean | -0.05205968675663031 |
| offset_std | 0.018638026063843946 |
| fold_scores | [1.7699731159722891, 1.7637143915766842, 1.7514312773233358, 1.7435282313017817] |
| fold_deltas | [0.0007293473802796324, 0.000989371281357343, 0.0004415270605417909, -2.2109715596574375e-05] |
| notes |  |

## Evidence row 7

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/primitive_validation_final6h_base.csv`

| Field | Value |
|---|---|
| name | recent_dist |
| family | raw_new |
| wcv | 1.7500901905888837 |
| base_wcv | 1.749803702558191 |
| delta | 0.00028648803069259246 |
| wins | 1 |
| wins_recent | 1 |
| latest_delta | -2.2109715596574375e-05 |
| worst_delta | 0.000989371281357343 |
| raw_delta | -0.00020476412946329365 |
| offset_mean | -0.05205968675663031 |
| offset_std | 0.018638026063843946 |
| fold_scores | [1.7699731159722891, 1.7637143915766842, 1.7514312773233358, 1.7435282313017817] |
| fold_deltas | [0.0007293473802796324, 0.000989371281357343, 0.0004415270605417909, -2.2109715596574375e-05] |
| notes |  |

## Evidence row 8

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/STABLE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | recent_dist |
| family | raw_new |
| wcv | 1.7500901905888837 |
| base_wcv | 1.749803702558191 |
| delta | 0.00028648803069259246 |
| wins | 1 |
| wins_recent | 1 |
| latest_delta | -2.2109715596574375e-05 |
| worst_delta | 0.000989371281357343 |
| raw_delta | -0.00020476412946329365 |
| offset_mean | -0.05205968675663031 |
| offset_std | 0.018638026063843946 |
| fold_scores | [1.7699731159722891, 1.7637143915766842, 1.7514312773233358, 1.7435282313017817] |
| fold_deltas | [0.0007293473802796324, 0.000989371281357343, 0.0004415270605417909, -2.2109715596574375e-05] |
| notes |  |

## Evidence row 9

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/ALL_VALIDATION.csv`

| Field | Value |
|---|---|
| name | recent_dist |
| family | raw_new |
| wcv | 1.7500901905888837 |
| base_wcv | 1.749803702558191 |
| delta | 0.00028648803069259246 |
| wins | 1 |
| wins_recent | 1 |
| latest_delta | -2.2109715596574375e-05 |
| worst_delta | 0.000989371281357343 |
| raw_delta | -0.00020476412946329365 |
| offset_mean | -0.05205968675663031 |
| offset_std | 0.018638026063843946 |
| fold_scores | [1.7699731159722891, 1.7637143915766842, 1.7514312773233358, 1.7435282313017817] |
| fold_deltas | [0.0007293473802796324, 0.000989371281357343, 0.0004415270605417909, -2.2109715596574375e-05] |
| notes |  |

## Evidence row 10

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_fast12.csv`

| Field | Value |
|---|---|
| name | recent_dist |
| family | raw_new |
| wcv | 1.7500901905888837 |
| base_wcv | 1.749803702558191 |
| delta | 0.00028648803069259246 |
| wins | 1 |
| wins_recent | 1 |
| latest_delta | -2.2109715596574375e-05 |
| worst_delta | 0.000989371281357343 |
| raw_delta | -0.00020476412946329365 |
| offset_mean | -0.05205968675663031 |
| offset_std | 0.018638026063843946 |
| fold_scores | [1.7699731159722891, 1.7637143915766842, 1.7514312773233358, 1.7435282313017817] |
| fold_deltas | [0.0007293473802796324, 0.000989371281357343, 0.0004415270605417909, -2.2109715596574375e-05] |
| notes |  |

## Evidence row 11

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_stable18.csv`

| Field | Value |
|---|---|
| name | recent_dist |
| family | raw_new |
| wcv | 1.7500901905888837 |
| base_wcv | 1.749803702558191 |
| delta | 0.00028648803069259246 |
| wins | 1 |
| wins_recent | 1 |
| latest_delta | -2.2109715596574375e-05 |
| worst_delta | 0.000989371281357343 |
| raw_delta | -0.00020476412946329365 |
| offset_mean | -0.05205968675663031 |
| offset_std | 0.018638026063843946 |
| fold_scores | [1.7699731159722891, 1.7637143915766842, 1.7514312773233358, 1.7435282313017817] |
| fold_deltas | [0.0007293473802796324, 0.000989371281357343, 0.0004415270605417909, -2.2109715596574375e-05] |
| notes |  |

## Evidence row 12

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_existing.csv`

| Field | Value |
|---|---|
| name | recent_dist |
| family | raw_new |
| wcv | 1.7500901905888837 |
| base_wcv | 1.749803702558191 |
| delta | 0.00028648803069259246 |
| wins | 1 |
| wins_recent | 1 |
| latest_delta | -2.2109715596574375e-05 |
| worst_delta | 0.000989371281357343 |
| raw_delta | -0.00020476412946329365 |
| offset_mean | -0.05205968675663031 |
| offset_std | 0.018638026063843946 |
| fold_scores | [1.7699731159722891, 1.7637143915766842, 1.7514312773233358, 1.7435282313017817] |
| fold_deltas | [0.0007293473802796324, 0.000989371281357343, 0.0004415270605417909, -2.2109715596574375e-05] |
| notes |  |

## Evidence row 13

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final.csv`

| Field | Value |
|---|---|
| name | recent_dist |
| family | raw_new |
| wcv | 1.7500901905888837 |
| base_wcv | 1.749803702558191 |
| delta | 0.00028648803069259246 |
| wins | 1 |
| wins_recent | 1 |
| latest_delta | -2.2109715596574375e-05 |
| worst_delta | 0.000989371281357343 |
| raw_delta | -0.00020476412946329365 |
| offset_mean | -0.05205968675663031 |
| offset_std | 0.018638026063843946 |
| fold_scores | [1.7699731159722891, 1.7637143915766842, 1.7514312773233358, 1.7435282313017817] |
| fold_deltas | [0.0007293473802796324, 0.000989371281357343, 0.0004415270605417909, -2.2109715596574375e-05] |
| notes |  |

## Evidence row 14

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final_reload.csv`

| Field | Value |
|---|---|
| name | recent_dist |
| family | raw_new |
| wcv | 1.7500901905888837 |
| base_wcv | 1.749803702558191 |
| delta | 0.00028648803069259246 |
| wins | 1 |
| wins_recent | 1 |
| latest_delta | -2.2109715596574375e-05 |
| worst_delta | 0.000989371281357343 |
| raw_delta | -0.00020476412946329365 |
| offset_mean | -0.05205968675663031 |
| offset_std | 0.018638026063843946 |
| fold_scores | [1.7699731159722891, 1.7637143915766842, 1.7514312773233358, 1.7435282313017817] |
| fold_deltas | [0.0007293473802796324, 0.000989371281357343, 0.0004415270605417909, -2.2109715596574375e-05] |
| notes |  |

## Evidence row 15

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_after_recent_hurdle_fast12.csv`

| Field | Value |
|---|---|
| name | recent_dist |
| family | raw_new |
| wcv | 1.7500901905888837 |
| base_wcv | 1.749803702558191 |
| delta | 0.00028648803069259246 |
| wins | 1 |
| wins_recent | 1 |
| latest_delta | -2.2109715596574375e-05 |
| worst_delta | 0.000989371281357343 |
| raw_delta | -0.00020476412946329365 |
| offset_mean | -0.05205968675663031 |
| offset_std | 0.018638026063843946 |
| fold_scores | [1.7699731159722891, 1.7637143915766842, 1.7514312773233358, 1.7435282313017817] |
| fold_deltas | [0.0007293473802796324, 0.000989371281357343, 0.0004415270605417909, -2.2109715596574375e-05] |
| notes |  |

## Evidence row 16

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_after_recent_hurdle_stable18.csv`

| Field | Value |
|---|---|
| name | recent_dist |
| family | raw_new |
| wcv | 1.7500901905888837 |
| base_wcv | 1.749803702558191 |
| delta | 0.00028648803069259246 |
| wins | 1 |
| wins_recent | 1 |
| latest_delta | -2.2109715596574375e-05 |
| worst_delta | 0.000989371281357343 |
| raw_delta | -0.00020476412946329365 |
| offset_mean | -0.05205968675663031 |
| offset_std | 0.018638026063843946 |
| fold_scores | [1.7699731159722891, 1.7637143915766842, 1.7514312773233358, 1.7435282313017817] |
| fold_deltas | [0.0007293473802796324, 0.000989371281357343, 0.0004415270605417909, -2.2109715596574375e-05] |
| notes |  |

## Evidence row 17

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_existing.csv`

| Field | Value |
|---|---|
| name | recent_dist |
| family | raw_new |
| wcv | 1.7500901905888837 |
| base_wcv | 1.749803702558191 |
| delta | 0.00028648803069259246 |
| wins | 1 |
| wins_recent | 1 |
| latest_delta | -2.2109715596574375e-05 |
| worst_delta | 0.000989371281357343 |
| raw_delta | -0.00020476412946329365 |
| offset_mean | -0.05205968675663031 |
| offset_std | 0.018638026063843946 |
| fold_scores | [1.7699731159722891, 1.7637143915766842, 1.7514312773233358, 1.7435282313017817] |
| fold_deltas | [0.0007293473802796324, 0.000989371281357343, 0.0004415270605417909, -2.2109715596574375e-05] |
| notes |  |

## Evidence row 18

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_final.csv`

| Field | Value |
|---|---|
| name | recent_dist |
| family | raw_new |
| wcv | 1.7500901905888837 |
| base_wcv | 1.749803702558191 |
| delta | 0.00028648803069259246 |
| wins | 1 |
| wins_recent | 1 |
| latest_delta | -2.2109715596574375e-05 |
| worst_delta | 0.000989371281357343 |
| raw_delta | -0.00020476412946329365 |
| offset_mean | -0.05205968675663031 |
| offset_std | 0.018638026063843946 |
| fold_scores | [1.7699731159722891, 1.7637143915766842, 1.7514312773233358, 1.7435282313017817] |
| fold_deltas | [0.0007293473802796324, 0.000989371281357343, 0.0004415270605417909, -2.2109715596574375e-05] |
| notes |  |

## Evidence row 19

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_final_reload.csv`

| Field | Value |
|---|---|
| name | recent_dist |
| family | raw_new |
| wcv | 1.7500901905888837 |
| base_wcv | 1.749803702558191 |
| delta | 0.00028648803069259246 |
| wins | 1 |
| wins_recent | 1 |
| latest_delta | -2.2109715596574375e-05 |
| worst_delta | 0.000989371281357343 |
| raw_delta | -0.00020476412946329365 |
| offset_mean | -0.05205968675663031 |
| offset_std | 0.018638026063843946 |
| fold_scores | [1.7699731159722891, 1.7637143915766842, 1.7514312773233358, 1.7435282313017817] |
| fold_deltas | [0.0007293473802796324, 0.000989371281357343, 0.0004415270605417909, -2.2109715596574375e-05] |
| notes |  |
