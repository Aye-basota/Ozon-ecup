# Teammate candidate — ridge_drop_recent_hurdle_fast12_s075

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__ridge_drop_recent_hurdle_fast12_s075`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `ridge_drop_recent_hurdle_fast12_s075`
- **Original source:** `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/combo_validation_extra90.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/primitive_validation_extra90.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/STABLE_BRANCH_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/combo_validation_final6h_base.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/primitive_validation_final6h_base.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/STABLE_BRANCH_VALIDATION.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/ALL_VALIDATION.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_fast12.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_stable18.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final_reload.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_after_recent_hurdle_fast12.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_after_recent_hurdle_stable18.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_final.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_final_reload.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Ridge, two-part / hurdle
- **Features:** 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`
- **Known score:** | wcv | 1.7483107325233276 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 17 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — ridge_drop_recent_hurdle_fast12_s075

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_drop_recent_hurdle_fast12_s075 |
| family | ridge_subset |
| wcv | 1.7483107325233276 |
| base_wcv | 1.749803702558191 |
| delta | -0.00149297003486355 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0015865020312400002 |
| worst_delta | 0.0 |
| raw_delta | -0.0022713418853271606 |
| offset_mean | -0.023604629417647134 |
| offset_std | 0.053700852492723786 |
| fold_scores | [1.7692437685920095, 1.760893031663813, 1.7494801110102927, 1.7419638389861383] |
| fold_deltas | [0.0, -0.0018319886315139033, -0.0015096392525013602, -0.0015865020312400002] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_direct', 'recent_dist', 'recent_hurdle_stable18'] |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/combo_validation_extra90.csv`

| Field | Value |
|---|---|
| name | ridge_drop_recent_hurdle_fast12_s075 |
| family | ridge_subset |
| wcv | 1.7483107325233276 |
| base_wcv | 1.749803702558191 |
| delta | -0.00149297003486355 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0015865020312400002 |
| worst_delta | 0.0 |
| raw_delta | -0.0022713418853271606 |
| offset_mean | -0.023604629417647134 |
| offset_std | 0.053700852492723786 |
| fold_scores | [1.7692437685920095, 1.760893031663813, 1.7494801110102927, 1.7419638389861383] |
| fold_deltas | [0.0, -0.0018319886315139033, -0.0015096392525013602, -0.0015865020312400002] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_direct', 'recent_dist', 'recent_hurdle_stable18'] |

## Evidence row 3

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/primitive_validation_extra90.csv`

| Field | Value |
|---|---|
| name | ridge_drop_recent_hurdle_fast12_s075 |
| family | ridge_subset |
| wcv | 1.7483107325233276 |
| base_wcv | 1.749803702558191 |
| delta | -0.00149297003486355 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0015865020312400002 |
| worst_delta | 0.0 |
| raw_delta | -0.0022713418853271606 |
| offset_mean | -0.023604629417647134 |
| offset_std | 0.053700852492723786 |
| fold_scores | [1.7692437685920095, 1.760893031663813, 1.7494801110102927, 1.7419638389861383] |
| fold_deltas | [0.0, -0.0018319886315139033, -0.0015096392525013602, -0.0015865020312400002] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_direct', 'recent_dist', 'recent_hurdle_stable18'] |

## Evidence row 4

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/STABLE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_drop_recent_hurdle_fast12_s075 |
| family | ridge_subset |
| wcv | 1.7483107325233276 |
| base_wcv | 1.749803702558191 |
| delta | -0.00149297003486355 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0015865020312400002 |
| worst_delta | 0.0 |
| raw_delta | -0.0022713418853271606 |
| offset_mean | -0.023604629417647134 |
| offset_std | 0.053700852492723786 |
| fold_scores | [1.7692437685920095, 1.760893031663813, 1.7494801110102927, 1.7419638389861383] |
| fold_deltas | [0.0, -0.0018319886315139033, -0.0015096392525013602, -0.0015865020312400002] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_direct', 'recent_dist', 'recent_hurdle_stable18'] |

## Evidence row 5

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_drop_recent_hurdle_fast12_s075 |
| family | ridge_subset |
| wcv | 1.7483107325233276 |
| base_wcv | 1.749803702558191 |
| delta | -0.00149297003486355 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0015865020312400002 |
| worst_delta | 0.0 |
| raw_delta | -0.0022713418853271606 |
| offset_mean | -0.023604629417647134 |
| offset_std | 0.053700852492723786 |
| fold_scores | [1.7692437685920095, 1.760893031663813, 1.7494801110102927, 1.7419638389861383] |
| fold_deltas | [0.0, -0.0018319886315139033, -0.0015096392525013602, -0.0015865020312400002] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_direct', 'recent_dist', 'recent_hurdle_stable18'] |

## Evidence row 6

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/combo_validation_final6h_base.csv`

| Field | Value |
|---|---|
| name | ridge_drop_recent_hurdle_fast12_s075 |
| family | ridge_subset |
| wcv | 1.7483107325233276 |
| base_wcv | 1.749803702558191 |
| delta | -0.00149297003486355 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0015865020312400002 |
| worst_delta | 0.0 |
| raw_delta | -0.0022713418853271606 |
| offset_mean | -0.023604629417647134 |
| offset_std | 0.053700852492723786 |
| fold_scores | [1.7692437685920095, 1.760893031663813, 1.7494801110102927, 1.7419638389861383] |
| fold_deltas | [0.0, -0.0018319886315139033, -0.0015096392525013602, -0.0015865020312400002] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_direct', 'recent_dist', 'recent_hurdle_stable18'] |

## Evidence row 7

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/primitive_validation_final6h_base.csv`

| Field | Value |
|---|---|
| name | ridge_drop_recent_hurdle_fast12_s075 |
| family | ridge_subset |
| wcv | 1.7483107325233276 |
| base_wcv | 1.749803702558191 |
| delta | -0.00149297003486355 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0015865020312400002 |
| worst_delta | 0.0 |
| raw_delta | -0.0022713418853271606 |
| offset_mean | -0.023604629417647134 |
| offset_std | 0.053700852492723786 |
| fold_scores | [1.7692437685920095, 1.760893031663813, 1.7494801110102927, 1.7419638389861383] |
| fold_deltas | [0.0, -0.0018319886315139033, -0.0015096392525013602, -0.0015865020312400002] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_direct', 'recent_dist', 'recent_hurdle_stable18'] |

## Evidence row 8

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/STABLE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_drop_recent_hurdle_fast12_s075 |
| family | ridge_subset |
| wcv | 1.7483107325233276 |
| base_wcv | 1.749803702558191 |
| delta | -0.00149297003486355 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0015865020312400002 |
| worst_delta | 0.0 |
| raw_delta | -0.0022713418853271606 |
| offset_mean | -0.023604629417647134 |
| offset_std | 0.053700852492723786 |
| fold_scores | [1.7692437685920095, 1.760893031663813, 1.7494801110102927, 1.7419638389861383] |
| fold_deltas | [0.0, -0.0018319886315139033, -0.0015096392525013602, -0.0015865020312400002] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_direct', 'recent_dist', 'recent_hurdle_stable18'] |

## Evidence row 9

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/ALL_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_drop_recent_hurdle_fast12_s075 |
| family | ridge_subset |
| wcv | 1.7483107325233276 |
| base_wcv | 1.749803702558191 |
| delta | -0.00149297003486355 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0015865020312400002 |
| worst_delta | 0.0 |
| raw_delta | -0.0022713418853271606 |
| offset_mean | -0.023604629417647134 |
| offset_std | 0.053700852492723786 |
| fold_scores | [1.7692437685920095, 1.760893031663813, 1.7494801110102927, 1.7419638389861383] |
| fold_deltas | [0.0, -0.0018319886315139033, -0.0015096392525013602, -0.0015865020312400002] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_direct', 'recent_dist', 'recent_hurdle_stable18'] |

## Evidence row 10

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_fast12.csv`

| Field | Value |
|---|---|
| name | ridge_drop_recent_hurdle_fast12_s075 |
| family | ridge_subset |
| wcv | 1.748343696237043 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014600063211479923 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0015360090549187966 |
| worst_delta | 0.0 |
| raw_delta | -0.002239700368966139 |
| offset_mean | -0.02382887025170121 |
| offset_std | 0.053548018147517924 |
| fold_scores | [1.7692437685920095, 1.7608871682557037, 1.7495056706881382, 1.7420143319624595] |
| fold_deltas | [0.0, -0.0018378520396231668, -0.0014840795746557944, -0.0015360090549187966] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_direct', 'recent_dist'] |

## Evidence row 11

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_stable18.csv`

| Field | Value |
|---|---|
| name | ridge_drop_recent_hurdle_fast12_s075 |
| family | ridge_subset |
| wcv | 1.7483107325233276 |
| base_wcv | 1.749803702558191 |
| delta | -0.00149297003486355 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0015865020312400002 |
| worst_delta | 0.0 |
| raw_delta | -0.0022713418853271606 |
| offset_mean | -0.023604629417647134 |
| offset_std | 0.053700852492723786 |
| fold_scores | [1.7692437685920095, 1.760893031663813, 1.7494801110102927, 1.7419638389861383] |
| fold_deltas | [0.0, -0.0018319886315139033, -0.0015096392525013602, -0.0015865020312400002] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_direct', 'recent_dist', 'recent_hurdle_stable18'] |

## Evidence row 12

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final.csv`

| Field | Value |
|---|---|
| name | ridge_drop_recent_hurdle_fast12_s075 |
| family | ridge_subset |
| wcv | 1.7483107325233276 |
| base_wcv | 1.749803702558191 |
| delta | -0.00149297003486355 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0015865020312400002 |
| worst_delta | 0.0 |
| raw_delta | -0.0022713418853271606 |
| offset_mean | -0.023604629417647134 |
| offset_std | 0.053700852492723786 |
| fold_scores | [1.7692437685920095, 1.760893031663813, 1.7494801110102927, 1.7419638389861383] |
| fold_deltas | [0.0, -0.0018319886315139033, -0.0015096392525013602, -0.0015865020312400002] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_direct', 'recent_dist', 'recent_hurdle_stable18'] |

## Evidence row 13

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final_reload.csv`

| Field | Value |
|---|---|
| name | ridge_drop_recent_hurdle_fast12_s075 |
| family | ridge_subset |
| wcv | 1.7483107325233276 |
| base_wcv | 1.749803702558191 |
| delta | -0.00149297003486355 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0015865020312400002 |
| worst_delta | 0.0 |
| raw_delta | -0.0022713418853271606 |
| offset_mean | -0.023604629417647134 |
| offset_std | 0.053700852492723786 |
| fold_scores | [1.7692437685920095, 1.760893031663813, 1.7494801110102927, 1.7419638389861383] |
| fold_deltas | [0.0, -0.0018319886315139033, -0.0015096392525013602, -0.0015865020312400002] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_direct', 'recent_dist', 'recent_hurdle_stable18'] |

## Evidence row 14

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_after_recent_hurdle_fast12.csv`

| Field | Value |
|---|---|
| name | ridge_drop_recent_hurdle_fast12_s075 |
| family | ridge_subset |
| wcv | 1.748343696237043 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014600063211479923 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0015360090549187966 |
| worst_delta | 0.0 |
| raw_delta | -0.002239700368966139 |
| offset_mean | -0.02382887025170121 |
| offset_std | 0.053548018147517924 |
| fold_scores | [1.7692437685920095, 1.7608871682557037, 1.7495056706881382, 1.7420143319624595] |
| fold_deltas | [0.0, -0.0018378520396231668, -0.0014840795746557944, -0.0015360090549187966] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_direct', 'recent_dist'] |

## Evidence row 15

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_after_recent_hurdle_stable18.csv`

| Field | Value |
|---|---|
| name | ridge_drop_recent_hurdle_fast12_s075 |
| family | ridge_subset |
| wcv | 1.7483107325233276 |
| base_wcv | 1.749803702558191 |
| delta | -0.00149297003486355 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0015865020312400002 |
| worst_delta | 0.0 |
| raw_delta | -0.0022713418853271606 |
| offset_mean | -0.023604629417647134 |
| offset_std | 0.053700852492723786 |
| fold_scores | [1.7692437685920095, 1.760893031663813, 1.7494801110102927, 1.7419638389861383] |
| fold_deltas | [0.0, -0.0018319886315139033, -0.0015096392525013602, -0.0015865020312400002] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_direct', 'recent_dist', 'recent_hurdle_stable18'] |

## Evidence row 16

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_final.csv`

| Field | Value |
|---|---|
| name | ridge_drop_recent_hurdle_fast12_s075 |
| family | ridge_subset |
| wcv | 1.7483107325233276 |
| base_wcv | 1.749803702558191 |
| delta | -0.00149297003486355 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0015865020312400002 |
| worst_delta | 0.0 |
| raw_delta | -0.0022713418853271606 |
| offset_mean | -0.023604629417647134 |
| offset_std | 0.053700852492723786 |
| fold_scores | [1.7692437685920095, 1.760893031663813, 1.7494801110102927, 1.7419638389861383] |
| fold_deltas | [0.0, -0.0018319886315139033, -0.0015096392525013602, -0.0015865020312400002] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_direct', 'recent_dist', 'recent_hurdle_stable18'] |

## Evidence row 17

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_final_reload.csv`

| Field | Value |
|---|---|
| name | ridge_drop_recent_hurdle_fast12_s075 |
| family | ridge_subset |
| wcv | 1.7483107325233276 |
| base_wcv | 1.749803702558191 |
| delta | -0.00149297003486355 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0015865020312400002 |
| worst_delta | 0.0 |
| raw_delta | -0.0022713418853271606 |
| offset_mean | -0.023604629417647134 |
| offset_std | 0.053700852492723786 |
| fold_scores | [1.7692437685920095, 1.760893031663813, 1.7494801110102927, 1.7419638389861383] |
| fold_deltas | [0.0, -0.0018319886315139033, -0.0015096392525013602, -0.0015865020312400002] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_direct', 'recent_dist', 'recent_hurdle_stable18'] |
