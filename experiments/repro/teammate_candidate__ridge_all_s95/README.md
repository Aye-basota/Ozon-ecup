# Teammate candidate — ridge_all_s95

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__ridge_all_s95`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `ridge_all_s95`
- **Original source:** `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/combo_validation_extra90.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/primitive_validation_extra90.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/STABLE_BRANCH_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/combo_validation_final6h_base.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/primitive_validation_final6h_base.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/STABLE_BRANCH_VALIDATION.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/ALL_VALIDATION.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_fast12.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_stable18.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_existing.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final_reload.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_after_recent_hurdle_fast12.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_after_recent_hurdle_stable18.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_existing.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_final.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_final_reload.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Ridge, two-part / hurdle
- **Features:** 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`
- **Known score:** | wcv | 1.7483082027845493 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 19 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — ridge_all_s95

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_all_s95 |
| family | ridge_shrink |
| wcv | 1.7483082027845493 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014954997736417396 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001641837719631667 |
| worst_delta | 0.0 |
| raw_delta | -0.0021912227605428215 |
| offset_mean | -0.011055020874916052 |
| offset_std | 0.06129997053274389 |
| fold_scores | [1.7692437685920095, 1.7609731942778746, 1.749541214559627, 1.7419085032977466] |
| fold_deltas | [0.0, -0.0017518260174522737, -0.0014485357031670532, -0.001641837719631667] |
| notes |  |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/combo_validation_extra90.csv`

| Field | Value |
|---|---|
| name | ridge_all_s95 |
| family | ridge_shrink |
| wcv | 1.7483082027845493 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014954997736417396 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001641837719631667 |
| worst_delta | 0.0 |
| raw_delta | -0.0021912227605428215 |
| offset_mean | -0.011055020874916052 |
| offset_std | 0.06129997053274389 |
| fold_scores | [1.7692437685920095, 1.7609731942778746, 1.749541214559627, 1.7419085032977466] |
| fold_deltas | [0.0, -0.0017518260174522737, -0.0014485357031670532, -0.001641837719631667] |
| notes |  |

## Evidence row 3

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/primitive_validation_extra90.csv`

| Field | Value |
|---|---|
| name | ridge_all_s95 |
| family | ridge_shrink |
| wcv | 1.7483082027845493 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014954997736417396 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001641837719631667 |
| worst_delta | 0.0 |
| raw_delta | -0.0021912227605428215 |
| offset_mean | -0.011055020874916052 |
| offset_std | 0.06129997053274389 |
| fold_scores | [1.7692437685920095, 1.7609731942778746, 1.749541214559627, 1.7419085032977466] |
| fold_deltas | [0.0, -0.0017518260174522737, -0.0014485357031670532, -0.001641837719631667] |
| notes |  |

## Evidence row 4

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/STABLE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_all_s95 |
| family | ridge_shrink |
| wcv | 1.7483082027845493 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014954997736417396 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001641837719631667 |
| worst_delta | 0.0 |
| raw_delta | -0.0021912227605428215 |
| offset_mean | -0.011055020874916052 |
| offset_std | 0.06129997053274389 |
| fold_scores | [1.7692437685920095, 1.7609731942778746, 1.749541214559627, 1.7419085032977466] |
| fold_deltas | [0.0, -0.0017518260174522737, -0.0014485357031670532, -0.001641837719631667] |
| notes |  |

## Evidence row 5

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_all_s95 |
| family | ridge_shrink |
| wcv | 1.7483082027845493 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014954997736417396 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001641837719631667 |
| worst_delta | 0.0 |
| raw_delta | -0.0021912227605428215 |
| offset_mean | -0.011055020874916052 |
| offset_std | 0.06129997053274389 |
| fold_scores | [1.7692437685920095, 1.7609731942778746, 1.749541214559627, 1.7419085032977466] |
| fold_deltas | [0.0, -0.0017518260174522737, -0.0014485357031670532, -0.001641837719631667] |
| notes |  |

## Evidence row 6

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/combo_validation_final6h_base.csv`

| Field | Value |
|---|---|
| name | ridge_all_s95 |
| family | ridge_shrink |
| wcv | 1.7483082027845493 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014954997736417396 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001641837719631667 |
| worst_delta | 0.0 |
| raw_delta | -0.0021912227605428215 |
| offset_mean | -0.011055020874916052 |
| offset_std | 0.06129997053274389 |
| fold_scores | [1.7692437685920095, 1.7609731942778746, 1.749541214559627, 1.7419085032977466] |
| fold_deltas | [0.0, -0.0017518260174522737, -0.0014485357031670532, -0.001641837719631667] |
| notes |  |

## Evidence row 7

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/primitive_validation_final6h_base.csv`

| Field | Value |
|---|---|
| name | ridge_all_s95 |
| family | ridge_shrink |
| wcv | 1.7483082027845493 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014954997736417396 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001641837719631667 |
| worst_delta | 0.0 |
| raw_delta | -0.0021912227605428215 |
| offset_mean | -0.011055020874916052 |
| offset_std | 0.06129997053274389 |
| fold_scores | [1.7692437685920095, 1.7609731942778746, 1.749541214559627, 1.7419085032977466] |
| fold_deltas | [0.0, -0.0017518260174522737, -0.0014485357031670532, -0.001641837719631667] |
| notes |  |

## Evidence row 8

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/STABLE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_all_s95 |
| family | ridge_shrink |
| wcv | 1.7483082027845493 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014954997736417396 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001641837719631667 |
| worst_delta | 0.0 |
| raw_delta | -0.0021912227605428215 |
| offset_mean | -0.011055020874916052 |
| offset_std | 0.06129997053274389 |
| fold_scores | [1.7692437685920095, 1.7609731942778746, 1.749541214559627, 1.7419085032977466] |
| fold_deltas | [0.0, -0.0017518260174522737, -0.0014485357031670532, -0.001641837719631667] |
| notes |  |

## Evidence row 9

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/ALL_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_all_s95 |
| family | ridge_shrink |
| wcv | 1.7483082027845493 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014954997736417396 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001641837719631667 |
| worst_delta | 0.0 |
| raw_delta | -0.0021912227605428215 |
| offset_mean | -0.011055020874916052 |
| offset_std | 0.06129997053274389 |
| fold_scores | [1.7692437685920095, 1.7609731942778746, 1.749541214559627, 1.7419085032977466] |
| fold_deltas | [0.0, -0.0017518260174522737, -0.0014485357031670532, -0.001641837719631667] |
| notes |  |

## Evidence row 10

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_fast12.csv`

| Field | Value |
|---|---|
| name | ridge_all_s95 |
| family | ridge_shrink |
| wcv | 1.7483078704213597 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014958321368313617 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016347490755828087 |
| worst_delta | 0.0 |
| raw_delta | -0.002191444337549046 |
| offset_mean | -0.01101031577883538 |
| offset_std | 0.061342912671433214 |
| fold_scores | [1.7692437685920095, 1.7609475569797228, 1.749538609558644, 1.7419155919417955] |
| fold_deltas | [0.0, -0.001777463315604022, -0.001451140704149978, -0.0016347490755828087] |
| notes |  |

## Evidence row 11

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_stable18.csv`

| Field | Value |
|---|---|
| name | ridge_all_s95 |
| family | ridge_shrink |
| wcv | 1.7483082027845493 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014954997736417396 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001641837719631667 |
| worst_delta | 0.0 |
| raw_delta | -0.0021912227605428215 |
| offset_mean | -0.011055020874916052 |
| offset_std | 0.06129997053274389 |
| fold_scores | [1.7692437685920095, 1.7609731942778746, 1.749541214559627, 1.7419085032977466] |
| fold_deltas | [0.0, -0.0017518260174522737, -0.0014485357031670532, -0.001641837719631667] |
| notes |  |

## Evidence row 12

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_existing.csv`

| Field | Value |
|---|---|
| name | ridge_all_s95 |
| family | ridge_shrink |
| wcv | 1.748406560834701 |
| base_wcv | 1.749803702558191 |
| delta | -0.0013971417234900584 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0014912684584631464 |
| worst_delta | 0.0 |
| raw_delta | -0.0020998317094857557 |
| offset_mean | -0.011406887675448755 |
| offset_std | 0.061074190412349585 |
| fold_scores | [1.7692437685920095, 1.7609530498924064, 1.7496189909180928, 1.7420590725589151] |
| fold_deltas | [0.0, -0.0017719704029204308, -0.0013707593447012112, -0.0014912684584631464] |
| notes |  |

## Evidence row 13

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final.csv`

| Field | Value |
|---|---|
| name | ridge_all_s95 |
| family | ridge_shrink |
| wcv | 1.7483082027845493 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014954997736417396 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001641837719631667 |
| worst_delta | 0.0 |
| raw_delta | -0.0021912227605428215 |
| offset_mean | -0.011055020874916052 |
| offset_std | 0.06129997053274389 |
| fold_scores | [1.7692437685920095, 1.7609731942778746, 1.749541214559627, 1.7419085032977466] |
| fold_deltas | [0.0, -0.0017518260174522737, -0.0014485357031670532, -0.001641837719631667] |
| notes |  |

## Evidence row 14

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final_reload.csv`

| Field | Value |
|---|---|
| name | ridge_all_s95 |
| family | ridge_shrink |
| wcv | 1.7483082027845493 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014954997736417396 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001641837719631667 |
| worst_delta | 0.0 |
| raw_delta | -0.0021912227605428215 |
| offset_mean | -0.011055020874916052 |
| offset_std | 0.06129997053274389 |
| fold_scores | [1.7692437685920095, 1.7609731942778746, 1.749541214559627, 1.7419085032977466] |
| fold_deltas | [0.0, -0.0017518260174522737, -0.0014485357031670532, -0.001641837719631667] |
| notes |  |

## Evidence row 15

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_after_recent_hurdle_fast12.csv`

| Field | Value |
|---|---|
| name | ridge_all_s95 |
| family | ridge_shrink |
| wcv | 1.7483078704213597 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014958321368313617 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016347490755828087 |
| worst_delta | 0.0 |
| raw_delta | -0.002191444337549046 |
| offset_mean | -0.01101031577883538 |
| offset_std | 0.061342912671433214 |
| fold_scores | [1.7692437685920095, 1.7609475569797228, 1.749538609558644, 1.7419155919417955] |
| fold_deltas | [0.0, -0.001777463315604022, -0.001451140704149978, -0.0016347490755828087] |
| notes |  |

## Evidence row 16

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_after_recent_hurdle_stable18.csv`

| Field | Value |
|---|---|
| name | ridge_all_s95 |
| family | ridge_shrink |
| wcv | 1.7483082027845493 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014954997736417396 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001641837719631667 |
| worst_delta | 0.0 |
| raw_delta | -0.0021912227605428215 |
| offset_mean | -0.011055020874916052 |
| offset_std | 0.06129997053274389 |
| fold_scores | [1.7692437685920095, 1.7609731942778746, 1.749541214559627, 1.7419085032977466] |
| fold_deltas | [0.0, -0.0017518260174522737, -0.0014485357031670532, -0.001641837719631667] |
| notes |  |

## Evidence row 17

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_existing.csv`

| Field | Value |
|---|---|
| name | ridge_all_s95 |
| family | ridge_shrink |
| wcv | 1.748406560834701 |
| base_wcv | 1.749803702558191 |
| delta | -0.0013971417234900584 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0014912684584631464 |
| worst_delta | 0.0 |
| raw_delta | -0.0020998317094857557 |
| offset_mean | -0.011406887675448755 |
| offset_std | 0.061074190412349585 |
| fold_scores | [1.7692437685920095, 1.7609530498924064, 1.7496189909180928, 1.7420590725589151] |
| fold_deltas | [0.0, -0.0017719704029204308, -0.0013707593447012112, -0.0014912684584631464] |
| notes |  |

## Evidence row 18

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_final.csv`

| Field | Value |
|---|---|
| name | ridge_all_s95 |
| family | ridge_shrink |
| wcv | 1.7483082027845493 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014954997736417396 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001641837719631667 |
| worst_delta | 0.0 |
| raw_delta | -0.0021912227605428215 |
| offset_mean | -0.011055020874916052 |
| offset_std | 0.06129997053274389 |
| fold_scores | [1.7692437685920095, 1.7609731942778746, 1.749541214559627, 1.7419085032977466] |
| fold_deltas | [0.0, -0.0017518260174522737, -0.0014485357031670532, -0.001641837719631667] |
| notes |  |

## Evidence row 19

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_final_reload.csv`

| Field | Value |
|---|---|
| name | ridge_all_s95 |
| family | ridge_shrink |
| wcv | 1.7483082027845493 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014954997736417396 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001641837719631667 |
| worst_delta | 0.0 |
| raw_delta | -0.0021912227605428215 |
| offset_mean | -0.011055020874916052 |
| offset_std | 0.06129997053274389 |
| fold_scores | [1.7692437685920095, 1.7609731942778746, 1.749541214559627, 1.7419085032977466] |
| fold_deltas | [0.0, -0.0017518260174522737, -0.0014485357031670532, -0.001641837719631667] |
| notes |  |
