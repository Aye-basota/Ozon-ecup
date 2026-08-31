# Teammate candidate — multiscale_direct

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__multiscale_direct`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `multiscale_direct`
- **Original source:** `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/combo_validation_extra90.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/primitive_validation_extra90.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/STABLE_BRANCH_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/combo_validation_final6h_base.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/primitive_validation_final6h_base.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/STABLE_BRANCH_VALIDATION.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/ALL_VALIDATION.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_fast12.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_stable18.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_existing.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final_reload.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_after_recent_hurdle_fast12.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_after_recent_hurdle_stable18.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_existing.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_final.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_final_reload.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** two-part / hurdle
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`
- **Known score:** | wcv | 1.760047439180695 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 19 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — multiscale_direct

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`

| Field | Value |
|---|---|
| name | multiscale_direct |
| family | raw_new |
| wcv | 1.760047439180695 |
| base_wcv | 1.749803702558191 |
| delta | 0.010243736622504033 |
| wins | 0 |
| wins_recent | 0 |
| latest_delta | 0.011281828419630546 |
| worst_delta | 0.011281828419630546 |
| raw_delta | 0.010728810844990127 |
| offset_mean | -0.06861270838302198 |
| offset_std | 0.015075426500793472 |
| fold_scores | [1.7762759032894748, 1.7711592583338476, 1.7608649530642964, 1.7548321694370088] |
| fold_deltas | [0.007032134697465331, 0.008434238038520725, 0.009875202801502336, 0.011281828419630546] |
| notes |  |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/combo_validation_extra90.csv`

| Field | Value |
|---|---|
| name | multiscale_direct |
| family | raw_new |
| wcv | 1.760047439180695 |
| base_wcv | 1.749803702558191 |
| delta | 0.010243736622504033 |
| wins | 0 |
| wins_recent | 0 |
| latest_delta | 0.011281828419630546 |
| worst_delta | 0.011281828419630546 |
| raw_delta | 0.010728810844990127 |
| offset_mean | -0.06861270838302198 |
| offset_std | 0.015075426500793472 |
| fold_scores | [1.7762759032894748, 1.7711592583338476, 1.7608649530642964, 1.7548321694370088] |
| fold_deltas | [0.007032134697465331, 0.008434238038520725, 0.009875202801502336, 0.011281828419630546] |
| notes |  |

## Evidence row 3

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/primitive_validation_extra90.csv`

| Field | Value |
|---|---|
| name | multiscale_direct |
| family | raw_new |
| wcv | 1.760047439180695 |
| base_wcv | 1.749803702558191 |
| delta | 0.010243736622504033 |
| wins | 0 |
| wins_recent | 0 |
| latest_delta | 0.011281828419630546 |
| worst_delta | 0.011281828419630546 |
| raw_delta | 0.010728810844990127 |
| offset_mean | -0.06861270838302198 |
| offset_std | 0.015075426500793472 |
| fold_scores | [1.7762759032894748, 1.7711592583338476, 1.7608649530642964, 1.7548321694370088] |
| fold_deltas | [0.007032134697465331, 0.008434238038520725, 0.009875202801502336, 0.011281828419630546] |
| notes |  |

## Evidence row 4

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/STABLE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | multiscale_direct |
| family | raw_new |
| wcv | 1.760047439180695 |
| base_wcv | 1.749803702558191 |
| delta | 0.010243736622504033 |
| wins | 0 |
| wins_recent | 0 |
| latest_delta | 0.011281828419630546 |
| worst_delta | 0.011281828419630546 |
| raw_delta | 0.010728810844990127 |
| offset_mean | -0.06861270838302198 |
| offset_std | 0.015075426500793472 |
| fold_scores | [1.7762759032894748, 1.7711592583338476, 1.7608649530642964, 1.7548321694370088] |
| fold_deltas | [0.007032134697465331, 0.008434238038520725, 0.009875202801502336, 0.011281828419630546] |
| notes |  |

## Evidence row 5

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`

| Field | Value |
|---|---|
| name | multiscale_direct |
| family | raw_new |
| wcv | 1.760047439180695 |
| base_wcv | 1.749803702558191 |
| delta | 0.010243736622504033 |
| wins | 0 |
| wins_recent | 0 |
| latest_delta | 0.011281828419630546 |
| worst_delta | 0.011281828419630546 |
| raw_delta | 0.010728810844990127 |
| offset_mean | -0.06861270838302198 |
| offset_std | 0.015075426500793472 |
| fold_scores | [1.7762759032894748, 1.7711592583338476, 1.7608649530642964, 1.7548321694370088] |
| fold_deltas | [0.007032134697465331, 0.008434238038520725, 0.009875202801502336, 0.011281828419630546] |
| notes |  |

## Evidence row 6

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/combo_validation_final6h_base.csv`

| Field | Value |
|---|---|
| name | multiscale_direct |
| family | raw_new |
| wcv | 1.760047439180695 |
| base_wcv | 1.749803702558191 |
| delta | 0.010243736622504033 |
| wins | 0 |
| wins_recent | 0 |
| latest_delta | 0.011281828419630546 |
| worst_delta | 0.011281828419630546 |
| raw_delta | 0.010728810844990127 |
| offset_mean | -0.06861270838302198 |
| offset_std | 0.015075426500793472 |
| fold_scores | [1.7762759032894748, 1.7711592583338476, 1.7608649530642964, 1.7548321694370088] |
| fold_deltas | [0.007032134697465331, 0.008434238038520725, 0.009875202801502336, 0.011281828419630546] |
| notes |  |

## Evidence row 7

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/primitive_validation_final6h_base.csv`

| Field | Value |
|---|---|
| name | multiscale_direct |
| family | raw_new |
| wcv | 1.760047439180695 |
| base_wcv | 1.749803702558191 |
| delta | 0.010243736622504033 |
| wins | 0 |
| wins_recent | 0 |
| latest_delta | 0.011281828419630546 |
| worst_delta | 0.011281828419630546 |
| raw_delta | 0.010728810844990127 |
| offset_mean | -0.06861270838302198 |
| offset_std | 0.015075426500793472 |
| fold_scores | [1.7762759032894748, 1.7711592583338476, 1.7608649530642964, 1.7548321694370088] |
| fold_deltas | [0.007032134697465331, 0.008434238038520725, 0.009875202801502336, 0.011281828419630546] |
| notes |  |

## Evidence row 8

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/STABLE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | multiscale_direct |
| family | raw_new |
| wcv | 1.760047439180695 |
| base_wcv | 1.749803702558191 |
| delta | 0.010243736622504033 |
| wins | 0 |
| wins_recent | 0 |
| latest_delta | 0.011281828419630546 |
| worst_delta | 0.011281828419630546 |
| raw_delta | 0.010728810844990127 |
| offset_mean | -0.06861270838302198 |
| offset_std | 0.015075426500793472 |
| fold_scores | [1.7762759032894748, 1.7711592583338476, 1.7608649530642964, 1.7548321694370088] |
| fold_deltas | [0.007032134697465331, 0.008434238038520725, 0.009875202801502336, 0.011281828419630546] |
| notes |  |

## Evidence row 9

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/ALL_VALIDATION.csv`

| Field | Value |
|---|---|
| name | multiscale_direct |
| family | raw_new |
| wcv | 1.760047439180695 |
| base_wcv | 1.749803702558191 |
| delta | 0.010243736622504033 |
| wins | 0 |
| wins_recent | 0 |
| latest_delta | 0.011281828419630546 |
| worst_delta | 0.011281828419630546 |
| raw_delta | 0.010728810844990127 |
| offset_mean | -0.06861270838302198 |
| offset_std | 0.015075426500793472 |
| fold_scores | [1.7762759032894748, 1.7711592583338476, 1.7608649530642964, 1.7548321694370088] |
| fold_deltas | [0.007032134697465331, 0.008434238038520725, 0.009875202801502336, 0.011281828419630546] |
| notes |  |

## Evidence row 10

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_fast12.csv`

| Field | Value |
|---|---|
| name | multiscale_direct |
| family | raw_new |
| wcv | 1.760047439180695 |
| base_wcv | 1.749803702558191 |
| delta | 0.010243736622504033 |
| wins | 0 |
| wins_recent | 0 |
| latest_delta | 0.011281828419630546 |
| worst_delta | 0.011281828419630546 |
| raw_delta | 0.010728810844990127 |
| offset_mean | -0.06861270838302198 |
| offset_std | 0.015075426500793472 |
| fold_scores | [1.7762759032894748, 1.7711592583338476, 1.7608649530642964, 1.7548321694370088] |
| fold_deltas | [0.007032134697465331, 0.008434238038520725, 0.009875202801502336, 0.011281828419630546] |
| notes |  |

## Evidence row 11

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_stable18.csv`

| Field | Value |
|---|---|
| name | multiscale_direct |
| family | raw_new |
| wcv | 1.760047439180695 |
| base_wcv | 1.749803702558191 |
| delta | 0.010243736622504033 |
| wins | 0 |
| wins_recent | 0 |
| latest_delta | 0.011281828419630546 |
| worst_delta | 0.011281828419630546 |
| raw_delta | 0.010728810844990127 |
| offset_mean | -0.06861270838302198 |
| offset_std | 0.015075426500793472 |
| fold_scores | [1.7762759032894748, 1.7711592583338476, 1.7608649530642964, 1.7548321694370088] |
| fold_deltas | [0.007032134697465331, 0.008434238038520725, 0.009875202801502336, 0.011281828419630546] |
| notes |  |

## Evidence row 12

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_existing.csv`

| Field | Value |
|---|---|
| name | multiscale_direct |
| family | raw_new |
| wcv | 1.760047439180695 |
| base_wcv | 1.749803702558191 |
| delta | 0.010243736622504033 |
| wins | 0 |
| wins_recent | 0 |
| latest_delta | 0.011281828419630546 |
| worst_delta | 0.011281828419630546 |
| raw_delta | 0.010728810844990127 |
| offset_mean | -0.06861270838302198 |
| offset_std | 0.015075426500793472 |
| fold_scores | [1.7762759032894748, 1.7711592583338476, 1.7608649530642964, 1.7548321694370088] |
| fold_deltas | [0.007032134697465331, 0.008434238038520725, 0.009875202801502336, 0.011281828419630546] |
| notes |  |

## Evidence row 13

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final.csv`

| Field | Value |
|---|---|
| name | multiscale_direct |
| family | raw_new |
| wcv | 1.760047439180695 |
| base_wcv | 1.749803702558191 |
| delta | 0.010243736622504033 |
| wins | 0 |
| wins_recent | 0 |
| latest_delta | 0.011281828419630546 |
| worst_delta | 0.011281828419630546 |
| raw_delta | 0.010728810844990127 |
| offset_mean | -0.06861270838302198 |
| offset_std | 0.015075426500793472 |
| fold_scores | [1.7762759032894748, 1.7711592583338476, 1.7608649530642964, 1.7548321694370088] |
| fold_deltas | [0.007032134697465331, 0.008434238038520725, 0.009875202801502336, 0.011281828419630546] |
| notes |  |

## Evidence row 14

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final_reload.csv`

| Field | Value |
|---|---|
| name | multiscale_direct |
| family | raw_new |
| wcv | 1.760047439180695 |
| base_wcv | 1.749803702558191 |
| delta | 0.010243736622504033 |
| wins | 0 |
| wins_recent | 0 |
| latest_delta | 0.011281828419630546 |
| worst_delta | 0.011281828419630546 |
| raw_delta | 0.010728810844990127 |
| offset_mean | -0.06861270838302198 |
| offset_std | 0.015075426500793472 |
| fold_scores | [1.7762759032894748, 1.7711592583338476, 1.7608649530642964, 1.7548321694370088] |
| fold_deltas | [0.007032134697465331, 0.008434238038520725, 0.009875202801502336, 0.011281828419630546] |
| notes |  |

## Evidence row 15

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_after_recent_hurdle_fast12.csv`

| Field | Value |
|---|---|
| name | multiscale_direct |
| family | raw_new |
| wcv | 1.760047439180695 |
| base_wcv | 1.749803702558191 |
| delta | 0.010243736622504033 |
| wins | 0 |
| wins_recent | 0 |
| latest_delta | 0.011281828419630546 |
| worst_delta | 0.011281828419630546 |
| raw_delta | 0.010728810844990127 |
| offset_mean | -0.06861270838302198 |
| offset_std | 0.015075426500793472 |
| fold_scores | [1.7762759032894748, 1.7711592583338476, 1.7608649530642964, 1.7548321694370088] |
| fold_deltas | [0.007032134697465331, 0.008434238038520725, 0.009875202801502336, 0.011281828419630546] |
| notes |  |

## Evidence row 16

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_after_recent_hurdle_stable18.csv`

| Field | Value |
|---|---|
| name | multiscale_direct |
| family | raw_new |
| wcv | 1.760047439180695 |
| base_wcv | 1.749803702558191 |
| delta | 0.010243736622504033 |
| wins | 0 |
| wins_recent | 0 |
| latest_delta | 0.011281828419630546 |
| worst_delta | 0.011281828419630546 |
| raw_delta | 0.010728810844990127 |
| offset_mean | -0.06861270838302198 |
| offset_std | 0.015075426500793472 |
| fold_scores | [1.7762759032894748, 1.7711592583338476, 1.7608649530642964, 1.7548321694370088] |
| fold_deltas | [0.007032134697465331, 0.008434238038520725, 0.009875202801502336, 0.011281828419630546] |
| notes |  |

## Evidence row 17

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_existing.csv`

| Field | Value |
|---|---|
| name | multiscale_direct |
| family | raw_new |
| wcv | 1.760047439180695 |
| base_wcv | 1.749803702558191 |
| delta | 0.010243736622504033 |
| wins | 0 |
| wins_recent | 0 |
| latest_delta | 0.011281828419630546 |
| worst_delta | 0.011281828419630546 |
| raw_delta | 0.010728810844990127 |
| offset_mean | -0.06861270838302198 |
| offset_std | 0.015075426500793472 |
| fold_scores | [1.7762759032894748, 1.7711592583338476, 1.7608649530642964, 1.7548321694370088] |
| fold_deltas | [0.007032134697465331, 0.008434238038520725, 0.009875202801502336, 0.011281828419630546] |
| notes |  |

## Evidence row 18

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_final.csv`

| Field | Value |
|---|---|
| name | multiscale_direct |
| family | raw_new |
| wcv | 1.760047439180695 |
| base_wcv | 1.749803702558191 |
| delta | 0.010243736622504033 |
| wins | 0 |
| wins_recent | 0 |
| latest_delta | 0.011281828419630546 |
| worst_delta | 0.011281828419630546 |
| raw_delta | 0.010728810844990127 |
| offset_mean | -0.06861270838302198 |
| offset_std | 0.015075426500793472 |
| fold_scores | [1.7762759032894748, 1.7711592583338476, 1.7608649530642964, 1.7548321694370088] |
| fold_deltas | [0.007032134697465331, 0.008434238038520725, 0.009875202801502336, 0.011281828419630546] |
| notes |  |

## Evidence row 19

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_final_reload.csv`

| Field | Value |
|---|---|
| name | multiscale_direct |
| family | raw_new |
| wcv | 1.760047439180695 |
| base_wcv | 1.749803702558191 |
| delta | 0.010243736622504033 |
| wins | 0 |
| wins_recent | 0 |
| latest_delta | 0.011281828419630546 |
| worst_delta | 0.011281828419630546 |
| raw_delta | 0.010728810844990127 |
| offset_mean | -0.06861270838302198 |
| offset_std | 0.015075426500793472 |
| fold_scores | [1.7762759032894748, 1.7711592583338476, 1.7608649530642964, 1.7548321694370088] |
| fold_deltas | [0.007032134697465331, 0.008434238038520725, 0.009875202801502336, 0.011281828419630546] |
| notes |  |
