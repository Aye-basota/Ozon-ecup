# Teammate candidate — ridge_all_s65

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__ridge_all_s65`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `ridge_all_s65`
- **Original source:** `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/combo_validation_extra90.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/primitive_validation_extra90.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/STABLE_BRANCH_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/combo_validation_final6h_base.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/primitive_validation_final6h_base.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/STABLE_BRANCH_VALIDATION.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/ALL_VALIDATION.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_fast12.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_stable18.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_existing.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final_reload.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_after_recent_hurdle_fast12.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_after_recent_hurdle_stable18.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_existing.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_final.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_final_reload.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Ridge, two-part / hurdle
- **Features:** 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`
- **Known score:** | wcv | 1.7483024351871654 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 19 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — ridge_all_s65

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_all_s65 |
| family | ridge_shrink |
| wcv | 1.7483024351871654 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015012673710256053 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001610920274576877 |
| worst_delta | 0.0 |
| raw_delta | -0.0022740695037589908 |
| offset_mean | -0.02981063489143307 |
| offset_std | 0.05000601355504332 |
| fold_scores | [1.7692437685920095, 1.7609639303785523, 1.749462383128989, 1.7419394207428014] |
| fold_deltas | [0.0, -0.0017610899167745941, -0.0015273671338049688, -0.001610920274576877] |
| notes |  |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/combo_validation_extra90.csv`

| Field | Value |
|---|---|
| name | ridge_all_s65 |
| family | ridge_shrink |
| wcv | 1.7483024351871654 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015012673710256053 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001610920274576877 |
| worst_delta | 0.0 |
| raw_delta | -0.0022740695037589908 |
| offset_mean | -0.02981063489143307 |
| offset_std | 0.05000601355504332 |
| fold_scores | [1.7692437685920095, 1.7609639303785523, 1.749462383128989, 1.7419394207428014] |
| fold_deltas | [0.0, -0.0017610899167745941, -0.0015273671338049688, -0.001610920274576877] |
| notes |  |

## Evidence row 3

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/primitive_validation_extra90.csv`

| Field | Value |
|---|---|
| name | ridge_all_s65 |
| family | ridge_shrink |
| wcv | 1.7483024351871654 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015012673710256053 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001610920274576877 |
| worst_delta | 0.0 |
| raw_delta | -0.0022740695037589908 |
| offset_mean | -0.02981063489143307 |
| offset_std | 0.05000601355504332 |
| fold_scores | [1.7692437685920095, 1.7609639303785523, 1.749462383128989, 1.7419394207428014] |
| fold_deltas | [0.0, -0.0017610899167745941, -0.0015273671338049688, -0.001610920274576877] |
| notes |  |

## Evidence row 4

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/STABLE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_all_s65 |
| family | ridge_shrink |
| wcv | 1.7483024351871654 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015012673710256053 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001610920274576877 |
| worst_delta | 0.0 |
| raw_delta | -0.0022740695037589908 |
| offset_mean | -0.02981063489143307 |
| offset_std | 0.05000601355504332 |
| fold_scores | [1.7692437685920095, 1.7609639303785523, 1.749462383128989, 1.7419394207428014] |
| fold_deltas | [0.0, -0.0017610899167745941, -0.0015273671338049688, -0.001610920274576877] |
| notes |  |

## Evidence row 5

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_all_s65 |
| family | ridge_shrink |
| wcv | 1.7483024351871654 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015012673710256053 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001610920274576877 |
| worst_delta | 0.0 |
| raw_delta | -0.0022740695037589908 |
| offset_mean | -0.02981063489143307 |
| offset_std | 0.05000601355504332 |
| fold_scores | [1.7692437685920095, 1.7609639303785523, 1.749462383128989, 1.7419394207428014] |
| fold_deltas | [0.0, -0.0017610899167745941, -0.0015273671338049688, -0.001610920274576877] |
| notes |  |

## Evidence row 6

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/combo_validation_final6h_base.csv`

| Field | Value |
|---|---|
| name | ridge_all_s65 |
| family | ridge_shrink |
| wcv | 1.7483024351871654 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015012673710256053 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001610920274576877 |
| worst_delta | 0.0 |
| raw_delta | -0.0022740695037589908 |
| offset_mean | -0.02981063489143307 |
| offset_std | 0.05000601355504332 |
| fold_scores | [1.7692437685920095, 1.7609639303785523, 1.749462383128989, 1.7419394207428014] |
| fold_deltas | [0.0, -0.0017610899167745941, -0.0015273671338049688, -0.001610920274576877] |
| notes |  |

## Evidence row 7

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/primitive_validation_final6h_base.csv`

| Field | Value |
|---|---|
| name | ridge_all_s65 |
| family | ridge_shrink |
| wcv | 1.7483024351871654 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015012673710256053 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001610920274576877 |
| worst_delta | 0.0 |
| raw_delta | -0.0022740695037589908 |
| offset_mean | -0.02981063489143307 |
| offset_std | 0.05000601355504332 |
| fold_scores | [1.7692437685920095, 1.7609639303785523, 1.749462383128989, 1.7419394207428014] |
| fold_deltas | [0.0, -0.0017610899167745941, -0.0015273671338049688, -0.001610920274576877] |
| notes |  |

## Evidence row 8

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/STABLE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_all_s65 |
| family | ridge_shrink |
| wcv | 1.7483024351871654 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015012673710256053 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001610920274576877 |
| worst_delta | 0.0 |
| raw_delta | -0.0022740695037589908 |
| offset_mean | -0.02981063489143307 |
| offset_std | 0.05000601355504332 |
| fold_scores | [1.7692437685920095, 1.7609639303785523, 1.749462383128989, 1.7419394207428014] |
| fold_deltas | [0.0, -0.0017610899167745941, -0.0015273671338049688, -0.001610920274576877] |
| notes |  |

## Evidence row 9

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/ALL_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_all_s65 |
| family | ridge_shrink |
| wcv | 1.7483024351871654 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015012673710256053 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001610920274576877 |
| worst_delta | 0.0 |
| raw_delta | -0.0022740695037589908 |
| offset_mean | -0.02981063489143307 |
| offset_std | 0.05000601355504332 |
| fold_scores | [1.7692437685920095, 1.7609639303785523, 1.749462383128989, 1.7419394207428014] |
| fold_deltas | [0.0, -0.0017610899167745941, -0.0015273671338049688, -0.001610920274576877] |
| notes |  |

## Evidence row 10

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_fast12.csv`

| Field | Value |
|---|---|
| name | ridge_all_s65 |
| family | ridge_shrink |
| wcv | 1.7483043449566518 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014993576015391123 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016013011770825702 |
| worst_delta | 0.0 |
| raw_delta | -0.0022720543369175654 |
| offset_mean | -0.029777190050940375 |
| offset_std | 0.05002442745755808 |
| fold_scores | [1.7692437685920095, 1.760943417779214, 1.749460562869244, 1.7419490398402957] |
| fold_deltas | [0.0, -0.0017816025161128124, -0.0015291873935501243, -0.0016013011770825702] |
| notes |  |

## Evidence row 11

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_stable18.csv`

| Field | Value |
|---|---|
| name | ridge_all_s65 |
| family | ridge_shrink |
| wcv | 1.7483024351871654 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015012673710256053 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001610920274576877 |
| worst_delta | 0.0 |
| raw_delta | -0.0022740695037589908 |
| offset_mean | -0.02981063489143307 |
| offset_std | 0.05000601355504332 |
| fold_scores | [1.7692437685920095, 1.7609639303785523, 1.749462383128989, 1.7419394207428014] |
| fold_deltas | [0.0, -0.0017610899167745941, -0.0015273671338049688, -0.001610920274576877] |
| notes |  |

## Evidence row 12

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_existing.csv`

| Field | Value |
|---|---|
| name | ridge_all_s65 |
| family | ridge_shrink |
| wcv | 1.7483835468112712 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014201557469199304 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0014868305047903707 |
| worst_delta | 0.0 |
| raw_delta | -0.002193423060041688 |
| offset_mean | -0.03004644447883026 |
| offset_std | 0.04982561253822636 |
| fold_scores | [1.7692437685920095, 1.760942007392296, 1.7495293336729405, 1.742063510512588] |
| fold_deltas | [0.0, -0.001783012903030956, -0.0014604165898535193, -0.0014868305047903707] |
| notes |  |

## Evidence row 13

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final.csv`

| Field | Value |
|---|---|
| name | ridge_all_s65 |
| family | ridge_shrink |
| wcv | 1.7483024351871654 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015012673710256053 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001610920274576877 |
| worst_delta | 0.0 |
| raw_delta | -0.0022740695037589908 |
| offset_mean | -0.02981063489143307 |
| offset_std | 0.05000601355504332 |
| fold_scores | [1.7692437685920095, 1.7609639303785523, 1.749462383128989, 1.7419394207428014] |
| fold_deltas | [0.0, -0.0017610899167745941, -0.0015273671338049688, -0.001610920274576877] |
| notes |  |

## Evidence row 14

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final_reload.csv`

| Field | Value |
|---|---|
| name | ridge_all_s65 |
| family | ridge_shrink |
| wcv | 1.7483024351871654 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015012673710256053 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001610920274576877 |
| worst_delta | 0.0 |
| raw_delta | -0.0022740695037589908 |
| offset_mean | -0.02981063489143307 |
| offset_std | 0.05000601355504332 |
| fold_scores | [1.7692437685920095, 1.7609639303785523, 1.749462383128989, 1.7419394207428014] |
| fold_deltas | [0.0, -0.0017610899167745941, -0.0015273671338049688, -0.001610920274576877] |
| notes |  |

## Evidence row 15

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_after_recent_hurdle_fast12.csv`

| Field | Value |
|---|---|
| name | ridge_all_s65 |
| family | ridge_shrink |
| wcv | 1.7483043449566518 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014993576015391123 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016013011770825702 |
| worst_delta | 0.0 |
| raw_delta | -0.0022720543369175654 |
| offset_mean | -0.029777190050940375 |
| offset_std | 0.05002442745755808 |
| fold_scores | [1.7692437685920095, 1.760943417779214, 1.749460562869244, 1.7419490398402957] |
| fold_deltas | [0.0, -0.0017816025161128124, -0.0015291873935501243, -0.0016013011770825702] |
| notes |  |

## Evidence row 16

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_after_recent_hurdle_stable18.csv`

| Field | Value |
|---|---|
| name | ridge_all_s65 |
| family | ridge_shrink |
| wcv | 1.7483024351871654 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015012673710256053 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001610920274576877 |
| worst_delta | 0.0 |
| raw_delta | -0.0022740695037589908 |
| offset_mean | -0.02981063489143307 |
| offset_std | 0.05000601355504332 |
| fold_scores | [1.7692437685920095, 1.7609639303785523, 1.749462383128989, 1.7419394207428014] |
| fold_deltas | [0.0, -0.0017610899167745941, -0.0015273671338049688, -0.001610920274576877] |
| notes |  |

## Evidence row 17

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_existing.csv`

| Field | Value |
|---|---|
| name | ridge_all_s65 |
| family | ridge_shrink |
| wcv | 1.7483835468112712 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014201557469199304 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0014868305047903707 |
| worst_delta | 0.0 |
| raw_delta | -0.002193423060041688 |
| offset_mean | -0.03004644447883026 |
| offset_std | 0.04982561253822636 |
| fold_scores | [1.7692437685920095, 1.760942007392296, 1.7495293336729405, 1.742063510512588] |
| fold_deltas | [0.0, -0.001783012903030956, -0.0014604165898535193, -0.0014868305047903707] |
| notes |  |

## Evidence row 18

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_final.csv`

| Field | Value |
|---|---|
| name | ridge_all_s65 |
| family | ridge_shrink |
| wcv | 1.7483024351871654 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015012673710256053 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001610920274576877 |
| worst_delta | 0.0 |
| raw_delta | -0.0022740695037589908 |
| offset_mean | -0.02981063489143307 |
| offset_std | 0.05000601355504332 |
| fold_scores | [1.7692437685920095, 1.7609639303785523, 1.749462383128989, 1.7419394207428014] |
| fold_deltas | [0.0, -0.0017610899167745941, -0.0015273671338049688, -0.001610920274576877] |
| notes |  |

## Evidence row 19

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_final_reload.csv`

| Field | Value |
|---|---|
| name | ridge_all_s65 |
| family | ridge_shrink |
| wcv | 1.7483024351871654 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015012673710256053 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001610920274576877 |
| worst_delta | 0.0 |
| raw_delta | -0.0022740695037589908 |
| offset_mean | -0.02981063489143307 |
| offset_std | 0.05000601355504332 |
| fold_scores | [1.7692437685920095, 1.7609639303785523, 1.749462383128989, 1.7419394207428014] |
| fold_deltas | [0.0, -0.0017610899167745941, -0.0015273671338049688, -0.001610920274576877] |
| notes |  |
