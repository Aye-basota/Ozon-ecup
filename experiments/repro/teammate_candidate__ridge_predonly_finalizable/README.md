# Teammate candidate — ridge_predonly_finalizable

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__ridge_predonly_finalizable`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `ridge_predonly_finalizable`
- **Original source:** `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/combo_validation_extra90.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/primitive_validation_extra90.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/STABLE_BRANCH_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/combo_validation_final6h_base.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/FAMILY_BEST.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/primitive_validation_final6h_base.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/STABLE_BRANCH_VALIDATION.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/ALL_VALIDATION.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_fast12.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_stable18.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_existing.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final_reload.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_after_recent_hurdle_fast12.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_after_recent_hurdle_stable18.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_existing.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_final.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_final_reload.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Ridge, two-part / hurdle
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`
- **Known score:** | wcv | 1.748685133520886 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 20 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — ridge_predonly_finalizable

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_predonly_finalizable |
| family | ridge_predonly |
| wcv | 1.748685133520886 |
| base_wcv | 1.749803702558191 |
| delta | -0.0011185690373051985 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0013661150134705036 |
| worst_delta | 0.0 |
| raw_delta | -0.0017360610687487288 |
| offset_mean | -0.005948782690852096 |
| offset_std | 0.06471133708466575 |
| fold_scores | [1.7692437685920095, 1.7617546931602568, 1.7500125099673756, 1.7421842260039078] |
| fold_deltas | [0.0, -0.0009703271350700593, -0.0009772402954184578, -0.0013661150134705036] |
| notes |  |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/combo_validation_extra90.csv`

| Field | Value |
|---|---|
| name | ridge_predonly_finalizable |
| family | ridge_predonly |
| wcv | 1.748685133520886 |
| base_wcv | 1.749803702558191 |
| delta | -0.0011185690373051985 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0013661150134705036 |
| worst_delta | 0.0 |
| raw_delta | -0.0017360610687487288 |
| offset_mean | -0.005948782690852096 |
| offset_std | 0.06471133708466575 |
| fold_scores | [1.7692437685920095, 1.7617546931602568, 1.7500125099673756, 1.7421842260039078] |
| fold_deltas | [0.0, -0.0009703271350700593, -0.0009772402954184578, -0.0013661150134705036] |
| notes |  |

## Evidence row 3

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/primitive_validation_extra90.csv`

| Field | Value |
|---|---|
| name | ridge_predonly_finalizable |
| family | ridge_predonly |
| wcv | 1.748685133520886 |
| base_wcv | 1.749803702558191 |
| delta | -0.0011185690373051985 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0013661150134705036 |
| worst_delta | 0.0 |
| raw_delta | -0.0017360610687487288 |
| offset_mean | -0.005948782690852096 |
| offset_std | 0.06471133708466575 |
| fold_scores | [1.7692437685920095, 1.7617546931602568, 1.7500125099673756, 1.7421842260039078] |
| fold_deltas | [0.0, -0.0009703271350700593, -0.0009772402954184578, -0.0013661150134705036] |
| notes |  |

## Evidence row 4

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/STABLE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_predonly_finalizable |
| family | ridge_predonly |
| wcv | 1.748685133520886 |
| base_wcv | 1.749803702558191 |
| delta | -0.0011185690373051985 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0013661150134705036 |
| worst_delta | 0.0 |
| raw_delta | -0.0017360610687487288 |
| offset_mean | -0.005948782690852096 |
| offset_std | 0.06471133708466575 |
| fold_scores | [1.7692437685920095, 1.7617546931602568, 1.7500125099673756, 1.7421842260039078] |
| fold_deltas | [0.0, -0.0009703271350700593, -0.0009772402954184578, -0.0013661150134705036] |
| notes |  |

## Evidence row 5

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_predonly_finalizable |
| family | ridge_predonly |
| wcv | 1.748685133520886 |
| base_wcv | 1.749803702558191 |
| delta | -0.0011185690373051985 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0013661150134705036 |
| worst_delta | 0.0 |
| raw_delta | -0.0017360610687487288 |
| offset_mean | -0.005948782690852096 |
| offset_std | 0.06471133708466575 |
| fold_scores | [1.7692437685920095, 1.7617546931602568, 1.7500125099673756, 1.7421842260039078] |
| fold_deltas | [0.0, -0.0009703271350700593, -0.0009772402954184578, -0.0013661150134705036] |
| notes |  |

## Evidence row 6

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/combo_validation_final6h_base.csv`

| Field | Value |
|---|---|
| name | ridge_predonly_finalizable |
| family | ridge_predonly |
| wcv | 1.748685133520886 |
| base_wcv | 1.749803702558191 |
| delta | -0.0011185690373051985 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0013661150134705036 |
| worst_delta | 0.0 |
| raw_delta | -0.0017360610687487288 |
| offset_mean | -0.005948782690852096 |
| offset_std | 0.06471133708466575 |
| fold_scores | [1.7692437685920095, 1.7617546931602568, 1.7500125099673756, 1.7421842260039078] |
| fold_deltas | [0.0, -0.0009703271350700593, -0.0009772402954184578, -0.0013661150134705036] |
| notes |  |

## Evidence row 7

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/FAMILY_BEST.csv`

| Field | Value |
|---|---|
| family | ridge_predonly |
| best_name | ridge_predonly_finalizable |
| delta | -0.0011185690373051985 |
| wins_recent | 3 |
| latest_delta | -0.0013661150134705036 |

## Evidence row 8

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/primitive_validation_final6h_base.csv`

| Field | Value |
|---|---|
| name | ridge_predonly_finalizable |
| family | ridge_predonly |
| wcv | 1.748685133520886 |
| base_wcv | 1.749803702558191 |
| delta | -0.0011185690373051985 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0013661150134705036 |
| worst_delta | 0.0 |
| raw_delta | -0.0017360610687487288 |
| offset_mean | -0.005948782690852096 |
| offset_std | 0.06471133708466575 |
| fold_scores | [1.7692437685920095, 1.7617546931602568, 1.7500125099673756, 1.7421842260039078] |
| fold_deltas | [0.0, -0.0009703271350700593, -0.0009772402954184578, -0.0013661150134705036] |
| notes |  |

## Evidence row 9

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/STABLE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_predonly_finalizable |
| family | ridge_predonly |
| wcv | 1.748685133520886 |
| base_wcv | 1.749803702558191 |
| delta | -0.0011185690373051985 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0013661150134705036 |
| worst_delta | 0.0 |
| raw_delta | -0.0017360610687487288 |
| offset_mean | -0.005948782690852096 |
| offset_std | 0.06471133708466575 |
| fold_scores | [1.7692437685920095, 1.7617546931602568, 1.7500125099673756, 1.7421842260039078] |
| fold_deltas | [0.0, -0.0009703271350700593, -0.0009772402954184578, -0.0013661150134705036] |
| notes |  |

## Evidence row 10

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/ALL_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_predonly_finalizable |
| family | ridge_predonly |
| wcv | 1.748685133520886 |
| base_wcv | 1.749803702558191 |
| delta | -0.0011185690373051985 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0013661150134705036 |
| worst_delta | 0.0 |
| raw_delta | -0.0017360610687487288 |
| offset_mean | -0.005948782690852096 |
| offset_std | 0.06471133708466575 |
| fold_scores | [1.7692437685920095, 1.7617546931602568, 1.7500125099673756, 1.7421842260039078] |
| fold_deltas | [0.0, -0.0009703271350700593, -0.0009772402954184578, -0.0013661150134705036] |
| notes |  |

## Evidence row 11

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_fast12.csv`

| Field | Value |
|---|---|
| name | ridge_predonly_finalizable |
| family | ridge_predonly |
| wcv | 1.7486888160259564 |
| base_wcv | 1.749803702558191 |
| delta | -0.0011148865322345416 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001360560492774221 |
| worst_delta | 0.0 |
| raw_delta | -0.0017357797354733862 |
| offset_mean | -0.006059972465566301 |
| offset_std | 0.06462758665756026 |
| fold_scores | [1.7692437685920095, 1.7617498922222683, 1.7500176107889922, 1.742189780524604] |
| fold_deltas | [0.0, -0.000975128073058551, -0.0009721394738018141, -0.001360560492774221] |
| notes |  |

## Evidence row 12

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_stable18.csv`

| Field | Value |
|---|---|
| name | ridge_predonly_finalizable |
| family | ridge_predonly |
| wcv | 1.748685133520886 |
| base_wcv | 1.749803702558191 |
| delta | -0.0011185690373051985 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0013661150134705036 |
| worst_delta | 0.0 |
| raw_delta | -0.0017360610687487288 |
| offset_mean | -0.005948782690852096 |
| offset_std | 0.06471133708466575 |
| fold_scores | [1.7692437685920095, 1.7617546931602568, 1.7500125099673756, 1.7421842260039078] |
| fold_deltas | [0.0, -0.0009703271350700593, -0.0009772402954184578, -0.0013661150134705036] |
| notes |  |

## Evidence row 13

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_existing.csv`

| Field | Value |
|---|---|
| name | ridge_predonly_finalizable |
| family | ridge_predonly |
| wcv | 1.7488130175461634 |
| base_wcv | 1.749803702558191 |
| delta | -0.0009906850120274886 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001174844250826812 |
| worst_delta | 0.0 |
| raw_delta | -0.0016236473917066027 |
| offset_mean | -0.006628629550135249 |
| offset_std | 0.06426341562137784 |
| fold_scores | [1.7692437685920095, 1.7617511541995423, 1.7501113030172368, 1.7423754967665515] |
| fold_deltas | [0.0, -0.0009738660957845369, -0.0008784472455571901, -0.001174844250826812] |
| notes |  |

## Evidence row 14

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final.csv`

| Field | Value |
|---|---|
| name | ridge_predonly_finalizable |
| family | ridge_predonly |
| wcv | 1.748685133520886 |
| base_wcv | 1.749803702558191 |
| delta | -0.0011185690373051985 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0013661150134705036 |
| worst_delta | 0.0 |
| raw_delta | -0.0017360610687487288 |
| offset_mean | -0.005948782690852096 |
| offset_std | 0.06471133708466575 |
| fold_scores | [1.7692437685920095, 1.7617546931602568, 1.7500125099673756, 1.7421842260039078] |
| fold_deltas | [0.0, -0.0009703271350700593, -0.0009772402954184578, -0.0013661150134705036] |
| notes |  |

## Evidence row 15

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final_reload.csv`

| Field | Value |
|---|---|
| name | ridge_predonly_finalizable |
| family | ridge_predonly |
| wcv | 1.748685133520886 |
| base_wcv | 1.749803702558191 |
| delta | -0.0011185690373051985 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0013661150134705036 |
| worst_delta | 0.0 |
| raw_delta | -0.0017360610687487288 |
| offset_mean | -0.005948782690852096 |
| offset_std | 0.06471133708466575 |
| fold_scores | [1.7692437685920095, 1.7617546931602568, 1.7500125099673756, 1.7421842260039078] |
| fold_deltas | [0.0, -0.0009703271350700593, -0.0009772402954184578, -0.0013661150134705036] |
| notes |  |

## Evidence row 16

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_after_recent_hurdle_fast12.csv`

| Field | Value |
|---|---|
| name | ridge_predonly_finalizable |
| family | ridge_predonly |
| wcv | 1.7486888160259564 |
| base_wcv | 1.749803702558191 |
| delta | -0.0011148865322345416 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001360560492774221 |
| worst_delta | 0.0 |
| raw_delta | -0.0017357797354733862 |
| offset_mean | -0.006059972465566301 |
| offset_std | 0.06462758665756026 |
| fold_scores | [1.7692437685920095, 1.7617498922222683, 1.7500176107889922, 1.742189780524604] |
| fold_deltas | [0.0, -0.000975128073058551, -0.0009721394738018141, -0.001360560492774221] |
| notes |  |

## Evidence row 17

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_after_recent_hurdle_stable18.csv`

| Field | Value |
|---|---|
| name | ridge_predonly_finalizable |
| family | ridge_predonly |
| wcv | 1.748685133520886 |
| base_wcv | 1.749803702558191 |
| delta | -0.0011185690373051985 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0013661150134705036 |
| worst_delta | 0.0 |
| raw_delta | -0.0017360610687487288 |
| offset_mean | -0.005948782690852096 |
| offset_std | 0.06471133708466575 |
| fold_scores | [1.7692437685920095, 1.7617546931602568, 1.7500125099673756, 1.7421842260039078] |
| fold_deltas | [0.0, -0.0009703271350700593, -0.0009772402954184578, -0.0013661150134705036] |
| notes |  |

## Evidence row 18

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_existing.csv`

| Field | Value |
|---|---|
| name | ridge_predonly_finalizable |
| family | ridge_predonly |
| wcv | 1.7488130175461634 |
| base_wcv | 1.749803702558191 |
| delta | -0.0009906850120274886 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001174844250826812 |
| worst_delta | 0.0 |
| raw_delta | -0.0016236473917066027 |
| offset_mean | -0.006628629550135249 |
| offset_std | 0.06426341562137784 |
| fold_scores | [1.7692437685920095, 1.7617511541995423, 1.7501113030172368, 1.7423754967665515] |
| fold_deltas | [0.0, -0.0009738660957845369, -0.0008784472455571901, -0.001174844250826812] |
| notes |  |

## Evidence row 19

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_final.csv`

| Field | Value |
|---|---|
| name | ridge_predonly_finalizable |
| family | ridge_predonly |
| wcv | 1.748685133520886 |
| base_wcv | 1.749803702558191 |
| delta | -0.0011185690373051985 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0013661150134705036 |
| worst_delta | 0.0 |
| raw_delta | -0.0017360610687487288 |
| offset_mean | -0.005948782690852096 |
| offset_std | 0.06471133708466575 |
| fold_scores | [1.7692437685920095, 1.7617546931602568, 1.7500125099673756, 1.7421842260039078] |
| fold_deltas | [0.0, -0.0009703271350700593, -0.0009772402954184578, -0.0013661150134705036] |
| notes |  |

## Evidence row 20

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_final_reload.csv`

| Field | Value |
|---|---|
| name | ridge_predonly_finalizable |
| family | ridge_predonly |
| wcv | 1.748685133520886 |
| base_wcv | 1.749803702558191 |
| delta | -0.0011185690373051985 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0013661150134705036 |
| worst_delta | 0.0 |
| raw_delta | -0.0017360610687487288 |
| offset_mean | -0.005948782690852096 |
| offset_std | 0.06471133708466575 |
| fold_scores | [1.7692437685920095, 1.7617546931602568, 1.7500125099673756, 1.7421842260039078] |
| fold_deltas | [0.0, -0.0009703271350700593, -0.0009772402954184578, -0.0013661150134705036] |
| notes |  |
