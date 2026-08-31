# Teammate candidate — bias_p_ridge_recentpow1p7_s075

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__bias_p_ridge_recentpow1p7_s075`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `bias_p_ridge_recentpow1p7_s075`
- **Original source:** `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/combo_validation_extra90.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/STABLE_BRANCH_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/combo_validation_final6h_base.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/STABLE_BRANCH_VALIDATION.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/ALL_VALIDATION.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_fast12.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_stable18.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_existing.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final_reload.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/SHORTLIST_VALIDATION.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Ridge, two-part / hurdle
- **Features:** 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`
- **Known score:** | wcv | 1.7481992525836516 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 13 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — bias_p_ridge_recentpow1p7_s075

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`

| Field | Value |
|---|---|
| name | bias_p_ridge_recentpow1p7_s075 |
| family | local_bias |
| wcv | 1.7481992525836516 |
| base_wcv | 1.749803702558191 |
| delta | -0.001604449974539494 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001735082914407382 |
| worst_delta | 0.0 |
| raw_delta | -0.002363989499594569 |
| offset_mean | -0.02567348892294259 |
| offset_std | 0.05289584881657064 |
| fold_scores | [1.7692437685920095, 1.7608712874843424, 1.749370095092578, 1.7418152581029709] |
| fold_deltas | [0.0, -0.001853732810984443, -0.0016196551702161166, -0.001735082914407382] |
| notes |  |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/combo_validation_extra90.csv`

| Field | Value |
|---|---|
| name | bias_p_ridge_recentpow1p7_s075 |
| family | local_bias |
| wcv | 1.7481992525836516 |
| base_wcv | 1.749803702558191 |
| delta | -0.001604449974539494 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001735082914407382 |
| worst_delta | 0.0 |
| raw_delta | -0.002363989499594569 |
| offset_mean | -0.02567348892294259 |
| offset_std | 0.05289584881657064 |
| fold_scores | [1.7692437685920095, 1.7608712874843424, 1.749370095092578, 1.7418152581029709] |
| fold_deltas | [0.0, -0.001853732810984443, -0.0016196551702161166, -0.001735082914407382] |
| notes |  |

## Evidence row 3

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/STABLE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | bias_p_ridge_recentpow1p7_s075 |
| family | local_bias |
| wcv | 1.7481992525836516 |
| base_wcv | 1.749803702558191 |
| delta | -0.001604449974539494 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001735082914407382 |
| worst_delta | 0.0 |
| raw_delta | -0.002363989499594569 |
| offset_mean | -0.02567348892294259 |
| offset_std | 0.05289584881657064 |
| fold_scores | [1.7692437685920095, 1.7608712874843424, 1.749370095092578, 1.7418152581029709] |
| fold_deltas | [0.0, -0.001853732810984443, -0.0016196551702161166, -0.001735082914407382] |
| notes |  |

## Evidence row 4

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`

| Field | Value |
|---|---|
| name | bias_p_ridge_recentpow1p7_s075 |
| family | local_bias |
| wcv | 1.7481992525836516 |
| base_wcv | 1.749803702558191 |
| delta | -0.001604449974539494 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001735082914407382 |
| worst_delta | 0.0 |
| raw_delta | -0.002363989499594569 |
| offset_mean | -0.02567348892294259 |
| offset_std | 0.05289584881657064 |
| fold_scores | [1.7692437685920095, 1.7608712874843424, 1.749370095092578, 1.7418152581029709] |
| fold_deltas | [0.0, -0.001853732810984443, -0.0016196551702161166, -0.001735082914407382] |
| notes |  |

## Evidence row 5

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/combo_validation_final6h_base.csv`

| Field | Value |
|---|---|
| name | bias_p_ridge_recentpow1p7_s075 |
| family | local_bias |
| wcv | 1.7481992525836516 |
| base_wcv | 1.749803702558191 |
| delta | -0.001604449974539494 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001735082914407382 |
| worst_delta | 0.0 |
| raw_delta | -0.002363989499594569 |
| offset_mean | -0.02567348892294259 |
| offset_std | 0.05289584881657064 |
| fold_scores | [1.7692437685920095, 1.7608712874843424, 1.749370095092578, 1.7418152581029709] |
| fold_deltas | [0.0, -0.001853732810984443, -0.0016196551702161166, -0.001735082914407382] |
| notes |  |

## Evidence row 6

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/STABLE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | bias_p_ridge_recentpow1p7_s075 |
| family | local_bias |
| wcv | 1.7481992525836516 |
| base_wcv | 1.749803702558191 |
| delta | -0.001604449974539494 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001735082914407382 |
| worst_delta | 0.0 |
| raw_delta | -0.002363989499594569 |
| offset_mean | -0.02567348892294259 |
| offset_std | 0.05289584881657064 |
| fold_scores | [1.7692437685920095, 1.7608712874843424, 1.749370095092578, 1.7418152581029709] |
| fold_deltas | [0.0, -0.001853732810984443, -0.0016196551702161166, -0.001735082914407382] |
| notes |  |

## Evidence row 7

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/ALL_VALIDATION.csv`

| Field | Value |
|---|---|
| name | bias_p_ridge_recentpow1p7_s075 |
| family | local_bias |
| wcv | 1.7481992525836516 |
| base_wcv | 1.749803702558191 |
| delta | -0.001604449974539494 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001735082914407382 |
| worst_delta | 0.0 |
| raw_delta | -0.002363989499594569 |
| offset_mean | -0.02567348892294259 |
| offset_std | 0.05289584881657064 |
| fold_scores | [1.7692437685920095, 1.7608712874843424, 1.749370095092578, 1.7418152581029709] |
| fold_deltas | [0.0, -0.001853732810984443, -0.0016196551702161166, -0.001735082914407382] |
| notes |  |

## Evidence row 8

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_fast12.csv`

| Field | Value |
|---|---|
| name | bias_p_ridge_recentpow1p7_s075 |
| family | local_bias |
| wcv | 1.748184955509061 |
| base_wcv | 1.749803702558191 |
| delta | -0.0016187470491300833 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.00174460544957733 |
| worst_delta | 0.0 |
| raw_delta | -0.0023791132542282747 |
| offset_mean | -0.025632838384945553 |
| offset_std | 0.05290564949268084 |
| fold_scores | [1.7692437685920095, 1.7608531997448904, 1.749344570002929, 1.741805735567801] |
| fold_deltas | [0.0, -0.0018718205504364427, -0.0016451802598649312, -0.00174460544957733] |
| notes |  |

## Evidence row 9

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_stable18.csv`

| Field | Value |
|---|---|
| name | bias_p_ridge_recentpow1p7_s075 |
| family | local_bias |
| wcv | 1.7481992525836516 |
| base_wcv | 1.749803702558191 |
| delta | -0.001604449974539494 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001735082914407382 |
| worst_delta | 0.0 |
| raw_delta | -0.002363989499594569 |
| offset_mean | -0.02567348892294259 |
| offset_std | 0.05289584881657064 |
| fold_scores | [1.7692437685920095, 1.7608712874843424, 1.749370095092578, 1.7418152581029709] |
| fold_deltas | [0.0, -0.001853732810984443, -0.0016196551702161166, -0.001735082914407382] |
| notes |  |

## Evidence row 10

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_existing.csv`

| Field | Value |
|---|---|
| name | bias_p_ridge_recentpow1p7_s075 |
| family | local_bias |
| wcv | 1.748285702789391 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015179997688001378 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001596469578764692 |
| worst_delta | 0.0 |
| raw_delta | -0.002279167220801881 |
| offset_mean | -0.025960163657536818 |
| offset_std | 0.05268854540186653 |
| fold_scores | [1.7692437685920095, 1.7608552456730764, 1.7494250775984481, 1.7419538714386136] |
| fold_deltas | [0.0, -0.001869774622250464, -0.0015646726643459008, -0.001596469578764692] |
| notes |  |

## Evidence row 11

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final.csv`

| Field | Value |
|---|---|
| name | bias_p_ridge_recentpow1p7_s075 |
| family | local_bias |
| wcv | 1.7481992525836516 |
| base_wcv | 1.749803702558191 |
| delta | -0.001604449974539494 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001735082914407382 |
| worst_delta | 0.0 |
| raw_delta | -0.002363989499594569 |
| offset_mean | -0.02567348892294259 |
| offset_std | 0.05289584881657064 |
| fold_scores | [1.7692437685920095, 1.7608712874843424, 1.749370095092578, 1.7418152581029709] |
| fold_deltas | [0.0, -0.001853732810984443, -0.0016196551702161166, -0.001735082914407382] |
| notes |  |

## Evidence row 12

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final_reload.csv`

| Field | Value |
|---|---|
| name | bias_p_ridge_recentpow1p7_s075 |
| family | local_bias |
| wcv | 1.7481992525836516 |
| base_wcv | 1.749803702558191 |
| delta | -0.001604449974539494 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001735082914407382 |
| worst_delta | 0.0 |
| raw_delta | -0.002363989499594569 |
| offset_mean | -0.02567348892294259 |
| offset_std | 0.05289584881657064 |
| fold_scores | [1.7692437685920095, 1.7608712874843424, 1.749370095092578, 1.7418152581029709] |
| fold_deltas | [0.0, -0.001853732810984443, -0.0016196551702161166, -0.001735082914407382] |
| notes |  |

## Evidence row 13

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/SHORTLIST_VALIDATION.csv`

| Field | Value |
|---|---|
| name | bias_p_ridge_recentpow1p7_s075 |
| family | local_bias |
| wcv | 1.7481992525836516 |
| base_wcv | 1.749803702558191 |
| delta | -0.001604449974539494 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001735082914407382 |
| worst_delta | 0.0 |
| raw_delta | -0.002363989499594569 |
| offset_mean | -0.02567348892294259 |
| offset_std | 0.05289584881657064 |
| fold_scores | [1.7692437685920095, 1.7608712874843424, 1.749370095092578, 1.7418152581029709] |
| fold_deltas | [0.0, -0.001853732810984443, -0.0016196551702161166, -0.001735082914407382] |
| notes |  |
