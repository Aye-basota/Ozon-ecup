# Teammate candidate — ridge_core_plus_multiscale_direct_s075

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__ridge_core_plus_multiscale_direct_s075`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `ridge_core_plus_multiscale_direct_s075`
- **Original source:** `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/combo_validation_extra90.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/primitive_validation_extra90.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/STABLE_BRANCH_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/combo_validation_final6h_base.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/primitive_validation_final6h_base.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/STABLE_BRANCH_VALIDATION.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/ALL_VALIDATION.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_fast12.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_stable18.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_existing.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final_reload.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/FINAL_CANDIDATE_METRICS.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_after_recent_hurdle_fast12.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_after_recent_hurdle_stable18.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_existing.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_final.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_final_reload.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Ridge, two-part / hurdle
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`
- **Known score:** | wcv | 1.7483788595948475 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 20 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — ridge_core_plus_multiscale_direct_s075

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_core_plus_multiscale_direct_s075 |
| family | ridge_subset |
| wcv | 1.7483788595948475 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014248429633435754 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0014725471438383764 |
| worst_delta | 0.0 |
| raw_delta | -0.002206776276751536 |
| offset_mean | -0.023977344300426615 |
| offset_std | 0.053418437117657966 |
| fold_scores | [1.7692437685920095, 1.7609081729928475, 1.749500107089172, 1.74207779387354] |
| fold_deltas | [0.0, -0.0018168473024793297, -0.0014896431736219906, -0.0014725471438383764] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct'] |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/combo_validation_extra90.csv`

| Field | Value |
|---|---|
| name | ridge_core_plus_multiscale_direct_s075 |
| family | ridge_subset |
| wcv | 1.7483788595948475 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014248429633435754 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0014725471438383764 |
| worst_delta | 0.0 |
| raw_delta | -0.002206776276751536 |
| offset_mean | -0.023977344300426615 |
| offset_std | 0.053418437117657966 |
| fold_scores | [1.7692437685920095, 1.7609081729928475, 1.749500107089172, 1.74207779387354] |
| fold_deltas | [0.0, -0.0018168473024793297, -0.0014896431736219906, -0.0014725471438383764] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct'] |

## Evidence row 3

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/primitive_validation_extra90.csv`

| Field | Value |
|---|---|
| name | ridge_core_plus_multiscale_direct_s075 |
| family | ridge_subset |
| wcv | 1.7483788595948475 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014248429633435754 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0014725471438383764 |
| worst_delta | 0.0 |
| raw_delta | -0.002206776276751536 |
| offset_mean | -0.023977344300426615 |
| offset_std | 0.053418437117657966 |
| fold_scores | [1.7692437685920095, 1.7609081729928475, 1.749500107089172, 1.74207779387354] |
| fold_deltas | [0.0, -0.0018168473024793297, -0.0014896431736219906, -0.0014725471438383764] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct'] |

## Evidence row 4

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/STABLE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_core_plus_multiscale_direct_s075 |
| family | ridge_subset |
| wcv | 1.7483788595948475 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014248429633435754 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0014725471438383764 |
| worst_delta | 0.0 |
| raw_delta | -0.002206776276751536 |
| offset_mean | -0.023977344300426615 |
| offset_std | 0.053418437117657966 |
| fold_scores | [1.7692437685920095, 1.7609081729928475, 1.749500107089172, 1.74207779387354] |
| fold_deltas | [0.0, -0.0018168473024793297, -0.0014896431736219906, -0.0014725471438383764] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct'] |

## Evidence row 5

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_core_plus_multiscale_direct_s075 |
| family | ridge_subset |
| wcv | 1.7483788595948475 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014248429633435754 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0014725471438383764 |
| worst_delta | 0.0 |
| raw_delta | -0.002206776276751536 |
| offset_mean | -0.023977344300426615 |
| offset_std | 0.053418437117657966 |
| fold_scores | [1.7692437685920095, 1.7609081729928475, 1.749500107089172, 1.74207779387354] |
| fold_deltas | [0.0, -0.0018168473024793297, -0.0014896431736219906, -0.0014725471438383764] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct'] |

## Evidence row 6

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/combo_validation_final6h_base.csv`

| Field | Value |
|---|---|
| name | ridge_core_plus_multiscale_direct_s075 |
| family | ridge_subset |
| wcv | 1.7483788595948475 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014248429633435754 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0014725471438383764 |
| worst_delta | 0.0 |
| raw_delta | -0.002206776276751536 |
| offset_mean | -0.023977344300426615 |
| offset_std | 0.053418437117657966 |
| fold_scores | [1.7692437685920095, 1.7609081729928475, 1.749500107089172, 1.74207779387354] |
| fold_deltas | [0.0, -0.0018168473024793297, -0.0014896431736219906, -0.0014725471438383764] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct'] |

## Evidence row 7

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/primitive_validation_final6h_base.csv`

| Field | Value |
|---|---|
| name | ridge_core_plus_multiscale_direct_s075 |
| family | ridge_subset |
| wcv | 1.7483788595948475 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014248429633435754 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0014725471438383764 |
| worst_delta | 0.0 |
| raw_delta | -0.002206776276751536 |
| offset_mean | -0.023977344300426615 |
| offset_std | 0.053418437117657966 |
| fold_scores | [1.7692437685920095, 1.7609081729928475, 1.749500107089172, 1.74207779387354] |
| fold_deltas | [0.0, -0.0018168473024793297, -0.0014896431736219906, -0.0014725471438383764] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct'] |

## Evidence row 8

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/STABLE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_core_plus_multiscale_direct_s075 |
| family | ridge_subset |
| wcv | 1.7483788595948475 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014248429633435754 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0014725471438383764 |
| worst_delta | 0.0 |
| raw_delta | -0.002206776276751536 |
| offset_mean | -0.023977344300426615 |
| offset_std | 0.053418437117657966 |
| fold_scores | [1.7692437685920095, 1.7609081729928475, 1.749500107089172, 1.74207779387354] |
| fold_deltas | [0.0, -0.0018168473024793297, -0.0014896431736219906, -0.0014725471438383764] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct'] |

## Evidence row 9

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/ALL_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_core_plus_multiscale_direct_s075 |
| family | ridge_subset |
| wcv | 1.7483788595948475 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014248429633435754 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0014725471438383764 |
| worst_delta | 0.0 |
| raw_delta | -0.002206776276751536 |
| offset_mean | -0.023977344300426615 |
| offset_std | 0.053418437117657966 |
| fold_scores | [1.7692437685920095, 1.7609081729928475, 1.749500107089172, 1.74207779387354] |
| fold_deltas | [0.0, -0.0018168473024793297, -0.0014896431736219906, -0.0014725471438383764] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct'] |

## Evidence row 10

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_fast12.csv`

| Field | Value |
|---|---|
| name | ridge_core_plus_multiscale_direct_s075 |
| family | ridge_subset |
| wcv | 1.7483788595948475 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014248429633435754 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0014725471438383764 |
| worst_delta | 0.0 |
| raw_delta | -0.002206776276751536 |
| offset_mean | -0.023977344300426615 |
| offset_std | 0.053418437117657966 |
| fold_scores | [1.7692437685920095, 1.7609081729928475, 1.749500107089172, 1.74207779387354] |
| fold_deltas | [0.0, -0.0018168473024793297, -0.0014896431736219906, -0.0014725471438383764] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct'] |

## Evidence row 11

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_stable18.csv`

| Field | Value |
|---|---|
| name | ridge_core_plus_multiscale_direct_s075 |
| family | ridge_subset |
| wcv | 1.7483788595948475 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014248429633435754 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0014725471438383764 |
| worst_delta | 0.0 |
| raw_delta | -0.002206776276751536 |
| offset_mean | -0.023977344300426615 |
| offset_std | 0.053418437117657966 |
| fold_scores | [1.7692437685920095, 1.7609081729928475, 1.749500107089172, 1.74207779387354] |
| fold_deltas | [0.0, -0.0018168473024793297, -0.0014896431736219906, -0.0014725471438383764] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct'] |

## Evidence row 12

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_existing.csv`

| Field | Value |
|---|---|
| name | ridge_core_plus_multiscale_direct_s075 |
| family | ridge_subset |
| wcv | 1.7483788595948475 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014248429633435754 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0014725471438383764 |
| worst_delta | 0.0 |
| raw_delta | -0.002206776276751536 |
| offset_mean | -0.023977344300426615 |
| offset_std | 0.053418437117657966 |
| fold_scores | [1.7692437685920095, 1.7609081729928475, 1.749500107089172, 1.74207779387354] |
| fold_deltas | [0.0, -0.0018168473024793297, -0.0014896431736219906, -0.0014725471438383764] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct'] |

## Evidence row 13

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final.csv`

| Field | Value |
|---|---|
| name | ridge_core_plus_multiscale_direct_s075 |
| family | ridge_subset |
| wcv | 1.7483788595948475 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014248429633435754 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0014725471438383764 |
| worst_delta | 0.0 |
| raw_delta | -0.002206776276751536 |
| offset_mean | -0.023977344300426615 |
| offset_std | 0.053418437117657966 |
| fold_scores | [1.7692437685920095, 1.7609081729928475, 1.749500107089172, 1.74207779387354] |
| fold_deltas | [0.0, -0.0018168473024793297, -0.0014896431736219906, -0.0014725471438383764] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct'] |

## Evidence row 14

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final_reload.csv`

| Field | Value |
|---|---|
| name | ridge_core_plus_multiscale_direct_s075 |
| family | ridge_subset |
| wcv | 1.7483788595948475 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014248429633435754 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0014725471438383764 |
| worst_delta | 0.0 |
| raw_delta | -0.002206776276751536 |
| offset_mean | -0.023977344300426615 |
| offset_std | 0.053418437117657966 |
| fold_scores | [1.7692437685920095, 1.7609081729928475, 1.749500107089172, 1.74207779387354] |
| fold_deltas | [0.0, -0.0018168473024793297, -0.0014896431736219906, -0.0014725471438383764] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct'] |

## Evidence row 15

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/FINAL_CANDIDATE_METRICS.csv`

| Field | Value |
|---|---|
| name | ridge_core_plus_multiscale_direct_s075 |
| family | ridge_subset |
| wcv | 1.7483788595948475 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014248429633435754 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0014725471438383764 |
| worst_delta | 0.0 |
| raw_delta | -0.002206776276751536 |
| offset_mean | -0.023977344300426615 |
| offset_std | 0.053418437117657966 |
| fold_scores | [1.7692437685920095, 1.7609081729928475, 1.749500107089172, 1.74207779387354] |
| fold_deltas | [0.0, -0.0018168473024793297, -0.0014896431736219906, -0.0014725471438383764] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct'] |
| oof_table_var | 0.0038672908914413277 |
| test_table_var | 0.004319911553832531 |
| var_ratio | 1.1170381735164776 |
| friend_corr | 0.9997607261258729 |
| friend_std_dz | 0.03611751926652844 |
| friend_mean_abs_dz | 0.027205361350818278 |
| friend_pct02 | 0.526268 |
| friend_pct05 | 0.149384 |
| friend_pct10 | 0.011288 |

## Evidence row 16

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_after_recent_hurdle_fast12.csv`

| Field | Value |
|---|---|
| name | ridge_core_plus_multiscale_direct_s075 |
| family | ridge_subset |
| wcv | 1.7483788595948475 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014248429633435754 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0014725471438383764 |
| worst_delta | 0.0 |
| raw_delta | -0.002206776276751536 |
| offset_mean | -0.023977344300426615 |
| offset_std | 0.053418437117657966 |
| fold_scores | [1.7692437685920095, 1.7609081729928475, 1.749500107089172, 1.74207779387354] |
| fold_deltas | [0.0, -0.0018168473024793297, -0.0014896431736219906, -0.0014725471438383764] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct'] |

## Evidence row 17

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_after_recent_hurdle_stable18.csv`

| Field | Value |
|---|---|
| name | ridge_core_plus_multiscale_direct_s075 |
| family | ridge_subset |
| wcv | 1.7483788595948475 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014248429633435754 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0014725471438383764 |
| worst_delta | 0.0 |
| raw_delta | -0.002206776276751536 |
| offset_mean | -0.023977344300426615 |
| offset_std | 0.053418437117657966 |
| fold_scores | [1.7692437685920095, 1.7609081729928475, 1.749500107089172, 1.74207779387354] |
| fold_deltas | [0.0, -0.0018168473024793297, -0.0014896431736219906, -0.0014725471438383764] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct'] |

## Evidence row 18

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_existing.csv`

| Field | Value |
|---|---|
| name | ridge_core_plus_multiscale_direct_s075 |
| family | ridge_subset |
| wcv | 1.7483788595948475 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014248429633435754 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0014725471438383764 |
| worst_delta | 0.0 |
| raw_delta | -0.002206776276751536 |
| offset_mean | -0.023977344300426615 |
| offset_std | 0.053418437117657966 |
| fold_scores | [1.7692437685920095, 1.7609081729928475, 1.749500107089172, 1.74207779387354] |
| fold_deltas | [0.0, -0.0018168473024793297, -0.0014896431736219906, -0.0014725471438383764] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct'] |

## Evidence row 19

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_final.csv`

| Field | Value |
|---|---|
| name | ridge_core_plus_multiscale_direct_s075 |
| family | ridge_subset |
| wcv | 1.7483788595948475 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014248429633435754 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0014725471438383764 |
| worst_delta | 0.0 |
| raw_delta | -0.002206776276751536 |
| offset_mean | -0.023977344300426615 |
| offset_std | 0.053418437117657966 |
| fold_scores | [1.7692437685920095, 1.7609081729928475, 1.749500107089172, 1.74207779387354] |
| fold_deltas | [0.0, -0.0018168473024793297, -0.0014896431736219906, -0.0014725471438383764] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct'] |

## Evidence row 20

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_final_reload.csv`

| Field | Value |
|---|---|
| name | ridge_core_plus_multiscale_direct_s075 |
| family | ridge_subset |
| wcv | 1.7483788595948475 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014248429633435754 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0014725471438383764 |
| worst_delta | 0.0 |
| raw_delta | -0.002206776276751536 |
| offset_mean | -0.023977344300426615 |
| offset_std | 0.053418437117657966 |
| fold_scores | [1.7692437685920095, 1.7609081729928475, 1.749500107089172, 1.74207779387354] |
| fold_deltas | [0.0, -0.0018168473024793297, -0.0014896431736219906, -0.0014725471438383764] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct'] |
