# Teammate candidate — ridge_all_s075

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__ridge_all_s075`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `ridge_all_s075`
- **Original source:** `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/combo_validation_extra90.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/primitive_validation_extra90.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/STABLE_BRANCH_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/combo_validation_final6h_base.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/FAMILY_BEST.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/primitive_validation_final6h_base.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/STABLE_BRANCH_VALIDATION.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/ALL_VALIDATION.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_fast12.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_stable18.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_existing.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final_reload.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_after_recent_hurdle_fast12.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_after_recent_hurdle_stable18.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_existing.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_final.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_final_reload.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/SHORTLIST_VALIDATION.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Ridge, two-part / hurdle
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`
- **Known score:** | wcv | 1.7482553756178343 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 21 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — ridge_all_s075

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_all_s075 |
| family | ridge_subset |
| wcv | 1.7482553756178343 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015483269403566999 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016711404303517696 |
| worst_delta | 0.0 |
| raw_delta | -0.0023258321498251947 |
| offset_mean | -0.023556581221166013 |
| offset_std | 0.05374460928372161 |
| fold_scores | [1.7692437685920095, 1.7609093293153943, 1.7494336505871262, 1.7418792005870265] |
| fold_deltas | [0.0, -0.001815690979932505, -0.0015560996756678325, -0.0016711404303517696] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_direct', 'recent_dist', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/combo_validation_extra90.csv`

| Field | Value |
|---|---|
| name | ridge_all_s075 |
| family | ridge_subset |
| wcv | 1.7482553756178343 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015483269403566999 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016711404303517696 |
| worst_delta | 0.0 |
| raw_delta | -0.0023258321498251947 |
| offset_mean | -0.023556581221166013 |
| offset_std | 0.05374460928372161 |
| fold_scores | [1.7692437685920095, 1.7609093293153943, 1.7494336505871262, 1.7418792005870265] |
| fold_deltas | [0.0, -0.001815690979932505, -0.0015560996756678325, -0.0016711404303517696] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_direct', 'recent_dist', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |

## Evidence row 3

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/primitive_validation_extra90.csv`

| Field | Value |
|---|---|
| name | ridge_all_s075 |
| family | ridge_subset |
| wcv | 1.7482553756178343 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015483269403566999 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016711404303517696 |
| worst_delta | 0.0 |
| raw_delta | -0.0023258321498251947 |
| offset_mean | -0.023556581221166013 |
| offset_std | 0.05374460928372161 |
| fold_scores | [1.7692437685920095, 1.7609093293153943, 1.7494336505871262, 1.7418792005870265] |
| fold_deltas | [0.0, -0.001815690979932505, -0.0015560996756678325, -0.0016711404303517696] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_direct', 'recent_dist', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |

## Evidence row 4

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/STABLE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_all_s075 |
| family | ridge_subset |
| wcv | 1.7482553756178343 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015483269403566999 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016711404303517696 |
| worst_delta | 0.0 |
| raw_delta | -0.0023258321498251947 |
| offset_mean | -0.023556581221166013 |
| offset_std | 0.05374460928372161 |
| fold_scores | [1.7692437685920095, 1.7609093293153943, 1.7494336505871262, 1.7418792005870265] |
| fold_deltas | [0.0, -0.001815690979932505, -0.0015560996756678325, -0.0016711404303517696] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_direct', 'recent_dist', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |

## Evidence row 5

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_all_s075 |
| family | ridge_subset |
| wcv | 1.7482553756178343 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015483269403566999 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016711404303517696 |
| worst_delta | 0.0 |
| raw_delta | -0.0023258321498251947 |
| offset_mean | -0.023556581221166013 |
| offset_std | 0.05374460928372161 |
| fold_scores | [1.7692437685920095, 1.7609093293153943, 1.7494336505871262, 1.7418792005870265] |
| fold_deltas | [0.0, -0.001815690979932505, -0.0015560996756678325, -0.0016711404303517696] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_direct', 'recent_dist', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |

## Evidence row 6

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/combo_validation_final6h_base.csv`

| Field | Value |
|---|---|
| name | ridge_all_s075 |
| family | ridge_subset |
| wcv | 1.7482553756178343 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015483269403566999 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016711404303517696 |
| worst_delta | 0.0 |
| raw_delta | -0.0023258321498251947 |
| offset_mean | -0.023556581221166013 |
| offset_std | 0.05374460928372161 |
| fold_scores | [1.7692437685920095, 1.7609093293153943, 1.7494336505871262, 1.7418792005870265] |
| fold_deltas | [0.0, -0.001815690979932505, -0.0015560996756678325, -0.0016711404303517696] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_direct', 'recent_dist', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |

## Evidence row 7

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/FAMILY_BEST.csv`

| Field | Value |
|---|---|
| family | ridge_subset |
| best_name | ridge_all_s075 |
| delta | -0.0015483269403566999 |
| wins_recent | 3 |
| latest_delta | -0.0016711404303517696 |

## Evidence row 8

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/primitive_validation_final6h_base.csv`

| Field | Value |
|---|---|
| name | ridge_all_s075 |
| family | ridge_subset |
| wcv | 1.7482553756178343 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015483269403566999 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016711404303517696 |
| worst_delta | 0.0 |
| raw_delta | -0.0023258321498251947 |
| offset_mean | -0.023556581221166013 |
| offset_std | 0.05374460928372161 |
| fold_scores | [1.7692437685920095, 1.7609093293153943, 1.7494336505871262, 1.7418792005870265] |
| fold_deltas | [0.0, -0.001815690979932505, -0.0015560996756678325, -0.0016711404303517696] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_direct', 'recent_dist', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |

## Evidence row 9

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/STABLE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_all_s075 |
| family | ridge_subset |
| wcv | 1.7482553756178343 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015483269403566999 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016711404303517696 |
| worst_delta | 0.0 |
| raw_delta | -0.0023258321498251947 |
| offset_mean | -0.023556581221166013 |
| offset_std | 0.05374460928372161 |
| fold_scores | [1.7692437685920095, 1.7609093293153943, 1.7494336505871262, 1.7418792005870265] |
| fold_deltas | [0.0, -0.001815690979932505, -0.0015560996756678325, -0.0016711404303517696] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_direct', 'recent_dist', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |

## Evidence row 10

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/ALL_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_all_s075 |
| family | ridge_subset |
| wcv | 1.7482553756178343 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015483269403566999 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016711404303517696 |
| worst_delta | 0.0 |
| raw_delta | -0.0023258321498251947 |
| offset_mean | -0.023556581221166013 |
| offset_std | 0.05374460928372161 |
| fold_scores | [1.7692437685920095, 1.7609093293153943, 1.7494336505871262, 1.7418792005870265] |
| fold_deltas | [0.0, -0.001815690979932505, -0.0015560996756678325, -0.0016711404303517696] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_direct', 'recent_dist', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |

## Evidence row 11

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_fast12.csv`

| Field | Value |
|---|---|
| name | ridge_all_s075 |
| family | ridge_subset |
| wcv | 1.7482567583377555 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015469442204357087 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016618744898704296 |
| worst_delta | 0.0 |
| raw_delta | -0.0023243363739783676 |
| offset_mean | -0.023518035306064413 |
| offset_std | 0.0537708663794488 |
| fold_scores | [1.7692437685920095, 1.7608868133724076, 1.7494315618773606, 1.7418884665275078] |
| fold_deltas | [0.0, -0.00183820692291925, -0.0015581883854334233, -0.0016618744898704296] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_direct', 'recent_dist', 'recent_hurdle_fast12'] |

## Evidence row 12

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_stable18.csv`

| Field | Value |
|---|---|
| name | ridge_all_s075 |
| family | ridge_subset |
| wcv | 1.7482553756178343 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015483269403566999 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016711404303517696 |
| worst_delta | 0.0 |
| raw_delta | -0.0023258321498251947 |
| offset_mean | -0.023556581221166013 |
| offset_std | 0.05374460928372161 |
| fold_scores | [1.7692437685920095, 1.7609093293153943, 1.7494336505871262, 1.7418792005870265] |
| fold_deltas | [0.0, -0.001815690979932505, -0.0015560996756678325, -0.0016711404303517696] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_direct', 'recent_dist', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |

## Evidence row 13

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_existing.csv`

| Field | Value |
|---|---|
| name | ridge_all_s075 |
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

## Evidence row 14

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final.csv`

| Field | Value |
|---|---|
| name | ridge_all_s075 |
| family | ridge_subset |
| wcv | 1.7482553756178343 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015483269403566999 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016711404303517696 |
| worst_delta | 0.0 |
| raw_delta | -0.0023258321498251947 |
| offset_mean | -0.023556581221166013 |
| offset_std | 0.05374460928372161 |
| fold_scores | [1.7692437685920095, 1.7609093293153943, 1.7494336505871262, 1.7418792005870265] |
| fold_deltas | [0.0, -0.001815690979932505, -0.0015560996756678325, -0.0016711404303517696] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_direct', 'recent_dist', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |

## Evidence row 15

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final_reload.csv`

| Field | Value |
|---|---|
| name | ridge_all_s075 |
| family | ridge_subset |
| wcv | 1.7482553756178343 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015483269403566999 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016711404303517696 |
| worst_delta | 0.0 |
| raw_delta | -0.0023258321498251947 |
| offset_mean | -0.023556581221166013 |
| offset_std | 0.05374460928372161 |
| fold_scores | [1.7692437685920095, 1.7609093293153943, 1.7494336505871262, 1.7418792005870265] |
| fold_deltas | [0.0, -0.001815690979932505, -0.0015560996756678325, -0.0016711404303517696] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_direct', 'recent_dist', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |

## Evidence row 16

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_after_recent_hurdle_fast12.csv`

| Field | Value |
|---|---|
| name | ridge_all_s075 |
| family | ridge_subset |
| wcv | 1.7482567583377555 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015469442204357087 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016618744898704296 |
| worst_delta | 0.0 |
| raw_delta | -0.0023243363739783676 |
| offset_mean | -0.023518035306064413 |
| offset_std | 0.0537708663794488 |
| fold_scores | [1.7692437685920095, 1.7608868133724076, 1.7494315618773606, 1.7418884665275078] |
| fold_deltas | [0.0, -0.00183820692291925, -0.0015581883854334233, -0.0016618744898704296] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_direct', 'recent_dist', 'recent_hurdle_fast12'] |

## Evidence row 17

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_after_recent_hurdle_stable18.csv`

| Field | Value |
|---|---|
| name | ridge_all_s075 |
| family | ridge_subset |
| wcv | 1.7482553756178343 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015483269403566999 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016711404303517696 |
| worst_delta | 0.0 |
| raw_delta | -0.0023258321498251947 |
| offset_mean | -0.023556581221166013 |
| offset_std | 0.05374460928372161 |
| fold_scores | [1.7692437685920095, 1.7609093293153943, 1.7494336505871262, 1.7418792005870265] |
| fold_deltas | [0.0, -0.001815690979932505, -0.0015560996756678325, -0.0016711404303517696] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_direct', 'recent_dist', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |

## Evidence row 18

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_existing.csv`

| Field | Value |
|---|---|
| name | ridge_all_s075 |
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

## Evidence row 19

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_final.csv`

| Field | Value |
|---|---|
| name | ridge_all_s075 |
| family | ridge_subset |
| wcv | 1.7482553756178343 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015483269403566999 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016711404303517696 |
| worst_delta | 0.0 |
| raw_delta | -0.0023258321498251947 |
| offset_mean | -0.023556581221166013 |
| offset_std | 0.05374460928372161 |
| fold_scores | [1.7692437685920095, 1.7609093293153943, 1.7494336505871262, 1.7418792005870265] |
| fold_deltas | [0.0, -0.001815690979932505, -0.0015560996756678325, -0.0016711404303517696] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_direct', 'recent_dist', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |

## Evidence row 20

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_final_reload.csv`

| Field | Value |
|---|---|
| name | ridge_all_s075 |
| family | ridge_subset |
| wcv | 1.7482553756178343 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015483269403566999 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016711404303517696 |
| worst_delta | 0.0 |
| raw_delta | -0.0023258321498251947 |
| offset_mean | -0.023556581221166013 |
| offset_std | 0.05374460928372161 |
| fold_scores | [1.7692437685920095, 1.7609093293153943, 1.7494336505871262, 1.7418792005870265] |
| fold_deltas | [0.0, -0.001815690979932505, -0.0015560996756678325, -0.0016711404303517696] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_direct', 'recent_dist', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |

## Evidence row 21

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/SHORTLIST_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_all_s075 |
| family | ridge_subset |
| wcv | 1.7482553756178343 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015483269403566999 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016711404303517696 |
| worst_delta | 0.0 |
| raw_delta | -0.0023258321498251947 |
| offset_mean | -0.023556581221166013 |
| offset_std | 0.05374460928372161 |
| fold_scores | [1.7692437685920095, 1.7609093293153943, 1.7494336505871262, 1.7418792005870265] |
| fold_deltas | [0.0, -0.001815690979932505, -0.0015560996756678325, -0.0016711404303517696] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_direct', 'recent_dist', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |
