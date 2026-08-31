# Teammate candidate — ridge_drop_recent_dist_s075

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__ridge_drop_recent_dist_s075`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `ridge_drop_recent_dist_s075`
- **Original source:** `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/combo_validation_extra90.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/primitive_validation_extra90.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/STABLE_BRANCH_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/combo_validation_final6h_base.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/primitive_validation_final6h_base.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/STABLE_BRANCH_VALIDATION.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/ALL_VALIDATION.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_fast12.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_stable18.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_existing.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final_reload.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_after_recent_hurdle_fast12.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_after_recent_hurdle_stable18.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_existing.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_final.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_final_reload.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/SHORTLIST_VALIDATION.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Ridge, two-part / hurdle
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`
- **Known score:** | wcv | 1.7482559921525223 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 20 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — ridge_drop_recent_dist_s075

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_drop_recent_dist_s075 |
| family | ridge_subset |
| wcv | 1.7482559921525223 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015477104056688044 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001653801782828479 |
| worst_delta | 0.0 |
| raw_delta | -0.0023262317516991853 |
| offset_mean | -0.0234721300339171 |
| offset_std | 0.053773673732904 |
| fold_scores | [1.7692437685920095, 1.760877976842981, 1.749416961533366, 1.7418965392345498] |
| fold_deltas | [0.0, -0.0018470434523458756, -0.0015727887294281206, -0.001653801782828479] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_direct', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/combo_validation_extra90.csv`

| Field | Value |
|---|---|
| name | ridge_drop_recent_dist_s075 |
| family | ridge_subset |
| wcv | 1.7482559921525223 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015477104056688044 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001653801782828479 |
| worst_delta | 0.0 |
| raw_delta | -0.0023262317516991853 |
| offset_mean | -0.0234721300339171 |
| offset_std | 0.053773673732904 |
| fold_scores | [1.7692437685920095, 1.760877976842981, 1.749416961533366, 1.7418965392345498] |
| fold_deltas | [0.0, -0.0018470434523458756, -0.0015727887294281206, -0.001653801782828479] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_direct', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |

## Evidence row 3

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/primitive_validation_extra90.csv`

| Field | Value |
|---|---|
| name | ridge_drop_recent_dist_s075 |
| family | ridge_subset |
| wcv | 1.7482559921525223 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015477104056688044 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001653801782828479 |
| worst_delta | 0.0 |
| raw_delta | -0.0023262317516991853 |
| offset_mean | -0.0234721300339171 |
| offset_std | 0.053773673732904 |
| fold_scores | [1.7692437685920095, 1.760877976842981, 1.749416961533366, 1.7418965392345498] |
| fold_deltas | [0.0, -0.0018470434523458756, -0.0015727887294281206, -0.001653801782828479] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_direct', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |

## Evidence row 4

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/STABLE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_drop_recent_dist_s075 |
| family | ridge_subset |
| wcv | 1.7482559921525223 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015477104056688044 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001653801782828479 |
| worst_delta | 0.0 |
| raw_delta | -0.0023262317516991853 |
| offset_mean | -0.0234721300339171 |
| offset_std | 0.053773673732904 |
| fold_scores | [1.7692437685920095, 1.760877976842981, 1.749416961533366, 1.7418965392345498] |
| fold_deltas | [0.0, -0.0018470434523458756, -0.0015727887294281206, -0.001653801782828479] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_direct', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |

## Evidence row 5

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_drop_recent_dist_s075 |
| family | ridge_subset |
| wcv | 1.7482559921525223 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015477104056688044 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001653801782828479 |
| worst_delta | 0.0 |
| raw_delta | -0.0023262317516991853 |
| offset_mean | -0.0234721300339171 |
| offset_std | 0.053773673732904 |
| fold_scores | [1.7692437685920095, 1.760877976842981, 1.749416961533366, 1.7418965392345498] |
| fold_deltas | [0.0, -0.0018470434523458756, -0.0015727887294281206, -0.001653801782828479] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_direct', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |

## Evidence row 6

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/combo_validation_final6h_base.csv`

| Field | Value |
|---|---|
| name | ridge_drop_recent_dist_s075 |
| family | ridge_subset |
| wcv | 1.7482559921525223 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015477104056688044 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001653801782828479 |
| worst_delta | 0.0 |
| raw_delta | -0.0023262317516991853 |
| offset_mean | -0.0234721300339171 |
| offset_std | 0.053773673732904 |
| fold_scores | [1.7692437685920095, 1.760877976842981, 1.749416961533366, 1.7418965392345498] |
| fold_deltas | [0.0, -0.0018470434523458756, -0.0015727887294281206, -0.001653801782828479] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_direct', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |

## Evidence row 7

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/primitive_validation_final6h_base.csv`

| Field | Value |
|---|---|
| name | ridge_drop_recent_dist_s075 |
| family | ridge_subset |
| wcv | 1.7482559921525223 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015477104056688044 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001653801782828479 |
| worst_delta | 0.0 |
| raw_delta | -0.0023262317516991853 |
| offset_mean | -0.0234721300339171 |
| offset_std | 0.053773673732904 |
| fold_scores | [1.7692437685920095, 1.760877976842981, 1.749416961533366, 1.7418965392345498] |
| fold_deltas | [0.0, -0.0018470434523458756, -0.0015727887294281206, -0.001653801782828479] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_direct', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |

## Evidence row 8

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/STABLE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_drop_recent_dist_s075 |
| family | ridge_subset |
| wcv | 1.7482559921525223 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015477104056688044 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001653801782828479 |
| worst_delta | 0.0 |
| raw_delta | -0.0023262317516991853 |
| offset_mean | -0.0234721300339171 |
| offset_std | 0.053773673732904 |
| fold_scores | [1.7692437685920095, 1.760877976842981, 1.749416961533366, 1.7418965392345498] |
| fold_deltas | [0.0, -0.0018470434523458756, -0.0015727887294281206, -0.001653801782828479] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_direct', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |

## Evidence row 9

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/ALL_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_drop_recent_dist_s075 |
| family | ridge_subset |
| wcv | 1.7482559921525223 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015477104056688044 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001653801782828479 |
| worst_delta | 0.0 |
| raw_delta | -0.0023262317516991853 |
| offset_mean | -0.0234721300339171 |
| offset_std | 0.053773673732904 |
| fold_scores | [1.7692437685920095, 1.760877976842981, 1.749416961533366, 1.7418965392345498] |
| fold_deltas | [0.0, -0.0018470434523458756, -0.0015727887294281206, -0.001653801782828479] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_direct', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |

## Evidence row 10

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_fast12.csv`

| Field | Value |
|---|---|
| name | ridge_drop_recent_dist_s075 |
| family | ridge_subset |
| wcv | 1.748251577272303 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015521252858879287 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001661968350114229 |
| worst_delta | 0.0 |
| raw_delta | -0.0023315765473953753 |
| offset_mean | -0.023611325725551868 |
| offset_std | 0.053678411260266845 |
| fold_scores | [1.7692437685920095, 1.7608757162217208, 1.7494178691777458, 1.741888372667264] |
| fold_deltas | [0.0, -0.0018493040736060617, -0.0015718810850482434, -0.001661968350114229] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_direct', 'recent_hurdle_fast12'] |

## Evidence row 11

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_stable18.csv`

| Field | Value |
|---|---|
| name | ridge_drop_recent_dist_s075 |
| family | ridge_subset |
| wcv | 1.7482559921525223 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015477104056688044 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001653801782828479 |
| worst_delta | 0.0 |
| raw_delta | -0.0023262317516991853 |
| offset_mean | -0.0234721300339171 |
| offset_std | 0.053773673732904 |
| fold_scores | [1.7692437685920095, 1.760877976842981, 1.749416961533366, 1.7418965392345498] |
| fold_deltas | [0.0, -0.0018470434523458756, -0.0015727887294281206, -0.001653801782828479] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_direct', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |

## Evidence row 12

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_existing.csv`

| Field | Value |
|---|---|
| name | ridge_drop_recent_dist_s075 |
| family | ridge_subset |
| wcv | 1.7483762211338256 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014274814243652992 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001480220002856747 |
| worst_delta | 0.0 |
| raw_delta | -0.00220835254867738 |
| offset_mean | -0.02387305477789095 |
| offset_std | 0.05350336852243025 |
| fold_scores | [1.7692437685920095, 1.7609050283502756, 1.7495071308996633, 1.7420701210145215] |
| fold_deltas | [0.0, -0.0018199919450512247, -0.001482619363130766, -0.001480220002856747] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_direct'] |

## Evidence row 13

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final.csv`

| Field | Value |
|---|---|
| name | ridge_drop_recent_dist_s075 |
| family | ridge_subset |
| wcv | 1.7482559921525223 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015477104056688044 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001653801782828479 |
| worst_delta | 0.0 |
| raw_delta | -0.0023262317516991853 |
| offset_mean | -0.0234721300339171 |
| offset_std | 0.053773673732904 |
| fold_scores | [1.7692437685920095, 1.760877976842981, 1.749416961533366, 1.7418965392345498] |
| fold_deltas | [0.0, -0.0018470434523458756, -0.0015727887294281206, -0.001653801782828479] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_direct', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |

## Evidence row 14

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final_reload.csv`

| Field | Value |
|---|---|
| name | ridge_drop_recent_dist_s075 |
| family | ridge_subset |
| wcv | 1.7482559921525223 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015477104056688044 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001653801782828479 |
| worst_delta | 0.0 |
| raw_delta | -0.0023262317516991853 |
| offset_mean | -0.0234721300339171 |
| offset_std | 0.053773673732904 |
| fold_scores | [1.7692437685920095, 1.760877976842981, 1.749416961533366, 1.7418965392345498] |
| fold_deltas | [0.0, -0.0018470434523458756, -0.0015727887294281206, -0.001653801782828479] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_direct', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |

## Evidence row 15

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_after_recent_hurdle_fast12.csv`

| Field | Value |
|---|---|
| name | ridge_drop_recent_dist_s075 |
| family | ridge_subset |
| wcv | 1.748251577272303 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015521252858879287 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001661968350114229 |
| worst_delta | 0.0 |
| raw_delta | -0.0023315765473953753 |
| offset_mean | -0.023611325725551868 |
| offset_std | 0.053678411260266845 |
| fold_scores | [1.7692437685920095, 1.7608757162217208, 1.7494178691777458, 1.741888372667264] |
| fold_deltas | [0.0, -0.0018493040736060617, -0.0015718810850482434, -0.001661968350114229] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_direct', 'recent_hurdle_fast12'] |

## Evidence row 16

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_after_recent_hurdle_stable18.csv`

| Field | Value |
|---|---|
| name | ridge_drop_recent_dist_s075 |
| family | ridge_subset |
| wcv | 1.7482559921525223 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015477104056688044 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001653801782828479 |
| worst_delta | 0.0 |
| raw_delta | -0.0023262317516991853 |
| offset_mean | -0.0234721300339171 |
| offset_std | 0.053773673732904 |
| fold_scores | [1.7692437685920095, 1.760877976842981, 1.749416961533366, 1.7418965392345498] |
| fold_deltas | [0.0, -0.0018470434523458756, -0.0015727887294281206, -0.001653801782828479] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_direct', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |

## Evidence row 17

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_existing.csv`

| Field | Value |
|---|---|
| name | ridge_drop_recent_dist_s075 |
| family | ridge_subset |
| wcv | 1.7483762211338256 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014274814243652992 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001480220002856747 |
| worst_delta | 0.0 |
| raw_delta | -0.00220835254867738 |
| offset_mean | -0.02387305477789095 |
| offset_std | 0.05350336852243025 |
| fold_scores | [1.7692437685920095, 1.7609050283502756, 1.7495071308996633, 1.7420701210145215] |
| fold_deltas | [0.0, -0.0018199919450512247, -0.001482619363130766, -0.001480220002856747] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_direct'] |

## Evidence row 18

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_final.csv`

| Field | Value |
|---|---|
| name | ridge_drop_recent_dist_s075 |
| family | ridge_subset |
| wcv | 1.7482559921525223 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015477104056688044 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001653801782828479 |
| worst_delta | 0.0 |
| raw_delta | -0.0023262317516991853 |
| offset_mean | -0.0234721300339171 |
| offset_std | 0.053773673732904 |
| fold_scores | [1.7692437685920095, 1.760877976842981, 1.749416961533366, 1.7418965392345498] |
| fold_deltas | [0.0, -0.0018470434523458756, -0.0015727887294281206, -0.001653801782828479] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_direct', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |

## Evidence row 19

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_final_reload.csv`

| Field | Value |
|---|---|
| name | ridge_drop_recent_dist_s075 |
| family | ridge_subset |
| wcv | 1.7482559921525223 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015477104056688044 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001653801782828479 |
| worst_delta | 0.0 |
| raw_delta | -0.0023262317516991853 |
| offset_mean | -0.0234721300339171 |
| offset_std | 0.053773673732904 |
| fold_scores | [1.7692437685920095, 1.760877976842981, 1.749416961533366, 1.7418965392345498] |
| fold_deltas | [0.0, -0.0018470434523458756, -0.0015727887294281206, -0.001653801782828479] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_direct', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |

## Evidence row 20

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/SHORTLIST_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_drop_recent_dist_s075 |
| family | ridge_subset |
| wcv | 1.7482559921525223 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015477104056688044 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001653801782828479 |
| worst_delta | 0.0 |
| raw_delta | -0.0023262317516991853 |
| offset_mean | -0.0234721300339171 |
| offset_std | 0.053773673732904 |
| fold_scores | [1.7692437685920095, 1.760877976842981, 1.749416961533366, 1.7418965392345498] |
| fold_deltas | [0.0, -0.0018470434523458756, -0.0015727887294281206, -0.001653801782828479] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_direct', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |
