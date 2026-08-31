# Teammate candidate — ridge_drop_recent_direct_s075

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__ridge_drop_recent_direct_s075`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `ridge_drop_recent_direct_s075`
- **Original source:** `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/combo_validation_extra90.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/primitive_validation_extra90.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/STABLE_BRANCH_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/combo_validation_final6h_base.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/primitive_validation_final6h_base.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/STABLE_BRANCH_VALIDATION.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/ALL_VALIDATION.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_fast12.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_stable18.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_existing.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final_reload.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_after_recent_hurdle_fast12.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_after_recent_hurdle_stable18.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_existing.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_final.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_final_reload.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Ridge, two-part / hurdle
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`
- **Known score:** | wcv | 1.748266298902999 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 19 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — ridge_drop_recent_direct_s075

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_drop_recent_direct_s075 |
| family | ridge_subset |
| wcv | 1.748266298902999 |
| base_wcv | 1.749803702558191 |
| delta | -0.001537403655192134 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001654896712939058 |
| worst_delta | 0.0 |
| raw_delta | -0.0023159114814465197 |
| offset_mean | -0.02358483593070413 |
| offset_std | 0.053706664843733515 |
| fold_scores | [1.7692437685920095, 1.7609239995512627, 1.7494347903537337, 1.7418954443044392] |
| fold_deltas | [0.0, -0.0018010207440641857, -0.001554959909060294, -0.001654896712939058] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_dist', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/combo_validation_extra90.csv`

| Field | Value |
|---|---|
| name | ridge_drop_recent_direct_s075 |
| family | ridge_subset |
| wcv | 1.748266298902999 |
| base_wcv | 1.749803702558191 |
| delta | -0.001537403655192134 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001654896712939058 |
| worst_delta | 0.0 |
| raw_delta | -0.0023159114814465197 |
| offset_mean | -0.02358483593070413 |
| offset_std | 0.053706664843733515 |
| fold_scores | [1.7692437685920095, 1.7609239995512627, 1.7494347903537337, 1.7418954443044392] |
| fold_deltas | [0.0, -0.0018010207440641857, -0.001554959909060294, -0.001654896712939058] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_dist', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |

## Evidence row 3

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/primitive_validation_extra90.csv`

| Field | Value |
|---|---|
| name | ridge_drop_recent_direct_s075 |
| family | ridge_subset |
| wcv | 1.748266298902999 |
| base_wcv | 1.749803702558191 |
| delta | -0.001537403655192134 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001654896712939058 |
| worst_delta | 0.0 |
| raw_delta | -0.0023159114814465197 |
| offset_mean | -0.02358483593070413 |
| offset_std | 0.053706664843733515 |
| fold_scores | [1.7692437685920095, 1.7609239995512627, 1.7494347903537337, 1.7418954443044392] |
| fold_deltas | [0.0, -0.0018010207440641857, -0.001554959909060294, -0.001654896712939058] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_dist', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |

## Evidence row 4

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/STABLE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_drop_recent_direct_s075 |
| family | ridge_subset |
| wcv | 1.748266298902999 |
| base_wcv | 1.749803702558191 |
| delta | -0.001537403655192134 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001654896712939058 |
| worst_delta | 0.0 |
| raw_delta | -0.0023159114814465197 |
| offset_mean | -0.02358483593070413 |
| offset_std | 0.053706664843733515 |
| fold_scores | [1.7692437685920095, 1.7609239995512627, 1.7494347903537337, 1.7418954443044392] |
| fold_deltas | [0.0, -0.0018010207440641857, -0.001554959909060294, -0.001654896712939058] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_dist', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |

## Evidence row 5

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_drop_recent_direct_s075 |
| family | ridge_subset |
| wcv | 1.748266298902999 |
| base_wcv | 1.749803702558191 |
| delta | -0.001537403655192134 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001654896712939058 |
| worst_delta | 0.0 |
| raw_delta | -0.0023159114814465197 |
| offset_mean | -0.02358483593070413 |
| offset_std | 0.053706664843733515 |
| fold_scores | [1.7692437685920095, 1.7609239995512627, 1.7494347903537337, 1.7418954443044392] |
| fold_deltas | [0.0, -0.0018010207440641857, -0.001554959909060294, -0.001654896712939058] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_dist', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |

## Evidence row 6

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/combo_validation_final6h_base.csv`

| Field | Value |
|---|---|
| name | ridge_drop_recent_direct_s075 |
| family | ridge_subset |
| wcv | 1.748266298902999 |
| base_wcv | 1.749803702558191 |
| delta | -0.001537403655192134 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001654896712939058 |
| worst_delta | 0.0 |
| raw_delta | -0.0023159114814465197 |
| offset_mean | -0.02358483593070413 |
| offset_std | 0.053706664843733515 |
| fold_scores | [1.7692437685920095, 1.7609239995512627, 1.7494347903537337, 1.7418954443044392] |
| fold_deltas | [0.0, -0.0018010207440641857, -0.001554959909060294, -0.001654896712939058] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_dist', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |

## Evidence row 7

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/primitive_validation_final6h_base.csv`

| Field | Value |
|---|---|
| name | ridge_drop_recent_direct_s075 |
| family | ridge_subset |
| wcv | 1.748266298902999 |
| base_wcv | 1.749803702558191 |
| delta | -0.001537403655192134 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001654896712939058 |
| worst_delta | 0.0 |
| raw_delta | -0.0023159114814465197 |
| offset_mean | -0.02358483593070413 |
| offset_std | 0.053706664843733515 |
| fold_scores | [1.7692437685920095, 1.7609239995512627, 1.7494347903537337, 1.7418954443044392] |
| fold_deltas | [0.0, -0.0018010207440641857, -0.001554959909060294, -0.001654896712939058] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_dist', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |

## Evidence row 8

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/STABLE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_drop_recent_direct_s075 |
| family | ridge_subset |
| wcv | 1.748266298902999 |
| base_wcv | 1.749803702558191 |
| delta | -0.001537403655192134 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001654896712939058 |
| worst_delta | 0.0 |
| raw_delta | -0.0023159114814465197 |
| offset_mean | -0.02358483593070413 |
| offset_std | 0.053706664843733515 |
| fold_scores | [1.7692437685920095, 1.7609239995512627, 1.7494347903537337, 1.7418954443044392] |
| fold_deltas | [0.0, -0.0018010207440641857, -0.001554959909060294, -0.001654896712939058] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_dist', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |

## Evidence row 9

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/ALL_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_drop_recent_direct_s075 |
| family | ridge_subset |
| wcv | 1.748266298902999 |
| base_wcv | 1.749803702558191 |
| delta | -0.001537403655192134 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001654896712939058 |
| worst_delta | 0.0 |
| raw_delta | -0.0023159114814465197 |
| offset_mean | -0.02358483593070413 |
| offset_std | 0.053706664843733515 |
| fold_scores | [1.7692437685920095, 1.7609239995512627, 1.7494347903537337, 1.7418954443044392] |
| fold_deltas | [0.0, -0.0018010207440641857, -0.001554959909060294, -0.001654896712939058] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_dist', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |

## Evidence row 10

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_fast12.csv`

| Field | Value |
|---|---|
| name | ridge_drop_recent_direct_s075 |
| family | ridge_subset |
| wcv | 1.7482550745308931 |
| base_wcv | 1.749803702558191 |
| delta | -0.001548628027297738 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016535303255991174 |
| worst_delta | 0.0 |
| raw_delta | -0.002326356442438815 |
| offset_mean | -0.023536865276973334 |
| offset_std | 0.053752017893981485 |
| fold_scores | [1.7692437685920095, 1.7608987513046994, 1.7494025903069395, 1.7418968106917792] |
| fold_deltas | [0.0, -0.0018262689906274332, -0.0015871599558545668, -0.0016535303255991174] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_dist', 'recent_hurdle_fast12'] |

## Evidence row 11

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_stable18.csv`

| Field | Value |
|---|---|
| name | ridge_drop_recent_direct_s075 |
| family | ridge_subset |
| wcv | 1.748266298902999 |
| base_wcv | 1.749803702558191 |
| delta | -0.001537403655192134 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001654896712939058 |
| worst_delta | 0.0 |
| raw_delta | -0.0023159114814465197 |
| offset_mean | -0.02358483593070413 |
| offset_std | 0.053706664843733515 |
| fold_scores | [1.7692437685920095, 1.7609239995512627, 1.7494347903537337, 1.7418954443044392] |
| fold_deltas | [0.0, -0.0018010207440641857, -0.001554959909060294, -0.001654896712939058] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_dist', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |

## Evidence row 12

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_existing.csv`

| Field | Value |
|---|---|
| name | ridge_drop_recent_direct_s075 |
| family | ridge_subset |
| wcv | 1.7483457621782095 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014579403799816148 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0015202697331160309 |
| worst_delta | 0.0 |
| raw_delta | -0.002237741423336903 |
| offset_mean | -0.023912742379886217 |
| offset_std | 0.053496571967613475 |
| fold_scores | [1.7692437685920095, 1.760895102292377, 1.74947797230557, 1.7420300712842622] |
| fold_deltas | [0.0, -0.0018299180029499595, -0.0015117779572240142, -0.0015202697331160309] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_dist'] |

## Evidence row 13

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final.csv`

| Field | Value |
|---|---|
| name | ridge_drop_recent_direct_s075 |
| family | ridge_subset |
| wcv | 1.748266298902999 |
| base_wcv | 1.749803702558191 |
| delta | -0.001537403655192134 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001654896712939058 |
| worst_delta | 0.0 |
| raw_delta | -0.0023159114814465197 |
| offset_mean | -0.02358483593070413 |
| offset_std | 0.053706664843733515 |
| fold_scores | [1.7692437685920095, 1.7609239995512627, 1.7494347903537337, 1.7418954443044392] |
| fold_deltas | [0.0, -0.0018010207440641857, -0.001554959909060294, -0.001654896712939058] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_dist', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |

## Evidence row 14

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final_reload.csv`

| Field | Value |
|---|---|
| name | ridge_drop_recent_direct_s075 |
| family | ridge_subset |
| wcv | 1.748266298902999 |
| base_wcv | 1.749803702558191 |
| delta | -0.001537403655192134 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001654896712939058 |
| worst_delta | 0.0 |
| raw_delta | -0.0023159114814465197 |
| offset_mean | -0.02358483593070413 |
| offset_std | 0.053706664843733515 |
| fold_scores | [1.7692437685920095, 1.7609239995512627, 1.7494347903537337, 1.7418954443044392] |
| fold_deltas | [0.0, -0.0018010207440641857, -0.001554959909060294, -0.001654896712939058] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_dist', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |

## Evidence row 15

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_after_recent_hurdle_fast12.csv`

| Field | Value |
|---|---|
| name | ridge_drop_recent_direct_s075 |
| family | ridge_subset |
| wcv | 1.7482550745308931 |
| base_wcv | 1.749803702558191 |
| delta | -0.001548628027297738 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016535303255991174 |
| worst_delta | 0.0 |
| raw_delta | -0.002326356442438815 |
| offset_mean | -0.023536865276973334 |
| offset_std | 0.053752017893981485 |
| fold_scores | [1.7692437685920095, 1.7608987513046994, 1.7494025903069395, 1.7418968106917792] |
| fold_deltas | [0.0, -0.0018262689906274332, -0.0015871599558545668, -0.0016535303255991174] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_dist', 'recent_hurdle_fast12'] |

## Evidence row 16

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_after_recent_hurdle_stable18.csv`

| Field | Value |
|---|---|
| name | ridge_drop_recent_direct_s075 |
| family | ridge_subset |
| wcv | 1.748266298902999 |
| base_wcv | 1.749803702558191 |
| delta | -0.001537403655192134 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001654896712939058 |
| worst_delta | 0.0 |
| raw_delta | -0.0023159114814465197 |
| offset_mean | -0.02358483593070413 |
| offset_std | 0.053706664843733515 |
| fold_scores | [1.7692437685920095, 1.7609239995512627, 1.7494347903537337, 1.7418954443044392] |
| fold_deltas | [0.0, -0.0018010207440641857, -0.001554959909060294, -0.001654896712939058] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_dist', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |

## Evidence row 17

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_existing.csv`

| Field | Value |
|---|---|
| name | ridge_drop_recent_direct_s075 |
| family | ridge_subset |
| wcv | 1.7483457621782095 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014579403799816148 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0015202697331160309 |
| worst_delta | 0.0 |
| raw_delta | -0.002237741423336903 |
| offset_mean | -0.023912742379886217 |
| offset_std | 0.053496571967613475 |
| fold_scores | [1.7692437685920095, 1.760895102292377, 1.74947797230557, 1.7420300712842622] |
| fold_deltas | [0.0, -0.0018299180029499595, -0.0015117779572240142, -0.0015202697331160309] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_dist'] |

## Evidence row 18

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_final.csv`

| Field | Value |
|---|---|
| name | ridge_drop_recent_direct_s075 |
| family | ridge_subset |
| wcv | 1.748266298902999 |
| base_wcv | 1.749803702558191 |
| delta | -0.001537403655192134 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001654896712939058 |
| worst_delta | 0.0 |
| raw_delta | -0.0023159114814465197 |
| offset_mean | -0.02358483593070413 |
| offset_std | 0.053706664843733515 |
| fold_scores | [1.7692437685920095, 1.7609239995512627, 1.7494347903537337, 1.7418954443044392] |
| fold_deltas | [0.0, -0.0018010207440641857, -0.001554959909060294, -0.001654896712939058] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_dist', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |

## Evidence row 19

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_final_reload.csv`

| Field | Value |
|---|---|
| name | ridge_drop_recent_direct_s075 |
| family | ridge_subset |
| wcv | 1.748266298902999 |
| base_wcv | 1.749803702558191 |
| delta | -0.001537403655192134 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001654896712939058 |
| worst_delta | 0.0 |
| raw_delta | -0.0023159114814465197 |
| offset_mean | -0.02358483593070413 |
| offset_std | 0.053706664843733515 |
| fold_scores | [1.7692437685920095, 1.7609239995512627, 1.7494347903537337, 1.7418954443044392] |
| fold_deltas | [0.0, -0.0018010207440641857, -0.001554959909060294, -0.001654896712939058] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'multiscale_direct', 'recent_dist', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |
