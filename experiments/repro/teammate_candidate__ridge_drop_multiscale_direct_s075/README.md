# Teammate candidate — ridge_drop_multiscale_direct_s075

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__ridge_drop_multiscale_direct_s075`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `ridge_drop_multiscale_direct_s075`
- **Original source:** `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/combo_validation_extra90.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/primitive_validation_extra90.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/STABLE_BRANCH_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/combo_validation_final6h_base.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/primitive_validation_final6h_base.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/STABLE_BRANCH_VALIDATION.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/ALL_VALIDATION.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_fast12.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_stable18.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_existing.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final_reload.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_after_recent_hurdle_fast12.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_after_recent_hurdle_stable18.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_existing.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_final.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_final_reload.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Ridge, two-part / hurdle
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`
- **Known score:** | wcv | 1.7482607225672138 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 19 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — ridge_drop_multiscale_direct_s075

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_drop_multiscale_direct_s075 |
| family | ridge_subset |
| wcv | 1.7482607225672138 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015429799909772335 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016645216840378563 |
| worst_delta | 0.0 |
| raw_delta | -0.002318059469876577 |
| offset_mean | -0.023395989128124794 |
| offset_std | 0.053885532262926375 |
| fold_scores | [1.7692437685920095, 1.7609209358906932, 1.749434660867022, 1.7418858193333404] |
| fold_deltas | [0.0, -0.0018040844046336435, -0.0015550893957720913, -0.0016645216840378563] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'recent_direct', 'recent_dist', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/combo_validation_extra90.csv`

| Field | Value |
|---|---|
| name | ridge_drop_multiscale_direct_s075 |
| family | ridge_subset |
| wcv | 1.7482607225672138 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015429799909772335 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016645216840378563 |
| worst_delta | 0.0 |
| raw_delta | -0.002318059469876577 |
| offset_mean | -0.023395989128124794 |
| offset_std | 0.053885532262926375 |
| fold_scores | [1.7692437685920095, 1.7609209358906932, 1.749434660867022, 1.7418858193333404] |
| fold_deltas | [0.0, -0.0018040844046336435, -0.0015550893957720913, -0.0016645216840378563] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'recent_direct', 'recent_dist', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |

## Evidence row 3

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/primitive_validation_extra90.csv`

| Field | Value |
|---|---|
| name | ridge_drop_multiscale_direct_s075 |
| family | ridge_subset |
| wcv | 1.7482607225672138 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015429799909772335 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016645216840378563 |
| worst_delta | 0.0 |
| raw_delta | -0.002318059469876577 |
| offset_mean | -0.023395989128124794 |
| offset_std | 0.053885532262926375 |
| fold_scores | [1.7692437685920095, 1.7609209358906932, 1.749434660867022, 1.7418858193333404] |
| fold_deltas | [0.0, -0.0018040844046336435, -0.0015550893957720913, -0.0016645216840378563] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'recent_direct', 'recent_dist', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |

## Evidence row 4

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/STABLE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_drop_multiscale_direct_s075 |
| family | ridge_subset |
| wcv | 1.7482607225672138 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015429799909772335 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016645216840378563 |
| worst_delta | 0.0 |
| raw_delta | -0.002318059469876577 |
| offset_mean | -0.023395989128124794 |
| offset_std | 0.053885532262926375 |
| fold_scores | [1.7692437685920095, 1.7609209358906932, 1.749434660867022, 1.7418858193333404] |
| fold_deltas | [0.0, -0.0018040844046336435, -0.0015550893957720913, -0.0016645216840378563] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'recent_direct', 'recent_dist', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |

## Evidence row 5

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_drop_multiscale_direct_s075 |
| family | ridge_subset |
| wcv | 1.7482607225672138 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015429799909772335 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016645216840378563 |
| worst_delta | 0.0 |
| raw_delta | -0.002318059469876577 |
| offset_mean | -0.023395989128124794 |
| offset_std | 0.053885532262926375 |
| fold_scores | [1.7692437685920095, 1.7609209358906932, 1.749434660867022, 1.7418858193333404] |
| fold_deltas | [0.0, -0.0018040844046336435, -0.0015550893957720913, -0.0016645216840378563] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'recent_direct', 'recent_dist', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |

## Evidence row 6

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/combo_validation_final6h_base.csv`

| Field | Value |
|---|---|
| name | ridge_drop_multiscale_direct_s075 |
| family | ridge_subset |
| wcv | 1.7482607225672138 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015429799909772335 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016645216840378563 |
| worst_delta | 0.0 |
| raw_delta | -0.002318059469876577 |
| offset_mean | -0.023395989128124794 |
| offset_std | 0.053885532262926375 |
| fold_scores | [1.7692437685920095, 1.7609209358906932, 1.749434660867022, 1.7418858193333404] |
| fold_deltas | [0.0, -0.0018040844046336435, -0.0015550893957720913, -0.0016645216840378563] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'recent_direct', 'recent_dist', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |

## Evidence row 7

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/primitive_validation_final6h_base.csv`

| Field | Value |
|---|---|
| name | ridge_drop_multiscale_direct_s075 |
| family | ridge_subset |
| wcv | 1.7482607225672138 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015429799909772335 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016645216840378563 |
| worst_delta | 0.0 |
| raw_delta | -0.002318059469876577 |
| offset_mean | -0.023395989128124794 |
| offset_std | 0.053885532262926375 |
| fold_scores | [1.7692437685920095, 1.7609209358906932, 1.749434660867022, 1.7418858193333404] |
| fold_deltas | [0.0, -0.0018040844046336435, -0.0015550893957720913, -0.0016645216840378563] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'recent_direct', 'recent_dist', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |

## Evidence row 8

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/STABLE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_drop_multiscale_direct_s075 |
| family | ridge_subset |
| wcv | 1.7482607225672138 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015429799909772335 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016645216840378563 |
| worst_delta | 0.0 |
| raw_delta | -0.002318059469876577 |
| offset_mean | -0.023395989128124794 |
| offset_std | 0.053885532262926375 |
| fold_scores | [1.7692437685920095, 1.7609209358906932, 1.749434660867022, 1.7418858193333404] |
| fold_deltas | [0.0, -0.0018040844046336435, -0.0015550893957720913, -0.0016645216840378563] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'recent_direct', 'recent_dist', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |

## Evidence row 9

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/ALL_VALIDATION.csv`

| Field | Value |
|---|---|
| name | ridge_drop_multiscale_direct_s075 |
| family | ridge_subset |
| wcv | 1.7482607225672138 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015429799909772335 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016645216840378563 |
| worst_delta | 0.0 |
| raw_delta | -0.002318059469876577 |
| offset_mean | -0.023395989128124794 |
| offset_std | 0.053885532262926375 |
| fold_scores | [1.7692437685920095, 1.7609209358906932, 1.749434660867022, 1.7418858193333404] |
| fold_deltas | [0.0, -0.0018040844046336435, -0.0015550893957720913, -0.0016645216840378563] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'recent_direct', 'recent_dist', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |

## Evidence row 10

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_fast12.csv`

| Field | Value |
|---|---|
| name | ridge_drop_multiscale_direct_s075 |
| family | ridge_subset |
| wcv | 1.748260374632766 |
| base_wcv | 1.749803702558191 |
| delta | -0.001543327925425094 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016731667473253165 |
| worst_delta | 0.0 |
| raw_delta | -0.0023209726221632937 |
| offset_mean | -0.023515657221735868 |
| offset_std | 0.05376359171543439 |
| fold_scores | [1.7692437685920095, 1.7609159761824564, 1.7494531260935358, 1.741877174270053] |
| fold_deltas | [0.0, -0.0018090441128704171, -0.0015366241692582605, -0.0016731667473253165] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'recent_direct', 'recent_dist', 'recent_hurdle_fast12'] |

## Evidence row 11

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_stable18.csv`

| Field | Value |
|---|---|
| name | ridge_drop_multiscale_direct_s075 |
| family | ridge_subset |
| wcv | 1.7482607225672138 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015429799909772335 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016645216840378563 |
| worst_delta | 0.0 |
| raw_delta | -0.002318059469876577 |
| offset_mean | -0.023395989128124794 |
| offset_std | 0.053885532262926375 |
| fold_scores | [1.7692437685920095, 1.7609209358906932, 1.749434660867022, 1.7418858193333404] |
| fold_deltas | [0.0, -0.0018040844046336435, -0.0015550893957720913, -0.0016645216840378563] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'recent_direct', 'recent_dist', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |

## Evidence row 12

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_existing.csv`

| Field | Value |
|---|---|
| name | ridge_drop_multiscale_direct_s075 |
| family | ridge_subset |
| wcv | 1.7483424128794964 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014612896786945662 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0015385927419104828 |
| worst_delta | 0.0 |
| raw_delta | -0.0022385692101918172 |
| offset_mean | -0.023663959877756616 |
| offset_std | 0.05369046611663058 |
| fold_scores | [1.7692437685920095, 1.7608996508803096, 1.749499784159019, 1.7420117482754678] |
| fold_deltas | [0.0, -0.0018253694150172084, -0.0014899661037750533, -0.0015385927419104828] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'recent_direct', 'recent_dist'] |

## Evidence row 13

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final.csv`

| Field | Value |
|---|---|
| name | ridge_drop_multiscale_direct_s075 |
| family | ridge_subset |
| wcv | 1.7482607225672138 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015429799909772335 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016645216840378563 |
| worst_delta | 0.0 |
| raw_delta | -0.002318059469876577 |
| offset_mean | -0.023395989128124794 |
| offset_std | 0.053885532262926375 |
| fold_scores | [1.7692437685920095, 1.7609209358906932, 1.749434660867022, 1.7418858193333404] |
| fold_deltas | [0.0, -0.0018040844046336435, -0.0015550893957720913, -0.0016645216840378563] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'recent_direct', 'recent_dist', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |

## Evidence row 14

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final_reload.csv`

| Field | Value |
|---|---|
| name | ridge_drop_multiscale_direct_s075 |
| family | ridge_subset |
| wcv | 1.7482607225672138 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015429799909772335 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016645216840378563 |
| worst_delta | 0.0 |
| raw_delta | -0.002318059469876577 |
| offset_mean | -0.023395989128124794 |
| offset_std | 0.053885532262926375 |
| fold_scores | [1.7692437685920095, 1.7609209358906932, 1.749434660867022, 1.7418858193333404] |
| fold_deltas | [0.0, -0.0018040844046336435, -0.0015550893957720913, -0.0016645216840378563] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'recent_direct', 'recent_dist', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |

## Evidence row 15

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_after_recent_hurdle_fast12.csv`

| Field | Value |
|---|---|
| name | ridge_drop_multiscale_direct_s075 |
| family | ridge_subset |
| wcv | 1.748260374632766 |
| base_wcv | 1.749803702558191 |
| delta | -0.001543327925425094 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016731667473253165 |
| worst_delta | 0.0 |
| raw_delta | -0.0023209726221632937 |
| offset_mean | -0.023515657221735868 |
| offset_std | 0.05376359171543439 |
| fold_scores | [1.7692437685920095, 1.7609159761824564, 1.7494531260935358, 1.741877174270053] |
| fold_deltas | [0.0, -0.0018090441128704171, -0.0015366241692582605, -0.0016731667473253165] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'recent_direct', 'recent_dist', 'recent_hurdle_fast12'] |

## Evidence row 16

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_after_recent_hurdle_stable18.csv`

| Field | Value |
|---|---|
| name | ridge_drop_multiscale_direct_s075 |
| family | ridge_subset |
| wcv | 1.7482607225672138 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015429799909772335 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016645216840378563 |
| worst_delta | 0.0 |
| raw_delta | -0.002318059469876577 |
| offset_mean | -0.023395989128124794 |
| offset_std | 0.053885532262926375 |
| fold_scores | [1.7692437685920095, 1.7609209358906932, 1.749434660867022, 1.7418858193333404] |
| fold_deltas | [0.0, -0.0018040844046336435, -0.0015550893957720913, -0.0016645216840378563] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'recent_direct', 'recent_dist', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |

## Evidence row 17

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_existing.csv`

| Field | Value |
|---|---|
| name | ridge_drop_multiscale_direct_s075 |
| family | ridge_subset |
| wcv | 1.7483424128794964 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014612896786945662 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0015385927419104828 |
| worst_delta | 0.0 |
| raw_delta | -0.0022385692101918172 |
| offset_mean | -0.023663959877756616 |
| offset_std | 0.05369046611663058 |
| fold_scores | [1.7692437685920095, 1.7608996508803096, 1.749499784159019, 1.7420117482754678] |
| fold_deltas | [0.0, -0.0018253694150172084, -0.0014899661037750533, -0.0015385927419104828] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'recent_direct', 'recent_dist'] |

## Evidence row 18

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_final.csv`

| Field | Value |
|---|---|
| name | ridge_drop_multiscale_direct_s075 |
| family | ridge_subset |
| wcv | 1.7482607225672138 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015429799909772335 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016645216840378563 |
| worst_delta | 0.0 |
| raw_delta | -0.002318059469876577 |
| offset_mean | -0.023395989128124794 |
| offset_std | 0.053885532262926375 |
| fold_scores | [1.7692437685920095, 1.7609209358906932, 1.749434660867022, 1.7418858193333404] |
| fold_deltas | [0.0, -0.0018040844046336435, -0.0015550893957720913, -0.0016645216840378563] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'recent_direct', 'recent_dist', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |

## Evidence row 19

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/primitive_validation_final_reload.csv`

| Field | Value |
|---|---|
| name | ridge_drop_multiscale_direct_s075 |
| family | ridge_subset |
| wcv | 1.7482607225672138 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015429799909772335 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016645216840378563 |
| worst_delta | 0.0 |
| raw_delta | -0.002318059469876577 |
| offset_mean | -0.023395989128124794 |
| offset_std | 0.053885532262926375 |
| fold_scores | [1.7692437685920095, 1.7609209358906932, 1.749434660867022, 1.7418858193333404] |
| fold_deltas | [0.0, -0.0018040844046336435, -0.0015550893957720913, -0.0016645216840378563] |
| notes | experts=['cap', 'unc', 'dist', 'hurdle', 'recent_direct', 'recent_dist', 'recent_hurdle_fast12', 'recent_hurdle_stable18'] |
