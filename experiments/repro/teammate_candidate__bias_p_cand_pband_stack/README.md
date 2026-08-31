# Teammate candidate — bias_p_cand_pband_stack

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__bias_p_cand_pband_stack`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `bias_p_cand_pband_stack`
- **Original source:** `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/combo_validation_extra90.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/STABLE_BRANCH_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/combo_validation_final6h_base.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/STABLE_BRANCH_VALIDATION.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/ALL_VALIDATION.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_fast12.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_stable18.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_existing.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final_reload.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/SHORTLIST_VALIDATION.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** two-part / hurdle
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`
- **Known score:** | wcv | 1.748171364580743 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 13 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — bias_p_cand_pband_stack

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`

| Field | Value |
|---|---|
| name | bias_p_cand_pband_stack |
| family | local_bias |
| wcv | 1.748171364580743 |
| base_wcv | 1.749803702558191 |
| delta | -0.0016323379774480173 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0017633092058610966 |
| worst_delta | 0.0 |
| raw_delta | -0.002389670664101118 |
| offset_mean | -0.029086178181156424 |
| offset_std | 0.05075206541112006 |
| fold_scores | [1.7692437685920095, 1.7608825806298283, 1.7493163210918354, 1.7417870318115172] |
| fold_deltas | [0.0, -0.0018424396654985742, -0.0016734291709585847, -0.0017633092058610966] |
| notes |  |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/combo_validation_extra90.csv`

| Field | Value |
|---|---|
| name | bias_p_cand_pband_stack |
| family | local_bias |
| wcv | 1.748171364580743 |
| base_wcv | 1.749803702558191 |
| delta | -0.0016323379774480173 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0017633092058610966 |
| worst_delta | 0.0 |
| raw_delta | -0.002389670664101118 |
| offset_mean | -0.029086178181156424 |
| offset_std | 0.05075206541112006 |
| fold_scores | [1.7692437685920095, 1.7608825806298283, 1.7493163210918354, 1.7417870318115172] |
| fold_deltas | [0.0, -0.0018424396654985742, -0.0016734291709585847, -0.0017633092058610966] |
| notes |  |

## Evidence row 3

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/STABLE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | bias_p_cand_pband_stack |
| family | local_bias |
| wcv | 1.748171364580743 |
| base_wcv | 1.749803702558191 |
| delta | -0.0016323379774480173 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0017633092058610966 |
| worst_delta | 0.0 |
| raw_delta | -0.002389670664101118 |
| offset_mean | -0.029086178181156424 |
| offset_std | 0.05075206541112006 |
| fold_scores | [1.7692437685920095, 1.7608825806298283, 1.7493163210918354, 1.7417870318115172] |
| fold_deltas | [0.0, -0.0018424396654985742, -0.0016734291709585847, -0.0017633092058610966] |
| notes |  |

## Evidence row 4

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`

| Field | Value |
|---|---|
| name | bias_p_cand_pband_stack |
| family | local_bias |
| wcv | 1.748171364580743 |
| base_wcv | 1.749803702558191 |
| delta | -0.0016323379774480173 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0017633092058610966 |
| worst_delta | 0.0 |
| raw_delta | -0.002389670664101118 |
| offset_mean | -0.029086178181156424 |
| offset_std | 0.05075206541112006 |
| fold_scores | [1.7692437685920095, 1.7608825806298283, 1.7493163210918354, 1.7417870318115172] |
| fold_deltas | [0.0, -0.0018424396654985742, -0.0016734291709585847, -0.0017633092058610966] |
| notes |  |

## Evidence row 5

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/combo_validation_final6h_base.csv`

| Field | Value |
|---|---|
| name | bias_p_cand_pband_stack |
| family | local_bias |
| wcv | 1.748171364580743 |
| base_wcv | 1.749803702558191 |
| delta | -0.0016323379774480173 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0017633092058610966 |
| worst_delta | 0.0 |
| raw_delta | -0.002389670664101118 |
| offset_mean | -0.029086178181156424 |
| offset_std | 0.05075206541112006 |
| fold_scores | [1.7692437685920095, 1.7608825806298283, 1.7493163210918354, 1.7417870318115172] |
| fold_deltas | [0.0, -0.0018424396654985742, -0.0016734291709585847, -0.0017633092058610966] |
| notes |  |

## Evidence row 6

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/STABLE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | bias_p_cand_pband_stack |
| family | local_bias |
| wcv | 1.748171364580743 |
| base_wcv | 1.749803702558191 |
| delta | -0.0016323379774480173 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0017633092058610966 |
| worst_delta | 0.0 |
| raw_delta | -0.002389670664101118 |
| offset_mean | -0.029086178181156424 |
| offset_std | 0.05075206541112006 |
| fold_scores | [1.7692437685920095, 1.7608825806298283, 1.7493163210918354, 1.7417870318115172] |
| fold_deltas | [0.0, -0.0018424396654985742, -0.0016734291709585847, -0.0017633092058610966] |
| notes |  |

## Evidence row 7

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/ALL_VALIDATION.csv`

| Field | Value |
|---|---|
| name | bias_p_cand_pband_stack |
| family | local_bias |
| wcv | 1.748171364580743 |
| base_wcv | 1.749803702558191 |
| delta | -0.0016323379774480173 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0017633092058610966 |
| worst_delta | 0.0 |
| raw_delta | -0.002389670664101118 |
| offset_mean | -0.029086178181156424 |
| offset_std | 0.05075206541112006 |
| fold_scores | [1.7692437685920095, 1.7608825806298283, 1.7493163210918354, 1.7417870318115172] |
| fold_deltas | [0.0, -0.0018424396654985742, -0.0016734291709585847, -0.0017633092058610966] |
| notes |  |

## Evidence row 8

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_fast12.csv`

| Field | Value |
|---|---|
| name | bias_p_cand_pband_stack |
| family | local_bias |
| wcv | 1.748161233917068 |
| base_wcv | 1.749803702558191 |
| delta | -0.0016424686411230737 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0017683585383232447 |
| worst_delta | 0.0 |
| raw_delta | -0.0024005978732854296 |
| offset_mean | -0.02906235368647811 |
| offset_std | 0.050749603749082016 |
| fold_scores | [1.7692437685920095, 1.760867274453863, 1.749296082855961, 1.741781982479055] |
| fold_deltas | [0.0, -0.0018577458414639203, -0.001693667406833077, -0.0017683585383232447] |
| notes |  |

## Evidence row 9

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_stable18.csv`

| Field | Value |
|---|---|
| name | bias_p_cand_pband_stack |
| family | local_bias |
| wcv | 1.748171364580743 |
| base_wcv | 1.749803702558191 |
| delta | -0.0016323379774480173 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0017633092058610966 |
| worst_delta | 0.0 |
| raw_delta | -0.002389670664101118 |
| offset_mean | -0.029086178181156424 |
| offset_std | 0.05075206541112006 |
| fold_scores | [1.7692437685920095, 1.7608825806298283, 1.7493163210918354, 1.7417870318115172] |
| fold_deltas | [0.0, -0.0018424396654985742, -0.0016734291709585847, -0.0017633092058610966] |
| notes |  |

## Evidence row 10

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_existing.csv`

| Field | Value |
|---|---|
| name | bias_p_cand_pband_stack |
| family | local_bias |
| wcv | 1.7482663833109626 |
| base_wcv | 1.749803702558191 |
| delta | -0.001537319247228434 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016145509898486399 |
| worst_delta | 0.0 |
| raw_delta | -0.0022945098439642506 |
| offset_mean | -0.029427522449362135 |
| offset_std | 0.0505203066543704 |
| fold_scores | [1.7692437685920095, 1.7608708226525196, 1.7493810038867883, 1.7419357900275296] |
| fold_deltas | [0.0, -0.0018541976428072537, -0.0016087463760057208, -0.0016145509898486399] |
| notes |  |

## Evidence row 11

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final.csv`

| Field | Value |
|---|---|
| name | bias_p_cand_pband_stack |
| family | local_bias |
| wcv | 1.748171364580743 |
| base_wcv | 1.749803702558191 |
| delta | -0.0016323379774480173 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0017633092058610966 |
| worst_delta | 0.0 |
| raw_delta | -0.002389670664101118 |
| offset_mean | -0.029086178181156424 |
| offset_std | 0.05075206541112006 |
| fold_scores | [1.7692437685920095, 1.7608825806298283, 1.7493163210918354, 1.7417870318115172] |
| fold_deltas | [0.0, -0.0018424396654985742, -0.0016734291709585847, -0.0017633092058610966] |
| notes |  |

## Evidence row 12

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final_reload.csv`

| Field | Value |
|---|---|
| name | bias_p_cand_pband_stack |
| family | local_bias |
| wcv | 1.748171364580743 |
| base_wcv | 1.749803702558191 |
| delta | -0.0016323379774480173 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0017633092058610966 |
| worst_delta | 0.0 |
| raw_delta | -0.002389670664101118 |
| offset_mean | -0.029086178181156424 |
| offset_std | 0.05075206541112006 |
| fold_scores | [1.7692437685920095, 1.7608825806298283, 1.7493163210918354, 1.7417870318115172] |
| fold_deltas | [0.0, -0.0018424396654985742, -0.0016734291709585847, -0.0017633092058610966] |
| notes |  |

## Evidence row 13

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/SHORTLIST_VALIDATION.csv`

| Field | Value |
|---|---|
| name | bias_p_cand_pband_stack |
| family | local_bias |
| wcv | 1.748171364580743 |
| base_wcv | 1.749803702558191 |
| delta | -0.0016323379774480173 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0017633092058610966 |
| worst_delta | 0.0 |
| raw_delta | -0.002389670664101118 |
| offset_mean | -0.029086178181156424 |
| offset_std | 0.05075206541112006 |
| fold_scores | [1.7692437685920095, 1.7608825806298283, 1.7493163210918354, 1.7417870318115172] |
| fold_deltas | [0.0, -0.0018424396654985742, -0.0016734291709585847, -0.0017633092058610966] |
| notes |  |
