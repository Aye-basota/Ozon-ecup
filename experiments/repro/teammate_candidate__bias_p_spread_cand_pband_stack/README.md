# Teammate candidate — bias_p_spread_cand_pband_stack

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__bias_p_spread_cand_pband_stack`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `bias_p_spread_cand_pband_stack`
- **Original source:** `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/combo_validation_extra90.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/STABLE_BRANCH_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/combo_validation_final6h_base.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/FAMILY_BEST.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/STABLE_BRANCH_VALIDATION.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/ALL_VALIDATION.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_fast12.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_stable18.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_existing.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final_reload.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/SHORTLIST_VALIDATION.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** two-part / hurdle
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`
- **Known score:** | wcv | 1.748168500786178 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 14 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — bias_p_spread_cand_pband_stack

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`

| Field | Value |
|---|---|
| name | bias_p_spread_cand_pband_stack |
| family | local_bias |
| wcv | 1.748168500786178 |
| base_wcv | 1.749803702558191 |
| delta | -0.0016352017720130323 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001778654917288458 |
| worst_delta | 0.0 |
| raw_delta | -0.0023930015715965722 |
| offset_mean | -0.029283748159694492 |
| offset_std | 0.05060871656005681 |
| fold_scores | [1.7692437685920095, 1.7608813038917057, 1.7493369116541326, 1.7417716861000898] |
| fold_deltas | [0.0, -0.0018437164036211318, -0.0016528386086613889, -0.001778654917288458] |
| notes |  |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/combo_validation_extra90.csv`

| Field | Value |
|---|---|
| name | bias_p_spread_cand_pband_stack |
| family | local_bias |
| wcv | 1.748168500786178 |
| base_wcv | 1.749803702558191 |
| delta | -0.0016352017720130323 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001778654917288458 |
| worst_delta | 0.0 |
| raw_delta | -0.0023930015715965722 |
| offset_mean | -0.029283748159694492 |
| offset_std | 0.05060871656005681 |
| fold_scores | [1.7692437685920095, 1.7608813038917057, 1.7493369116541326, 1.7417716861000898] |
| fold_deltas | [0.0, -0.0018437164036211318, -0.0016528386086613889, -0.001778654917288458] |
| notes |  |

## Evidence row 3

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/STABLE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | bias_p_spread_cand_pband_stack |
| family | local_bias |
| wcv | 1.748168500786178 |
| base_wcv | 1.749803702558191 |
| delta | -0.0016352017720130323 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001778654917288458 |
| worst_delta | 0.0 |
| raw_delta | -0.0023930015715965722 |
| offset_mean | -0.029283748159694492 |
| offset_std | 0.05060871656005681 |
| fold_scores | [1.7692437685920095, 1.7608813038917057, 1.7493369116541326, 1.7417716861000898] |
| fold_deltas | [0.0, -0.0018437164036211318, -0.0016528386086613889, -0.001778654917288458] |
| notes |  |

## Evidence row 4

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`

| Field | Value |
|---|---|
| name | bias_p_spread_cand_pband_stack |
| family | local_bias |
| wcv | 1.748168500786178 |
| base_wcv | 1.749803702558191 |
| delta | -0.0016352017720130323 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001778654917288458 |
| worst_delta | 0.0 |
| raw_delta | -0.0023930015715965722 |
| offset_mean | -0.029283748159694492 |
| offset_std | 0.05060871656005681 |
| fold_scores | [1.7692437685920095, 1.7608813038917057, 1.7493369116541326, 1.7417716861000898] |
| fold_deltas | [0.0, -0.0018437164036211318, -0.0016528386086613889, -0.001778654917288458] |
| notes |  |

## Evidence row 5

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/combo_validation_final6h_base.csv`

| Field | Value |
|---|---|
| name | bias_p_spread_cand_pband_stack |
| family | local_bias |
| wcv | 1.748168500786178 |
| base_wcv | 1.749803702558191 |
| delta | -0.0016352017720130323 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001778654917288458 |
| worst_delta | 0.0 |
| raw_delta | -0.0023930015715965722 |
| offset_mean | -0.029283748159694492 |
| offset_std | 0.05060871656005681 |
| fold_scores | [1.7692437685920095, 1.7608813038917057, 1.7493369116541326, 1.7417716861000898] |
| fold_deltas | [0.0, -0.0018437164036211318, -0.0016528386086613889, -0.001778654917288458] |
| notes |  |

## Evidence row 6

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/FAMILY_BEST.csv`

| Field | Value |
|---|---|
| family | local_bias |
| best_name | bias_p_spread_cand_pband_stack |
| delta | -0.0016352017720130323 |
| wins_recent | 3 |
| latest_delta | -0.001778654917288458 |

## Evidence row 7

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/STABLE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | bias_p_spread_cand_pband_stack |
| family | local_bias |
| wcv | 1.748168500786178 |
| base_wcv | 1.749803702558191 |
| delta | -0.0016352017720130323 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001778654917288458 |
| worst_delta | 0.0 |
| raw_delta | -0.0023930015715965722 |
| offset_mean | -0.029283748159694492 |
| offset_std | 0.05060871656005681 |
| fold_scores | [1.7692437685920095, 1.7608813038917057, 1.7493369116541326, 1.7417716861000898] |
| fold_deltas | [0.0, -0.0018437164036211318, -0.0016528386086613889, -0.001778654917288458] |
| notes |  |

## Evidence row 8

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/ALL_VALIDATION.csv`

| Field | Value |
|---|---|
| name | bias_p_spread_cand_pband_stack |
| family | local_bias |
| wcv | 1.748168500786178 |
| base_wcv | 1.749803702558191 |
| delta | -0.0016352017720130323 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001778654917288458 |
| worst_delta | 0.0 |
| raw_delta | -0.0023930015715965722 |
| offset_mean | -0.029283748159694492 |
| offset_std | 0.05060871656005681 |
| fold_scores | [1.7692437685920095, 1.7608813038917057, 1.7493369116541326, 1.7417716861000898] |
| fold_deltas | [0.0, -0.0018437164036211318, -0.0016528386086613889, -0.001778654917288458] |
| notes |  |

## Evidence row 9

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_fast12.csv`

| Field | Value |
|---|---|
| name | bias_p_spread_cand_pband_stack |
| family | local_bias |
| wcv | 1.7481579786891848 |
| base_wcv | 1.749803702558191 |
| delta | -0.0016457238690063213 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001781952218672389 |
| worst_delta | 0.0 |
| raw_delta | -0.0024039870449407215 |
| offset_mean | -0.029287398274934543 |
| offset_std | 0.05059474476774249 |
| fold_scores | [1.7692437685920095, 1.7608614697916558, 1.7493139654432006, 1.7417683887987059] |
| fold_deltas | [0.0, -0.001863550503671041, -0.0016757848195934066, -0.001781952218672389] |
| notes |  |

## Evidence row 10

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_stable18.csv`

| Field | Value |
|---|---|
| name | bias_p_spread_cand_pband_stack |
| family | local_bias |
| wcv | 1.748168500786178 |
| base_wcv | 1.749803702558191 |
| delta | -0.0016352017720130323 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001778654917288458 |
| worst_delta | 0.0 |
| raw_delta | -0.0023930015715965722 |
| offset_mean | -0.029283748159694492 |
| offset_std | 0.05060871656005681 |
| fold_scores | [1.7692437685920095, 1.7608813038917057, 1.7493369116541326, 1.7417716861000898] |
| fold_deltas | [0.0, -0.0018437164036211318, -0.0016528386086613889, -0.001778654917288458] |
| notes |  |

## Evidence row 11

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_existing.csv`

| Field | Value |
|---|---|
| name | bias_p_spread_cand_pband_stack |
| family | local_bias |
| wcv | 1.748264656855798 |
| base_wcv | 1.749803702558191 |
| delta | -0.001539045702393042 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016257500677374637 |
| worst_delta | 0.0 |
| raw_delta | -0.002295876222025021 |
| offset_mean | -0.029628168049936836 |
| offset_std | 0.05039037645345135 |
| fold_scores | [1.7692437685920095, 1.7608637363687865, 1.7494004709775652, 1.7419245909496408] |
| fold_deltas | [0.0, -0.0018612839265403913, -0.001589279285228784, -0.0016257500677374637] |
| notes |  |

## Evidence row 12

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final.csv`

| Field | Value |
|---|---|
| name | bias_p_spread_cand_pband_stack |
| family | local_bias |
| wcv | 1.748168500786178 |
| base_wcv | 1.749803702558191 |
| delta | -0.0016352017720130323 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001778654917288458 |
| worst_delta | 0.0 |
| raw_delta | -0.0023930015715965722 |
| offset_mean | -0.029283748159694492 |
| offset_std | 0.05060871656005681 |
| fold_scores | [1.7692437685920095, 1.7608813038917057, 1.7493369116541326, 1.7417716861000898] |
| fold_deltas | [0.0, -0.0018437164036211318, -0.0016528386086613889, -0.001778654917288458] |
| notes |  |

## Evidence row 13

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final_reload.csv`

| Field | Value |
|---|---|
| name | bias_p_spread_cand_pband_stack |
| family | local_bias |
| wcv | 1.748168500786178 |
| base_wcv | 1.749803702558191 |
| delta | -0.0016352017720130323 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001778654917288458 |
| worst_delta | 0.0 |
| raw_delta | -0.0023930015715965722 |
| offset_mean | -0.029283748159694492 |
| offset_std | 0.05060871656005681 |
| fold_scores | [1.7692437685920095, 1.7608813038917057, 1.7493369116541326, 1.7417716861000898] |
| fold_deltas | [0.0, -0.0018437164036211318, -0.0016528386086613889, -0.001778654917288458] |
| notes |  |

## Evidence row 14

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/SHORTLIST_VALIDATION.csv`

| Field | Value |
|---|---|
| name | bias_p_spread_cand_pband_stack |
| family | local_bias |
| wcv | 1.748168500786178 |
| base_wcv | 1.749803702558191 |
| delta | -0.0016352017720130323 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001778654917288458 |
| worst_delta | 0.0 |
| raw_delta | -0.0023930015715965722 |
| offset_mean | -0.029283748159694492 |
| offset_std | 0.05060871656005681 |
| fold_scores | [1.7692437685920095, 1.7608813038917057, 1.7493369116541326, 1.7417716861000898] |
| fold_deltas | [0.0, -0.0018437164036211318, -0.0016528386086613889, -0.001778654917288458] |
| notes |  |
