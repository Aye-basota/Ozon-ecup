# Teammate candidate — cand_pband_stack

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__cand_pband_stack`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `cand_pband_stack`
- **Original source:** `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/combo_validation_extra90.csv | пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/STABLE_BRANCH_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/combo_validation_final6h_base.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/FAMILY_BEST.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/STABLE_BRANCH_VALIDATION.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/ALL_VALIDATION.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_fast12.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_stable18.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_existing.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final_reload.csv | пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/SHORTLIST_VALIDATION.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** two-part / hurdle
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`
- **Known score:** | wcv | 1.7481751650423416 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 14 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — cand_pband_stack

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`

| Field | Value |
|---|---|
| name | cand_pband_stack |
| family | candidate_pband |
| wcv | 1.7481751650423416 |
| base_wcv | 1.749803702558191 |
| delta | -0.0016285375158494079 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0017811694816183277 |
| worst_delta | 0.0 |
| raw_delta | -0.0023862428420028023 |
| offset_mean | -0.028808447769434038 |
| offset_std | 0.05092850593128465 |
| fold_scores | [1.7692437685920095, 1.7609406294275092, 1.7493372689755042, 1.74176917153576] |
| fold_deltas | [0.0, -0.0017843908678176756, -0.0016524812872897865, -0.0017811694816183277] |
| notes |  |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/combo_validation_extra90.csv`

| Field | Value |
|---|---|
| name | cand_pband_stack |
| family | candidate_pband |
| wcv | 1.7481751650423416 |
| base_wcv | 1.749803702558191 |
| delta | -0.0016285375158494079 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0017811694816183277 |
| worst_delta | 0.0 |
| raw_delta | -0.0023862428420028023 |
| offset_mean | -0.028808447769434038 |
| offset_std | 0.05092850593128465 |
| fold_scores | [1.7692437685920095, 1.7609406294275092, 1.7493372689755042, 1.74176917153576] |
| fold_deltas | [0.0, -0.0017843908678176756, -0.0016524812872897865, -0.0017811694816183277] |
| notes |  |

## Evidence row 3

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/STABLE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | cand_pband_stack |
| family | candidate_pband |
| wcv | 1.7481751650423416 |
| base_wcv | 1.749803702558191 |
| delta | -0.0016285375158494079 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0017811694816183277 |
| worst_delta | 0.0 |
| raw_delta | -0.0023862428420028023 |
| offset_mean | -0.028808447769434038 |
| offset_std | 0.05092850593128465 |
| fold_scores | [1.7692437685920095, 1.7609406294275092, 1.7493372689755042, 1.74176917153576] |
| fold_deltas | [0.0, -0.0017843908678176756, -0.0016524812872897865, -0.0017811694816183277] |
| notes |  |

## Evidence row 4

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`

| Field | Value |
|---|---|
| name | cand_pband_stack |
| family | candidate_pband |
| wcv | 1.7481751650423416 |
| base_wcv | 1.749803702558191 |
| delta | -0.0016285375158494079 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0017811694816183277 |
| worst_delta | 0.0 |
| raw_delta | -0.0023862428420028023 |
| offset_mean | -0.028808447769434038 |
| offset_std | 0.05092850593128465 |
| fold_scores | [1.7692437685920095, 1.7609406294275092, 1.7493372689755042, 1.74176917153576] |
| fold_deltas | [0.0, -0.0017843908678176756, -0.0016524812872897865, -0.0017811694816183277] |
| notes |  |

## Evidence row 5

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/combo_validation_final6h_base.csv`

| Field | Value |
|---|---|
| name | cand_pband_stack |
| family | candidate_pband |
| wcv | 1.7481751650423416 |
| base_wcv | 1.749803702558191 |
| delta | -0.0016285375158494079 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0017811694816183277 |
| worst_delta | 0.0 |
| raw_delta | -0.0023862428420028023 |
| offset_mean | -0.028808447769434038 |
| offset_std | 0.05092850593128465 |
| fold_scores | [1.7692437685920095, 1.7609406294275092, 1.7493372689755042, 1.74176917153576] |
| fold_deltas | [0.0, -0.0017843908678176756, -0.0016524812872897865, -0.0017811694816183277] |
| notes |  |

## Evidence row 6

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/FAMILY_BEST.csv`

| Field | Value |
|---|---|
| family | candidate_pband |
| best_name | cand_pband_stack |
| delta | -0.0016285375158494079 |
| wins_recent | 3 |
| latest_delta | -0.0017811694816183277 |

## Evidence row 7

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/STABLE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | cand_pband_stack |
| family | candidate_pband |
| wcv | 1.7481751650423416 |
| base_wcv | 1.749803702558191 |
| delta | -0.0016285375158494079 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0017811694816183277 |
| worst_delta | 0.0 |
| raw_delta | -0.0023862428420028023 |
| offset_mean | -0.028808447769434038 |
| offset_std | 0.05092850593128465 |
| fold_scores | [1.7692437685920095, 1.7609406294275092, 1.7493372689755042, 1.74176917153576] |
| fold_deltas | [0.0, -0.0017843908678176756, -0.0016524812872897865, -0.0017811694816183277] |
| notes |  |

## Evidence row 8

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/ALL_VALIDATION.csv`

| Field | Value |
|---|---|
| name | cand_pband_stack |
| family | candidate_pband |
| wcv | 1.7481751650423416 |
| base_wcv | 1.749803702558191 |
| delta | -0.0016285375158494079 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0017811694816183277 |
| worst_delta | 0.0 |
| raw_delta | -0.0023862428420028023 |
| offset_mean | -0.028808447769434038 |
| offset_std | 0.05092850593128465 |
| fold_scores | [1.7692437685920095, 1.7609406294275092, 1.7493372689755042, 1.74176917153576] |
| fold_deltas | [0.0, -0.0017843908678176756, -0.0016524812872897865, -0.0017811694816183277] |
| notes |  |

## Evidence row 9

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_fast12.csv`

| Field | Value |
|---|---|
| name | cand_pband_stack |
| family | candidate_pband |
| wcv | 1.7481648633547773 |
| base_wcv | 1.749803702558191 |
| delta | -0.0016388392034137667 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0017865173721591798 |
| worst_delta | 0.0 |
| raw_delta | -0.0023973336933982735 |
| offset_mean | -0.02878459787848895 |
| offset_std | 0.050926412678527586 |
| fold_scores | [1.7692437685920095, 1.7609215852762823, 1.749318855503833, 1.741763823645219] |
| fold_deltas | [0.0, -0.0018034350190445636, -0.0016708947589609835, -0.0017865173721591798] |
| notes |  |

## Evidence row 10

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_after_recent_hurdle_stable18.csv`

| Field | Value |
|---|---|
| name | cand_pband_stack |
| family | candidate_pband |
| wcv | 1.7481751650423416 |
| base_wcv | 1.749803702558191 |
| delta | -0.0016285375158494079 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0017811694816183277 |
| worst_delta | 0.0 |
| raw_delta | -0.0023862428420028023 |
| offset_mean | -0.028808447769434038 |
| offset_std | 0.05092850593128465 |
| fold_scores | [1.7692437685920095, 1.7609406294275092, 1.7493372689755042, 1.74176917153576] |
| fold_deltas | [0.0, -0.0017843908678176756, -0.0016524812872897865, -0.0017811694816183277] |
| notes |  |

## Evidence row 11

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_existing.csv`

| Field | Value |
|---|---|
| name | cand_pband_stack |
| family | candidate_pband |
| wcv | 1.7482705892914419 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015331132667488609 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016328848723803358 |
| worst_delta | 0.0 |
| raw_delta | -0.002290766803055444 |
| offset_mean | -0.02914794972509797 |
| offset_std | 0.05069843955237111 |
| fold_scores | [1.7692437685920095, 1.760923719421542, 1.749406995694139, 1.741917456144998] |
| fold_deltas | [0.0, -0.001801300873784939, -0.001582754568655087, -0.0016328848723803358] |
| notes |  |

## Evidence row 12

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final.csv`

| Field | Value |
|---|---|
| name | cand_pband_stack |
| family | candidate_pband |
| wcv | 1.7481751650423416 |
| base_wcv | 1.749803702558191 |
| delta | -0.0016285375158494079 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0017811694816183277 |
| worst_delta | 0.0 |
| raw_delta | -0.0023862428420028023 |
| offset_mean | -0.028808447769434038 |
| offset_std | 0.05092850593128465 |
| fold_scores | [1.7692437685920095, 1.7609406294275092, 1.7493372689755042, 1.74176917153576] |
| fold_deltas | [0.0, -0.0017843908678176756, -0.0016524812872897865, -0.0017811694816183277] |
| notes |  |

## Evidence row 13

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/combo_validation_final_reload.csv`

| Field | Value |
|---|---|
| name | cand_pband_stack |
| family | candidate_pband |
| wcv | 1.7481751650423416 |
| base_wcv | 1.749803702558191 |
| delta | -0.0016285375158494079 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0017811694816183277 |
| worst_delta | 0.0 |
| raw_delta | -0.0023862428420028023 |
| offset_mean | -0.028808447769434038 |
| offset_std | 0.05092850593128465 |
| fold_scores | [1.7692437685920095, 1.7609406294275092, 1.7493372689755042, 1.74176917153576] |
| fold_deltas | [0.0, -0.0017843908678176756, -0.0016524812872897865, -0.0017811694816183277] |
| notes |  |

## Evidence row 14

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/SHORTLIST_VALIDATION.csv`

| Field | Value |
|---|---|
| name | cand_pband_stack |
| family | candidate_pband |
| wcv | 1.7481751650423416 |
| base_wcv | 1.749803702558191 |
| delta | -0.0016285375158494079 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0017811694816183277 |
| worst_delta | 0.0 |
| raw_delta | -0.0023862428420028023 |
| offset_mean | -0.028808447769434038 |
| offset_std | 0.05092850593128465 |
| fold_scores | [1.7692437685920095, 1.7609406294275092, 1.7493372689755042, 1.74176917153576] |
| fold_deltas | [0.0, -0.0017843908678176756, -0.0016524812872897865, -0.0017811694816183277] |
| notes |  |
