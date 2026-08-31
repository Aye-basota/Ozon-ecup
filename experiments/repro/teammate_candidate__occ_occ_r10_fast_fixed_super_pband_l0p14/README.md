# Teammate candidate — occ_occ_r10_fast_fixed__super_pband_l0p14

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__occ_occ_r10_fast_fixed_super_pband_l0p14`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `occ_occ_r10_fast_fixed__super_pband_l0p14`
- **Original source:** `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Unknown / not recoverable from repository history
- **Features:** occurrence features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`
- **Known score:** | wcv | 1.748421553905575 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 2 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — occ_occ_r10_fast_fixed__super_pband_l0p14

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`

| Field | Value |
|---|---|
| name | occ_occ_r10_fast_fixed__super_pband_l0p14 |
| family | occurrence_overlay |
| wcv | 1.748421553905575 |
| base_wcv | 1.749803702558191 |
| delta | -0.0013821486526161071 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001612950548770975 |
| worst_delta | 0.00023607115140222845 |
| raw_delta | -0.002176236910935847 |
| offset_mean | -0.00825814766931072 |
| offset_std | 0.05158149392100334 |
| fold_scores | [1.7694798397434117, 1.7612614482014473, 1.7497053621721148, 1.7419373904686073] |
| fold_deltas | [0.00023607115140222845, -0.0014635720938795949, -0.0012843880906792116, -0.001612950548770975] |
| notes | raw=occ_r10_fast;base=super_pband_l0p14;adaptive=False |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | occ_occ_r10_fast_fixed__super_pband_l0p14 |
| family | occurrence_overlay |
| wcv | 1.748421553905575 |
| base_wcv | 1.749803702558191 |
| delta | -0.0013821486526161071 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001612950548770975 |
| worst_delta | 0.00023607115140222845 |
| raw_delta | -0.002176236910935847 |
| offset_mean | -0.00825814766931072 |
| offset_std | 0.05158149392100334 |
| fold_scores | [1.7694798397434117, 1.7612614482014473, 1.7497053621721148, 1.7419373904686073] |
| fold_deltas | [0.00023607115140222845, -0.0014635720938795949, -0.0012843880906792116, -0.001612950548770975] |
| notes | raw=occ_r10_fast;base=super_pband_l0p14;adaptive=False |
