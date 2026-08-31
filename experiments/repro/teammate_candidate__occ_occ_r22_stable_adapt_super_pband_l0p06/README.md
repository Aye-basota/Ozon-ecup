# Teammate candidate — occ_occ_r22_stable_adapt__super_pband_l0p06

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__occ_occ_r22_stable_adapt_super_pband_l0p06`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `occ_occ_r22_stable_adapt__super_pband_l0p06`
- **Original source:** `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Unknown / not recoverable from repository history
- **Features:** occurrence features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`
- **Known score:** | wcv | 1.748261022575114 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 2 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — occ_occ_r22_stable_adapt__super_pband_l0p06

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`

| Field | Value |
|---|---|
| name | occ_occ_r22_stable_adapt__super_pband_l0p06 |
| family | occurrence_overlay |
| wcv | 1.748261022575114 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015426799830768297 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0017926591607275721 |
| worst_delta | 1.521810399363055e-05 |
| raw_delta | -0.0018588960753154638 |
| offset_mean | 0.008075706542555783 |
| offset_std | 0.07294727098414965 |
| fold_scores | [1.7692589866960031, 1.7614324110735675, 1.7494325187325923, 1.7417576818566507] |
| fold_deltas | [1.521810399363055e-05, -0.0012926092217593332, -0.0015572315302017081, -0.0017926591607275721] |
| notes | raw=occ_r22_stable;base=super_pband_l0p06;adaptive=True |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | occ_occ_r22_stable_adapt__super_pband_l0p06 |
| family | occurrence_overlay |
| wcv | 1.748261022575114 |
| base_wcv | 1.749803702558191 |
| delta | -0.0015426799830768297 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0017926591607275721 |
| worst_delta | 1.521810399363055e-05 |
| raw_delta | -0.0018588960753154638 |
| offset_mean | 0.008075706542555783 |
| offset_std | 0.07294727098414965 |
| fold_scores | [1.7692589866960031, 1.7614324110735675, 1.7494325187325923, 1.7417576818566507] |
| fold_deltas | [1.521810399363055e-05, -0.0012926092217593332, -0.0015572315302017081, -0.0017926591607275721] |
| notes | raw=occ_r22_stable;base=super_pband_l0p06;adaptive=True |
