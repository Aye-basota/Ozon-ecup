# Teammate candidate — occ_occ_r22_stable_fixed__table_core

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__occ_occ_r22_stable_fixed_table_core`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `occ_occ_r22_stable_fixed__table_core`
- **Original source:** `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Unknown / not recoverable from repository history
- **Features:** occurrence features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`
- **Known score:** | wcv | 1.7499925031463213 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 2 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — occ_occ_r22_stable_fixed__table_core

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`

| Field | Value |
|---|---|
| name | occ_occ_r22_stable_fixed__table_core |
| family | occurrence_overlay |
| wcv | 1.7499925031463213 |
| base_wcv | 1.749803702558191 |
| delta | 0.0001888005881303408 |
| wins | 0 |
| wins_recent | 0 |
| latest_delta | 0.00023983542427252758 |
| worst_delta | 0.00023983542427252758 |
| raw_delta | -0.00028325069982955143 |
| offset_mean | -0.054038192540588365 |
| offset_std | 0.027981202000366003 |
| fold_scores | [1.7692589866960031, 1.7629034434515614, 1.751125065515622, 1.7437901764416508] |
| fold_deltas | [1.521810399363055e-05, 0.00017842315623450844, 0.000135315252828061, 0.00023983542427252758] |
| notes | raw=occ_r22_stable;base=table_core;adaptive=False |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | occ_occ_r22_stable_fixed__table_core |
| family | occurrence_overlay |
| wcv | 1.7499925031463213 |
| base_wcv | 1.749803702558191 |
| delta | 0.0001888005881303408 |
| wins | 0 |
| wins_recent | 0 |
| latest_delta | 0.00023983542427252758 |
| worst_delta | 0.00023983542427252758 |
| raw_delta | -0.00028325069982955143 |
| offset_mean | -0.054038192540588365 |
| offset_std | 0.027981202000366003 |
| fold_scores | [1.7692589866960031, 1.7629034434515614, 1.751125065515622, 1.7437901764416508] |
| fold_deltas | [1.521810399363055e-05, 0.00017842315623450844, 0.000135315252828061, 0.00023983542427252758] |
| notes | raw=occ_r22_stable;base=table_core;adaptive=False |
