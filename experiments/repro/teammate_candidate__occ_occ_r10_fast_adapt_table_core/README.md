# Teammate candidate — occ_occ_r10_fast_adapt__table_core

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__occ_occ_r10_fast_adapt_table_core`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `occ_occ_r10_fast_adapt__table_core`
- **Original source:** `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Unknown / not recoverable from repository history
- **Features:** occurrence features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`
- **Known score:** | wcv | 1.7496529312046991 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 2 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — occ_occ_r10_fast_adapt__table_core

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`

| Field | Value |
|---|---|
| name | occ_occ_r10_fast_adapt__table_core |
| family | occurrence_overlay |
| wcv | 1.7496529312046991 |
| base_wcv | 1.749803702558191 |
| delta | -0.00015077135349192972 |
| wins | 2 |
| wins_recent | 2 |
| latest_delta | -0.00021464393008052696 |
| worst_delta | 0.00023607115140222845 |
| raw_delta | -0.001015541803265855 |
| offset_mean | -0.012575110464038745 |
| offset_std | 0.0481370745040157 |
| fold_scores | [1.7694798397434117, 1.7628419995415776, 1.7507361381363844, 1.7433356970872977] |
| fold_deltas | [0.00023607115140222845, 0.00011697924625075906, -0.00025361212640961917, -0.00021464393008052696] |
| notes | raw=occ_r10_fast;base=table_core;adaptive=True |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | occ_occ_r10_fast_adapt__table_core |
| family | occurrence_overlay |
| wcv | 1.7496529312046991 |
| base_wcv | 1.749803702558191 |
| delta | -0.00015077135349192972 |
| wins | 2 |
| wins_recent | 2 |
| latest_delta | -0.00021464393008052696 |
| worst_delta | 0.00023607115140222845 |
| raw_delta | -0.001015541803265855 |
| offset_mean | -0.012575110464038745 |
| offset_std | 0.0481370745040157 |
| fold_scores | [1.7694798397434117, 1.7628419995415776, 1.7507361381363844, 1.7433356970872977] |
| fold_deltas | [0.00023607115140222845, 0.00011697924625075906, -0.00025361212640961917, -0.00021464393008052696] |
| notes | raw=occ_r10_fast;base=table_core;adaptive=True |
