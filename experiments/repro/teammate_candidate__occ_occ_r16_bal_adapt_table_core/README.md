# Teammate candidate — occ_occ_r16_bal_adapt__table_core

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__occ_occ_r16_bal_adapt_table_core`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `occ_occ_r16_bal_adapt__table_core`
- **Original source:** `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Unknown / not recoverable from repository history
- **Features:** occurrence features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`
- **Known score:** | wcv | 1.74980754702804 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 2 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — occ_occ_r16_bal_adapt__table_core

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`

| Field | Value |
|---|---|
| name | occ_occ_r16_bal_adapt__table_core |
| family | occurrence_overlay |
| wcv | 1.74980754702804 |
| base_wcv | 1.749803702558191 |
| delta | 3.844469849016245e-06 |
| wins | 1 |
| wins_recent | 1 |
| latest_delta | -5.863123435778661e-05 |
| worst_delta | 0.00015298417706111067 |
| raw_delta | -0.0008416956356468368 |
| offset_mean | -0.01318395517365369 |
| offset_std | 0.05039992444035932 |
| fold_scores | [1.7693134296601989, 1.762878004472388, 1.7510275221378655, 1.7434917097830205] |
| fold_deltas | [6.966106818939721e-05, 0.00015298417706111067, 3.77718750714795e-05, -5.863123435778661e-05] |
| notes | raw=occ_r16_bal;base=table_core;adaptive=True |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | occ_occ_r16_bal_adapt__table_core |
| family | occurrence_overlay |
| wcv | 1.74980754702804 |
| base_wcv | 1.749803702558191 |
| delta | 3.844469849016245e-06 |
| wins | 1 |
| wins_recent | 1 |
| latest_delta | -5.863123435778661e-05 |
| worst_delta | 0.00015298417706111067 |
| raw_delta | -0.0008416956356468368 |
| offset_mean | -0.01318395517365369 |
| offset_std | 0.05039992444035932 |
| fold_scores | [1.7693134296601989, 1.762878004472388, 1.7510275221378655, 1.7434917097830205] |
| fold_deltas | [6.966106818939721e-05, 0.00015298417706111067, 3.77718750714795e-05, -5.863123435778661e-05] |
| notes | raw=occ_r16_bal;base=table_core;adaptive=True |
