# Teammate candidate — occ_occ_r12_wide_fixed__table_core

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__occ_occ_r12_wide_fixed_table_core`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `occ_occ_r12_wide_fixed__table_core`
- **Original source:** `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Unknown / not recoverable from repository history
- **Features:** occurrence features, 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`
- **Known score:** | wcv | 1.7500792119499793 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 2 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — occ_occ_r12_wide_fixed__table_core

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`

| Field | Value |
|---|---|
| name | occ_occ_r12_wide_fixed__table_core |
| family | occurrence_overlay |
| wcv | 1.7500792119499793 |
| base_wcv | 1.749803702558191 |
| delta | 0.00027550939178818307 |
| wins | 0 |
| wins_recent | 0 |
| latest_delta | 0.00016225852275653097 |
| worst_delta | 0.00051998519465557 |
| raw_delta | -0.0003589304093545446 |
| offset_mean | -0.047782786361238304 |
| offset_std | 0.030225784639696145 |
| fold_scores | [1.769763753786665, 1.7630940295645174, 1.7513838925032275, 1.7437125995401348] |
| fold_deltas | [0.00051998519465557, 0.00036900926919058996, 0.00039414224043343715, 0.00016225852275653097] |
| notes | raw=occ_r12_wide;base=table_core;adaptive=False |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | occ_occ_r12_wide_fixed__table_core |
| family | occurrence_overlay |
| wcv | 1.7500792119499793 |
| base_wcv | 1.749803702558191 |
| delta | 0.00027550939178818307 |
| wins | 0 |
| wins_recent | 0 |
| latest_delta | 0.00016225852275653097 |
| worst_delta | 0.00051998519465557 |
| raw_delta | -0.0003589304093545446 |
| offset_mean | -0.047782786361238304 |
| offset_std | 0.030225784639696145 |
| fold_scores | [1.769763753786665, 1.7630940295645174, 1.7513838925032275, 1.7437125995401348] |
| fold_deltas | [0.00051998519465557, 0.00036900926919058996, 0.00039414224043343715, 0.00016225852275653097] |
| notes | raw=occ_r12_wide;base=table_core;adaptive=False |
