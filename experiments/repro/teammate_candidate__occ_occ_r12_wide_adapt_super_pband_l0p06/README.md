# Teammate candidate — occ_occ_r12_wide_adapt__super_pband_l0p06

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__occ_occ_r12_wide_adapt_super_pband_l0p06`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `occ_occ_r12_wide_adapt__super_pband_l0p06`
- **Original source:** `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Unknown / not recoverable from repository history
- **Features:** occurrence features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`
- **Known score:** | wcv | 1.748332387865427 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 2 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — occ_occ_r12_wide_adapt__super_pband_l0p06

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`

| Field | Value |
|---|---|
| name | occ_occ_r12_wide_adapt__super_pband_l0p06 |
| family | occurrence_overlay |
| wcv | 1.748332387865427 |
| base_wcv | 1.749803702558191 |
| delta | -0.001471314692764188 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0018252031642240762 |
| worst_delta | 0.00051998519465557 |
| raw_delta | -0.001786132243014175 |
| offset_mean | 0.006520482070299365 |
| offset_std | 0.07118052495290526 |
| fold_scores | [1.769763753786665, 1.7616165052020798, 1.749546987741336, 1.7417251378531542] |
| fold_deltas | [0.00051998519465557, -0.0011085150932470533, -0.0014427625214579187, -0.0018252031642240762] |
| notes | raw=occ_r12_wide;base=super_pband_l0p06;adaptive=True |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | occ_occ_r12_wide_adapt__super_pband_l0p06 |
| family | occurrence_overlay |
| wcv | 1.748332387865427 |
| base_wcv | 1.749803702558191 |
| delta | -0.001471314692764188 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0018252031642240762 |
| worst_delta | 0.00051998519465557 |
| raw_delta | -0.001786132243014175 |
| offset_mean | 0.006520482070299365 |
| offset_std | 0.07118052495290526 |
| fold_scores | [1.769763753786665, 1.7616165052020798, 1.749546987741336, 1.7417251378531542] |
| fold_deltas | [0.00051998519465557, -0.0011085150932470533, -0.0014427625214579187, -0.0018252031642240762] |
| notes | raw=occ_r12_wide;base=super_pband_l0p06;adaptive=True |
