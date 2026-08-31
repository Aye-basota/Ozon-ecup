# Teammate candidate — occ_occ_r16_bal_fixed__super_pband_l0p06

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__occ_occ_r16_bal_fixed_super_pband_l0p06`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `occ_occ_r16_bal_fixed__super_pband_l0p06`
- **Original source:** `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Unknown / not recoverable from repository history
- **Features:** occurrence features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`
- **Known score:** | wcv | 1.748589327919988 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 2 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — occ_occ_r16_bal_fixed__super_pband_l0p06

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`

| Field | Value |
|---|---|
| name | occ_occ_r16_bal_fixed__super_pband_l0p06 |
| family | occurrence_overlay |
| wcv | 1.748589327919988 |
| base_wcv | 1.749803702558191 |
| delta | -0.0012143746382029085 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0014042002827576727 |
| worst_delta | 6.966106818939721e-05 |
| raw_delta | -0.0020255038942885496 |
| offset_mean | -0.012645069695793003 |
| offset_std | 0.05158733162281663 |
| fold_scores | [1.7693134296601989, 1.7612932398726515, 1.7499427208793388, 1.7421461407346206] |
| fold_deltas | [6.966106818939721e-05, -0.0014317804226753328, -0.0010470293834552447, -0.0014042002827576727] |
| notes | raw=occ_r16_bal;base=super_pband_l0p06;adaptive=False |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | occ_occ_r16_bal_fixed__super_pband_l0p06 |
| family | occurrence_overlay |
| wcv | 1.748589327919988 |
| base_wcv | 1.749803702558191 |
| delta | -0.0012143746382029085 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0014042002827576727 |
| worst_delta | 6.966106818939721e-05 |
| raw_delta | -0.0020255038942885496 |
| offset_mean | -0.012645069695793003 |
| offset_std | 0.05158733162281663 |
| fold_scores | [1.7693134296601989, 1.7612932398726515, 1.7499427208793388, 1.7421461407346206] |
| fold_deltas | [6.966106818939721e-05, -0.0014317804226753328, -0.0010470293834552447, -0.0014042002827576727] |
| notes | raw=occ_r16_bal;base=super_pband_l0p06;adaptive=False |
