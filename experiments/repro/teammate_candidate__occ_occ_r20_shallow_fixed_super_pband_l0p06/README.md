# Teammate candidate — occ_occ_r20_shallow_fixed__super_pband_l0p06

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__occ_occ_r20_shallow_fixed_super_pband_l0p06`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `occ_occ_r20_shallow_fixed__super_pband_l0p06`
- **Original source:** `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Unknown / not recoverable from repository history
- **Features:** occurrence features, 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`
- **Known score:** | wcv | 1.748503732276408 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 2 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — occ_occ_r20_shallow_fixed__super_pband_l0p06

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`

| Field | Value |
|---|---|
| name | occ_occ_r20_shallow_fixed__super_pband_l0p06 |
| family | occurrence_overlay |
| wcv | 1.748503732276408 |
| base_wcv | 1.749803702558191 |
| delta | -0.0012999702817827548 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0014330431874858274 |
| worst_delta | 3.360996094081692e-05 |
| raw_delta | -0.0021212420143171744 |
| offset_mean | -0.01290260956118161 |
| offset_std | 0.050373993861612475 |
| fold_scores | [1.7692773785529503, 1.7613180145210778, 1.7496760484779696, 1.7421172978298924] |
| fold_deltas | [3.360996094081692e-05, -0.001407005774249015, -0.0013137017848243726, -0.0014330431874858274] |
| notes | raw=occ_r20_shallow;base=super_pband_l0p06;adaptive=False |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | occ_occ_r20_shallow_fixed__super_pband_l0p06 |
| family | occurrence_overlay |
| wcv | 1.748503732276408 |
| base_wcv | 1.749803702558191 |
| delta | -0.0012999702817827548 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0014330431874858274 |
| worst_delta | 3.360996094081692e-05 |
| raw_delta | -0.0021212420143171744 |
| offset_mean | -0.01290260956118161 |
| offset_std | 0.050373993861612475 |
| fold_scores | [1.7692773785529503, 1.7613180145210778, 1.7496760484779696, 1.7421172978298924] |
| fold_deltas | [3.360996094081692e-05, -0.001407005774249015, -0.0013137017848243726, -0.0014330431874858274] |
| notes | raw=occ_r20_shallow;base=super_pband_l0p06;adaptive=False |
