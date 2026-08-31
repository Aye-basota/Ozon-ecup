# Teammate candidate — occ_occ_r14_multiscale_adapt__super_pband_l0p06

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__occ_occ_r14_multiscale_adapt_super_pband_l0p06`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `occ_occ_r14_multiscale_adapt__super_pband_l0p06`
- **Original source:** `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Unknown / not recoverable from repository history
- **Features:** occurrence features, 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`
- **Known score:** | wcv | 1.7483837189448275 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 2 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — occ_occ_r14_multiscale_adapt__super_pband_l0p06

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`

| Field | Value |
|---|---|
| name | occ_occ_r14_multiscale_adapt__super_pband_l0p06 |
| family | occurrence_overlay |
| wcv | 1.7483837189448275 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014199836133633218 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0018177279663174062 |
| worst_delta | 0.00100588980305516 |
| raw_delta | -0.001972953320496259 |
| offset_mean | -0.005175910612754965 |
| offset_std | 0.06512829757526392 |
| fold_scores | [1.7702496583950647, 1.7621018186144966, 1.7493603960349677, 1.7417326130510609] |
| fold_deltas | [0.00100588980305516, -0.0006232016808302987, -0.0016293542278262851, -0.0018177279663174062] |
| notes | raw=occ_r14_multiscale;base=super_pband_l0p06;adaptive=True |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | occ_occ_r14_multiscale_adapt__super_pband_l0p06 |
| family | occurrence_overlay |
| wcv | 1.7483837189448275 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014199836133633218 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0018177279663174062 |
| worst_delta | 0.00100588980305516 |
| raw_delta | -0.001972953320496259 |
| offset_mean | -0.005175910612754965 |
| offset_std | 0.06512829757526392 |
| fold_scores | [1.7702496583950647, 1.7621018186144966, 1.7493603960349677, 1.7417326130510609] |
| fold_deltas | [0.00100588980305516, -0.0006232016808302987, -0.0016293542278262851, -0.0018177279663174062] |
| notes | raw=occ_r14_multiscale;base=super_pband_l0p06;adaptive=True |
