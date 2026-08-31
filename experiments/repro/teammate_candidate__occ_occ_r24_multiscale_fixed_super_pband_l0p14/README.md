# Teammate candidate — occ_occ_r24_multiscale_fixed__super_pband_l0p14

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__occ_occ_r24_multiscale_fixed_super_pband_l0p14`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `occ_occ_r24_multiscale_fixed__super_pband_l0p14`
- **Original source:** `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv | пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Unknown / not recoverable from repository history
- **Features:** occurrence features, 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`
- **Known score:** | wcv | 1.7489005651342369 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 2 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — occ_occ_r24_multiscale_fixed__super_pband_l0p14

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/ALL_FINAL6H_VALIDATION.csv`

| Field | Value |
|---|---|
| name | occ_occ_r24_multiscale_fixed__super_pband_l0p14 |
| family | occurrence_overlay |
| wcv | 1.7489005651342369 |
| base_wcv | 1.749803702558191 |
| delta | -0.0009031374239541368 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0011406322728029217 |
| worst_delta | 0.0008209556030795984 |
| raw_delta | -0.0017139279514692518 |
| offset_mean | -0.009789741943745281 |
| offset_std | 0.05077554489198499 |
| fold_scores | [1.770064724195089, 1.7616540138779775, 1.7502145137764766, 1.7424097087445753] |
| fold_deltas | [0.0008209556030795984, -0.0010710064173493627, -0.0007752364863173877, -0.0011406322728029217] |
| notes | raw=occ_r24_multiscale;base=super_pband_l0p14;adaptive=False |

## Evidence row 2

Source: `пайплайн сокомандника/review_bundles/final6h_REVIEW_BUNDLE_20260823_204823_extracted/results/OCCURRENCE_BRANCH_VALIDATION.csv`

| Field | Value |
|---|---|
| name | occ_occ_r24_multiscale_fixed__super_pband_l0p14 |
| family | occurrence_overlay |
| wcv | 1.7489005651342369 |
| base_wcv | 1.749803702558191 |
| delta | -0.0009031374239541368 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0011406322728029217 |
| worst_delta | 0.0008209556030795984 |
| raw_delta | -0.0017139279514692518 |
| offset_mean | -0.009789741943745281 |
| offset_std | 0.05077554489198499 |
| fold_scores | [1.770064724195089, 1.7616540138779775, 1.7502145137764766, 1.7424097087445753] |
| fold_deltas | [0.0008209556030795984, -0.0010710064173493627, -0.0007752364863173877, -0.0011406322728029217] |
| notes | raw=occ_r24_multiscale;base=super_pband_l0p14;adaptive=False |
