# Teammate candidate — xmeta_fast4_p23_l23_plain__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__xmeta_fast4_p23_l23_plain_blend_ridge_recentpow1p7_s075_greedy35_finalizable_pr85`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `xmeta_fast4_p23_l23_plain__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85`
- **Original source:** `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Ridge, blend
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`
- **Known score:** | wcv | 1.7480725784802114 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 1 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — xmeta_fast4_p23_l23_plain__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/extra90_REVIEW_BUNDLE_20260823_222555_extracted/results/ALL_EXTRA90_VALIDATION.csv`

| Field | Value |
|---|---|
| name | xmeta_fast4_p23_l23_plain__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85 |
| family | xmeta_plain |
| wcv | 1.7480725784802114 |
| base_wcv | 1.749803702558191 |
| delta | -0.0017311240779797287 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.002067385601278282 |
| worst_delta | 0.0 |
| raw_delta | -0.001408612387330018 |
| offset_mean | 0.016078684228153436 |
| offset_std | 0.09839352558964988 |
| fold_scores | [1.7692437685920095, 1.7617849500765779, 1.749102841282301, 1.7414829554161] |
| fold_deltas | [0.0, -0.0009400702187489784, -0.0018869089804929295, -0.002067385601278282] |
| notes | subset=['occ_r10_fast', 'occ_r12_wide', 'occ_r14_multiscale', 'occ_r16_bal'];power=2.3;leaves=23;base=blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85 |
