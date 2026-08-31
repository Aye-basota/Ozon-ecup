# Teammate candidate — ridge_core_plus_recent_direct_s075__slotbeta875

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__ridge_core_plus_recent_direct_s075_slotbeta875`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `ridge_core_plus_recent_direct_s075__slotbeta875`
- **Original source:** `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/FINAL_CANDIDATE_METRICS.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Ridge
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** | wcv | 1.7484107925600434 |
- **Known score:** | wcv | 1.7484107925600434 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 1 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — ridge_core_plus_recent_direct_s075__slotbeta875

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/FINAL_CANDIDATE_METRICS.csv`

| Field | Value |
|---|---|
| name | ridge_core_plus_recent_direct_s075__slotbeta875 |
| family | ridge_subset_slotstrength |
| wcv | 1.7484107925600434 |
| base_wcv | 1.749803702558191 |
| delta | -0.0013929099981477104 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0014511537994339463 |
| worst_delta | 0.0 |
| raw_delta | -0.0021673603421816937 |
| offset_mean | -0.029663537865314404 |
| offset_std | 0.05004940599603792 |
| fold_scores | [1.7692437685920095, 1.7609719123019225, 1.7495451993653102, 1.7420991872179443] |
| fold_deltas | [0.0, -0.0017531079934043348, -0.001444550897483854, -0.0014511537994339463] |
| notes | beta=0.875 |
| oof_table_var | 0.002962755516866121 |
| test_table_var | 0.0033109395590422966 |
| var_ratio | 1.1175203421929563 |
| friend_corr | 0.999817030286236 |
| friend_std_dz | 0.031623846816634514 |
| friend_mean_abs_dz | 0.02381461879555961 |
| friend_pct02 | 0.473984 |
| friend_pct05 | 0.10622 |
| friend_pct10 | 0.005768 |
