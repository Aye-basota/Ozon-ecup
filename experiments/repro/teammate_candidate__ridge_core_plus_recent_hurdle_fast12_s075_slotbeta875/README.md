# Teammate candidate — ridge_core_plus_recent_hurdle_fast12_s075__slotbeta875

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__ridge_core_plus_recent_hurdle_fast12_s075_slotbeta875`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `ridge_core_plus_recent_hurdle_fast12_s075__slotbeta875`
- **Original source:** `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/FINAL_CANDIDATE_METRICS.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Ridge, two-part / hurdle
- **Features:** 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** | wcv | 1.7483025495257658 |
- **Known score:** | wcv | 1.7483025495257658 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 1 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — ridge_core_plus_recent_hurdle_fast12_s075__slotbeta875

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/FINAL_CANDIDATE_METRICS.csv`

| Field | Value |
|---|---|
| name | ridge_core_plus_recent_hurdle_fast12_s075__slotbeta875 |
| family | ridge_subset_slotstrength |
| wcv | 1.7483025495257658 |
| base_wcv | 1.749803702558191 |
| delta | -0.001501153032425151 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0016091315458124367 |
| worst_delta | 0.0 |
| raw_delta | -0.0022762556180515444 |
| offset_mean | -0.029382630694123578 |
| offset_std | 0.05023629896311449 |
| fold_scores | [1.7692437685920095, 1.760980050830407, 1.7494511742152845, 1.7419412094715658] |
| fold_deltas | [0.0, -0.0017449694649198655, -0.00153857604750951, -0.0016091315458124367] |
| notes | beta=0.875 |
| oof_table_var | 0.003026786518865363 |
| test_table_var | 0.003703172561721399 |
| var_ratio | 1.2234667158189898 |
| friend_corr | 0.9997958436953933 |
| friend_std_dz | 0.03345254461761981 |
| friend_mean_abs_dz | 0.02517571209061937 |
| friend_pct02 | 0.495544 |
| friend_pct05 | 0.122896 |
| friend_pct10 | 0.007788 |
