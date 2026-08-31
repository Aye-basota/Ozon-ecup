# Teammate candidate — ridge_core_plus_recent_hurdle_fast12_s075__slotbeta750

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__ridge_core_plus_recent_hurdle_fast12_s075_slotbeta750`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `ridge_core_plus_recent_hurdle_fast12_s075__slotbeta750`
- **Original source:** `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/FINAL_CANDIDATE_METRICS.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Ridge, two-part / hurdle
- **Features:** 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** | wcv | 1.7483896625244009 |
- **Known score:** | wcv | 1.7483896625244009 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 1 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — ridge_core_plus_recent_hurdle_fast12_s075__slotbeta750

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/FINAL_CANDIDATE_METRICS.csv`

| Field | Value |
|---|---|
| name | ridge_core_plus_recent_hurdle_fast12_s075__slotbeta750 |
| family | ridge_subset_slotstrength |
| wcv | 1.7483896625244009 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014140400337899095 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.001509806285723947 |
| worst_delta | 0.0 |
| raw_delta | -0.002158761942076944 |
| offset_mean | -0.03525505576289768 |
| offset_std | 0.04675738674496263 |
| fold_scores | [1.7692437685920095, 1.76108272876934, 1.7495278584705232, 1.7420405347316543] |
| fold_deltas | [0.0, -0.0016422915259868276, -0.001461891792270853, -0.001509806285723947] |
| notes | beta=0.75 |
| oof_table_var | 0.0022237615240643487 |
| test_table_var | 0.0027206982086116394 |
| var_ratio | 1.2234667158189896 |
| friend_corr | 0.9998497159314416 |
| friend_std_dz | 0.02867843635528312 |
| friend_mean_abs_dz | 0.021583567316596036 |
| friend_pct02 | 0.434092 |
| friend_pct05 | 0.079008 |
| friend_pct10 | 0.00338 |
