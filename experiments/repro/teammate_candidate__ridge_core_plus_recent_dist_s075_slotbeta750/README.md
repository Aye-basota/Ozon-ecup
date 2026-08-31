# Teammate candidate — ridge_core_plus_recent_dist_s075__slotbeta750

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__ridge_core_plus_recent_dist_s075_slotbeta750`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `ridge_core_plus_recent_dist_s075__slotbeta750`
- **Original source:** `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/FINAL_CANDIDATE_METRICS.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Ridge
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** | wcv | 1.7484576710662523 |
- **Known score:** | wcv | 1.7484576710662523 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 1 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — ridge_core_plus_recent_dist_s075__slotbeta750

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/FINAL_CANDIDATE_METRICS.csv`

| Field | Value |
|---|---|
| name | ridge_core_plus_recent_dist_s075__slotbeta750 |
| family | ridge_subset_slotstrength |
| wcv | 1.7484576710662523 |
| base_wcv | 1.749803702558191 |
| delta | -0.0013460314919385929 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0013974431773537965 |
| worst_delta | 0.0 |
| raw_delta | -0.0020878581271269367 |
| offset_mean | -0.035458887793215724 |
| offset_std | 0.04666380013287705 |
| fold_scores | [1.7692437685920095, 1.7610538898859092, 1.7495725837274407, 1.7421528978400245] |
| fold_deltas | [0.0, -0.001671130409417687, -0.001417166535353287, -0.0013974431773537965] |
| notes | beta=0.75 |
| oof_table_var | 0.002214901496936888 |
| test_table_var | 0.0024809130826742294 |
| var_ratio | 1.1201008650295392 |
| friend_corr | 0.9998625485757763 |
| friend_std_dz | 0.027381709523373016 |
| friend_mean_abs_dz | 0.020659203310303534 |
| friend_pct02 | 0.416532 |
| friend_pct05 | 0.068848 |
| friend_pct10 | 0.00242 |
