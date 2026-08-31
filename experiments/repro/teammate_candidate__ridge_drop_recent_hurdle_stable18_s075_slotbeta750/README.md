# Teammate candidate — ridge_drop_recent_hurdle_stable18_s075__slotbeta750

## Catalogue metadata

- **Catalogue ID:** `teammate_candidate__ridge_drop_recent_hurdle_stable18_s075_slotbeta750`
- **Namespace:** `teammate_candidate`
- **Experiment ID:** `ridge_drop_recent_hurdle_stable18_s075__slotbeta750`
- **Original source:** `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/FINAL_CANDIDATE_METRICS.csv`
- **Source ref:** `external teammate review bundles`
- **Source commit:** `NOT_IN_GIT_HISTORY`
- **Kind:** completed teammate candidate/subrun
- **Model:** Ridge, two-part / hurdle
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** | wcv | 1.7483859421644192 |
- **Known score:** | wcv | 1.7483859421644192 |
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact result rows and runner code survive; checkpoint bank and raw data are external
- **Notes:** Recovered from 1 result-table row(s); no score or parameter was inferred.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Teammate candidate — ridge_drop_recent_hurdle_stable18_s075__slotbeta750

This candidate was recovered from completed review-bundle result tables.

## Evidence row 1

Source: `пайплайн сокомандника/review_bundles/fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted/results/FINAL_CANDIDATE_METRICS.csv`

| Field | Value |
|---|---|
| name | ridge_drop_recent_hurdle_stable18_s075__slotbeta750 |
| family | ridge_subset_slotstrength |
| wcv | 1.7483859421644192 |
| base_wcv | 1.749803702558191 |
| delta | -0.0014177603937718015 |
| wins | 3 |
| wins_recent | 3 |
| latest_delta | -0.0015077061474995102 |
| worst_delta | 0.0 |
| raw_delta | -0.0021613038829067103 |
| offset_mean | -0.03525832510189352 |
| offset_std | 0.04677430504406777 |
| fold_scores | [1.7692437685920095, 1.761040558357888, 1.7495307920498682, 1.7420426348698788] |
| fold_deltas | [0.0, -0.0016844619374387637, -0.0014589582129258538, -0.0015077061474995102] |
| notes | beta=0.75 |
| oof_table_var | 0.0022681171202219774 |
| test_table_var | 0.0027425809212591285 |
| var_ratio | 1.2091884042525618 |
| friend_corr | 0.9998482225114247 |
| friend_std_dz | 0.028794606942648272 |
| friend_mean_abs_dz | 0.02170711948218738 |
| friend_pct02 | 0.43708 |
| friend_pct05 | 0.080292 |
| friend_pct10 | 0.00348 |
