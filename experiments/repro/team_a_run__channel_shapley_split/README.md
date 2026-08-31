# Logged run — CHANNEL-SHAPLEY-SPLIT

## Catalogue metadata

- **Catalogue ID:** `team_a_run__channel_shapley_split`
- **Namespace:** `team_a_run`
- **Experiment ID:** `CHANNEL-SHAPLEY-SPLIT`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** logged run / arm
- **Model:** Unknown / not recoverable from repository history
- **Features:** Search/Catalog decomposition, channel Shapley, 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** wcv:** Unknown / not recoverable from repository history
- **Known score:** Unknown / not recoverable from repository history
- **Seed:** params:** {"rounds":300,"early_stopping":false,"pilot_fold":"2025-10-16","search_seed":42,"catalog_seed":43,"shuffle":"train-cutoff x stable-z-decile","alpha_grid":[0,0.25,0.5,1]}
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — CHANNEL-SHAPLEY-SPLIT

This run was recovered from `experiments/log.csv`.

- **exp_id:** CHANNEL-SHAPLEY-SPLIT
- **timestamp:** 2026-08-24T02:02:57
- **commit:** a28a71f
- **description:** exp_052 audit-gated Search/Catalog Shapley monetary heads versus amount-matched shuffled-channel control
- **scenario:** S1
- **n_features:** 227
- **model:** lgb-direct-two-head
- **params:** {"rounds":300,"early_stopping":false,"pilot_fold":"2025-10-16","search_seed":42,"catalog_seed":43,"shuffle":"train-cutoff x stable-z-decile","alpha_grid":[0,0.25,0.5,1]}
- **cutoffs:** pilot 2025-10-16
- **L:** existing clean recipe
- **panel_blocks:** 3
- **fold_scores:** [1.741278566]
- **cv_mean:** 1.741278566
- **cv_std:** 0
- **bias_mean:** Unknown / not recoverable from repository history
- **best_offset:** -0.035387154
- **cv_mean_calib:** 1.741278566
- **delta_vs_b0:** 0.000000000
- **runtime_s:** 709.605
- **verdict:** REJECT
- **conclusion:** Preflight GO: oracle -0.482936 shape-only and predictability Spearman 0.278576 4/4. Pilot REAL-SHUF +0.001265695; both directions selected alpha=0; primary two-sided delta 0; fixed positive alphas worsen; full folds/test/LB/submission not run. Details exp_052
- **wcv:** Unknown / not recoverable from repository history
- **fold_cal:** [1.741278566]
- **mean_z:** Unknown / not recoverable from repository history
- **lb_public:** Unknown / not recoverable from repository history
