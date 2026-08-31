# Logged run — BURST-GAP-ETX

## Catalogue metadata

- **Catalogue ID:** `team_a_run__burst_gap_etx`
- **Namespace:** `team_a_run`
- **Experiment ID:** `BURST-GAP-ETX`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** logged run / arm
- **Model:** Unknown / not recoverable from repository history
- **Features:** gap/burst features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** wcv:** 1.747509863
- **Known score:** wcv:** 1.747509863
- **Seed:** params:** {"rounds":200,"num_leaves":31,"min_data_in_leaf":2000,"learning_rate":0.03,"lambda_l2":20,"max_bin":63,"burst_threshold":3,"episode_features":12,"scales":[0,0.25,0.5,1],"split":"splitmix64(user_id)&1","seed":42}
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — BURST-GAP-ETX

This run was recovered from `experiments/log.csv`.

- **exp_id:** BURST-GAP-ETX
- **timestamp:** 2026-08-24T16:00:23
- **commit:** a28a71f
- **description:** exp_054 fixed threshold-3 activity episodes and explicit inactivity gaps, audit-gated against matched joint shuffle
- **scenario:** S1
- **n_features:** 273
- **model:** artifact-only-burst-gap-cpu-probe
- **params:** {"rounds":200,"num_leaves":31,"min_data_in_leaf":2000,"learning_rate":0.03,"lambda_l2":20,"max_bin":63,"burst_threshold":3,"episode_features":12,"scales":[0,0.25,0.5,1],"split":"splitmix64(user_id)&1","seed":42}
- **cutoffs:** donor 09-04/09-18/10-02; recipient 10-16
- **L:** event-window365
- **panel_blocks:** 3
- **fold_scores:** [1.766883357, 1.760509577, 1.748629224, 1.741278566]
- **cv_mean:** 1.754325181
- **cv_std:** 0.009995
- **bias_mean:** Unknown / not recoverable from repository history
- **best_offset:** -0.035387154
- **cv_mean_calib:** 1.754325181
- **delta_vs_b0:** 0.000000000
- **runtime_s:** 494.0
- **verdict:** NO_GO_PREFLIGHT
- **conclusion:** Exact baseline and novelty audits PASS, but REAL and SHUFFLED both selected residual scales 0/0; late delta 0 and REAL-SHUFFLED 0; both recipient halves tie. Mid-activity spread 0.01998<0.03 and AUC excess +0.0000207<0.002. GPU/full folds/test/LB/submission not run. Details exp_054
- **wcv:** 1.747509863
- **fold_cal:** [1.766883357, 1.760509577, 1.748629224, 1.741278566]
- **mean_z:** Unknown / not recoverable from repository history
- **lb_public:** Unknown / not recoverable from repository history
