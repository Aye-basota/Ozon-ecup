# Logged run — LANDMARK-MEMORY-055

## Catalogue metadata

- **Catalogue ID:** `team_a_run__landmark_memory_055`
- **Namespace:** `team_a_run`
- **Experiment ID:** `LANDMARK-MEMORY-055`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** logged run / arm
- **Model:** Unknown / not recoverable from repository history
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** wcv:** 1.747509863
- **Known score:** wcv:** 1.747509863
- **Seed:** params:** {"rounds":200,"num_leaves":31,"min_data_in_leaf":2000,"learning_rate":0.03,"lambda_l2":20,"max_bin":63,"lags":[30,45,60,75,90,105,120,135,150,165,180,195,210,225,240,255],"memory_summaries":8,"scales":[0,0.25,0.5,1],"shuffle":"cutoff x lag x donor-only past30-GMV decile x side","split":"splitmix64(user_id)&1","seed":42}
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — LANDMARK-MEMORY-055

This run was recovered from `experiments/log.csv`.

- **exp_id:** LANDMARK-MEMORY-055
- **timestamp:** 2026-08-24T16:07:25
- **commit:** a28a71f
- **description:** exp_055 cutoff-safe retrospective state-to-realized-30d-GMV landmark memory versus matched shuffled-outcome control
- **scenario:** S1
- **n_features:** 269
- **model:** artifact-only-landmark-memory-cpu-probe
- **params:** {"rounds":200,"num_leaves":31,"min_data_in_leaf":2000,"learning_rate":0.03,"lambda_l2":20,"max_bin":63,"lags":[30,45,60,75,90,105,120,135,150,165,180,195,210,225,240,255],"memory_summaries":8,"scales":[0,0.25,0.5,1],"shuffle":"cutoff x lag x donor-only past30-GMV decile x side","split":"splitmix64(user_id)&1","seed":42}
- **cutoffs:** donor 09-04/09-18/10-02; recipient 10-16
- **L:** landmarks 30..255
- **panel_blocks:** 3
- **fold_scores:** [1.766883357, 1.760509577, 1.748629224, 1.741278566]
- **cv_mean:** 1.754325181
- **cv_std:** 0.009995
- **bias_mean:** Unknown / not recoverable from repository history
- **best_offset:** -0.035387154
- **cv_mean_calib:** 1.754325181
- **delta_vs_b0:** 0.000000000
- **runtime_s:** 861.260
- **verdict:** NO_GO_PREFLIGHT
- **conclusion:** Exact baseline/leakage/shuffle audits PASS. REAL and SHUFFLED both selected residual scales 0/0; late delta 0, REAL-SHUFFLED 0, both recipient halves tie; pooled partial residual corr 0.005814<0.02. Raw same-user target corr 0.498 is not incremental utility. GPU/pilot/full folds/test/LB/submission not run. Details exp_055
- **wcv:** 1.747509863
- **fold_cal:** [1.766883357, 1.760509577, 1.748629224, 1.741278566]
- **mean_z:** Unknown / not recoverable from repository history
- **lb_public:** Unknown / not recoverable from repository history
