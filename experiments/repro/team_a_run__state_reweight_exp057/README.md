# Logged run — STATE_REWEIGHT_EXP057

## Catalogue metadata

- **Catalogue ID:** `team_a_run__state_reweight_exp057`
- **Namespace:** `team_a_run`
- **Experiment ID:** `STATE_REWEIGHT_EXP057`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** logged run / arm
- **Model:** Unknown / not recoverable from repository history
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** wcv:** Unknown / not recoverable from repository history
- **Known score:** Unknown / not recoverable from repository history
- **Seed:** params:** {"base":"historical S1-E02 exact replay","domain_features":101,"domain_rounds":200,"odds_clip":[0.25,4.0],"user_total_cap":"2x median","arms":["UNIFORM","SHUFFLED","STATE_MATCH"],"unc_rounds":600,"slot_weight":0.20,"seed":42,"threads":4}
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — STATE_REWEIGHT_EXP057

This run was recovered from `experiments/log.csv`.

- **exp_id:** STATE_REWEIGHT_EXP057
- **timestamp:** 2026-08-24T21:27:18+03:00
- **commit:** a28a71f
- **description:** exp_057 target-free semantic production-state reweighting of fixed historical UNC slot vs exact within-state shuffled-weight control
- **scenario:** S1-pilot
- **n_features:** 236/101
- **model:** lgb-direct-weighted
- **params:** {"base":"historical S1-E02 exact replay","domain_features":101,"domain_rounds":200,"odds_clip":[0.25,4.0],"user_total_cap":"2x median","arms":["UNIFORM","SHUFFLED","STATE_MATCH"],"unc_rounds":600,"slot_weight":0.20,"seed":42,"threads":4}
- **cutoffs:** 24 @ step 7; pilot 2025-10-16
- **L:** Unknown / not recoverable from repository history
- **panel_blocks:** 3
- **fold_scores:** [1.741294735]
- **cv_mean:** 1.741294735
- **cv_std:** 0
- **bias_mean:** Unknown / not recoverable from repository history
- **best_offset:** -0.035914134
- **cv_mean_calib:** 1.741294735
- **delta_vs_b0:** 0.000016169
- **runtime_s:** 897.3
- **verdict:** REJECT
- **conclusion:** Exact UNC and STRONGEST replay PASS. ESS 79.30%, domain AUC 0.7213, no tiny stratum. MATCHED-SHUFFLED standalone +0.000254069; fixed slot +0.000002324; MATCHED-STRONGEST +0.000016169; halves -0.0000821/+0.0000860. Full folds/test/LB/submission not run. Details exp_057
- **wcv:** Unknown / not recoverable from repository history
- **fold_cal:** [1.741294735]
- **mean_z:** 2.667433263
- **lb_public:** Unknown / not recoverable from repository history
