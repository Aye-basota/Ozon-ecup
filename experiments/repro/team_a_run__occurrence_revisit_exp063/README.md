# Logged run — OCCURRENCE_REVISIT_EXP063

## Catalogue metadata

- **Catalogue ID:** `team_a_run__occurrence_revisit_exp063`
- **Namespace:** `team_a_run`
- **Experiment ID:** `OCCURRENCE_REVISIT_EXP063`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** logged run / arm
- **Model:** Unknown / not recoverable from repository history
- **Features:** occurrence features, 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** wcv:** 1.747520360
- **Known score:** wcv:** 1.747520360
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — OCCURRENCE_REVISIT_EXP063

This run was recovered from `experiments/log.csv`.

- **exp_id:** OCCURRENCE_REVISIT_EXP063
- **timestamp:** 2026-08-25T03:31:00+03:00
- **commit:** a28a71f
- **description:** exp_063 existing two-part occurrence member vs same-feature direct control against exact exp_037
- **scenario:** S1-artifact-integration
- **n_features:** 227
- **model:** fixed-log-member-nested-lofo
- **params:** {"base":"STRONGEST_CURRENT","real":"S1-E11 two-part","control":"S1-E10 direct","alpha_grid":[0,0.025,0.05,0.075,0.1,0.15,0.2],"selection":"LOFO with canonical donor weights; ties smallest alpha","training":"NONE"}
- **cutoffs:** val 2025-09-04/09-18/10-02/10-16
- **L:** 180
- **panel_blocks:** 3
- **fold_scores:** [1.766951070,1.760573673,1.748619614,1.741278566]
- **cv_mean:** 1.754355731
- **cv_std:** 0.010012
- **bias_mean:** Unknown / not recoverable from repository history
- **best_offset:** Unknown / not recoverable from repository history
- **cv_mean_calib:** 1.754355731
- **delta_vs_b0:** 0.000010498
- **runtime_s:** 3.5
- **verdict:** REJECT
- **conclusion:** All provenance/base replay audits PASS. E11 held alphas .075/.075/.05/0; nested delta +0.000010498, 1/4, latest 0. Direct E10 selects 0/0/0/0. Best fixed E11 .05 only -0.000009786. Test/LB/submission not run; direct occurrence integration closed. Details exp_063
- **wcv:** 1.747520360
- **fold_cal:** [1.766951070,1.760573673,1.748619614,1.741278566]
- **mean_z:** Unknown / not recoverable from repository history
- **lb_public:** Unknown / not recoverable from repository history
