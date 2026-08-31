# Logged run — FRESH-CONTRAST-MOE

## Catalogue metadata

- **Catalogue ID:** `team_a_run__fresh_contrast_moe`
- **Namespace:** `team_a_run`
- **Experiment ID:** `FRESH-CONTRAST-MOE`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** logged run / arm
- **Model:** dilated TCN
- **Features:** freshness/conditional features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** conclusion:** Nested LOFO FRESH -0.000225, 4/4; 10-16 -0.000253; selected GLOBAL alpha=1 on all folds. VOL +0.000008; FRESH-VOL -0.000233; A/B -0.000249/-0.000201; HIGH16 -0.000027 2/4. Validation gate failed; test inference and submission skipped. Details exp_040
- **Known score:** wcv:** 1.74728
- **Seed:** params:** {"alphas": [0.0, 0.25, 0.5, 0.75, 1.0], "base": "STRONGEST_CURRENT", "contrasts": ["FRESH", "VOL"], "head_seed": 42, "preprocess": "donor-fold 0.5/99.5 winsor; gate; center", "split": "splitmix64(user_id)&1", "variants": ["GLOBAL", "HIGH16"]}
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — FRESH-CONTRAST-MOE

This run was recovered from `experiments/log.csv`.

- **exp_id:** FRESH-CONTRAST-MOE
- **timestamp:** 2026-08-21T15:00:00
- **commit:** a28a71f
- **description:** exp_040 full two-sided EXP-032B d_fresh residual correction of STRONGEST_CURRENT; GLOBAL/HIGH16 nested LOFO and VOL control
- **scenario:** S1
- **n_features:** Unknown / not recoverable from repository history
- **model:** frozen-tcn-head+residual
- **params:** {"alphas": [0.0, 0.25, 0.5, 0.75, 1.0], "base": "STRONGEST_CURRENT", "contrasts": ["FRESH", "VOL"], "head_seed": 42, "preprocess": "donor-fold 0.5/99.5 winsor; gate; center", "split": "splitmix64(user_id)&1", "variants": ["GLOBAL", "HIGH16"]}
- **cutoffs:** reuse CLEAN 18/20/22/24 + 13 EXTRA; two-sided user cross-fit
- **L:** Unknown / not recoverable from repository history
- **panel_blocks:** 3
- **fold_scores:** [1.76664, 1.76035, 1.74844, 1.74103]
- **cv_mean:** Unknown / not recoverable from repository history
- **cv_std:** Unknown / not recoverable from repository history
- **bias_mean:** Unknown / not recoverable from repository history
- **best_offset:** Unknown / not recoverable from repository history
- **cv_mean_calib:** 1.75411
- **delta_vs_b0:** -0.000225
- **runtime_s:** 500
- **verdict:** REJECT
- **conclusion:** Nested LOFO FRESH -0.000225, 4/4; 10-16 -0.000253; selected GLOBAL alpha=1 on all folds. VOL +0.000008; FRESH-VOL -0.000233; A/B -0.000249/-0.000201; HIGH16 -0.000027 2/4. Validation gate failed; test inference and submission skipped. Details exp_040
- **wcv:** 1.74728
- **fold_cal:** [1.76664, 1.76035, 1.74844, 1.74103]
- **mean_z:** 2.68819
- **lb_public:** Unknown / not recoverable from repository history
