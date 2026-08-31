# Logged run — S1-GAPAXIS

## Catalogue metadata

- **Catalogue ID:** `team_a_run__s1_gapaxis`
- **Namespace:** `team_a_run`
- **Experiment ID:** `S1-GAPAXIS`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** logged run / arm
- **Model:** Unknown / not recoverable from repository history
- **Features:** gap/burst features, 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** model:** validation
- **Known score:** wcv:** 1.75637
- **Seed:** params:** {"actual_gaps": [35, 63, 91, 126], "capacity": "per probe and gap", "k11_control": true, "n_cutoffs": 5, "requested_gaps": [30, 60, 90, 120], "seed": 42}
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — S1-GAPAXIS

This run was recovered from `experiments/log.csv`.

- **exp_id:** S1-GAPAXIS
- **timestamp:** 2026-08-12T15:26:40
- **commit:** 3c62fa3
- **description:** STRATEGY_01 gap-axis k=5 with k=11 latest-fold control
- **scenario:** S2
- **n_features:** 227
- **model:** validation
- **params:** {"actual_gaps": [35, 63, 91, 126], "capacity": "per probe and gap", "k11_control": true, "n_cutoffs": 5, "requested_gaps": [30, 60, 90, 120], "seed": 42}
- **cutoffs:** last 5 eligible @ step 7
- **L:** 0
- **panel_blocks:** 3
- **fold_scores:** [1.77979, 1.76936, 1.75667, 1.75032]
- **cv_mean:** 1.76403
- **cv_std:** 0.01139
- **bias_mean:** 0.01215
- **best_offset:** 0.01215
- **cv_mean_calib:** 1.76384
- **delta_vs_b0:** Unknown / not recoverable from repository history
- **runtime_s:** Unknown / not recoverable from repository history
- **verdict:** REJECT
- **conclusion:** E10 gCV worsens by +0.004951 from actual gap 35 to 126, but the registered bias slope fails (+0.000543/day vs a negative accept band). E03a narrows its deficit but never changes rank; k11 is non-monotone. Do not use gCV as a selection gate; details in exp_019.
- **wcv:** 1.75637
- **fold_cal:** [1.77818, 1.76936, 1.7566, 1.75027]
- **mean_z:** 2.61348
- **lb_public:** Unknown / not recoverable from repository history
