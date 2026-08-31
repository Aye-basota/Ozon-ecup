# Logged run — BUYCTRL-DET

## Catalogue metadata

- **Catalogue ID:** `team_a_run__buyctrl_det`
- **Namespace:** `team_a_run`
- **Experiment ID:** `BUYCTRL-DET`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** logged run / arm
- **Model:** dilated TCN, sequence model
- **Features:** history-depth features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** wcv:** Unknown / not recoverable from repository history
- **Known score:** conclusion:** BUYTRUE-BUYSHUF mean +0.000436113, median +0.000219994, sd 0.000743770, better 1/3; BUYTRUE-BASE +0.000038615. Aux learns strongly (AUC 0.845890 vs 0.536505; BCE 0.469958 vs 0.668484), direct RMSLE does not transfer. Other folds/test/LB/submission skipped. Details exp_045
- **Seed:** description:** exp_045 deterministic plain SEQ-01 with true buy30 BCE versus cutoff-wise shuffled-label control; seeds 42/43/44
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — BUYCTRL-DET

This run was recovered from `experiments/log.csv`.

- **exp_id:** BUYCTRL-DET
- **timestamp:** 2026-08-23T04:11:16
- **commit:** a28a71f
- **description:** exp_045 deterministic plain SEQ-01 with true buy30 BCE versus cutoff-wise shuffled-label control; seeds 42/43/44
- **scenario:** S1
- **n_features:** 17
- **model:** tcn+aux
- **params:** {"arms": ["BASE", "BUYTRUE", "BUYSHUF"], "epochs": 4, "lambda_aux": 0.1, "aux_head": "Linear(192,1) train-only", "depth_aug": 0.0, "workers": 1, "materialized_plans": true, "deterministic_cuda": true, "endpoint": "end epoch 4"}
- **cutoffs:** 24 CLEAN @ step 7
- **L:** 365
- **panel_blocks:** 3
- **fold_scores:** [1.746287310, 1.745377789, 1.748103917]
- **cv_mean:** 1.746589672
- **cv_std:** 0.001133287
- **bias_mean:** Unknown / not recoverable from repository history
- **best_offset:** Unknown / not recoverable from repository history
- **cv_mean_calib:** 1.746589672
- **delta_vs_b0:** 0.000038615
- **runtime_s:** 36022.7
- **verdict:** FAIL
- **conclusion:** BUYTRUE-BUYSHUF mean +0.000436113, median +0.000219994, sd 0.000743770, better 1/3; BUYTRUE-BASE +0.000038615. Aux learns strongly (AUC 0.845890 vs 0.536505; BCE 0.469958 vs 0.668484), direct RMSLE does not transfer. Other folds/test/LB/submission skipped. Details exp_045
- **wcv:** Unknown / not recoverable from repository history
- **fold_cal:** Unknown / not recoverable from repository history
- **mean_z:** Unknown / not recoverable from repository history
- **lb_public:** Unknown / not recoverable from repository history
