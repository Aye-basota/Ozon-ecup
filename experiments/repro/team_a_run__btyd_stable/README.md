# Logged run — BTYD-STABLE

## Catalogue metadata

- **Catalogue ID:** `team_a_run__btyd_stable`
- **Namespace:** `team_a_run`
- **Experiment ID:** `BTYD-STABLE`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** logged run / arm
- **Model:** BG/NBD, BTYD
- **Features:** freshness/conditional features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** wcv:** 1.747240681
- **Known score:** wcv:** 1.747240681
- **Seed:** params:** {"candidate": "BTYD05", "cutoff": "2026-02-13", "split": "splitmix64(user_id)&1", "origin": "2024-12-31", "K": 3.0, "starts": 3, "weight": 0.05, "optimizer": "analytic-jac-lbfgsb-bfgs-v1", "seed": 42}
- **Postprocessing:** None documented
- **Submission:** conclusion:** Optimizer convergence blocker resolved without model/gate change. New nested LOFO -0.000269182 4/4; fixed 0.05 delta -0.000320983 4/4; residual alignment 4/4; OOF/test predictive stability and support PASS. submission_BTYD05.csv created; FRESH retraining stopped after one honest candidate. Details exp_051
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — BTYD-STABLE

This run was recovered from `experiments/log.csv`.

- **exp_id:** BTYD-STABLE
- **timestamp:** 2026-08-23T23:42:55
- **commit:** a28a71f
- **description:** exp_051 unchanged BG/NBD with analytic Jacobian strict deterministic fit applied equally to OOF and production
- **scenario:** S1
- **n_features:** Unknown / not recoverable from repository history
- **model:** bgnbd-analytic-jac
- **params:** {"candidate": "BTYD05", "cutoff": "2026-02-13", "split": "splitmix64(user_id)&1", "origin": "2024-12-31", "K": 3.0, "starts": 3, "weight": 0.05, "optimizer": "analytic-jac-lbfgsb-bfgs-v1", "seed": 42}
- **cutoffs:** OOF 09-04/09-18/10-02/10-16 + test 2026-02-13
- **L:** common-origin
- **panel_blocks:** 3
- **fold_scores:** [1.766228688, 1.759938649, 1.748454148, 1.741085954]
- **cv_mean:** 1.753926860
- **cv_std:** 0.009776
- **bias_mean:** Unknown / not recoverable from repository history
- **best_offset:** Unknown / not recoverable from repository history
- **cv_mean_calib:** 1.753926860
- **delta_vs_b0:** -0.000269182
- **runtime_s:** Unknown / not recoverable from repository history
- **verdict:** CASE_B_PASS
- **conclusion:** Optimizer convergence blocker resolved without model/gate change. New nested LOFO -0.000269182 4/4; fixed 0.05 delta -0.000320983 4/4; residual alignment 4/4; OOF/test predictive stability and support PASS. submission_BTYD05.csv created; FRESH retraining stopped after one honest candidate. Details exp_051
- **wcv:** 1.747240681
- **fold_cal:** [1.766228688, 1.759938649, 1.748454148, 1.741085954]
- **mean_z:** Unknown / not recoverable from repository history
- **lb_public:** Unknown / not recoverable from repository history
