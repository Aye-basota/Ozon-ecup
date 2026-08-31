# Logged run — ZERO2D-SHRINK

## Catalogue metadata

- **Catalogue ID:** `team_a_run__zero2d_shrink`
- **Namespace:** `team_a_run`
- **Experiment ID:** `ZERO2D-SHRINK`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** logged run / arm
- **Model:** Unknown / not recoverable from repository history
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** wcv:** 1.747485
- **Known score:** conclusion:** Honest ZERO2D -0.000024756, 2/4, 10-16 -0.000029746, eta=1 all; AMOUNT-ONLY -0.000034321; shuffled-p0 -0.000026348; zero error worsened while positives improved. Explicit post-REJECT LB probe prepared: test regime PASS, Var ratio 1.0438, SHA256 9f1cf32671fb18291659b61da232244d370f7c9af2e0cf9d8aebf9eba406d461; local verdict remains REJECT. Details exp_042
- **Seed:** params:** {"amount_edges": [1, 3, 10, 30, 50, 100], "base": "STRONGEST_CURRENT", "eta_grid": [0.25, 0.5, 0.75, 1.0], "min_cell_rows": 500, "p0_bins": 5, "shrinkage": 20000, "shuffle_seed": 42}
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — ZERO2D-SHRINK

This run was recovered from `experiments/log.csv`.

- **exp_id:** ZERO2D-SHRINK
- **timestamp:** 2026-08-21T22:37:15
- **commit:** a28a71f
- **description:** exp_042 ZERO2D negative-only EB/isotonic residual correction by fixed amount bins and S1-DIST p0
- **scenario:** S1
- **n_features:** Unknown / not recoverable from repository history
- **model:** oof-eb-isotonic
- **params:** {"amount_edges": [1, 3, 10, 30, 50, 100], "base": "STRONGEST_CURRENT", "eta_grid": [0.25, 0.5, 0.75, 1.0], "min_cell_rows": 500, "p0_bins": 5, "shrinkage": 20000, "shuffle_seed": 42}
- **cutoffs:** OOF 09-04/09-18/10-02/10-16
- **L:** Unknown / not recoverable from repository history
- **panel_blocks:** 3
- **fold_scores:** [1.766919174, 1.760514513, 1.748584459, 1.741248820]
- **cv_mean:** 1.75432
- **cv_std:** 0.01001
- **bias_mean:** Unknown / not recoverable from repository history
- **best_offset:** Unknown / not recoverable from repository history
- **cv_mean_calib:** 1.75432
- **delta_vs_b0:** -0.000025
- **runtime_s:** 9.6
- **verdict:** REJECT
- **conclusion:** Honest ZERO2D -0.000024756, 2/4, 10-16 -0.000029746, eta=1 all; AMOUNT-ONLY -0.000034321; shuffled-p0 -0.000026348; zero error worsened while positives improved. Explicit post-REJECT LB probe prepared: test regime PASS, Var ratio 1.0438, SHA256 9f1cf32671fb18291659b61da232244d370f7c9af2e0cf9d8aebf9eba406d461; local verdict remains REJECT. Details exp_042
- **wcv:** 1.747485
- **fold_cal:** [1.766919174, 1.760514513, 1.748584459, 1.741248820]
- **mean_z:** 2.68392
- **lb_public:** Unknown / not recoverable from repository history
