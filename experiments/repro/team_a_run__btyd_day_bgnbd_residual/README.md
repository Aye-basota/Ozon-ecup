# Logged run — BTYD-DAY-BGNBD-RESIDUAL

## Catalogue metadata

- **Catalogue ID:** `team_a_run__btyd_day_bgnbd_residual`
- **Namespace:** `team_a_run`
- **Experiment ID:** `BTYD-DAY-BGNBD-RESIDUAL`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** logged run / arm
- **Model:** BG/NBD, BTYD, blend
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** wcv:** 1.747240678
- **Known score:** wcv:** 1.747240678
- **Seed:** params:** {"origin": "2024-12-31", "event": "gmv>0 purchase day", "split": "splitmix64(user_id)&1", "horizon": 30, "count_cap": 30, "K": 3.0, "starts": 3, "blend_grid": [0.0, 0.025, 0.05, 0.1, 0.15], "seed": 42}
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — BTYD-DAY-BGNBD-RESIDUAL

This run was recovered from `experiments/log.csv`.

- **exp_id:** BTYD-DAY-BGNBD-RESIDUAL
- **timestamp:** 2026-08-23T19:15:28
- **commit:** a28a71f
- **description:** exp_047 common-origin purchase-day BG/NBD residual member with two-sided user cross-fit and fixed S2 monetary/aggregation
- **scenario:** S1
- **n_features:** Unknown / not recoverable from repository history
- **model:** bgnbd+fixed-s2-monetary
- **params:** {"origin": "2024-12-31", "event": "gmv>0 purchase day", "split": "splitmix64(user_id)&1", "horizon": 30, "count_cap": 30, "K": 3.0, "starts": 3, "blend_grid": [0.0, 0.025, 0.05, 0.1, 0.15], "seed": 42}
- **cutoffs:** OOF 09-04/09-18/10-02/10-16
- **L:** common-origin
- **panel_blocks:** 3
- **fold_scores:** [1.766228688, 1.759938643, 1.748454140, 1.741085955]
- **cv_mean:** 1.753926857
- **cv_std:** 0.009776452
- **bias_mean:** Unknown / not recoverable from repository history
- **best_offset:** Unknown / not recoverable from repository history
- **cv_mean_calib:** 1.753926857
- **delta_vs_b0:** -0.000269184
- **runtime_s:** 294.0
- **verdict:** REJECT
- **conclusion:** Honest nested LOFO -0.000269184, 4/4, 10-16 -0.000192612, selected weights 0.05/0.05/0.10/0.10. Below preregistered -0.0003 gate; classic BTYD family closed. No test inference, production, LB, or submission. Details exp_047
- **wcv:** 1.747240678
- **fold_cal:** [1.766228688, 1.759938643, 1.748454140, 1.741085955]
- **mean_z:** Unknown / not recoverable from repository history
- **lb_public:** Unknown / not recoverable from repository history
