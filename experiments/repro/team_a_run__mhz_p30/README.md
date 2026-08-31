# Logged run — MHZ-P30

## Catalogue metadata

- **Catalogue ID:** `team_a_run__mhz_p30`
- **Namespace:** `team_a_run`
- **Experiment ID:** `MHZ-P30`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** logged run / arm
- **Model:** Unknown / not recoverable from repository history
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** wcv:** 1.75267
- **Known score:** wcv:** 1.75267
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — MHZ-P30

This run was recovered from `experiments/log.csv`.

- **exp_id:** MHZ-P30
- **timestamp:** 2026-08-13T03:33:27
- **commit:** 34a2335
- **description:** exp_024 MHZ арка P30: aux = ['b30_p', 'b30_lo', 'val_mu', 'tp30']
- **scenario:** S1
- **n_features:** 255
- **model:** direct+aux
- **params:** {"aux": ["b30_p", "b30_lo", "val_mu", "tp30"]}
- **cutoffs:** Unknown / not recoverable from repository history
- **L:** Unknown / not recoverable from repository history
- **panel_blocks:** Unknown / not recoverable from repository history
- **fold_scores:** [1.77572, 1.76685, 1.7542, 1.74669]
- **cv_mean:** 1.76087
- **cv_std:** 0.01120
- **bias_mean:** -0.05496
- **best_offset:** -0.05497
- **cv_mean_calib:** 1.75986
- **delta_vs_b0:** Unknown / not recoverable from repository history
- **runtime_s:** Unknown / not recoverable from repository history
- **verdict:** Unknown / not recoverable from repository history
- **conclusion:** Unknown / not recoverable from repository history
- **wcv:** 1.75267
- **fold_cal:** [1.77346, 1.76603, 1.75389, 1.74612]
- **mean_z:** 2.68059
- **lb_public:** Unknown / not recoverable from repository history
