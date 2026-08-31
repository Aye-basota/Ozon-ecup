# Logged run — MHZ-FULL

## Catalogue metadata

- **Catalogue ID:** `team_a_run__mhz_full`
- **Namespace:** `team_a_run`
- **Experiment ID:** `MHZ-FULL`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** logged run / arm
- **Model:** Unknown / not recoverable from repository history
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** wcv:** 1.75234
- **Known score:** wcv:** 1.75234
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — MHZ-FULL

This run was recovered from `experiments/log.csv`.

- **exp_id:** MHZ-FULL
- **timestamp:** 2026-08-13T03:33:30
- **commit:** 34a2335
- **description:** exp_024 MHZ арка FULL: aux = ['haz_p7', 'haz_p14', 'haz_p21', 'haz_p30', 'haz_p45', 'haz_p60', 'haz_edays', 'haz_h2', 'haz_h3', 'haz_h4', 'haz_h5', 'haz_h6', 'haz_lo7', 'haz_lo30', 'haz_lo60', 'haz_sl730', 'haz_sl3060', 'cnt_p0', 'cnt_en', 'cnt_ge2', 'cnt_ge4', 'cnt_mix', 'b30_p', 'b30_lo', 'val_mu', 'tp30', 'tp_cnt']
- **scenario:** S1
- **n_features:** 255
- **model:** direct+aux
- **params:** {"aux": ["haz_p7", "haz_p14", "haz_p21", "haz_p30", "haz_p45", "haz_p60", "haz_edays", "haz_h2", "haz_h3", "haz_h4", "haz_h5", "haz_h6", "haz_lo7", "haz_lo30", "haz_lo60", "haz_sl730", "haz_sl3060", "cnt_p0", "cnt_en", "cnt_ge2", "cnt_ge4", "cnt_mix", "b30_p", "b30_lo", "val_mu", "tp30", "tp_cnt"]}
- **cutoffs:** Unknown / not recoverable from repository history
- **L:** Unknown / not recoverable from repository history
- **panel_blocks:** Unknown / not recoverable from repository history
- **fold_scores:** [1.77536, 1.76653, 1.75373, 1.7465]
- **cv_mean:** 1.76053
- **cv_std:** 0.01117
- **bias_mean:** -0.05565
- **best_offset:** -0.05565
- **cv_mean_calib:** 1.75950
- **delta_vs_b0:** Unknown / not recoverable from repository history
- **runtime_s:** Unknown / not recoverable from repository history
- **verdict:** Unknown / not recoverable from repository history
- **conclusion:** Unknown / not recoverable from repository history
- **wcv:** 1.75234
- **fold_cal:** [1.77304, 1.76573, 1.75342, 1.74588]
- **mean_z:** 2.68128
- **lb_public:** Unknown / not recoverable from repository history
