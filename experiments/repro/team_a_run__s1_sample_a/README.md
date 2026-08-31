# Logged run — S1-SAMPLE-A

## Catalogue metadata

- **Catalogue ID:** `team_a_run__s1_sample_a`
- **Namespace:** `team_a_run`
- **Experiment ID:** `S1-SAMPLE-A`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** logged run / arm
- **Model:** Unknown / not recoverable from repository history
- **Features:** 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** conclusion:** Avg3 wCV 1.750569 vs capacity-matched baseline 1.750456: +0.000113; 2/4 folds and the latest fold is worse by +0.000194. Training rows +9.23% without a quality gain. Exclude from pipeline; details in exp_020.
- **Known score:** conclusion:** Avg3 wCV 1.750569 vs capacity-matched baseline 1.750456: +0.000113; 2/4 folds and the latest fold is worse by +0.000194. Training rows +9.23% without a quality gain. Exclude from pipeline; details in exp_020.
- **Seed:** params:** {"baseline_rounds": 300, "only_change": "train sample panel", "rounds": 200, "seeds": [42, 43, 44], "train_blocks": 0}
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — S1-SAMPLE-A

This run was recovered from `experiments/log.csv`.

- **exp_id:** S1-SAMPLE-A
- **timestamp:** 2026-08-12T15:08:55
- **commit:** 27098ea
- **description:** STRATEGY_02 Variant A: train_blocks=0 with a separate capacity curve and avg3
- **scenario:** S1
- **n_features:** 227
- **model:** direct
- **params:** {"baseline_rounds": 300, "only_change": "train sample panel", "rounds": 200, "seeds": [42, 43, 44], "train_blocks": 0}
- **cutoffs:** all @ step 7
- **L:** 0
- **panel_blocks:** 3
- **fold_scores:** [1.77179, 1.76447, 1.75225, 1.74476]
- **cv_mean:** 1.75832
- **cv_std:** 0.01049
- **bias_mean:** -0.05019
- **best_offset:** -0.05019
- **cv_mean_calib:** 1.75746
- **delta_vs_b0:** Unknown / not recoverable from repository history
- **runtime_s:** Unknown / not recoverable from repository history
- **verdict:** REJECT
- **conclusion:** Avg3 wCV 1.750569 vs capacity-matched baseline 1.750456: +0.000113; 2/4 folds and the latest fold is worse by +0.000194. Training rows +9.23% without a quality gain. Exclude from pipeline; details in exp_020.
- **wcv:** 1.75057
- **fold_cal:** [1.77018, 1.76389, 1.75195, 1.7441]
- **mean_z:** 2.67582
- **lb_public:** Unknown / not recoverable from repository history
