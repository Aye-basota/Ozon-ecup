# Logged run — S1-SAMPLE-B

## Catalogue metadata

- **Catalogue ID:** `team_a_run__s1_sample_b`
- **Namespace:** `team_a_run`
- **Experiment ID:** `S1-SAMPLE-B`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** logged run / arm
- **Model:** dilated TCN
- **Features:** 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** conclusion:** Capacity-matched dense wCV 1.752339 vs baseline 1.751076: +0.001263, 0/4 folds. Weighted AUC -0.000416; rec_buy 15-60 RMSLE/AUC +0.001557/-0.001208; 2-15 purchase days +0.001512/-0.000935. Train rows within +/-1% each fold. Dense supervision gate for HDN/TCN failed; details exp_022.
- **Known score:** conclusion:** Capacity-matched dense wCV 1.752339 vs baseline 1.751076: +0.001263, 0/4 folds. Weighted AUC -0.000416; rec_buy 15-60 RMSLE/AUC +0.001557/-0.001208; 2-15 purchase days +0.001512/-0.000935. Train rows within +/-1% each fold. Dense supervision gate for HDN/TCN failed; details exp_022.
- **Seed:** params:** {"baseline_rounds": 300, "dense_rounds": 200, "dense_step": 3, "row_frac": 0.422, "seed": 42}
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — S1-SAMPLE-B

This run was recovered from `experiments/log.csv`.

- **exp_id:** S1-SAMPLE-B
- **timestamp:** 2026-08-12T20:19:27
- **commit:** 34a2335
- **description:** STRATEGY_02 Variant B: dense step 3 at matched train volume
- **scenario:** S1
- **n_features:** 227
- **model:** direct
- **params:** {"baseline_rounds": 300, "dense_rounds": 200, "dense_step": 3, "row_frac": 0.422, "seed": 42}
- **cutoffs:** 57 @ step 3, row_frac 0.422
- **L:** Unknown / not recoverable from repository history
- **panel_blocks:** 3
- **fold_scores:** [1.77383, 1.76646, 1.75450, 1.74643]
- **cv_mean:** 1.76031
- **cv_std:** 0.01057
- **bias_mean:** -0.05532
- **best_offset:** -0.05532
- **cv_mean_calib:** 1.75933
- **delta_vs_b0:** Unknown / not recoverable from repository history
- **runtime_s:** Unknown / not recoverable from repository history
- **verdict:** FAIL
- **conclusion:** Capacity-matched dense wCV 1.752339 vs baseline 1.751076: +0.001263, 0/4 folds. Weighted AUC -0.000416; rec_buy 15-60 RMSLE/AUC +0.001557/-0.001208; 2-15 purchase days +0.001512/-0.000935. Train rows within +/-1% each fold. Dense supervision gate for HDN/TCN failed; details exp_022.
- **wcv:** 1.75234
- **fold_cal:** [1.77172, 1.76582, 1.75414, 1.74564]
- **mean_z:** 2.68095
- **lb_public:** Unknown / not recoverable from repository history
