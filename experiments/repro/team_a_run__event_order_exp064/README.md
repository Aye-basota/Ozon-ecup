# Logged run — EVENT_ORDER_EXP064

## Catalogue metadata

- **Catalogue ID:** `team_a_run__event_order_exp064`
- **Namespace:** `team_a_run`
- **Experiment ID:** `EVENT_ORDER_EXP064`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** logged run / arm
- **Model:** Unknown / not recoverable from repository history
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** conclusion:** Cutoff/alignment/finiteness/state-multiset tests and exact transition-count parity PASS; shuffle changes 97.62-97.77% feature rows. REAL SHUFFLED CONTROL all select scale 0 on all folds and both halves; Delta wCV=0. Full model/test/LB not run. Details exp_064
- **Known score:** conclusion:** Cutoff/alignment/finiteness/state-multiset tests and exact transition-count parity PASS; shuffle changes 97.62-97.77% feature rows. REAL SHUFFLED CONTROL all select scale 0 on all folds and both halves; Delta wCV=0. Full model/test/LB not run. Details exp_064
- **Seed:** params:** {"base":"STRONGEST_CURRENT","new_features":15,"controls":11,"rounds":120,"num_leaves":15,"min_data_in_leaf":1000,"lambda_l2":50,"scales":[0,0.25,0.5,0.75,1],"placebo":"complete 3-bit daily state permuted within user over fixed dates","split":"splitmix64(user_id) 4-way + two-sided scale","seed":42,"threads":4}
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — EVENT_ORDER_EXP064

This run was recovered from `experiments/log.csv`.

- **exp_id:** EVENT_ORDER_EXP064
- **timestamp:** 2026-08-25T03:39:00+03:00
- **commit:** a28a71f
- **description:** exp_064 explicit daily Search/Cart/Buy transition order vs exact within-user state-multiset shuffle
- **scenario:** S1-preflight
- **n_features:** 242
- **model:** artifact-only-cross-user-lgb-residual
- **params:** {"base":"STRONGEST_CURRENT","new_features":15,"controls":11,"rounds":120,"num_leaves":15,"min_data_in_leaf":1000,"lambda_l2":50,"scales":[0,0.25,0.5,0.75,1],"placebo":"complete 3-bit daily state permuted within user over fixed dates","split":"splitmix64(user_id) 4-way + two-sided scale","seed":42,"threads":4}
- **cutoffs:** val 2025-09-04/09-18/10-02/10-16
- **L:** 90
- **panel_blocks:** 3
- **fold_scores:** [1.766883356,1.760509577,1.748629224,1.741278566]
- **cv_mean:** 1.754325181
- **cv_std:** 0.009995
- **bias_mean:** Unknown / not recoverable from repository history
- **best_offset:** Unknown / not recoverable from repository history
- **cv_mean_calib:** 1.754325181
- **delta_vs_b0:** 0
- **runtime_s:** 62.95
- **verdict:** REJECT
- **conclusion:** Cutoff/alignment/finiteness/state-multiset tests and exact transition-count parity PASS; shuffle changes 97.62-97.77% feature rows. REAL SHUFFLED CONTROL all select scale 0 on all folds and both halves; Delta wCV=0. Full model/test/LB not run. Details exp_064
- **wcv:** 1.747509862
- **fold_cal:** [1.766883356,1.760509577,1.748629224,1.741278566]
- **mean_z:** Unknown / not recoverable from repository history
- **lb_public:** Unknown / not recoverable from repository history
