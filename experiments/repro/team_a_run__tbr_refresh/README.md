# Logged run — TBR-REFRESH

## Catalogue metadata

- **Catalogue ID:** `team_a_run__tbr_refresh`
- **Namespace:** `team_a_run`
- **Experiment ID:** `TBR-REFRESH`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** logged run / arm
- **Model:** sequence model, blend
- **Features:** freshness/conditional features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** wcv:** 1.747507416
- **Known score:** wcv:** 1.747507416
- **Seed:** params:** {"components": ["UNC", "CAP"], "rounds": [200, 250, 300, 600], "primary_round": 300, "seeds": [42, 43, 44], "weights": {"CAP": 0.1, "UNC": 0.2, "S1-DIST": 0.25, "ETX-AVG3": 0.225, "SEQ-AVG3": 0.225}, "replay": "bitwise", "early_stopping": false}
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — TBR-REFRESH

This run was recovered from `experiments/log.csv`.

- **exp_id:** TBR-REFRESH
- **timestamp:** 2026-08-23T17:30:00
- **commit:** a28a71f
- **description:** exp_046 controlled production refresh of UNC/CAP: fixed 300 rounds and AVG3 seeds 42/43/44 inside exact STRONGEST_CURRENT
- **scenario:** S1
- **n_features:** 236/195
- **model:** direct+fixed-blend
- **params:** {"components": ["UNC", "CAP"], "rounds": [200, 250, 300, 600], "primary_round": 300, "seeds": [42, 43, 44], "weights": {"CAP": 0.1, "UNC": 0.2, "S1-DIST": 0.25, "ETX-AVG3": 0.225, "SEQ-AVG3": 0.225}, "replay": "bitwise", "early_stopping": false}
- **cutoffs:** 18/20/22/24 @ step 7
- **L:** UNC=None;CAP=180
- **panel_blocks:** 3
- **fold_scores:** Unknown / not recoverable from repository history
- **cv_mean:** Unknown / not recoverable from repository history
- **cv_std:** Unknown / not recoverable from repository history
- **bias_mean:** Unknown / not recoverable from repository history
- **best_offset:** Unknown / not recoverable from repository history
- **cv_mean_calib:** 1.75431
- **delta_vs_b0:** -0.000002
- **runtime_s:** 4741.4
- **verdict:** REJECT
- **conclusion:** Historical H and fresh A match bitwise. B-A AVG3@600 -0.000052560 4/4; C-A rounds-only +0.000033136 1/4; primary D-A/H -0.000002447 3/4 and 10-16 -0.000004781; interaction +0.000016977. No production/test/submission; no basis for DIST-AVG3. Details exp_046
- **wcv:** 1.747507416
- **fold_cal:** [1.766828342, 1.760479849, 1.748658229, 1.741273786]
- **mean_z:** 2.687775865
- **lb_public:** Unknown / not recoverable from repository history
