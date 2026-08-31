# Logged run — S04-SEQ-FRESH

## Catalogue metadata

- **Catalogue ID:** `team_a_run__s04_seq_fresh`
- **Namespace:** `team_a_run`
- **Experiment ID:** `S04-SEQ-FRESH`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** logged run / arm
- **Model:** dilated TCN, sequence model
- **Features:** freshness/conditional features, history-depth features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** description:** exp_032 S04-SEQ: conditional intensity head на CLEAN+EXTRA поверх замороженного энкодера, 4 фолда x 3 сида головы, метрика на группе A
- **Known score:** conclusion:** FRESH-CLEAN -0.00128 wCV(группа A) на 4/4 фолдах, 3/3 сида головы; контроль объёма -0.00008; отравление интенсива +0.006 (на 09-04 -0.124 = уровень модели, форма общая)
- **Seed:** conclusion:** FRESH-CLEAN -0.00128 wCV(группа A) на 4/4 фолдах, 3/3 сида головы; контроль объёма -0.00008; отравление интенсива +0.006 (на 09-04 -0.124 = уровень модели, форма общая)
- **Postprocessing:** conclusion:** FRESH-CLEAN -0.00128 wCV(группа A) на 4/4 фолдах, 3/3 сида головы; контроль объёма -0.00008; отравление интенсива +0.006 (на 09-04 -0.124 = уровень модели, форма общая)
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — S04-SEQ-FRESH

This run was recovered from `experiments/log.csv`.

- **exp_id:** S04-SEQ-FRESH
- **timestamp:** 2026-08-19T20:20:00
- **commit:** a28a71f
- **description:** exp_032 S04-SEQ: conditional intensity head на CLEAN+EXTRA поверх замороженного энкодера, 4 фолда x 3 сида головы, метрика на группе A
- **scenario:** S1
- **n_features:** 17
- **model:** tcn+head
- **params:** {"encoder": "SEQ-D3A-BASE-S42-V* frozen", "head": "Linear(192,64)-GELU-Drop-Linear(64,1)", "lr": 0.001, "wd": 0.01, "batch": 8192, "epochs": 4, "head_seeds": [42, 43, 44], "extra_cutoffs": 13, "extra_depth_clip": 289, "user_split": "splitmix64(user_id)&1", "control": ["COND-CLEAN", "COND-VOL"]}
- **cutoffs:** CLEAN 18..24 + EXTRA 13 @ step 7
- **L:** Unknown / not recoverable from repository history
- **panel_blocks:** 3
- **fold_scores:** [1.7688, 1.76318, 1.7505, 1.73888]
- **cv_mean:** Unknown / not recoverable from repository history
- **cv_std:** Unknown / not recoverable from repository history
- **bias_mean:** Unknown / not recoverable from repository history
- **best_offset:** Unknown / not recoverable from repository history
- **cv_mean_calib:** Unknown / not recoverable from repository history
- **delta_vs_b0:** -0.00128
- **runtime_s:** Unknown / not recoverable from repository history
- **verdict:** CONTINUE
- **conclusion:** FRESH-CLEAN -0.00128 wCV(группа A) на 4/4 фолдах, 3/3 сида головы; контроль объёма -0.00008; отравление интенсива +0.006 (на 09-04 -0.124 = уровень модели, форма общая)
- **wcv:** 1.74721
- **fold_cal:** [1.7688, 1.76318, 1.7505, 1.73888]
- **mean_z:** Unknown / not recoverable from repository history
- **lb_public:** Unknown / not recoverable from repository history
