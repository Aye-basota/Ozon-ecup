# Logged run — S04-PROD-FRESH

## Catalogue metadata

- **Catalogue ID:** `team_a_run__s04_prod_fresh`
- **Namespace:** `team_a_run`
- **Experiment ID:** `S04-PROD-FRESH`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** logged run / arm
- **Model:** dilated TCN, sequence model
- **Features:** freshness/conditional features, history-depth features, 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** description:** exp_032b EXP-032B: боевой экстенсив 1-p0 из S1-DIST вместо переобученной головы P(y>0); тот же mu_FRESH из exp_032, 4 фолда x 3 сида головы, метрика на группе A
- **Known score:** wcv:** 1.74514
- **Seed:** params:** {"P_prod": "S1-DIST 1-p0 (exp_014 \u043a\u043e\u043d\u0444\u0438\u0433, 250 \u0440\u0430\u0443\u043d\u0434\u043e\u0432)", "encoder": "SEQ-D3A-BASE-S42-V* \u0437\u0430\u043c\u043e\u0440\u043e\u0436\u0435\u043d", "extra_depth_clip": 289, "head_seeds": [42, 43, 44], "metric_group": "A (splitmix64 & 1 == 0)"}
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — S04-PROD-FRESH

This run was recovered from `experiments/log.csv`.

- **exp_id:** S04-PROD-FRESH
- **timestamp:** 2026-08-19T21:29:30
- **commit:** a28a71f
- **description:** exp_032b EXP-032B: боевой экстенсив 1-p0 из S1-DIST вместо переобученной головы P(y>0); тот же mu_FRESH из exp_032, 4 фолда x 3 сида головы, метрика на группе A
- **scenario:** S1
- **n_features:** 227
- **model:** tcn+dist
- **params:** {"P_prod": "S1-DIST 1-p0 (exp_014 \u043a\u043e\u043d\u0444\u0438\u0433, 250 \u0440\u0430\u0443\u043d\u0434\u043e\u0432)", "encoder": "SEQ-D3A-BASE-S42-V* \u0437\u0430\u043c\u043e\u0440\u043e\u0436\u0435\u043d", "extra_depth_clip": 289, "head_seeds": [42, 43, 44], "metric_group": "A (splitmix64 & 1 == 0)"}
- **cutoffs:** 18/20/22/24 @ step 7 + 13 EXTRA только в интенсив
- **L:** 0
- **panel_blocks:** 3
- **fold_scores:** [1.76808, 1.76082, 1.74744, 1.73721]
- **cv_mean:** Unknown / not recoverable from repository history
- **cv_std:** Unknown / not recoverable from repository history
- **bias_mean:** Unknown / not recoverable from repository history
- **best_offset:** Unknown / not recoverable from repository history
- **cv_mean_calib:** Unknown / not recoverable from repository history
- **delta_vs_b0:** Unknown / not recoverable from repository history
- **runtime_s:** 5314
- **verdict:** PASS (гейт), но в смесь не годится
- **conclusion:** FRESH-CLEAN -0.00101 4/4; налог двухчастной схемы +0.00088 -> -0.00146; итог против BASE-1HEAD -0.00247 4/4. НО: подстановка в боевую смесь при фиксированных весах хуже на 0/4 в слоте SEQ (разнообразие Var(z-z_tab) 0.041 -> 0.015, corr остатков 0.9933 -> 0.9975), в слоте DIST -0.00005 при поле 0.0005. Интенсив SEQ хуже интенсива донора на +0.00257, 0/4. Дальше — fine-tune энкодера SEQ-D3A на conditional loss, а не гибрид.
- **wcv:** 1.74514
- **fold_cal:** [1.76808, 1.76082, 1.74744, 1.73721]
- **mean_z:** Unknown / not recoverable from repository history
- **lb_public:** Unknown / not recoverable from repository history
