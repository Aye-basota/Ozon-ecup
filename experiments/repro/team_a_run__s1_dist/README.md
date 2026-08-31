# Logged run — S1-DIST

## Catalogue metadata

- **Catalogue ID:** `team_a_run__s1_dist`
- **Namespace:** `team_a_run`
- **Experiment ID:** `S1-DIST`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** logged run / arm
- **Model:** distribution head, calibration diagnostic
- **Features:** 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** conclusion:** E0. Голова распределения лучше S1-E10 на 4 фолдах из 4: калиброванный OOF 1.75744 против 1.75889 (-0.00145). Пре-регистрированный порог >=0.003 на >=3 фолдах НЕ выполнен: эталон N12 (-0.00396) снимался на базе из 3 cutoff'ов, на плотной сетке остаётся -0.0007..-0.0019. Var(z-z_E10)=0.01320 при пороге разнообразия 0.10 — та же функция, а не новый класс. Смесь 0.15/0.30/0.10/0.45 с S1-E10/S1-E02/S1-E03a: OOF 1.75645 против 1.75716 у S1-BEST, лучше на 4 фолдах из 4. Сабмит submissions
- **Known score:** verdict:** ПРИНЯТО; LB 1.6507774106
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** conclusion:** E0. Голова распределения лучше S1-E10 на 4 фолдах из 4: калиброванный OOF 1.75744 против 1.75889 (-0.00145). Пре-регистрированный порог >=0.003 на >=3 фолдах НЕ выполнен: эталон N12 (-0.00396) снимался на базе из 3 cutoff'ов, на плотной сетке остаётся -0.0007..-0.0019. Var(z-z_E10)=0.01320 при пороге разнообразия 0.10 — та же функция, а не новый класс. Смесь 0.15/0.30/0.10/0.45 с S1-E10/S1-E02/S1-E03a: OOF 1.75645 против 1.75716 у S1-BEST, лучше на 4 фолдах из 4. Сабмит submissions
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — S1-DIST

This run was recovered from `experiments/log.csv`.

- **exp_id:** S1-DIST
- **timestamp:** 2026-08-11T19:52:11
- **commit:** 0ef1396
- **description:** E0: голова распределения (multiclass 16 бинов z, ẑ=sum p_k m_k) на признаках S1-E10, 250 раундов
- **scenario:** S1
- **n_features:** 227
- **model:** dist
- **params:** {"L": null, "cutoffs": "all", "min_history": 90, "model": "dist", "n_bins": 16, "norm_long": true, "panel_blocks": 3, "rounds": 250, "step": 7, "train_blocks": 1}
- **cutoffs:** 24 @ step 7
- **L:** Unknown / not recoverable from repository history
- **panel_blocks:** 3
- **fold_scores:** [1.77183, 1.7644, 1.7522, 1.74491]
- **cv_mean:** 1.75834
- **cv_std:** 0.01045
- **bias_mean:** -0.05147
- **best_offset:** -0.05000
- **cv_mean_calib:** 1.75744
- **delta_vs_b0:** Unknown / not recoverable from repository history
- **runtime_s:** 3952.0
- **verdict:** ПРИНЯТО; LB 1.6507774106
- **conclusion:** E0. Голова распределения лучше S1-E10 на 4 фолдах из 4: калиброванный OOF 1.75744 против 1.75889 (-0.00145). Пре-регистрированный порог >=0.003 на >=3 фолдах НЕ выполнен: эталон N12 (-0.00396) снимался на базе из 3 cutoff'ов, на плотной сетке остаётся -0.0007..-0.0019. Var(z-z_E10)=0.01320 при пороге разнообразия 0.10 — та же функция, а не новый класс. Смесь 0.15/0.30/0.10/0.45 с S1-E10/S1-E02/S1-E03a: OOF 1.75645 против 1.75716 у S1-BEST, лучше на 4 фолдах из 4. Сабмит submissions/submission_dist_head.csv -> LB 1.6507774106 против 1.6512803 (-0.0005029), ЛУЧШИЙ результат проекта. Перенос OOF->LB оказался 0.64x, а не 1.13-1.20x по R10: константа калибровалась на ухудшениях и на улучшение не переносится. Детали exp_014
- **wcv:** 1.75062
- **fold_cal:** [1.77015, 1.76382, 1.75183, 1.74426]
- **mean_z:** 2.67693
- **lb_public:** 1.6507774 (в смеси S1-DIST-MIX)
