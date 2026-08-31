# Logged run — S1-DIST-F4

## Catalogue metadata

- **Catalogue ID:** `team_a_run__s1_dist_f4`
- **Namespace:** `team_a_run`
- **Experiment ID:** `S1-DIST-F4`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** logged run / arm
- **Model:** calibration diagnostic
- **Features:** 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** description:** S1-DIST, обученная только на обучающей выборке фолда 2025-10-16 (24 cutoff'а вместо 29)
- **Known score:** conclusion:** Только тестовая модель, отдельной CV нет: обучена на той же выборке, что и фолд 2025-10-16, поэтому её единственный валидационный скор — 1.74491 (калибр. 1.74427) на этом фолде. 1.744 — свойство ФОЛДА, а не модели. Отправлены оба сабмита (чистая модель и контрольная смесь), лучший дал LB 1.6512012383165489 — хуже S1-DIST-MIX на +0.00042. Отказ от пяти самых свежих cutoff'ов стоит примерно 0.0004 на LB, обе закономерности проекта (плотная сетка, свежие cutoff'ы) устояли. Ключевое: э
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** conclusion:** Только тестовая модель, отдельной CV нет: обучена на той же выборке, что и фолд 2025-10-16, поэтому её единственный валидационный скор — 1.74491 (калибр. 1.74427) на этом фолде. 1.744 — свойство ФОЛДА, а не модели. Отправлены оба сабмита (чистая модель и контрольная смесь), лучший дал LB 1.6512012383165489 — хуже S1-DIST-MIX на +0.00042. Отказ от пяти самых свежих cutoff'ов стоит примерно 0.0004 на LB, обе закономерности проекта (плотная сетка, свежие cutoff'ы) устояли. Ключевое: э
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — S1-DIST-F4

This run was recovered from `experiments/log.csv`.

- **exp_id:** S1-DIST-F4
- **timestamp:** 2026-08-11T21:09:59
- **commit:** 521835d
- **description:** S1-DIST, обученная только на обучающей выборке фолда 2025-10-16 (24 cutoff'а вместо 29)
- **scenario:** S1
- **n_features:** 227
- **model:** dist
- **params:** {"L": null, "cutoffs": "train_for_fold 2025-10-16", "min_history": 90, "model": "dist", "n_bins": 16, "norm_long": true, "rounds": 250, "step": 7, "train_blocks": 1}
- **cutoffs:** 24 @ step 7 (2025-04-03..2025-09-11)
- **L:** Unknown / not recoverable from repository history
- **panel_blocks:** 3
- **fold_scores:** [1.74491]
- **cv_mean:** Unknown / not recoverable from repository history
- **cv_std:** Unknown / not recoverable from repository history
- **bias_mean:** Unknown / not recoverable from repository history
- **best_offset:** Unknown / not recoverable from repository history
- **cv_mean_calib:** 1.74427
- **delta_vs_b0:** Unknown / not recoverable from repository history
- **runtime_s:** 1242.0
- **verdict:** REJECT
- **conclusion:** Только тестовая модель, отдельной CV нет: обучена на той же выборке, что и фолд 2025-10-16, поэтому её единственный валидационный скор — 1.74491 (калибр. 1.74427) на этом фолде. 1.744 — свойство ФОЛДА, а не модели. Отправлены оба сабмита (чистая модель и контрольная смесь), лучший дал LB 1.6512012383165489 — хуже S1-DIST-MIX на +0.00042. Отказ от пяти самых свежих cutoff'ов стоит примерно 0.0004 на LB, обе закономерности проекта (плотная сетка, свежие cutoff'ы) устояли. Ключевое: эта дельта СТРУКТУРНО НЕИЗМЕРИМА локально — обучающая выборка фолда 10-16 и есть эти 24 cutoff'а. Детали exp_015, правило в exp_016 §5
- **wcv:** Unknown / not recoverable from repository history
- **fold_cal:** Unknown / not recoverable from repository history
- **mean_z:** Unknown / not recoverable from repository history
- **lb_public:** 1.6512012 (лучший из двух)
