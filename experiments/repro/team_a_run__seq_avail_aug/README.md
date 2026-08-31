# Logged run — SEQ-AVAIL-AUG

## Catalogue metadata

- **Catalogue ID:** `team_a_run__seq_avail_aug`
- **Namespace:** `team_a_run`
- **Experiment ID:** `SEQ-AVAIL-AUG`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** logged run / arm
- **Model:** dilated TCN, sequence model
- **Features:** history-depth features, 227 tabular features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** wcv:** Unknown / not recoverable from repository history
- **Known score:** conclusion:** OOD устранён, но gate провален. BASE/B/A25 RMSLE_cal 1.74808/1.74986/1.74913; AUC 0.84248/0.84123/0.84190. availprobe +0.00651 -> -0.00020/-0.00001, край availcurve стал плоским. Var(z_aug-z_base)=0.03236/0.03044 при seed-pair 0.02270; corr остатков 0.99471/0.99502. Cross-depth 212->289 (+77 дней): gain -0.00308 у BASE сжался до -0.00165/-0.00206, optimum 275 -> 261/254. Инвариантность куплена частичным игнорированием длинной истории. STOP: без 4 folds, 3 seeds, LOFO и LB-submit; p
- **Seed:** conclusion:** OOD устранён, но gate провален. BASE/B/A25 RMSLE_cal 1.74808/1.74986/1.74913; AUC 0.84248/0.84123/0.84190. availprobe +0.00651 -> -0.00020/-0.00001, край availcurve стал плоским. Var(z_aug-z_base)=0.03236/0.03044 при seed-pair 0.02270; corr остатков 0.99471/0.99502. Cross-depth 212->289 (+77 дней): gain -0.00308 у BASE сжался до -0.00165/-0.00206, optimum 275 -> 261/254. Инвариантность куплена частичным игнорированием длинной истории. STOP: без 4 folds, 3 seeds, LOFO и LB-submit; p
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — SEQ-AVAIL-AUG

This run was recovered from `experiments/log.csv`.

- **exp_id:** SEQ-AVAIL-AUG
- **timestamp:** 2026-08-14T03:20:00
- **commit:** 560b24b
- **description:** exp_029: train-only augmentation границы avail; быстрый gate BASE/B/A25 на fold 2025-10-16 seed 42
- **scenario:** S1
- **n_features:** 17
- **model:** tcn-diagnostic
- **params:** {"architecture_changed": false, "base": "none", "variants": {"B": "avail_bnd p=.5 full=.5", "A25": "avail_drop p=.25"}, "seed": 42, "fold": "2025-10-16", "epochs": 4}
- **cutoffs:** 24 @ step 7
- **L:** Unknown / not recoverable from repository history
- **panel_blocks:** 3
- **fold_scores:** [1.74808, 1.74986, 1.74913]
- **cv_mean:** Unknown / not recoverable from repository history
- **cv_std:** Unknown / not recoverable from repository history
- **bias_mean:** Unknown / not recoverable from repository history
- **best_offset:** Unknown / not recoverable from repository history
- **cv_mean_calib:** Unknown / not recoverable from repository history
- **delta_vs_b0:** Unknown / not recoverable from repository history
- **runtime_s:** Unknown / not recoverable from repository history
- **verdict:** REJECT
- **conclusion:** OOD устранён, но gate провален. BASE/B/A25 RMSLE_cal 1.74808/1.74986/1.74913; AUC 0.84248/0.84123/0.84190. availprobe +0.00651 -> -0.00020/-0.00001, край availcurve стал плоским. Var(z_aug-z_base)=0.03236/0.03044 при seed-pair 0.02270; corr остатков 0.99471/0.99502. Cross-depth 212->289 (+77 дней): gain -0.00308 у BASE сжался до -0.00165/-0.00206, optimum 275 -> 261/254. Инвариантность куплена частичным игнорированием длинной истории. STOP: без 4 folds, 3 seeds, LOFO и LB-submit; production clip289. Детали exp_029.
- **wcv:** Unknown / not recoverable from repository history
- **fold_cal:** Unknown / not recoverable from repository history
- **mean_z:** Unknown / not recoverable from repository history
- **lb_public:** Unknown / not recoverable from repository history
