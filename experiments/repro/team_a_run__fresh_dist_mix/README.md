# Logged run — FRESH-DIST-MIX

## Catalogue metadata

- **Catalogue ID:** `team_a_run__fresh_dist_mix`
- **Namespace:** `team_a_run`
- **Experiment ID:** `FRESH-DIST-MIX`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** logged run / arm
- **Model:** Unknown / not recoverable from repository history
- **Features:** freshness/conditional features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** conclusion:** Все 13 поздних target-окон физически наблюдаемы, train вырос бы с 6,065,972 до 9,184,968 строк (+51.4%), но T>=2025-10-17 запрещены: 250k пользователей отобраны по гарантированной активности в 2025-11-16..2026-02-13. На 2026-01-14 P(any activity next30)=1.0 по конструкции; e08 уже дал около +0.054 RMSLE и bias +0.366 на clean holdout. По стоп-правилу обучение, late pseudo-validation на загрязнённых targets, test diagnostics и submission не выполнялись. Детали exp_028.
- **Known score:** conclusion:** Все 13 поздних target-окон физически наблюдаемы, train вырос бы с 6,065,972 до 9,184,968 строк (+51.4%), но T>=2025-10-17 запрещены: 250k пользователей отобраны по гарантированной активности в 2025-11-16..2026-02-13. На 2026-01-14 P(any activity next30)=1.0 по конструкции; e08 уже дал около +0.054 RMSLE и bias +0.366 на clean holdout. По стоп-правилу обучение, late pseudo-validation на загрязнённых targets, test diagnostics и submission не выполнялись. Детали exp_028.
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — FRESH-DIST-MIX

This run was recovered from `experiments/log.csv`.

- **exp_id:** FRESH-DIST-MIX
- **timestamp:** 2026-08-13T22:10:46
- **commit:** 560b24b
- **description:** exp_028: аудит добавления максимально свежих полностью наблюдаемых cutoff'ов в production S1-DIST-MIX
- **scenario:** S1
- **n_features:** Unknown / not recoverable from repository history
- **model:** audit-only
- **params:** {"base_cutoffs": 29, "extra_cutoffs": 13, "base_rows": 6065972, "extra_rows": 3118996, "last_cutoff": "2026-01-14", "target": "(T,T+30]", "stop_reason": "guaranteed activity panel selection contamination"}
- **cutoffs:** 29 clean + 13 contaminated (2025-10-22..2026-01-14)
- **L:** Unknown / not recoverable from repository history
- **panel_blocks:** 3
- **fold_scores:** Unknown / not recoverable from repository history
- **cv_mean:** Unknown / not recoverable from repository history
- **cv_std:** Unknown / not recoverable from repository history
- **bias_mean:** Unknown / not recoverable from repository history
- **best_offset:** Unknown / not recoverable from repository history
- **cv_mean_calib:** Unknown / not recoverable from repository history
- **delta_vs_b0:** Unknown / not recoverable from repository history
- **runtime_s:** Unknown / not recoverable from repository history
- **verdict:** REJECT
- **conclusion:** Все 13 поздних target-окон физически наблюдаемы, train вырос бы с 6,065,972 до 9,184,968 строк (+51.4%), но T>=2025-10-17 запрещены: 250k пользователей отобраны по гарантированной активности в 2025-11-16..2026-02-13. На 2026-01-14 P(any activity next30)=1.0 по конструкции; e08 уже дал около +0.054 RMSLE и bias +0.366 на clean holdout. По стоп-правилу обучение, late pseudo-validation на загрязнённых targets, test diagnostics и submission не выполнялись. Детали exp_028.
- **wcv:** Unknown / not recoverable from repository history
- **fold_cal:** Unknown / not recoverable from repository history
- **mean_z:** Unknown / not recoverable from repository history
- **lb_public:** Unknown / not recoverable from repository history
