# Logged run — AUTHORITATIVE_LATEST_EXP067

## Catalogue metadata

- **Catalogue ID:** `team_a_run__authoritative_latest_exp067`
- **Namespace:** `team_a_run`
- **Experiment ID:** `AUTHORITATIVE_LATEST_EXP067`
- **Original source:** `experiments/log.csv`
- **Source ref:** `working tree`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** logged run / arm
- **Model:** blend
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** wcv:** Unknown / not recoverable from repository history
- **Known score:** conclusion:** TEST reconstruction PASS max log error 8.88e-16; source SHA 7ef5b2c5...e722, rebuilt SHA a9dc2dab...a1. LB 1.6492175622 is EXTERNALLY_REPORTED only; CAP_LINEAGE UNKNOWN; canonical OOF missing for B/X3, so latest is not CV/LOFO/private-safe anchor. Details exp_067
- **Seed:** Seed from src/config.py unless the preserved card explicitly states otherwise
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** PARTIAL: exact parameters are preserved in the log row; model artifacts may be external/ignored
- **Notes:** Run-level entry; a numbered experiment card may document the wider experiment family.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# Logged run — AUTHORITATIVE_LATEST_EXP067

This run was recovered from `experiments/log.csv`.

- **exp_id:** AUTHORITATIVE_LATEST_EXP067
- **timestamp:** 2026-08-25T16:08:05+03:00
- **commit:** a28a71f
- **description:** exp_067 exact latest TEST reconstruction plus LB/CAP/OOF provenance audit
- **scenario:** production-integration-audit
- **n_features:** 0
- **model:** frozen-log-blend-audit
- **params:** {"recipe":{"friend":0.12,"occ_meta_B":0.16,"occ_raw_X3":0.72},"space":"log1p","floor":"max(z,0)","training":"NONE","prefix":"AUTHORITATIVE_LATEST_AUDIT_20260825_160144_V2"}
- **cutoffs:** test 2026-02-13; canonical val required 2025-09-04/09-18/10-02/10-16
- **L:** Unknown / not recoverable from repository history
- **panel_blocks:** Unknown / not recoverable from repository history
- **fold_scores:** Unknown / not recoverable from repository history
- **cv_mean:** Unknown / not recoverable from repository history
- **cv_std:** Unknown / not recoverable from repository history
- **bias_mean:** Unknown / not recoverable from repository history
- **best_offset:** Unknown / not recoverable from repository history
- **cv_mean_calib:** Unknown / not recoverable from repository history
- **delta_vs_b0:** Unknown / not recoverable from repository history
- **runtime_s:** 14.0
- **verdict:** CONTINUE_PROVENANCE
- **conclusion:** TEST reconstruction PASS max log error 8.88e-16; source SHA 7ef5b2c5...e722, rebuilt SHA a9dc2dab...a1. LB 1.6492175622 is EXTERNALLY_REPORTED only; CAP_LINEAGE UNKNOWN; canonical OOF missing for B/X3, so latest is not CV/LOFO/private-safe anchor. Details exp_067
- **wcv:** Unknown / not recoverable from repository history
- **fold_cal:** Unknown / not recoverable from repository history
- **mean_z:** 2.329302564
- **lb_public:** 1.6492175622
