# EXP071_ETX_FRESH_CONTRAST

## Catalogue metadata

- **Catalogue ID:** `new_direction__exp071_etx_fresh_contrast`
- **Namespace:** `new_direction`
- **Experiment ID:** `EXP071_ETX_FRESH_CONTRAST`
- **Original source:** `research/new_directions/EXP071_ETX_FRESH_CONTRAST`
- **Source ref:** `origin/team-a late research package`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** late research direction / experiment package
- **Model:** dilated TCN, sequence model
- **Features:** freshness/conditional features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** ## 5. Full fold and wCV metrics
- **Known score:** Unknown / not recoverable from repository history
- **Seed:** The OOF raw file is explicitly marked `PILOT_ONLY_SEED42` after a pilot rejection. TEST and candidate artifacts are not fabricated when production is not authorized.
- **Postprocessing:** None documented
- **Submission:** EXTRA contributes only positive conditional-amount rows from the opposite splitmix64 user side. It never updates the ETX encoder, tokenizer, zero/nonzero probability, EXP-037 components, validation labels, eligibility, or normalization. Production status: `{"status": "SKIPPED", "reason": "REJECT_PILOT", "rule": "production inference is authorized only after provisional PASS TYPE A/B", "public_lb_used": false, "submission_uploaded": false}`. Public LB use and upload are both false.
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when the data/frozen artifacts named by the report are present
- **Notes:** Directory-level audit unit: 19 files, 1 launcher/helper scripts, 1 preserved report documents. Numeric claims are copied from those reports.

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# EXP071_ETX_FRESH_CONTRAST

## 1. Verdict

**REJECT_PILOT** — **DO_NOT_ADD**.

## 2. Exact hypothesis

A frozen ETX-01 seed-42 final query-token embedding may encode amount timing differently from the historical frozen TCN/D3A embedding, making `mu_ETX_FRESH - mu_ETX_CLEAN` both more useful for EXP-037 and more orthogonal to the existing SEQ-FRESH correction.

## 3. Encoder/checkpoint parity

See `encoder_parity.json`. All four fold configs and the TEST config match the registered ETX-01 architecture. The external hook replays the original forward and returns the 128-dimensional final normalized query token without changing weights. Hard hook parity status: **PASS**, with hook-vs-original error `0.0`. The archived 2025-10-16 OOF array also replays exactly. The prior saved TEST DCW array shows recorded bf16/SDPA runtime drift (max `0.0625`, RMS `0.0026638` in direct ETX log output); this drift is between runtimes, while the hook and original forward in the current frozen runtime are identical.

## 4. Pilot decision

Pilot status: **REJECT_PILOT**. Metrics: `{"exp037": 1.7412785664162476, "etx_real": 1.7413885973157768, "etx_vol": 1.7411954693233225, "seq_fresh": 1.7410251443249476, "seq_plus_etx_orth": 1.7412377317952712, "real_delta": 0.00011003089952921208, "vol_delta": -8.309709292508849e-05, "real_minus_vol": 0.00019312799245430057, "orth_incremental_delta": 0.0002125874703235997}`. The gate uses fixed unit scale; the displayed alpha grid is diagnostic and was not used to select on the held-out fold.

## 5. Full fold and wCV metrics

The full four-fold/wCV phase was **not run**, exactly as required by the pilot gate. The retained `fold_metrics.csv` rows are pilot-only diagnostics, not a four-fold estimate.

## 6. REAL vs VOL evidence

See `real_vs_vol.csv` and `user_half_metrics.csv`. Equal-volume rows use the exact historical earliest-third CLEAN-positive resampling rule with canonical RNG seed 42 and equal optimization steps.

## 7. ETX-FRESH vs existing SEQ-FRESH

See `seq_vs_etx_fresh.csv`. Existing SEQ-FRESH is read unchanged from `06_ALIGNED_OOF.pred_fresh_contrast`.

## 8. Incremental orthogonal component

Pilot gamma was estimated without targets from the three donor-panel correction vectors. The pilot comparison used fixed unit scale; nested beta selection was not authorized after rejection.

## 9. OOF correction diversity

`diversity_oof.csv` contains pilot-only correction correlations. The full donor-fold projection and unexplained-variance endpoint were not run; `oof_projection_metrics.json` records the early stop.

## 10. TEST distance outside the current geometry span

See `test_span_projection.json`. TEST conditional-head inference is skipped unless OOF evidence provisionally satisfies PASS TYPE A or TYPE B.

## 11. Leakage and production-regime audits

EXTRA contributes only positive conditional-amount rows from the opposite splitmix64 user side. It never updates the ETX encoder, tokenizer, zero/nonzero probability, EXP-037 components, validation labels, eligibility, or normalization. Production status: `{"status": "SKIPPED", "reason": "REJECT_PILOT", "rule": "production inference is authorized only after provisional PASS TYPE A/B", "public_lb_used": false, "submission_uploaded": false}`. Public LB use and upload are both false.

## 12. Runtime and disk

`{"finalize_process_seconds": 0.008026838302612305, "experiment_elapsed_seconds": 859.3315832614899, "pilot_runtime_seconds": 613.4752097129822, "persistent_bytes": 1671536, "persistent_gb": 0.001671536, "disk_limit_gb": 6, "within_disk_limit": true, "removed_cache_bytes": 1419739163, "platform": "Windows-11-10.0.26200-SP0"}`. Temporary embedding caches were removed during finalization.

## 13. Standardized artifacts and SHA256

Hashes are in `checksums.sha256`; source provenance and checkpoint hashes are in `artifact_manifest.csv`.

- `C:\Users\Admin\Desktop\e-cup-research-clean\research\new_directions\EXP071_ETX_FRESH_CONTRAST\etx_fresh_raw_OOF.parquet` — SHA256 `8632235409d4dc9306e74908e38df1985450566de06bc3f037c1e18423b69757`
- `C:\Users\Admin\Desktop\e-cup-research-clean\research\new_directions\EXP071_ETX_FRESH_CONTRAST\etx_fresh_raw_TEST.parquet` — **NOT PRODUCED** (REJECT_PILOT early-stop policy)
- `C:\Users\Admin\Desktop\e-cup-research-clean\research\new_directions\EXP071_ETX_FRESH_CONTRAST\etx_fresh_contrast_OOF.parquet` — **NOT PRODUCED** (REJECT_PILOT early-stop policy)
- `C:\Users\Admin\Desktop\e-cup-research-clean\research\new_directions\EXP071_ETX_FRESH_CONTRAST\etx_fresh_contrast_TEST.parquet` — **NOT PRODUCED** (REJECT_PILOT early-stop policy)
- `C:\Users\Admin\Desktop\e-cup-research-clean\research\new_directions\EXP071_ETX_FRESH_CONTRAST\etx_fresh_contrast_TEST.csv` — **NOT PRODUCED** (REJECT_PILOT early-stop policy)

The OOF raw file is explicitly marked `PILOT_ONLY_SEED42` after a pilot rejection. TEST and candidate artifacts are not fabricated when production is not authorized.

## 14. Recommendation

**DO_NOT_ADD**. No leaderboard upload was made and geometry weights were not refit.
