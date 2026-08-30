# EXP069 BTYD05 FRESH1 production report

## 1. Verdict

**PASS_TYPE_A**  
Recommendation: **ADD_TO_SUBMISSION_GEOMETRY**

## 2. Exact hypothesis

The historical positive-target FRESH conditional residual correction relative to EXP-037 contains signed signal complementary to fixed 5% stable BTYD, and an exact-semantics production rebuild can yield a useful TEST vector outside the current geometry span.

## 3. OOF baseline parity

All 770,616 unique canonical `(fold,user_id)` rows and targets align exactly. EXP-037 reconstructs from CAP/UNC/DIST/SEQ-AVG3/ETX-AVG3 with maximum log error `5.912e-08`; BTYD05 maximum log error is `1.138e-07`. Registered parity passed. EXP-037 wCV is `1.747509862493`.

## 4. Fold and wCV results

- FRESH delta wCV: `-0.000224956` (4/4; latest `-0.000253422`).
- Fixed BTYD05 delta wCV: `-0.000320983` (4/4).
- Fixed BTYD05_FRESH1: wCV `1.747042922755`, delta `-0.000466940`, folds `[-0.0008351544339089, -0.0007037874058728, -0.0003959749775164, -0.0003971833647016]`.
- User-cluster bootstrap intervals are in `bootstrap_metrics.csv`; combined 95% interval is `[-0.0005853406363249, -0.0003456223065615]`.

## 5. FRESH vs VOL control

REAL-minus-VOL is `-0.000232049` wCV. The matched-volume control is neutral while REAL improves. Both splitmix user halves have negative combined deltas: `[-0.000563384799926, -0.000370572922387]`.

## 6. BTYD/FRESH complementarity

FRESH adds `-0.000143021` nested wCV beyond fixed BTYD05, with 4/4 held-out folds improving. Fixed combined gain is materially larger than either component alone.

## 7. OOF correction diversity

Donor-fold ridge projection leaves `0.449` of combined correction variance unexplained and `0.025515` unexplained RMS. FRESH alone is `0.970` unexplained.

## 8. TEST distance outside the 65-source span

Rank-57 affine-span distance is `0.010679` RMS; orthogonal norm fraction is `0.127`. Nearest source is `candidate_B_BTYD05_HEDGE.csv` at `0.015509` RMS. Numerical rank changes `57 -> 58`.

## 9. Production and leakage audits

The encoder used only the 29 CLEAN cutoff grid through 2025-10-16. EXTRA comprises only positive-target rows at the 13 preregistered cutoffs and updates only conditional amount heads. Two splitmix donor sides and seeds 42/43/44 were averaged in log space. The encoder checksum was unchanged. Frozen OOF preprocessing parameters were applied to TEST without TEST centering or variance matching. Production regime: **PASS**; schema: **PASS**.

The leave-one-fold-out production bridge has calibrated wCV discrepancy `+0.000e+00` and preserves all four fold signs. Its maximum saved-vector RMS difference is `0.031244`; each fold difference is a pure constant offset (maximum within-fold difference SD `1.735e-17`), which the canonical fold log-offset evaluator removes exactly.

The TEST extensive probability is explicitly marked reconstructed: same CLEAN-only S1-DIST recipe, not a byte-exact recovery of the historical TEST trajectory. Its provenance and mismatch diagnostics are retained in `production_training_audit.json` and `production_regime.json`; no guessed formula or TEST target calibration was used.

## 10. Runtime and disk

- OOF analysis: `60.3s`.
- Production encoder: `5729.9s`.
- Embedding plus conditional heads: `570.3s`.
- Peak new persistent disk: `0.136 GB` (budget pass: `True`). No persistent temporary embedding caches were written.

## 11. Saved artifacts and SHA256

```json
[
  {
    "path": "C:\\Users\\Admin\\Desktop\\e-cup-research-clean\\research\\new_directions\\EXP069_BTYD05_FRESH1_PROD\\fresh_conditional_TEST.parquet",
    "bytes": 20899834,
    "sha256": "f1b8c3b65c654630abd90425956490d9ad0f2b8d165d5f1fed0da0f8feeaad91"
  },
  {
    "path": "C:\\Users\\Admin\\Desktop\\e-cup-research-clean\\research\\new_directions\\EXP069_BTYD05_FRESH1_PROD\\btyd05_fresh1_OOF.parquet",
    "bytes": 26547745,
    "sha256": "472be8cad64c43ec2c2783fea6497b20fe02ffefa0d49344e3ddb78fbb81a07b"
  },
  {
    "path": "C:\\Users\\Admin\\Desktop\\e-cup-research-clean\\research\\new_directions\\EXP069_BTYD05_FRESH1_PROD\\btyd05_fresh1_TEST.parquet",
    "bytes": 7846686,
    "sha256": "68f5f9e86110e4b7a59776997521c33f7dc80c28b918fea59d9c223825ae19b0"
  },
  {
    "path": "C:\\Users\\Admin\\Desktop\\e-cup-research-clean\\research\\new_directions\\EXP069_BTYD05_FRESH1_PROD\\btyd05_fresh1_TEST.csv",
    "bytes": 6362147,
    "sha256": "5170617dd5c538691ee107ad2732342799c298e7beb2130e9760a5fa205b13b7"
  }
]
```

Complete hashes are in `checksums.sha256`; full input provenance is in `artifact_manifest.csv`.

## 12. Recommendation

**ADD_TO_SUBMISSION_GEOMETRY**

No submission was uploaded and no public-LB equation or score was used for evaluation, selection, scaling, or level fitting.
