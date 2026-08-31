# LATEST-DELTA-COMPATIBILITY — prerequisite report

## Verdict

`BLOCKED_NO_CANONICAL_LATEST_OOF`

Primary experiment was stopped before candidate construction. No model was
trained, no OOF was reconstructed, no public-LB information was used for a
decision, and no submission was created.

## Required authoritative input

The registered prerequisite requires one exact canonical four-fold OOF object
with the following aligned columns:

```text
z_latest
z_STRONGEST_CURRENT
target
user_id
fold
```

It must represent raw log-space predictions before fold calibration. Exact
alignment must be provable by `(fold,user_id)`.

## What was found

- No path or content reference named `AUTHORITATIVE-LATEST-INTEGRATION` exists
  in the current worktree.
- The previous integration audit is `exp_065 / FINAL_INTEGRATION_EXP065`. It
  independently rebuilds `STRONGEST_CURRENT` and packages BTYD05, but does not
  provide OOF `latest`. Its manifest SHA256 is
  `442ba96c2ed536d8e0684d2a4daad4aed0b444ea63ff72abbb441aa63ceae5e9`.
- Before writing this result, a header inventory covered 627 NPZ files and 83
  Parquet files under `artifacts/`, `research/`, and `weights_archives/`. None
  contained a field named `z_latest`; none contained the required five-field
  schema. The subsequently created `oof_candidates.npz` is an explicit
  zero-row blocked marker, not a canonical input.
- The teammate bundle documents production-only `latest`:
  `0.12*STRONGEST_CURRENT + 0.16*occ_meta_B + 0.72*occ_raw_X3`, followed by a
  zero floor. It contains test/production CSV components and public LB, not the
  required canonical four-fold OOF.

## Why reconstruction is forbidden

Test predictions have no validation targets or four-fold donor/holdout
identity. Public LB is a single aggregate and cannot recover row-level OOF.
Building `latest` from clean-fold occurrence candidates would also invent a
different recipe: project records say those clean-fold candidates are worse
than `exp_037` on 4/4 folds, while production weights were LB-calibrated.

Therefore no test array, public score, native second-line folds, or inferred
occurrence recipe was promoted to project wCV.

## Family status

| Family | Status | Reason |
|---|---|---|
| SAFE-ANCHOR | BLOCKED_NOT_RUN | `d_anchor=z_STRONGEST_CURRENT-z_latest` is undefined without canonical `z_latest`. |
| BTYD05 | BLOCKED_NOT_RUN | Exact exp_051 BTYD05 exists, but its delta cannot be evaluated against missing canonical `latest`. |
| FRESH | BLOCKED_NOT_RUN | Exact exp_040 outer correction exists; alignment to canonical `latest` cannot be checked. |
| SEQ65 | BLOCKED_NOT_RUN | Exact fixed recipe can be restored relative to STRONGEST, but the new base is missing. |
| Negative control | BLOCKED_NOT_RUN | A control without the real base/targets would not define a decision comparison. |

No fixed alpha curve, outer-fold alpha, held-out delta, nested wCV delta, fold
sign, 2025-10-16 result, residual correlation, decomposition, AUC, segment,
hash-half, or occurrence-direction correlation is reported. Empty values in the
CSV artifacts mean **not evaluated**, not zero.

## Test regime and private safety

Validation gate was never reached, so test-regime audit is not authorized.
`latest` effective CAP lineage is not established by an authoritative audit.
The fact that 12% of the documented production recipe is
`STRONGEST_CURRENT` does not establish private safety of the remaining 88%,
and adding BTYD/FRESH/SEQ65 would not fix that unknown lineage.

## Unblock contract

Resume only when an authoritative artifact supplies the five required fields
for all four canonical folds and passes:

1. unique `(fold,user_id)` rows and exact row count/support;
2. exact target and `STRONGEST_CURRENT` replay against canonical project OOF;
3. finite raw log-space `z_latest` before fold calibration;
4. registered provenance for the occurrence components and effective CAP
   lineage;
5. production arrays for any family that later passes the validation gate.

At that point the frozen alpha grids and LOFO rules in the task can be applied
without changing this blocked result retroactively.
