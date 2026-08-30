# exp_049 — corrected EXP-048 same-fold analysis / production audit

- **Дата:** 2026-08-23
- **Тип:** analysis/production only; model training = **NO**
- **Кандидат:** `BTYD05_FRESH1` (fixed weights)
- **Verdict:** **REJECT**

## Method correction

EXP-048 mixed standard 4-fold 1:2:4:8 with matched 3-fold 1:2:4, so its
selection penalty also contained removal of `2025-10-16`.  This follow-up uses
exactly `09-04/09-18/10-02`, weights `1:2:4`, for A/B/C.  C is explicitly
conditional on `k>0`; reference probabilities were renormalized after removing
`pi_ref(k=0)=0.004951` and are not called fully identified CV.

## Corrected endpoint

| scheme | delta | folds | signs |
|---|---:|---:|---|
| A_STANDARD_3F | -0.000547 | 3/3 | `['-', '-', '-']` |
| C_MATCHED_KPOS_3F | -0.000551 | 3/3 | `['-', '-', '-']` |

Bootstrap re-estimates weighted calibration inside every one of 500 cluster
replicates: `P(delta<0)=1.000` and 95% interval
`[-0.000672, -0.000426]`.

The EXP-048 shuffle contradiction was an estimand/reporting bug: the displayed
`-0.000544` was the matched candidate effect, but the interval and `outside`
boolean were computed for `matched-standard` selection shift.  The corrected
signal shuffle compares like with like and passes;
the selection-k shuffle remains a separate sensitivity diagnostic.

Missing `k=0` sensitivity is recorded in `missing_k0_sensitivity.csv`.  It is
scenario analysis only; no result is presented as fully identified matched-CV.

## Production audit

Status: **FAIL_MISSING_EXACT_PRODUCTION_SUPPORT**.  STRONGEST_CURRENT test rows/order,
finiteness, official level normalization and reconstruction all pass.  Exact
BTYD and FRESH production predictions do not exist in the authorized artifacts:
EXP-047 explicitly stopped before test inference, while EXP-040 stopped before
production inference and did not persist conditional-head weights.  Averaging
fold BTYD fits, choosing the latest fit, or carrying OOF FRESH corrections to
test would invent an unregistered production recipe.  Therefore correction
quantiles/variance ratio/support on test are not identified, the audit cannot
PASS, and no submission slot is spent.

## Final gate

Validation evidence meets the fixed statistical gate: **PREFERRED**.
Production-support audit is a hard failure, so final verdict is **REJECT**.
Submission path/hash: **not created**.

Artifacts: `research/strategies/results/SELMATCH_EXP049/`.
