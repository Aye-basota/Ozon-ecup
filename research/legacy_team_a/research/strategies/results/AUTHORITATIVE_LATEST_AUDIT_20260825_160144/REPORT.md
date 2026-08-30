# AUTHORITATIVE-LATEST-INTEGRATION audit

## Verdict

**CONTINUE_PROVENANCE**

`latest.csv` is numerically reconstructed from the frozen component submissions,
but the public score is only externally reported and canonical row-level OOF for
the two late occurrence/Ridge components is absent from the bundle.

## Production-state classification

- `best_public_observed`: `latest/latest.csv`, public LB `1.64921756224069`, **EXTERNALLY_REPORTED**; LB event date is not recorded, evidence was present by the 2026-08-24 provenance audit.
- `best_exactly_reproducible`: `STRONGEST_CURRENT` / `exp_037` (end-to-end package, recorded LB, canonical four-fold OOF).
- `research_private_safe_anchor`: `STRONGEST_CURRENT` / `exp_037`.

## Test reconstruction

- Recipe: `0.12 * friend + 0.16 * occ_meta_B + 0.72 * occ_raw_X3` in `z=log1p(predict)` space.
- Full policy: component nonnegative validation, convex log blend, `z=max(z,0)`, `predict=expm1(z)`; no post-blend level normalization.
- Rows/schema/order: `250000` rows, `user_id,predict`, exact sample order.
- Source SHA-256: `7ef5b2c58925bd28c5bc7eb83b9cfd4785c608a0c8b2a6d7a3277730cba8e722`.
- Reconstructed SHA-256: `a9dc2dabdc693cd510c0428c501154898ec11f1e15b0694261471757cae274a1`.
- Byte-identical: `false` (CSV writer formatting is recorded separately from numeric equality).
- Required max error: `8.8817841970012523e-16` (floor `5.0e-07`).
- Reconstructed roundtrip max error: `4.4408920985006262e-16`.

## Lineage and CAP

`friend.csv` is byte-identical to `STRONGEST_CURRENT` and contains CAP at 10%.
Both late submissions preserve the 45% fixed SEQ/ETX slot and replace the 55%
table slot through `friend + 0.55 * (candidate_table - table_core)`, followed by
fixed level handling. Expanding the final blend proves 45% shared neural anchor,
6.6% original table core, 8.8% `occ_meta_B` table candidate and 39.6%
`occ_raw_X3` table candidate. The directly fixed CAP coefficient is therefore
1.2%, while extra CAP dependence inside the learned candidate tables cannot be
reduced to a documented scalar from the supplied artifacts.

```text
CAP_LINEAGE = UNKNOWN
PRIVATE_SAFE_STATUS = UNRESOLVED
```

The three latest components are not independent models: both occurrence
components share the same fixed neural anchor and Ridge/greedy table ancestry.
This is intentional table-slot replacement, but it is double use of common
ancestry rather than three independent signals.

## OOF

`friend` canonical OOF is available and aligned on the four project folds.
Canonical row-level OOF for `occ_meta_B` and `occ_raw_X3` is missing; the bundle
explicitly omits the multi-GB research cache. Summary validation CSVs cannot
replace row-level OOF and cannot prove target equality or absence of in-sample
stacking for these exact production CSVs.

```text
CANONICAL_OOF = MISSING
```

No project wCV, segment diagnostics, canonical latest NPZ, or model-dependent
level audit was synthesized.

## LB provenance

The number `1.64921756224069` occurs in textual README/provenance documents and
is described there as coming from a transmitted external journal. It does not
occur in the three relevant `RUN_MANIFEST.json` files or a SHA-to-score registry.
Accordingly it is classified as `EXTERNALLY_REPORTED`, not independently verified.
