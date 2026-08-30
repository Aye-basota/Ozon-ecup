# EXP-065 — FINAL-INTEGRATION Hypothesis Card (frozen before results)

- **Date:** 2026-08-25
- **Author:** A1
- **Frozen development reference:** `STRONGEST-CURRENT / exp_037`
- **No leaderboard send:** mandatory

## Objective

Refresh the strongest legitimate candidate after EXP-061..064, independently reconstruct its production prediction from primitive component artifacts, recheck the critical sequence-slot regime gate, and package exactly two schema-verified canonical CSVs. This is an integration/audit experiment, not a new performance claim or blend search.

## Candidate choice frozen before audit

- **A — canonical:** exact exp_037 recipe, because no session candidate passed even the development gate and the teammate public leader is explicitly public-LB calibrated and worse on all canonical absolute clean folds.
- **B — independent hedge:** fixed `BTYD05` from exp_051, because it is the strongest pre-existing non-neural production-support PASS (`fixed .05 Delta wCV=-0.000321`, 4/4, test correction variance ratio `1.1734`). It does not replace A and is below the current production research threshold.
- Public-LB-calibrated teammate blends, level probes, and neighboring sequence weights are ineligible for A/B selection.

## Exact A recipe

In `log1p` space: `.10 S1-CAP + .20 S1-UNC + .25 S1-DIST + .075 each of three SEQ clip289 seeds + .075 each of three ETX DCW seeds`; shift only to fixed mean level `2.3293`, floor at zero, `expm1`, six-decimal CSV. All nine component user arrays must match sample order exactly.

## Audits and gates

1. Rebuilt A must be byte-identical to registered exp_037 CSV SHA256 `abc2218b...e04bda`.
2. Exact schema/order, 250,000 unique users, finite/nonnegative predictions, level in anchor band, and source hashes must pass for A and B.
3. Critical `Var(ETX-AVG3 - SEQ-AVG3)_test / Var(...)_OOF` must be inside frozen `0.6..1.2`; report all component-to-ensemble and pairwise variance ratios diagnostically without using them to tune weights.
4. B source must be byte-identical to exp_051 SHA256 `c3cfb4d...c2932`; exp_051 production support and variance ratio `1.1734` must still be present and PASS.
5. Copy exactly A and B into `submissions/FINAL_20260825_A1/`, produce a manifest with hashes/provenance/statistics, and do not upload.

Any mandatory audit failure is a true integration blocker and neither file may be presented as ready. Passing means `ACCEPT` the refreshed two-candidate package while retaining exp_037 as strongest current.
