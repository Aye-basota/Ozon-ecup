# EXP-064 — EVENT-ORDER Hypothesis Card (frozen before results)

- **Date:** 2026-08-25
- **Author:** A1
- **Development reference:** exact `STRONGEST-CURRENT / exp_037`, wCV `1.7475098625`

## Hypothesis

The direction of transitions between a user's observed daily funnel states—Search, Cart, and Buy—contains intent-resolution information not present in 7/14/30/60/90-day marginal counts. Explicit 90-day transition motifs and gaps may complement the champion even though ETX/TCN learn a latent sequence representation.

## Why genuinely new information

Two users can have identical activity dates and identical multisets of daily Search/Cart/Buy states but opposite order (for example Search→Cart→Buy versus Buy→Cart→Search). Canonical tabular columns, OPEN-FUNNEL marginals, and platform normalization do not identify this difference. The placebo below preserves those observables exactly and destroys only order.

## Closest prior + critical difference

- `exp_054` used hand-fixed activity episodes based on thresholded total intensity; EXP-064 uses channel-state transition direction and a within-user exact state-multiset control.
- `exp_061` summarized Search/Cart after the last purchase but discarded the full intermediate path; EXP-064 represents adjacent active-day motifs throughout 90 days.
- ETX/TCN can encode order, but the falsification asks whether an explicit low-variance transition statistic explains residuals not reliably extracted by those models.

## Exact representation and placebo

Using only rows with `T-90 < event_date <= T`, encode each observed day as three bits `(searches>0, to_cart>0, to_ord>0 or gmv>0)`. Add only new `eo_*` columns through opt-in `build_features(cutoff_date, event_order_source=...)`: transition/change/repeat counts, up/down stage counts, Search→Cart/Buy, Cart→Buy, no-buy↔buy, buy→buy, recent-30 up/down counts, up/down calendar-gap means, and last-transition code.

`SHUFFLED` deterministically permutes the complete three-bit state vector among the same user's observed dates before transition aggregation. It preserves user, cutoff, active dates/gaps, number of rows, and the exact per-user daily-state multiset. `CONTROL_ONLY` uses the same fixed champion controls without `eo_*` columns.

## Cheapest falsification

Build REAL and SHUFFLED opt-in features for all four canonical folds. Join them to exact exp_037 OOF, then use the already validated four-way cross-user residual probe and two-sided donor-selected scale grid `{0,.25,.5,.75,1}`. No GPU, full production model, or test inference is allowed before this gate.

## Success gate

- Cutoff, unique-key, row-order, finiteness, state-multiset-by-construction unit tests, exact per-user transition-count parity, and shuffle-movement audits pass (`>20%` of user feature rows must change).
- REAL improves at least `3/4` folds including `2025-10-16`.
- Nested two-sided `Delta wCV <= -0.0005`; REAL beats SHUFFLED by at least `0.0003` wCV.
- Both user halves choose nonzero scales; gain is not confined to a tiny transition-support segment.

Passing means `CONTINUE` to one canonical low-capacity tabular model pilot and then test-support audit. It is not production acceptance.

## Kill gate

Any audit failure, latest-fold non-improvement, `Delta wCV > -0.0005`, REAL−SHUFFLED advantage smaller than `0.0003`, or zero held scale => `REJECT`. Do not rescue this exact family with state alphabets, window/transition-lag sweeps, higher-capacity learners, segment gates, or neural retraining.
