# RECENCY-RIDGE-ON-PREDICTIONS — report

## Outcome

**Verdict: `BLOCKED_HISTORICAL_REPLAY`.** The historical reference CSV is present,
but the exact row-level OOF/test prediction bank and `meta_raw` matrices used to fit
the Ridge are absent. The required formula replay error therefore cannot be computed;
the `5e-07` gate is not claimed from a circular read of the final CSV.

No new Ridge, lambda/scale search, canonical LOFO, controls, test candidate, submission,
or leaderboard upload was run.

## Phase A — what was recovered exactly

- Reference: `пайплайн сокомандника\review_bundles\fixedstack_combo10h_REVIEW_BUNDLE_20260823_054654_extracted\submissions\submission_combo10h_candidate_1_ridge_drop_recent_hurdle_stable18_s075.csv`
- SHA-256: `95965c33bfe32378227e39ec1e0a792fed19cc29dc73d84b8b83bc0cda447959`; rows: 250,000; exact common user order.
- Historical table member order: `cap, unc, dist, hurdle, multiscale_direct, recent_direct, recent_dist, recent_hurdle_fast12`.
- Target: `log1p(y) - table_core`; expanding prior-fold walk-forward, with the first
  fold equal to `table_core` (not four-fold outer LOFO).
- Ridge: `StandardScaler(copy=False)` then `Ridge(alpha=150, solver='lsqr',
  tol=1e-4, fit_intercept=True)`; fold weights `1:2:4:8` go only to Ridge;
  correction clipped to `[-2,2]`, fixed scale `0.75`.
- Finalization: `candidate_table=clip(table_core+0.75*d,0,20)`, then
  `friend+0.55*(candidate_table-table_core)`, fixed mean level `2.3293`, one final clip,
  `predict=expm1(z)`, pandas CSV serialization.
- `recent_dist`: norm-long dist, temporal tau 120, 330 rounds.
- `recent_hurdle_fast12`: two-part model on the last 12 eligible cutoffs, tau 70,
  430 rounds. Despite the winner name, `recent_hurdle_stable18` was dropped.

The external description “prediction-level Ridge” is not exact: the winning recipe
has `include_meta=True` and prepends up to 72 raw activity/recency columns. The separate
`ridge_predonly_finalizable` is a different candidate. Exact meta column names and fitted
coefficients are unavailable because the cap OOF checkpoints were omitted.

The source tree contains 629 NPZ files, but zero of the 32
expected historical OOF checkpoints and zero of the 6 missing helper
test checkpoints. The package does retain ready CAP/UNC/DIST TEST arrays, which are
insufficient to refit Ridge without donor OOF, targets, `p/mu`, and `meta_raw`.

The number `1.6492897556391737` is recorded as `known_ridge_submission_public` in the
later Final6h manifest, but no SHA-to-score row binds it to the exact reference CSV.
It is therefore recorded as family-level/unbound evidence, not as verified LB for SHA
`95965c33bfe32378227e39ec1e0a792fed19cc29dc73d84b8b83bc0cda447959`.

## Phase B — redundancy audit

The exact historical winner is not a direct component of `friend`, `occ_meta_B`,
`occ_raw_X3`, or `latest`. `friend` is byte-identical to `STRONGEST_CURRENT` and predates
the Ridge. Both occurrence components instead descend from a later, distinct anchor:
`blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85`. That anchor uses all nine finalizable members (including stable18),
weights folds as `(1:2:4:8)^1.7`, then adaptively blends Ridge with greedy35.

Nevertheless the historical function is nearly absorbed on TEST: against `latest`,
`corr=0.999968243561` and
`Var(z_historical-z_latest)=0.000159636752562`.
Against the later Ridge/greedy anchor, `corr=0.999989817609`
and variance difference is `5.53331324853e-05`.
These are production-TEST geometry diagnostics only. Target residual correlations and
`corr(z_ridge-z_latest, log1p(y)-z_latest)` are unavailable without canonical aligned OOF.

The authoritative TEST `latest` recipe replays independently from friend/B/X3 to
`8.88e-16` max log error, but canonical row-level OOF for B/X3 is still
missing, consistent with `exp_066/067`.

## Phases C and test regime

Not run. Phase A failed first, and the independent prerequisite—canonical four-fold OOF
`latest`—is also absent. `CAP_LINEAGE=UNKNOWN`; private safety and production parity
remain unresolved. All requested downstream CSV/JSON/NPZ artifacts are explicit blocked
markers rather than synthetic metrics.
