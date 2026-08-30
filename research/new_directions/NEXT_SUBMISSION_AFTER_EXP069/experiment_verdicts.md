# Independent verdicts — EXP069, EXP070, EXP071

Every number below was recomputed from saved predictions with the clean repo's canonical
evaluator (`src/metrics/rmsle.py`: per-fold optimal global log offset, non-negative
predictions, wCV weights 1:2:4:8), not with any experiment's private copy. The OOF
baseline is `pred_exp037` from `06_ALIGNED_OOF.parquet` throughout. The public incumbent
was never used as an OOF baseline, residual target or selection signal.

---

## EXP071_ETX_FRESH_CONTRAST — **CONFIRMED_REJECT**

Reported verdict `REJECT_PILOT / DO_NOT_ADD` is independently confirmed. Every pilot
metric reproduces to **0.000e+00**:

| quantity | recomputed | reported | diff |
|---|---|---|---|
| EXP-037, fold 2025-10-16 | 1.741278566416 | 1.741278566416 | 0 |
| ETX-FRESH (unit scale) | 1.741388597316 | 1.741388597316 | 0 |
| ETX-VOL (matched-volume placebo) | 1.741195469323 | 1.741195469323 | 0 |
| SEQ-FRESH | 1.741025144325 | 1.741025144325 | 0 |
| SEQ-FRESH + ETX orthogonal | 1.741237731795 | 1.741237731795 | 0 |
| REAL delta vs EXP-037 | **+0.000110031** | +0.000110031 | 0 |
| REAL − VOL | **+0.000193128** | +0.000193128 | 0 |
| orthogonal incremental vs SEQ-FRESH | **+0.000212587** | +0.000212587 | 0 |
| user half A / B (REAL − VOL) | +0.000166589 / +0.000222821 | same | 0 |

Verification performed: exact row alignment to the canonical `2025-10-16` fold
(197,379 rows, keys equal); `user_side` equals `splitmix64(user_id)&1`; targets from the
canonical bank; correction preprocessing replayed from the registered clip bounds and
centres; scale = 1 as registered (the alpha grid is diagnostic and reproduces at every
grid point); `corr(ETX, SEQ-FRESH) = 0.2402781055` and `gamma = 0.1400473076` reproduce
exactly. Encoder-parity evidence: hook-vs-original forward error `0.0`, archived
2025-10-16 OOF array replays exactly; the recorded bf16/SDPA TEST drift
(RMS `0.002664`, max `0.0625`) is a between-runtime artefact and is not implicated in
this fold's numbers because no TEST inference was run.

**All three failure directions agree:** REAL is worse than EXP-037, REAL is worse than
its own matched-volume placebo, and the ETX-orthogonal component makes SEQ-FRESH worse.
Both user halves point the same (wrong) way. Nothing was rescued: no sign flip, the
placebo was not promoted, no favourable half or alpha was selected, and no four-fold run
was started after the gate failed.

**Scientific scope of the rejection — precise statement.** EXP071 rejects *FRESH
conditional-amount supervision on this frozen ETX-01 seed-42 representation*, and only
that. It does **not** reject ETX representations in general (ETX-AVG3 carries weight
0.225 inside EXP-037 and is not challenged), and it does **not** reject FRESH supervision
in general — the same mechanism on the frozen SEQ-D3A TCN embedding is exactly what
EXP069 validates 4/4 folds. The failed object is the *pairing*: the ETX final query-token
embedding does not encode the amount-timing information that the fresh EXTRA cutoffs
carry. Diagnostic support: the ETX correction correlates only `0.2403` with the working
SEQ-FRESH correction and `0.0040` with the base residual — it is a different, and empty,
direction rather than a weaker copy of a good one.

---

## EXP070_COUNT_VALUE_MOE — **COMPUTE_INCOMPLETE** (with a control-fragile weak signal)

Not "confirmed reject" and not a pass. Everything that exists reproduces exactly; what
would decide it does not exist.

Reproduction (from `count_value_moe_raw_OOF.parquet` and the `_fold_*.npz` caches):

| candidate | 2025-09-04 | 2025-09-18 | 2025-10-16 | max diff vs report |
|---|---|---|---|---|
| EXP-037 | 1.766883356768 | 1.760509576735 | 1.741278566416 | 0 |
| COUNT_REAL | 1.769165238855 | 1.763482272285 | 1.743399240024 | 0 |
| COUNT_SHUFFLED | 1.768684854404 | 1.763273346741 | 1.743778955725 | 0 |
| REPLACE_REAL_BETA1 | 1.766780903844 | 1.760542273041 | 1.741164059510 | 0 |
| REPLACE_SHUFFLED_BETA1 | 1.766600799507 | 1.760455977283 | 1.741280484514 | 0 |
| ADD10_REAL | 1.766913350500 | 1.760596041657 | 1.741259190342 | 2.2e-16 |

Formula verified: the DIST replacement is exactly `z_exp037 + 0.25·(z_count − z_dist)`
(z_base is byte-equal to `pred_exp037`, `z_predict = z_base + correction`), and the
add-one endpoint is exactly `0.90·z_exp037 + 0.10·z_count`. Label construction, five
frozen bins, target window `(T, T+30]`, probability row sums (`max |Σp−1| = 5.157e-08`),
canonical keys and targets all check out; C4 frequency `3.135%` in the oldest training
panel is above the `0.5%` fallback threshold, so five bins were correctly retained.
Runtime stop at `6984 s` with `2025-10-02` untrained is confirmed by the artifacts.

Answers to the registered questions:

1. **Was the pilot pass materially above noise, or marginal?** *Marginal.* The gate was
   ≤ −0.00010 on both arms; the observed values were −0.000114507 (vs EXP-037) and
   −0.000116425 (vs shuffled) — 1.15× the threshold. A user-cluster bootstrap of the
   three-fold 1:2:8 replacement delta gives 95% `[-0.000156, -0.0000073]`, `P(Δ<0)=0.983`:
   the interval nearly touches zero.
2. **Did the REAL−SHUFFLED advantage shrink across folds?** *It inverts.* Per fold:
   `2025-09-04 +0.000180`, `2025-09-18 +0.000086`, `2025-10-16 −0.000116`. The placebo
   **beats** the real model on 2 of the 3 completed folds. The only negative value is the
   latest fold, which carries weight 8 of 11 in the diagnostic aggregate — the entire
   `−0.0000526` 1:2:8 REAL−SHUFFLED figure is that one fold.
3. **Is there evidence of fold decay?** There is evidence of fold *dependence*, in the
   favourable direction over time (real-vs-EXP-037 deltas: `−0.000102 / +0.000033 /
   −0.000115`), but with an inconsistent sign and a control that only cooperates on the
   most recent fold. That is a legitimate recency story and equally a legitimate
   one-fold-fluke story; three folds cannot separate them.
4. **Is it worth finishing?** *Yes, cheaply, and only as a decisive test.* The missing
   `2025-10-02` fold is the single observation that would either give a canonical
   four-fold wCV plus an honest LOFO, or kill the mechanism. It is currently the cheapest
   decisive experiment on the board.
5. **Can it be completed materially faster?** *Yes, roughly 4× cheaper than a rerun.*
   The runner already has `save_fold_cache`/`load_fold_cache`, the three completed folds
   are cached in `_fold_*.npz`, `_label_cache/label_20251002_b3.parquet` already exists,
   and the expert LightGBM Dataset parent is already shared between the real and shuffled
   arms. Completing only the missing fold means one fold's 12 model fits (~40–50 min at
   the observed ~2,330 s/fold) instead of ~116 min for all three. Further savings are
   available but unnecessary: loading only the 227 frozen feature columns, and reusing the
   binned parent across the five experts (already done).

**Not usable in this submission**, as its own report concludes: no canonical four-fold
wCV, no honest LOFO, no TEST vector, no production audit. Nothing was inferred from the
incomplete fold set.

---

## EXP069_BTYD05_FRESH1_PRODUCTION — **PASS_TYPE_A independently confirmed**

### Tests (Phase 4.1)

| suite | command | result | runtime |
|---|---|---|---|
| EXP069 lineage (the reported 40) | `python -m pytest src/test_fresh_contrast.py src/test_seq_cond.py -q` (in `OZON-E-CUP`) | **40 passed**, exit 0 | 23.3 s |
| component suites (superset) | `python -m pytest src/test_seq.py src/test_dist.py src/test_zero2d_shrink.py src/test_validation.py src/test_btyd_stable_fit.py -q` | **116 passed**, exit 0 | 16.8 s |
| production/schema validator | `python validate_outputs.py` | `PASS`; csv-vs-parquet max abs error `0.0`; production formula max abs error `0.0`; 30/30 checksums; 0 manifest failures | 3 s |

The "40 tests" claim is exact: `test_fresh_contrast.py` collects 9 and `test_seq_cond.py`
collects 31. No formula, alignment, leakage, schema or production test failed.

### Canonical OOF parity (Phase 4.2)

770,616 rows, 770,616 unique `(fold, user_id)` keys, fold sizes
`188,518 / 191,025 / 193,694 / 197,379`; row order identical across the aligned bank and
both EXP069 OOF files; target max abs difference `0.0`; EXP-037 rebuilds from
`0.10 CAP + 0.20 UNC + 0.25 DIST + 0.225 SEQ-AVG3 + 0.225 ETX-AVG3` with max log error
`9.92e-08`; BTYD05 rebuilds with `1.14e-07`; the saved correction equals the historical
`fresh_processed_nested` vector exactly (`0.0`); `splitmix64(user_id)&1` matches the saved
`user_side` on all rows; all predictions finite and non-negative.

| quantity | recomputed | registered | difference |
|---|---|---|---|
| EXP-037 wCV | 1.747509862493216 | 1.7475098625201952 | −2.70e-11 |
| FRESH delta wCV | −0.000224956 | −0.000224956 | ~1e-15 |
| BTYD05 delta wCV | −0.000320983 | −0.000320983 | ~1e-15 |
| **BTYD05_FRESH1 delta wCV** | **−0.000466940** | −0.000466940 | ~1e-15 |
| fold deltas | −0.000835154 / −0.000703787 / −0.000395975 / −0.000397183 | same | ~1e-15 |
| latest-fold delta | −0.000397183 | −0.000397183 | — |
| folds improved | 4 / 4 | 4 / 4 | — |
| VOL placebo delta | **+0.000007093** | +0.000007093 | — |
| REAL FRESH − VOL | **−0.000232049** | −0.000232049 | — |
| user half A / B | −0.000563385 / −0.000370573 | same | — |
| bootstrap 95% (500 user clusters, seed 42) | **[−0.000585341, −0.000345622]**, P(Δ<0)=1.00 | [−0.0005853406, −0.0003456223] | — |

All headline claims reproduce. The registered `1.7475098625201952` differs from the
recomputed `1.747509862493216` by `2.7e-11`, which is float32 storage in the aligned bank,
far inside the experiment's own `2e-5` parity gate.

### TEST artifacts (Phase 4.3)

250,000 rows in every TEST artifact; 250,000 unique `user_id`; order byte-equal to
`sample_submit.csv`; finite and non-negative; CSV-vs-parquet max abs error `0.0`;
independent SHA256 of all four declared artifacts matches the report and the ledger.

### Correction vs baseline separation (Phase 4.4)

`d_combined = z_exp069 − z_base_used` decomposes as
`d_btyd + d_fresh` with max abs error **1.19e-15**, where
`d_btyd = 0.05·(z_btyd − z_base_used)` and `d_fresh` is the explicitly saved
`fresh_conditional_TEST.correction` column (recovered, not inferred).

Baseline reconstruction drift, `z_base_used − exact EXP-037 formula on the aligned TEST
component arrays`:

| statistic | value |
|---|---|
| RMS | 0.135343 |
| mean | −0.135341 |
| sd after removing the mean | **0.000830** |
| max abs | 0.135362 |
| quantiles 0.001 → 0.999 | −0.135362 → −0.133042 |
| rows floored at zero | 273 |

The drift is a **pure level shift** — the documented rescale of `pred_exp037_rebuilt` to
production mean log level 2.3293 — plus a 273-row flooring nonlinearity. It is not in the
deployed step at all: the deployment uses only `d_fresh`, a standalone additive vector
that never touches `z_base_used`. Level shifts are inside the geometry span anyway
(constant-direction residual `7.89e-09`), so the projection would have removed one.

### Span expansion (Phase 4.5) — reproduced exactly

Using the same `Z.npz` (hash verified) and the same `build_basis(tol=1e-12)`:

| quantity | recomputed | reported |
|---|---|---|
| unique sources / basis rank | 65 / 57 | 65 / 57 |
| RMS distance from affine span | **0.010679245743308086** | 0.010679245743308086 |
| orthogonal norm fraction | **0.12719419305538987** | 0.12719419305538987 |
| nearest source | candidate_B_BTYD05_HEDGE.csv, 0.015509304981 | same |
| numerical rank | **57 → 58** | 57 → 58 |

### Verdict

`PASS_TYPE_A` is **independently confirmed**. Both registered gates hold on recomputed
values: delta ≤ −0.00035 (−0.000467), ≥3/4 folds improved (4/4), latest fold negative,
REAL−VOL ≤ −0.00010 (−0.000232), both user halves negative, production regime PASS,
schema PASS.

**Carried limitations** (both affect the uncertainty budget, neither invalidates the pass):

1. The TEST extensive probability `p_dist` is a same-recipe CLEAN-only S1-DIST
   *reconstruction*, and the raw contrast is literally `p_dist · (μ_FRESH − μ_CLEAN)`.
   `p_dist` therefore multiplies the correction per user, so the reconstruction limitation
   touches the **new orthogonal component**, not only the parallel part. `p_dist` is
   well-behaved (mean 0.580, range [0.0205, 0.9998], no zeros), but its per-user accuracy
   is bounded by the rebuild's mean `|Δz| = 0.0445`.
2. The TEST correction is materially *smaller* than the OOF one: raw sd ratio `0.370`,
   processed sd ratio `0.470`, orthogonal-component RMS ratio `0.405`. Part of this is
   benign (TEST averages two donor sides × three seeds; the historical OOF is a
   single-seed cross-fit, so TEST is a lower-variance estimate of the same contrast), but
   it is not fully explained by averaging and it means α = 1.0 on TEST already applies
   only ≈ 0.40 of the OOF-validated correction magnitude.
