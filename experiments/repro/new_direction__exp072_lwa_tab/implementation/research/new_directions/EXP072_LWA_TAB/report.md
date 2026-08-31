# EXP072 LWA TAB — final report

## 1. Verdict

**REJECT** — pilot gate failure: `G2`.

The legal late-window conditional-amount channel does not scale in the preregistered full-capacity tabular form. Full four-fold validation, seed robustness, bootstrap analysis, OOF vector production, and all TEST inference were stopped exactly at the pilot gate.

Recommendation: **CLOSE_FAMILY**.

## 2. Pilot metrics and gate arithmetic

- EXP-037: `1.741278566416`
- FRESH: `1.741983057925`; delta vs EXP-037 `+0.000704492`
- VOL: `1.742260827848`
- FRESH_NOOV: `1.742108480674`
- REAL - VOL: `-0.000277770`
- NOOV - VOL: `-0.000152347`; required `<= -0.000138885`
- Unexplained variance: `0.952719`
- `corr(d_LWA,d_FRESH_EXP069)`: `+0.133565`

- `G1`: **PASS** — `{"pass": true, "value": -0.00027776992298922387, "threshold": -0.0001, "margin": 0.00017776992298922388, "formula": "score(FRESH)-score(VOL)"}`
- `G2`: **FAIL** — `{"pass": false, "value": 0.0007044915083431746, "threshold": 0.0, "margin": -0.0007044915083431746, "formula": "score(FRESH)-score(EXP037)"}`
- `G3`: **PASS** — `{"pass": true, "left": -0.00015234717354850602, "right": -0.00013888496149461194, "margin": 1.3462212053894085e-05, "formula": "score(FRESH_NOOV)-score(VOL) <= 0.5*(score(FRESH)-score(VOL))"}`
- `G4_unexplained`: **PASS** — `{"pass": true, "value": 0.9527185747059694, "threshold": 0.5, "margin": 0.45271857470596943}`
- `G4_corr_exp069`: **PASS** — `{"pass": true, "value": 0.1335650978246083, "absolute_value": 0.1335650978246083, "threshold": 0.85, "margin": 0.7164349021753917}`

The fixed-alpha pilot is the only predictive estimate authorized after rejection. Canonical four-fold wCV, nested delta, span-orthogonal nested delta, bootstrap interval, user-half full-validation results, and seed spread are **not estimated**.

## 3. Controls and diversity

The VOL arm used exactly the same number of additional rows as EXTRA, drawn with replacement from the earliest one-third of CLEAN positive cutoff slots with RNG seed 42. NOOV retained 9/13 EXTRA cutoffs. The pilot target-free least-squares projection used the 13 nonredundant aligned OOF directions and left `0.953` of centered variance unexplained.

## 4. Leakage and provenance

All CLEAN cutoffs satisfy `T+30<=V`; EXTRA contributes only positive-target rows; each donor splitmix side predicts only the opposite recipient side; the exact EXP069 OOF `p_dist` vectors are frozen and shared byte-identically by all arms; bounds and centers come only from the three donor panels; no public-LB value, score, geometry weight, or reconstructed champion OOF entered a label, weight, bound, level, projection coefficient, or selection.

The review packet's 13 claimed EXTRA `panel_*_b3.parquet` caches were absent. Exact canonical three-block eligibility was reconstructed in memory from raw events and equality-checked against two existing b3 caches. No external cache was written.

## 5. Runtime and artifacts

Pilot wall time: `1067.1s` (budget `3000s`; within budget `True`). Peak observed RSS: `4,843,343,872` bytes. Temporary data remained in memory and was released; no persistent model or feature cache was created.

Input hashes and row counts are in `artifact_manifest.csv`. Output hashes are in `checksums.sha256`.

No `lwa_tab_OOF.parquet`, `lwa_tab_TEST.parquet`, `lwa_tab_TEST.csv`, or any other TEST vector exists.

## 6. Limitations

The run stopped on the single latest-fold pilot, so it cannot estimate four-fold stability, bootstrap uncertainty, user-half aggregate effects, seed spread, TEST span distance, or OOF/TEST magnitude parity. The frozen OOF `p_dist` is the exact EXP069 fold source; the saved EXP069 TEST `p_dist` was never read for inference. The canonical folds still end four months and one holiday season before TEST, but the early stop prevents any TEST claim.

## 7. Recommendation and next measurement

**CLOSE_FAMILY.** The single most informative next measurement would be a genuinely different legal late-window data channel with a preregistered matched-row control; another model, seed, round count, or feature manipulation on this same conditional-positive tabular channel is not authorized by this experiment.
