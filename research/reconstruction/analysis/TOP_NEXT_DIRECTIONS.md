# Top next directions

Shortlist deliberately contains three directions. Expected gains refer to leaderboard RMSLE unless explicitly labeled OOF. Probabilities are broad decision ranges, not calibrated frequencies.

## 1. Fixed `SEQ65 + BTYD05` compound — exploitation

### 1. Thesis

Materialize one locked no-training submission:

```text
z_seq65 = .10 CAP + .10 UNC + .15 DIST + .325 ETX-AVG3 + .325 SEQ-AVG3
z_final_raw = .95 * z_seq65 + .05 * z_BTYD
```

Then apply the project level policy once: global shift to mean raw `z=2.3293`, floor at zero, `expm1`.

### 2. Evidence

- `team_a_current:EXP-059`: fixed sequence-slot reweight `−0.000237898`, 4/4.
- `team_a_current:EXP-051`: fixed BTYD05 `−0.000320983`, 4/4; production variance ratio `1.1734` and exact test support.
- New aligned OOF: compound `1.746947164` vs `EXP-037 1.747509863`, **`−0.000562699`**, folds `−0.000988/−0.000795/−0.000510/−0.000478`.
- Correction Pearson `−0.01263`; interaction `−0.00000382`; paired user-cluster SE `0.0000586`.
- New combined test/OOF correction variance ratio `1.2167`, inside historical `[0.6,1.4]` gate.
- Nested 2D audit independently gives `−0.0005656`, nearly identical to fixed recipe.

### 3. Why it may still improve

Neither gain is in `EXP-037`; `EXP-059` changes representation balance, `EXP-051` injects a point-process/count×value residual. Они никогда не встречались вместе. Near-zero correction correlation и near-zero interaction показывают отсутствие absorption.

### 4. Proposed experiment

Independent hash-bound reconstruction from primitive arrays, not from already shifted CSV:

1. verify all nine `EXP-059` component hashes and BTYD `test_raw.npz` hash;
2. align all 250,000 user IDs to sample order;
3. rebuild OOF and require exact numbers above;
4. build exactly one CSV with `.65/.05` frozen;
5. record SHA256; no neighboring weights and no post-LB adjustment.

### 5. Expected result

- Local OOF is already measured: `−0.000563`.
- Likely LB gain: **`−0.0002…−0.0005`**.
- Upside: around `−0.0007`.
- Probability of positive LB gain: **70–85%**.
- Probability of gain `>0.001`: **5–10%**.

Historical OOF→LB transfer for strong-line improvements was roughly 0.57–0.71, but range widens for regime shift; this is why expected LB magnitude is below OOF delta.

### 6. Cost

- Training: none.
- CPU: 2–5 minutes.
- RAM: <1 GB.
- Storage: about 5 MB CSV plus small manifest.
- Complexity: low.
- Private-LB robustness: medium-high; 4/4, paired SE and test ratio pass, but gain concentrates in zeros/high GMV.
- Dependencies: all artifacts already exist.

### 7. Kill criterion

Kill immediately if independent rebuild differs by `>5e-7` in log-space, any OOF fold loses, total delta is weaker than `−0.0003`, combined test/OOF correction variance leaves `[0.6,1.4]`, or user alignment/hash verification fails.

## 2. Exact FRESH production add-on to the frozen compound — exploitation with training

### 1. Thesis

Reproduce only the already measured `EXP-040` FRESH−CLEAN correction under an exact deterministic production recipe and add it once to the frozen direction 1 compound. No new architecture, alpha, segments or weight grid.

### 2. Evidence

- `team_a_current:EXP-040`: FRESH correction `−0.000224956`, 4/4, paired SE `0.0000375`.
- `team_a_current:EXP-049`: BTYD05+FRESH `−0.000547` on corrected 3-fold protocol, 3/3; cluster bootstrap P(delta<0)=1.0; signal shuffle passed.
- New full 4-fold calculation: BTYD05+FRESH `−0.000466940`, 4/4.
- New frozen triple `SEQ65+BTYD05+FRESH`: wCV `1.746788684`, **`−0.000721179`**, 4/4; incremental over direction 1 `−0.000158480`.
- FRESH vs SEQ65 correction corr `−0.0638`; FRESH vs BTYD corr `+0.1762`.

### 3. Contradictory evidence

- Pair B+FRESH is antagonistic: observed `−0.000467` vs arithmetic `−0.000546`.
- Exact TEST encoder and conditional heads do not exist.
- Historical fold encoders used a `workers=3` race policy; exact trajectory is not recoverable from saved state.
- `EXP-051` estimated five new encoder runs plus 30 conditional-head fits; therefore current OOF artifact alone cannot authorize a submission.
- Incremental gain over direction 1 is only `0.000158`, smaller than total FRESH standalone effect.

### 4. Why it may still improve

FRESH captures conditional regime contrast not contained in BTYD or simple sequence-slot reweight. The triple improves every fold and its total gain is materially larger than direction 1. Missing value is production support, not local residual evidence.

### 5. Proposed experiment

1. Freeze original TCN architecture, two-sided hash split, CLEAN/FRESH heads, GLOBAL mapping, alpha=1, donor-safe winsorization and depth clip 289.
2. Make encoder deterministic before training; pre-register expected parity tolerance rather than pretending to replay the racy trajectory.
3. Rebuild 4 OOF folds + one production encoder and 30 heads.
4. First gate: new OOF correction must reproduce saved `EXP-040` total within `±0.00005`, sign 4/4.
5. Second gate: add only to frozen direction 1; no re-optimization.
6. Production gate: correction variance ratio `[0.6,1.4]`, support/clipping and user alignment pass.

### 6. Expected result

- Incremental likely LB gain over direction 1: **`−0.00005…−0.00025`**.
- Total likely gain vs `EXP-037`: **`−0.0003…−0.0007`**.
- Upside total: around `−0.0009`.
- Probability of positive incremental gain: **55–70%**.
- Probability total gain `>0.001`: **below 5%**.

### 7. Cost

- Compute: roughly 5 encoder trajectories + 30 small head fits; plan for 4–8 GPU-hours and 1–2 CPU-hours, subject to measured hardware throughput.
- Storage: likely 5–15 GB checkpoints/predictions.
- Complexity: medium-high because determinism and cross-fit must be preserved.
- Risk: medium; implementation/trajectory shift is the main risk.
- Private-LB robustness: medium if production variance passes; residual regime overlaps BTYD.
- Dependencies: direction 1 frozen; original FRESH code/tests; full data and GPU.

### 8. Kill criterion

Kill if deterministic rebuild fails the `±0.00005` OOF parity band, incremental triple gain is weaker than `−0.00010`, fewer than 3/4 folds improve, correction variance ratio leaves `[0.6,1.4]`, or production clipping/support exceeds preregistered OOF bounds. Do not rescue with alpha/segment/weight tuning.

## 3. Canonical four-fold rebuild of teammate occurrence/meta line — exploration

### 1. Thesis

Do not submit or tune `latest.csv` as a black box. Recreate canonical row-level OOF for its two missing components (`occ_meta_B`, `occ_raw_X3`) on the exact `EXP-037` folds and evaluate one locked transform against the real champion.

### 2. Evidence

- Teammate fixedstack candidates report `−0.00146…−0.00155` vs their base on three recent folds.
- Final6h/extra90 report `−0.00165…−0.00182`, recent 3/3, latest-fold up to `−0.00207`.
- Test submissions and `latest.csv` recipe are reconstructible; `latest=.12 friend+.16 occ_meta_B+.72 occ_raw_X3` exactly.
- Externally reported public `1.649217562` is lower than champion `1.6496571`, so the branch cannot be dismissed outright.
- Prediction differences to friend are small but nonzero: corr `0.99963–0.99972`, std delta `0.038–0.045`.

### 3. Contradictory evidence

- Teammate validation base is `1.749803703`, already worse than `EXP-037 1.747509863`.
- Candidate fold 1 delta is exactly 0; the apparent gain is measured on only three recent folds.
- Final candidate absolute wCV remains around `1.7480`, not better than the canonical champion.
- Hundreds of Ridge/meta/occurrence variants were searched; repeated-selection optimism is substantial.
- Canonical OOF for final components is absent; public score lacks SHA binding.

### 4. Why it may still improve

Occurrence models and raw/meta table stacks target regimes where compound gains are largest (future zero/nonbuyer and recency). The line may contain residual complementarity even if its standalone absolute CV is worse. That question has never been tested on common OOF with fold-safe weights.

### 5. Proposed experiment

1. Parent: exact `EXP-037` OOF and fold definitions.
2. Rebuild only the two named final teammate components with exact train coverage and row keys; no new family search.
3. Freeze at most three recipes before results: raw X3, meta B, recorded `.12/.16/.72 latest` transform.
4. Evaluate absolute four-fold scores and a nested LOFO blend with `EXP-037`/direction 1.
5. Require nonzero improvement on fold 1, ≥3/4 overall, and test/OOF variance `[0.6,1.4]`.
6. Public LB is not used for weight selection.

### 6. Expected result

- Likely honest gain vs `EXP-037`: **`0…−0.0006`**.
- Upside: **`−0.0010…−0.0013`** if recent-fold effect survives common OOF.
- Probability of positive gain: **40–60%**.
- Probability of gain `>0.001`: **10–20%**.

### 7. Cost

- Compute: estimated 6–10 hours using the historical fixedstack/final6h runtimes as lower bounds; cached pieces may reduce this.
- Storage: likely 10–30 GB for checkpoints and canonical OOF/test banks.
- Complexity: high; provenance, fold coverage and raw meta features must be reconstructed.
- Risk: high repeated-selection and implementation drift.
- Private-LB robustness: unresolved until fold 1/common OOF passes.
- Dependencies: teammate scripts/data/checkpoints and exact `EXP-037` row keys.

### 8. Kill criterion

Kill if common OOF cannot be reconstructed exactly, fold 1 does not improve, fixed/nested blend delta is weaker than `−0.0003`, absolute candidate remains worse without residual complementarity, or test/OOF variance leaves `[0.6,1.4]`. Do not open another meta-search before passing this gate.

## Priority and portfolio

| rank | direction | strategy | expected value |
|---:|---|---|---|
| 1 | fixed SEQ65+BTYD05 | exploitation | highest; ready, no training |
| 2 | exact FRESH add-on | exploitation | smaller incremental EV, strong OOF evidence |
| 3 | canonical teammate occurrence/meta | exploration | lower probability, largest plausible upside |

Recommended budget split: roughly 75% exploitation / 25% exploration until direction 1 has a frozen artifact and direction 2 passes production support.
