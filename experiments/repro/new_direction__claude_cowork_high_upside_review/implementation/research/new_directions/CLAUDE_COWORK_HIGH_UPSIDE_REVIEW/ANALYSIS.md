# High-upside research review — independent audit and experiment selection

Reviewer: independent research review session, 2026-08-26.
Scope: `C:/Users/Admin/Desktop/e-cup-research-clean` plus its two registered external workspaces
(`C:/Users/Admin/Desktop/OZON-E-CUP`, `C:/Users/Admin/Desktop/submission_geometry_research`).
Every number below was recomputed from primary artifacts in this session unless explicitly
attributed to a registry card. No model was trained, no submission was produced, no public
score was used to select anything.

---

## 0. What was actually read

Primary artifacts (not summaries): `registry/{experiments,models,submissions,source_manifest}.csv`;
`gpt_pro_research_packet/{01..16}` including `06_ALIGNED_OOF.parquet` (770,616 rows, 18 cols) and
`07_ALIGNED_TEST.parquet` (250,000 rows, 30 cols); `02_EXPERIMENT_REGISTRY.csv` (82 rows);
`05_OOF_INVENTORY.csv` (82 rows); `submission_geometry/cache/Z.npz` (67 vectors → 65 unique) with
`Z_meta.json`, `core.py`, `directions.py`, `geomlib.py`; the full `NEXT_SUBMISSION_AFTER_EXP069`
artifact set (report, `historical_transfer.csv`, `oof_component_metrics.csv`, `oof_nested_alpha.csv`,
`exp069_component_decomposition.csv`, `score_estimate.json`, `test_regime.json`); EXP069/070/071
reports, configs, pilot metrics and `fresh_conditional_{OOF,TEST}.parquet`; `src/features/canonical.py`,
`src/data/loaders.py`, `config/competition.yaml`; the raw event panel `data/raw/train.parquet`
(30,631,006 rows); and the `data/processed` cache inventory (649 files).

Parity check of my own evaluator against the registered numbers: EXP-037 wCV reproduces to
`1.747509867` on the float32 aligned packet (registered `1.7475098625201952`, difference `4.6e-8`);
FRESH add-one nested `−0.000225` (registry `−0.000225`); BTYD add-one nested `−0.000269`
(packet `−0.000269`); E11 occurrence add-one `+5.2e-6`, 1/4 (EXP063 `+1.05e-5`, 1/4). The pipeline
used for every claim below therefore reproduces the project's own primary results.

**No EXP072 description exists anywhere in the three workspaces.** The highest registered id is
`EXP068` in the historical registry plus `EXP069/070/071` in `research/new_directions`.
**The next free id is `EXP072`.**

---

## 1. Why the 0.001 target requires a different *kind* of experiment, not a bigger one

### 1.1 The incumbent is an LB-fitted optimum inside a span that contains every OOF source

The incumbent `1.6466079084` is an affine log-space combination of 65 unique submitted vectors,
fitted to public-LB score equations inside a rank-57/58 difference span. I rebuilt that basis from
`Z.npz` (reference `last (1).csv`, mean-N metric, tol `1e-12` → rank 58; the constant vector lies
inside the span, residual RMS `5.4e-12`) and projected every model-family TEST vector onto it:

| TEST vector | RMS deviation from reference | span-orthogonal RMS | orthogonal fraction |
|---|---:|---:|---:|
| `pred_cap` | 0.29279 | 0.01952 | 6.7 % |
| `pred_hurdle_e11` | 0.18897 | 0.00995 | 5.3 % |
| `pred_ridge15` | 0.27109 | 0.00921 | 3.4 % |
| `pred_unc` | 0.21123 | 0.00545 | 2.6 % |
| `pred_seq_avg3` | 0.13239 | 0.00295 | 2.2 % |
| `pred_etx_avg3` | 0.17873 | 0.00262 | 1.5 % |
| `pred_btyd` | 0.53345 | 0.00536 | 1.0 % |
| `pred_dist` | 0.18609 | 0.00109 | 0.6 % |
| every submitted geometry source | — | `0.00000` | 0 % |
| **EXP069 FRESH correction** | 0.01220 | **0.01068** | **87.5 %** |

Everything the project has ever submitted is, by construction, inside the span. The geometry has
already priced that span on the real TEST target and has consumed ~94 % of its own estimated
remaining safe budget. **Any candidate expressible as a recombination of existing sources is
already deployed.** This is the single structural fact that governs the whole review.

### 1.2 The total in-span headroom is smaller than the target

Measured on the canonical aligned OOF bank (14 structurally different prediction families,
fold-demeaned, EXP-037 as baseline):

| Estimator over the *entire existing OOF bank* | ΔwCV vs EXP-037 | per-fold |
|---|---:|---|
| best single add-one (nested α) — `pred_btyd05` | −0.000342 | 4/4 |
| **fold-safe linear stack (fit 3 folds, evaluate held-out)** | **−0.000796** | `[−0.00141, −0.00123, −0.00078, −0.00062]` |
| regime-aware stack, 5 quantile regimes of `z_base` | −0.000927 | `[−0.00155, −0.00126, −0.00093, −0.00077]` |
| regime-aware stack, 10 regimes | −0.000870 | — |
| regime-aware stack, 5/10 dispersion regimes | −0.000888 / −0.000808 | — |
| **in-fold linear ORACLE (cheating, uses the fold's own target)** | **−0.001081** | `[−0.00180, −0.00142, −0.00098, −0.00098]` |

Three conclusions, all load-bearing:

1. Even an *unrealisable* in-fold oracle over every family this project has ever built reaches only
   `−0.00108` wCV. A single new candidate at `−0.0015…−0.002` would have to be roughly twice the
   combined oracle value of the entire project history.
2. Conditional/regime-aware stacking buys at most `−0.00013` beyond a plain linear stack. The
   "nonlinear conditional stack" family is quantitatively dead, before any training.
3. Every one of these gains decays monotonically toward the latest fold. The measurable headroom
   is shrinking as the folds approach TEST, and TEST is four further months out.

### 1.3 The residual is almost entirely irreducible by anything in the bank

Fold-demeaned residual variance against EXP-037 is `3.032…3.122`. The fraction explained by the
whole 14-source difference span is `0.20 % / 0.16 % / 0.11 % / 0.11 %` (folds 1→4). A `−0.001` wCV
improvement corresponds to explaining `0.115 %` of residual variance; a `−0.001` public improvement
corresponds to explaining `0.122 %` of TEST residual variance. Meanwhile a perfect zero/non-zero
classifier would be worth `−0.591` wCV and per-row model selection over 15 sources `−0.261` wCV —
both are pure oracles that use the realised target and prove nothing except that the error mass is
overwhelmingly irreducible uncertainty about *whether* a user buys.

### 1.4 The raw data admits no new input channel

`data/raw/train.parquet` is a daily user-level panel: `event_date, user_id, search, cat,
has_search_to_{cart,ord}, has_cat_to_{cart,ord}, search_to_{cart,ord}, cat_to_{cart,ord},
gmv_search, gmv_cat, to_cart, to_ord, gmv, searches`. **There are no item ids, no category ids, no
prices, no sessions.** Every model in the project reads the same 11 numeric channels over the same
409 days. Ideas that require product/category identity, user–item affinity, price elasticity or
basket structure are not merely untried — they are impossible. The remaining degrees of freedom are
(a) history encoding, (b) target formulation, (c) loss/head, (d) which rows are trained on.

---

## 2. My own CV→LB transfer audit and the threshold I register

### 2.1 The exact local model, validated on primary evidence

For a step `D` added to a baseline with RMSLE `S`, `ΔRMSLE = (−2A + Q)/(2S)` with `A = mean(r·D)`
and `Q = mean(D²)`. Defining the scale-free unit alignment `a = A / rms(D)`, the best achievable
improvement over `α` is `Δ = −a²/(2S)`. This model is not a heuristic: `submission_LEVEL_MINUS_006`
is a pure level shift whose entire public loss must equal its own second-order term — observed
`+0.00108317`, predicted `+0.00108713`, agreement `4e-6` (`historical_transfer.csv`). I reproduced
the OOF side directly: FRESH's orthogonal `a_oof = 0.024628` predicts `−0.0001735`; the measured
fixed-α=1 orthogonal delta is `−0.0001727`.

### 2.2 Two transfer regimes, not one

| Case | honest OOF Δ | public Δ | gain transfer |
|---|---:|---:|---:|
| S1-DIST added to the then-current mix | −0.00071 (4/4) | −0.000503 | **0.71** (card records 0.64) |
| SEQ-01 blend | ≈ −0.0011 | −0.000601 | ≈ 0.55 |
| EXP-037 vs SEQ-01-MIX | −0.00092 LOFO (4/4) | −0.000519 | **0.564** — the card calls this the *third independent confirmation* of ≈0.57 |
| HOLIDAY-YOY | +0.00207 | +0.00108 | 0.52 |
| S1-MIX-E11 | −0.00038 | +0.00023 | **−0.60 (inverted)** |
| BTYD05 hedge vs STRONGEST | −0.000321 | +0.000274 | **−0.85 (inverted)** |
| SEQ65 vs STRONGEST | −0.000238 | +0.000537 | **−2.26 (inverted)** |
| ZERO2D | −0.0000248 (2/4) | −0.000110 | 4.4 (noise) |

The pattern is clean and it is *not* "transfer is negative". Transfer is `0.52–0.71` for **large
deltas from a genuinely new family added to a blend that had not yet been LB-optimised**, and it is
zero-to-inverted for **small deltas compared against a champion that was itself selected on public
LB** — where a winner's-curse penalty of about one public σ (`~1e-4`) fully accounts for the sign.
The `historical_transfer.csv` unit-alignment transfers (`−0.134, −0.310, −0.033, −6.04, −0.676`) are
all drawn from the second regime and all from in-span directions; the packet itself states that the
closer analogue — an out-of-span direction with canonical OOF — *cannot be constructed*, because
every source with canonical OOF returns an OOF-orthogonal component of `~4e-15`.

**Registered threshold (mine, from this evidence):**

- required TEST unit alignment for public `−0.001` (scale-free): `a_TEST = sqrt(2 × 1.6466 × 0.001) = 0.0574`;
- at gain transfer `τ = 0.55…0.67`, **required span-orthogonal nested ΔwCV ≈ `−0.0015 … −0.0018`**,
  central value **`−0.00175`**;
- the absolute floor, only if transfer were perfect (`τ = 1`, unsupported), is `−0.00094`;
- required orthogonal `a_oof`: `0.0418` for `−0.0005`, `0.0591` for `−0.001`, `0.0782` for `−0.00175`.

The user's prior of `−0.0014…−0.0018` is therefore **confirmed**, with one hard rider the prior did
not contain: this threshold is only valid if the **TEST-side correction is produced at the same
magnitude as the OOF correction**. EXP069's TEST correction was `0.405×` its OOF magnitude, which
inflates the requirement by roughly `1/0.405² ≈ 6×`. Magnitude parity between the OOF and TEST
production paths is not a detail; it is a first-order design constraint.

### 2.3 Where that leaves the target

FRESH — the only span-orthogonal direction the project owns — has orthogonal `a = 0.0237`. Reaching
`−0.001` public from a single new experiment needs `3.3×` that alignment, i.e. `11×` the explained
residual variance, on a direction that must also stay out of span. Nothing in eight months of this
project has produced anything of that size against the *current* baseline; the largest new-family
numbers on record (`DIST −0.00145` vs E10, `S04 conditional-positive −0.00128` vs its matched
control) were measured against far weaker baselines that no longer exist.

**Honest verdict: `≈0.001` public from one experiment is low probability (< 10 %).** The realistic
path is TYPE BASIS — two or three independently validated out-of-span directions of
`−0.0005 … −0.001` each, honestly combined on OOF before anything touches TEST. §6 states what that
requires numerically. This review therefore selects the experiment most likely to produce the
**second credible basis direction**, and says so instead of dressing a `−0.0003` candidate as a
`0.001` solution.

---

## 3. Critique of the proposed EXP072 (SEQ/ETX temporal distributional heads)

The proposal: freeze existing SEQ/ETX temporal representations, replace the point-regression head
with a matched 16-bin distributional head mirroring S1-DIST, controlled comparison
`same encoder + same rows + same embeddings`, `DIRECT head ↔ DIST16 head`.

I tried to falsify it against the ten questions posed, and it does not survive.

**(1) Has a distributional/quantile head already been run on SEQ or ETX?** Not exactly — no
registry row is a DIST16 head on a frozen SEQ/ETX embedding. Novelty of the exact object: genuine
but narrow.

**(2) Is the historical DIST gain a LightGBM regularization effect?** The S1-DIST card says so in
its own words: `Var(z − z_E10) = 0.01320` against a diversity threshold of `0.10` — *"the same
function, not a new class"*. Its pre-registered gate (`≥0.003` on ≥3 folds) was **not met**; the
dense-grid gain was `−0.0007…−0.0019`; OOF→LB transfer was `0.64`, not the `1.13–1.20` the project
had predicted. DIST's value was a bias/variance trade for a *tree* ensemble (softmax over 16 bins
plus a bin-mean readout is a different estimator of `E[z|x]` for trees). A neural head already
minimises the same objective smoothly on 6M rows; swapping to 16-way cross-entropy is a
regularisation change, not a new information channel.

**(3) Is the DIST direction still informative once temporal representations exist?**
This is the decisive measurement, and it is not favourable. On canonical OOF, orthogonalising the
DIST difference direction against `{SEQ-AVG3, ETX-AVG3, SEQ-D3A-AVG3}`:

- unexplained variance of DIST after projection: `0.657` (so the *vector* is not redundant);
- **unit alignment of the orthogonal part with the EXP-037 residual: `a = 0.00257`**;
- implied best-case ΔwCV: **`−1.9e-6`**.

The part of the distributional formulation that the temporal models do not already contain carries
essentially **no residual-aligned signal**. Symmetrically, `SEQ-AVG3 ⊥ {CAP,UNC,DIST}` gives
`a = 0.0054` and `ETX-AVG3 ⊥ {CAP,UNC,DIST}` gives `a = −0.0054`. Whatever `DIST × temporal`
interaction exists, its measurable residual alignment in the existing bank is two orders of
magnitude below the threshold.

**(4) Can frozen temporal embeddings be obtained honestly for all rows?** Yes for OOF: the
`S04SEQ_emb_SEQ-D3A-BASE-S42-V*_{clean,extra,val}_X.npy` caches exist for all four folds (~9 GB).
No for TEST without work: EXP069's reconnaissance records that
`artifacts/model_SEQ-D3A-BASE-S42-TEST.pt` **does not exist**; EXP069 had to train a production
encoder (5,729.9 s) to get one. ETX TEST checkpoints do exist (EXP071 verified them), but the ETX
side is closed (see 8).

**(5) Compute.** OOF pilot on cached embeddings is cheap; a full four-fold run plus a TEST
production encoder is `2–4 h` GPU on an RTX 4060 Ti 8 GB. Not prohibitive, but 4–8× the selected
alternative for materially worse priors.

**(6) Encoder/checkpoint mismatch between OOF and TEST.** Yes, and it is the exact failure mode
that cost EXP069 its magnitude: the OOF side is a single-seed cross-fit while TEST averaged two
donor sides × three seeds, and the TEST extensive probability is a reconstruction
(`reference_reproduced = false`, mean `|Δz| = 0.0445`). The registered outcome was a TEST correction
at `0.405×` OOF magnitude. A DIST-head variant inherits every one of these.

**(7) Correlation with S1-DIST.** The correction-correlation matrix on canonical OOF gives
`corr(d_dist, d_seq_avg3) = −0.487`, `corr(d_dist, d_etx_avg3) = −0.320`,
`corr(d_dist, d_hurdle_e11) = +0.500`, `corr(d_dist, d_holiday_yoy) = +0.700`. A DIST head on a SEQ
encoder sits inside a densely occupied part of the correction space.

**(8) Is magnitude the zone of strength for SEQ/ETX?** The ETX card is explicit: `AUC(y>0) 0.84135`
for ETX-AVG3 versus `0.84159` for SEQ-AVG3 — *"ETX wins on magnitude, not on ranking"*. So the
temporal models are already the magnitude specialists; a distributional head is a magnitude-side
reformulation of a magnitude-side model.

**(9) Does discretisation hurt the theoretical optimum?** Yes, mildly and unavoidably: 16-bin
`Σ p_k m_k` is a quantised estimator of `E[z|x]`; on tabular trees the variance reduction paid for
that bias, and the payment was already collected once. The most likely outcome is a slightly worse
copy of DIST living in DIST's own direction.

**(10) Base rate for "swap/add a head on a frozen or fine-tuned temporal encoder".** Five attempts
are on record and four failed *against matched controls*:
`EXP-038 FNL` — future funnel/cart heads on SEQ-D3A: the content-free `BUYCTRL` control scored
**better** than the real arms (`1.745910/1.746030` vs `1.746970/1.745950`, BASE `1.747130`,
noise-repeat control `1.747460`); `EXP-032B` hybrid — PASS on its own gate but `0/4` in the
production mixture, diversity collapsing `Var(z−z_tab) 0.041 → 0.015`, residual corr
`0.9933 → 0.9975`; `EXP-044 FRESH-COND-FT` encoder fine-tune — `FRESH−VOL −0.000088` against a
`−0.0003` gate, REJECT; `EXP-071 ETX-FRESH` — REAL worse than EXP-037 *and* worse than its
volume-matched placebo, CONFIRMED_REJECT. The single success is `EXP-069 FRESH`, whose orthogonal
value is `−0.00017`.

**Assessment (deliberately coarse):**

| quantity | EXP072-DIST as proposed |
|---|---|
| P(pilot positive on latest fold) | medium-low |
| P(full four-fold positive) | low-medium |
| P(nested orthogonal ≤ −0.0005) | low |
| P(nested orthogonal ≤ −0.001) | very low |
| P(reaching −0.00175, the ~0.001-public threshold) | very low |
| span expansion potential | medium (a head-swap contrast is a difference vector, so it will have some orthogonal part) but the *aligned* part is measured at `a≈0.0026` |
| compute | 2–4 h GPU plus a TEST production encoder |
| implementation risk | medium (embedding/checkpoint parity, TEST encoder absent) |
| production risk | high (inherits EXP069's magnitude-loss failure mode) |

**Verdict: the mechanism is not wrong, but it is measurably small.** It is a reformulation of
information the ensemble already holds, in a part of the correction space that is already dense, and
it belongs to the exact experiment class whose matched controls have failed four times out of five.
It should not be first priority.

---

## 4. Independent scan — internal shortlist and what killed most of it

Five mechanisms were considered seriously; two were killed before costing anything.

**(i) Temporal distributional target (EXP072 as proposed).** Kept, demoted — §3.

**(ii) Future accumulation / multi-horizon target structure (GMV or purchase-days at 7/14/21/30,
cumulative shape, acceleration).** *Killed on primary evidence.* This is auxiliary-head supervision
on the same encoder with the same inputs — precisely the object `EXP-038 FNL` built the `BUYCTRL`
control for, and the control won. `MHZ-{BASE,SELF,P30,FULL}` multi-horizon occurrence/hazard scored
`1.75191…1.75267` against `1.74751`; on the aligned bank `pred_mhz_full` has partial alignment
`0.00135` (δ `−5.2e-7`) — the direction is empty. The distinct part of the idea (cumulative GMV
trajectory rather than funnel events or hazards) adds **no new input information**, only a smoother
target, so the same control applies unchanged. Novelty: low. Expected value: low.

**(iii) Recover late-model canonical OOF (`occ_raw_X3`, `occ_meta_B`, SEQ65, Ridge stack) and build
a nonlinear conditional stack.** *Killed twice over.* Feasibility: `05_OOF_INVENTORY.csv` records
that `occ_meta_B` and `occ_raw_X3` are missing 80 fold NPZ files and ~9.7 GB of cache, EXP068 is
missing 32/32 historical OOF and 6/6 helper TEST checkpoints, and SEQ65 has no row-level OOF —
recovery means **retraining**, not loading. Value: even granted free recovery, all of these are
inside the 65-source span, so the geometry has already priced them; and the oracle analysis in §1.2
caps the whole family at `−0.00108` in-fold and `−0.00093` regime-aware. Oracle headroom below the
`0.002–0.003` kill line by a factor of two to three. **Rejected before training, exactly as the
oracle gate requires.**

**(iv) New target decompositions materially different from E11 / DIST16 / ZERO2D / EXP070
count-value.** Nothing survived that is both different and supported. The zero/positive split is the
dominant error mass (39.1 % of rows, 44 % of squared log error) but every implementation of it —
E11, ZERO2D, S04 two-part, BLOCK4 — either failed its control or collapsed in the mixture, and the
oracle for a *perfect* zero classifier (`−0.591`) is unattainable by definition.

**(v) Late-window (EXTRA-cutoff) supervision — the one channel the ensemble does not contain.**
Selected. §5.

---

## 5. The selected direction and the evidence behind it

### 5.1 The only genuinely new information channel in this project

Every one of the 65 submitted vectors was trained inside the clean corridor ending `2025-10-16`.
Exactly one object in the entire project has ever legally consumed data after that date: the
`EXTRA` cutoff set `2025-10-22 … 2026-01-14` (13 weekly cutoffs), used by `EXP-032B/EXP-040/EXP-069`
for **positive-target conditional-amount supervision only**. The full target is illegal there —
the 250,000 users were selected on guaranteed activity in `2025-11-16…2026-02-13`, so
`P(any activity next 30d) → 1` by construction and a full-target model on late cutoffs costs
`+0.054 RMSLE` with bias `+0.366` on a clean holdout (`FRESH-DIST-MIX` / exp_028). But the
*conditional amount given a purchase* is not manufactured by that selection rule, which is why the
conditional-amount channel is legal and why it is the only part of the model that can be refreshed.

That this channel is the source of the project's only out-of-span direction is now measurable, and
I verified it independently. Orthogonalising every aligned OOF direction against the other thirteen:

| direction | orthogonal RMS | partial `a` | best-case ΔwCV | unexplained var |
|---|---:|---:|---:|---:|
| **`pred_fresh_contrast`** | 0.02515 | **0.02374** | **−1.61e-4** | 0.943 |
| `pred_seq_d3a_avg3` | 0.08332 | 0.01575 | −7.10e-5 | 0.471 |
| `pred_hurdle_e11` | 0.08969 | 0.01243 | −4.42e-5 | 0.637 |
| `pred_ridge15` | 0.20140 | −0.00886 | −2.24e-5 | 0.934 |
| `pred_block4_saf` | 0.04399 | 0.00343 | −3.36e-6 | 0.984 |
| `pred_mhz_full` | 0.10830 | 0.00135 | −5.21e-7 | 0.704 |
| CAP/UNC/DIST/SEQ/ETX/BTYD | ~0 | ~0 | ~0 | ~0 (exactly collinear with EXP-037) |

FRESH is the only family with both a large unexplained fraction **and** a residual-aligned
orthogonal component. Its TEST vector is `87.5 %` out of the geometry span and raised the numerical
rank `57 → 58` — the first time the span moved in this project.

### 5.2 A confound I tested and did **not** confirm

The reported *nested* orthogonal fold deltas of FRESH (`+1.19e-5, −1.02e-5, −1.27e-4, −2.59e-4`)
track the calendar overlap between EXTRA target windows and fold target windows almost exactly
(0, 0, 10 and 54 overlapping days). If the orthogonal gain were calendar overlap, it would not
transfer to TEST at all, whose target window `2026-02-14…2026-03-15` overlaps nothing.

I tested this directly by computing, **per fold**, the alignment of the FRESH correction after
orthogonalising it against all other OOF directions *within that fold*:

| fold | raw `a` | partial `a` | orthogonal RMS | best-case Δ | overlap days |
|---|---:|---:|---:|---:|---:|
| 2025-09-04 | 0.02954 | 0.02633 | 0.02452 | −1.98e-4 | 0 |
| 2025-09-18 | 0.02404 | 0.02282 | 0.02260 | −1.49e-4 | 0 |
| 2025-10-02 | 0.02609 | 0.02657 | 0.02386 | −2.02e-4 | 10 |
| 2025-10-16 | 0.02971 | 0.01751 | 0.02775 | −8.77e-5 | 54 |

The orthogonal alignment is **flat to slightly declining**, and the *most* overlapped fold is the
**weakest**. The overlap story is not supported; the reported nested pattern is an artefact of
donor-fold-fitted projection, not of calendar leakage. **The FRESH channel is real.** The honest
caveats that survive are different and milder: the orthogonal alignment does not grow toward TEST
(so no recency bonus should be assumed), and the TEST magnitude loss (`0.405×`) remains unexplained.
Because the confound is cheap to re-test under a stricter protocol and its failure mode would be
catastrophic for transfer, the selected experiment retains a no-overlap arm as a mandatory control.

### 5.3 The regime the channel is supposed to capture — measured from the raw panel

I recomputed the eligibility panel and 30-day target directly from `train.parquet` on 41 weekly
cutoffs (`2025-04-03 … 2026-01-08`; dense matrix, 250,000 × 409):

| cutoff | eligible n | `P(y>0)` | `E[log1p y \| y>0]` | segment spread (16+ buy-days − 0 buy-days) |
|---|---:|---:|---:|---:|
| 2025-04-03 | 170,705 | 0.5444 | 4.2645 | 1.927 |
| 2025-09-04 (fold 1) | 188,518 | 0.6002 | 4.3183 | 1.937 |
| 2025-10-16 (fold 4) | 197,379 | 0.6129 | 4.2938 | 1.957 |
| 2025-11-27 (peak) | 209,608 | 0.6347 | 4.4100 | 2.037 |
| 2026-01-08 (trough) | 227,042 | 0.5682 | 4.1608 | 1.906 |

Two facts matter. First, the panel grows from 170.7k to 227.0k as the guaranteed-activity window is
approached — the documented extensive-margin contamination, visible and confirmed. Second, the
**conditional amount regime genuinely moves**: the level by `0.25` log units across the year and the
cross-segment spread by `0.13`. The level part is worthless (per-fold log offset removes it on OOF;
the geometry intercept and the demonstrated near-optimality of this family's TEST level remove it
there). The **differential** part — roughly `0.05–0.13` log units of segment-dependent movement,
concentrated in exactly the window no submitted model has seen — is the exploitable quantity, and
today it is extracted by a single `Linear(192,64)→GELU→Linear(64,1)` head (~12k parameters) sitting
on a frozen encoder that was trained for a *different* objective.

That gap between the size of the signal and the capacity pointed at it is the reason this is the
highest-EV experiment on the board.

---

## 6. Ranking

See `CANDIDATE_RANKING.csv`. Summary of the three that survived:

| rank | candidate | oracle headroom | expected orthogonal nested ΔwCV | P(≥0.0005) | P(≥0.001) | P(≥0.00175) | runtime |
|---|---|---|---|---|---|---|---|
| 1 | **EXP072 LWA-TAB** — late-window conditional-amount transfer, tabular, full capacity | not capped by any existing-source oracle (new data channel); FRESH proves `a=0.0237` exists at 12k parameters | **−0.0002 … −0.0008** | low-medium | low | very low | **~45 min CPU pilot**, ~2.5 h full |
| 2 | EXP072-DIST as proposed — DIST16 head on frozen SEQ/ETX | `a(DIST ⊥ SEQ/ETX) = 0.0026` → `−1.9e-6` | −0.0000 … −0.0003 | low | very low | very low | 2–4 h GPU |
| 3 | EXP070 four-fold completion | in-span; family capped at `−0.00093` regime-aware and already priced by geometry | −0.0000 … −0.0001 | very low | ~0 | ~0 | 40–50 min |

Rank 3 is retained only as **scientific closure**: `2025-10-02` was never trained, so EXP070 has no
canonical wCV and no honest LOFO, and its placebo currently beats the real model on 2 of 3 completed
folds. Finishing it is cheap and settles a hanging question, but its endpoint
`z_exp037 + 0.25·(z_count − z_dist)` is a DIST replacement living entirely inside the priced span,
so its public expectation is ~0 whatever it returns. It is not an EV competitor.

---

## 7. Decision

**DECISION C — REPLACE.** EXP072 as proposed (SEQ/ETX temporal distributional heads) is **rejected
as first priority** on the measured evidence in §3: the residual-aligned component of a
distributional reformulation that the temporal models do not already contain is `a = 0.0026`
(`−1.9e-6` wCV), and four of the five head-swap experiments on record lost to their matched
controls. It is not deleted — it is demoted to rank 2 and, if it is ever run, it should be run
*inside* the late-window channel rather than on clean-corridor data.

**Selected: `EXP072 — LWA-TAB` (Late-Window conditional-Amount Transfer, tabular).** The id `EXP072`
is the next free id after an explicit registry check; only its content changes.

One falsifiable hypothesis: *the legal late-window conditional-amount channel — the only training
data outside the 65-source span's information set — is under-exploited by a factor of at least two,
and replacing the 12k-parameter frozen-encoder head with a full-capacity tabular conditional-amount
model on the frozen 227-column `LnormNone` feature family produces a span-orthogonal nested ΔwCV
against EXP-037 of `≤ −0.0005`, with `REAL − VOL ≤ −0.0003`, surviving removal of every EXTRA
cutoff whose target window overlaps the evaluated fold.*

Why it has the highest expected value:

1. It is the only mechanism with a **confirmed PASS**, a **confirmed out-of-span TEST geometry**
   (87.5 %), and an **independently verified non-zero orthogonal alignment** (`a = 0.0237`, §5.1).
2. Its current implementation is demonstrably capacity-starved relative to the signal measured in
   §5.3, and the tabular channel is where this project's amount modelling has always been strongest.
3. Its cost is trivial: every prerequisite is cached. The `feat_*_LnormNone.parquet` family covers
   **all 13 EXTRA cutoffs and the TEST cutoff `20260213`**; panel caches exist through `20260213`;
   the extensive-probability vector `p_dist` is already saved by EXP069 for both OOF and TEST.
   The pilot is CPU-only LightGBM, ~45 minutes.
4. It is the only design that can **fix the `0.405×` magnitude loss**, because both the OOF and the
   TEST side are produced by the identical estimator with identical seeds and rounds — no
   reconstructed checkpoint, no side/seed averaging asymmetry. Per §2.2 that alone is worth up to a
   factor of six in deployed value.
5. It has a genuinely cheap kill: if `REAL − VOL` on the latest fold is above `−0.0001`, or the
   no-overlap arm keeps less than half the gain, or the centred correction's span-orthogonal
   fraction is below `0.5`, the family is closed and the project stops spending on it — which is
   itself worth more than another `−0.0002` in-span candidate.

Realistic expectation, stated without decoration: **span-orthogonal nested ΔwCV `−0.0002 … −0.0008`**,
most likely near `−0.0004`. That is `PASS_BASIS`, not `PASS_BIG`. Probability of clearing the
`−0.00175` threshold that `≈0.001` public requires: **very low**.

---

## 8. If multiple basis directions are the real answer — and they are

The evidence says the `0.001` target is reachable, if at all, only as a sum of independent
out-of-span directions. Stating that arithmetic explicitly so it is not fudged later:

- Two orthogonal directions with alignments `a₁`, `a₂` and correction correlation `ρ` give a
  combined alignment `a² = (a₁² + a₂² − 2ρ a₁a₂)/(1 − ρ²)`. For `a₁ = a₂` and small `ρ`, the
  combined value is `≈ 2a₁²/(1+ρ)`.
- Target `a_combined = 0.0782` (the `−0.00175` threshold). With `a₁ = 0.0237` (FRESH as it stands),
  a partner direction would need `a₂ ≈ 0.075` at `ρ = 0` — i.e. the partner must essentially carry
  the whole target alone. **Two FRESH-sized directions are not enough.**
- With `a₁ = a₂ = 0.05` and `ρ = 0.2`, `a_combined = 0.0645` → nested ΔwCV `−0.00119` → public
  `≈ −0.00068` at `τ = 0.57`. That is the realistic best case for a two-direction basis, and it
  requires *both* directions to be twice FRESH.
- With `a₁ = 0.0237`, `a₂ = 0.045`, `ρ = 0.5`: `a_combined = 0.0448` → `−0.00057` nested → public
  `≈ −0.00033`. This is the honest median outcome if EXP072 LWA-TAB passes.

Required correction correlation between the new direction and FRESH: `ρ ≤ 0.6` for the combination
to be worth running; `ρ ≤ 0.3` for it to be worth much. The combination must be fitted **on OOF
only**, with nested per-fold weight selection over a `{0, 0.25, 0.5, 0.75, 1.0}` grid for each
direction, and only then projected out of the 65-source span and added to the TEST geometry — never
the reverse order, and never with a weight informed by any public score.

Practical implication: the next two experiments should be planned as a pair. `EXP072 LWA-TAB` is
selected because it is the candidate most likely to become the second usable basis vector, and
because its failure closes the last open family cheaply.

---

## 9. Standing rules re-affirmed for whatever runs next

- No residual is ever built against `1.6466079084`; the only honest OOF anchor is `pred_exp037`.
- No public-LB geometry weight is projected onto OOF rows; no synthetic champion OOF is constructed.
- Folds are exactly `2025-09-04 / 2025-09-18 / 2025-10-02 / 2025-10-16`, weights `1:2:4:8`,
  per-fold global log offset, 770,616 rows.
- EXTRA cutoffs may contribute **positive-target conditional-amount rows only**, from the opposite
  `splitmix64(user_id) & 1` side, and may never touch eligibility, the extensive probability,
  feature scaling, EXP-037 components or validation labels.
- TEST inference is authorised only after an honest OOF PASS; α is fixed before any upload; no
  public score is used for selection, scaling or levelling.
