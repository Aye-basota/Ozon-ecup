# EXP-038 — Future-Funnel Supervision (FNL) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use
> checkbox (`- [ ]`) syntax. This plan is executed inline in the authoring session, so
> implementation code lives in `src/fnl.py` rather than being duplicated here; every task
> below fixes the *interfaces*, the *tests that must exist*, and the *acceptance command*.

**Goal:** Measure whether *future* Search/Cart labels drawn from `(T, T+h]` give the
SEQ-D3A encoder information that purchase supervision (`z30`, `buy30`) does not — using
four arms (BASE / BUYCTRL / CART / FUNNEL) that share encoder, batch order, seed and
compute budget.

**Architecture:** One new module `src/fnl.py` runs **all four arms through one code path**;
an arm is just a list of auxiliary heads (possibly empty). The encoder, depth policy, input
channels, cutoff grid, panels, optimizer recipe and the main `z30` target are the confirmed
`SEQ-D3A` ones and are not touched. Auxiliary heads are a single `Linear(3H → M)` probe on
the same pooled vector the main head reads, zero-initialised, built **after** the TCN — so
the encoder's initial weights and the batch-order stream are bitwise identical across arms
and `λ = 0` recovers BASE exactly. Labels come from a new dense raw-count array
`seq_fut_v1.npy`, built the same way as `seq_gmv_v1.npy`.

**Tech Stack:** Python 3, numpy, polars, PyTorch 2.11+cu126 (eager — `torch.compile` does
not work on this Windows box), pytest. Reused project modules: `src.seq`, `src.features`,
`src.report`, `src.tracking`, `src.validation`, `src.ptime_eval`, `src.merge_oof`.

**Spec:** the session prompt (requirements reproduced under *Global Constraints*), plus
`AGENTS.md`, `STATE.md`, `experiments/exp_024_multihorizon_hazard.md`,
`experiments/exp_030c_seq_d3a_multiseed.md`, `experiments/exp_036…037`.

## Global Constraints

- **Do not modify** `src/validation.py`, `src/config.py`, `src/seq.py`, `src/etx.py`.
  `src/seq.py` is production code for `SEQ-AVG3`; `src/fnl.py` imports from it, never edits it.
- Seed from `config.SEED` (42). Pilot is **fold `2025-10-16`, seed 42 only**. No 4-fold,
  no multi-seed, no test model, no submission until the gate passes.
- Encoder recipe frozen: `hidden=64 blocks=8 kernel=3 dropout=0.10 batch=1024 chunk=256
  lr=3e-3 wd=1e-2 epochs=4 warmup=300`, AdamW cosine, bf16 autocast, 17 channels, window
  365, `--depth-aug 0.5` with grid `90 120 150 180 220 254 289`, train panel 1-block,
  val panel 3-block, train cutoffs `T + 30 ≤ V`.
- Auxiliary labels are built **strictly** from `(T, T+h]`. Horizon `h` is legal for a train
  cutoff `T` in fold `V` only if `T + h ≤ V`. No cutoff later than
  `CORRIDOR_END = 2025-10-16` is used for supervision of any kind.
- Loss: `L = MSE(z30) + λ · s_z · mean_m(L_m / s_m)`, with `s_z = Var(z30)` on train and
  `s_m` the constant-predictor loss of head `m` on train. λ grid is exactly `{0.1, 0.3}` —
  **no further tuning permitted**. If both λ give the same conclusion, stop.
- Inference prediction is the **direct `z30` head only**; auxiliary heads never post-process.
- Arms are exactly: `BASE` (no heads); `BUYCTRL` (`buy30`); `CART` (`any_cart_7/14/30`,
  `log1p(to_cart_30)`); `FUNNEL` (CART + `any_search_7/14/30`, `log1p(searches_30)`).
  No other heads "for count".
- Data (`data/`, `submissions/`) is never committed.
- Every long run needs `PYTHONIOENCODING=utf-8` (cp1251 console kills the process on `Δ`).
- Never run two processes that both load the dense panel: 33.9 GB RAM, ~16.8 GB free, one
  training process needs 6–8 GB (`ecup-seq-gpu-budget`).

## File Structure

| file | responsibility |
|---|---|
| `src/fnl.py` | future-count array, label builder, arm registry, aux model, training loop, CLI |
| `src/test_fnl.py` | anti-leakage tests + the equivalence controls that make arms comparable |
| `research/strategies/results/FNL1/run.sh` | the 7-run pilot queue, strictly sequential |
| `research/strategies/results/FNL1/analyze.py` | all diagnostics → CSVs + decision table |
| `experiments/exp_038_fnl_future_funnel.md` | the card: 5 questions + verdict |

---

### Task 1: Future-count array and leakage-safe label builder

**Files:** create `src/fnl.py`, create `src/test_fnl.py`.

**Interfaces produced:**
- `FUT_COLS = ("searches", "to_cart")`; `FUT_NPY = DATA_PROCESSED / "seq_fut_v1.npy"`,
  shape `(250_000, 409, 2)` `uint16` (observed maxima 630 and 1535).
- `build_future(force: bool = False) -> None`, `future() -> np.ndarray`
- `@dataclass(frozen=True) Head(name: str, kind: str, src: str, h: int)`
  with `kind ∈ {"bin","reg"}`, `src ∈ {"searches","to_cart","gmv"}`
- `ARMS: dict[str, tuple[Head, ...]]`
- `aux_labels(T, rows, heads) -> np.ndarray` `(n, M) float32`
- `aux_scales(A, heads) -> (s, b)` — constant-predictor loss and its optimal parameter
- `fold_cutoffs_for_heads(V, heads) -> list[dt.date]`
- `build_index_aux(cuts, blocks, heads) -> (ci, ri, zy, A)`

**Why a new array:** `seq_panel_v1.npy` stores `log1p` per channel, and
`log1p(Σx) ≠ Σ log1p(x)`, so the `log1p(to_cart_30)` / `log1p(searches_30)` labels need raw
daily counts. Binary labels come from the same array (window sum > 0), so both label kinds
have one source and one code path.

- [ ] **Step 1: Write the failing tests in `src/test_fnl.py`**
  - `test_future_array_matches_raw_counts` — dense array equals the log on sampled rows.
  - `test_labels_read_only_their_window[7|14|30]` — labels equal a polars recomputation over
    `(T, T+h]` only.
  - `test_regression_label_is_log1p_of_the_window_sum`
  - `test_buy30_label_equals_the_main_target_being_positive` — `buy30 ≡ target_at(T) > 0`
    row for row, i.e. the new labelling is not weaker on leakage than the target itself.
  - `test_every_train_cutoff_is_legal_for_every_horizon[arm]` — `T + h ≤ V` and
    `T ≤ CORRIDOR_END` for all folds, all arms.
  - `test_aux_horizons_never_exceed_the_main_target_horizon[arm]` — since every `h ≤ 30`,
    `fold_cutoffs_for_heads(V) == seq.fold_cutoffs(V)`; the cutoff grid does not change.
  - `test_validation_labels_do_not_touch_the_poisoned_window[arm]` —
    `V + h < 2025-11-16` for all four folds.
  - `test_scales_are_the_constant_predictor_loss` — `H(p̄)` for binary, `Var` for regression;
    biases `logit(p̄)` and `mean`.
  - `test_index_labels_line_up_with_rows_and_cutoffs`
- [ ] **Step 2: Run — expect `ModuleNotFoundError: src.fnl`**
      `PYTHONIOENCODING=utf-8 python -m pytest src/test_fnl.py -q`
- [ ] **Step 3: Implement the label half of `src/fnl.py`**
- [ ] **Step 4: `python -m src.fnl build` then rerun pytest — all green**
- [ ] **Step 5: Commit** `feat(fnl): future funnel labels + anti-leakage tests`

---

### Task 2: Auxiliary-head model that leaves BASE bitwise unchanged

**Files:** modify `src/fnl.py`, `src/test_fnl.py`.

**Interfaces produced:**
- `build_net(cfg, heads, aux_bias) -> FunnelNet` with `.tcn` (the unmodified `seq` TCN),
  `.aux` (`nn.Linear(3*hidden, M)` or `None`), `forward(x) -> z` (main head only, so
  `seq.predict` works unchanged) and `forward_all(x) -> (z, aux_logits)`.

**Decisions locked here:**
- The TCN is constructed **first** so it draws exactly the same randoms as BASE; the aux
  head is constructed after. This is what makes the arms paired.
- The aux head is a bare linear probe — no hidden layer. The experiment asks whether the
  labels change the *representation*; head capacity would let the head absorb the task
  instead. Cost: +193/+772/+1544 parameters against 245 633.
- Aux weights zero, bias = constant predictor, mirroring `z0` on the main head: at step 0
  the normalized aux loss is exactly 1.0 and its gradient into the encoder is exactly 0.

- [ ] **Step 1: Write the failing tests**
  - `test_encoder_init_is_identical_across_arms[arm]` — `build_net(...).tcn.state_dict()`
    equals `seq.build_model(cfg).state_dict()` tensor for tensor at the same seed.
  - `test_aux_head_starts_at_the_constant_predictor[arm]`
  - `test_forward_returns_the_same_z_for_every_arm[arm]`
- [ ] **Step 2: Run — expect `ImportError: cannot import name 'build_net'`**
- [ ] **Step 3: Implement `build_net` / `FunnelNet`**
- [ ] **Step 4: Rerun pytest — all green**
- [ ] **Step 5: Commit** `feat(fnl): aux-head model with bitwise-identical encoder init`

---

### Task 3: Training loop, normalized loss, and the λ=0 ≡ BASE control

**Files:** modify `src/fnl.py`, `src/test_fnl.py`.

**Interfaces produced:**
- `class AuxBatcher(seq.Batcher)` — same `_plan()`, yields `(x, y, a)`
- `aux_loss(logits, targets, kinds, scales) -> Tensor` = `mean_m(L_m / s_m)`
- `fit_model(cuts, cfg, heads, lam, eval_fn=None) -> (model, dev, cfg, hist)`
- `predict_aux(model, T, rows, cfg, dev) -> np.ndarray` `(n, M)` logits
- `train_fold(V, cfg, heads, lam, curve, ckpt, n_cutoffs, val_frac) -> dict` with keys
  `user_id, z, y, aux_pred, aux_true, hist, head_names, head_kinds`

**Decisions locked here:**
- The main term is **not** normalized: its gradient must stay identical to BASE, otherwise
  `λ = 0` would not reproduce the base. `s_z` rescales the normalized aux into the main
  term's units, so `λ = 0.3` reads directly as "the auxiliary task weighs 30% of the main
  one at initialisation" (`L_z30 = s_z` there, because the main bias is init to `mean z`).
- Heads are **averaged**, not summed. BUYCTRL has 1 head and FUNNEL has 8; summing would
  give FUNNEL 8× the auxiliary gradient and the comparison would measure regularisation
  strength instead of the information source.
- `clip_grad_norm_` stays a **single global clip** over all parameters — changing the
  optimizer recipe is forbidden. The pre-clip norm is logged per epoch so the card can say
  whether clipping started to discriminate between arms.

- [ ] **Step 1: Write the failing tests**
  - `test_batch_order_does_not_depend_on_the_arm` — `_plan()` identical for all four arms.
  - `test_batch_labels_match_the_rows_of_the_batch`
  - `test_inputs_are_bitwise_the_base_inputs` — aux labels never enter the input tensor.
  - `test_aux_loss_is_one_at_the_constant_predictor`
  - `test_lambda_zero_is_bitwise_the_base_run` — tiny 1-epoch config, `FUNNEL @ λ=0` gives
    exactly the same `z` as `BASE`.
- [ ] **Step 2: Run — expect `ImportError: cannot import name 'AuxBatcher'`**
- [ ] **Step 3: Implement the training section**
- [ ] **Step 4: Rerun pytest — all green**
- [ ] **Step 5: Commit** `feat(fnl): normalized aux loss with lambda=0 == BASE control`

---

### Task 4: Fold runner, CLI and smoke check

**Files:** modify `src/fnl.py`, `src/test_fnl.py`.

**Interfaces produced:** CLI `python -m src.fnl {build,smoke,fold}`. Per run:
`artifacts/oof_<EXP>-V1016.npz`, `artifacts/fnl_<EXP>-V1016.npz`
(`aux_pred`, `aux_true`, `head_names`, `head_kinds`), `artifacts/curve_<EXP>-V1016.json`,
`artifacts/model_<EXP>-V1016.pt`.

- [ ] **Step 1: Write the failing tests** — `test_arms_are_exactly_the_four_specified`
      (exact head names, in order) and `test_cli_exposes_build_smoke_fold`.
- [ ] **Step 2: Run — expect `ImportError: cannot import name 'main'`**
- [ ] **Step 3: Implement `train_fold`, `cmd_fold`, `cmd_smoke`, `main`.**
      `train_fold` asserts `V + h < 2025-11-16` for every head before training.
- [ ] **Step 4: `python -m src.fnl smoke` then full pytest.** Smoke must print head
      prevalences near `any_cart_7≈0.45 any_cart_30≈0.77 any_search_7≈0.69 any_search_30≈0.93`.
- [ ] **Step 5: Commit** `feat(fnl): fold runner and CLI`

---

### Task 5: Run queue — 7 pilot runs on fold 2025-10-16, seed 42

**Files:** create `research/strategies/results/FNL1/run.sh`, `.../README.md`.

Runs, in this order (most informative first, so the answer arrives before the replicate):

| # | exp id | arm | λ |
|---|---|---|---|
| 1 | `FNL-BASE-L00-S42` | BASE | — |
| 2 | `FNL-FUNNEL-L30-S42` | FUNNEL | 0.3 |
| 3 | `FNL-BUYCTRL-L30-S42` | BUYCTRL | 0.3 |
| 4 | `FNL-CART-L30-S42` | CART | 0.3 |
| 5 | `FNL-FUNNEL-L10-S42` | FUNNEL | 0.1 |
| 6 | `FNL-BUYCTRL-L10-S42` | BUYCTRL | 0.1 |
| 7 | `FNL-CART-L10-S42` | CART | 0.1 |

Strictly sequential (RAM), `--curve` on every run (epoch curves are a required diagnostic
and cost ~1 min), skip a run whose OOF already exists so the queue is restartable.
Budget: 7 × ~75 min ≈ 9 h on the RTX 4060 Ti in eager mode.

- [ ] **Step 1: Write `run.sh`**
- [ ] **Step 2: `bash -n run.sh` → SYNTAX_OK**
- [ ] **Step 3: Launch in the background**
- [ ] **Step 4: `ls artifacts/oof_FNL-*-V1016.npz | wc -l` → 7**
- [ ] **Step 5: Commit the scripts (not the artifacts)**

---

### Task 6: Diagnostics that make the experiment informative

**Files:** create `research/strategies/results/FNL1/analyze.py`.
Writes `folds.csv`, `segments.csv`, `aux_auc.csv`, `diversity.csv`, `curves.csv`.

Calibration rule: per-fold optimal log shift computed on the **whole fold**, never inside a
segment (`rmsle_diagnostics` §3, `ptime_eval` docstring) — segment recalibration is closed.

1. **`folds.csv`** — per arm: `rmsle`, `rmsle_cal`, `offset`, `bias`, `mean_z`,
   `auc_y30_pos`, `d_cal_vs_BASE`, `d_cal_vs_BUYCTRL` (matched λ).
2. **`segments.csv`** — per arm × segment from `ptime_eval.segments`, restricted to the
   fold: `n`, `mse_share`, `rmsle`, `auc`, deltas to BASE and to matched-λ BUYCTRL. Must
   include by name: `ВСЕ`, `rec_buy 15-60`, `w180_days_buy 2-15`,
   `пересечение 2-15 x 15-60`, `никогда не покупал`, `w180_days_buy 16+` (high activity).
3. **`aux_auc.csv`** — for every arm and every head it was trained on: AUC of the head's own
   prediction against its own label (binary) or `corr`/`R²` (regression), **and** the AUC of
   the same label scored by that arm's main `z` head. The second column is the one that
   discriminates: it says whether the funnel label carries anything the point forecast does
   not already rank — the exact question `exp_024` answered negatively for hazard/count.
4. **`diversity.csv`** — `Var(z_arm − z_BASE)`, `corr(z_arm, z_BASE)`,
   `corr(resid_arm, resid_BASE)` with `resid = log1p(y) − z_cal`.
5. **`curves.csv`** — per epoch: `rmsle_cal`, `train_mse`, `train_aux`, `grad_norm`.

- [ ] **Step 1: Write `analyze.py`**
- [ ] **Step 2: Run it; the printed table has one row per arm with both delta columns**
- [ ] **Step 3: Commit**

---

### Task 7: Verdict, experiment card, project bookkeeping

**Files:** create `experiments/exp_038_fnl_future_funnel.md`; modify `experiments/log.csv`,
`experiments/README.md`, `STATE.md`.

- [ ] **Step 1: Apply the gate** on `2025-10-16` calibrated RMSLE from `folds.csv`:

```text
STRONG CONTINUE : FUNNEL − BASE ≤ −0.003  AND  FUNNEL − BUYCTRL ≤ −0.001
                  AND rec_buy 15–60 does not get worse
CONTINUE        : FUNNEL − BASE ≤ −0.0015 AND a clear gain over BUYCTRL,
                  or a strong AUC gain in the problem segments
REJECT          : |FUNNEL − BUYCTRL| ≤ 0.0005, or the aux heads are well predicted
                  while the main GMV/buy30 prediction does not improve
```

Use the better λ for FUNNEL and the **matched** λ for BUYCTRL. If the two λ disagree in
sign, report both and call the result unresolved at this sample size — do not pick the
favourable one.

- [ ] **Step 2: Write the card**, answering explicitly and in order:
  1. Do future Cart/Search labels give signal beyond `buy30`?
  2. Which source is more useful, Search or Cart (`CART` vs `FUNNEL`)?
  3. Did the main bottleneck `rec_buy 15–60` improve?
  4. Does any improvement come through activity ranking (AUC) or GMV magnitude
     (RMSLE among `y>0`)?
  5. Is there a realistic path to a **large** gain rather than `1e-4`?

  The card must also state the power limitation plainly: one fold, one seed; the paired
  per-fold delta sd for this encoder on this fold across seeds is **0.00038**
  (`exp_030c`), so the REJECT band `±0.0003…0.0005` sits at the resolution limit of a
  single run. What makes a REJECT defensible is the agreement of the two λ plus the
  aux-AUC diagnostics, not the RMSLE number alone.

- [ ] **Step 3: On REJECT, name the closed class** — one paragraph in the card and one line
  in STATE.md «Не повторять»: *future funnel labels (Search/Cart at any horizon ≤ 30) as
  auxiliary supervision for a sequence encoder give nothing beyond purchase supervision* —
  so no future agent retries it with `to_ord`, `cat`, per-category carts, longer horizons,
  or contrastive intent objectives.

- [ ] **Step 4: On CONTINUE / STRONG CONTINUE, write the follow-up plan (do not run it)** —
  4 folds → 3 seeds of the winning arm → honest LOFO inside
  `0.10·CAP + 0.20·UNC + 0.25·DIST + 0.225·ETX-AVG3 + 0.225·SEQ-AVG3`, CAP non-zero, test
  level `L* = 2.3293`, blending in `log1p`, test depth policy `clip289` / `DCW`, and the
  `exp_036` regime gate (`Var(Δz)` test/OOF ratio must land in 0.6–1.2×). No LB submission
  if the final gain is only ≈ −0.0005; the bar is 4-fold `ΔwCV ≤ −0.002`, ideally
  `−0.003…−0.005` before ensembling. Check specifically whether the encoder improves the
  **mid-activity zone**, where ETX has not broken through.

- [ ] **Step 5: Update the journals and commit**

---

## Self-Review

**Spec coverage.** Frozen base architecture → Global Constraints + Task 3. Single fold/seed
→ Global Constraints + Task 5. Four arms → Task 1 `ARMS`, Task 4 test. Label semantics and
`T + h ≤ V` → Task 1 tests. No dirty late cutoffs → `fold_cutoffs_for_heads` assertion +
test. Automated anti-leakage tests → Tasks 1 and 3. Loss normalization and the
`λ ∈ {0.1, 0.3}` grid → Task 3 + Task 5 queue. Direct `z30` head at inference →
`FunnelNet.forward` + Task 2 test. All eleven required diagnostics → Task 6 items 1–5.
`FUNNEL vs BUYCTRL` primary and `CART vs FUNNEL` secondary → Task 6 `folds.csv` columns and
Task 7 gate. Gate thresholds → Task 7 Step 1. Post-gate plan → Task 7 Step 4. Forbidden
list (no MHZ, no hazard, no blend-weight search, no new calibration, no bigger net, no
handcrafted features, no dirty labels) → nothing in Tasks 1–7 touches any of them. Five
report questions and the final verdict → Task 7 Step 2. Closed-class statement on REJECT →
Task 7 Step 3.

**Placeholders.** None. Task 6 names the exact CSV columns instead of saying "write
diagnostics"; implementation code is deliberately in `src/fnl.py` because this plan is
executed inline, not handed to a cold worker.

**Type consistency.** `Head(name, kind, src, h)` is used identically in Tasks 1–4.
`aux_scales` returns `(s, b)` everywhere. `build_index_aux` returns `(ci, ri, zy, A)` and
`AuxBatcher.__init__` takes `A` in the fifth position. `fit_model(cuts, cfg, heads, lam,
eval_fn)` matches its call in `train_fold`. `train_fold` returns a dict whose keys
(`user_id, z, y, aux_pred, aux_true, hist, head_names, head_kinds`) are exactly the keys
`cmd_fold` reads.
