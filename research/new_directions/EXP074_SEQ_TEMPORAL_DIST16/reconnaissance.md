# EXP074 reconnaissance

## Registry

`EXP074` is free in both registries inspected:

- `C:/Users/Admin/Desktop/e-cup-research-clean/registry/experiments.csv`;
- `C:/Users/Admin/Desktop/submission_geometry_research/gpt_pro_research_packet/02_EXPERIMENT_REGISTRY.csv`.

No existing experiment was renamed or edited.

## Exact historical S1-DIST formulation

Authoritative implementation: `C:/Users/Admin/Desktop/OZON-E-CUP/src/models.py`, SHA256
`c0dd7b75dc15d72a12b2f2c672ca6f7050f02034e0763fc8057d1e7b9db0ada4`.

- Work in `z = log1p(GMV30)`.
- There are 16 classes. Edge 0 is fixed at `1e-9`, separating exact zero from positive values.
- The other 14 boundaries are computed on the current fold's training rows only as
  `quantile(z[z>0], [1/15, 2/15, ..., 14/15])`.
- Labels are `searchsorted(edges, z, side="right")`.
- Each centre is the mean training `z` in that class. An empty class inherits the nearest
  non-empty centre on its left.
- Prediction is the raw multiclass probability vector multiplied by the training centres:
  `z_dist = probabilities @ centres`.

Authoritative parameters: `C:/Users/Admin/Desktop/OZON-E-CUP/src/config.py`, SHA256
`8dd82be5291da32971ac285b13e3eb02c5702bc0e5eaec9152e7002159dcf324`.
S1-DIST used 250 rounds, learning rate `0.05`, `num_leaves=127`,
`min_data_in_leaf=200`, `feature_fraction=0.7`, `bagging_fraction=0.8`,
`bagging_freq=1`, `lambda_l2=5`, `max_bin=63`, `force_row_wise=true`, seed `42`.
DIST overrides the objective/metric with `multiclass`/`multi_logloss` and
`num_class=16`; matched DIRECT retains `regression`/`rmse`.

## Exact current SEQ representation and builder

Authoritative encoder: `C:/Users/Admin/Desktop/OZON-E-CUP/src/seq.py`, SHA256
`d95e0101692169781ea1ca6c585629cbb2f06d340e39f8afeb5a8e8e164c88b5`.

- This is historical plain `SEQ-01`, not D3A: 17 ordered channels, hidden width 64,
  eight causal dilated TCN blocks, 365-day input.
- Channel order is `present, cat, buy, ponly, searches, search_to_cart, search_to_ord,
  cat_to_cart, cat_to_ord, to_cart, to_ord, gmv_search, gmv_cat, gmv, avail, dow_sin,
  dow_cos`.
- The exact final representation before the head is 192-dimensional:
  `concat(final_hidden[:, -1], mean_time(final_hidden), max_time(final_hidden))`.
- `fold_cutoffs(V)` uses the canonical cutoff grid and enforces `T + 30 days <= V`.
  `build_index(..., blocks=1)` builds CLEAN one-block training rows and `log1p(GMV30)`;
  validation uses the canonical three-block panel.

An embedding hook already exists in `C:/Users/Admin/Desktop/OZON-E-CUP/src/seq_cond.py`,
SHA256 `a2e499d17911d2c88a9581d54714902f50ba7b9b499c47e5a5770ab32053252e`.
`_pool` implements the exact production pooling, while `embed` runs `model.encode` under
`torch.no_grad`. Thus the representation is not ambiguous. The required historical weights
are the missing part.

## Checkpoint audit and hard block

The authoritative archive is
`C:/Users/Admin/Desktop/OZON-E-CUP/weights_archives/TCN_SEQ-01_weights.zip`, SHA256
`686932a06cbf2f44835a41eec6fff4722c58464e344583da6727d6e818c757f3`.

| current SEQ-AVG3 member | 09-04 | 09-18 | 10-02 | 10-16 |
|---|---:|---:|---:|---:|
| historical plain SEQ-01 seed 42 | missing | missing | missing | missing |
| historical plain SEQ-01 seed 43 | present | present | present | missing |
| historical plain SEQ-01 seed 44 | present | present | present | present |

The historical `exp_026_seq_seed_averaging.md` explicitly records why: the original
`SEQ-01` run did not save weights; checkpoint saving was added only afterward. The exact
historical seed-42 production checkpoint is also absent. Saved OOF/test predictions cannot
recover the pre-head embedding.

Available seed-42 files are later retrains or different encoders. They cannot be silently
substituted:

| candidate vs exact historical SEQ-01 prediction | fold | max abs log error | RMS log error |
|---|---|---:|---:|
| `SEQ-D3A-BASE-S42` | 2025-10-16 | 1.703125 | 0.164831179 |
| `SEQ-03A-BASE-S42` | 2025-10-16 | 1.703125 | 0.164831179 |
| `SEQ-01C-S42` compiled control | 2025-09-04 | 1.509765625 | 0.144327147 |

All comparisons have identical `user_id` and target vectors. Every candidate misses the
required `1e-6` parity threshold by many orders of magnitude.

### Exhaustive recovery pass over `OZON-E-CUP`

A second pass did not trust filenames. It recursively opened all nine ZIP archives,
loaded all 203 `.pt`/`.pth`/`.ckpt` files, and found 169 files compatible with the exact
plain-TCN state-key contract. Tensor-content hashing reduced those to 134 unique states;
there were no load failures. Each unique state was run through the historical SEQ-01
architecture on 256 fixed canonical rows for each OOF fold and production TEST, and its
head output was compared with the retained historical scalar prediction.

| prediction context | exact matches (`<=1e-6`) | best max abs log error |
|---|---:|---:|
| 2025-09-04 | 0 | 0.46875 |
| 2025-09-18 | 0 | 0.43359375 |
| 2025-10-02 | 0 | 0.4375 |
| 2025-10-16 | 0 | 0.50000 |
| production TEST, clip 289 | 0 | 0.40625 |

The large deliverable archive contains seed-43/44 production SEQ models and the seed-42
scalar TEST prediction, but no seed-42 model. Review-bundle and teammate-copy archives
also contain no missing state. All Git refs, reflogs and stashes were checked. A full
unreachable-object scan covered 264 blobs and 105 trees; its only three large blobs were
CSV payloads, not Torch or ZIP data. The only reachable model archives are the already
audited DETSEQ01, SEQ-01 and SEQ-D3A archives.

The machine-readable inventory and functional result are preserved in
`deep_recovery_audit.json`. Therefore the missing state is not merely misnamed or hidden
inside this repository.

## Canonical analysis and deployment inputs

- `06_ALIGNED_OOF.parquet`: 770,616 unique `(fold,user_id)` rows; all four canonical
  folds; required EXP-037/diversity columns present; finite and nonnegative; SHA256
  `7e64510b4c019ddcded125ba356517eeb2e31c32ecc11c856581f0c632c2de3f`.
- `07_ALIGNED_TEST.parquet`: 250,000 unique users; required columns present; finite and
  nonnegative; exact sample order; SHA256
  `d466eed2f649d51fc37a93e63d19f8f0c7a72d3e373d43c77085d8f55c112e2c`.
- Canonical order: `C:/Users/Admin/Desktop/OZON-E-CUP/data/raw/sample_submit.csv`, SHA256
  `06a433b0ac32f7c0292ce3cb994c1684b4156b392f30fe537ea6a44d0bc4c1b1`.
- Geometry bank: 67 vectors, two exact duplicates, 65 unique sources, documented
  difference rank 57. `Z.npz` SHA256
  `443e80472f321e796faccfe0e9ea6954653397eb74311e7abef7471ea3064dfe`.
- Exact TEST-only incumbent anchor: `SUBMIT_NEXT_BEST.csv`, SHA256
  `95f3fa982d8173e5382b199888748f57601b86fe8d0dcaa692984dde67d34677`.

## Decision

The prompt requires `TECHNICAL_BLOCK` when the exact current seed-42 checkpoint is absent.
Therefore encoder parity, pilot heads, full folds, multi-seed, TEST and submission are not
run. Retraining seed 42, using D3A, or using a later deterministic plain-SEQ retrain would
change the encoder and would answer a different hypothesis.
