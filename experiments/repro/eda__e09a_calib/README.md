# e09a_calib

## Catalogue metadata

- **Catalogue ID:** `eda__e09a_calib`
- **Namespace:** `eda`
- **Experiment ID:** `e09a_calib`
- **Original source:** `research/eda/e09a_calib.py`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** EDA experiment/script
- **Model:** LightGBM
- **Features:** gap/burst features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Unknown / not recoverable from repository history
- **Known score:** print(f"    {nm:8s} n={msk.sum():7,} ({100*msk.mean():5.1f}%)  RMSLE={np.sqrt(e2[msk].mean()):.4f}"
- **Seed:** lambda_l2=5.0, verbose=-1, seed=SEED, num_threads=10, force_row_wise=True)
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when the inputs/checkpoints named by the preserved runner are available

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# e09a_calib

Original script: `research/eda/e09a_calib.py`

```python
"""Stage 8a (lean): anchored target, log-offset calibration, error anatomy. One training matrix."""
import datetime as dt
import gc
import os
import time

import lightgbm as lgb
import numpy as np
import psutil

import fe

SEED = 42
t0 = time.time()
PROC = psutil.Process(os.getpid())


def mem(tag=""):
    print(f"    [mem {PROC.memory_info().rss/2**30:5.2f} GB] {tag}", flush=True)


fe.load()
CUTS = [dt.date(2025, 4, 15), dt.date(2025, 5, 15), dt.date(2025, 6, 15),
        dt.date(2025, 7, 15), dt.date(2025, 8, 15)]
V1 = dt.date(2025, 9, 16)
V2 = dt.date(2025, 10, 16)
EX = ["w30_gmv", "w90_gmv", "w365_gmv", "w180_days_buy", "rec_buy"]

Xtr_parts, ytr_parts, extr_parts, cut_id = [], [], [], []
val = {}
for T in CUTS + [V1, V2]:
    X, Y = fe.make_xy(T, 3)
    FEATS = [c for c in X.columns if c != "user_id"]
    A = X.select(FEATS).to_numpy().astype(np.float32)
    ex = {c: X[c].to_numpy().astype(np.float64) for c in EX}
    y = Y["y"].to_numpy()
    if T in CUTS:
        Xtr_parts.append(A); ytr_parts.append(y); extr_parts.append(ex)
        cut_id.append(np.full(len(y), CUTS.index(T)))
    else:
        val[T] = (A, y, ex)
    del X, Y
    gc.collect()
    print(f"built {T} n={len(y):,}", flush=True)
    mem()

Xtr = np.vstack(Xtr_parts)
ytr = np.concatenate(ytr_parts)
cid = np.concatenate(cut_id)
extr = {c: np.concatenate([e[c] for e in extr_parts]) for c in EX}
del Xtr_parts, ytr_parts, extr_parts
gc.collect()
mem(f"train matrix {Xtr.shape}")

P = dict(objective="regression", metric="rmse", learning_rate=0.05, num_leaves=127,
         min_data_in_leaf=200, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
         lambda_l2=5.0, verbose=-1, seed=SEED, num_threads=10, force_row_wise=True)
ROUNDS = 700
DTR = lgb.Dataset(Xtr, np.zeros(len(ytr)), free_raw_data=False)
DTR.construct()
mem("lgb dataset constructed")


def fit(label, sub=None):
    d = DTR.subset(np.where(sub)[0]) if sub is not None else DTR
    d = d.set_label(label if sub is None else label[sub])
    m = lgb.train(P, d, num_boost_round=ROUNDS)
    gc.collect()
    return m


def rm(y, z):
    return float(np.sqrt(np.mean((np.log1p(y) - z) ** 2)))


print("\n" + "=" * 100)
print("EXP 7: ANCHORED (delta) TARGET vs raw level")
print("=" * 100, flush=True)
base = {}
for a in [None, "w30_gmv", "w90_gmv", "w365_gmv"]:
    off = np.log1p(extr[a]) if a else 0.0
    m = fit(np.log1p(ytr) - off)
    out = []
    for VV in [V1, V2]:
        A, y, ex = val[VV]
        z = m.predict(A) + (np.log1p(ex[a]) if a else 0.0)
        out.append((rm(y, z), float(np.log1p(y).mean() - z.mean())))
        if a is None:
            base[VV] = (z, y, ex)
    print(f"  anchor={str(a):10s}  V1 RMSLE={out[0][0]:.5f} bias={out[0][1]:+.4f}  |  "
          f"V2 RMSLE={out[1][0]:.5f} bias={out[1][1]:+.4f}", flush=True)
    del m
    gc.collect()
    mem()

print("\n" + "=" * 100)
print("EXP 8: GLOBAL LOG-OFFSET — gain and stability")
print("=" * 100, flush=True)
grid = np.linspace(-0.4, 0.4, 161)
for VV in [V1, V2]:
    z, y, _ = base[VV]
    ly = np.log1p(y)
    sc = np.array([np.sqrt(np.mean((ly - (z + d)) ** 2)) for d in grid])
    i = int(sc.argmin())
    print(f"  {VV}: RMSLE={rm(y,z):.5f}  best_offset={grid[i]:+.4f} -> {sc[i]:.5f} "
          f"(gain {rm(y,z)-sc[i]:+.5f})  mean_bias={ly.mean()-z.mean():+.4f}", flush=True)

print("\n  --- expanding window: offset vs train->val gap ---", flush=True)
for VV, kmax in [(dt.date(2025, 6, 15), 1), (dt.date(2025, 7, 15), 2),
                 (dt.date(2025, 8, 15), 3), (V1, 5), (V2, 5)]:
    sub = cid <= kmax - 1
    if sub.sum() == 0:
        continue
    m = fit(np.log1p(ytr), sub)
    if VV in val:
        A, y, _ = val[VV]
    else:
        j = CUTS.index(VV)
        A, y = Xtr[cid == j], ytr[cid == j]
    z = m.predict(A)
    ly = np.log1p(y)
    off = float(ly.mean() - z.mean())
    print(f"    train cutoffs 0..{kmax-1} -> val {VV}: RMSLE={rm(y,z):.5f} offset={off:+.4f} "
          f"corrected={float(np.sqrt(np.mean((ly-(z+off))**2))):.5f}", flush=True)
    del m
    gc.collect()

print("\n" + "=" * 100)
print("EXP 10: ERROR ANATOMY on V1 + segment-specific offsets")
print("=" * 100, flush=True)
z, y, ex = base[V1]
ly = np.log1p(y)
e2 = (ly - z) ** 2
print(f"  total RMSLE={np.sqrt(e2.mean()):.5f}")
for nm, msk in [("y == 0", y == 0), ("y > 0", y > 0)]:
    print(f"    {nm:8s} n={msk.sum():7,} ({100*msk.mean():5.1f}%)  RMSLE={np.sqrt(e2[msk].mean()):.4f}"
          f"  share_MSE={100*e2[msk].sum()/e2.sum():5.1f}%")
nb = np.nan_to_num(ex["w180_days_buy"], nan=-1)
print("\n  by w180_days_buy:")
for lo, hi in [(0, 0), (1, 1), (2, 3), (4, 7), (8, 15), (16, 10 ** 6)]:
    m_ = (nb >= lo) & (nb <= hi)
    if m_.sum() == 0:
        continue
    sc = np.array([np.sqrt(np.mean((ly[m_] - (z[m_] + d)) ** 2)) for d in grid])
    i = int(sc.argmin())
    print(f"    buydays180 {lo:3d}-{hi:<7d} n={m_.sum():7,} ({100*m_.mean():5.1f}%) "
          f"RMSLE={np.sqrt(e2[m_].mean()):.4f} shareMSE={100*e2[m_].sum()/e2.sum():5.1f}% "
          f"P(y>0)={float((y[m_]>0).mean()):.3f} best_off={grid[i]:+.3f}->{sc[i]:.4f}")
rb = ex["rec_buy"]
print("\n  by rec_buy:")
for lo, hi, nm in [(0, 7, "0-7"), (8, 30, "8-30"), (31, 90, "31-90"),
                   (91, 180, "91-180"), (181, 10 ** 6, "181+")]:
    m_ = (~np.isnan(rb)) & (rb >= lo) & (rb <= hi)
    if m_.sum() == 0:
        continue
    print(f"    rec_buy {nm:8s} n={m_.sum():7,} ({100*m_.mean():5.1f}%) "
          f"RMSLE={np.sqrt(e2[m_].mean()):.4f} shareMSE={100*e2[m_].sum()/e2.sum():5.1f}%")
m_ = np.isnan(rb)
print(f"    never bought    n={m_.sum():7,} ({100*m_.mean():5.1f}%) "
      f"RMSLE={np.sqrt(e2[m_].mean()):.4f} shareMSE={100*e2[m_].sum()/e2.sum():5.1f}%")
print(f"\ntotal {time.time()-t0:.0f}s", flush=True)

```
