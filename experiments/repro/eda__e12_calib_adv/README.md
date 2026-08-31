# e12_calib_adv

## Catalogue metadata

- **Catalogue ID:** `eda__e12_calib_adv`
- **Namespace:** `eda`
- **Experiment ID:** `e12_calib_adv`
- **Original source:** `research/eda/e12_calib_adv.py`
- **Source ref:** `working tree`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** EDA experiment/script
- **Model:** LightGBM
- **Features:** gap/burst features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** """Stage 11 (memory-lean): offset calibration, error anatomy, adversarial validation, test level."""
- **Known score:** print(f"    {nm:8s} n={msk.sum():7,} ({100*msk.mean():5.1f}%) RMSLE={np.sqrt(e2[msk].mean()):.4f}"
- **Seed:** idx = np.random.RandomState(SEED).permutation(len(X))
- **Postprocessing:** print("E. FEATURE-LEVEL SHIFT clean-val -> TEST")
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when the inputs/checkpoints named by the preserved runner are available

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# e12_calib_adv

Original script: `research/eda/e12_calib_adv.py`

```python
"""Stage 11 (memory-lean): offset calibration, error anatomy, adversarial validation, test level."""
import datetime as dt
import gc
import os
import time

import lightgbm as lgb
import numpy as np
import psutil
from sklearn.metrics import roc_auc_score

import fe

SEED = 42
t0 = time.time()
PR = psutil.Process(os.getpid())


def log(*a):
    print(f"[{time.time()-t0:6.0f}s {PR.memory_info().rss/2**30:4.2f}GB]", *a, flush=True)


CUTS = [dt.date(2025, 6, 15), dt.date(2025, 7, 15), dt.date(2025, 8, 15)]
V1 = dt.date(2025, 9, 16)
V2 = dt.date(2025, 10, 16)
TEST = dt.date(2026, 2, 13)
EX = ["w30_gmv", "w365_gmv", "w180_days_buy", "rec_buy", "w30_days_present", "tenure"]

fe.load()
parts, ys, val, exs = [], [], {}, []
FEATS = None
for T in CUTS + [V1, V2, TEST]:
    u = fe.panel_users(T, 3)
    X = fe.build_features(T, u)
    if FEATS is None:
        FEATS = [c for c in X.columns if c != "user_id"]
    A = X.select(FEATS).to_numpy().astype(np.float32)
    ex = {c: X[c].to_numpy().astype(np.float64) for c in EX}
    if T == TEST:
        val[T] = (A, None, ex)
    else:
        y = fe.target(T, u)["y"].to_numpy()
        if T in CUTS:
            parts.append(A); ys.append(y); exs.append(ex)
        else:
            val[T] = (A, y, ex)
    del X, u
    gc.collect()
    log(f"built {T} n={A.shape[0]:,}")

Xtr = np.vstack(parts); ytr = np.concatenate(ys)
del parts, ys, exs
gc.collect()
log(f"train {Xtr.shape}")

P = dict(objective="regression", metric="rmse", learning_rate=0.05, num_leaves=127,
         min_data_in_leaf=200, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
         lambda_l2=5.0, verbose=-1, seed=SEED, num_threads=10, max_bin=63, force_row_wise=True)
m = lgb.train(P, lgb.Dataset(Xtr, np.log1p(ytr)), num_boost_round=600)
log("model trained")
del Xtr, ytr
gc.collect()


def rm(y, z):
    return float(np.sqrt(np.mean((np.log1p(y) - z) ** 2)))


grid = np.linspace(-0.5, 0.5, 201)
print("\n" + "=" * 100)
print("A. OUT-OF-TIME BIAS AND OPTIMAL LOG-OFFSET (train = Jun/Jul/Aug 2025)")
print("=" * 100)
preds = {}
for VV in [V1, V2]:
    A, y, ex = val[VV]
    z = m.predict(A)
    preds[VV] = (z, y, ex)
    ly = np.log1p(y)
    sc = np.array([np.sqrt(np.mean((ly - (z + d)) ** 2)) for d in grid])
    i = int(sc.argmin())
    print(f"  val {VV} (gap {(VV-max(CUTS)).days:3d}d): RMSLE={rm(y,z):.5f}  mean_bias={ly.mean()-z.mean():+.4f}"
          f"  best_offset={grid[i]:+.4f} -> {sc[i]:.5f}  gain={rm(y,z)-sc[i]:+.5f}")
    print(f"      mean log1p: pred={z.mean():.4f}  true={ly.mean():.4f}   "
          f"P(y>0)={float((y>0).mean()):.4f}")

print("\n" + "=" * 100)
print("B. TEST-CUTOFF PREDICTION LEVEL vs the anchors from sample_submit")
print("=" * 100)
A, _, ex = val[TEST]
zt = m.predict(A)
print(f"  test panel n={A.shape[0]:,}")
print(f"  mean(log1p(pred)) on TEST      = {zt.mean():.4f}")
print(f"  trailing-30d level m_x         = 2.2421   (sample_submit)")
print("  anchor scenarios for E[log1p(y_test)]: 2.3308 (typical drift) .. 2.4180 (YoY analogue)")
print(f"  => raw model is {2.3308-zt.mean():+.4f} .. {2.4180-zt.mean():+.4f} away from the anchor band")
print(f"  implied share of zero-ish preds: P(pred<0.5)={float((np.expm1(zt)<0.5).mean()):.4f}")

print("\n" + "=" * 100)
print("C. ERROR ANATOMY on V1")
print("=" * 100)
z, y, ex = preds[V1]
ly = np.log1p(y); e2 = (ly - z) ** 2
print(f"  total RMSLE={np.sqrt(e2.mean()):.5f}")
for nm, msk in [("y == 0", y == 0), ("y > 0", y > 0)]:
    print(f"    {nm:8s} n={msk.sum():7,} ({100*msk.mean():5.1f}%) RMSLE={np.sqrt(e2[msk].mean()):.4f}"
          f" shareMSE={100*e2[msk].sum()/e2.sum():5.1f}%")
nb = np.nan_to_num(ex["w180_days_buy"], nan=-1)
print("\n  by w180_days_buy (history purchase frequency):")
for lo, hi in [(0, 0), (1, 1), (2, 3), (4, 7), (8, 15), (16, 10 ** 6)]:
    k = (nb >= lo) & (nb <= hi)
    if k.sum() == 0:
        continue
    sc = np.array([np.sqrt(np.mean((ly[k] - (z[k] + d)) ** 2)) for d in grid])
    i = int(sc.argmin())
    print(f"    {lo:3d}-{hi:<7d} n={k.sum():7,} ({100*k.mean():5.1f}%) RMSLE={np.sqrt(e2[k].mean()):.4f}"
          f" shareMSE={100*e2[k].sum()/e2.sum():5.1f}% P(y>0)={float((y[k]>0).mean()):.3f}"
          f" best_off={grid[i]:+.3f}->{sc[i]:.4f}")
kk = np.isnan(ex["rec_buy"])
print(f"\n    never bought      n={kk.sum():7,} ({100*kk.mean():5.1f}%) RMSLE={np.sqrt(e2[kk].mean()):.4f}"
      f" shareMSE={100*e2[kk].sum()/e2.sum():5.1f}%")

print("\n" + "=" * 100)
print("D. ADVERSARIAL VALIDATION")
print("=" * 100)
PA = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=63,
          min_data_in_leaf=200, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
          verbose=-1, seed=SEED, num_threads=10, max_bin=63, force_row_wise=True)


def adv(Aa, Ab, nm):
    X = np.vstack([Aa, Ab]); yy = np.r_[np.zeros(len(Aa)), np.ones(len(Ab))]
    idx = np.random.RandomState(SEED).permutation(len(X))
    X, yy = X[idx], yy[idx]
    c = int(0.7 * len(X))
    mm = lgb.train(PA, lgb.Dataset(X[:c], yy[:c]), num_boost_round=250)
    auc = roc_auc_score(yy[c:], mm.predict(X[c:]))
    imp = sorted(zip(FEATS, mm.feature_importance("gain")), key=lambda t: -t[1])[:10]
    print(f"  {nm}: AUC={auc:.4f}")
    print("    " + ", ".join(f"{n}" for n, _ in imp))
    del X, yy, mm
    gc.collect()
    return auc


adv(val[V1][0], val[TEST][0], f"{V1} vs TEST")
adv(val[V2][0], val[TEST][0], f"{V2} vs TEST")
adv(val[V1][0], val[V2][0], f"{V1} vs {V2} (two historical)")

print("\n" + "=" * 100)
print("E. FEATURE-LEVEL SHIFT clean-val -> TEST")
print("=" * 100)
iv = {c: i for i, c in enumerate(FEATS)}
watch = ["w30_gmv", "w90_gmv", "w365_gmv", "w30_days_present", "w30_days_buy", "w180_days_buy",
         "w30_searches", "w30_orders", "tenure", "rec_buy", "rec_any", "w30_ponly_share",
         "w30_presence_rate", "lifetime_buyrate", "trend_gmv_30_90"]
print(f"  {'feature':26s} {'V2 mean':>12s} {'TEST mean':>12s} {'ratio':>8s}")
for c in watch:
    a = float(np.nanmean(val[V2][0][:, iv[c]])); b = float(np.nanmean(val[TEST][0][:, iv[c]]))
    print(f"  {c:26s} {a:12.4f} {b:12.4f} {b/(a+1e-9):8.3f}")
log("done")

```
