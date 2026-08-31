# e13_capped

## Catalogue metadata

- **Catalogue ID:** `eda__e13_capped`
- **Namespace:** `eda`
- **Experiment ID:** `e13_capped`
- **Original source:** `research/eda/e13_capped.py`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** EDA experiment/script
- **Model:** LightGBM
- **Features:** gap/burst features, history-depth features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** print("ADVERSARIAL VALIDATION WITH CAPPED FEATURES")
- **Known score:** Unknown / not recoverable from repository history
- **Seed:** idx = np.random.RandomState(SEED).permutation(len(X))
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when the inputs/checkpoints named by the preserved runner are available

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# e13_capped

Original script: `research/eda/e13_capped.py`

```python
"""Stage 12: history-depth artefact.

At cutoff T the available history is (T - 2025-01-01) days: 89d at the earliest clean
cutoff, 409d at the test cutoff. So `tenure`, `all_*`, `w365_*` mean different things at
different cutoffs -> extrapolation failure. Fix: truncate every cutoff to a FIXED window L
(including the test) and drop unbounded features.
"""
import datetime as dt
import gc
import os
import time

import lightgbm as lgb
import numpy as np
import polars as pl
import psutil
from sklearn.metrics import roc_auc_score

import fe

SEED = 42
t0 = time.time()
PR = psutil.Process(os.getpid())
L = 180                    # fixed history depth used at EVERY cutoff incl. test


def log(*a):
    print(f"[{time.time()-t0:6.0f}s {PR.memory_info().rss/2**30:4.2f}GB]", *a, flush=True)


_full = None


def capped_features(T, users, L=L):
    """build_features on a df truncated to (T-L, T]; only windows <= L survive."""
    global _full
    if _full is None:
        _full = fe._CACHE["df"]
    fe._CACHE["df"] = _full.filter(pl.col("event_date") > T - dt.timedelta(days=L))
    try:
        X = fe.build_features(T, users)
    finally:
        fe._CACHE["df"] = _full
    drop = [c for c in X.columns if c.startswith("w365") or c.startswith("all_")
            or c in ("tenure", "first_buy_age")
            or "_365" in c or "lifetime_gmv_per_day" in c]
    return X.drop(drop)


fe.load()
CUTS = [dt.date(2025, 6, 15), dt.date(2025, 7, 15), dt.date(2025, 8, 15)]
V1, V2, TEST = dt.date(2025, 9, 16), dt.date(2025, 10, 16), dt.date(2026, 2, 13)

parts, ys, val = [], [], {}
FE = None
for T in CUTS + [V1, V2, TEST]:
    u = fe.panel_users(T, 3)
    X = capped_features(T, u)
    if FE is None:
        FE = [c for c in X.columns if c != "user_id"]
    A = X.select(FE).to_numpy().astype(np.float32)
    if T == TEST:
        val[T] = (A, None)
    else:
        y = fe.target(T, u)["y"].to_numpy()
        (parts.append(A) or ys.append(y)) if T in CUTS else val.setdefault(T, (A, y))
    del X, u
    gc.collect()
    log(f"built {T} n={A.shape[0]:,} feats={len(FE)}")

Xtr = np.vstack(parts); ytr = np.concatenate(ys)
del parts, ys
gc.collect()

P = dict(objective="regression", metric="rmse", learning_rate=0.05, num_leaves=127,
         min_data_in_leaf=200, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
         lambda_l2=5.0, verbose=-1, seed=SEED, num_threads=10, max_bin=63, force_row_wise=True)
m = lgb.train(P, lgb.Dataset(Xtr, np.log1p(ytr)), num_boost_round=600)
log("trained")
del Xtr, ytr
gc.collect()


def rm(y, z):
    return float(np.sqrt(np.mean((np.log1p(y) - z) ** 2)))


print("\n" + "=" * 100)
print(f"CAPPED FEATURES (history truncated to last {L}d at every cutoff, no unbounded features)")
print("=" * 100)
grid = np.linspace(-0.5, 0.5, 201)
for VV in [V1, V2]:
    A, y = val[VV]
    z = m.predict(A)
    ly = np.log1p(y)
    sc = np.array([np.sqrt(np.mean((ly - (z + d)) ** 2)) for d in grid])
    i = int(sc.argmin())
    print(f"  val {VV} (gap {(VV-max(CUTS)).days:3d}d): RMSLE={rm(y,z):.5f}  "
          f"mean_bias={ly.mean()-z.mean():+.4f}  best_off={grid[i]:+.4f} -> {sc[i]:.5f}")
A, _ = val[TEST]
zt = m.predict(A)
print(f"\n  mean(log1p(pred)) on TEST = {zt.mean():.4f}")
print("  anchor band 2.2791 (median drift) .. 2.4080 (YoY analogue);  UNCAPPED model gave 2.6465")
print(f"  => capped model is {2.2791-zt.mean():+.4f} .. {2.4080-zt.mean():+.4f} from the band")

print("\n" + "=" * 100)
print("ADVERSARIAL VALIDATION WITH CAPPED FEATURES")
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
    imp = sorted(zip(FE, mm.feature_importance("gain")), key=lambda t: -t[1])[:8]
    print(f"  {nm}: AUC={auc:.4f}   (uncapped был 0.9930 / 0.9912 / 0.9811)")
    print("    " + ", ".join(n for n, _ in imp))
    del X, yy, mm
    gc.collect()


adv(val[V1][0], val[TEST][0], f"{V1} vs TEST")
adv(val[V2][0], val[TEST][0], f"{V2} vs TEST")
adv(val[V1][0], val[V2][0], f"{V1} vs {V2}")

print("\n" + "=" * 100)
print("FEATURE SHIFT AFTER CAPPING")
print("=" * 100)
iv = {c: i for i, c in enumerate(FE)}
for c in ["w30_gmv", "w90_gmv", "w180_gmv", "w30_days_buy", "w180_days_buy", "rec_buy",
          "rec_any", "w30_ponly_share", "trend_gmv_30_90", "gap_mean"]:
    if c not in iv:
        continue
    a = float(np.nanmean(val[V2][0][:, iv[c]])); b = float(np.nanmean(val[TEST][0][:, iv[c]]))
    print(f"  {c:22s} V2={a:11.4f}  TEST={b:11.4f}  ratio={b/(a+1e-9):6.3f}")
log("done")

```
