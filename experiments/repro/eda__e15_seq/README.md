# e15_seq

## Catalogue metadata

- **Catalogue ID:** `eda__e15_seq`
- **Namespace:** `eda`
- **Experiment ID:** `e15_seq`
- **Original source:** `research/eda/e15_seq.py`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** EDA experiment/script
- **Model:** LightGBM, blend
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Unknown / not recoverable from repository history
- **Known score:** Unknown / not recoverable from repository history
- **Seed:** lambda_l2=5.0, verbose=-1, seed=SEED, num_threads=10, max_bin=63, force_row_wise=True)
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when the inputs/checkpoints named by the preserved runner are available

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# e15_seq

Original script: `research/eda/e15_seq.py`

```python
"""Stage 14: sequence probe (fixed #weeks) + structural model + 3-way diversity/blending."""
import datetime as dt
import gc
import math
import os
import time

import lightgbm as lgb
import numpy as np
import polars as pl
import psutil

import fe

SEED = 42
NW = 20                       # 140 days of weekly history, available at every cutoff
NMAX, QN = 30, 11
t0 = time.time()
PR = psutil.Process(os.getpid())
_gx, _gw = np.polynomial.hermite_e.hermegauss(QN)
_gw = _gw / _gw.sum()
_LG = np.array([math.lgamma(n + 1) for n in range(1, NMAX + 1)])


def log(*a):
    print(f"[{time.time()-t0:6.0f}s {PR.memory_info().rss/2**30:4.2f}GB]", *a, flush=True)


def weekly(T, users, nw=NW):
    df = (fe.load().lazy()
          .filter((pl.col("event_date") <= T) & (pl.col("event_date") > T - dt.timedelta(days=7 * nw)))
          .with_columns(wk=((pl.lit(T) - pl.col("event_date")).dt.total_days() // 7).cast(pl.Int32))
          .group_by(["user_id", "wk"])
          .agg(g=pl.col("gmv").sum(), a=pl.len(), o=pl.col("to_ord").sum(),
               s=pl.col("searches").sum()).collect())
    out = np.zeros((users.height, 4 * nw), dtype=np.float32)
    pos = {u: i for i, u in enumerate(users["user_id"].to_list())}
    ui = np.array([pos.get(u, -1) for u in df["user_id"].to_list()])
    wk = df["wk"].to_numpy()
    ok = (ui >= 0) & (wk >= 0) & (wk < nw)
    for j, c in enumerate(["g", "a", "o", "s"]):
        out[ui[ok], j * nw + wk[ok]] = np.log1p(df[c].to_numpy()[ok])
    return out


def value_stats(T, users):
    h = (fe.load().lazy().filter((pl.col("event_date") <= T) & (pl.col("gmv") > 0))
         .group_by("user_id").agg(k=pl.len(), s=pl.col("gmv").log().sum(),
                                  ss=(pl.col("gmv").log() ** 2).sum()).collect())
    d = (users.join(h, on="user_id", how="left")
         .with_columns(pl.col("k").fill_null(0), pl.col("s").fill_null(0.0),
                       pl.col("ss").fill_null(0.0)).sort("user_id"))
    k = d["k"].to_numpy().astype(float); s = d["s"].to_numpy(); ss = d["ss"].to_numpy()
    mpop = s.sum() / max(k.sum(), 1.0)
    vpop = max(ss.sum() / max(k.sum(), 1.0) - mpop ** 2, 1e-3)
    K = 3.0
    mu = (s + K * mpop) / (k + K)
    mk = np.where(k > 0, s / np.maximum(k, 1), 0.0)
    within = np.where(k >= 2, np.maximum(ss - k * mk ** 2, 0) / np.maximum(k - 1, 1), vpop)
    return mu, np.sqrt(np.maximum((k * within + K * vpop) / (k + K), 1e-3))


def n_target(T, users, h=30):
    a, b = T + dt.timedelta(days=1), T + dt.timedelta(days=h)
    n = (fe.load().lazy().filter((pl.col("event_date") >= a) & (pl.col("event_date") <= b)
                                 & (pl.col("gmv") > 0))
         .group_by("user_id").agg(pl.len().alias("n")).collect())
    return (users.join(n, on="user_id", how="left")
            .with_columns(pl.col("n").fill_null(0)).sort("user_id")["n"].to_numpy())


def expected_log1p(lam, mu, sig):
    ns = np.arange(1, NMAX + 1, dtype=float)
    out = np.zeros(len(lam)); step = 40000
    for i in range(0, len(lam), step):
        l = lam[i:i + step, None]; m = mu[i:i + step, None]; s2 = sig[i:i + step, None] ** 2
        pmf = np.exp(-l + np.log(np.maximum(l, 1e-12)) * ns[None, :] - _LG[None, :])
        sS2 = np.log1p((np.exp(np.minimum(s2, 20)) - 1.0) / ns[None, :])
        muS = np.log(ns)[None, :] + m + s2 / 2.0 - sS2 / 2.0
        acc = np.zeros(len(l))
        for q in range(QN):
            acc += (pmf * np.log1p(np.exp(np.clip(muS + np.sqrt(sS2) * _gx[q], -30, 30)))).sum(1) * _gw[q]
        out[i:i + step] = acc
    return out


fe.load()
CUTS = [dt.date(2025, 6, 15), dt.date(2025, 7, 15), dt.date(2025, 8, 15)]
V = dt.date(2025, 9, 16)
D = {}
for T in CUTS + [V]:
    u = fe.panel_users(T, 3)
    X = fe.build_features(T, u)
    FE = [c for c in X.columns if c != "user_id"]
    mu, sg = value_stats(T, u)
    D[T] = dict(A=X.select(FE).to_numpy().astype(np.float32), S=weekly(T, u),
                y=fe.target(T, u)["y"].to_numpy(), n=n_target(T, u), mu=mu, sg=sg)
    del X, u
    gc.collect()
    log(f"built {T} n={len(D[T]['y']):,} agg={len(FE)} seq={D[T]['S'].shape[1]}")
fe._CACHE.clear()
gc.collect()

P = dict(objective="regression", metric="rmse", learning_rate=0.05, num_leaves=127,
         min_data_in_leaf=200, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
         lambda_l2=5.0, verbose=-1, seed=SEED, num_threads=10, max_bin=63, force_row_wise=True)
R = 600
ytr = np.concatenate([D[T]["y"] for T in CUTS])
ntr = np.concatenate([D[T]["n"] for T in CUTS])
yv = D[V]["y"]; lyv = np.log1p(yv)


def rm(z):
    return float(np.sqrt(np.mean((lyv - z) ** 2)))


print("\n" + "=" * 96)
print(f"A. SEQUENCE PROBE: aggregates vs + {4*NW} weekly log-features vs weekly ONLY")
print("=" * 96, flush=True)
Atr = np.vstack([D[T]["A"] for T in CUTS])
m0 = lgb.train(P, lgb.Dataset(Atr, np.log1p(ytr)), num_boost_round=R)
z_agg = m0.predict(D[V]["A"])
print(f"  aggregates only        RMSLE={rm(z_agg):.5f}  mean(z)={z_agg.mean():.4f}", flush=True)

Str = np.vstack([D[T]["S"] for T in CUTS])
m1 = lgb.train(P, lgb.Dataset(np.hstack([Atr, Str]), np.log1p(ytr)), num_boost_round=R)
z_both = m1.predict(np.hstack([D[V]["A"], D[V]["S"]]))
print(f"  aggregates + weekly    RMSLE={rm(z_both):.5f}  delta={rm(z_both)-rm(z_agg):+.5f}", flush=True)

m2 = lgb.train(P, lgb.Dataset(Str, np.log1p(ytr)), num_boost_round=R)
z_seq = m2.predict(D[V]["S"])
print(f"  weekly ONLY            RMSLE={rm(z_seq):.5f}  (zero hand-made aggregates)", flush=True)
del Atr, Str
gc.collect()

print("\n" + "=" * 96)
print("B. STRUCTURAL: Poisson(buy-days) x lognormal(value) -> exact E[log1p(S)]")
print("=" * 96, flush=True)
mc = lgb.train(dict(P, objective="poisson", metric="poisson"),
               lgb.Dataset(np.vstack([D[T]["A"] for T in CUTS]), ntr), num_boost_round=R)
lam = np.maximum(mc.predict(D[V]["A"]), 1e-6)
mu, sg = D[V]["mu"], D[V]["sg"]
print(f"  lambda mean={lam.mean():.4f}  true={D[V]['n'].mean():.4f}")
z_raw = expected_log1p(lam, mu, sg)
print(f"  raw           RMSLE={rm(z_raw):.5f}  mean(z)={z_raw.mean():.4f}  true={lyv.mean():.4f}")
best = (9.9, 1.0, 0.0)
for fs in [0.7, 0.8, 0.9, 1.0]:
    for dm in [-0.4, -0.3, -0.2, -0.1, 0.0]:
        s = rm(expected_log1p(lam, mu + dm, sg * fs))
        if s < best[0]:
            best = (s, fs, dm)
z_str = expected_log1p(lam, mu + best[2], sg * best[1])
print(f"  calibrated (sigma x{best[1]}, mu{best[2]:+.2f})  RMSLE={best[0]:.5f}  "
      f"mean(z)={z_str.mean():.4f}", flush=True)

print("\n" + "=" * 96)
print("C. DIVERSITY AND BLENDING (all in log space)")
print("=" * 96)
names = {"agg": z_agg, "seq_only": z_seq, "agg+seq": z_both, "structural": z_str}
print("  residual correlations:")
ks = list(names)
for i in range(len(ks)):
    for j in range(i + 1, len(ks)):
        a, b = names[ks[i]], names[ks[j]]
        print(f"    {ks[i]:11s} vs {ks[j]:11s}  pred={np.corrcoef(a,b)[0,1]:.4f}  "
              f"resid={np.corrcoef(lyv-a, lyv-b)[0,1]:.4f}")
print("\n  pairwise blends with 'agg':")
for nm in ["seq_only", "agg+seq", "structural"]:
    bb = min(((rm((1 - w) * z_agg + w * names[nm]), w) for w in np.arange(0, 1.01, 0.05)))
    print(f"    agg + {nm:11s}: best w={bb[1]:.2f} -> RMSLE={bb[0]:.5f} "
          f"(agg alone {rm(z_agg):.5f}, gain {rm(z_agg)-bb[0]:+.5f})")
best3 = min(((rm(a * z_agg + b * z_both + c * z_str), a, b, c)
             for a in np.arange(0, 1.01, 0.1) for b in np.arange(0, 1.01 - a, 0.1)
             for c in [round(1 - a - b, 2)]))
print(f"\n  best 3-way agg/agg+seq/structural = {best3[1]:.1f}/{best3[2]:.1f}/{best3[3]:.1f}"
      f" -> RMSLE={best3[0]:.5f}  (gain vs agg alone {rm(z_agg)-best3[0]:+.5f})")
log("done")

```
