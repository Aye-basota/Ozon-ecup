# e14_struct_seq

## Catalogue metadata

- **Catalogue ID:** `eda__e14_struct_seq`
- **Namespace:** `eda`
- **Experiment ID:** `e14_struct_seq`
- **Original source:** `research/eda/e14_struct_seq.py`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** EDA experiment/script
- **Model:** LightGBM, blend
- **Features:** See preserved experiment card and implementation
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Unknown / not recoverable from repository history
- **Known score:** Unknown / not recoverable from repository history
- **Seed:** bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, seed=SEED,
- **Postprocessing:** None documented
- **Submission:** None documented
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when the inputs/checkpoints named by the preserved runner are available

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# e14_struct_seq

Original script: `research/eda/e14_struct_seq.py`

```python
"""Stage 13: (A) structural generative model vs direct GBM, (B) weekly-sequence probe."""
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
t0 = time.time()
PR = psutil.Process(os.getpid())


def log(*a):
    print(f"[{time.time()-t0:6.0f}s {PR.memory_info().rss/2**30:4.2f}GB]", *a, flush=True)


CUTS = [dt.date(2025, 6, 15), dt.date(2025, 7, 15), dt.date(2025, 8, 15)]
V = dt.date(2025, 9, 16)
NMAX, QN = 30, 11
_gx, _gw = np.polynomial.hermite_e.hermegauss(QN)
_gw = _gw / _gw.sum()
_LG = np.array([math.lgamma(n + 1) for n in range(1, NMAX + 1)])


def value_stats(T, users):
    """Empirical-Bayes mu_i, sigma_i for log(daily gmv | buy), history <= T only."""
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
    sig = np.sqrt(np.maximum((k * within + K * vpop) / (k + K), 1e-3))
    return mu, sig, mpop, math.sqrt(vpop)


def n_target(T, users, h=30):
    a, b = T + dt.timedelta(days=1), T + dt.timedelta(days=h)
    n = (fe.load().lazy().filter((pl.col("event_date") >= a) & (pl.col("event_date") <= b)
                                 & (pl.col("gmv") > 0))
         .group_by("user_id").agg(pl.len().alias("n")).collect())
    return (users.join(n, on="user_id", how="left")
            .with_columns(pl.col("n").fill_null(0)).sort("user_id")["n"].to_numpy())


def expected_log1p(lam, mu, sig):
    """E[log1p(S)],  S|n ~ Fenton-Wilkinson lognormal,  n ~ Poisson(lam)."""
    ns = np.arange(1, NMAX + 1, dtype=float)
    out = np.zeros(len(lam))
    step = 40000
    for i in range(0, len(lam), step):
        l = lam[i:i + step, None]; m = mu[i:i + step, None]; s2 = sig[i:i + step, None] ** 2
        pmf = np.exp(-l + np.log(np.maximum(l, 1e-12)) * ns[None, :] - _LG[None, :])
        sS2 = np.log1p((np.exp(np.minimum(s2, 20)) - 1.0) / ns[None, :])
        muS = np.log(ns)[None, :] + m + s2 / 2.0 - sS2 / 2.0
        sS = np.sqrt(sS2)
        acc = np.zeros(len(l))
        for q in range(QN):
            acc += (pmf * np.log1p(np.exp(np.clip(muS + sS * _gx[q], -30, 30)))).sum(1) * _gw[q]
        out[i:i + step] = acc
    return out


def weekly(T, users, nw=26):
    df = (fe.load().lazy()
          .filter((pl.col("event_date") <= T) & (pl.col("event_date") > T - dt.timedelta(days=7 * nw)))
          .with_columns(wk=((pl.lit(T) - pl.col("event_date")).dt.total_days() // 7).cast(pl.Int32))
          .group_by(["user_id", "wk"])
          .agg(g=pl.col("gmv").sum(), a=pl.len(), o=pl.col("to_ord").sum(),
               s=pl.col("searches").sum()).collect())
    out = users.clone()
    for nm in ["g", "a", "o", "s"]:
        p = df.pivot(values=nm, index="user_id", on="wk", aggregate_function="first")
        p.columns = ["user_id"] + [f"{nm}w{c}" for c in p.columns[1:]]
        out = out.join(p, on="user_id", how="left")
    out = out.fill_null(0).sort("user_id")
    for nm in ["g", "o", "s"]:
        out = out.with_columns([pl.col(c).log1p().alias(c) for c in out.columns if c.startswith(nm + "w")])
    return out


fe.load()
D = {}
for T in CUTS + [V]:
    u = fe.panel_users(T, 3)
    X = fe.build_features(T, u)
    FE = [c for c in X.columns if c != "user_id"]
    W = weekly(T, u)
    WF = [c for c in W.columns if c != "user_id"]
    mu, sg, mp, sp = value_stats(T, u)
    D[T] = dict(A=X.select(FE).to_numpy().astype(np.float32),
                S=W.select(WF).to_numpy().astype(np.float32),
                y=fe.target(T, u)["y"].to_numpy(), n=n_target(T, u), mu=mu, sg=sg)
    del X, W, u
    gc.collect()
    log(f"built {T} n={len(D[T]['y']):,} agg={len(FE)} seq={len(WF)} mu_pop={mp:.3f} sig_pop={sp:.3f}")

fe._CACHE.clear()          # raw 30M-row frame no longer needed
gc.collect()
log("raw frame released")

P = dict(learning_rate=0.05, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.7,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, seed=SEED,
         num_threads=10, max_bin=63, force_row_wise=True)
R = 600
Atr = np.vstack([D[T]["A"] for T in CUTS])
ytr = np.concatenate([D[T]["y"] for T in CUTS])
ntr = np.concatenate([D[T]["n"] for T in CUTS])
yv = D[V]["y"]; lyv = np.log1p(yv)


def rm(z):
    return float(np.sqrt(np.mean((lyv - z) ** 2)))


print("\n" + "=" * 96)
print("A. DIRECT GBM (reference)")
print("=" * 96, flush=True)
m_dir = lgb.train(dict(P, objective="regression", metric="rmse"),
                  lgb.Dataset(Atr, np.log1p(ytr)), num_boost_round=R)
z_dir = m_dir.predict(D[V]["A"])
print(f"  direct RMSLE={rm(z_dir):.5f}   mean(z)={z_dir.mean():.4f} true={lyv.mean():.4f}")
log("direct done")

print("\n" + "=" * 96)
print("B. STRUCTURAL: Poisson(buy-days) x lognormal(value) -> exact E[log1p(S)]")
print("=" * 96, flush=True)
m_cnt = lgb.train(dict(P, objective="poisson", metric="poisson"),
                  lgb.Dataset(Atr, ntr), num_boost_round=R)
lam = np.maximum(m_cnt.predict(D[V]["A"]), 1e-6)
mu, sg = D[V]["mu"], D[V]["sg"]
print(f"  lambda mean={lam.mean():.4f}  true mean n={D[V]['n'].mean():.4f}")
z_str = expected_log1p(lam, mu, sg)
print(f"  structural RMSLE={rm(z_str):.5f}  mean(z)={z_str.mean():.4f}")
best = (9.9, 1.0, 0.0)
for fs in [0.8, 1.0, 1.2]:
    for dm in [-0.3, -0.15, 0.0, 0.15]:
        s = rm(expected_log1p(lam, mu + dm, sg * fs))
        if s < best[0]:
            best = (s, fs, dm)
print(f"  best value-model tweak: sigma x{best[1]}, mu{best[2]:+.2f} -> RMSLE={best[0]:.5f}")
z_str = expected_log1p(lam, mu + best[2], sg * best[1])
log("structural done")

print("\n" + "=" * 96)
print("C. SEQUENCE: aggregates + 104 weekly log-features")
print("=" * 96, flush=True)
Str = np.hstack([Atr, np.vstack([D[T]["S"] for T in CUTS])])
m_seq = lgb.train(dict(P, objective="regression", metric="rmse"),
                  lgb.Dataset(Str, np.log1p(ytr)), num_boost_round=R)
z_seq = m_seq.predict(np.hstack([D[V]["A"], D[V]["S"]]))
print(f"  agg+seq RMSLE={rm(z_seq):.5f}   delta vs direct={rm(z_seq)-rm(z_dir):+.5f}")
del Str
gc.collect()
log("seq done")

print("\n" + "=" * 96)
print("D. DIVERSITY / BLENDING")
print("=" * 96)
for nm, z in [("structural", z_str), ("sequence", z_seq)]:
    print(f"  corr(pred)      direct vs {nm:11s} = {np.corrcoef(z_dir, z)[0,1]:.4f}")
    print(f"  corr(residual)  direct vs {nm:11s} = {np.corrcoef(lyv-z_dir, lyv-z)[0,1]:.4f}")
print()
for w in [0.0, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
    print(f"  direct {1-w:.1f} + structural {w:.1f}: RMSLE={rm((1-w)*z_dir + w*z_str):.5f}")
print()
for w in [0.3, 0.5, 0.7]:
    print(f"  direct {1-w:.1f} + sequence   {w:.1f}: RMSLE={rm((1-w)*z_dir + w*z_seq):.5f}")
print()
best3 = min(((rm(a * z_dir + b * z_seq + (1 - a - b) * z_str), a, b)
             for a in np.arange(0, 1.01, 0.1) for b in np.arange(0, 1.01 - a, 0.1)))
print(f"  best 3-way: direct={best3[1]:.1f} seq={best3[2]:.1f} struct={1-best3[1]-best3[2]:.1f} "
      f"-> RMSLE={best3[0]:.5f}   (direct alone {rm(z_dir):.5f})")
log("done")

```
