"""Stage 7: real experiments. Panel rule, cutoff choice, direct vs two-part, seasonal calibration."""
import datetime as dt
import time

import lightgbm as lgb
import numpy as np
import polars as pl

import fe

OUT = r"C:\Users\Admin\AppData\Local\Temp\claude\C--Users-Admin-Desktop-OZON-E-CUP\f013b07e-ef3c-43c2-884c-856362ff21fa\scratchpad"
SEED = 42

t0 = time.time()
fe.load()
print(f"loaded in {time.time()-t0:.1f}s")

# Cutoffs.  Target of T must NOT overlap the guaranteed window [2025-11-16..2026-02-13]
# => T + 30 < 2025-11-16 => T <= 2025-10-16.  Panel needs 90d history => T >= 2025-03-31.
TRAIN_CUTS = [dt.date(2025, 4, 15), dt.date(2025, 5, 15), dt.date(2025, 6, 15),
              dt.date(2025, 7, 15), dt.date(2025, 8, 15)]
VAL_CUT = dt.date(2025, 9, 16)          # clean: target 2025-09-17..2025-10-16
VAL2_CUT = dt.date(2025, 10, 16)        # clean: target 2025-10-17..2025-11-15
DIRTY_CUT = dt.date(2026, 1, 14)        # contaminated: target == guaranteed block 0

cache = {}


def get(T, n_blocks=3):
    k = (T, n_blocks)
    if k not in cache:
        t = time.time()
        cache[k] = fe.make_xy(T, n_blocks)
        print(f"  built {T} blocks={n_blocks}: n={cache[k][0].height:,} "
              f"feats={cache[k][0].width-1} in {time.time()-t:.1f}s")
    return cache[k]


print("\n=== building datasets ===")
for T in TRAIN_CUTS + [VAL_CUT, VAL2_CUT]:
    get(T)
get(DIRTY_CUT)

FEATS = [c for c in cache[(VAL_CUT, 3)][0].columns if c != "user_id"]
print(f"\nn features = {len(FEATS)}")


def to_np(X):
    return X.select(FEATS).to_numpy().astype(np.float32)


def fit_predict(train_cuts, val_cut, params=None, num_round=900, two_part=False, weights=None):
    Xs, ys = [], []
    ws = []
    for i, T in enumerate(train_cuts):
        X, Y = get(T)
        Xs.append(to_np(X))
        ys.append(Y["y"].to_numpy())
        ws.append(np.full(X.height, 1.0 if weights is None else weights[i]))
    Xtr = np.vstack(Xs)
    ytr = np.concatenate(ys)
    wtr = np.concatenate(ws)
    Xv, Yv = get(val_cut)
    Xva, yva = to_np(Xv), Yv["y"].to_numpy()

    p = dict(objective="regression", metric="rmse", learning_rate=0.05, num_leaves=127,
             min_data_in_leaf=200, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
             lambda_l2=5.0, verbose=-1, seed=SEED, num_threads=12)
    if params:
        p.update(params)

    if not two_part:
        d = lgb.Dataset(Xtr, np.log1p(ytr), weight=wtr)
        m = lgb.train(p, d, num_boost_round=num_round)
        z = m.predict(Xva)
        return np.expm1(z), yva, m, z
    # two-part: P(y>0) * E[log1p(y)|y>0]
    pc = dict(p)
    pc.update(objective="binary", metric="binary_logloss")
    mc = lgb.train(pc, lgb.Dataset(Xtr, (ytr > 0).astype(int), weight=wtr), num_boost_round=num_round)
    msk = ytr > 0
    mr = lgb.train(p, lgb.Dataset(Xtr[msk], np.log1p(ytr[msk]), weight=wtr[msk]),
                   num_boost_round=num_round)
    pr = mc.predict(Xva)
    mu = mr.predict(Xva)
    z = pr * mu
    return np.expm1(z), yva, (mc, mr), z


def report(name, pred, y, z=None):
    s = fe.rmsle(y, pred)
    extra = ""
    if z is not None:
        extra = f"  mean(log1p(pred))={z.mean():.4f}  true mean(log1p(y))={np.log1p(y).mean():.4f}  bias={np.log1p(y).mean()-z.mean():+.4f}"
    print(f"  {name:52s} RMSLE={s:.5f}{extra}")
    return s


print("\n" + "=" * 110)
print("EXP 1: naive baselines on the CLEAN validation cutoff", VAL_CUT)
print("=" * 110)
Xv, Yv = get(VAL_CUT)
yv = Yv["y"].to_numpy()
report("persistence: gmv last 30d", Xv["w30_gmv"].to_numpy(), yv)
report("gmv last 90d / 3", Xv["w90_gmv"].to_numpy() / 3, yv)
report("gmv last 180d / 6", Xv["w180_gmv"].to_numpy() / 6, yv)
report("gmv last 365d / 12.17", Xv["w365_gmv"].to_numpy() / 12.1667, yv)
zg = np.linspace(0, 4, 401)
best = min(((fe.rmsle(yv, np.expm1(np.full(len(yv), c))), c) for c in zg))
report(f"best constant (log-space {best[1]:.3f})", np.expm1(np.full(len(yv), best[1])), yv)

print("\n" + "=" * 110)
print("EXP 2: PANEL RULE — does re-applying the organiser rule matter?")
print("=" * 110)
X3, Y3 = get(VAL_CUT, 3)
X1, Y1 = get(VAL_CUT, 1)
print(f"  panel 3-block: n={X3.height:,}  P(y>0)={float((Y3['y']>0).mean()):.4f}  "
      f"m_y={float(np.log1p(Y3['y'].to_numpy()).mean()):.4f}")
print(f"  panel 1-block: n={X1.height:,}  P(y>0)={float((Y1['y']>0).mean()):.4f}  "
      f"m_y={float(np.log1p(Y1['y'].to_numpy()).mean()):.4f}")
print("  -> the TEST panel is 3-block by construction; a 1-block panel is a different population")

print("\n" + "=" * 110)
print("EXP 3: single recent cutoff vs multi-cutoff training (validated on the CLEAN cutoff)")
print("=" * 110)
res = {}
p1, y1, m1, z1 = fit_predict([dt.date(2025, 8, 15)], VAL_CUT)
res["single T=2025-08-15"] = report("train 1 cutoff (nearest)", p1, y1, z1)
p2, y2, m2, z2 = fit_predict(TRAIN_CUTS, VAL_CUT)
res["multi 5 cutoffs"] = report("train 5 cutoffs", p2, y2, z2)

print("\n" + "=" * 110)
print("EXP 4: DIRECT log1p regression vs TWO-PART (classifier x conditional log-mean)")
print("=" * 110)
p3, y3, m3, z3 = fit_predict(TRAIN_CUTS, VAL_CUT, two_part=True)
res["two-part"] = report("two-part P(buy) x E[log1p|buy]", p3, y3, z3)
for a in [0.3, 0.5, 0.7]:
    zb = a * z2 + (1 - a) * z3
    report(f"blend direct/two-part a={a}", np.expm1(zb), y2, zb)

print("\n" + "=" * 110)
print("EXP 5: THE TRAP — training on the contaminated cutoff T=2026-01-14")
print("=" * 110)
Xd, Yd = get(DIRTY_CUT)
print(f"  T={DIRTY_CUT}: P(any activity next 30d)=1.0000 by construction, "
      f"P(y>0)={float((Yd['y']>0).mean()):.4f}")
pd_, yd_, md_, zd_ = fit_predict([DIRTY_CUT], VAL_CUT)
res["single dirty cutoff"] = report("train on 2026-01-14 only -> clean val", pd_, yd_, zd_)
pdd, ydd, mdd, zdd = fit_predict(TRAIN_CUTS, DIRTY_CUT)
report("train clean -> validate on DIRTY cutoff (optimistic!)", pdd, ydd, zdd)

print("\n" + "=" * 110)
print("EXP 6: feature importance (top 30) of the multi-cutoff direct model")
print("=" * 110)
imp = sorted(zip(FEATS, m2.feature_importance("gain")), key=lambda t: -t[1])
for n, v in imp[:30]:
    print(f"  {n:32s} {v:14,.0f}")
print("\n  ... presence-only features rank:")
for n, v in imp:
    if "ponly" in n or "presence_only" in n:
        print(f"  {n:32s} {v:14,.0f}  (rank {[a for a,_ in imp].index(n)+1})")

np.save(OUT + r"\z_direct.npy", z2)
np.save(OUT + r"\z_twopart.npy", z3)
np.save(OUT + r"\y_val.npy", y2)
print(f"\ntotal {time.time()-t0:.0f}s")
