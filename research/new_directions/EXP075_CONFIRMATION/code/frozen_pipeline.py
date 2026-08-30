"""Faithful port of the frozen EXP075 A1/A2 pipeline (run_a1_clean_forward.py +
run_a2_cnn_pilot.py) to the audit container.  Features / architecture /
hyperparameters / objective / preprocessing are unchanged.  The only differences:
  * the 11-channel panel is rebuilt from raw train.parquet (verified bitwise
    identical to the historical seq_panel_v1.npy on a 300-user sample);
  * LightGBM num_threads=2 (deterministic=True + force_col_wise=True make
    LightGBM results independent of thread count);
  * torch runs on CPU (no CUDA in the audit container);
  * the `validation_cutoff <= 2025-10-16` guard is lifted so that late
    confirmation folds can be built.  Nothing else is relaxed.
"""
from __future__ import annotations
import datetime as dt, gc, json, math, os, time
import numpy as np, pandas as pd, polars as pl, lightgbm as lgb

RAW = "/mnt/user-data/uploads/OZON-E-CUP/data/raw/train.parquet"
SAMPLE = "/mnt/user-data/uploads/OZON-E-CUP/data/raw/sample_submit.csv"
WORK = "/home/claude/work"
DATA_START = dt.date(2025, 1, 1); DATA_END = dt.date(2026, 2, 13)
N_DAYS = (DATA_END - DATA_START).days + 1
FOLDS = [dt.date(2025,9,4), dt.date(2025,9,18), dt.date(2025,10,2), dt.date(2025,10,16)]
FOLD_WEIGHTS = np.asarray([1.,2.,4.,8.])
TRAIN_LAGS = [77, 63, 49, 35]
WINDOWS = [7,14,30,60,90,180,365]
RAW_CHANNELS = ["cat","searches","search_to_cart","search_to_ord","cat_to_cart",
                "cat_to_ord","to_cart","to_ord","gmv_search","gmv_cat","gmv"]
NCH = len(RAW_CHANNELS)
SEED = 42; CHUNK = 16_000
CTX_DIM = len(WINDOWS)*NCH*2 + NCH + 1
T0 = time.time()
def log(*v): print(f"[{time.time()-T0:8.1f}s]", *v, flush=True)

def day_index(v: dt.date) -> int: return (v - DATA_START).days

def stable_half(u):
    x = np.asarray(u, dtype=np.uint64)
    x = x ^ (x >> np.uint64(30)); x = x*np.uint64(0xBF58476D1CE4E5B9)
    x = x ^ (x >> np.uint64(27)); x = x*np.uint64(0x94D049BB133111EB)
    x = x ^ (x >> np.uint64(31))
    return (x & np.uint64(1)).astype(np.int8)

def hash64(v):
    x = np.asarray(v, dtype=np.uint64)
    x = x ^ (x >> np.uint64(30)); x = x*np.uint64(0xBF58476D1CE4E5B9)
    x = x ^ (x >> np.uint64(27)); x = x*np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))

def lgb_params(kind):
    common = dict(objective="regression_l2", metric="rmse", max_bin=63,
        bagging_fraction=0.8, bagging_freq=1, deterministic=True, force_col_wise=True,
        verbosity=-1, num_threads=2, seed=SEED, feature_fraction_seed=SEED,
        bagging_seed=SEED, data_random_seed=SEED)
    if kind == "baseline":
        return dict(common, learning_rate=0.035, num_leaves=63, min_data_in_leaf=800,
                    feature_fraction=0.8, lambda_l2=20.0)
    return dict(common, learning_rate=0.03, num_leaves=31, min_data_in_leaf=1000,
                feature_fraction=0.65, lambda_l2=30.0)

def train_lgb(X, y, kind, rounds):
    ds = lgb.Dataset(X, label=np.asarray(y, dtype=np.float32), free_raw_data=True)
    m = lgb.train(lgb_params(kind), ds, num_boost_round=rounds)
    del ds; gc.collect()
    return m

class CleanData:
    def __init__(self):
        self.panel = np.load(f"{WORK}/panel11.npy", mmap_mode="r")
        self.gmv = np.load(f"{WORK}/gmv.npy", mmap_mode="r")
        self.uid = np.load(f"{WORK}/uid.npy")
        assert self.panel.shape == (250_000, N_DAYS, NCH)
        assert self.gmv.shape == (250_000, N_DAYS)
        assert np.all(self.uid[1:] > self.uid[:-1])
        self._cache = {}
    def rows(self, user_ids):
        v = np.asarray(user_ids, dtype=np.int64)
        i = np.searchsorted(self.uid, v)
        assert i.max(initial=0) < len(self.uid) and np.array_equal(self.uid[i], v)
        return i.astype(np.int32)
    def raw_cutoff_frame(self, cutoff):
        if cutoff in self._cache: return self._cache[cutoff]
        ts, te = cutoff + dt.timedelta(days=1), cutoff + dt.timedelta(days=30)
        assert te <= DATA_END, f"target end {te} beyond raw data"
        bx = []
        for b in range(3):
            end = cutoff - dt.timedelta(days=30*b); start = end - dt.timedelta(days=29)
            bx.append(pl.col("event_date").is_between(start, end, closed="both").any().alias(f"b{b}"))
        elig = (pl.scan_parquet(RAW)
                .filter(pl.col("event_date").is_between(cutoff-dt.timedelta(days=89), cutoff, closed="both"))
                .group_by("user_id").agg(bx)
                .filter(pl.all_horizontal([pl.col(f"b{i}") for i in range(3)]))
                .select("user_id").collect().sort("user_id"))
        fut = (pl.scan_parquet(RAW)
               .filter(pl.col("event_date").is_between(ts, te, closed="both"))
               .group_by("user_id").agg([
                   pl.col("gmv").sum().alias("target_y30"),
                   (pl.col("gmv") > 0).sum().alias("target_purchase_days"),
                   pl.len().alias("target_events")]).collect())
        f = (elig.join(fut, on="user_id", how="left")
             .with_columns([pl.col("target_y30").fill_null(0.0),
                            pl.col("target_purchase_days").fill_null(0),
                            pl.col("target_events").fill_null(0)])
             .sort("user_id").to_pandas())
        r = self.rows(f["user_id"].to_numpy()); d = day_index(cutoff)
        mm = self.gmv[r, d+1:d+31].sum(axis=1)
        err = float(np.max(np.abs(mm - f["target_y30"].to_numpy())))
        assert err <= 1e-8, f"raw/mmap target mismatch {cutoff}: {err}"
        f["cutoff"] = cutoff.isoformat()
        f["target_log"] = np.log1p(f["target_y30"].to_numpy(dtype=np.float64))
        self._cache[cutoff] = f
        log("raw cutoff", cutoff, "rows", len(f), "target parity", err,
            "inactive%%=%.3f" % (100*np.mean(f.target_events.to_numpy()==0)))
        return f
    def _padded(self, rows, cutoff, history):
        d = day_index(cutoff); start = d - history + 1; lo = max(0, start)
        out = np.zeros((len(rows), history, NCH), dtype=np.float32)
        if lo <= d:
            out[:, lo-start:, :] = np.asarray(self.panel[rows, lo:d+1, :], dtype=np.float32)
        assert DATA_START + dt.timedelta(days=d) <= cutoff
        return out
    def context_features(self, rows, cutoff, out=None):
        if out is None: out = np.empty((len(rows), CTX_DIM), dtype=np.float32)
        for s in range(0, len(rows), CHUNK):
            e = min(s+CHUNK, len(rows)); rr = rows[s:e]
            seq = self._padded(rr, cutoff, 365)
            cols = []
            for w in WINDOWS:
                tail = seq[:, -w:, :]
                cols.extend([tail.sum(axis=1), (tail > 0).sum(axis=1, dtype=np.int32).astype(np.float32)])
            nz = seq > 0
            rec = np.full((len(rr), NCH), 366.0, dtype=np.float32)
            has = nz.any(axis=1); rev = nz[:, ::-1, :].argmax(axis=1).astype(np.float32)
            rec[has] = rev[has]
            avail = np.full((len(rr),1), min(day_index(cutoff)+1, 365), dtype=np.float32)
            out[s:e] = np.concatenate([*cols, rec, avail], axis=1)
            del seq, nz, rec, cols, rev
        return out
    def weekly_only(self, rows, cutoff, history=365, out=None):
        nw = math.ceil(history/7)
        if out is None: out = np.empty((len(rows), nw, NCH), dtype=np.float32)
        pad = nw*7 - history
        for s in range(0, len(rows), CHUNK):
            e = min(s+CHUNK, len(rows)); rr = rows[s:e]
            seq = self._padded(rr, cutoff, history)
            sw = np.pad(seq, ((0,0),(pad,0),(0,0))) if pad else seq
            out[s:e] = sw.reshape(len(rr), nw, 7, NCH).sum(axis=2)
            del seq, sw
        return out
    def candidate_into(self, X, pos, rows, cutoff, history, ctx):
        nw = math.ceil(history/7); tdim = (nw+28)*NCH
        pad = nw*7 - history
        for s in range(0, len(rows), CHUNK):
            e = min(s+CHUNK, len(rows)); rr = rows[s:e]
            seq = self._padded(rr, cutoff, history)
            sw = np.pad(seq, ((0,0),(pad,0),(0,0))) if pad else seq
            weekly = sw.reshape(len(rr), nw, 7, NCH).sum(axis=2)
            daily = seq[:, -28:, :]
            X[pos+s:pos+e, :nw*NCH] = weekly.reshape(len(rr), -1)
            X[pos+s:pos+e, nw*NCH:tdim] = daily.reshape(len(rr), -1)
            del seq, sw, weekly, daily
        X[pos:pos+len(rows), tdim:] = ctx
        return tdim

def project_candidate(u, baseline_z):
    u0 = np.asarray(u, float) - float(np.mean(u))
    x = np.asarray(baseline_z, float) - float(np.mean(baseline_z))
    den = float(np.dot(x, x)); beta = 0.0 if den == 0 else float(np.dot(x, u0)/den)
    perp = u0 - beta*x; perp -= float(np.mean(perp))
    beta2 = 0.0 if den == 0 else float(np.dot(x, perp)/den)
    perp -= beta2*x; perp -= float(np.mean(perp))
    return perp, {"mean_removed": float(np.mean(u)), "beta1": beta, "beta2": beta2,
                  "max_projection_after_second_pass": abs(float(np.dot(x, perp)/max(den,1e-300)))}

def correlation(x, y):
    x = np.asarray(x, float) - float(np.mean(x)); y = np.asarray(y, float) - float(np.mean(y))
    d = math.sqrt(float(np.dot(x,x)*np.dot(y,y)))
    return 0.0 if d == 0 else float(np.dot(x,y)/d)
