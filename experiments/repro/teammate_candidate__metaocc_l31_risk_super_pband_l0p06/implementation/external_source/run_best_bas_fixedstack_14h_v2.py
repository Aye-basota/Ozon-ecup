#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E-CUP 2026 / Track 3 — fixed champion + full-user research stack (14h).

PURPOSE
-------
This runner deliberately NEVER retrains the teammate's STRONGEST_CURRENT neural
ensemble.  The packaged production prediction is a fixed anchor.  We reuse the
already-computed full-user OOF CAP/UNC/DIST/HURDLE artifacts from previous
best_bas runs, train only NEW table/meta experts, validate the new table signal
on the teammate's four uncontaminated folds, then replace/reshape the 55% table
slot of STRONGEST_CURRENT while preserving its 45% SEQ/ETX slot exactly.

Expected layout
---------------
  src/DL/best_bas/run_best_bas_fixedstack_14h.py
  src/DL/best_bas/run_best_bas_research_23h.py
  src/DL/best_bas/submission_STRONGEST_CURRENT/
  src/DL/best_bas/_best_bas_research/checkpoints/folds/{cap,unc,dist,hurdle}__*.npz

The previous full-user CAP/UNC/DIST/HURDLE OOF is REQUIRED and is never retrained.
New experts are trained on all available users (row_frac=1.0), never mod-4/25%.

Research order (most promising first)
-------------------------------------
A. Existing-artifact stackers (minutes):
   - S06-style Ridge stack/residual (teammate backlog), prediction-only + rich-meta;
   - constrained simplex / greedy ensemble selection;
   - Phase12 RMSLE-effective q*=clip(z/mu,0,1);
   - p-band local mixture with shrinkage;
   - conservative occurrence Platt recalibration inside the TABLE slot only.
B. New full-user table experts (hours, resumable), in priority order:
   - recent_hurdle; multiscale_direct; recent_direct; recent_dist.
   After each family the whole cheap stack search is re-run.
C. Nonlinear residual LGBM + robust combinations of independently validated
   mechanisms.  No class1 hard router / no previous ranker_safe repetition.
D. Final refit ONLY of new experts that are actually needed by the shortlisted
   candidate recipes.  STRONGEST_CURRENT stays frozen.
E. Emit 5-7 candidate submissions, selected for validation quality + temporal
   robustness + test-regime plausibility + diversity from STRONGEST_CURRENT and
   the previous ranker_safe/class1_occ submissions if they are found locally.

Validation
----------
Primary folds are exactly teammate clean folds:
  2025-09-04, 2025-09-18, 2025-10-02, 2025-10-16, weights 1:2:4:8.
Meta learners are WALK-FORWARD: prediction for fold i is trained only on folds < i.
The first fold therefore falls back to the original table slot for meta methods.
Raw new experts follow the teammate purge rule through src.train.Setup.

Important limitation
--------------------
We intentionally do not rebuild historical SEQ/ETX OOF.  Consequently local
selection measures the quality of the replacement TABLE signal, not an exact
reconstruction of final STRONGEST_CURRENT OOF.  This is the correct compromise
under the user's constraint not to spend ~10h/epoch rebuilding the fixed model.
We compensate with 4-fold consistency, recent-fold checks, OOF->test variance
checks, and only modify the table contribution; the proven SEQ/ETX production
slot remains byte-for-byte unchanged in the final anchor prediction.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import gc
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import traceback
import warnings
import zipfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names.*")

SCRIPT_VERSION = "fixedstack_14h_2026-08-22_006"
FOLDS = ("2025-09-04", "2025-09-18", "2025-10-02", "2025-10-16")
FW = np.asarray([1.0, 2.0, 4.0, 8.0], dtype=np.float64)
LEVEL = 2.3293
EPS = 1e-7
TABLE_WEIGHT = 0.55
CORE_TABLE_WEIGHTS = {"cap": 0.10/TABLE_WEIGHT, "unc": 0.20/TABLE_WEIGHT, "dist": 0.25/TABLE_WEIGHT}
KNOWN_LB = {"STRONGEST_CURRENT": 1.6496571, "ranker_safe": 1.654133685532829,
            "class1_occ": 1.688068573391526}

# -----------------------------------------------------------------------------
# generic utilities
# -----------------------------------------------------------------------------
def now() -> str: return dt.datetime.now().isoformat(timespec="seconds")
def log(*x: Any) -> None: print(f"[{now()}]", *x, flush=True)

def jdefault(x: Any):
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return float(x)
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, Path): return str(x)
    if isinstance(x, dt.date): return x.isoformat()
    raise TypeError(type(x).__name__)

def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=jdefault), encoding="utf-8")
    os.replace(tmp, path)

def save_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    import pandas as pd
    path.parent.mkdir(parents=True, exist_ok=True)
    flat=[]
    for r in rows:
        q={}
        for k,v in r.items():
            q[k]=json.dumps(v,ensure_ascii=False,default=jdefault) if isinstance(v,(dict,list,tuple,np.ndarray)) else v
        flat.append(q)
    pd.DataFrame(flat).to_csv(path,index=False)

def append_error(path: Path, stage: str, name: str, fold: str, exc: BaseException) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a",encoding="utf-8") as f:
        f.write(json.dumps(dict(time=now(),stage=stage,name=name,fold=fold,error=repr(exc),traceback=traceback.format_exc()),ensure_ascii=False)+"\n")

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()

def truez(y): return np.log1p(np.maximum(np.asarray(y,np.float64),0.0))
def clipz(z): return np.clip(np.nan_to_num(np.asarray(z,np.float64),nan=0.0,posinf=20.0,neginf=0.0),0.0,20.0)
def rms_z(y,z): return float(np.sqrt(np.mean((truez(y)-clipz(z))**2)))
def wavg(a): return float(np.dot(np.asarray(a,np.float64),FW)/FW.sum())

def calibrate(y,z,iters=40):
    yy=truez(y);zz=np.asarray(z,np.float64);d=float((yy-zz).mean())
    for _ in range(iters):
        m=zz+d>0
        if not m.any():break
        nd=float((yy[m]-zz[m]).mean())
        if abs(nd-d)<1e-12:d=nd;break
        d=nd
    return d,rms_z(y,zz+d)

def level_test(z,level=LEVEL):
    z=clipz(z);d=float(level-z.mean());return clipz(z+d),d

def sigmoid(x):
    x=np.clip(np.asarray(x,np.float64),-35,35);return 1/(1+np.exp(-x))
def logit(p):
    p=np.clip(np.asarray(p,np.float64),1e-6,1-1e-6);return np.log(p/(1-p))
def align(src_uid,arr,dst_uid):
    su=np.asarray(src_uid,np.int64);du=np.asarray(dst_uid,np.int64);a=np.asarray(arr)
    if np.array_equal(su,du):return a
    o=np.argsort(su);ss=su[o];p=np.searchsorted(ss,du)
    if (p>=len(ss)).any() or not np.array_equal(ss[p],du):raise ValueError("user_id sets differ")
    return a[o[p]]

def rank01(x):
    x=np.asarray(x,np.float64);o=np.argsort(x,kind="mergesort");r=np.empty(len(x),np.float32)
    if len(x)<=1:r[:]=.5
    else:r[o]=np.arange(len(x),dtype=np.float32)/(len(x)-1)
    return r

def candidate_distance(a,b):
    a=np.asarray(a,np.float64);b=np.asarray(b,np.float64);d=a-b
    return dict(corr=float(np.corrcoef(a,b)[0,1]),mean_abs=float(np.mean(np.abs(d))),std=float(np.std(d)),
                pct05=float(np.mean(np.abs(d)>.05)),pct10=float(np.mean(np.abs(d)>.10)))

@dataclasses.dataclass
class Budget:
    started: float
    max_hours: float
    reserve: float=2.25
    @property
    def elapsed(self):return (time.time()-self.started)/3600
    @property
    def remaining(self):return self.max_hours-self.elapsed
    def can_start(self,est,extra=0):return self.remaining>float(est)+self.reserve+float(extra)

# -----------------------------------------------------------------------------
# previous runner, paths, cache repair
# -----------------------------------------------------------------------------
def import_prev(base: Path):
    cands=[base/"run_best_bas_research_23h.py"]+sorted(base.glob("run_best_bas_research*.py"),key=lambda p:p.stat().st_mtime,reverse=True)
    for p in cands:
        if not p.exists():continue
        spec=importlib.util.spec_from_file_location("fixedstack_prev",p)
        if spec is None or spec.loader is None:continue
        m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
        if all(hasattr(m,x) for x in ("Context","Budget","discover_package","discover_raw_and_sample","configure_pipeline","verify_friend_package","fold_ckpt","load_fold","build_test_meta_raw")):
            return m,p
    raise FileNotFoundError("run_best_bas_research_23h.py not found beside this script")

def parquet_ok(p: Path)->bool:
    try:
        if p.stat().st_size<8:return False
        with p.open("rb") as f:
            return f.read(4)==b"PAR1" and (f.seek(-4,os.SEEK_END) is not None) and f.read(4)==b"PAR1"
    except Exception:return False

def repair_cache(ctx)->list[str]:
    root=Path(ctx.work)/"cache"/"processed";removed=[]
    if not root.exists():return removed
    for p in root.rglob("*.parquet"):
        if parquet_ok(p):continue
        try:p.unlink();removed.append(str(p));log("CACHE REPAIR",p.name)
        except Exception as e:raise RuntimeError(f"cannot delete corrupted cache {p}: {e}")
    return removed

def install_atomic_cache(ctx):
    F=ctx.features_mod
    if getattr(F,"_fixedstack_atomic",False):return
    default_L=getattr(F,"HISTORY_L",180)
    def safe(T,L=default_L,norm_long=False):
        p=F.DATA_PROCESSED/f"feat_{F._tag(T)}_L{'norm' if norm_long else ''}{L}.parquet"
        if p.exists():
            try:return F.pl.read_parquet(p)
            except Exception:
                p.unlink(missing_ok=True);log("CACHE SELF-HEAL",p.name)
        x=F.build_features(T,L,norm_long);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f".tmp.{os.getpid()}.parquet")
        tmp.unlink(missing_ok=True);x.write_parquet(tmp)
        if not parquet_ok(tmp):tmp.unlink(missing_ok=True);raise RuntimeError(f"truncated parquet write {tmp}")
        os.replace(tmp,p);return x
    F.features_cached=safe;F._fixedstack_atomic=True

def discover_work(base: Path, explicit: str|None)->Path:
    """Find a reusable previous work directory without touching missing stale paths.

    A previous version unconditionally added preferred directory names to the
    candidate list.  On Windows Path.resolve() may keep a missing path, and the
    later p.stat() then raises FileNotFoundError.  We only admit directories
    that actually contain checkpoints/folds and score defensively.
    """
    if explicit:
        p=Path(explicit)
        p=p if p.is_absolute() else base/p
        folds_dir=p/"checkpoints"/"folds"
        if not folds_dir.is_dir():
            raise FileNotFoundError(f"Reuse directory has no checkpoints/folds: {p}")
        return p.resolve()

    preferred=[base/"_best_bas_research", base/"_best_bas_phase15_friend"]
    discovered=list(base.glob("_best_bas*"))
    seen=[]
    for p in preferred + discovered:
        try:
            if not p.is_dir() or not (p/"checkpoints"/"folds").is_dir():
                continue
            rp=p.resolve()
        except OSError:
            continue
        if rp not in seen:
            seen.append(rp)

    if not seen:
        raise FileNotFoundError(
            f"No reusable _best_bas.../checkpoints/folds directory found under {base}"
        )

    def score(p):
        try:
            f=p/"checkpoints"/"folds"
            required=sum(
                (f/f"{n}__{fold}.npz").exists()
                for n in ("cap","unc","dist","hurdle")
                for fold in FOLDS
            )
            return (required, len(list(f.glob("*.npz"))), p.stat().st_mtime)
        except OSError:
            return (-1, -1, -1.0)

    seen=[p for p in seen if score(p)[0] >= 0]
    if not seen:
        raise FileNotFoundError(
            f"Reusable work directories were found under {base}, but none are readable"
        )
    seen.sort(key=score, reverse=True)
    best=seen[0]
    log("REUSE WORK", best, "score", score(best))
    return best

def valid_npz(path:Path,need=("user_id","y","z")):
    try:
        with np.load(path,allow_pickle=False) as d:
            if not set(need).issubset(d.files):return False
            n=len(d["user_id"]);return n>0 and all((np.ndim(d[k])==0 or len(d[k])==n) for k in need)
    except Exception:return False

def load_npz(path:Path):
    with np.load(path,allow_pickle=False) as d:return {k:d[k] for k in d.files}

def save_npz_atomic(path:Path,**kw):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f".tmp.{os.getpid()}.npz")
    np.savez_compressed(tmp,**kw)
    with np.load(tmp,allow_pickle=False) as d:_=d[d.files[0]]
    os.replace(tmp,path)

# -----------------------------------------------------------------------------
# core bank — FIXED existing artifacts only
# -----------------------------------------------------------------------------
def load_core(prev,ctx):
    bank={}
    for fold in FOLDS:
        ds={}
        for n in ("cap","unc","dist","hurdle"):
            p=prev.fold_ckpt(ctx,n,fold);need=("user_id","y","z","p","mu") if n=="hurdle" else ("user_id","y","z")
            if not valid_npz(p,need):
                raise RuntimeError(f"Required existing full-user artifact missing/corrupt: {p}. This runner refuses to retrain the fixed base.")
            ds[n]=load_npz(p)
        uid=ds["cap"]["user_id"].astype(np.int64);y=ds["cap"]["y"].astype(np.float64)
        r={"uid":uid,"y":y,"true_z":truez(y)}
        for n in ("cap","unc","dist","hurdle"):
            r[n]=align(ds[n]["user_id"],ds[n]["z"],uid).astype(np.float64)
        r["p"]=np.clip(align(ds["hurdle"]["user_id"],ds["hurdle"]["p"],uid).astype(np.float64),EPS,1-EPS)
        r["mu"]=np.maximum(align(ds["hurdle"]["user_id"],ds["hurdle"]["mu"],uid).astype(np.float64),0)
        cap=ds["cap"]
        if "meta_raw" not in cap:
            raise RuntimeError(f"{fold}: cap checkpoint has no meta_raw; previous full-user runner is required")
        r["meta_raw"]=align(cap["user_id"],cap["meta_raw"],uid).astype(np.float32)
        r["meta_names"]=cap.get("meta_names",np.asarray([f"m{i}" for i in range(r['meta_raw'].shape[1])])).astype(str).tolist()
        r["table_core"]=sum(CORE_TABLE_WEIGHTS[n]*r[n] for n in CORE_TABLE_WEIGHTS)
        bank[fold]=r
    return bank

def score_table(name,preds,bank,family,rows,notes=""):
    scores=[];base_scores=[];raw=[];rawb=[];offs=[];deltas=[]
    for f in FOLDS:
        r=bank[f];z=np.asarray(preds[f],np.float64);off,sc=calibrate(r["y"],z);bo,bsc=calibrate(r["y"],r["table_core"])
        scores.append(sc);base_scores.append(bsc);raw.append(rms_z(r["y"],z));rawb.append(rms_z(r["y"],r["table_core"]));offs.append(off);deltas.append(sc-bsc)
    row=dict(name=name,family=family,wcv=wavg(scores),base_wcv=wavg(base_scores),delta=wavg(deltas),wins=int(sum(d<0 for d in deltas)),
             wins_recent=int(sum(d<0 for d in deltas[1:])),latest_delta=float(deltas[-1]),worst_delta=float(max(deltas)),
             raw_delta=wavg(np.asarray(raw)-np.asarray(rawb)),offset_mean=float(np.mean(offs)),offset_std=float(np.std(offs)),
             fold_scores=scores,fold_deltas=deltas,notes=notes)
    rows.append(row);return row

# -----------------------------------------------------------------------------
# feature matrices for meta learners
# -----------------------------------------------------------------------------
def pred_features(rec,experts,include_meta=True):
    P=np.stack([rec[n] for n in experts],1).astype(np.float32)
    core=rec["table_core"].astype(np.float32);p=rec["p"].astype(np.float32);mu=rec["mu"].astype(np.float32)
    derived=np.column_stack([core,p,mu,np.log1p(mu),p*(1-p),P.mean(1),P.std(1),P.min(1),P.max(1),P.max(1)-P.min(1)])
    diffs=np.column_stack([P[:,j]-core for j in range(P.shape[1])])
    pieces=[P,derived.astype(np.float32),diffs.astype(np.float32)]
    if include_meta:pieces.insert(0,np.asarray(rec["meta_raw"],np.float32))
    X=np.column_stack(pieces).astype(np.float32);return np.nan_to_num(X,nan=0,posinf=20,neginf=-20)

def concat_train(bank,ids,experts,include_meta=True):
    X=[];y=[];w=[];q=[]
    for j in ids:
        r=bank[FOLDS[j]];X.append(pred_features(r,experts,include_meta));y.append(r["true_z"]-r["table_core"])
        w.append(np.full(len(r["uid"]),FW[j],np.float32));q.append(np.clip(r["true_z"]/np.maximum(r["mu"],.15),0,1))
    return np.vstack(X),np.concatenate(y),np.concatenate(w),np.concatenate(q)

def make_ridge(alpha):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    return make_pipeline(StandardScaler(copy=False),Ridge(alpha=float(alpha),solver="lsqr",tol=1e-4))

def make_lgbm(seed=42,objective="huber"):
    import lightgbm as lgb
    return lgb.LGBMRegressor(objective=objective,n_estimators=360,learning_rate=.035,num_leaves=31,min_child_samples=450,
                             subsample=.86,colsample_bytree=.78,reg_lambda=18.,reg_alpha=1.5,max_bin=127,random_state=seed,n_jobs=max(2,min(14,os.cpu_count() or 8)),verbosity=-1)

def walk_residual(bank,experts,kind,alpha=100.,shrink=.75,include_meta=True,seed=42):
    out={};models=[]
    for i,f in enumerate(FOLDS):
        r=bank[f]
        if i==0:out[f]=r["table_core"].copy();models.append(None);continue
        X,y,w,_=concat_train(bank,list(range(i)),experts,include_meta);Xt=pred_features(r,experts,include_meta)
        if kind=="ridge":m=make_ridge(alpha);m.fit(X,y,**{"ridge__sample_weight":w})
        else:m=make_lgbm(seed+i);m.fit(X,y,sample_weight=w)
        d=np.clip(m.predict(Xt),-2.0,2.0);out[f]=clipz(r["table_core"]+float(shrink)*d);models.append(m)
        del X,y,w,Xt,d;gc.collect()
    return out,models

def final_residual(bank,test,experts,kind,alpha=100.,shrink=.75,include_meta=True,seed=42):
    X,y,w,_=concat_train(bank,list(range(4)),experts,include_meta);Xt=pred_features(test,experts,include_meta)
    if kind=="ridge":m=make_ridge(alpha);m.fit(X,y,**{"ridge__sample_weight":w})
    else:m=make_lgbm(seed);m.fit(X,y,sample_weight=w)
    d=np.clip(m.predict(Xt),-2.0,2.0);z=clipz(test["table_core"]+float(shrink)*d)
    del X,y,w,Xt,d,m;gc.collect();return z

# Effective RMSLE q target: learn how much of positive magnitude should be active.
def walk_q(bank,experts,shrink=1.0,seed=4100):
    out={}
    for i,f in enumerate(FOLDS):
        r=bank[f]
        if i==0:out[f]=r["table_core"].copy();continue
        X,_,w,q=concat_train(bank,list(range(i)),experts,True);Xt=pred_features(r,experts,True)
        m=make_lgbm(seed+i,objective="regression");qw=w*np.minimum(np.concatenate([bank[FOLDS[j]]["mu"]**2 for j in range(i)]),25.0).astype(np.float32)
        m.fit(X,q,sample_weight=qw);qh=np.clip(m.predict(Xt),0,1)
        base_pm=r["p"]*r["mu"];resid=r["table_core"]-base_pm;raw=qh*r["mu"]+resid
        out[f]=clipz((1-shrink)*r["table_core"]+shrink*raw)
        del X,w,q,Xt,qw,qh,m,raw;gc.collect()
    return out

def final_q(bank,test,experts,shrink=1.0,seed=4100):
    X,_,w,q=concat_train(bank,list(range(4)),experts,True);Xt=pred_features(test,experts,True)
    mus=np.concatenate([bank[f]["mu"] for f in FOLDS]);qw=w*np.minimum(mus**2,25.0).astype(np.float32)
    m=make_lgbm(seed,objective="regression");m.fit(X,q,sample_weight=qw);qh=np.clip(m.predict(Xt),0,1)
    raw=qh*test["mu"]+(test["table_core"]-test["p"]*test["mu"]);z=clipz((1-shrink)*test["table_core"]+shrink*raw)
    del X,w,q,Xt,m,qh,qw;gc.collect();return z

# -----------------------------------------------------------------------------
# convex / greedy / p-band ensembles
# -----------------------------------------------------------------------------
def fit_simplex(bank,ids,experts,lam=.03,min_new=0.,new_names=()):
    from scipy.optimize import minimize
    Z=[];y=[];sw=[]
    for j in ids:
        r=bank[FOLDS[j]];Z.append(np.stack([r[n] for n in experts],1));y.append(r["true_z"]);sw.append(np.full(len(r["uid"]),FW[j]))
    Z=np.vstack(Z);y=np.concatenate(y);sw=np.concatenate(sw);sw=sw/sw.mean()
    prior=np.zeros(len(experts));
    for n,w in CORE_TABLE_WEIGHTS.items():
        if n in experts:prior[experts.index(n)]=w
    if prior.sum()==0:prior[:]=1/len(prior)
    else:prior/=prior.sum()
    def obj(x):
        e=Z@x-y;return float(np.mean(sw*e*e)+lam*np.sum((x-prior)**2))
    cons=[{"type":"eq","fun":lambda x:float(x.sum()-1)}]
    inds=[experts.index(n) for n in new_names if n in experts]
    if min_new>0 and inds:cons.append({"type":"ineq","fun":lambda x:float(x[inds].sum()-min_new)})
    res=minimize(obj,prior,method="SLSQP",bounds=[(0,1)]*len(experts),constraints=cons,options={"maxiter":250,"ftol":1e-11})
    x=np.clip(res.x if res.success else prior,0,None);x=x/max(x.sum(),EPS);return x

def walk_simplex(bank,experts,lam=.03,min_new=0,new_names=()):
    out={};weights=[]
    for i,f in enumerate(FOLDS):
        if i==0:out[f]=bank[f]["table_core"].copy();weights.append(None);continue
        w=fit_simplex(bank,list(range(i)),experts,lam,min_new,new_names);out[f]=clipz(sum(w[k]*bank[f][n] for k,n in enumerate(experts)));weights.append(w)
    return out,weights

def final_simplex(bank,test,experts,lam=.03,min_new=0,new_names=()):
    w=fit_simplex(bank,list(range(4)),experts,lam,min_new,new_names);return clipz(sum(w[k]*test[n] for k,n in enumerate(experts))),w

def fit_greedy(bank,ids,experts,steps=25):
    # Caruana-style selection with replacement, optimized on log-space MSE.
    Z=[];y=[];sw=[]
    for j in ids:
        r=bank[FOLDS[j]];Z.append(np.stack([r[n] for n in experts],1));y.append(r["true_z"]);sw.append(np.full(len(r["uid"]),FW[j]))
    Z=np.vstack(Z);y=np.concatenate(y);sw=np.concatenate(sw);sw=sw/sw.mean();counts=np.zeros(len(experts),int);cur=None
    for t in range(steps):
        best=None
        for k in range(len(experts)):
            pr=Z[:,k] if cur is None else (cur*t+Z[:,k])/(t+1);loss=float(np.mean(sw*(pr-y)**2))
            if best is None or loss<best[0]:best=(loss,k,pr)
        counts[best[1]]+=1;cur=best[2]
    return counts/counts.sum()

def walk_greedy(bank,experts,steps=25):
    out={};ws=[]
    for i,f in enumerate(FOLDS):
        if i==0:out[f]=bank[f]["table_core"].copy();ws.append(None);continue
        w=fit_greedy(bank,list(range(i)),experts,steps);out[f]=clipz(sum(w[k]*bank[f][n] for k,n in enumerate(experts)));ws.append(w)
    return out,ws

def final_greedy(bank,test,experts,steps=25):
    w=fit_greedy(bank,list(range(4)),experts,steps);return clipz(sum(w[k]*test[n] for k,n in enumerate(experts))),w

def fit_pband(bank,ids,experts,bands=(0,.2,.4,.6,.8,1.0001),lam=.08):
    global_w=fit_simplex(bank,ids,experts,lam=.04);W=[]
    from scipy.optimize import minimize
    for b in range(len(bands)-1):
        Z=[];y=[];sw=[]
        for j in ids:
            r=bank[FOLDS[j]];m=(r["p"]>=bands[b])&(r["p"]<bands[b+1])
            if m.sum()<1000:continue
            Z.append(np.stack([r[n][m] for n in experts],1));y.append(r["true_z"][m]);sw.append(np.full(m.sum(),FW[j]))
        if not Z:W.append(global_w);continue
        Z=np.vstack(Z);y=np.concatenate(y);sw=np.concatenate(sw);sw/=sw.mean()
        def obj(x):return float(np.mean(sw*(Z@x-y)**2)+lam*np.sum((x-global_w)**2))
        res=minimize(obj,global_w,method="SLSQP",bounds=[(0,1)]*len(experts),constraints=[{"type":"eq","fun":lambda x:float(x.sum()-1)}],options={"maxiter":180,"ftol":1e-10})
        x=np.clip(res.x if res.success else global_w,0,None);W.append(x/max(x.sum(),EPS))
    return np.asarray(W)

def apply_pband(rec,experts,W,bands=(0,.2,.4,.6,.8,1.0001)):
    P=np.stack([rec[n] for n in experts],1);p=np.asarray(rec["p"],np.float64);z=np.empty(len(p),np.float64)
    centers=np.asarray([(bands[i]+bands[i+1])/2 for i in range(len(bands)-1)],np.float64)
    # Vectorised region assignment; each region is already strongly shrunk to the
    # global mixture during fit, which is safer than brittle hard model routing.
    idx=np.argmin(np.abs(p[:,None]-centers[None,:]),axis=1)
    for k in range(len(centers)):
        m=idx==k
        if m.any():z[m]=P[m]@W[k]
    return clipz(z)

def walk_pband(bank,experts):
    out={};Ws=[]
    for i,f in enumerate(FOLDS):
        if i==0:out[f]=bank[f]["table_core"].copy();Ws.append(None);continue
        W=fit_pband(bank,list(range(i)),experts);out[f]=apply_pband(bank[f],experts,W);Ws.append(W)
    return out,Ws

def final_pband(bank,test,experts):
    W=fit_pband(bank,list(range(4)),experts);return apply_pband(test,experts,W),W

# -----------------------------------------------------------------------------
# occurrence Platt inside table slot
# -----------------------------------------------------------------------------
def fit_platt(bank,ids):
    from sklearn.linear_model import LogisticRegression
    X=[];y=[];w=[]
    for j in ids:
        r=bank[FOLDS[j]];X.append(logit(r["p"]).reshape(-1,1));y.append((r["y"]>0).astype(np.int8));w.append(np.full(len(r["uid"]),FW[j]))
    X=np.vstack(X);y=np.concatenate(y);w=np.concatenate(w);m=LogisticRegression(C=.15,solver="lbfgs",max_iter=150);m.fit(X,y,sample_weight=w);return m

def occurrence_apply(rec,m,guard=False):
    pc=m.predict_proba(logit(rec["p"]).reshape(-1,1))[:,1];basepm=rec["p"]*rec["mu"];res=rec["table_core"]-basepm
    if guard:pc=np.where(pc<rec["p"],pc,.75*rec["p"]+.25*pc)
    return clipz(pc*rec["mu"]+res)

def walk_occ(bank,guard=False):
    out={}
    for i,f in enumerate(FOLDS):
        if i==0:out[f]=bank[f]["table_core"].copy();continue
        m=fit_platt(bank,list(range(i)));out[f]=occurrence_apply(bank[f],m,guard)
    return out

def final_occ(bank,test,guard=False):return occurrence_apply(test,fit_platt(bank,list(range(4))),guard)

# -----------------------------------------------------------------------------
# new ALL-USER table experts — no teammate model retraining
# -----------------------------------------------------------------------------
def expert_spec(ctx,name):
    T=ctx.train_mod
    common=dict(learning_rate=.035,num_leaves=63,min_data_in_leaf=220,feature_fraction=.82,bagging_fraction=.90,bagging_freq=1,lambda_l2=14.,lambda_l1=1.,max_bin=127)
    if name=="recent_hurdle":return T.Setup(L=0,min_history=90,step=7,panel_blocks=3,train_blocks=1,model="two_part",rounds=520,norm_long=True,weight_tau=105,params=common),"all"
    if name=="multiscale_direct":return T.Setup(L=180,min_history=90,step=7,panel_blocks=3,train_blocks=1,model="direct",rounds=650,params={**common,"num_leaves":95,"min_data_in_leaf":190}),"multiscale"
    if name=="recent_direct":return T.Setup(L=0,min_history=90,step=7,panel_blocks=3,train_blocks=1,model="direct",rounds=650,norm_long=True,weight_tau=105,params={**common,"num_leaves":95,"min_data_in_leaf":180}),"all"
    if name=="recent_dist":return T.Setup(L=0,min_history=90,step=7,panel_blocks=3,train_blocks=1,model="dist",rounds=330,norm_long=True,weight_tau=120,params=common),"all"
    raise KeyError(name)

def multiscale_features(feats):
    wins=("w7_","w14_","w30_","w60_","w90_","w180_");prefix=("rec_","trend_","dlog_","gap_","buygap_","pt_");exact={"weekend_share","tenure_frac","first_buy_frac","gap_max_frac"}
    out=[c for c in feats if c.startswith(wins) or c.startswith(prefix) or c in exact];seen=[];s=set()
    for c in out:
        if c not in s:seen.append(c);s.add(c)
    return seen

def new_fold_path(ctx,name,fold):return Path(ctx.checkpoints)/"folds"/f"{name}__{fold}.npz"
def new_test_path(ctx,name):return Path(ctx.checkpoints)/"test"/f"{name}_test.npz"

def train_expert_fold(ctx,name,fold):
    p=new_fold_path(ctx,name,fold);need=("user_id","y","z","p","mu") if name=="recent_hurdle" else ("user_id","y","z")
    if valid_npz(p,need):log("reuse new expert",p.name);return load_npz(p)
    s,mode=expert_spec(ctx,name);F,T,M=ctx.features_mod,ctx.train_mod,ctx.models_mod;V=dt.date.fromisoformat(fold);cuts=s.train_cutoffs(V)
    Xv,yv=T.xy(V,s);feats=T.select_features(F.feature_names(Xv),s.drop_groups,s.keep_only)
    if mode=="multiscale":feats=multiscale_features(feats)
    if len(feats)<18:raise RuntimeError(f"{name}: only {len(feats)} features")
    log("NEW FULL-USER",name,fold,"cuts",len(cuts),"features",len(feats));X,y,w=T.assemble(cuts,s,feats,V);n=len(y);T._XY.clear();box=[X];del X;gc.collect()
    # IMPORTANT: fit_free forwards temporal sample weights into LightGBM datasets.
    # recent_* experts would silently lose their defining recency weighting if we
    # called models.make_datasets(..., None, ...) directly.
    model=T.fit_free(s,box,y,w if s.weight_tau else None)
    A=F.to_np(Xv,feats);extra={}
    if s.model=="direct":z=model.predict(A)
    elif s.model=="dist":z=M.predict_dist(model,A)
    else:
        clf,reg=model;pp=np.clip(clf.predict(A),EPS,1-EPS);mu=np.maximum(reg.predict(A),0);z=pp*mu;extra=dict(p=pp.astype(np.float32),mu=mu.astype(np.float32))
    save_npz_atomic(p,user_id=Xv["user_id"].to_numpy().astype(np.int64),y=np.asarray(yv,np.float32),z=np.asarray(clipz(z),np.float32),**extra)
    off,sc=calibrate(yv,z);log("NEW DONE",name,fold,"train_rows",f"{n:,}","cal",f"{sc:.6f}","off",f"{off:+.4f}")
    del model,A,Xv,y,w,box;T._XY.clear();gc.collect();return load_npz(p)

def train_expert_test(ctx,name):
    p=new_test_path(ctx,name);need=("user_id","z","p","mu") if name=="recent_hurdle" else ("user_id","z")
    if valid_npz(p,need):log("reuse final expert",p.name);return load_npz(p)
    s,mode=expert_spec(ctx,name);F,T,M,C=ctx.features_mod,ctx.train_mod,ctx.models_mod,ctx.config
    Xt,_=F.make_xy(C.CUTOFF_TEST,s.L,s.panel_blocks,with_target=False,norm_long=s.norm_long);feats=T.select_features(F.feature_names(Xt),s.drop_groups,s.keep_only)
    if mode=="multiscale":feats=multiscale_features(feats)
    cuts=s.grid();log("FINAL NEW FULL-USER",name,"cuts",len(cuts),"features",len(feats));X,y,w=T.assemble(cuts,s,feats,C.CUTOFF_TEST);T._XY.clear();box=[X];del X;gc.collect()
    model=T.fit_free(s,box,y,w if s.weight_tau else None)
    A=F.to_np(Xt,feats);extra={}
    if s.model=="direct":z=model.predict(A)
    elif s.model=="dist":z=M.predict_dist(model,A)
    else:
        clf,reg=model;pp=np.clip(clf.predict(A),EPS,1-EPS);mu=np.maximum(reg.predict(A),0);z=pp*mu;extra=dict(p=pp.astype(np.float32),mu=mu.astype(np.float32))
    save_npz_atomic(p,user_id=Xt["user_id"].to_numpy().astype(np.int64),z=np.asarray(clipz(z),np.float32),**extra)
    del model,A,Xt,y,w,box;T._XY.clear();gc.collect();return load_npz(p)

def add_expert_to_bank(ctx,bank,name):
    for f in FOLDS:
        d=train_expert_fold(ctx,name,f);r=bank[f];r[name]=align(d["user_id"],d["z"],r["uid"]).astype(np.float64)
        if name=="recent_hurdle":r["p_recent_hurdle"]=align(d["user_id"],d["p"],r["uid"]);r["mu_recent_hurdle"]=align(d["user_id"],d["mu"],r["uid"])

# -----------------------------------------------------------------------------
# candidate definitions / research
# -----------------------------------------------------------------------------
@dataclasses.dataclass
class Spec:
    name:str;kind:str;family:str;params:dict[str,Any];experts:tuple[str,...]

def available_experts(bank):
    pref=["cap","unc","dist","hurdle","recent_hurdle","multiscale_direct","recent_direct","recent_dist"]
    return [n for n in pref if all(n in bank[f] for f in FOLDS)]

def make_specs(bank):
    ex=tuple(available_experts(bank));new=tuple(n for n in ex if n not in ("cap","unc","dist","hurdle"));specs=[]
    # Backlog S06 and robust variants.
    for alpha in (30.,150.,700.):
        specs.append(Spec(f"ridge_meta_a{int(alpha)}_s075","ridge_resid","ridge_meta",dict(alpha=alpha,shrink=.75,meta=True),ex))
    specs.append(Spec("ridge_predonly_a80_s1","ridge_resid","ridge_predonly",dict(alpha=80.,shrink=1.,meta=False),ex))
    specs += [Spec("lgbm_resid_s045","lgbm_resid","lgbm_meta",dict(shrink=.45),ex),Spec("lgbm_resid_s075","lgbm_resid","lgbm_meta",dict(shrink=.75),ex)]
    specs += [Spec("effective_q_s060","q","effective_q",dict(shrink=.60),ex),Spec("effective_q_s100","q","effective_q",dict(shrink=1.0),ex)]
    specs += [Spec("simplex_anchor","simplex","simplex",dict(lam=.06,min_new=0.,new=new),ex),Spec("simplex_free","simplex","simplex",dict(lam=.0,min_new=0.,new=new),ex)]
    if new:
        specs += [Spec("simplex_new15","simplex","simplex_new",dict(lam=.05,min_new=.15,new=new),ex),Spec("simplex_new25","simplex","simplex_new",dict(lam=.04,min_new=.25,new=new),ex)]
    specs += [Spec("greedy25","greedy","greedy",dict(steps=25),ex),Spec("pband_soft","pband","pband",{},ex),Spec("occ_platt_guard","occ","occ_cal",dict(guard=True),ex)]
    return specs

def eval_spec(spec,bank):
    if spec.kind=="ridge_resid":p,_=walk_residual(bank,list(spec.experts),"ridge",spec.params["alpha"],spec.params["shrink"],spec.params["meta"],4200)
    elif spec.kind=="lgbm_resid":p,_=walk_residual(bank,list(spec.experts),"lgbm",100,spec.params["shrink"],True,4300)
    elif spec.kind=="q":p=walk_q(bank,list(spec.experts),spec.params["shrink"],4400)
    elif spec.kind=="simplex":p,_=walk_simplex(bank,list(spec.experts),spec.params["lam"],spec.params["min_new"],spec.params["new"])
    elif spec.kind=="greedy":p,_=walk_greedy(bank,list(spec.experts),spec.params["steps"])
    elif spec.kind=="pband":p,_=walk_pband(bank,list(spec.experts))
    elif spec.kind=="occ":p=walk_occ(bank,spec.params["guard"])
    else:raise KeyError(spec.kind)
    return p

def final_spec(spec,bank,test):
    if spec.kind=="ridge_resid":return final_residual(bank,test,list(spec.experts),"ridge",spec.params["alpha"],spec.params["shrink"],spec.params["meta"],4200),{}
    if spec.kind=="lgbm_resid":return final_residual(bank,test,list(spec.experts),"lgbm",100,spec.params["shrink"],True,4300),{}
    if spec.kind=="q":return final_q(bank,test,list(spec.experts),spec.params["shrink"],4400),{}
    if spec.kind=="simplex":
        z,w=final_simplex(bank,test,list(spec.experts),spec.params["lam"],spec.params["min_new"],spec.params["new"]);return z,{"weights":dict(zip(spec.experts,w.tolist()))}
    if spec.kind=="greedy":
        z,w=final_greedy(bank,test,list(spec.experts),spec.params["steps"]);return z,{"weights":dict(zip(spec.experts,w.tolist()))}
    if spec.kind=="pband":
        z,W=final_pband(bank,test,list(spec.experts));return z,{"band_weights":W.tolist()}
    if spec.kind=="occ":return final_occ(bank,test,spec.params["guard"]),{}
    raise KeyError(spec.kind)

def research(bank,results_dir,tag="final"):
    rows=[];preds={};specmap={}
    # raw experts themselves are also candidates.
    for n in available_experts(bank):
        if n in ("cap","unc","dist","hurdle"):continue
        p={f:bank[f][n] for f in FOLDS};score_table(n,p,bank,"raw_new",rows);preds[n]=p;specmap[n]=Spec(n,"raw","raw_new",{},(n,))
    for s in make_specs(bank):
        try:
            p=eval_spec(s,bank);score_table(s.name,p,bank,s.family,rows);preds[s.name]=p;specmap[s.name]=s
        except Exception as e:
            append_error(results_dir/"errors.jsonl","research",s.name,"all",e);log("SPEC FAILED",s.name,repr(e))
    # fixed consensus of independent mechanisms only if both held-out predictions exist.
    defs=[("consensus_ridge_q","ridge_meta_a150_s075","effective_q_s060"),
          ("consensus_ridge_pband","ridge_meta_a150_s075","pband_soft"),
          ("consensus_lgbm_greedy","lgbm_resid_s045","greedy25")]
    for name,a,b in defs:
        if a in preds and b in preds:
            p={f:.5*preds[a][f]+.5*preds[b][f] for f in FOLDS};score_table(name,p,bank,"consensus",rows,notes=f"0.5*{a}+0.5*{b}");preds[name]=p
            specmap[name]=Spec(name,"consensus","consensus",dict(a=a,b=b),(a,b))
    rows.sort(key=lambda r:(r["delta"],r["worst_delta"]));save_csv(results_dir/f"candidate_validation_{tag}.csv",rows)
    return rows,preds,specmap

# -----------------------------------------------------------------------------
# test bank / candidate finalization
# -----------------------------------------------------------------------------
def component_test(package,name):
    p=package/"artifacts"/"predictions";return np.load(p/f"uid_{name}.npy"),np.load(p/f"ztest_{name}.npy")

def build_test(prev,ctx,friend,needed_raw):
    uid=np.asarray(friend["uid"],np.int64);r={"uid":uid,"friend":np.asarray(friend["z"],np.float64)}
    for n,src in (("cap","S1-CAP"),("unc","S1-UNC"),("dist","S1-DIST")):
        u,z=component_test(ctx.package,src);r[n]=align(u,z,uid).astype(np.float64)
    r["table_core"]=sum(CORE_TABLE_WEIGHTS[n]*r[n] for n in CORE_TABLE_WEIGHTS)
    hp=Path(ctx.checkpoints)/"test"/"hurdle_test.npz"
    if not valid_npz(hp,("user_id","z","p","mu")):
        log("Existing final hurdle missing; training ONLY this non-friend helper on all users")
        d=prev.train_table_test_hurdle(ctx)
    else:d=load_npz(hp)
    r["hurdle"]=align(d["user_id"],d["z"],uid);r["p"]=np.clip(align(d["user_id"],d["p"],uid),EPS,1-EPS);r["mu"]=np.maximum(align(d["user_id"],d["mu"],uid),0)
    u,X,names=prev.build_test_meta_raw(ctx);r["meta_raw"]=align(u,X,uid).astype(np.float32);r["meta_names"]=names
    for n in needed_raw:
        d=train_expert_test(ctx,n);r[n]=align(d["user_id"],d["z"],uid)
        if n=="recent_hurdle":r["p_recent_hurdle"]=align(d["user_id"],d["p"],uid);r["mu_recent_hurdle"]=align(d["user_id"],d["mu"],uid)
    return r

def final_candidate_table(name,specmap,bank,test,cache):
    if name in cache:return cache[name]
    s=specmap[name]
    if s.kind=="raw":z=test[s.experts[0]];meta={}
    elif s.kind=="consensus":
        a=final_candidate_table(s.params["a"],specmap,bank,test,cache)[0];b=final_candidate_table(s.params["b"],specmap,bank,test,cache)[0];z=.5*a+.5*b;meta={"parts":[s.params["a"],s.params["b"]]}
    else:z,meta=final_spec(s,bank,test)
    cache[name]=(clipz(z),meta);return cache[name]

def transform_to_friend(friend_z,core_test,candidate_table,beta=1.0):
    z=np.asarray(friend_z,np.float64)+TABLE_WEIGHT*float(beta)*(np.asarray(candidate_table)-np.asarray(core_test));return level_test(z,LEVEL)[0]

def regime_metrics(oof,bank,test_table,core_test,final_z,friend_z):
    oo=[]
    for f in FOLDS:
        d=np.asarray(oof[f])-bank[f]["table_core"];d=d-d.mean();oo.append(d)
    oo=np.concatenate(oo);td=np.asarray(test_table)-core_test;td=td-td.mean();vo=float(np.var(oo));vt=float(np.var(td));ratio=vt/max(vo,1e-12)
    dd=candidate_distance(final_z,friend_z)
    return dict(oof_table_var=vo,test_table_var=vt,var_ratio=ratio,friend_corr=dd["corr"],friend_std_dz=dd["std"],friend_mean_abs_dz=dd["mean_abs"],friend_pct05=dd["pct05"],friend_pct10=dd["pct10"])

def locate_previous_submissions(base,friend_uid):
    import pandas as pd
    patterns=["*continue12h*safe*csv","*continue12h*class1_occ*csv"]
    out={}
    for pat in patterns:
        for p in base.rglob(pat):
            try:
                d=pd.read_csv(p)
                if len(d)!=len(friend_uid) or not {"user_id","predict"}.issubset(d.columns):continue
                z=np.log1p(np.maximum(d.predict.to_numpy(np.float64),0));out[p.stem]=align(d.user_id.to_numpy(np.int64),z,friend_uid);break
            except Exception:continue
    return out

def candidate_rank(rows):
    # Quality first; penalize temporal inconsistency / positive latest fold / OOD variance.
    for r in rows:
        r["selection_score"]=r["delta"]+0.65*max(r["latest_delta"],0)+0.35*max(r["worst_delta"],0)+0.0015*abs(math.log(max(r.get("var_ratio",1),1e-4)))
    return sorted(rows,key=lambda r:(r["selection_score"],r["delta"]))

def choose_output(rows,testmap,friend,oldsubs,n=7):
    # Prefer actual table improvement: first fold of meta models is baseline, so 2/3 recent wins is meaningful.
    good=[r for r in rows if r["delta"]<-.00035 and r["latest_delta"]<=.00015 and r["wins_recent"]>=2 and .35<=r["var_ratio"]<=2.1 and
          (r["friend_std_dz"]>=.010 or r["friend_pct05"]>=.035)]
    ranked=candidate_rank(good);chosen=[];family_count={}
    for r in ranked:
        if len(chosen)>=n:break
        if family_count.get(r["family"],0)>=2:continue
        z=testmap[r["name"]]
        # Explicitly avoid reproducing the two already-tested bad directions.
        if oldsubs and any(candidate_distance(z,v)["corr"]>.9993 for v in oldsubs.values()):continue
        if chosen and not all((candidate_distance(z,testmap[q["name"]])["std"]>=.008 or candidate_distance(z,testmap[q["name"]])["pct05"]>=.025) for q in chosen):continue
        chosen.append(r);family_count[r["family"]]=family_count.get(r["family"],0)+1
    # Controlled fallback: near-ties only, never another over_guard-like disaster.
    if len(chosen)<min(5,n):
        near=candidate_rank([r for r in rows if r not in chosen and r["delta"]<=.0010 and r["latest_delta"]<=.0015 and r["wins_recent"]>=1 and .28<=r["var_ratio"]<=2.6 and (r["friend_std_dz"]>=.007 or r["friend_pct05"]>=.02)])
        for r in near:
            if len(chosen)>=n:break
            z=testmap[r["name"]]
            if all(candidate_distance(z,testmap[q["name"]])["std"]>=.006 for q in chosen):chosen.append(r)
    return chosen[:n]

# -----------------------------------------------------------------------------
# leaderboard diagnostics — do NOT optimize candidates on public LB
# -----------------------------------------------------------------------------
def leaderboard_geometry(friend_z,oldsubs):
    rows=[];base=KNOWN_LB["STRONGEST_CURRENT"]
    for stem,z in oldsubs.items():
        key="ranker_safe" if "safe" in stem.lower() else "class1_occ" if "occ" in stem.lower() else None
        if not key:continue
        d=np.asarray(z)-friend_z;v=float(np.mean(d*d));score=KNOWN_LB[key]
        # From L(a)=L0^2+2 a <e,d>+a^2<d,d>, infer derivative at a=0 using a=1 observation.
        cross=(score**2-base**2-v)/2;alpha=float(np.clip(-cross/max(v,1e-12),-1,1));best2=base**2-(cross*cross/max(v,1e-12)) if v>0 else base**2
        rows.append(dict(direction=key,lb_base=base,lb_full=score,full_d2=v,inferred_opt_alpha=alpha,inferred_best_lb=float(math.sqrt(max(best2,0)))))
    return rows

# -----------------------------------------------------------------------------
# self test
# -----------------------------------------------------------------------------
def self_test():
    rng=np.random.default_rng(7);bank={}
    for j,f in enumerate(FOLDS):
        n=2500;X=rng.normal(size=(n,24)).astype(np.float32);p=sigmoid(.6*X[:,0]-.3*X[:,1]);mu=np.maximum(1.4+.5*X[:,2],.1);tz=np.maximum(p*mu+.25*X[:,3]+rng.normal(scale=.75,size=n),0);y=np.expm1(tz)
        cap=np.maximum(tz+rng.normal(scale=.42,size=n),0);unc=np.maximum(tz+rng.normal(scale=.39,size=n),0);dist=np.maximum(tz+rng.normal(scale=.37,size=n),0);hur=np.maximum(p*mu+rng.normal(scale=.36,size=n),0)
        rec={"uid":np.arange(n)+j*10000,"y":y,"true_z":tz,"meta_raw":X,"meta_names":[f"m{k}" for k in range(X.shape[1])],"p":p,"mu":mu,"cap":cap,"unc":unc,"dist":dist,"hurdle":hur}
        rec["table_core"]=sum(CORE_TABLE_WEIGHTS[k]*rec[k] for k in CORE_TABLE_WEIGHTS);bank[f]=rec
    rows,preds,specmap=research(bank,Path("/tmp/fixedstack_selftest"),"selftest")
    test={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in bank[FOLDS[-1]].items()};test["friend"]=test["table_core"]+.1*rng.normal(size=len(test["uid"]));cache={}
    for r in rows[:5]:
        z,_=final_candidate_table(r["name"],specmap,bank,test,cache);assert len(z)==len(test["uid"]) and np.isfinite(z).all()
    print("SELF-TEST OK",len(rows),"candidates",flush=True)

# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--max-hours",type=float,default=14.0);ap.add_argument("--threads",type=int,default=max(4,min(14,os.cpu_count() or 8)))
    ap.add_argument("--reuse-work-dir",type=str,default=None);ap.add_argument("--no-install",action="store_true");ap.add_argument("--preflight-only",action="store_true");ap.add_argument("--self-test",action="store_true")
    args=ap.parse_args()
    if args.self_test:self_test();return
    started=time.time();budget=Budget(started,args.max_hours);base=Path(__file__).resolve().parent;prev,prev_path=import_prev(base);package=prev.discover_package(base);raw,sample=prev.discover_raw_and_sample(base,package);prev.ensure_dependencies(package,args.no_install)
    work=discover_work(base,args.reuse_work_dir);out=base/"_best_bas_fixedstack_14h";results=out/"results";subs=out/"submissions";results.mkdir(parents=True,exist_ok=True);subs.mkdir(parents=True,exist_ok=True)
    ctx=prev.Context(base_dir=base,package=package,pipeline=package/"pipeline",raw=raw,sample=sample,work=work,results=results,submissions=subs,checkpoints=work/"checkpoints",budget=prev.Budget(started,args.max_hours,max(args.max_hours-2.5,0),1.5))
    prev.configure_pipeline(ctx,args.threads);repair=repair_cache(ctx);install_atomic_cache(ctx);friend=prev.verify_friend_package(package);errors=results/"errors.jsonl"
    log("FIXEDSTACK",SCRIPT_VERSION);log("FRIEND exact rebuild error",friend.get("max_log_error"));log("No SEQ/ETX friend model will be retrained.")
    bank=load_core(prev,ctx);base_rows=[];score_table("table_core",{f:bank[f]["table_core"] for f in FOLDS},bank,"baseline",base_rows);score_table("hurdle",{f:bank[f]["hurdle"] for f in FOLDS},bank,"existing",base_rows);save_csv(results/"core_validation.csv",base_rows)
    manifest=dict(version=SCRIPT_VERSION,started=now(),previous_runner=str(prev_path),package=str(package),work=str(work),raw=str(raw),sample=str(sample),repair=repair,friend_sha=friend.get("ref_sha256"),friend_rebuild_error=friend.get("max_log_error"),known_lb=KNOWN_LB,args=vars(args))
    atomic_json(results/"RUN_START.json",manifest)
    if args.preflight_only:
        log("PREFLIGHT OK: fixed base OOF exists, friend reconstruction exact, no base retraining");return

    # Stage 0: cheap research before spending hours on new models.
    log("STAGE 0: existing-artifact stack/q/pband research")
    rows,preds,specmap=research(bank,results,"stage0")
    for r in rows[:12]:log(" ",r["name"],f"delta={r['delta']:+.6f}","recent wins",r["wins_recent"],"latest",f"{r['latest_delta']:+.6f}")

    # Stage 1: expensive NEW families, all users.  Re-run cheap research after each.
    plan=[("recent_hurdle",1.3),("multiscale_direct",2.0),("recent_direct",2.3),("recent_dist",2.7)]
    runtime=[];completed=[]
    for name,est in plan:
        # Need room for final refit(s), final meta research and file writing.
        if not budget.can_start(est,extra=1.25):log("SKIP NEW FAMILY by budget",name,"remaining",f"{budget.remaining:.2f}h");continue
        t=time.time();ok=True
        try:
            add_expert_to_bank(ctx,bank,name);completed.append(name)
        except Exception as e:
            ok=False;append_error(errors,"new_expert",name,"all",e);log("NEW FAMILY FAILED",name,repr(e))
        runtime.append(dict(stage="new_family",name=name,hours=(time.time()-t)/3600,ok=ok,remaining=budget.remaining))
        save_csv(results/"runtime.csv",runtime)
        if ok:
            log("RESEARCH REFRESH after",name);rows,preds,specmap=research(bank,results,f"after_{name}")
            top=rows[:8]
            for r in top:log(" ",r["name"],f"delta={r['delta']:+.6f}","wins_recent",r["wins_recent"],"latest",f"{r['latest_delta']:+.6f}")

    # Final research with all completed full-user experts.
    rows,preds,specmap=research(bank,results,"final")
    save_csv(results/"runtime.csv",runtime)
    log("TOP TABLE VALIDATION")
    for r in rows[:20]:log(f" {r['name']:28s} d={r['delta']:+.6f} wins={r['wins']}/4 recent={r['wins_recent']}/3 latest={r['latest_delta']:+.6f} family={r['family']}")

    # Pre-shortlist recipes before expensive final refits.  Keep enough diversity.
    pre=[r for r in rows if r["delta"]<.0005 and r["latest_delta"]<.0008]
    pre=candidate_rank(pre);pre_names=[];families={}
    for r in pre:
        if len(pre_names)>=14:break
        if families.get(r["family"],0)>=3:continue
        pre_names.append(r["name"]);families[r["family"]]=families.get(r["family"],0)+1
    # Determine raw dependencies actually needed by these recipes.  Because most stack specs contain all available experts,
    # final fitting needs each completed raw expert. If budget is low we drop raw-dependent specs rather than half-refit.
    raw_needed=set()
    for n in pre_names:
        s=specmap[n]
        if s.kind=="raw":raw_needed.add(s.experts[0])
        elif s.kind not in ("consensus",):raw_needed.update(x for x in s.experts if x in completed)
        else:
            for part in (s.params["a"],s.params["b"]):
                ps=specmap[part];raw_needed.update(x for x in ps.experts if x in completed)
    # If remaining budget is tight, restrict candidate recipes to existing-only bank rather than risk no submissions.
    if raw_needed and budget.remaining < 2.3:
        log("LOW FINAL BUDGET -> restricting to existing-only recipes");raw_needed.clear();pre_names=[n for n in pre_names if not any(x in completed for x in specmap[n].experts)][:12]

    log("FINAL TEST BUILD; raw experts needed",sorted(raw_needed),"remaining",f"{budget.remaining:.2f}h")
    test=build_test(prev,ctx,friend,sorted(raw_needed));oldsubs=locate_previous_submissions(base,np.asarray(friend["uid"],np.int64));save_csv(results/"leaderboard_direction_diagnostics.csv",leaderboard_geometry(np.asarray(friend["z"],np.float64),oldsubs))

    # Finalize pre-candidates. If a recipe references a raw expert that was not refit, skip it.
    cache={};testmap={};finalrows=[];recipe_rows=[]
    for r in rows:
        if r["name"] not in pre_names:continue
        s=specmap[r["name"]]
        deps=set()
        if s.kind=="raw":deps.add(s.experts[0])
        elif s.kind=="consensus":
            for part in (s.params["a"],s.params["b"]):deps.update(x for x in specmap[part].experts if x in completed)
        else:deps.update(x for x in s.experts if x in completed)
        if any(x not in test for x in deps):continue
        try:
            table_z,meta=final_candidate_table(r["name"],specmap,bank,test,cache);final_z=transform_to_friend(friend["z"],test["table_core"],table_z,1.0);reg=regime_metrics(preds[r["name"]],bank,table_z,test["table_core"],final_z,friend["z"])
            rr=dict(r);rr.update(reg);finalrows.append(rr);testmap[r["name"]]=final_z;recipe_rows.append(dict(name=r["name"],kind=s.kind,family=s.family,params=s.params,meta=meta))
        except Exception as e:append_error(errors,"finalize",r["name"],"test",e);log("FINALIZE FAILED",r["name"],repr(e))
    finalrows=candidate_rank(finalrows);save_csv(results/"candidate_recipes.csv",recipe_rows)

    # Always construct MATERIAL table-slot-strength variants of the best genuinely
    # improving recipes.  beta=.72/.85 means replacing 39.6%/46.75% of the final
    # prediction through the 55% table slot — this is not a cosmetic 95/5 blend.
    # Their validation is recomputed explicitly; no score is copied from beta=1.
    seed_rows=[r for r in finalrows if r["delta"]<0 and r["wins_recent"]>=2][:4]
    for r in seed_rows:
        name=r["name"];table_z,_=final_candidate_table(name,specmap,bank,test,cache)
        for beta in (.72,.85):
            nn=f"{name}__slotbeta{int(beta*100)}"
            if nn in testmap:continue
            op={f:clipz(bank[f]["table_core"]+beta*(preds[name][f]-bank[f]["table_core"])) for f in FOLDS}
            tmp=[];rr=score_table(nn,op,bank,r["family"]+"_slotstrength",tmp,notes=f"beta={beta} of validated table replacement")
            table_beta=clipz(test["table_core"]+beta*(table_z-test["table_core"]));z=transform_to_friend(friend["z"],test["table_core"],table_beta,1.0)
            reg=regime_metrics(op,bank,table_beta,test["table_core"],z,friend["z"]);rr.update(reg);finalrows.append(rr);testmap[nn]=z
    finalrows=candidate_rank(finalrows);save_csv(results/"candidate_summary.csv",finalrows)

    chosen=choose_output(finalrows,testmap,np.asarray(friend["z"],np.float64),oldsubs,n=7)
    if len(chosen)<5:
        log("WARNING only",len(chosen),"quality/diversity candidates passed. Will emit available safe set, not knowingly bad files.")
    import pandas as pd
    sample_df=pd.read_csv(sample);sample_uid=sample_df["user_id"].to_numpy(np.int64) if "user_id" in sample_df else np.asarray(friend["uid"],np.int64)
    selection=[]
    for i,r in enumerate(chosen,1):
        z=align(friend["uid"],testmap[r["name"]],sample_uid);pred=np.maximum(np.expm1(np.clip(z,0,20)),0);df=pd.DataFrame({"user_id":sample_uid,"predict":pred})
        if len(df)!=250000 or df.user_id.duplicated().any() or df.predict.isna().any() or (df.predict<0).any():raise RuntimeError("bad submission")
        path=subs/f"submission_fixedstack_candidate_{i}_{r['name']}.csv";df.to_csv(path,index=False)
        rr=dict(rank=i,name=r["name"],file=str(path),sha256=sha256(path),delta_table=r["delta"],latest_delta=r["latest_delta"],wins_recent=r["wins_recent"],family=r["family"],friend_corr=r["friend_corr"],friend_std_dz=r["friend_std_dz"],friend_pct05=r["friend_pct05"],var_ratio=r["var_ratio"]);selection.append(rr);log("CANDIDATE",i,r["name"],"table d",f"{r['delta']:+.6f}","corr friend",f"{r['friend_corr']:.6f}")
    save_csv(results/"FINAL_CANDIDATES.csv",selection)
    # diversity matrix incl previous submissions
    div=[];allz={"STRONGEST_CURRENT":np.asarray(friend["z"],np.float64),**{r["name"]:testmap[r["name"]] for r in chosen},**{f"OLD_{k}":v for k,v in oldsubs.items()}}
    ks=list(allz)
    for i in range(len(ks)):
        for j in range(i+1,len(ks)):
            div.append(dict(a=ks[i],b=ks[j],**candidate_distance(allz[ks[i]],allz[ks[j]])))
    save_csv(results/"DIVERSITY.csv",div)
    runtime_hours=(time.time()-started)/3600;manifest.update(dict(finished=now(),runtime_hours=runtime_hours,completed_new_experts=completed,raw_final_refits=sorted(raw_needed),selection=selection,remaining_hours=budget.remaining));atomic_json(results/"RUN_MANIFEST.json",manifest)
    report=["E-CUP fixed champion + full-user stack 14h",f"runtime_hours={runtime_hours:.3f}","STRONGEST_CURRENT was NEVER retrained.",f"completed_new_experts={completed}","", "Top table-validation candidates:"]
    for r in finalrows[:25]:report.append(f"{r['name']:32s} delta={r['delta']:+.6f} recent={r['wins_recent']}/3 latest={r['latest_delta']:+.6f} var_ratio={r.get('var_ratio',float('nan')):.3f} friend_corr={r.get('friend_corr',float('nan')):.6f}")
    report += ["","Emitted candidates:"]+[f"{x['rank']}. {x['name']} table_delta={x['delta_table']:+.6f} corr_friend={x['friend_corr']:.6f} file={x['file']}" for x in selection]
    (results/"REPORT_RU.txt").write_text("\n".join(report),encoding="utf-8")
    bundle=base/f"fixedstack_14h_REVIEW_BUNDLE_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    with zipfile.ZipFile(bundle,"w",zipfile.ZIP_DEFLATED) as zf:
        for p in results.iterdir():
            if p.is_file() and p.suffix.lower() in {".csv",".json",".txt",".jsonl"}:zf.write(p,arcname=f"results/{p.name}")
        for p in subs.glob("*.csv"):zf.write(p,arcname=f"submissions/{p.name}")
    log("DONE",f"{runtime_hours:.2f}h","candidates",len(selection));log("REPORT",results/"REPORT_RU.txt");log("BUNDLE",bundle)

if __name__=="__main__":main()
