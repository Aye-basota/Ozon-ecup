# -----------------------------------------------------------------------------

## Catalogue metadata

- **Catalogue ID:** `teammate_research__continue_best_bas_final6h`
- **Namespace:** `teammate_research`
- **Experiment ID:** `continue_best_bas_final6h`
- **Original source:** `пайплайн сокомандника/research_scripts/continue_best_bas_final6h.py`
- **Source ref:** `working tree`
- **Source commit:** `a28a71fb2d0194052014c542f36d180dfe74bcf9`
- **Kind:** teammate research runner
- **Model:** LightGBM, Ridge, two-part / hurdle, ensemble, blend
- **Features:** recency, freshness/conditional features, occurrence features, history-depth features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** Validation
- **Known score:** Unknown / not recoverable from repository history
- **Seed:** max_bin=127,random_state=seed,n_jobs=max(2,min(10,os.cpu_count() or 8)),verbosity=-1)
- **Postprocessing:** # Candidate-level Ridge over already cross-fitted candidate predictions.
- **Submission:** for p in subs.glob("submission_final6h_*.csv"):zf.write(p,arcname=f"submissions/{p.name}")
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when the inputs/checkpoints named by the preserved runner are available

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# continue_best_bas_final6h

Original script: `пайплайн сокомандника/research_scripts/continue_best_bas_final6h.py`

```python
from __future__ import annotations

"""E-CUP 2026 / Track 3 — final 6h continuation over fixed STRONGEST_CURRENT.

Purpose
-------
This continuation never retrains the teammate's STRONGEST_CURRENT ensemble.
It reuses all saved CAP/UNC/DIST/Hurdle/new-expert OOF/test artifacts, finishes
missing memory-safe final pieces, then spends the remaining budget on NEW
occurrence-only all-user models and on combinations of historically successful
ideas:

A) stable stacking branch
   * temporal Ridge residual stack (recent folds upweighted),
   * candidate-level greedy / p-band / local-bias ensembles,
   * second-level super-Ridge over already cross-fitted candidates,
   * stable18 hurdle finalization in a clean sequential two-part process.

B) occurrence-specialist branch
   * several all-user recent occurrence-only LightGBM variants (different
     recency windows / capacity / feature views),
   * temporal calibration of their probabilities,
   * meta-occurrence stacking,
   * false-one / severe-over risk used only as a continuous TRUST GATE,
     never as an additive magnitude correction,
   * occurrence correction over the best stable table stack while preserving
     the strong positive-value magnitude signal.

Validation
----------
Uses the same four clean teammate folds as the fixedstack runs:
2025-09-04, 2025-09-18, 2025-10-02, 2025-10-16 with 1:2:4:8 weighting.
Meta models are walk-forward: target fold i uses only folds < i.
Raw occurrence models are trained from historical cutoffs preceding each target
fold and therefore can produce honest predictions on all four folds.

Runtime
-------
Default wall budget: 6h. Heavy raw training always occurs in fresh child
processes. The parent keeps at least ~0.65h for finalization and file writing.
If the first experiments finish unusually quickly, the script keeps launching
additional occurrence variants from a prioritized queue instead of terminating
prematurely. No sleeps are used.

Expected location
-----------------
Place beside:
  run_best_bas_fixedstack_14h_v2.py
  continue_fixedstack_combo_10h.py
in src/DL/best_bas/ and run from repository root.
"""

import argparse
import dataclasses
import datetime as dt
import gc
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np

SCRIPT_VERSION = "final6h_fixedfriend_2026-08-23_001"
FOLDS = ("2025-09-04", "2025-09-18", "2025-10-02", "2025-10-16")
FW = np.asarray([1.0, 2.0, 4.0, 8.0], dtype=np.float64)
EPS = 1e-7
TABLE_WEIGHT = 0.55
RESERVE_HOURS = 0.65
KNOWN_FRIEND_PUBLIC = 1.6496571
KNOWN_RIDGE_SUB_PUBLIC = 1.6492897556391737


def now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def log(*x: Any) -> None:
    print(f"[{now()}]", *x, flush=True)


def jdefault(x: Any):
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    if dataclasses.is_dataclass(x):
        return dataclasses.asdict(x)
    raise TypeError(type(x).__name__)


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=jdefault), encoding="utf-8")
    os.replace(tmp, path)


def save_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    import pandas as pd
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_csv(path, index=False)


def append_error(path: Path, stage: str, name: str, fold: str, exc: BaseException) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "time": now(), "stage": stage, "name": name, "fold": fold,
        "error": repr(exc), "traceback": traceback.format_exc(),
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def discover_parent_scripts(base: Path):
    combo_candidates = [
        base / "continue_fixedstack_combo_10h.py",
        base / "continue_fixedstack_combo_10h_v2.py",
    ]
    combo_path = next((p for p in combo_candidates if p.exists()), None)
    if combo_path is None:
        raise FileNotFoundError("Не найден continue_fixedstack_combo_10h.py рядом с новым файлом")
    combo = import_module(combo_path, "combo10_parent_final6h")
    fixed, fixed_path = combo.import_fixed(base)
    prev, prev_path = fixed.import_prev(base)
    return combo, combo_path, fixed, fixed_path, prev, prev_path


def clipz(z):
    return np.maximum(np.asarray(z, np.float64), 0.0)


def logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1-1e-6)
    return np.log(p / (1-p))


def sigmoid(x):
    x = np.asarray(x, np.float64)
    out = np.empty_like(x)
    m = x >= 0
    out[m] = 1.0 / (1.0 + np.exp(-x[m]))
    ex = np.exp(x[~m])
    out[~m] = ex / (1.0 + ex)
    return out


def wavg(a):
    a = np.asarray(a, np.float64)
    return float(np.dot(a, FW) / FW.sum())


def align(src_uid, arr, dst_uid):
    src_uid = np.asarray(src_uid, np.int64)
    dst_uid = np.asarray(dst_uid, np.int64)
    arr = np.asarray(arr)
    if np.array_equal(src_uid, dst_uid):
        return arr
    order = np.argsort(src_uid)
    pos = np.searchsorted(src_uid[order], dst_uid)
    if np.any(pos >= len(src_uid)) or not np.array_equal(src_uid[order][pos], dst_uid):
        raise ValueError("user_id alignment mismatch")
    return arr[order][pos]


class Budget:
    def __init__(self, started: float, max_hours: float, reserve: float = RESERVE_HOURS):
        self.started = started
        self.max_hours = float(max_hours)
        self.reserve = float(reserve)

    @property
    def elapsed(self) -> float:
        return (time.time() - self.started) / 3600.0

    @property
    def remaining(self) -> float:
        return self.max_hours - self.elapsed

    def can_start(self, estimate: float, extra: float = 0.0) -> bool:
        return self.remaining > self.reserve + extra + estimate


# -----------------------------------------------------------------------------
# raw all-user occurrence family
# -----------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class OccCfg:
    name: str
    maxcuts: int
    tau: float
    rounds: int
    leaves: int
    min_leaf: int
    feature_mode: str = "all"  # all | multiscale
    feature_fraction: float = .82


# Priority: first configs represent genuinely different recency/capacity choices.
# Extra configs keep the run productive if the machine trains faster than expected.
OCC_QUEUE = [
    OccCfg("occ_r10_fast", 10, 55.0, 380, 31, 520, "all", .82),
    OccCfg("occ_r16_bal", 16, 100.0, 440, 47, 430, "all", .84),
    OccCfg("occ_r22_stable", 22, 180.0, 500, 31, 650, "all", .80),
    OccCfg("occ_r14_multiscale", 14, 85.0, 430, 47, 430, "multiscale", .90),
    OccCfg("occ_r18_wide", 18, 125.0, 470, 63, 420, "all", .76),
    OccCfg("occ_r24_multiscale", 24, 220.0, 520, 31, 700, "multiscale", .88),
    OccCfg("occ_r12_wide", 12, 70.0, 430, 79, 380, "all", .72),
    OccCfg("occ_r20_shallow", 20, 155.0, 500, 23, 760, "all", .90),
]
OCC_MAP = {x.name: x for x in OCC_QUEUE}


def occ_fold_path(ctx, name: str, fold: str) -> Path:
    return Path(ctx.checkpoints) / "folds" / f"{name}__{fold}.npz"


def occ_test_path(ctx, name: str) -> Path:
    return Path(ctx.checkpoints) / "test" / f"{name}_test.npz"


def valid_npz(path: Path, need=("user_id",)) -> bool:
    if not path.exists() or path.stat().st_size < 256:
        return False
    try:
        with np.load(path, allow_pickle=False) as d:
            if not set(need).issubset(d.files):
                return False
            n = len(d["user_id"])
            if n == 0:
                return False
            for k in need:
                if len(d[k]) != n:
                    return False
                if k != "user_id" and not np.isfinite(d[k]).all():
                    return False
        return True
    except Exception:
        return False


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def save_npz_atomic(path: Path, **kw) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}.npz")
    np.savez_compressed(tmp, **kw)
    with np.load(tmp, allow_pickle=False) as d:
        _ = d[d.files[0]]
    os.replace(tmp, path)


def configure_child_context(base: Path, args):
    combo, _, fixed, _, prev, _ = discover_parent_scripts(base)
    package = prev.discover_package(base)
    raw, sample = prev.discover_raw_and_sample(base, package)
    prev.ensure_dependencies(package, args.no_install)
    work = fixed.discover_work(base, args.reuse_work_dir)
    out = base / "_best_bas_final6h"
    results = out / "results"; subs = out / "submissions"
    results.mkdir(parents=True, exist_ok=True); subs.mkdir(parents=True, exist_ok=True)
    started = time.time()
    ctx = prev.Context(base_dir=base, package=package, pipeline=package/"pipeline", raw=raw, sample=sample,
                       work=work, results=results, submissions=subs, checkpoints=work/"checkpoints",
                       budget=prev.Budget(started, 99, 98, 1))
    prev.configure_pipeline(ctx, max(2, int(args.threads)))
    fixed.repair_cache(ctx); fixed.install_atomic_cache(ctx)
    return combo, fixed, prev, ctx


def _select_occ_features(fixed, feats: list[str], mode: str) -> list[str]:
    if mode == "multiscale":
        xs = fixed.multiscale_features(feats)
        # keep enough signal even if helper returns a narrow view
        return xs if len(xs) >= 40 else feats
    return feats


def _lgb_binary_params(ctx, cfg: OccCfg):
    M = ctx.models_mod
    # Reuse teammate defaults and only change capacity / binary objective.
    base = dict(num_leaves=cfg.leaves, min_data_in_leaf=cfg.min_leaf,
                feature_fraction=cfg.feature_fraction, bagging_fraction=.90,
                bagging_freq=1, lambda_l2=14.0, lambda_l1=1.0, max_bin=127,
                learning_rate=.035, verbosity=-1, num_threads=max(2, int(getattr(ctx, "threads", 6) if hasattr(ctx, "threads") else 6)) )
    try:
        return M._params(base, objective="binary", metric="binary_logloss")
    except Exception:
        p = dict(objective="binary", metric="binary_logloss", learning_rate=.035,
                 num_leaves=cfg.leaves, min_data_in_leaf=cfg.min_leaf,
                 feature_fraction=cfg.feature_fraction, bagging_fraction=.90, bagging_freq=1,
                 lambda_l2=14.0, lambda_l1=1.0, max_bin=127, verbosity=-1,
                 num_threads=max(2, min(10, os.cpu_count() or 6)), seed=42)
        return p


def train_occ_child(base: Path, args, name: str, fold: str) -> None:
    combo, fixed, prev, ctx = configure_child_context(base, args)
    if name not in OCC_MAP:
        raise KeyError(name)
    cfg = OCC_MAP[name]
    T, F = ctx.train_mod, ctx.features_mod
    s, _ = fixed.expert_spec(ctx, "recent_hurdle")
    s.weight_tau = cfg.tau
    C = ctx.config
    is_test = fold == "TEST"
    if is_test:
        path = occ_test_path(ctx, name)
        if valid_npz(path, ("user_id", "p")):
            log("CHILD OCC reuse", name, "TEST"); return
        Xv, _ = F.make_xy(C.CUTOFF_TEST, s.L, s.panel_blocks, with_target=False, norm_long=s.norm_long)
        feats = T.select_features(F.feature_names(Xv), s.drop_groups, s.keep_only)
        feats = _select_occ_features(fixed, feats, cfg.feature_mode)
        cuts = list(s.grid())[-cfg.maxcuts:]
        V = C.CUTOFF_TEST
    else:
        path = occ_fold_path(ctx, name, fold)
        if valid_npz(path, ("user_id", "y", "p")):
            log("CHILD OCC reuse", name, fold); return
        V = dt.date.fromisoformat(fold)
        cuts = list(s.train_cutoffs(V))[-cfg.maxcuts:]
        Xv, yv = T.xy(V, s)
        feats = T.select_features(F.feature_names(Xv), s.drop_groups, s.keep_only)
        feats = _select_occ_features(fixed, feats, cfg.feature_mode)
    log("CHILD OCC", name, fold, "cuts", len(cuts), "features", len(feats))
    X, y, w = T.assemble(cuts, s, feats, V)
    # Construct one LightGBM Dataset, then drop the dense multi-million-row X.
    import lightgbm as lgb
    params = _lgb_binary_params(ctx, cfg)
    ds = lgb.Dataset(X, (y > 0).astype(np.int8), weight=w, params=params, free_raw_data=True).construct()
    del X
    T._XY.clear(); gc.collect()
    model = lgb.train(params, ds, num_boost_round=cfg.rounds)
    del ds; gc.collect()
    A = F.to_np(Xv, feats)
    pp = np.clip(model.predict(A), EPS, 1-EPS).astype(np.float32)
    uid = Xv["user_id"].to_numpy().astype(np.int64)
    if is_test:
        save_npz_atomic(path, user_id=uid, p=pp)
    else:
        save_npz_atomic(path, user_id=uid, y=np.asarray(yv, np.float32), p=pp)
    log("CHILD OCC DONE", name, fold, "train_rows", f"{len(y):,}", "file", path)
    del model, A, Xv, y, w, pp
    T._XY.clear(); gc.collect()


def train_hurdle_sequential_child(base: Path, args, name: str, fold: str) -> None:
    """Memory-safe sequential two-part fit for fast12/stable18.

    The old helper constructed classifier and positive-regression datasets at the
    same time.  This version trains / frees the classifier Dataset before building
    the positive Dataset, materially reducing peak RAM.
    """
    combo, fixed, prev, ctx = configure_child_context(base, args)
    if name not in combo.MEM_VARIANTS:
        raise KeyError(name)
    maxcuts, tau, rounds = combo.MEM_VARIANTS[name]
    T, F, M = ctx.train_mod, ctx.features_mod, ctx.models_mod
    s, _ = fixed.expert_spec(ctx, "recent_hurdle")
    s.weight_tau = tau; s.rounds = rounds
    C = ctx.config
    is_test = fold == "TEST"
    path = combo.variant_test_path(ctx, name) if is_test else combo.variant_fold_path(ctx, name, fold)
    need = ("user_id", "z", "p", "mu") if is_test else ("user_id", "y", "z", "p", "mu")
    if valid_npz(path, need):
        log("CHILD HURDLE reuse", name, fold); return
    if is_test:
        V = C.CUTOFF_TEST
        Xv, _ = F.make_xy(V, s.L, s.panel_blocks, with_target=False, norm_long=s.norm_long)
        cuts = list(s.grid())[-maxcuts:]
    else:
        V = dt.date.fromisoformat(fold)
        Xv, yv = T.xy(V, s)
        cuts = list(s.train_cutoffs(V))[-maxcuts:]
    feats = T.select_features(F.feature_names(Xv), s.drop_groups, s.keep_only)
    log("CHILD HURDLE SEQ", name, fold, "cuts", len(cuts), "features", len(feats))
    X, y, w = T.assemble(cuts, s, feats, V)
    import lightgbm as lgb
    try:
        p_base = M._params(s.params)
        p_clf = M._params(s.params, objective="binary", metric="binary_logloss")
    except Exception:
        p_base = dict(objective="regression", metric="rmse", learning_rate=.05, verbosity=-1,
                      num_threads=max(2, min(8, os.cpu_count() or 6)))
        p_clf = dict(p_base, objective="binary", metric="binary_logloss")
    ds_clf = lgb.Dataset(X, (y > 0).astype(np.int8), weight=w, params=p_clf, free_raw_data=True).construct()
    clf = lgb.train(p_clf, ds_clf, num_boost_round=rounds)
    del ds_clf; gc.collect()
    pos = y > 0
    # Positive matrix is smaller; after copying it, free the huge all-row matrix.
    Xp = np.asarray(X[pos], np.float32)
    yp = np.log1p(np.asarray(y[pos], np.float64))
    wp = np.asarray(w[pos], np.float32)
    del X, y, w, pos
    T._XY.clear(); gc.collect()
    ds_reg = lgb.Dataset(Xp, yp, weight=wp, params=p_base, free_raw_data=True).construct()
    del Xp, yp, wp; gc.collect()
    reg = lgb.train(p_base, ds_reg, num_boost_round=rounds)
    del ds_reg; gc.collect()
    A = F.to_np(Xv, feats)
    pp = np.clip(clf.predict(A), EPS, 1-EPS)
    mu = np.maximum(reg.predict(A), 0.0)
    z = clipz(pp * mu)
    kw = dict(user_id=Xv["user_id"].to_numpy().astype(np.int64),
              z=np.asarray(z, np.float32), p=np.asarray(pp, np.float32), mu=np.asarray(mu, np.float32))
    if not is_test:
        kw["y"] = np.asarray(yv, np.float32)
    save_npz_atomic(path, **kw)
    log("CHILD HURDLE SEQ DONE", name, fold, "file", path)
    del clf, reg, A, Xv, pp, mu, z
    T._XY.clear(); gc.collect()


def child_main(args) -> None:
    base = Path(__file__).resolve().parent
    if args.child_occ:
        train_occ_child(base, args, args.child_occ, args.child_fold)
        return
    if args.child_hurdle:
        train_hurdle_sequential_child(base, args, args.child_hurdle, args.child_fold)
        return
    raise RuntimeError("child mode without task")


def run_child(script: Path, args, kind: str, name: str, fold: str, errors: Path) -> tuple[bool, float]:
    cmd = [sys.executable, str(script), "--reuse-work-dir", str(args.reuse_work_dir or ""),
           "--threads", str(args.child_threads), "--no-install"]
    # Empty reuse path is undesirable for argparse; omit it.
    if not args.reuse_work_dir:
        cmd = [sys.executable, str(script), "--threads", str(args.child_threads), "--no-install"]
    if kind == "occ":
        cmd += ["--child-occ", name, "--child-fold", fold]
    else:
        cmd += ["--child-hurdle", name, "--child-fold", fold]
    log("SPAWN", kind, name, fold)
    t = time.time()
    try:
        rc = subprocess.run(cmd, check=False).returncode
        ok = rc == 0
        if not ok:
            log("CHILD FAILED", kind, name, fold, "returncode", rc)
        return ok, (time.time()-t)/3600.0
    except Exception as exc:
        append_error(errors, f"child_{kind}", name, fold, exc)
        return False, (time.time()-t)/3600.0


# -----------------------------------------------------------------------------
# compact bank / research helpers
# -----------------------------------------------------------------------------

def load_full_bank(combo, fixed, prev, ctx):
    bank = fixed.load_core(prev, ctx)
    combo.load_existing_new_oof(fixed, ctx, bank)
    for name in combo.MEM_VARIANTS:
        if all(valid_npz(combo.variant_fold_path(ctx, name, f), ("user_id","y","z","p","mu")) for f in FOLDS):
            for f in FOLDS:
                d = load_npz(combo.variant_fold_path(ctx, name, f))
                r = bank[f]
                r[name] = align(d["user_id"], d["z"], r["uid"]).astype(np.float32)
                r[f"p_{name}"] = align(d["user_id"], d["p"], r["uid"]).astype(np.float32)
                r[f"mu_{name}"] = align(d["user_id"], d["mu"], r["uid"]).astype(np.float32)
    return bank


def load_occ_into_bank(bank, ctx, completed_names: Sequence[str]) -> list[str]:
    loaded = []
    for name in completed_names:
        if not all(valid_npz(occ_fold_path(ctx,name,f),("user_id","y","p")) for f in FOLDS):
            continue
        for f in FOLDS:
            d = load_npz(occ_fold_path(ctx,name,f)); r = bank[f]
            r[f"p_{name}"] = np.clip(align(d["user_id"], d["p"], r["uid"]), EPS, 1-EPS).astype(np.float32)
        loaded.append(name)
    return loaded


def score_table(fixed, name, preds, bank, family, rows, notes=""):
    return fixed.score_table(name, preds, bank, family, rows, notes=notes)


def candidate_index(rows):
    return {r["name"]: r for r in rows}


def choose_top_pred_names(rows, predpool, maxn=8):
    # Prefer robust recent winners and architectural diversity.
    ranked = sorted(rows, key=lambda r:(r["delta"], r["latest_delta"]))
    out=[]; fam={}
    for r in ranked:
        if r["name"] not in predpool or r["delta"] >= 0 or r["wins_recent"] < 2:
            continue
        if fam.get(r["family"],0) >= 2:
            continue
        out.append(r["name"]); fam[r["family"]]=fam.get(r["family"],0)+1
        if len(out)>=maxn: break
    return out


def _super_features(rec, names, predpool_fold):
    P = np.stack([np.asarray(predpool_fold[n], np.float32) for n in names], axis=1)
    core=np.asarray(rec["table_core"],np.float32); p=np.asarray(rec["p"],np.float32); mu=np.asarray(rec["mu"],np.float32)
    extras=np.column_stack([core,p,mu,np.log1p(mu),p*(1-p),P.mean(1),P.std(1),P.max(1)-P.min(1)])
    return np.column_stack([P, extras]).astype(np.float32)


def walk_super_ridge(bank, names, predpool, alpha=150., shrink=.75, power=1.7, recent_k=None):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    out={}
    for i,f in enumerate(FOLDS):
        if i==0:
            out[f]=np.asarray(bank[f]["table_core"],np.float64).copy(); continue
        ids=list(range(i));
        if recent_k is not None: ids=ids[-int(recent_k):]
        X=[];y=[];w=[]
        for j in ids:
            ff=FOLDS[j]; X.append(_super_features(bank[ff],names,{n:predpool[n][ff] for n in names}))
            y.append(np.asarray(bank[ff]["true_z"]-bank[ff]["table_core"],np.float32))
            w.append(np.full(len(bank[ff]["uid"]),FW[j]**power,np.float32))
        X=np.vstack(X); y=np.concatenate(y); w=np.concatenate(w)
        sc=StandardScaler(copy=False); Xs=sc.fit_transform(X)
        m=Ridge(alpha=float(alpha),solver="lsqr",tol=1e-4); m.fit(Xs,y,sample_weight=w)
        Xt=_super_features(bank[f],names,{n:predpool[n][f] for n in names}); Xt=sc.transform(Xt)
        d=np.clip(m.predict(Xt),-1.5,1.5)
        out[f]=clipz(bank[f]["table_core"]+float(shrink)*d)
        del X,y,w,Xs,Xt,d,m,sc;gc.collect()
    return out


def final_super_ridge(bank,test,names,predpool,testpool,alpha=150.,shrink=.75,power=1.7,recent_k=None):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    ids=list(range(4));
    if recent_k is not None: ids=ids[-int(recent_k):]
    X=[];y=[];w=[]
    for j in ids:
        ff=FOLDS[j]; X.append(_super_features(bank[ff],names,{n:predpool[n][ff] for n in names}))
        y.append(np.asarray(bank[ff]["true_z"]-bank[ff]["table_core"],np.float32));w.append(np.full(len(bank[ff]["uid"]),FW[j]**power,np.float32))
    X=np.vstack(X);y=np.concatenate(y);w=np.concatenate(w)
    sc=StandardScaler(copy=False);Xs=sc.fit_transform(X);m=Ridge(alpha=float(alpha),solver="lsqr",tol=1e-4);m.fit(Xs,y,sample_weight=w)
    Xt=_super_features(test,names,testpool);Xt=sc.transform(Xt);d=np.clip(m.predict(Xt),-1.5,1.5)
    z=clipz(test["table_core"]+float(shrink)*d)
    del X,y,w,Xs,Xt,d,m,sc;gc.collect();return z


@dataclasses.dataclass
class NewRecipe:
    name: str
    kind: str
    params: dict[str,Any]
    family: str


def extend_stable_research(combo,fixed,bank,rows,predpool,recipes,results):
    newrecipes:dict[str,NewRecipe]={}
    top=choose_top_pred_names(rows,predpool,8)
    if len(top)<3:
        return rows,predpool,newrecipes

    def add(name,p,family,rec,notes=""):
        if name in predpool:return
        score_table(fixed,name,p,bank,family,rows,notes);predpool[name]=p;newrecipes[name]=rec

    # Candidate-level Ridge over already cross-fitted candidate predictions.
    for alpha in (20.,100.,400.):
        for shrink in (.65,.80,.95):
            name=f"superridge_a{int(alpha)}_s{int(shrink*100):02d}"
            p=walk_super_ridge(bank,top,predpool,alpha,shrink,1.7,None)
            add(name,p,"super_ridge",NewRecipe(name,"super_ridge",{"names":top,"alpha":alpha,"shrink":shrink,"power":1.7,"recent_k":None},"super_ridge"),notes=str(top))
    # Recent-only meta is deliberately separate: temporal shift is strong.
    for alpha in (80.,250.):
        name=f"superridge_recent2_a{int(alpha)}_s80"
        p=walk_super_ridge(bank,top,predpool,alpha,.80,2.0,2)
        add(name,p,"super_ridge_recent",NewRecipe(name,"super_ridge",{"names":top,"alpha":alpha,"shrink":.80,"power":2.0,"recent_k":2},"super_ridge_recent"),notes=str(top))

    # Re-stack the actually strongest candidate predictions, not only primitives.
    ranked=choose_top_pred_names(rows,predpool,10)
    if len(ranked)>=4:
        for lam in (.06,.14,.28):
            name=f"super_pband_l{str(lam).replace('.','p')}"
            p,_=combo.walk_candidate_pband(bank,ranked[:7],predpool,lam=lam,prior_name=ranked[0])
            add(name,p,"super_pband",NewRecipe(name,"cand_pband",{"names":ranked[:7],"lam":lam,"prior":ranked[0]},"super_pband"))
        for lam in (.05,.15):
            name=f"super_simplex_l{str(lam).replace('.','p')}"
            p,_=combo.walk_candidate_simplex(bank,ranked[:8],predpool,lam=lam,prior_name=ranked[0])
            add(name,p,"super_simplex",NewRecipe(name,"cand_simplex",{"names":ranked[:8],"lam":lam,"prior":ranked[0]},"super_simplex"))

    rows.sort(key=lambda r:(r["delta"],r["latest_delta"]))
    save_csv(results/"STABLE_BRANCH_VALIDATION.csv",rows)
    return rows,predpool,newrecipes


# -----------------------------------------------------------------------------
# occurrence integration / meta occurrence / risk-gated correction
# -----------------------------------------------------------------------------

def p_apply(base_table,p_base,mu,p_new,down=1.0,up=.15,shift=0.0,risk=None,threshold=None):
    pp=sigmoid(logit(p_new)+float(shift))
    delta=pp-np.asarray(p_base,np.float64)
    strength=np.where(delta<0,float(down),float(up))
    if risk is not None:
        rr=np.clip(np.asarray(risk,np.float64),0,1)
        # Error detectors are only a trust signal. Never direct magnitude correction.
        strength=np.where(delta<0,strength*(.30+.70*rr),strength*(.70+.30*(1-rr)))
    if threshold is not None:
        active=np.abs(delta)>=float(threshold)
        strength=np.where(active,strength,0.0)
    return clipz(np.asarray(base_table,np.float64)+strength*delta*np.asarray(mu,np.float64))


def fit_occ_params_on_past(bank,ids,pkey,base_oof=None,risk_oof=None):
    # Small regularized grid. Objective is weighted MSE in z-space + weak prior.
    best=None
    shifts=(-.22,-.14,-.08,0.0,.06)
    downs=(.45,.65,.85,1.0)
    ups=(.05,.12,.22)
    ths=(None,.025,.05)
    for sh in shifts:
      for dn in downs:
       for up in ups:
        for th in ths:
         num=0.;den=0.
         for j in ids:
            f=FOLDS[j];r=bank[f];base=np.asarray(base_oof[f] if base_oof is not None else r["table_core"],np.float64)
            risk=None if risk_oof is None else risk_oof[f]
            z=p_apply(base,r["p"],r["mu"],r[pkey],dn,up,sh,risk,th)
            e=(z-r["true_z"])**2; w=FW[j]
            num+=w*float(np.mean(e));den+=w
         if den==0:continue
         # discourage extreme shifts / full replacement from one old fold
         obj=num/den + .0015*sh*sh + .00025*(dn-.75)**2 + .00020*(up-.12)**2
         if best is None or obj<best[0]:best=(obj,sh,dn,up,th)
    return best[1:]


def walk_occ_candidate(bank,pkey,base_oof=None,risk_oof=None,adaptive=True,fixed_params=(-.08,.75,.12,.025)):
    out={};pars=[]
    for i,f in enumerate(FOLDS):
        r=bank[f];base=np.asarray(base_oof[f] if base_oof is not None else r["table_core"],np.float64)
        if adaptive and i>0:
            pa=fit_occ_params_on_past(bank,list(range(i)),pkey,base_oof,risk_oof)
        else:pa=fixed_params
        sh,dn,up,th=pa;risk=None if risk_oof is None else risk_oof[f]
        out[f]=p_apply(base,r["p"],r["mu"],r[pkey],dn,up,sh,risk,th);pars.append(pa)
    return out,pars


def final_occ_candidate(bank,test,pkey,ptest,base_oof,base_test,risk_oof=None,risk_test=None,adaptive=True,fixed_params=(-.08,.75,.12,.025)):
    pa=fit_occ_params_on_past(bank,list(range(4)),pkey,base_oof,risk_oof) if adaptive else fixed_params
    sh,dn,up,th=pa
    z=p_apply(base_test,test["p"],test["mu"],ptest,dn,up,sh,risk_test,th)
    return z,pa


def occ_meta_features(rec, occ_names):
    cols=[logit(rec["p"]),np.asarray(rec["p"],np.float64),np.log1p(np.asarray(rec["mu"],np.float64)),np.asarray(rec["table_core"],np.float64)]
    for n in occ_names:cols.extend([logit(rec[f"p_{n}"]),np.asarray(rec[f"p_{n}"],np.float64)-np.asarray(rec["p"],np.float64)])
    # Raw behavioral features are already saved by the full-user cap runner.
    meta=np.asarray(rec["meta_raw"],np.float32)
    # Cap dimension to keep meta classifier light but retain deterministic first columns.
    if meta.shape[1]>96: meta=meta[:,:96]
    return np.column_stack([meta]+cols).astype(np.float32)


def make_occ_meta(seed=42,leaves=31):
    import lightgbm as lgb
    return lgb.LGBMClassifier(n_estimators=420,learning_rate=.03,num_leaves=leaves,max_depth=-1,
                              min_child_samples=450,subsample=.88,colsample_bytree=.78,
                              reg_lambda=18.,reg_alpha=1.2,max_bin=127,random_state=seed,
                              n_jobs=max(2,min(10,os.cpu_count() or 8)),verbosity=-1)


def walk_meta_occ(bank,occ_names,power=1.7,leaves=31):
    out={};models=[]
    for i,f in enumerate(FOLDS):
        if i==0:
            out[f]=np.asarray(bank[f]["p"],np.float64).copy();models.append(None);continue
        X=[];y=[];w=[]
        for j in range(i):
            ff=FOLDS[j];X.append(occ_meta_features(bank[ff],occ_names));y.append((bank[ff]["y"]>0).astype(np.int8));w.append(np.full(len(bank[ff]["uid"]),FW[j]**power,np.float32))
        X=np.vstack(X);y=np.concatenate(y);w=np.concatenate(w);m=make_occ_meta(7100+i,leaves);m.fit(X,y,sample_weight=w)
        out[f]=np.clip(m.predict_proba(occ_meta_features(bank[f],occ_names))[:,1],EPS,1-EPS);models.append(m)
        del X,y,w,m;gc.collect()
    return out


def final_meta_occ(bank,test,occ_names,ptest,power=1.7,leaves=31):
    X=[];y=[];w=[]
    for j,f in enumerate(FOLDS):X.append(occ_meta_features(bank[f],occ_names));y.append((bank[f]["y"]>0).astype(np.int8));w.append(np.full(len(bank[f]["uid"]),FW[j]**power,np.float32))
    X=np.vstack(X);y=np.concatenate(y);w=np.concatenate(w);m=make_occ_meta(7900,leaves);m.fit(X,y,sample_weight=w)
    tr=dict(test)
    for n in occ_names:tr[f"p_{n}"]=ptest[n]
    p=np.clip(m.predict_proba(occ_meta_features(tr,occ_names))[:,1],EPS,1-EPS)
    del X,y,w,m;gc.collect();return p


def risk_features(rec,occ_names):
    X=occ_meta_features(rec,occ_names)
    extra=np.column_stack([np.asarray(rec["table_core"])-np.asarray(rec["p"])*np.asarray(rec["mu"]),
                           np.asarray(rec["p"])*np.asarray(rec["mu"]),
                           np.asarray(rec["p"])*(1-np.asarray(rec["p"]))]).astype(np.float32)
    return np.column_stack([X,extra]).astype(np.float32)


def make_risk_model(seed=42):
    import lightgbm as lgb
    return lgb.LGBMClassifier(n_estimators=320,learning_rate=.035,num_leaves=23,min_child_samples=550,
                              subsample=.90,colsample_bytree=.72,reg_lambda=22.,reg_alpha=1.5,
                              max_bin=127,random_state=seed,n_jobs=max(2,min(10,os.cpu_count() or 8)),verbosity=-1)


def _risk_labels(rec):
    zero=np.asarray(rec["y"])<=0
    fo=(zero & (np.asarray(rec["p"])>=.5)).astype(np.int8)
    over=((np.asarray(rec["table_core"])-np.asarray(rec["true_z"]))>1.0).astype(np.int8)
    return fo,over


def walk_risk_gate(bank,occ_names,power=1.7):
    out={}
    for i,f in enumerate(FOLDS):
        if i==0:out[f]=np.full(len(bank[f]["uid"]),.5,np.float64);continue
        X=[];yf=[];yo=[];w=[]
        for j in range(i):
            ff=FOLDS[j];a,b=_risk_labels(bank[ff]);X.append(risk_features(bank[ff],occ_names));yf.append(a);yo.append(b);w.append(np.full(len(a),FW[j]**power,np.float32))
        X=np.vstack(X);yf=np.concatenate(yf);yo=np.concatenate(yo);w=np.concatenate(w)
        mf=make_risk_model(8100+i);mo=make_risk_model(8200+i);mf.fit(X,yf,sample_weight=w);mo.fit(X,yo,sample_weight=w)
        Xt=risk_features(bank[f],occ_names);pf=mf.predict_proba(Xt)[:,1];po=mo.predict_proba(Xt)[:,1];out[f]=np.sqrt(np.clip(pf*po,0,1))
        del X,yf,yo,w,Xt,mf,mo,pf,po;gc.collect()
    return out


def final_risk_gate(bank,test,occ_names,ptest,power=1.7):
    X=[];yf=[];yo=[];w=[]
    for j,f in enumerate(FOLDS):
        a,b=_risk_labels(bank[f]);X.append(risk_features(bank[f],occ_names));yf.append(a);yo.append(b);w.append(np.full(len(a),FW[j]**power,np.float32))
    X=np.vstack(X);yf=np.concatenate(yf);yo=np.concatenate(yo);w=np.concatenate(w);mf=make_risk_model(8901);mo=make_risk_model(8902);mf.fit(X,yf,sample_weight=w);mo.fit(X,yo,sample_weight=w)
    tr=dict(test)
    for n in occ_names:tr[f"p_{n}"]=ptest[n]
    Xt=risk_features(tr,occ_names);pf=mf.predict_proba(Xt)[:,1];po=mo.predict_proba(Xt)[:,1];r=np.sqrt(np.clip(pf*po,0,1))
    del X,yf,yo,w,Xt,mf,mo,pf,po;gc.collect();return r


# -----------------------------------------------------------------------------
# finalization utilities / selection
# -----------------------------------------------------------------------------

def candidate_distance(a,b):
    a=np.asarray(a,np.float64);b=np.asarray(b,np.float64);d=a-b
    if np.std(a)<1e-12 or np.std(b)<1e-12:corr=0.0
    else:corr=float(np.corrcoef(a,b)[0,1])
    return {"corr":corr,"std":float(np.std(d)),"mae":float(np.mean(np.abs(d))),
            "pct02":float(np.mean(np.abs(d)>.02)),"pct05":float(np.mean(np.abs(d)>.05)),"pct10":float(np.mean(np.abs(d)>.10))}


def evaluate_occurrence_pool(fixed,bank,base_oof_candidates,occ_names,rows,predpool,results):
    occ_rows=[];occ_specs={}
    # Raw occurrence configs over table core and over top stable candidates.
    for n in occ_names:
        pkey=f"p_{n}"
        for adaptive in (False,True):
            tag="adapt" if adaptive else "fixed"
            for base_name,base_oof in base_oof_candidates:
                name=f"occ_{n}_{tag}__{base_name}"
                p,_=walk_occ_candidate(bank,pkey,base_oof,None,adaptive)
                rr=score_table(fixed,name,p,bank,"occurrence_overlay",rows,notes=f"raw={n};base={base_name};adaptive={adaptive}")
                predpool[name]=p;occ_rows.append(rr);occ_specs[name]={"kind":"raw_occ","occ":n,"base":base_name,"adaptive":adaptive,"risk":False}
    # Meta occurrence combines all completed raw occurrence models.
    if len(occ_names)>=2:
        for leaves in (23,31):
            pm=walk_meta_occ(bank,occ_names,power=1.7,leaves=leaves)
            key=f"p_metaocc_l{leaves}"
            for f in FOLDS:bank[f][key]=pm[f]
            # Risk gate is trained once per meta variant.
            risk=walk_risk_gate(bank,occ_names,power=1.7)
            for base_name,base_oof in base_oof_candidates:
                for gated in (False,True):
                    name=f"metaocc_l{leaves}_{'risk' if gated else 'plain'}__{base_name}"
                    p,_=walk_occ_candidate(bank,key,base_oof,risk if gated else None,True)
                    rr=score_table(fixed,name,p,bank,"occurrence_meta_risk" if gated else "occurrence_meta",rows,notes=f"base={base_name};occ={occ_names}")
                    predpool[name]=p;occ_rows.append(rr);occ_specs[name]={"kind":"meta_occ","leaves":leaves,"base":base_name,"risk":gated,"occ_names":list(occ_names)}
    occ_rows.sort(key=lambda r:(r["delta"],r["latest_delta"]))
    save_csv(results/"OCCURRENCE_BRANCH_VALIDATION.csv",occ_rows)
    return occ_specs


def find_recipe_any(name,combo_recipes,newrecipes):
    if name in newrecipes:return ("new",newrecipes[name])
    if name in combo_recipes:return ("combo",combo_recipes[name])
    raise KeyError(name)


def finalize_stable_candidate(name, combo, fixed, bank, test, predpool, combo_recipes, newrecipes, cache):
    if name in cache:return cache[name]
    kind,rec=find_recipe_any(name,combo_recipes,newrecipes)
    if kind=="combo":
        z=combo.finalize_recipe(name,combo_recipes,predpool,bank,test,fixed,cache)
    else:
        p=rec.params
        if rec.kind=="super_ridge":
            testpool={n:finalize_stable_candidate(n,combo,fixed,bank,test,predpool,combo_recipes,newrecipes,cache) for n in p["names"]}
            z=final_super_ridge(bank,test,p["names"],predpool,testpool,p["alpha"],p["shrink"],p["power"],p["recent_k"])
        elif rec.kind=="cand_pband":
            names=p["names"];testpool={n:finalize_stable_candidate(n,combo,fixed,bank,test,predpool,combo_recipes,newrecipes,cache) for n in names}
            W,bands=combo.fit_candidate_pband(bank,list(range(4)),names,predpool,p["lam"],p["prior"])
            z=combo.apply_candidate_pband(test,names,testpool,W,bands)
        elif rec.kind=="cand_simplex":
            names=p["names"];testpool={n:finalize_stable_candidate(n,combo,fixed,bank,test,predpool,combo_recipes,newrecipes,cache) for n in names}
            z,_=combo.final_candidate_simplex(bank,names,predpool,testpool,p["lam"],p["prior"])
        else:raise KeyError(rec.kind)
    cache[name]=z;return z


def build_test_bank(combo,fixed,prev,ctx,friend):
    test=combo.build_test_core_only(fixed,prev,ctx,friend)
    # Load every final expert that exists; missing optional artifacts stay optional.
    names=set(combo.finalizable_experts({f:{} for f in FOLDS}))|set(combo.MEM_VARIANTS)|{"multiscale_direct","recent_direct","recent_dist","recent_hurdle"}
    for n in sorted(names):
        if n in ("cap","unc","dist","hurdle"):continue
        try:combo.load_final_raw_into_test(test,ctx,n)
        except Exception:pass
    return test


def select_branch(rows,families,require_recent=3):
    cand=[r for r in rows if r.get("family") in families and r["delta"]<0 and r["wins_recent"]>=require_recent and r["latest_delta"]<0]
    if not cand and require_recent>2:
        cand=[r for r in rows if r.get("family") in families and r["delta"]<0 and r["wins_recent"]>=2 and r["latest_delta"]<=.0001]
    return min(cand,key=lambda r:(r["delta"],r["latest_delta"])) if cand else None


def locate_old_submissions(base, friend_uid):
    import pandas as pd
    out={}
    for root in (base/"_best_bas_combo_10h"/"submissions",base/"_best_bas_continue_12h"/"submissions"):
        if not root.exists():continue
        for p in root.glob("*.csv"):
            try:
                d=pd.read_csv(p)
                if {"user_id","predict"}.issubset(d.columns) and len(d)==250000:
                    out[p.stem]=align(d.user_id.to_numpy(np.int64),np.log1p(np.maximum(d.predict.to_numpy(np.float64),0)),friend_uid)
            except Exception:pass
    return out


def ensure_friend_submission(friend_z, uid, sample_path, outpath):
    import pandas as pd
    s=pd.read_csv(sample_path);su=s["user_id"].to_numpy(np.int64)
    z=align(uid,friend_z,su);pred=np.maximum(np.expm1(np.clip(z,0,20)),0)
    pd.DataFrame({"user_id":su,"predict":pred}).to_csv(outpath,index=False)


# -----------------------------------------------------------------------------
# smoke test
# -----------------------------------------------------------------------------

def self_test():
    import types
    rng=np.random.default_rng(7);n=1800
    bank={};predpool={}
    class Fixed:
        @staticmethod
        def score_table(name,preds,bank,family,rows,notes=""):
            ss=[];bb=[];dd=[]
            for f in FOLDS:
                z=np.asarray(preds[f]);y=bank[f]["true_z"];b=bank[f]["table_core"]
                a=float(np.sqrt(np.mean((z-y)**2)));c=float(np.sqrt(np.mean((b-y)**2)));ss.append(a);bb.append(c);dd.append(a-c)
            r={"name":name,"family":family,"wcv":wavg(ss),"base_wcv":wavg(bb),"delta":wavg(dd),"wins":sum(x<0 for x in dd),"wins_recent":sum(x<0 for x in dd[1:]),"latest_delta":dd[-1],"worst_delta":max(dd),"fold_scores":ss,"fold_deltas":dd,"notes":notes};rows.append(r);return r
    for j,f in enumerate(FOLDS):
        X=rng.normal(size=(n,40)).astype(np.float32);p=sigmoid(.6*X[:,0]-.25*X[:,1]);mu=np.maximum(1.5+.45*X[:,2],.1);tz=np.maximum(p*mu+.22*X[:,3]+rng.normal(scale=.72,size=n),0)
        core=np.maximum(tz+rng.normal(scale=.42,size=n),0);rec={"uid":np.arange(n)+j*n,"y":np.expm1(tz),"true_z":tz,"p":p,"mu":mu,"table_core":core,"meta_raw":X}
        rec["p_occ_r10_fast"]=np.clip(sigmoid(logit(p)+.25*X[:,4]-.12),.001,.999)
        rec["p_occ_r16_bal"]=np.clip(sigmoid(logit(p)+.18*X[:,5]-.10),.001,.999)
        bank[f]=rec
        predpool["a"]={**predpool.get("a",{}),f:np.maximum(core+.10*X[:,6],0)}
        predpool["b"]={**predpool.get("b",{}),f:np.maximum(core-.08*X[:,7],0)}
    rows=[]
    for k in ("a","b"):Fixed.score_table(k,predpool[k],bank,"x",rows)
    p=walk_super_ridge(bank,["a","b"],predpool,80,.75,1.7,None);Fixed.score_table("super",p,bank,"super_ridge",rows)
    pm=walk_meta_occ(bank,["occ_r10_fast","occ_r16_bal"],1.7,23)
    for f in FOLDS:bank[f]["p_metaocc_l23"]=pm[f]
    risk=walk_risk_gate(bank,["occ_r10_fast","occ_r16_bal"],1.7)
    po,_=walk_occ_candidate(bank,"p_metaocc_l23",p,risk,True);Fixed.score_table("occ",po,bank,"occurrence_meta_risk",rows)
    assert len(rows)>=4 and all(np.isfinite(r["delta"]) for r in rows)
    print("SELF-TEST OK",len(rows),flush=True)


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--max-hours",type=float,default=6.0)
    ap.add_argument("--threads",type=int,default=max(4,min(10,os.cpu_count() or 8)))
    ap.add_argument("--child-threads",type=int,default=6)
    ap.add_argument("--reuse-work-dir",type=str,default=None)
    ap.add_argument("--no-install",action="store_true")
    ap.add_argument("--preflight-only",action="store_true")
    ap.add_argument("--self-test",action="store_true")
    ap.add_argument("--child-occ",type=str,default=None,help=argparse.SUPPRESS)
    ap.add_argument("--child-hurdle",type=str,default=None,help=argparse.SUPPRESS)
    ap.add_argument("--child-fold",type=str,default="TEST",help=argparse.SUPPRESS)
    args=ap.parse_args()
    if args.self_test:self_test();return
    if args.child_occ or args.child_hurdle:child_main(args);return

    started=time.time();budget=Budget(started,args.max_hours)
    base=Path(__file__).resolve().parent
    combo,combo_path,fixed,fixed_path,prev,prev_path=discover_parent_scripts(base)
    package=prev.discover_package(base);raw,sample=prev.discover_raw_and_sample(base,package);prev.ensure_dependencies(package,args.no_install)
    work=fixed.discover_work(base,args.reuse_work_dir)
    # Pin all child processes to the exact same checkpoint directory.
    args.reuse_work_dir=str(work)
    out=base/"_best_bas_final6h";results=out/"results";subs=out/"submissions";results.mkdir(parents=True,exist_ok=True);subs.mkdir(parents=True,exist_ok=True)
    ctx=prev.Context(base_dir=base,package=package,pipeline=package/"pipeline",raw=raw,sample=sample,work=work,results=results,submissions=subs,checkpoints=work/"checkpoints",budget=prev.Budget(started,args.max_hours,max(args.max_hours-.8,0),.6))
    prev.configure_pipeline(ctx,args.threads);fixed.repair_cache(ctx);fixed.install_atomic_cache(ctx)
    friend=prev.verify_friend_package(package);errors=results/"errors.jsonl"
    log("FINAL6H",SCRIPT_VERSION,"work",work)
    log("FRIEND exact rebuild error",friend.get("max_log_error"),"known public",KNOWN_FRIEND_PUBLIC,"best fixedstack public",KNOWN_RIDGE_SUB_PUBLIC)
    log("STRONGEST_CURRENT / teammate SEQ+ETX are NEVER retrained.")

    # Preflight: mandatory compact checkpoints and stable18 OOF availability.
    core_ok=all(combo.npz_valid(Path(ctx.checkpoints)/"folds"/f"{n}__{f}.npz",("user_id","y","z")) for n in ("cap","unc","dist") for f in FOLDS)
    hurdle_ok=all(combo.npz_valid(Path(ctx.checkpoints)/"folds"/f"hurdle__{f}.npz",("user_id","y","z","p","mu")) for f in FOLDS)
    if not (core_ok and hurdle_ok):raise RuntimeError("Mandatory fixedstack OOF missing; refusing to retrain teammate/base models")
    stable_oof=all(valid_npz(combo.variant_fold_path(ctx,"recent_hurdle_stable18",f),("user_id","y","z","p","mu")) for f in FOLDS)
    fast_oof=all(valid_npz(combo.variant_fold_path(ctx,"recent_hurdle_fast12",f),("user_id","y","z","p","mu")) for f in FOLDS)
    if args.preflight_only:
        log("PREFLIGHT OK","core",core_ok,"hurdle",hurdle_ok,"fast12_oof",fast_oof,"stable18_oof",stable_oof,"remaining",f"{budget.remaining:.2f}h")
        return

    manifest={"version":SCRIPT_VERSION,"started":now(),"combo_parent":str(combo_path),"fixed_parent":str(fixed_path),"previous_runner":str(prev_path),"work":str(work),"package":str(package),"args":vars(args),"friend_rebuild_error":friend.get("max_log_error"),"known_friend_public":KNOWN_FRIEND_PUBLIC,"known_ridge_submission_public":KNOWN_RIDGE_SUB_PUBLIC}
    atomic_json(results/"RUN_START.json",manifest)
    runtime=[]
    script=Path(__file__).resolve()

    # 1) Finish the one missing piece from the previous winning branch FIRST.
    for hname in ("recent_hurdle_fast12","recent_hurdle_stable18"):
        hp=combo.variant_test_path(ctx,hname)
        if valid_npz(hp,("user_id","z","p","mu")):
            log("REUSE FINAL HURDLE",hname);continue
        # stable18 is valuable enough to spend up to ~1.3h here.
        if not budget.can_start(.35,extra=.55):
            log("SKIP HURDLE FINAL by budget",hname);continue
        ok,h=run_child(script,args,"hurdle",hname,"TEST",errors);runtime.append({"stage":"hurdle_final","name":hname,"fold":"TEST","hours":h,"ok":ok,"remaining":budget.remaining});save_csv(results/"runtime.csv",runtime)

    # 2) Train raw occurrence-only families. Heavy work happens before the parent
    #    loads the compact OOF bank, keeping RAM pressure low.
    completed_occ=[];family_times=[]
    for cfg_idx,cfg in enumerate(OCC_QUEUE):
        # If a whole family already exists it costs nothing and should be used.
        tasks=[f for f in FOLDS if not valid_npz(occ_fold_path(ctx,cfg.name,f),("user_id","y","p"))]
        if not valid_npz(occ_test_path(ctx,cfg.name),("user_id","p")):tasks.append("TEST")
        if not tasks:
            completed_occ.append(cfg.name);log("REUSE OCC FAMILY",cfg.name);continue
        # After one measured family, project the next full family conservatively.
        est_family=(np.median(family_times)*1.20 if family_times else .80)
        if not budget.can_start(est_family,extra=.35):
            log("STOP OCC QUEUE before",cfg.name,"remaining",f"{budget.remaining:.2f}h","est",f"{est_family:.2f}h");break
        log("OCC FAMILY START",cfg.name,"tasks",tasks,"remaining",f"{budget.remaining:.2f}h")
        ft=time.time();okfam=True
        measured=[]
        for ti,fold in enumerate(tasks):
            # Per-child guard learned from completed children.
            child_est=max(.10,1.25*np.median(measured)) if measured else .16
            left=len(tasks)-ti
            if budget.remaining <= budget.reserve + .35 + child_est*left:
                log("OCC FAMILY budget stop",cfg.name,"before",fold,"remaining",f"{budget.remaining:.2f}h");okfam=False;break
            ok,h=run_child(script,args,"occ",cfg.name,fold,errors);measured.append(h);runtime.append({"stage":"occ_raw","name":cfg.name,"fold":fold,"hours":h,"ok":ok,"remaining":budget.remaining});save_csv(results/"runtime.csv",runtime)
            if not ok:okfam=False;break
        total=(time.time()-ft)/3600.;family_times.append(total)
        full=all(valid_npz(occ_fold_path(ctx,cfg.name,f),("user_id","y","p")) for f in FOLDS) and valid_npz(occ_test_path(ctx,cfg.name),("user_id","p"))
        if full:
            completed_occ.append(cfg.name);log("OCC FAMILY COMPLETE",cfg.name,"hours",f"{total:.2f}")
        elif not okfam:log("OCC FAMILY INCOMPLETE",cfg.name)
        # Continue queue while there is meaningful research budget; no early exit simply because first configs were fast.
        if budget.remaining <= budget.reserve + .85:break

    # 3) Load compact OOF bank and perform all cheap combinations.
    bank=load_full_bank(combo,fixed,prev,ctx);completed_occ=load_occ_into_bank(bank,ctx,completed_occ)
    log("OCC COMPLETE BANK",completed_occ,"remaining",f"{budget.remaining:.2f}h")
    rows,predpool,combo_recipes,experts=combo.build_primitive_research(fixed,bank,results,"final6h_base")
    rows,predpool,combo_recipes=combo.build_combo_research(fixed,bank,rows,predpool,combo_recipes,experts,results,"final6h_base")
    rows,predpool,newrecipes=extend_stable_research(combo,fixed,bank,rows,predpool,combo_recipes,results)
    rows.sort(key=lambda r:(r["delta"],r["latest_delta"]))
    log("TOP STABLE BRANCH")
    for r in rows[:15]:log(" ",r["name"],f"d={r['delta']:+.6f}","recent",r["wins_recent"],"latest",f"{r['latest_delta']:+.6f}")

    # Stable bases supplied to occurrence overlay: table_core + best robust stable candidates.
    stable_candidates=[r for r in rows if r["delta"]<0 and r["wins_recent"]>=3 and r["latest_delta"]<0 and r["name"] in predpool]
    stable_candidates=sorted(stable_candidates,key=lambda r:(r["delta"],r["latest_delta"]))[:4]
    base_oof_candidates=[("table_core",{f:bank[f]["table_core"] for f in FOLDS})]+[(r["name"],predpool[r["name"]]) for r in stable_candidates]
    occ_specs={}
    if completed_occ:
        occ_specs=evaluate_occurrence_pool(fixed,bank,base_oof_candidates,completed_occ,rows,predpool,results)
        rows.sort(key=lambda r:(r["delta"],r["latest_delta"]))
        log("TOP AFTER OCCURRENCE")
        for r in rows[:20]:log(" ",r["name"],f"d={r['delta']:+.6f}","recent",r["wins_recent"],"latest",f"{r['latest_delta']:+.6f}",r["family"])

    save_csv(results/"ALL_FINAL6H_VALIDATION.csv",rows)
    # Explicit research conclusions table by family.
    famrows=[]
    for fam in sorted(set(r["family"] for r in rows)):
        g=[r for r in rows if r["family"]==fam];b=min(g,key=lambda x:x["delta"]);famrows.append({"family":fam,"best_name":b["name"],"delta":b["delta"],"wins_recent":b["wins_recent"],"latest_delta":b["latest_delta"]})
    save_csv(results/"FAMILY_BEST.csv",famrows)

    # 4) Choose two DIFFERENT branches before finalization.
    A=select_branch(rows,{"adaptive_blend","local_bias","candidate_pband","super_ridge","super_ridge_recent","super_pband","super_simplex","ridge_temporal","ridge_subset"},3)
    B=select_branch(rows,{"occurrence_overlay","occurrence_meta","occurrence_meta_risk"},3)
    if A is None:
        A=min([r for r in rows if r["delta"]<0 and r["wins_recent"]>=2],key=lambda r:r["delta"])
    if B is None:
        # Honest fallback: different local-specialist branch, still locally improving.
        alts=[r for r in rows if r["family"] in {"local_bias","candidate_pband","pband","super_pband","simplex","greedy"} and r["delta"]<0 and r["wins_recent"]>=2]
        B=min(alts,key=lambda r:(r["delta"],r["latest_delta"])) if alts else min([r for r in rows if r["name"]!=A["name"] and r["delta"]<0],key=lambda r:r["delta"])
    log("SELECTED VALIDATION BRANCH A",A["name"],A["family"],f"d={A['delta']:+.6f}")
    log("SELECTED VALIDATION BRANCH B",B["name"],B["family"],f"d={B['delta']:+.6f}")

    # 5) Finalize. No heavy training should remain except compact meta fits.
    test=build_test_bank(combo,fixed,prev,ctx,friend)
    for n in completed_occ:
        d=load_npz(occ_test_path(ctx,n));test[f"p_{n}"]=np.clip(align(d["user_id"],d["p"],test["uid"]),EPS,1-EPS)
    cache={}

    def final_stable(name):
        if name=="table_core":return np.asarray(test["table_core"],np.float64)
        return finalize_stable_candidate(name,combo,fixed,bank,test,predpool,combo_recipes,newrecipes,cache)

    def finalize_occ_row(q):
        if q["name"] not in occ_specs:
            return final_stable(q["name"])
        sp=occ_specs[q["name"]];base_name=sp["base"]
        base_test=final_stable(base_name)
        base_oof={f:(bank[f]["table_core"] if base_name=="table_core" else predpool[base_name][f]) for f in FOLDS}
        if sp["kind"]=="raw_occ":
            ptest=np.asarray(test[f"p_{sp['occ']}"])
            z,_=final_occ_candidate(bank,test,f"p_{sp['occ']}",ptest,base_oof,base_test,None,None,sp["adaptive"])
            return z
        names=sp["occ_names"];ptest_map={n:np.asarray(test[f"p_{n}"]) for n in names}
        pmeta=final_meta_occ(bank,test,names,ptest_map,1.7,sp["leaves"])
        pkey=f"p_metaocc_l{sp['leaves']}"
        if any(pkey not in bank[f] for f in FOLDS):
            pm=walk_meta_occ(bank,names,1.7,sp["leaves"])
            for ff in FOLDS:bank[ff][pkey]=pm[ff]
        risk_oof=walk_risk_gate(bank,names,1.7) if sp["risk"] else None
        risk_test=final_risk_gate(bank,test,names,ptest_map,1.7) if sp["risk"] else None
        z,_=final_occ_candidate(bank,test,pkey,pmeta,base_oof,base_test,risk_oof,risk_test,True)
        return z

    # Finalize A robustly: if its optional raw dependency is unavailable, move to
    # the next validated A-family candidate rather than crashing at the end.
    A_pool=[A]+[r for r in sorted(rows,key=lambda r:(r["delta"],r["latest_delta"]))
                if r["name"]!=A["name"] and r["family"] in {"adaptive_blend","local_bias","candidate_pband","super_ridge","super_ridge_recent","super_pband","super_simplex","ridge_temporal","ridge_subset"}
                and r["delta"]<0 and r["wins_recent"]>=2 and r["latest_delta"]<=.0001]
    A_table=None
    for q in A_pool:
        try:
            A_table=final_stable(q["name"]);A=q;break
        except Exception as exc:
            append_error(errors,"final_A",q["name"],"test",exc);log("A candidate not finalizable",q["name"],repr(exc))
    if A_table is None:
        raise RuntimeError("No validated A-branch candidate could be finalized")
    A_final=fixed.transform_to_friend(np.asarray(friend["z"],np.float64),test["table_core"],A_table,1.0)

    # Finalize B from the occurrence branch first. If raw occurrence is unavailable
    # or failed validation/finalization, use a genuinely different locally-winning
    # p-band/local-bias candidate.
    B_pool=[B]+[r for r in sorted(rows,key=lambda r:(r["delta"],r["latest_delta"]))
                if r["name"] not in {A["name"],B["name"]} and r["delta"]<0 and r["wins_recent"]>=2 and r["latest_delta"]<=.0001
                and r["family"] in {"occurrence_overlay","occurrence_meta","occurrence_meta_risk","local_bias","candidate_pband","pband","super_pband","simplex","greedy"}]
    B_table=None;B_final=None;dAB=None
    for q in B_pool:
        try:
            tb=finalize_occ_row(q);zq=fixed.transform_to_friend(np.asarray(friend["z"],np.float64),test["table_core"],tb,1.0);dd=candidate_distance(A_final,zq)
            # First accept robust occurrence even if moderately close. Non-occurrence
            # fallback must provide visible distributional diversity.
            distinct=(dd["std"]>=.004 or dd["pct02"]>=.06)
            if q["family"].startswith("occurrence") or distinct:
                B=q;B_table=tb;B_final=zq;dAB=dd;break
        except Exception as exc:
            append_error(errors,"final_B",q["name"],"test",exc);log("B candidate not finalizable",q["name"],repr(exc))
    if B_final is None:
        # Last-resort locally improving finalizable candidate, still never a
        # deliberately bad diversity file.
        for q in sorted(rows,key=lambda r:(r["delta"],r["latest_delta"])):
            if q["name"]==A["name"] or q["delta"]>=0 or q["wins_recent"]<2:continue
            try:
                tb=finalize_occ_row(q);zq=fixed.transform_to_friend(np.asarray(friend["z"],np.float64),test["table_core"],tb,1.0);dd=candidate_distance(A_final,zq)
                B=q;B_table=tb;B_final=zq;dAB=dd;break
            except Exception:continue
    if B_final is None:raise RuntimeError("No validated B-branch candidate could be finalized")

    # Final validation metadata and public-anchor geometry.
    oldsubs=locate_old_submissions(base,np.asarray(friend["uid"],np.int64))
    metrics=[]
    for label,r,z in (("A",A,A_final),("B",B,B_final)):
        dF=candidate_distance(z,friend["z"]);row={"branch":label,"name":r["name"],"family":r["family"],"delta_table":r["delta"],"wins_recent":r["wins_recent"],"latest_delta":r["latest_delta"],**{f"friend_{k}":v for k,v in dF.items()}}
        for on,oz in oldsubs.items():row[f"corr__{on}"]=candidate_distance(z,oz)["corr"]
        metrics.append(row)
    save_csv(results/"FINAL_TWO_METRICS.csv",metrics)
    save_csv(results/"PAIR_DIVERSITY.csv",[{"a":A["name"],"b":B["name"],**dAB}])

    import pandas as pd
    sample_df=pd.read_csv(sample);suid=sample_df["user_id"].to_numpy(np.int64)
    submissions=[]
    for label,r,z in (("A",A,A_final),("B",B,B_final)):
        zz=align(friend["uid"],z,suid);pred=np.maximum(np.expm1(np.clip(zz,0,20)),0)
        df=pd.DataFrame({"user_id":suid,"predict":pred})
        if len(df)!=250000 or df.user_id.duplicated().any() or df.predict.isna().any() or (df.predict<0).any():raise RuntimeError("bad submission")
        fn=subs/f"submission_final6h_{label}_{r['name']}.csv";df.to_csv(fn,index=False);submissions.append({"branch":label,"name":r["name"],"file":str(fn),"delta_table":r["delta"],"latest_delta":r["latest_delta"],"family":r["family"],**candidate_distance(z,friend["z"])})
        log("SUBMISSION",label,r["name"],f"d={r['delta']:+.6f}","corr_friend",f"{submissions[-1]['corr']:.6f}")
    save_csv(results/"FINAL_SUBMISSIONS.csv",submissions)

    # Detailed report / bundle.
    runtime_h=(time.time()-started)/3600
    manifest.update({"finished":now(),"runtime_hours":runtime_h,"remaining_hours":budget.remaining,"completed_occurrence_families":completed_occ,"branch_A":submissions[0],"branch_B":submissions[1]})
    atomic_json(results/"RUN_MANIFEST.json",manifest)
    lines=["E-CUP final 6h continuation",f"runtime_hours={runtime_h:.3f}","STRONGEST_CURRENT was NEVER retrained.",f"known_friend_public={KNOWN_FRIEND_PUBLIC}",f"known_previous_ridge_public={KNOWN_RIDGE_SUB_PUBLIC}","",f"Completed occurrence families: {completed_occ}","","Best by family:"]
    for x in sorted(famrows,key=lambda r:r["delta"]):lines.append(f"{x['family']:26s} {x['best_name']:60s} d={x['delta']:+.6f} recent={x['wins_recent']}/3 latest={x['latest_delta']:+.6f}")
    lines += ["","Final submissions:"]+[f"{x['branch']}: {x['name']} d={x['delta_table']:+.6f} corr_friend={x['corr']:.6f} std={x['std']:.5f} file={x['file']}" for x in submissions]
    (results/"REPORT_RU.txt").write_text("\n".join(lines),encoding="utf-8")
    bundle=base/f"final6h_REVIEW_BUNDLE_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    with zipfile.ZipFile(bundle,"w",zipfile.ZIP_DEFLATED) as zf:
        for p in results.iterdir():
            if p.is_file() and p.suffix.lower() in {".csv",".json",".txt",".jsonl"}:zf.write(p,arcname=f"results/{p.name}")
        for p in subs.glob("submission_final6h_*.csv"):zf.write(p,arcname=f"submissions/{p.name}")
    log("DONE",f"{runtime_h:.2f}h","remaining",f"{budget.remaining:.2f}h")
    log("BUNDLE",bundle)


if __name__=="__main__":
    main()

```
