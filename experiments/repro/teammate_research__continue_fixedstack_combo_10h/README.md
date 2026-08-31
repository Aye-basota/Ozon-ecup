# !/usr/bin/env python3

## Catalogue metadata

- **Catalogue ID:** `teammate_research__continue_fixedstack_combo_10h`
- **Namespace:** `teammate_research`
- **Experiment ID:** `continue_fixedstack_combo_10h`
- **Original source:** `пайплайн сокомандника/research_scripts/continue_fixedstack_combo_10h.py`
- **Source ref:** `working tree`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** teammate research runner
- **Model:** LightGBM, Ridge, two-part / hurdle, blend
- **Features:** recency, freshness/conditional features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** E-CUP 2026 / Track 3 — continuation after fixedstack validation (9–10h).
- **Known score:** Unknown / not recoverable from repository history
- **Seed:** for r in seeds:
- **Postprocessing:** # Candidate-level nonnegative stack.  Uses only already cross-fitted candidate predictions.
- **Submission:** path=subs/f"submission_combo10h_candidate_{i}_{r['name']}.csv";df.to_csv(path,index=False)
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when the inputs/checkpoints named by the preserved runner are available

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# continue_fixedstack_combo_10h

Original script: `пайплайн сокомандника/research_scripts/continue_fixedstack_combo_10h.py`

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E-CUP 2026 / Track 3 — continuation after fixedstack validation (9–10h).

This script is intentionally a CONTINUATION, not a restart.

What it reuses
--------------
* teammate STRONGEST_CURRENT submission/prediction package (never retrained);
* _best_bas_research/checkpoints/folds/{cap,unc,dist,hurdle}__*.npz;
* already completed OOF experts from run_best_bas_fixedstack_14h_v2.py:
  recent_hurdle, multiscale_direct, recent_direct, recent_dist;
* already completed final test experts if present:
  multiscale_direct_test, recent_direct_test, recent_dist_test;
* previous bad-LB submissions only as NEGATIVE-DIRECTION diagnostics.

What is new
-----------
1) A wider, but still leakage-safe, search around the actually winning Ridge signal:
   * Ridge shrink / recency weighting / recent-fold training;
   * Ridge expert-subset search (do weak raw experts really help only jointly?);
   * candidate-level adaptive blends and nonnegative stacking;
   * local p-band trust (how much of the Ridge correction to apply in each regime);
   * p x disagreement local trust;
   * p-band and p x disagreement residual-bias calibration;
   * hierarchical combinations of the above.
2) Two NEW all-user hurdle experts with explicitly recent training windows.  They
   are trained on every user, but use fewer historical cutoffs, which is both a
   temporal-shift experiment and a memory-safe alternative to the failed huge
   final recent_hurdle fit.
3) Final raw models are trained in FRESH CHILD PROCESSES one at a time.  The OOF
   bank is released before spawning children, eliminating the memory fragmentation
   that caused LightGBMError: bad allocation in the previous run.
4) Final submissions are emitted only from locally improving recipes.  We prefer
   3/3 recent-fold wins and different error directions; a deliberately bad model
   can no longer be selected merely because it is diverse.

Expected placement
------------------
  src/DL/best_bas/continue_fixedstack_combo_10h.py
  src/DL/best_bas/run_best_bas_fixedstack_14h_v2.py
  src/DL/best_bas/run_best_bas_research_23h.py
  src/DL/best_bas/submission_STRONGEST_CURRENT/
  src/DL/best_bas/_best_bas_research/

Run
---
  python continue_fixedstack_combo_10h.py --max-hours 9.5

The script is resumable.  Re-running it reuses all valid .npz checkpoints.
"""
from __future__ import annotations

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

SCRIPT_VERSION = "fixedstack_combo_10h_2026-08-23_001"
FOLDS = ("2025-09-04", "2025-09-18", "2025-10-02", "2025-10-16")
FW = np.asarray([1.0, 2.0, 4.0, 8.0], dtype=np.float64)
TABLE_WEIGHT = 0.55
LEVEL = 2.3293
EPS = 1e-7


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
    if isinstance(x, dt.date):
        return x.isoformat()
    raise TypeError(type(x).__name__)


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=jdefault), encoding="utf-8")
    os.replace(tmp, path)


def save_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    import pandas as pd
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = []
    for r in rows:
        q = {}
        for k, v in r.items():
            if isinstance(v, (dict, list, tuple, np.ndarray)):
                q[k] = json.dumps(v, ensure_ascii=False, default=jdefault)
            else:
                q[k] = v
        flat.append(q)
    pd.DataFrame(flat).to_csv(path, index=False)


def append_error(path: Path, stage: str, name: str, fold: str, exc: BaseException) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "time": now(), "stage": stage, "name": name, "fold": fold,
            "error": repr(exc), "traceback": traceback.format_exc(),
        }, ensure_ascii=False) + "\n")


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def import_fixed(base: Path):
    cands = [base / "run_best_bas_fixedstack_14h_v2.py", base / "run_best_bas_fixedstack_14h.py"]
    cands += sorted(base.glob("run_best_bas_fixedstack*.py"), key=lambda p: p.stat().st_mtime, reverse=True)
    seen = set()
    for p in cands:
        if not p.exists() or p.resolve() == Path(__file__).resolve() or p in seen:
            continue
        seen.add(p)
        try:
            m = import_module(p, "fixedstack_parent")
            need = ("import_prev", "discover_work", "load_core", "research", "valid_npz", "load_npz",
                    "candidate_distance", "transform_to_friend", "component_test", "repair_cache", "install_atomic_cache")
            if all(hasattr(m, x) for x in need):
                return m, p
        except Exception:
            continue
    raise FileNotFoundError("run_best_bas_fixedstack_14h_v2.py not found beside continuation script")


def clipz(z):
    return np.clip(np.nan_to_num(np.asarray(z, np.float64), nan=0.0, posinf=20.0, neginf=0.0), 0.0, 20.0)


def truez(y):
    return np.log1p(np.maximum(np.asarray(y, np.float64), 0.0))


def rms_z(y, z):
    return float(np.sqrt(np.mean((truez(y) - clipz(z)) ** 2)))


def wavg(x):
    return float(np.dot(np.asarray(x, np.float64), FW) / FW.sum())


def align(src_uid, arr, dst_uid):
    su = np.asarray(src_uid, np.int64)
    du = np.asarray(dst_uid, np.int64)
    a = np.asarray(arr)
    if np.array_equal(su, du):
        return a
    o = np.argsort(su)
    ss = su[o]
    p = np.searchsorted(ss, du)
    if (p >= len(ss)).any() or not np.array_equal(ss[p], du):
        raise ValueError("user_id sets differ")
    return a[o[p]]


def level_test(z, level=LEVEL):
    z = clipz(z)
    d = float(level - z.mean())
    return clipz(z + d), d


def candidate_distance(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    d = a - b
    c = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else 1.0
    return {
        "corr": c,
        "mean_abs": float(np.mean(np.abs(d))),
        "std": float(np.std(d)),
        "pct02": float(np.mean(np.abs(d) > .02)),
        "pct05": float(np.mean(np.abs(d) > .05)),
        "pct10": float(np.mean(np.abs(d) > .10)),
    }


@dataclasses.dataclass
class Budget:
    started: float
    max_hours: float
    final_reserve: float = 1.25

    @property
    def elapsed(self):
        return (time.time() - self.started) / 3600.0

    @property
    def remaining(self):
        return self.max_hours - self.elapsed

    def can_start(self, estimate_h: float, extra: float = 0.0):
        return self.remaining > estimate_h + extra + self.final_reserve


# -----------------------------------------------------------------------------
# checkpoint helpers
# -----------------------------------------------------------------------------

def npz_valid(path: Path, need=("user_id", "z")) -> bool:
    if not path.exists() or path.stat().st_size < 256:
        return False
    try:
        with np.load(path, allow_pickle=False) as d:
            if not all(k in d.files for k in need):
                return False
            n = len(d["user_id"])
            if n == 0 or len(d["z"]) != n:
                return False
            if not np.isfinite(d["z"][: min(n, 1000)]).all():
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


# -----------------------------------------------------------------------------
# already-completed OOF expert loading
# -----------------------------------------------------------------------------

EXISTING_NEW = ("recent_hurdle", "multiscale_direct", "recent_direct", "recent_dist")
MEM_VARIANTS = {
    # name: max recent cutoffs, temporal tau, rounds
    "recent_hurdle_fast12": (12, 70.0, 430),
    "recent_hurdle_stable18": (18, 135.0, 460),
}


def load_existing_new_oof(fixed, ctx, bank) -> list[str]:
    loaded = []
    for name in EXISTING_NEW:
        ok = True
        parts = {}
        for f in FOLDS:
            p = Path(ctx.checkpoints) / "folds" / f"{name}__{f}.npz"
            need = ("user_id", "y", "z", "p", "mu") if name == "recent_hurdle" else ("user_id", "y", "z")
            if not npz_valid(p, need):
                ok = False
                break
            parts[f] = load_npz(p)
        if not ok:
            log("OOF existing new expert incomplete -> not used", name)
            continue
        for f in FOLDS:
            r = bank[f]
            d = parts[f]
            r[name] = align(d["user_id"], d["z"], r["uid"]).astype(np.float32)
            if name == "recent_hurdle":
                r["p_recent_hurdle"] = align(d["user_id"], d["p"], r["uid"]).astype(np.float32)
                r["mu_recent_hurdle"] = align(d["user_id"], d["mu"], r["uid"]).astype(np.float32)
        loaded.append(name)
        log("REUSED OOF", name)
    return loaded


def variant_fold_path(ctx, name: str, fold: str) -> Path:
    return Path(ctx.checkpoints) / "folds" / f"{name}__{fold}.npz"


def variant_test_path(ctx, name: str) -> Path:
    return Path(ctx.checkpoints) / "test" / f"{name}_test.npz"


def train_hurdle_variant_fold(ctx, fixed, name: str, fold: str) -> dict[str, np.ndarray]:
    """All-user hurdle with a capped recent cutoff window.

    Fold training is small enough for the normal two_part implementation.  The
    final test version is trained in a clean child process with a lower-memory
    sequential algorithm.
    """
    p = variant_fold_path(ctx, name, fold)
    if npz_valid(p, ("user_id", "y", "z", "p", "mu")):
        log("reuse OOF memory-safe hurdle", p.name)
        return load_npz(p)
    if name not in MEM_VARIANTS:
        raise KeyError(name)
    maxcuts, tau, rounds = MEM_VARIANTS[name]
    T, F, M = ctx.train_mod, ctx.features_mod, ctx.models_mod
    s, _ = fixed.expert_spec(ctx, "recent_hurdle")
    s.weight_tau = tau
    s.rounds = rounds
    V = dt.date.fromisoformat(fold)
    cuts = list(s.train_cutoffs(V))[-maxcuts:]
    Xv, yv = T.xy(V, s)
    feats = T.select_features(F.feature_names(Xv), s.drop_groups, s.keep_only)
    log("NEW ALL-USER", name, fold, "recent cuts", len(cuts), "features", len(feats))
    X, y, w = T.assemble(cuts, s, feats, V)
    n = len(y)
    T._XY.clear()
    box = [X]
    del X
    gc.collect()
    model = T.fit_free(s, box, y, w)
    A = F.to_np(Xv, feats)
    clf, reg = model
    pp = np.clip(clf.predict(A), EPS, 1 - EPS)
    mu = np.maximum(reg.predict(A), 0.0)
    z = pp * mu
    save_npz_atomic(p,
                    user_id=Xv["user_id"].to_numpy().astype(np.int64),
                    y=np.asarray(yv, np.float32), z=np.asarray(clipz(z), np.float32),
                    p=np.asarray(pp, np.float32), mu=np.asarray(mu, np.float32))
    off, sc = fixed.calibrate(yv, z)
    log("NEW DONE", name, fold, "rows", f"{n:,}", "cal", f"{sc:.6f}", "off", f"{off:+.4f}")
    del model, A, Xv, y, w, box, pp, mu, z
    T._XY.clear()
    gc.collect()
    return load_npz(p)


def add_variant_oof(ctx, fixed, bank, name: str) -> None:
    for f in FOLDS:
        d = train_hurdle_variant_fold(ctx, fixed, name, f)
        r = bank[f]
        r[name] = align(d["user_id"], d["z"], r["uid"]).astype(np.float32)
        r[f"p_{name}"] = align(d["user_id"], d["p"], r["uid"]).astype(np.float32)
        r[f"mu_{name}"] = align(d["user_id"], d["mu"], r["uid"]).astype(np.float32)


# -----------------------------------------------------------------------------
# meta features / custom Ridge variants
# -----------------------------------------------------------------------------

def expert_spread(rec: dict[str, Any], experts: Sequence[str]) -> np.ndarray:
    P = np.stack([np.asarray(rec[n], np.float32) for n in experts], axis=1)
    return np.std(P, axis=1).astype(np.float32)


def pred_features(rec, experts, include_meta=True):
    P = np.stack([np.asarray(rec[n], np.float32) for n in experts], axis=1)
    core = np.asarray(rec["table_core"], np.float32)
    p = np.asarray(rec["p"], np.float32)
    mu = np.asarray(rec["mu"], np.float32)
    derived = np.column_stack([
        core, p, mu, np.log1p(mu), p * (1 - p),
        P.mean(1), P.std(1), P.min(1), P.max(1), P.max(1) - P.min(1),
    ]).astype(np.float32)
    diffs = np.column_stack([P[:, j] - core for j in range(P.shape[1])]).astype(np.float32)
    pieces = [P, derived, diffs]
    if include_meta:
        pieces.insert(0, np.asarray(rec["meta_raw"], np.float32))
    X = np.column_stack(pieces).astype(np.float32, copy=False)
    return np.nan_to_num(X, nan=0.0, posinf=20.0, neginf=-20.0)


def make_ridge(alpha: float):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    return make_pipeline(StandardScaler(copy=False), Ridge(alpha=float(alpha), solver="lsqr", tol=1e-4))


def concat_custom(bank, ids: Sequence[int], experts: Sequence[str], include_meta: bool,
                  weight_power: float = 1.0):
    Xs, ys, ws = [], [], []
    for j in ids:
        r = bank[FOLDS[j]]
        Xs.append(pred_features(r, experts, include_meta))
        ys.append(np.asarray(r["true_z"] - r["table_core"], np.float32))
        ws.append(np.full(len(r["uid"]), float(FW[j] ** weight_power), np.float32))
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(ws)


@dataclasses.dataclass
class Recipe:
    name: str
    kind: str
    family: str
    params: dict[str, Any]


def walk_ridge(bank, experts: Sequence[str], alpha=150.0, shrink=.75, include_meta=True,
               weight_power=1.0, recent_k: int | None = None):
    out = {}
    for i, f in enumerate(FOLDS):
        r = bank[f]
        if i == 0:
            out[f] = np.asarray(r["table_core"], np.float64).copy()
            continue
        ids = list(range(i))
        if recent_k is not None:
            ids = ids[-int(recent_k):]
        X, y, w = concat_custom(bank, ids, experts, include_meta, weight_power)
        Xt = pred_features(r, experts, include_meta)
        m = make_ridge(alpha)
        m.fit(X, y, **{"ridge__sample_weight": w})
        d = np.clip(m.predict(Xt), -2.0, 2.0)
        out[f] = clipz(np.asarray(r["table_core"], np.float64) + float(shrink) * d)
        del X, y, w, Xt, d, m
        gc.collect()
    return out


def final_ridge(bank, test, experts: Sequence[str], alpha=150.0, shrink=.75,
                include_meta=True, weight_power=1.0, recent_k: int | None = None):
    ids = list(range(4))
    if recent_k is not None:
        ids = ids[-int(recent_k):]
    X, y, w = concat_custom(bank, ids, experts, include_meta, weight_power)
    Xt = pred_features(test, experts, include_meta)
    m = make_ridge(alpha)
    m.fit(X, y, **{"ridge__sample_weight": w})
    d = np.clip(m.predict(Xt), -2.0, 2.0)
    z = clipz(np.asarray(test["table_core"], np.float64) + float(shrink) * d)
    del X, y, w, Xt, d, m
    gc.collect()
    return z


def score_candidate(fixed, name: str, p: dict[str, np.ndarray], bank, family: str,
                    rows: list[dict[str, Any]], notes=""):
    return fixed.score_table(name, p, bank, family, rows, notes=notes)


# -----------------------------------------------------------------------------
# candidate-level safe combiners
# -----------------------------------------------------------------------------

def fit_blend_alpha(bank, ids, A, B, prior=.75, lam=.12):
    num = 0.0
    den = 0.0
    count = 0.0
    for j in ids:
        f = FOLDS[j]
        y = np.asarray(bank[f]["true_z"], np.float64)
        a = np.asarray(A[f], np.float64)
        b = np.asarray(B[f], np.float64)
        # Remove fold intercept because score_table calibrates a global offset.
        y = y - y.mean()
        a = a - a.mean()
        b = b - b.mean()
        d = a - b
        w = float(FW[j])
        num += w * float(np.dot(d, y - b))
        den += w * float(np.dot(d, d))
        count += w * len(y)
    if den <= 1e-12:
        raw = prior
    else:
        raw = num / den
    # ridge toward prior, expressed on the natural alpha scale
    strength = den / max(count, 1.0)
    alpha = (strength * raw + lam * prior) / (strength + lam)
    return float(np.clip(alpha, 0.0, 1.0))


def walk_adaptive_blend(bank, A, B, prior=.75, lam=.12):
    out, pars = {}, []
    for i, f in enumerate(FOLDS):
        if i == 0:
            out[f] = np.asarray(bank[f]["table_core"], np.float64).copy()
            pars.append(None)
            continue
        a = fit_blend_alpha(bank, list(range(i)), A, B, prior, lam)
        out[f] = clipz(a * np.asarray(A[f]) + (1 - a) * np.asarray(B[f]))
        pars.append(a)
    return out, pars


def final_adaptive_blend(bank, At, Bt, A_oof, B_oof, prior=.75, lam=.12):
    a = fit_blend_alpha(bank, list(range(4)), A_oof, B_oof, prior, lam)
    return clipz(a * At + (1 - a) * Bt), a


def fit_candidate_simplex(bank, ids, names, predpool, lam=.12, prior_name: str | None = None):
    from scipy.optimize import minimize
    Zs, ys, ws = [], [], []
    for j in ids:
        f = FOLDS[j]
        Zs.append(np.stack([predpool[n][f] for n in names], axis=1).astype(np.float64))
        ys.append(np.asarray(bank[f]["true_z"], np.float64))
        ws.append(np.full(len(bank[f]["uid"]), FW[j], np.float64))
    Z = np.vstack(Zs)
    y = np.concatenate(ys)
    sw = np.concatenate(ws)
    sw /= sw.mean()
    prior = np.full(len(names), 1.0 / len(names), np.float64)
    if prior_name in names:
        prior[:] = .15 / max(len(names) - 1, 1)
        prior[names.index(prior_name)] = .85
        prior /= prior.sum()
    # centre once so weights are optimized for shape, not fold-level intercept
    def obj(x):
        pr = Z @ x
        e = pr - y
        return float(np.mean(sw * e * e) + lam * np.sum((x - prior) ** 2))
    res = minimize(obj, prior, method="SLSQP", bounds=[(0, 1)] * len(names),
                   constraints=[{"type": "eq", "fun": lambda x: float(np.sum(x) - 1.0)}],
                   options={"maxiter": 220, "ftol": 1e-10})
    x = np.clip(res.x if res.success else prior, 0, None)
    x /= max(x.sum(), EPS)
    return x


def walk_candidate_simplex(bank, names, predpool, lam=.12, prior_name=None):
    out, weights = {}, []
    for i, f in enumerate(FOLDS):
        if i == 0:
            out[f] = np.asarray(bank[f]["table_core"], np.float64).copy()
            weights.append(None)
            continue
        w = fit_candidate_simplex(bank, list(range(i)), names, predpool, lam, prior_name)
        out[f] = clipz(sum(w[k] * predpool[n][f] for k, n in enumerate(names)))
        weights.append(w)
    return out, weights


def final_candidate_simplex(bank, names, predpool, testpool, lam=.12, prior_name=None):
    w = fit_candidate_simplex(bank, list(range(4)), names, predpool, lam, prior_name)
    return clipz(sum(w[k] * testpool[n] for k, n in enumerate(names))), w


def fit_candidate_pband(bank, ids, names, predpool,
                        bands=(0, .2, .4, .6, .8, 1.0001), lam=.18, prior_name=None):
    global_w = fit_candidate_simplex(bank, ids, names, predpool, lam=.16, prior_name=prior_name)
    from scipy.optimize import minimize
    W = []
    for b in range(len(bands) - 1):
        Zs, ys, ws = [], [], []
        for j in ids:
            f = FOLDS[j]
            r = bank[f]
            m = (r["p"] >= bands[b]) & (r["p"] < bands[b + 1])
            if int(m.sum()) < 3000:
                continue
            Zs.append(np.stack([predpool[n][f][m] for n in names], axis=1))
            ys.append(np.asarray(r["true_z"])[m])
            ws.append(np.full(int(m.sum()), FW[j]))
        if not Zs:
            W.append(global_w.copy())
            continue
        Z = np.vstack(Zs).astype(np.float64)
        y = np.concatenate(ys).astype(np.float64)
        sw = np.concatenate(ws).astype(np.float64)
        sw /= sw.mean()
        def obj(x):
            e = Z @ x - y
            return float(np.mean(sw * e * e) + lam * np.sum((x - global_w) ** 2))
        res = minimize(obj, global_w, method="SLSQP", bounds=[(0, 1)] * len(names),
                       constraints=[{"type": "eq", "fun": lambda x: float(np.sum(x) - 1)}],
                       options={"maxiter": 180, "ftol": 1e-10})
        x = np.clip(res.x if res.success else global_w, 0, None)
        x /= max(x.sum(), EPS)
        W.append(x)
    return np.asarray(W), np.asarray(bands, np.float64)


def apply_candidate_pband(rec, names, testpool, W, bands):
    p = np.asarray(rec["p"], np.float64)
    P = np.stack([testpool[n] for n in names], axis=1)
    z = np.empty(len(p), np.float64)
    idx = np.clip(np.searchsorted(bands, p, side="right") - 1, 0, len(bands) - 2)
    for b in range(len(bands) - 1):
        m = idx == b
        if m.any():
            z[m] = P[m] @ W[b]
    return clipz(z)


def walk_candidate_pband(bank, names, predpool, lam=.18, prior_name=None):
    out, pars = {}, []
    for i, f in enumerate(FOLDS):
        if i == 0:
            out[f] = np.asarray(bank[f]["table_core"], np.float64).copy()
            pars.append(None)
            continue
        W, bands = fit_candidate_pband(bank, list(range(i)), names, predpool, lam=lam, prior_name=prior_name)
        out[f] = apply_candidate_pband(bank[f], names, {n: predpool[n][f] for n in names}, W, bands)
        pars.append(W)
    return out, pars


# -----------------------------------------------------------------------------
# local trust / residual calibration on top of an already-good candidate
# -----------------------------------------------------------------------------

def fit_band_alpha(bank, ids, cand_oof, experts, two_d=False, shrink_n=12000.0):
    pbands = np.asarray([0, .15, .30, .50, .70, .85, 1.0001], np.float64)
    # spread thresholds are global over train ids to keep application deterministic
    spreads = np.concatenate([expert_spread(bank[FOLDS[j]], experts) for j in ids])
    sq = np.quantile(spreads, [1/3, 2/3]) if two_d and len(spreads) else np.asarray([])
    shape = (len(pbands) - 1, 3 if two_d else 1)
    num = np.zeros(shape, np.float64)
    den = np.zeros(shape, np.float64)
    cnt = np.zeros(shape, np.float64)
    # First compute a global alpha for shrinkage target.
    gnum = gden = 0.0
    for j in ids:
        f = FOLDS[j]
        r = bank[f]
        d = np.asarray(cand_oof[f], np.float64) - np.asarray(r["table_core"], np.float64)
        target = np.asarray(r["true_z"], np.float64) - np.asarray(r["table_core"], np.float64)
        # remove per-fold intercept
        target -= target.mean(); d -= d.mean()
        w = float(FW[j])
        gnum += w * float(np.dot(d, target)); gden += w * float(np.dot(d, d))
    ga = float(np.clip(gnum / max(gden, 1e-12), 0.0, 1.35))
    for j in ids:
        f = FOLDS[j]
        r = bank[f]
        d = np.asarray(cand_oof[f], np.float64) - np.asarray(r["table_core"], np.float64)
        target = np.asarray(r["true_z"], np.float64) - np.asarray(r["table_core"], np.float64)
        target -= target.mean(); d -= d.mean()
        pi = np.clip(np.searchsorted(pbands, np.asarray(r["p"]), side="right") - 1, 0, len(pbands)-2)
        if two_d:
            sp = expert_spread(r, experts)
            si = np.searchsorted(sq, sp, side="right")
        else:
            si = np.zeros(len(d), np.int8)
        w = float(FW[j])
        for a in range(shape[0]):
            for b in range(shape[1]):
                m = (pi == a) & (si == b)
                if not m.any():
                    continue
                dm, tm = d[m], target[m]
                num[a, b] += w * float(np.dot(dm, tm))
                den[a, b] += w * float(np.dot(dm, dm))
                cnt[a, b] += w * int(m.sum())
    A = np.full(shape, ga, np.float64)
    for a in range(shape[0]):
        for b in range(shape[1]):
            raw = num[a, b] / max(den[a, b], 1e-12)
            raw = float(np.clip(raw, 0.0, 1.35))
            s = cnt[a, b] / (cnt[a, b] + shrink_n)
            A[a, b] = s * raw + (1 - s) * ga
    return {"pbands": pbands, "spread_q": sq, "alpha": A, "global_alpha": ga}


def apply_band_alpha(rec, base_z, params, experts):
    pb = params["pbands"]
    A = params["alpha"]
    pi = np.clip(np.searchsorted(pb, np.asarray(rec["p"]), side="right") - 1, 0, len(pb)-2)
    if A.shape[1] > 1:
        sp = expert_spread(rec, experts)
        si = np.searchsorted(params["spread_q"], sp, side="right")
    else:
        si = np.zeros(len(pi), np.int8)
    core = np.asarray(rec["table_core"], np.float64)
    d = np.asarray(base_z, np.float64) - core
    aa = A[pi, si]
    return clipz(core + aa * d)


def walk_band_alpha(bank, base_oof, experts, two_d=False, shrink_n=12000.0):
    out, pars = {}, []
    for i, f in enumerate(FOLDS):
        if i == 0:
            out[f] = np.asarray(bank[f]["table_core"], np.float64).copy(); pars.append(None); continue
        par = fit_band_alpha(bank, list(range(i)), base_oof, experts, two_d=two_d, shrink_n=shrink_n)
        out[f] = apply_band_alpha(bank[f], base_oof[f], par, experts)
        pars.append(par)
    return out, pars


def fit_bias_cells(bank, ids, base_oof, experts, two_d=False, shrink_n=15000.0, strength=.65):
    pbands = np.asarray([0, .15, .30, .50, .70, .85, 1.0001], np.float64)
    spreads = np.concatenate([expert_spread(bank[FOLDS[j]], experts) for j in ids])
    sq = np.quantile(spreads, [1/3, 2/3]) if two_d and len(spreads) else np.asarray([])
    shape = (len(pbands)-1, 3 if two_d else 1)
    sm = np.zeros(shape, np.float64); wt = np.zeros(shape, np.float64)
    for j in ids:
        f = FOLDS[j]; r = bank[f]
        resid = np.asarray(r["true_z"], np.float64) - np.asarray(base_oof[f], np.float64)
        # Global mean is handled by calibration / final level, so only local deviation matters.
        resid -= resid.mean()
        pi = np.clip(np.searchsorted(pbands, np.asarray(r["p"]), side="right") - 1, 0, len(pbands)-2)
        if two_d:
            si = np.searchsorted(sq, expert_spread(r, experts), side="right")
        else:
            si = np.zeros(len(resid), np.int8)
        w = float(FW[j])
        for a in range(shape[0]):
            for b in range(shape[1]):
                m = (pi == a) & (si == b)
                if not m.any(): continue
                sm[a,b] += w * float(resid[m].sum()); wt[a,b] += w * int(m.sum())
    B = np.zeros(shape, np.float64)
    for a in range(shape[0]):
        for b in range(shape[1]):
            raw = sm[a,b] / max(wt[a,b], 1e-12)
            s = wt[a,b] / (wt[a,b] + shrink_n)
            B[a,b] = float(np.clip(strength * s * raw, -.18, .18))
    return {"pbands": pbands, "spread_q": sq, "bias": B}


def apply_bias_cells(rec, base_z, params, experts):
    pb = params["pbands"]; B = params["bias"]
    pi = np.clip(np.searchsorted(pb, np.asarray(rec["p"]), side="right") - 1, 0, len(pb)-2)
    if B.shape[1] > 1:
        si = np.searchsorted(params["spread_q"], expert_spread(rec, experts), side="right")
    else:
        si = np.zeros(len(pi), np.int8)
    return clipz(np.asarray(base_z, np.float64) + B[pi, si])


def walk_bias_cells(bank, base_oof, experts, two_d=False, shrink_n=15000.0, strength=.65):
    out, pars = {}, []
    for i, f in enumerate(FOLDS):
        if i == 0:
            out[f] = np.asarray(bank[f]["table_core"], np.float64).copy(); pars.append(None); continue
        par = fit_bias_cells(bank, list(range(i)), base_oof, experts, two_d, shrink_n, strength)
        out[f] = apply_bias_cells(bank[f], base_oof[f], par, experts)
        pars.append(par)
    return out, pars


# -----------------------------------------------------------------------------
# research construction
# -----------------------------------------------------------------------------

def finalizable_experts(bank) -> list[str]:
    # Original recent_hurdle deliberately excluded: its full test refit caused the
    # bad-allocation crash and added only ~5e-5 to Ridge in the prior run.
    names = ["cap", "unc", "dist", "hurdle", "multiscale_direct", "recent_direct", "recent_dist",
             "recent_hurdle_fast12", "recent_hurdle_stable18"]
    return [n for n in names if all(n in bank[f] for f in FOLDS)]


def build_primitive_research(fixed, bank, results: Path, tag: str):
    rows: list[dict[str, Any]] = []
    predpool: dict[str, dict[str, np.ndarray]] = {}
    recipes: dict[str, Recipe] = {}
    ex = finalizable_experts(bank)
    core = [n for n in ("cap", "unc", "dist", "hurdle") if n in ex]
    new = [n for n in ex if n not in core]

    def add(name, p, family, recipe, notes=""):
        score_candidate(fixed, name, p, bank, family, rows, notes)
        predpool[name] = p; recipes[name] = recipe

    # Raw experts for diagnostics, not necessarily output candidates.
    for n in new:
        p = {f: np.asarray(bank[f][n], np.float64) for f in FOLDS}
        add(n, p, "raw_new", Recipe(n, "raw", "raw_new", {"expert": n}))

    # Expert-set search.  We do not assume that every individually weak model
    # should enter the same Ridge.  This is cheap and was not explored enough in
    # the crashed run.
    sets: list[tuple[str, list[str]]] = [("all", ex), ("core", core)]
    if new:
        for drop in new:
            sets.append((f"drop_{drop}", [x for x in ex if x != drop]))
        for n in new:
            sets.append((f"core_plus_{n}", core + [n]))
    uniq = []
    seen = set()
    for lab, xs in sets:
        key = tuple(xs)
        if key not in seen and len(xs) >= 4:
            seen.add(key); uniq.append((lab, xs))

    for lab, xs in uniq:
        p = walk_ridge(bank, xs, alpha=150, shrink=.75, include_meta=True)
        add(f"ridge_{lab}_s075", p, "ridge_subset",
            Recipe(f"ridge_{lab}_s075", "ridge", "ridge_subset",
                   {"experts": xs, "alpha": 150., "shrink": .75, "meta": True, "weight_power": 1., "recent_k": None}),
            notes=f"experts={xs}")

    # Shrink sweep around the stable all/finalizable expert set.
    for s in (.55, .65, .75, .85, .95):
        name = f"ridge_all_s{int(s*100):02d}"
        if name in predpool: continue
        p = walk_ridge(bank, ex, alpha=150, shrink=s, include_meta=True)
        add(name, p, "ridge_shrink", Recipe(name, "ridge", "ridge_shrink",
                                             {"experts": ex, "alpha":150., "shrink":s, "meta":True,
                                              "weight_power":1., "recent_k":None}))

    # More aggressive temporal weighting / last-two-fold meta training.
    for power in (1.35, 1.70):
        name = f"ridge_recentpow{str(power).replace('.','p')}_s075"
        p = walk_ridge(bank, ex, alpha=150, shrink=.75, include_meta=True, weight_power=power)
        add(name, p, "ridge_temporal", Recipe(name, "ridge", "ridge_temporal",
                                               {"experts": ex, "alpha":150., "shrink":.75, "meta":True,
                                                "weight_power":power, "recent_k":None}))
    name = "ridge_recent2_s075"
    p = walk_ridge(bank, ex, alpha=150, shrink=.75, include_meta=True, recent_k=2)
    add(name, p, "ridge_temporal", Recipe(name, "ridge", "ridge_temporal",
                                           {"experts": ex, "alpha":150., "shrink":.75, "meta":True,
                                            "weight_power":1., "recent_k":2}))

    # Prediction-only Ridge is structurally different and previously improved 3/3.
    name = "ridge_predonly_finalizable"
    p = walk_ridge(bank, ex, alpha=80, shrink=1.0, include_meta=False)
    add(name, p, "ridge_predonly", Recipe(name, "ridge", "ridge_predonly",
                                           {"experts": ex, "alpha":80., "shrink":1., "meta":False,
                                            "weight_power":1., "recent_k":None}))

    # Nonnegative methods on the same finalizable raw bank.
    p, _ = fixed.walk_simplex(bank, ex, lam=.02, min_new=0, new_names=())
    add("simplex_finalizable", p, "simplex", Recipe("simplex_finalizable", "raw_simplex", "simplex", {"experts": ex, "lam": .02}))
    p, _ = fixed.walk_greedy(bank, ex, steps=35)
    add("greedy35_finalizable", p, "greedy", Recipe("greedy35_finalizable", "raw_greedy", "greedy", {"experts": ex, "steps":35}))
    p, _ = fixed.walk_pband(bank, ex)
    add("pband_finalizable", p, "pband", Recipe("pband_finalizable", "raw_pband", "pband", {"experts": ex}))
    p = fixed.walk_occ(bank, guard=True)
    add("occ_platt_guard", p, "occ_cal", Recipe("occ_platt_guard", "occ", "occ_cal", {"guard": True}))

    rows.sort(key=lambda r: (r["delta"], r["worst_delta"]))
    save_csv(results / f"primitive_validation_{tag}.csv", rows)
    return rows, predpool, recipes, ex


def build_combo_research(fixed, bank, rows, predpool, recipes, experts, results: Path, tag: str):
    """Build only combinations of candidates that already improve or are strong complementary controls."""
    def add(name, p, family, recipe, notes=""):
        if name in predpool: return
        score_candidate(fixed, name, p, bank, family, rows, notes)
        predpool[name] = p; recipes[name] = recipe

    ranked = sorted(rows, key=lambda r: (r["delta"], r["latest_delta"]))
    good = [r["name"] for r in ranked if r["delta"] < 0 and r["wins_recent"] >= 2]
    ridge_names = [n for n in good if recipes[n].family.startswith("ridge")]
    R = ridge_names[0] if ridge_names else good[0]
    alternatives = []
    for cand in ("pband_finalizable", "ridge_predonly_finalizable", "greedy35_finalizable",
                 "simplex_finalizable", "occ_platt_guard"):
        if cand in predpool and cand != R:
            alternatives.append(cand)

    # Adaptive pairwise blends.  Weight is fitted only on PAST folds.
    for B in alternatives[:5]:
        for prior in (.70, .85):
            name = f"blend_{R}__{B}_pr{int(prior*100)}"
            p, _ = walk_adaptive_blend(bank, predpool[R], predpool[B], prior=prior, lam=.12)
            add(name, p, "adaptive_blend", Recipe(name, "blend", "adaptive_blend",
                                                   {"a":R,"b":B,"prior":prior,"lam":.12}))

    # Candidate-level nonnegative stack.  Uses only already cross-fitted candidate predictions.
    top_stack = []
    for n in [R, "pband_finalizable", "ridge_predonly_finalizable", "greedy35_finalizable", "simplex_finalizable", "occ_platt_guard"]:
        if n in predpool and n not in top_stack:
            top_stack.append(n)
    if len(top_stack) >= 3:
        for lam in (.08, .20):
            name = f"cand_simplex_l{str(lam).replace('.','p')}"
            p, _ = walk_candidate_simplex(bank, top_stack, predpool, lam=lam, prior_name=R)
            add(name, p, "candidate_simplex", Recipe(name, "cand_simplex", "candidate_simplex",
                                                      {"names":top_stack,"lam":lam,"prior":R}))
        name = "cand_pband_stack"
        p, _ = walk_candidate_pband(bank, top_stack[:5], predpool, lam=.20, prior_name=R)
        add(name, p, "candidate_pband", Recipe(name, "cand_pband", "candidate_pband",
                                                {"names":top_stack[:5],"lam":.20,"prior":R}))

    # Trust the Ridge correction differently by purchase-probability regime and by
    # raw-expert disagreement.  This directly combines the successful Ridge and
    # Phase12 local-specialist ideas without hard routing.
    for two_d, suffix, shrink_n in ((False, "p", 12000.), (True, "p_spread", 18000.)):
        name = f"trust_{suffix}_{R}"
        p, _ = walk_band_alpha(bank, predpool[R], experts, two_d=two_d, shrink_n=shrink_n)
        add(name, p, "local_trust", Recipe(name, "band_alpha", "local_trust",
                                            {"base":R,"experts":experts,"two_d":two_d,"shrink_n":shrink_n}))

    # Local bias correction is intentionally low-capacity: error detectors were
    # good at finding errors but magnitude models were unstable.  A shrunk cell
    # mean is much safer than a free additive corrector.
    seed_bases = [R]
    if "cand_pband_stack" in predpool: seed_bases.append("cand_pband_stack")
    for base in seed_bases:
        for two_d, suffix, strength in ((False, "p", .55), (True, "p_spread", .45)):
            name = f"bias_{suffix}_{base}"
            p, _ = walk_bias_cells(bank, predpool[base], experts, two_d=two_d,
                                   shrink_n=18000. if two_d else 14000., strength=strength)
            add(name, p, "local_bias", Recipe(name, "bias", "local_bias",
                                               {"base":base,"experts":experts,"two_d":two_d,
                                                "shrink_n":18000. if two_d else 14000.,"strength":strength}))

    # One explicit hierarchy: trust-shrink first, then local bias.  Both stages are
    # cross-fitted, so this is still honest walk-forward validation.
    trust_name = f"trust_p_spread_{R}"
    if trust_name in predpool:
        name = f"hier_trust_bias_{R}"
        p, _ = walk_bias_cells(bank, predpool[trust_name], experts, two_d=True,
                               shrink_n=22000., strength=.40)
        add(name, p, "hierarchical", Recipe(name, "bias", "hierarchical",
                                             {"base":trust_name,"experts":experts,"two_d":True,
                                              "shrink_n":22000.,"strength":.40}))

    rows.sort(key=lambda r: (r["delta"], r["worst_delta"]))
    save_csv(results / f"combo_validation_{tag}.csv", rows)
    return rows, predpool, recipes


# -----------------------------------------------------------------------------
# finalization recipes
# -----------------------------------------------------------------------------

def fit_final_band_alpha(bank, base_oof, experts, two_d, shrink_n):
    return fit_band_alpha(bank, list(range(4)), base_oof, experts, two_d, shrink_n)


def fit_final_bias(bank, base_oof, experts, two_d, shrink_n, strength):
    return fit_bias_cells(bank, list(range(4)), base_oof, experts, two_d, shrink_n, strength)


def final_raw_simplex(fixed, bank, test, experts, lam):
    return fixed.final_simplex(bank, test, experts, lam=lam, min_new=0, new_names=())[0]


def final_raw_greedy(fixed, bank, test, experts, steps):
    return fixed.final_greedy(bank, test, experts, steps=steps)[0]


def final_raw_pband(fixed, bank, test, experts):
    return fixed.final_pband(bank, test, experts)[0]


def finalize_recipe(name: str, recipes: dict[str, Recipe], predpool, bank, test, fixed,
                    cache: dict[str, np.ndarray]) -> np.ndarray:
    if name in cache:
        return cache[name]
    r = recipes[name]
    p = r.params
    if r.kind == "raw":
        z = np.asarray(test[p["expert"]], np.float64)
    elif r.kind == "ridge":
        z = final_ridge(bank, test, p["experts"], p["alpha"], p["shrink"], p["meta"], p["weight_power"], p["recent_k"])
    elif r.kind == "raw_simplex":
        z = final_raw_simplex(fixed, bank, test, p["experts"], p["lam"])
    elif r.kind == "raw_greedy":
        z = final_raw_greedy(fixed, bank, test, p["experts"], p["steps"])
    elif r.kind == "raw_pband":
        z = final_raw_pband(fixed, bank, test, p["experts"])
    elif r.kind == "occ":
        z = fixed.final_occ(bank, test, p["guard"])
    elif r.kind == "blend":
        a = finalize_recipe(p["a"], recipes, predpool, bank, test, fixed, cache)
        b = finalize_recipe(p["b"], recipes, predpool, bank, test, fixed, cache)
        z, _ = final_adaptive_blend(bank, a, b, predpool[p["a"]], predpool[p["b"]], p["prior"], p["lam"])
    elif r.kind == "cand_simplex":
        names = p["names"]
        testpool = {n: finalize_recipe(n, recipes, predpool, bank, test, fixed, cache) for n in names}
        z, _ = final_candidate_simplex(bank, names, predpool, testpool, p["lam"], p["prior"])
    elif r.kind == "cand_pband":
        names = p["names"]
        testpool = {n: finalize_recipe(n, recipes, predpool, bank, test, fixed, cache) for n in names}
        W, bands = fit_candidate_pband(bank, list(range(4)), names, predpool, p["lam"], p["prior"])
        z = apply_candidate_pband(test, names, testpool, W, bands)
    elif r.kind == "band_alpha":
        base = finalize_recipe(p["base"], recipes, predpool, bank, test, fixed, cache)
        par = fit_final_band_alpha(bank, predpool[p["base"]], p["experts"], p["two_d"], p["shrink_n"])
        z = apply_band_alpha(test, base, par, p["experts"])
    elif r.kind == "bias":
        base = finalize_recipe(p["base"], recipes, predpool, bank, test, fixed, cache)
        par = fit_final_bias(bank, predpool[p["base"]], p["experts"], p["two_d"], p["shrink_n"], p["strength"])
        z = apply_bias_cells(test, base, par, p["experts"])
    else:
        raise KeyError((name, r.kind))
    cache[name] = clipz(z)
    return cache[name]


# -----------------------------------------------------------------------------
# test bank, final raw training in fresh processes
# -----------------------------------------------------------------------------

def build_test_core_only(fixed, prev, ctx, friend):
    uid = np.asarray(friend["uid"], np.int64)
    r: dict[str, Any] = {"uid": uid, "friend": np.asarray(friend["z"], np.float64)}
    for n, src in (("cap", "S1-CAP"), ("unc", "S1-UNC"), ("dist", "S1-DIST")):
        u, z = fixed.component_test(ctx.package, src)
        r[n] = align(u, z, uid).astype(np.float32)
    r["table_core"] = sum(fixed.CORE_TABLE_WEIGHTS[n] * np.asarray(r[n], np.float64) for n in fixed.CORE_TABLE_WEIGHTS)
    hp = Path(ctx.checkpoints) / "test" / "hurdle_test.npz"
    if not npz_valid(hp, ("user_id", "z", "p", "mu")):
        raise RuntimeError("hurdle_test.npz is required from the previous continuation; refusing to retrain fixed/base helper here")
    d = load_npz(hp)
    r["hurdle"] = align(d["user_id"], d["z"], uid).astype(np.float32)
    r["p"] = np.clip(align(d["user_id"], d["p"], uid).astype(np.float32), EPS, 1-EPS)
    r["mu"] = np.maximum(align(d["user_id"], d["mu"], uid).astype(np.float32), 0)
    u, X, names = prev.build_test_meta_raw(ctx)
    r["meta_raw"] = align(u, X, uid).astype(np.float32)
    r["meta_names"] = names
    return r


def raw_test_path(ctx, name):
    return Path(ctx.checkpoints) / "test" / f"{name}_test.npz"


def load_final_raw_into_test(test, ctx, name):
    p = raw_test_path(ctx, name)
    need = ("user_id", "z", "p", "mu") if name.startswith("recent_hurdle") else ("user_id", "z")
    if not npz_valid(p, need):
        raise FileNotFoundError(p)
    d = load_npz(p)
    test[name] = align(d["user_id"], d["z"], test["uid"]).astype(np.float32)
    if name.startswith("recent_hurdle") and "p" in d:
        test[f"p_{name}"] = align(d["user_id"], d["p"], test["uid"]).astype(np.float32)
        test[f"mu_{name}"] = align(d["user_id"], d["mu"], test["uid"]).astype(np.float32)


def child_command(script: Path, work: Path, name: str, threads: int):
    return [sys.executable, str(script), "--child-final", name, "--reuse-work-dir", str(work), "--threads", str(threads)]


def spawn_final_child(script: Path, work: Path, name: str, threads: int, errors: Path) -> bool:
    log("FRESH-PROCESS FINAL TRAIN", name, "threads", threads)
    cmd = child_command(script, work, name, threads)
    try:
        rc = subprocess.run(cmd, check=False).returncode
        if rc != 0:
            log("CHILD FAILED", name, "returncode", rc)
            return False
        return True
    except Exception as e:
        append_error(errors, "child_final", name, "test", e)
        log("CHILD EXCEPTION", name, repr(e))
        return False


def configure_context_for_child(base: Path, args):
    fixed, fixed_path = import_fixed(base)
    prev, prev_path = fixed.import_prev(base)
    package = prev.discover_package(base)
    raw, sample = prev.discover_raw_and_sample(base, package)
    prev.ensure_dependencies(package, args.no_install)
    work = fixed.discover_work(base, args.reuse_work_dir)
    dummy = base / "_best_bas_combo_10h"
    results = dummy / "results"; subs = dummy / "submissions"
    results.mkdir(parents=True, exist_ok=True); subs.mkdir(parents=True, exist_ok=True)
    started = time.time()
    ctx = prev.Context(base_dir=base, package=package, pipeline=package/"pipeline", raw=raw, sample=sample,
                       work=work, results=results, submissions=subs, checkpoints=work/"checkpoints",
                       budget=prev.Budget(started, 99, 98, 1))
    prev.configure_pipeline(ctx, max(2, int(args.threads)))
    fixed.repair_cache(ctx); fixed.install_atomic_cache(ctx)
    return fixed, prev, ctx, package, raw, sample, fixed_path, prev_path


def train_variant_test_lowmem(ctx, fixed, name: str):
    """Fresh-process final fit of a recent hurdle variant.

    Uses the same all-user two-part training semantics as OOF.  Because the child
    has no OOF bank / research matrices resident, peak RAM is dramatically lower.
    Capped recent cutoffs additionally keep the dense matrix bounded.
    """
    if name not in MEM_VARIANTS:
        raise KeyError(name)
    p = variant_test_path(ctx, name)
    if npz_valid(p, ("user_id", "z", "p", "mu")):
        log("CHILD reuse final", p.name); return
    maxcuts, tau, rounds = MEM_VARIANTS[name]
    T, F = ctx.train_mod, ctx.features_mod
    C = ctx.config
    s, _ = fixed.expert_spec(ctx, "recent_hurdle")
    s.weight_tau = tau; s.rounds = rounds
    Xt, _ = F.make_xy(C.CUTOFF_TEST, s.L, s.panel_blocks, with_target=False, norm_long=s.norm_long)
    feats = T.select_features(F.feature_names(Xt), s.drop_groups, s.keep_only)
    cuts = list(s.grid())[-maxcuts:]
    log("CHILD FINAL ALL-USER", name, "recent cuts", len(cuts), "features", len(feats))
    X, y, w = T.assemble(cuts, s, feats, C.CUTOFF_TEST)
    T._XY.clear(); box = [X]; del X; gc.collect()
    model = T.fit_free(s, box, y, w)
    A = F.to_np(Xt, feats)
    clf, reg = model
    pp = np.clip(clf.predict(A), EPS, 1-EPS); mu = np.maximum(reg.predict(A), 0); z = pp * mu
    save_npz_atomic(p, user_id=Xt["user_id"].to_numpy().astype(np.int64), z=np.asarray(clipz(z), np.float32),
                    p=np.asarray(pp, np.float32), mu=np.asarray(mu, np.float32))
    log("CHILD FINAL DONE", name, "rows", f"{len(y):,}", "file", p)
    del model, A, Xt, y, w, box, pp, mu, z
    T._XY.clear(); gc.collect()


def child_main(args):
    base = Path(__file__).resolve().parent
    fixed, prev, ctx, *_ = configure_context_for_child(base, args)
    name = args.child_final
    if name in MEM_VARIANTS:
        train_variant_test_lowmem(ctx, fixed, name)
        return
    # Existing direct/dist finals, if unexpectedly absent, are trained in a clean
    # process.  Original recent_hurdle is intentionally NOT retried here.
    if name in ("multiscale_direct", "recent_direct", "recent_dist"):
        d = fixed.train_expert_test(ctx, name)
        log("CHILD FINAL DONE", name, "n", len(d["user_id"]))
        return
    raise KeyError(name)


# -----------------------------------------------------------------------------
# candidate metrics and selection
# -----------------------------------------------------------------------------

def regime_metrics(oof, bank, test_table, core_test, final_z, friend_z, fixed):
    ds = []
    for f in FOLDS:
        d = np.asarray(oof[f], np.float64) - np.asarray(bank[f]["table_core"], np.float64)
        d -= d.mean(); ds.append(d)
    oo = np.concatenate(ds)
    td = np.asarray(test_table, np.float64) - np.asarray(core_test, np.float64); td -= td.mean()
    vo = float(np.var(oo)); vt = float(np.var(td)); ratio = vt / max(vo, 1e-12)
    dd = candidate_distance(final_z, friend_z)
    return {"oof_table_var":vo, "test_table_var":vt, "var_ratio":ratio,
            "friend_corr":dd["corr"], "friend_std_dz":dd["std"],
            "friend_mean_abs_dz":dd["mean_abs"], "friend_pct02":dd["pct02"],
            "friend_pct05":dd["pct05"], "friend_pct10":dd["pct10"]}


def ranking_score(r):
    # Quality dominates.  Latest / worst positive deltas are strongly penalized.
    return (float(r["delta"]) + .7 * max(float(r["latest_delta"]), 0.0)
            + .35 * max(float(r["worst_delta"]), 0.0)
            + .0010 * abs(math.log(max(float(r.get("var_ratio", 1.0)), 1e-4))))


def select_submission_candidates(rows, testmap, friend_z, oldsubs, n=5):
    for r in rows:
        r["selection_score"] = ranking_score(r)
    # First pool: robust improvements.  Meta first fold is baseline, hence 3/3
    # recent wins is the strongest evidence we have.
    robust = [r for r in rows if r["delta"] < -0.00035 and r["wins_recent"] >= 2
              and r["latest_delta"] < 0.00005 and .30 <= r.get("var_ratio", 1) <= 2.6]
    robust.sort(key=lambda r: (r["selection_score"], r["delta"]))
    chosen = []
    fam = {}
    for r in robust:
        if len(chosen) >= n: break
        if r["name"] not in testmap: continue
        if fam.get(r["family"], 0) >= 2: continue
        z = testmap[r["name"]]
        # Never reproduce the two already-tested bad LB directions.
        if any(candidate_distance(z, v)["corr"] > .9995 for v in oldsubs.values()):
            continue
        # First two can be close if both are exceptionally good; later candidates
        # must bring a genuinely different error direction.
        if len(chosen) >= 2:
            if not all(candidate_distance(z, testmap[q["name"]])["std"] >= .0045 or
                       candidate_distance(z, testmap[q["name"]])["pct02"] >= .08 for q in chosen):
                continue
        chosen.append(r); fam[r["family"]] = fam.get(r["family"], 0) + 1
    # Safe fallback: still locally improving; no deliberately bad diversity file.
    if len(chosen) < min(3, n):
        more = [r for r in rows if r not in chosen and r["delta"] < 0 and r["latest_delta"] <= .0002 and r["wins_recent"] >= 2 and r["name"] in testmap]
        more.sort(key=lambda r: (r["selection_score"], r["delta"]))
        for r in more:
            if len(chosen) >= n: break
            chosen.append(r)
    return chosen[:n]


def locate_previous_submissions(base: Path, friend_uid):
    import pandas as pd
    out = {}
    pats = ["*continue12h*safe*csv", "*continue12h*class1_occ*csv"]
    for pat in pats:
        for p in base.rglob(pat):
            try:
                d = pd.read_csv(p)
                if len(d) != len(friend_uid) or not {"user_id", "predict"}.issubset(d.columns): continue
                z = np.log1p(np.maximum(d["predict"].to_numpy(np.float64), 0))
                out[p.stem] = align(d["user_id"].to_numpy(np.int64), z, friend_uid)
                break
            except Exception:
                continue
    return out


def oracle_diagnostics(bank, predpool, names):
    rows = []
    for f in FOLDS:
        y = np.asarray(bank[f]["true_z"], np.float64)
        P = np.stack([predpool[n][f] for n in names], axis=1)
        best = np.min((P - y[:, None]) ** 2, axis=1)
        rows.append({"fold":f, "n_models":len(names), "oracle_rmsle":float(np.sqrt(best.mean()))})
    return rows


# -----------------------------------------------------------------------------
# self-test
# -----------------------------------------------------------------------------

def self_test():
    rng = np.random.default_rng(123)
    # lightweight synthetic bank with all expert names used by recipes
    bank = {}
    for j, f in enumerate(FOLDS):
        n = 1800
        meta = rng.normal(size=(n, 18)).astype(np.float32)
        p = 1 / (1 + np.exp(-(.6*meta[:,0]-.25*meta[:,1])))
        mu = np.maximum(1.5 + .4*meta[:,2], .1)
        tz = np.maximum(p*mu + .2*meta[:,3] + rng.normal(scale=.65,size=n), 0)
        y = np.expm1(tz)
        rec = {"uid":np.arange(n)+j*10000,"y":y,"true_z":tz,"meta_raw":meta,
               "meta_names":[f"m{k}" for k in range(meta.shape[1])],"p":p,"mu":mu}
        for k, sig in (("cap",.42),("unc",.39),("dist",.37),("hurdle",.38),
                       ("multiscale_direct",.41),("recent_direct",.40),("recent_dist",.385),
                       ("recent_hurdle_fast12",.365)):
            rec[k] = np.maximum(tz+rng.normal(scale=sig,size=n),0).astype(np.float32)
        rec["table_core"] = .10/.55*rec["cap"] + .20/.55*rec["unc"] + .25/.55*rec["dist"]
        bank[f] = rec
    class F:
        CORE_TABLE_WEIGHTS={"cap":.10/.55,"unc":.20/.55,"dist":.25/.55}
        score_table=staticmethod(lambda name,preds,bank,family,rows,notes="": _score_test(name,preds,bank,family,rows,notes))
        walk_simplex=staticmethod(lambda bank,ex,lam=.02,min_new=0,new_names=(): _simplex_test(bank,ex))
        walk_greedy=staticmethod(lambda bank,ex,steps=35: _simplex_test(bank,ex))
        walk_pband=staticmethod(lambda bank,ex: _simplex_test(bank,ex))
        walk_occ=staticmethod(lambda bank,guard=True: {f:bank[f]["table_core"] for f in FOLDS})
    # Add minimal final APIs and exercise recursive finalization too.
    F.final_simplex=staticmethod(lambda bank,test,ex,lam=.02,min_new=0,new_names=(): (np.mean(np.stack([test[n] for n in ex],1),1), np.ones(len(ex))/len(ex)))
    F.final_greedy=staticmethod(lambda bank,test,ex,steps=35: (np.mean(np.stack([test[n] for n in ex],1),1), np.ones(len(ex))/len(ex)))
    F.final_pband=staticmethod(lambda bank,test,ex: (np.mean(np.stack([test[n] for n in ex],1),1), np.ones((5,len(ex)))/len(ex)))
    F.final_occ=staticmethod(lambda bank,test,guard=True: np.asarray(test["table_core"],np.float64))
    rows,pool,recipes,ex = build_primitive_research(F,bank,Path("/tmp/combo_selftest"),"x")
    rows,pool,recipes = build_combo_research(F,bank,rows,pool,recipes,ex,Path("/tmp/combo_selftest"),"x")
    assert len(rows) >= 10 and all(np.isfinite(r["delta"]) for r in rows)
    test={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in bank[FOLDS[-1]].items()}
    cache={}
    tested=0
    for r in rows:
        try:
            z=finalize_recipe(r["name"],recipes,pool,bank,test,F,cache)
            assert len(z)==len(test["uid"]) and np.isfinite(z).all()
            tested+=1
            if tested>=12:break
        except Exception:
            continue
    assert tested>=8, tested
    print("SELF-TEST OK",len(rows),"candidates finalizers",tested,flush=True)


def _score_test(name,preds,bank,family,rows,notes=""):
    sc=[];bs=[];ds=[]
    for f in FOLDS:
        a=rms_z(bank[f]["y"],preds[f]);b=rms_z(bank[f]["y"],bank[f]["table_core"]);sc.append(a);bs.append(b);ds.append(a-b)
    r={"name":name,"family":family,"wcv":wavg(sc),"base_wcv":wavg(bs),"delta":wavg(ds),"wins":sum(x<0 for x in ds),
       "wins_recent":sum(x<0 for x in ds[1:]),"latest_delta":ds[-1],"worst_delta":max(ds),"fold_scores":sc,"fold_deltas":ds,"notes":notes}
    rows.append(r);return r


def _simplex_test(bank,ex):
    out={f:np.mean(np.stack([bank[f][n] for n in ex],1),1) for f in FOLDS}
    return out,None


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-hours", type=float, default=9.5)
    ap.add_argument("--threads", type=int, default=max(4, min(12, os.cpu_count() or 8)))
    ap.add_argument("--child-threads", type=int, default=5)
    ap.add_argument("--reuse-work-dir", type=str, default=None)
    ap.add_argument("--no-install", action="store_true")
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--child-final", type=str, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.self_test:
        self_test(); return
    if args.child_final:
        child_main(args); return

    started = time.time(); budget = Budget(started, args.max_hours)
    base = Path(__file__).resolve().parent
    fixed, fixed_path = import_fixed(base)
    prev, prev_path = fixed.import_prev(base)
    package = prev.discover_package(base)
    raw, sample = prev.discover_raw_and_sample(base, package)
    prev.ensure_dependencies(package, args.no_install)
    work = fixed.discover_work(base, args.reuse_work_dir)
    out = base / "_best_bas_combo_10h"
    results = out / "results"; subs = out / "submissions"
    results.mkdir(parents=True, exist_ok=True); subs.mkdir(parents=True, exist_ok=True)
    ctx = prev.Context(base_dir=base, package=package, pipeline=package/"pipeline", raw=raw, sample=sample,
                       work=work, results=results, submissions=subs, checkpoints=work/"checkpoints",
                       budget=prev.Budget(started, args.max_hours, max(args.max_hours-1.5,0), 1.0))
    prev.configure_pipeline(ctx, args.threads)
    fixed.repair_cache(ctx); fixed.install_atomic_cache(ctx)
    friend = prev.verify_friend_package(package)
    errors = results / "errors.jsonl"
    log("COMBO CONTINUATION", SCRIPT_VERSION)
    log("WORK", work)
    log("FRIEND rebuild error", friend.get("max_log_error"))
    log("Fixed teammate SEQ/ETX/base will NOT be retrained.")

    bank = fixed.load_core(prev, ctx)
    loaded = load_existing_new_oof(fixed, ctx, bank)
    if args.preflight_only:
        log("PREFLIGHT OK; existing new OOF", loaded, "remaining", f"{budget.remaining:.2f}h")
        return

    manifest = {"version":SCRIPT_VERSION,"started":now(),"fixed_parent":str(fixed_path),"previous_runner":str(prev_path),
                "work":str(work),"package":str(package),"loaded_existing_oof":loaded,"args":vars(args),
                "friend_rebuild_error":friend.get("max_log_error")}
    atomic_json(results/"RUN_START.json",manifest)

    # A. First extended search from everything already trained.  This is cheap and
    # gives us a baseline before spending time on new recent hurdle variants.
    log("A: EXTENDED COMBINATION RESEARCH on existing OOF")
    rows, predpool, recipes, ex = build_primitive_research(fixed, bank, results, "existing")
    rows, predpool, recipes = build_combo_research(fixed, bank, rows, predpool, recipes, ex, results, "existing")
    for r in rows[:15]:
        log(f"  {r['name'][:52]:52s} d={r['delta']:+.6f} recent={r['wins_recent']}/3 latest={r['latest_delta']:+.6f}")

    runtime=[]
    # B. New all-user recent hurdle variants.  They are ordered by expected value.
    for name, est in (("recent_hurdle_fast12", .85), ("recent_hurdle_stable18", 1.15)):
        if not budget.can_start(est, extra=2.0):
            log("SKIP NEW VARIANT by budget", name, "remaining", f"{budget.remaining:.2f}h")
            continue
        t=time.time();ok=True
        try:
            add_variant_oof(ctx, fixed, bank, name)
        except Exception as e:
            ok=False;append_error(errors,"variant_oof",name,"all",e);log("VARIANT FAILED",name,repr(e))
        runtime.append({"stage":"variant_oof","name":name,"hours":(time.time()-t)/3600,"ok":ok,"remaining":budget.remaining})
        save_csv(results/"runtime.csv",runtime)
        if ok:
            log("RESEARCH REFRESH after",name)
            rows,predpool,recipes,ex=build_primitive_research(fixed,bank,results,f"after_{name}")
            rows,predpool,recipes=build_combo_research(fixed,bank,rows,predpool,recipes,ex,results,f"after_{name}")
            for r in rows[:12]:log(" ",r["name"],f"d={r['delta']:+.6f}","recent",r["wins_recent"],"latest",f"{r['latest_delta']:+.6f}")

    # Final cheap research one more time from complete OOF bank.
    rows,predpool,recipes,ex=build_primitive_research(fixed,bank,results,"final")
    rows,predpool,recipes=build_combo_research(fixed,bank,rows,predpool,recipes,ex,results,"final")
    rows.sort(key=lambda r:(r["delta"],r["latest_delta"]))
    save_csv(results/"ALL_VALIDATION.csv",rows)
    oracle_names=[r["name"] for r in rows[:min(12,len(rows))] if r["name"] in predpool]
    save_csv(results/"ORACLE_DIAGNOSTICS.csv",oracle_diagnostics(bank,predpool,oracle_names))
    log("TOP FINAL VALIDATION")
    for r in rows[:25]:
        log(f" {r['name'][:50]:50s} d={r['delta']:+.6f} recent={r['wins_recent']}/3 latest={r['latest_delta']:+.6f} family={r['family']}")

    # Shortlist before final raw training. Only locally improving / recent-stable recipes.
    shortlist=[r for r in rows if r["delta"]<-.00025 and r["wins_recent"]>=2 and r["latest_delta"]<.00025]
    shortlist=sorted(shortlist,key=lambda r:(r["delta"],r["latest_delta"]))[:22]
    shortlist_names=[r["name"] for r in shortlist]
    save_csv(results/"SHORTLIST_VALIDATION.csv",shortlist)

    # Raw dependencies recursively gathered from recipes.
    def deps(name, seen=None):
        seen=set() if seen is None else seen
        if name in seen:return set()
        seen.add(name);r=recipes[name];p=r.params
        if r.kind=="raw":return {p["expert"]}
        if r.kind=="ridge":return set(p["experts"])
        if r.kind in ("raw_simplex","raw_greedy","raw_pband"):return set(p["experts"])
        if r.kind=="occ":return set()
        if r.kind=="blend":return deps(p["a"],seen)|deps(p["b"],seen)
        if r.kind in ("cand_simplex","cand_pband"):
            out=set()
            for n in p["names"]:out|=deps(n,seen)
            return out
        if r.kind in ("band_alpha","bias"):return deps(p["base"],seen)|set(p["experts"])
        return set()
    raw_needed=set()
    for n in shortlist_names:raw_needed|=deps(n)
    raw_needed={x for x in raw_needed if x not in ("cap","unc","dist","hurdle")}
    log("RAW TEST DEPENDENCIES",sorted(raw_needed),"remaining",f"{budget.remaining:.2f}h")

    # Free all OOF/research matrices before child final fits.  Children start with a
    # clean address space; this is the critical bad-allocation fix.
    del bank,predpool,rows,recipes,ex
    gc.collect()
    try:
        ctx.train_mod._XY.clear()
    except Exception:pass
    gc.collect()

    # Train only missing final raw experts, one clean child at a time. Original
    # recent_hurdle is NOT among finalizable dependencies by construction.
    for name in sorted(raw_needed):
        p=raw_test_path(ctx,name)
        need=("user_id","z","p","mu") if name.startswith("recent_hurdle") else ("user_id","z")
        if npz_valid(p,need):
            log("REUSE FINAL RAW",name);continue
        est=.7 if name in MEM_VARIANTS else 2.4
        if not budget.can_start(est,extra=.65):
            log("SKIP FINAL RAW by budget",name);continue
        ok=spawn_final_child(Path(__file__).resolve(),work,name,max(2,min(args.child_threads,args.threads)),errors)
        if not ok:log("Final raw unavailable; dependent candidates will be skipped",name)

    # Reload compact OOF bank and recompute cheap recipes after children exit.
    log("RELOAD OOF BANK after clean final training")
    bank=fixed.load_core(prev,ctx);load_existing_new_oof(fixed,ctx,bank)
    for name in MEM_VARIANTS:
        if all(npz_valid(variant_fold_path(ctx,name,f),("user_id","y","z","p","mu")) for f in FOLDS):
            for f in FOLDS:
                d=load_npz(variant_fold_path(ctx,name,f));bank[f][name]=align(d["user_id"],d["z"],bank[f]["uid"]).astype(np.float32)
    rows,predpool,recipes,ex=build_primitive_research(fixed,bank,results,"final_reload")
    rows,predpool,recipes=build_combo_research(fixed,bank,rows,predpool,recipes,ex,results,"final_reload")
    rows.sort(key=lambda r:(r["delta"],r["latest_delta"]))

    # Build compact test bank only now.  No multi-million-row training matrix is
    # alive at this point.
    test=build_test_core_only(fixed,prev,ctx,friend)
    for name in sorted(set(finalizable_experts(bank))-{"cap","unc","dist","hurdle"}):
        try:load_final_raw_into_test(test,ctx,name)
        except FileNotFoundError:log("FINAL RAW missing -> recipes using it will be skipped",name)

    oldsubs=locate_previous_submissions(base,np.asarray(friend["uid"],np.int64))
    cache={};testmap={};finalrows=[]
    for r in rows:
        if not (r["delta"]<.00035 and r["wins_recent"]>=2 and r["latest_delta"]<.0005):continue
        name=r["name"]
        missing=[x for x in deps(name) if x not in test and x not in ("cap","unc","dist","hurdle")]
        if missing:continue
        try:
            table_z=finalize_recipe(name,recipes,predpool,bank,test,fixed,cache)
            final_z=fixed.transform_to_friend(np.asarray(friend["z"],np.float64),test["table_core"],table_z,1.0)
            reg=regime_metrics(predpool[name],bank,table_z,test["table_core"],final_z,friend["z"],fixed)
            rr=dict(r);rr.update(reg);finalrows.append(rr);testmap[name]=final_z
        except Exception as e:
            append_error(errors,"finalize",name,"test",e);log("FINALIZE FAILED",name,repr(e))

    # Material slot-strength variants of the best 3 mechanisms.  These alter
    # 41.25%, 48.1%, or 55% of the final model through the table slot and are
    # explicitly revalidated; no cosmetic 95/5 blends.
    seeds=sorted([r for r in finalrows if r["delta"]<0 and r["wins_recent"]>=2],key=lambda r:r["delta"])[:4]
    for r in seeds:
        name=r["name"]
        base_table=finalize_recipe(name,recipes,predpool,bank,test,fixed,cache)
        for beta in (.75,.875):
            nn=f"{name}__slotbeta{int(beta*1000):03d}"
            op={f:clipz(bank[f]["table_core"]+beta*(predpool[name][f]-bank[f]["table_core"])) for f in FOLDS}
            tmp=[];rr=fixed.score_table(nn,op,bank,r["family"]+"_slotstrength",tmp,notes=f"beta={beta}")
            table_beta=clipz(test["table_core"]+beta*(base_table-test["table_core"]))
            z=fixed.transform_to_friend(friend["z"],test["table_core"],table_beta,1.0)
            reg=regime_metrics(op,bank,table_beta,test["table_core"],z,friend["z"],fixed);rr.update(reg)
            finalrows.append(rr);testmap[nn]=z

    finalrows.sort(key=lambda r:(ranking_score(r),r["delta"]))
    save_csv(results/"FINAL_CANDIDATE_METRICS.csv",finalrows)
    chosen=select_submission_candidates(finalrows,testmap,np.asarray(friend["z"],np.float64),oldsubs,n=5)

    import pandas as pd
    sample_df=pd.read_csv(sample)
    sample_uid=sample_df["user_id"].to_numpy(np.int64) if "user_id" in sample_df else np.asarray(friend["uid"],np.int64)
    selection=[]
    for i,r in enumerate(chosen,1):
        z=align(friend["uid"],testmap[r["name"]],sample_uid)
        pred=np.maximum(np.expm1(np.clip(z,0,20)),0)
        df=pd.DataFrame({"user_id":sample_uid,"predict":pred})
        if len(df)!=250000 or df.user_id.duplicated().any() or df.predict.isna().any() or (df.predict<0).any():
            raise RuntimeError("bad submission")
        path=subs/f"submission_combo10h_candidate_{i}_{r['name']}.csv";df.to_csv(path,index=False)
        dd=candidate_distance(testmap[r["name"]],friend["z"])
        sr={"rank":i,"name":r["name"],"file":str(path),"delta_table":r["delta"],"latest_delta":r["latest_delta"],
            "wins_recent":r["wins_recent"],"family":r["family"],"friend_corr":dd["corr"],"friend_std_dz":dd["std"],
            "friend_pct05":dd["pct05"],"var_ratio":r.get("var_ratio")}
        selection.append(sr);log("SUBMISSION",i,r["name"],"d",f"{r['delta']:+.6f}","recent",r["wins_recent"],"corr",f"{dd['corr']:.6f}")
    save_csv(results/"FINAL_SELECTION.csv",selection)

    # Pairwise diversity including old bad-LB directions and friend anchor.
    div=[];allz={"STRONGEST_CURRENT":np.asarray(friend["z"],np.float64),**{r["name"]:testmap[r["name"]] for r in chosen},**{f"OLD_{k}":v for k,v in oldsubs.items()}}
    ks=list(allz)
    for i in range(len(ks)):
        for j in range(i+1,len(ks)):
            div.append({"a":ks[i],"b":ks[j],**candidate_distance(allz[ks[i]],allz[ks[j]])})
    save_csv(results/"DIVERSITY.csv",div)

    runtime_h=(time.time()-started)/3600
    manifest.update({"finished":now(),"runtime_hours":runtime_h,"remaining_hours":budget.remaining,"selection":selection,
                     "finalizable_experts":finalizable_experts(bank)})
    atomic_json(results/"RUN_MANIFEST.json",manifest)
    report=["E-CUP fixedstack combo continuation",f"runtime_hours={runtime_h:.3f}",
            "STRONGEST_CURRENT was NEVER retrained.","",
            "Top validation candidates:"]
    for r in finalrows[:30]:
        report.append(f"{r['name']:55s} delta={r['delta']:+.6f} recent={r['wins_recent']}/3 latest={r['latest_delta']:+.6f} family={r['family']} var_ratio={r.get('var_ratio',float('nan')):.3f}")
    report += ["","Emitted submissions:"]+[f"{x['rank']}. {x['name']} delta={x['delta_table']:+.6f} corr_friend={x['friend_corr']:.6f} file={x['file']}" for x in selection]
    (results/"REPORT_RU.txt").write_text("\n".join(report),encoding="utf-8")
    bundle=base/f"fixedstack_combo10h_REVIEW_BUNDLE_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    with zipfile.ZipFile(bundle,"w",zipfile.ZIP_DEFLATED) as zf:
        for p in results.iterdir():
            if p.is_file() and p.suffix.lower() in {".csv",".json",".txt",".jsonl"}:zf.write(p,arcname=f"results/{p.name}")
        for p in subs.glob("*.csv"):zf.write(p,arcname=f"submissions/{p.name}")
    log("DONE",f"{runtime_h:.2f}h","submissions",len(selection))
    log("REPORT",results/"REPORT_RU.txt")
    log("BUNDLE",bundle)


if __name__ == "__main__":
    main()

```
