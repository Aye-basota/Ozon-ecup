#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E-CUP 2026 / Track 3 — emergency 12h continuation for teammate STRONGEST_CURRENT.

PURPOSE
-------
This file is intended to be placed next to the PREVIOUS runner in:

    src/DL/best_bas/

and next to the unpacked teammate package:

    src/DL/best_bas/submission_STRONGEST_CURRENT/

It does NOT restart the old 23h research from scratch.  It reuses the previous
runner's fold/test checkpoints in-place, completes only truly missing mandatory
TABLE checkpoints if required, skips new expensive neural OOF training, and
spends the remaining budget on the post-Phase-14 directions:

  1) class1-specialized experts (p>=0.5), especially conservative experts for
     false-one / overprediction risk;
  2) temporally honest error-detector probabilities used only as ROUTER FEATURES,
     never as direct additive correction magnitudes;
  3) a single LambdaRank expert ranker (oracle distillation / expert ranking)
     instead of another bank of independent pairwise classifiers;
  4) routing only inside classifier=1, leaving class0 untouched by default.

Temporal hygiene:
  * error scores for validation fold i are learned only from folds < i;
  * specialists for fold i are learned only from folds < i;
  * ranker for fold i is learned only from already OOT predictions in folds < i;
  * final test models may use all four mature validation folds.

The script writes a COMMON report that inventories the previous run (including
its manifest/progress/errors/runtime if present) and appends the continuation
results, detector AUCs, class1 segment scores, expert-ranker diagnostics and the
three final submissions.

Recommended emergency launch with ~12h left:

    python continue_best_bas_12h_v2.py --max-hours 11.25

By default it reserves ~45 minutes and refuses to start optional work too close
to the deadline.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import importlib.util
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

import numpy as np

VERSION = "best_bas_continue_12h_phase15_v3_cache_repair_2026-08-21"
FOLDS = ("2025-09-04", "2025-09-18", "2025-10-02", "2025-10-16")
FOLD_WEIGHTS = np.asarray([1.0, 2.0, 4.0, 8.0], dtype=np.float64)
EPS = 1e-7


def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def log(*x: Any) -> None:
    print(f"[{now_iso()}]", *x, flush=True)


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    tmp.replace(path)


def _json_default(x: Any):
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return float(x)
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, Path): return str(x)
    raise TypeError(type(x).__name__)


def safe_read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import pandas as pd
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    flat = []
    for r in rows:
        q = {}
        for k, v in r.items():
            if isinstance(v, (dict, list, tuple, np.ndarray)):
                q[k] = json.dumps(v, ensure_ascii=False, default=_json_default)
            else:
                q[k] = v
        flat.append(q)
    pd.DataFrame(flat).to_csv(path, index=False)


def import_previous_runner(base: Path, explicit: str | None):
    candidates: list[Path] = []
    if explicit:
        candidates.append((base / explicit).resolve() if not Path(explicit).is_absolute() else Path(explicit).resolve())
    candidates += [
        base / "run_best_bas_research_23h.py",
        base / "run_best_bas_research.py",
    ]
    candidates += sorted(base.glob("run_best_bas*.py"), key=lambda p: p.stat().st_mtime, reverse=True)
    this = Path(__file__).resolve()
    seen = set()
    for p in candidates:
        p = p.resolve()
        if p == this or p in seen or not p.exists():
            continue
        seen.add(p)
        try:
            spec = importlib.util.spec_from_file_location("best_bas_previous_runner", p)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            required = ["Context", "Budget", "discover_package", "discover_raw_and_sample", "configure_pipeline",
                        "verify_friend_package", "fold_ckpt", "load_fold", "build_fold_bank", "meta_matrix",
                        "train_table_fold", "build_test_record", "calibrate", "level_calibrate_test",
                        "validate_submission_frame", "sha256"]
            if all(hasattr(mod, n) for n in required):
                log("Previous runner:", p)
                return mod, p
        except Exception as e:
            log("Не удалось импортировать", p.name, "->", repr(e))
    raise FileNotFoundError(
        "Не найден предыдущий run_best_bas*.py. Положите continue_best_bas_12h_v2.py рядом с прошлым файлом."
    )


def score_workdir(p: Path) -> tuple[int, int, float]:
    ck = p / "checkpoints"
    if not ck.is_dir():
        return (-1, -1, 0.0)
    fold_files = list((ck / "folds").glob("*.npz")) if (ck / "folds").exists() else []
    test_files = list((ck / "test").glob("*.npz")) if (ck / "test").exists() else []
    mt = max([f.stat().st_mtime for f in fold_files + test_files] + [p.stat().st_mtime])
    return (len(fold_files), len(test_files), mt)


def discover_previous_work(base: Path, explicit: str | None, new_work: Path) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_absolute(): p = base / p
        p = p.resolve()
        if not (p / "checkpoints").is_dir():
            raise FileNotFoundError(f"В {p} нет checkpoints/")
        return p
    names = ["_best_bas_phase15_friend", "_best_bas_research", "_best_bas_phase15", "_best_bas_research_23h"]
    cands = [base / n for n in names if (base / n).is_dir()]
    cands += [p for p in base.glob("_best_bas*") if p.is_dir() and p.resolve() != new_work.resolve()]
    uniq = []
    seen = set()
    for p in cands:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp); uniq.append(rp)
    scored = [(score_workdir(p), p) for p in uniq]
    scored = [x for x in scored if x[0][0] >= 0]
    if not scored:
        raise FileNotFoundError(
            "Не найдена рабочая папка предыдущего запуска (_best_bas_.../checkpoints). "
            "Если вы её переименовали, передайте --previous-work-dir PATH."
        )
    scored.sort(key=lambda x: x[0], reverse=True)
    log("Previous work candidates:")
    for sc, p in scored[:6]: log(" ", p.name, "fold_npz=", sc[0], "test_npz=", sc[1])
    return scored[0][1]


class WallBudget:
    def __init__(self, max_hours: float, reserve_hours: float = 0.75):
        self.started = time.time()
        self.max_hours = float(max_hours)
        self.reserve_hours = float(reserve_hours)
    @property
    def elapsed(self) -> float: return (time.time() - self.started) / 3600.0
    @property
    def remaining(self) -> float: return self.max_hours - self.elapsed
    def can_start(self, estimate: float, extra_reserve: float = 0.0) -> bool:
        return self.remaining > float(estimate) + self.reserve_hours + float(extra_reserve)


def inventory_previous(prev_work: Path) -> list[dict[str, Any]]:
    rows = []
    for sub in ("checkpoints/folds", "checkpoints/test", "results", "submissions"):
        root = prev_work / sub
        if not root.exists():
            continue
        for p in root.glob("*"):
            if not p.is_file(): continue
            rows.append(dict(section=sub, file=p.name, bytes=p.stat().st_size,
                             mtime=dt.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")))
    return rows


def parquet_has_valid_magic(path: Path) -> bool:
    """Cheap detection of truncated parquet files left by a full disk/crash.

    A parquet file must start and end with the four bytes PAR1.  The exact
    failure seen after the interrupted run ("file must end with PAR1") is
    therefore detectable without loading the multi-GB feature table.
    """
    try:
        if path.stat().st_size < 8:
            return False
        with path.open("rb") as fh:
            head = fh.read(4)
            fh.seek(-4, os.SEEK_END)
            tail = fh.read(4)
        return head == b"PAR1" and tail == b"PAR1"
    except Exception:
        return False


def repair_processed_cache(ctx, *, verbose: bool = True) -> list[str]:
    """Delete ONLY malformed derived parquet caches, never raw train.parquet.

    The previous runner redirects src.config.DATA_PROCESSED to
    <old_work>/cache/processed.  Everything there is reproducible derived
    state (feat_*, panel_*, etc.), so a truncated file is safe to remove and
    rebuild.  Valid caches are left untouched.
    """
    root = Path(ctx.work) / "cache" / "processed"
    removed: list[str] = []
    if not root.exists():
        return removed
    for p in sorted(root.rglob("*.parquet")):
        if parquet_has_valid_magic(p):
            continue
        try:
            size = p.stat().st_size
        except Exception:
            size = -1
        try:
            p.unlink()
            removed.append(str(p))
            if verbose:
                log("CACHE REPAIR: removed truncated parquet:", p.name, f"({size} bytes)")
        except Exception as e:
            raise RuntimeError(f"Не удалось удалить повреждённый cache parquet {p}: {e}") from e
    return removed


def install_atomic_feature_cache(ctx) -> None:
    """Make future feature-cache writes atomic and self-healing for this process.

    src.features.features_cached originally writes directly to the final
    parquet filename.  A full disk can leave a file that exists but is
    truncated, and every restart then trusts it.  This runtime patch writes to
    *.tmp.parquet first and os.replace()s only after a successful write.
    """
    F = ctx.features_mod
    if getattr(F, "_best_bas_atomic_cache_patch", False):
        return
    original = F.features_cached
    default_L = getattr(F, "HISTORY_L", 180)

    def safe_features_cached(T, L=default_L, norm_long=False):
        p = F.DATA_PROCESSED / f"feat_{F._tag(T)}_L{'norm' if norm_long else ''}{L}.parquet"
        if p.exists():
            try:
                return F.pl.read_parquet(p)
            except Exception as e:
                msg = str(e).lower()
                if ("par1" not in msg and "out of specification" not in msg and
                        "parquet" not in msg and "computeerror" not in type(e).__name__.lower()):
                    raise
                log("CACHE REPAIR: unreadable feature cache -> rebuild:", p.name, "|", repr(e))
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass
        f = F.build_features(T, L, norm_long)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + f".tmp.{os.getpid()}.parquet")
        try:
            if tmp.exists():
                tmp.unlink()
            f.write_parquet(tmp)
            # Validate footer before publishing the cache filename.
            if not parquet_has_valid_magic(tmp):
                raise RuntimeError(f"Temporary parquet is truncated after write: {tmp}")
            os.replace(tmp, p)
        except Exception:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            raise
        return f

    F.features_cached = safe_features_cached
    F._best_bas_atomic_cache_patch = True
    F._best_bas_original_features_cached = original
    log("Installed atomic/self-healing feature parquet cache")


def npz_structurally_readable(path: Path) -> bool:
    """Check the ZIP central directory and required keys without loading meta_raw."""
    import zipfile
    try:
        with zipfile.ZipFile(path, "r") as zf:
            # Opening the central directory catches the common disk-full/truncate case.
            names = set(zf.namelist())
            if not names:
                return False
        with np.load(path, allow_pickle=True) as d:
            keys = set(d.files)
            if not {"user_id", "z"}.issubset(keys):
                return False
            # Materialise small/essential arrays so CRC/truncation is detected.
            _ = np.asarray(d["user_id"])
            _ = np.asarray(d["z"])
        return True
    except Exception:
        return False


def repair_corrupt_checkpoints(ctx) -> list[str]:
    """Remove only unreadable generated NPZ checkpoints so they can be rebuilt."""
    removed: list[str] = []
    root = Path(ctx.checkpoints)
    if not root.exists():
        return removed
    for sub in (root / "folds", root / "test"):
        if not sub.exists():
            continue
        for p in sorted(sub.glob("*.npz")):
            if npz_structurally_readable(p):
                continue
            try:
                size = p.stat().st_size
            except Exception:
                size = -1
            try:
                p.unlink()
                removed.append(str(p))
                log("CHECKPOINT REPAIR: removed unreadable npz:", p.name, f"({size} bytes)")
            except Exception as e:
                raise RuntimeError(f"Не удалось удалить повреждённый checkpoint {p}: {e}") from e
    return removed


def is_cache_corruption_error(exc: BaseException) -> bool:
    s = (type(exc).__name__ + " " + str(exc)).lower()
    return ("par1" in s or "file out of specification" in s or
            ("parquet" in s and ("computeerror" in s or "invalid" in s or "corrupt" in s)))


def previous_checkpoint_counts(prev, ctx) -> dict[str, Any]:
    detail: dict[str, list[str]] = {}
    for model in ("cap", "unc", "dist", "hurdle", "hurdle240", "hurdle320", "seq42", "etx42"):
        detail[model] = [f for f in FOLDS if prev.fold_ckpt(ctx, model, f).exists()]
    return {"detail": detail,
            "core_ready": all(len(detail[x]) == 4 for x in ("cap", "unc", "dist", "hurdle")),
            "seq42_complete": len(detail["seq42"]) == 4,
            "etx42_complete": len(detail["etx42"]) == 4}


def historical_runtime_estimate(prev_work: Path, variant: str, default: float = 0.6) -> float:
    p = prev_work / "results" / "runtime.csv"
    try:
        import pandas as pd
        d = pd.read_csv(p)
        x = d[(d.get("stage") == "table") & (d.get("model") == variant)]["hours"].dropna().astype(float)
        if len(x): return float(max(0.15, x.median() * 1.30))
    except Exception:
        pass
    return default


def ensure_core(prev, ctx, wall: WallBudget, row_frac: float, runtime_rows: list[dict[str, Any]]) -> None:
    """Complete only missing core fold checkpoints; never trains neural OOF."""
    for variant in ("cap", "unc", "dist", "hurdle"):
        for fold in FOLDS:
            path = prev.fold_ckpt(ctx, variant, fold)
            # Continuation uses only the full hurdle checkpoint. Old 240/320 snapshots
            # are not required and must never force a costly retrain.
            if path.exists():
                continue
            est = historical_runtime_estimate(ctx.work, variant)
            if not wall.can_start(est, extra_reserve=2.0):
                raise RuntimeError(
                    f"Не хватает безопасного времени на недостающий обязательный {variant}/{fold}. "
                    f"remaining={wall.remaining:.2f}h, estimate≈{est:.2f}h."
                )
            log("RESUME missing core:", variant, fold, f"estimate≈{est:.2f}h")
            t = time.time()
            try:
                prev.train_table_fold(ctx, variant, fold, row_frac=row_frac)
            except Exception as e:
                if not is_cache_corruption_error(e):
                    raise
                log("Detected parquet cache corruption during", f"{variant}/{fold}", "-> repair + one retry")
                removed = repair_processed_cache(ctx)
                if not removed:
                    log("No bad PAR1 footer found; retrying once because Polars reported parquet corruption")
                # train.py caches assembled cutoff tables in-memory; after a failed
                # assembly they may contain partial state.  Clear before retry.
                try:
                    ctx.train_mod._XY.clear()
                except Exception:
                    pass
                gc.collect()
                prev.train_table_fold(ctx, variant, fold, row_frac=row_frac)
            runtime_rows.append(dict(stage="resume_core", model=variant, fold=fold,
                                     hours=(time.time()-t)/3600.0))



def build_core_bank(prev, ctx, neural_complete: dict[str,bool]) -> dict[str,dict[str,Any]]:
    """Load only checkpoints needed by the continuation; no hurdle240/320 dependency."""
    bank={}
    proxy_weights=getattr(prev,"PROXY_COMPONENT_WEIGHTS",{"cap":.10,"unc":.20,"dist":.25,"seq42":.225,"etx42":.225})
    for fold in FOLDS:
        cap=prev.load_fold(prev.fold_ckpt(ctx,"cap",fold));prev.validate_fold_record(cap,f"cap/{fold}")
        uid=np.asarray(cap["user_id"],dtype=np.int64);y=np.asarray(cap["y"],dtype=np.float64)
        rec={"user_id":uid,"y":y,"cap":np.asarray(cap["z"],dtype=np.float64),
             "meta_raw":np.asarray(cap["meta_raw"],dtype=np.float32),
             "meta_names":[str(x) for x in cap["meta_names"].tolist()]}
        for n in ("unc","dist","hurdle"):
            d=prev.align_to_uid(prev.load_fold(prev.fold_ckpt(ctx,n,fold)),uid)
            prev.validate_fold_record(d,f"{n}/{fold}")
            if not np.allclose(d["y"],y,atol=1e-5,rtol=1e-6): raise AssertionError(f"target mismatch {n}/{fold}")
            rec[n]=np.asarray(d["z"],dtype=np.float64)
            if n=="hurdle":
                rec["p_hurdle"]=np.clip(np.asarray(d["p"],dtype=np.float64),EPS,1-EPS)
                rec["mu"]=np.maximum(np.asarray(d["mu"],dtype=np.float64),0)
        for fam in ("seq42","etx42"):
            if neural_complete.get(fam):
                d=prev.align_to_uid(prev.load_fold(prev.fold_ckpt(ctx,fam,fold)),uid)
                prev.validate_fold_record(d,f"{fam}/{fold}")
                rec[fam]=np.asarray(d["z"],dtype=np.float64)
        # structural historical proxy: missing optional sequence family falls back to DIST
        rec["proxy"]=clip_z(sum(float(w)*np.asarray(rec.get(n,rec["dist"]),dtype=np.float64)
                                 for n,w in proxy_weights.items()))
        rec["hurdle_temporal"]=rec["hurdle"].copy()
        bank[fold]=rec
    return bank


def build_test_record_core(prev, ctx, friend: dict[str,Any], neural_complete: dict[str,bool]) -> dict[str,Any]:
    """Future record that reuses even an older minimal hurdle_test checkpoint if possible."""
    uid=np.asarray(friend["uid"],dtype=np.int64)
    rec={"user_id":uid,"friend":np.asarray(friend["z"],dtype=np.float64)}
    mapping={"cap":"S1-CAP","unc":"S1-UNC","dist":"S1-DIST"}
    if neural_complete.get("seq42"): mapping["seq42"]="SEQ-01"
    if neural_complete.get("etx42"): mapping["etx42"]="ETX-01-S42-DCW"
    for key,comp in mapping.items():
        u,z=prev.load_component_test(ctx.package,comp)
        d=prev.align_to_uid({"user_id":np.asarray(u,dtype=np.int64),"z":np.asarray(z,dtype=np.float64)},uid)
        rec[key]=clip_z(d["z"])
    hp=ctx.checkpoints/"test"/"hurdle_test.npz"
    h=None
    if hp.exists():
        try:
            d=prev.load_fold(hp)
            if {"user_id","z","p","mu"}.issubset(d): h=prev.align_to_uid(d,uid);log("reuse minimal/full hurdle_test.npz")
        except Exception as e:
            log("Existing hurdle_test unreadable; rebuild:",repr(e))
    if h is None:
        h=prev.align_to_uid(prev.train_table_test_hurdle(ctx),uid)
    rec["hurdle"]=clip_z(h["z"]);rec["hurdle_temporal"]=rec["hurdle"].copy()
    rec["p_hurdle"]=np.clip(np.asarray(h["p"],dtype=np.float64),EPS,1-EPS)
    rec["mu"]=np.maximum(np.asarray(h["mu"],dtype=np.float64),0)
    u_meta,Xraw,names=prev.build_test_meta_raw(ctx)
    md=prev.align_to_uid({"user_id":np.asarray(u_meta,dtype=np.int64),"X":np.asarray(Xraw)},uid)
    rec["meta_raw"]=np.asarray(md["X"],dtype=np.float32);rec["meta_names"]=list(names)
    proxy_weights=getattr(prev,"PROXY_COMPONENT_WEIGHTS",{"cap":.10,"unc":.20,"dist":.25,"seq42":.225,"etx42":.225})
    rec["proxy"]=clip_z(sum(float(w)*np.asarray(rec.get(n,rec["dist"]),dtype=np.float64)
                              for n,w in proxy_weights.items()))
    return rec


def weighted_cv(a: Iterable[float]) -> float:
    a = np.asarray(list(a), dtype=np.float64)
    return float(np.dot(a, FOLD_WEIGHTS) / FOLD_WEIGHTS.sum())


def clip_z(x: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0, posinf=20.0, neginf=0.0), 0.0, 20.0)


def make_clf(seed: int, threads: int, n_estimators: int = 220):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        n_estimators=n_estimators, learning_rate=0.035, num_leaves=31, max_depth=7,
        min_child_samples=180, subsample=0.88, subsample_freq=1, colsample_bytree=0.82,
        reg_alpha=3.0, reg_lambda=18.0, random_state=seed, n_jobs=threads, verbosity=-1,
    )


def make_reg(seed: int, threads: int, objective: str = "huber", alpha: float | None = None,
             n_estimators: int = 260):
    from lightgbm import LGBMRegressor
    kw = dict(
        n_estimators=n_estimators, learning_rate=0.03, num_leaves=31, max_depth=7,
        min_child_samples=180, subsample=0.90, subsample_freq=1, colsample_bytree=0.84,
        reg_alpha=2.0, reg_lambda=18.0, objective=objective, random_state=seed,
        n_jobs=threads, verbosity=-1,
    )
    if alpha is not None: kw["alpha"] = alpha
    return LGBMRegressor(**kw)


def base_matrix(prev, rec: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    X, names = prev.meta_matrix(rec)
    # Defensive memory cap. Keep the first 88 stable columns if an older/newer runner exposed more.
    if X.shape[1] > 64:
        X = X[:, :64]; names = names[:64]
    return np.asarray(X, dtype=np.float32), list(names)


def error_labels(rec: dict[str, Any]) -> dict[str, np.ndarray]:
    y = np.asarray(rec["y"], dtype=np.float64)
    ly = np.log1p(y)
    p = np.asarray(rec["p_hurdle"], dtype=np.float64)
    base = np.asarray(rec["proxy"], dtype=np.float64)
    c1 = p >= 0.5
    resid = base - ly
    return {
        "false_one": (c1 & (y <= 0)).astype(np.int8),
        "false_zero": ((~c1) & (y > 0)).astype(np.int8),
        "over": (c1 & (resid > 0.75)).astype(np.int8),
        "under": (c1 & (resid < -0.75)).astype(np.int8),
        "cat": (np.abs(resid) > 1.50).astype(np.int8),
    }


RISK_KEYS = ("false_one", "false_zero", "over", "under", "cat")


def heuristic_risks(rec: dict[str, Any]) -> dict[str, np.ndarray]:
    p = np.asarray(rec["p_hurdle"], dtype=np.float64)
    disagreement = np.zeros(len(p), dtype=np.float64)
    es = [np.asarray(rec[n], dtype=np.float64) for n in ("cap", "unc", "dist", "hurdle") if n in rec]
    if es:
        disagreement = np.std(np.column_stack(es), axis=1)
        disagreement = np.clip(disagreement / (np.nanquantile(disagreement, .90) + 1e-6), 0, 1)
    return {
        "false_one": np.clip((p - .5) * 1.1 + .25 * disagreement, 0, 1),
        "false_zero": np.clip((.5 - p) * 1.1 + .15 * disagreement, 0, 1),
        "over": np.clip(.55 * p + .45 * disagreement, 0, 1),
        "under": np.clip(.35 * (1-p) + .45 * disagreement, 0, 1),
        "cat": np.clip(disagreement, 0, 1),
    }


def concat_folds(prev, bank: dict[str, dict[str, Any]], idxs: list[int]):
    Xs, ys, ps, folds = [], [], [], []
    names = None
    for j in idxs:
        rec = bank[FOLDS[j]]
        X, nm = base_matrix(prev, rec)
        if names is None: names = nm
        elif names != nm: raise AssertionError("base feature schema changed")
        Xs.append(X); ys.append(np.asarray(rec["y"], dtype=np.float64)); ps.append(np.asarray(rec["p_hurdle"], dtype=np.float64))
        folds.append(np.full(len(rec["y"]), j, dtype=np.int8))
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(ps), np.concatenate(folds), names


def fit_overtime_error_detectors(prev, bank: dict[str, dict[str, Any]], threads: int,
                                 report: list[dict[str, Any]], seed: int = 15100) -> None:
    from sklearn.metrics import roc_auc_score
    for i, fold in enumerate(FOLDS):
        rec = bank[fold]
        if i == 0:
            risks = heuristic_risks(rec)
            for k, v in risks.items(): rec[f"risk_{k}"] = v
            continue
        Xtr, _, _, _, _ = concat_folds(prev, bank, list(range(i)))
        Xt, _ = base_matrix(prev, rec)
        labs_hist = {k: [] for k in RISK_KEYS}
        for j in range(i):
            L = error_labels(bank[FOLDS[j]])
            for k in RISK_KEYS: labs_hist[k].append(L[k])
        truth = error_labels(rec)
        for ki, k in enumerate(RISK_KEYS):
            ytr = np.concatenate(labs_hist[k])
            rate = float(ytr.mean())
            if rate < .005 or rate > .995:
                pr = np.full(len(rec["y"]), rate, dtype=np.float64)
            else:
                clf = make_clf(seed + 100*i + ki, threads, n_estimators=200)
                clf.fit(Xtr, ytr)
                pr = clf.predict_proba(Xt)[:, 1]
                del clf
            rec[f"risk_{k}"] = np.clip(pr, EPS, 1-EPS)
            auc = float("nan")
            try:
                if len(np.unique(truth[k])) == 2:
                    auc = float(roc_auc_score(truth[k], pr))
            except Exception:
                pass
            report.append(dict(fold=fold, detector=k, train_positive_rate=rate,
                               val_positive_rate=float(truth[k].mean()), auc=auc))
        del Xtr, Xt
        gc.collect()


def recency_weights(fold_ids: np.ndarray) -> np.ndarray:
    # Recent anchors matter more, but avoid an extreme 1:2:4:8 dominance inside model fitting.
    w = 1.0 + 0.55 * np.asarray(fold_ids, dtype=np.float64)
    return w / w.mean()


def fit_specialists_one(prev, bank: dict[str, dict[str, Any]], target_idx: int, threads: int,
                        seed: int = 16100) -> None:
    rec = bank[FOLDS[target_idx]]
    Xt, _ = base_matrix(prev, rec)
    if target_idx == 0:
        for n in ("class1_direct", "class1_over_guard", "class1_q35", "class1_recent", "class1_occ", "class1_riskblend"):
            rec[n] = np.asarray(rec["proxy"], dtype=np.float64).copy()
        return
    Xtr, ytr, ptr, fids, _ = concat_folds(prev, bank, list(range(target_idx)))
    ztr = np.log1p(ytr)
    c1 = ptr >= .5
    wt = recency_weights(fids)
    if c1.sum() < 5000:
        raise RuntimeError("Too few class1 historical rows")

    # A. general class1 z expert.
    reg = make_reg(seed + target_idx, threads, "huber", n_estimators=260)
    reg.fit(Xtr[c1], ztr[c1], sample_weight=wt[c1])
    z_direct = clip_z(reg.predict(Xt)); del reg

    # B. over-guard: past false-ones and large overpredictions receive extra weight.
    # Labels are historical only, so this is temporally valid and NOT a direct correction model.
    base_hist = np.concatenate([bank[FOLDS[j]]["proxy"] for j in range(target_idx)])
    over_amt = base_hist - ztr
    wguard = wt * (1.0 + 2.5*((ytr<=0)&c1) + 1.6*(over_amt>.75) + 0.5*(over_amt>1.5))
    reg = make_reg(seed + 100 + target_idx, threads, "huber", n_estimators=280)
    reg.fit(Xtr[c1], ztr[c1], sample_weight=wguard[c1])
    z_guard = clip_z(reg.predict(Xt)); del reg

    # C. conservative quantile expert for the known high-overprediction class1 zone.
    reg = make_reg(seed + 200 + target_idx, threads, "quantile", alpha=.35, n_estimators=240)
    reg.fit(Xtr[c1], ztr[c1], sample_weight=wt[c1])
    z_q35 = clip_z(reg.predict(Xt)); del reg

    # D. recent-only class1 expert (last two mature regimes only).
    recent_idxs = list(range(max(0, target_idx-2), target_idx))
    Xr, yr, pr, fr, _ = concat_folds(prev, bank, recent_idxs)
    zr = np.log1p(yr); mr = pr >= .5; wr = recency_weights(fr)
    reg = make_reg(seed + 300 + target_idx, threads, "huber", n_estimators=240)
    reg.fit(Xr[mr], zr[mr], sample_weight=wr[mr])
    z_recent = clip_z(reg.predict(Xt)); del reg

    # E. tree hurdle INSIDE class1: separate occurrence from positive magnitude.
    yb = (ytr > 0).astype(np.int8)
    clf = make_clf(seed + 400 + target_idx, threads, n_estimators=220)
    clf.fit(Xtr[c1], yb[c1], sample_weight=wt[c1])
    pocc = np.clip(clf.predict_proba(Xt)[:,1], EPS, 1-EPS); del clf
    pos = c1 & (ytr > 0)
    reg = make_reg(seed + 450 + target_idx, threads, "huber", n_estimators=240)
    reg.fit(Xtr[pos], ztr[pos], sample_weight=wt[pos])
    mu_pos = clip_z(reg.predict(Xt)); del reg
    z_occ = clip_z(pocc * mu_pos)

    c1t = np.asarray(rec["p_hurdle"]) >= .5
    # Keep class0 untouched. This is central to the Phase14 conclusion.
    for name, zz in (("class1_direct", z_direct), ("class1_over_guard", z_guard),
                     ("class1_q35", z_q35), ("class1_recent", z_recent), ("class1_occ", z_occ)):
        out = np.asarray(rec["proxy"], dtype=np.float64).copy()
        out[c1t] = zz[c1t]
        rec[name] = clip_z(out)

    # Risk blend: only highly suspicious class1 rows move toward q35/over-guard.
    risk = np.sqrt(np.clip(rec["risk_false_one"],0,1) * np.clip(rec["risk_over"],0,1))
    rc1 = risk[c1t]
    thr = float(np.quantile(rc1, .60)) if len(rc1) else 1.0
    alpha = np.zeros(len(risk), dtype=np.float64)
    if len(rc1):
        den = max(float(np.quantile(rc1,.92) - thr), 1e-6)
        alpha[c1t] = np.clip((risk[c1t]-thr)/den, 0, 1) * .80
    target = .55*z_guard + .45*z_q35
    rec["class1_riskblend"] = clip_z((1-alpha)*rec["proxy"] + alpha*target)

    del Xtr, Xt, Xr, ytr, ptr, fids, yr, pr, fr
    gc.collect()


SPECIALISTS = ("class1_direct", "class1_over_guard", "class1_q35", "class1_recent", "class1_occ", "class1_riskblend")


def fit_all_overtime_specialists(prev, bank, threads: int) -> None:
    for i, fold in enumerate(FOLDS):
        log("CLASS1 specialists ->", fold)
        fit_specialists_one(prev, bank, i, threads)


def expert_names(bank: dict[str, dict[str, Any]], neural_complete: dict[str, bool]) -> list[str]:
    names = ["proxy", "cap", "unc", "dist", "hurdle", *SPECIALISTS]
    if neural_complete.get("seq42") and all("seq42" in bank[f] for f in FOLDS): names.append("seq42")
    if neural_complete.get("etx42") and all("etx42" in bank[f] for f in FOLDS): names.append("etx42")
    # Stable unique order.
    return list(dict.fromkeys(names))


def risk_matrix(rec: dict[str, Any]) -> np.ndarray:
    return np.column_stack([np.asarray(rec[f"risk_{k}"], dtype=np.float32) for k in RISK_KEYS])


def make_ranker(seed: int, threads: int, n_estimators: int = 220):
    from lightgbm import LGBMRanker
    return LGBMRanker(
        objective="lambdarank", metric="ndcg", ndcg_at=[1,3],
        n_estimators=n_estimators, learning_rate=.035, num_leaves=31, max_depth=7,
        min_child_samples=120, subsample=.88, subsample_freq=1, colsample_bytree=.82,
        reg_alpha=2.0, reg_lambda=18.0, random_state=seed, n_jobs=threads, verbosity=-1,
    )


def deterministic_subset(uid: np.ndarray, max_users: int, seed: int) -> np.ndarray:
    uid = np.asarray(uid, dtype=np.uint64)
    # deterministic 64-bit mix, no Python hash randomization
    x = uid ^ np.uint64(seed * 0x9E3779B1)
    x ^= x >> np.uint64(30); x *= np.uint64(0xBF58476D1CE4E5B9)
    x ^= x >> np.uint64(27); x *= np.uint64(0x94D049BB133111EB)
    x ^= x >> np.uint64(31)
    if len(uid) <= max_users: return np.arange(len(uid), dtype=np.int64)
    return np.argpartition(x, max_users)[:max_users].astype(np.int64)


def ranker_rows(prev, rec: dict[str, Any], experts: list[str], idx: np.ndarray | None = None,
                include_labels: bool = True):
    Xb, _ = base_matrix(prev, rec)
    if idx is None: idx = np.arange(len(rec["y"]), dtype=np.int64)
    Xb = Xb[idx]
    R = risk_matrix(rec)[idx]
    E = len(experts); n = len(idx)
    ez = np.column_stack([np.asarray(rec[e], dtype=np.float32)[idx] for e in experts])
    proxy = np.asarray(rec["proxy"], dtype=np.float32)[idx]
    mean_e = ez.mean(axis=1); std_e = ez.std(axis=1)

    # Repeat user context E times; expert-specific block adds low-memory discriminative features.
    ctx = np.column_stack([Xb, R, np.asarray(rec["p_hurdle"],dtype=np.float32)[idx],
                           np.asarray(rec["mu"],dtype=np.float32)[idx], mean_e, std_e]).astype(np.float32)
    Xrep = np.repeat(ctx, E, axis=0)
    expert_z = ez.reshape(-1)
    proxy_rep = np.repeat(proxy, E)
    mean_rep = np.repeat(mean_e, E)
    sp = np.column_stack([expert_z, expert_z-proxy_rep, np.abs(expert_z-proxy_rep), expert_z-mean_rep]).astype(np.float32)
    onehot = np.tile(np.eye(E, dtype=np.float32), (n,1))
    X = np.column_stack([Xrep, sp, onehot]).astype(np.float32)
    groups = np.full(n, E, dtype=np.int32)
    labels = None
    if include_labels:
        ly = np.log1p(np.asarray(rec["y"],dtype=np.float64)[idx])[:,None]
        losses = (ez.astype(np.float64)-ly)**2
        order = np.argsort(losses, axis=1)  # best first
        rel = np.zeros((n,E), dtype=np.int32)
        # E-1 for best ... 0 for worst.
        ranks = np.arange(E-1,-1,-1,dtype=np.int32)
        rows = np.arange(n)[:,None]
        rel[rows, order] = ranks[None,:]
        labels = rel.reshape(-1)
    return X, labels, groups, ez


def predict_ranker(prev, model, rec: dict[str, Any], experts: list[str], mask: np.ndarray,
                   chunk_users: int = 35000):
    n = len(rec["y"]) if "y" in rec else len(rec["user_id"])
    chosen = np.asarray(rec["proxy"], dtype=np.float64).copy()
    margin = np.zeros(n, dtype=np.float64)
    chosen_idx = np.zeros(n, dtype=np.int16)
    ids = np.flatnonzero(mask)
    for start in range(0, len(ids), chunk_users):
        ix = ids[start:start+chunk_users]
        X, _, _, ez = ranker_rows(prev, rec, experts, idx=ix, include_labels=False)
        sc = model.predict(X).reshape(len(ix), len(experts))
        k = np.argmax(sc, axis=1)
        chosen[ix] = ez[np.arange(len(ix)), k]
        chosen_idx[ix] = k
        if len(experts) >= 2:
            part = np.partition(sc, -2, axis=1)
            margin[ix] = part[:,-1] - part[:,-2]
        else:
            margin[ix] = 1.0
        del X, sc, ez
    return chosen, margin, chosen_idx


def add_ranker_candidates(prev, bank: dict[str, dict[str, Any]], experts: list[str], threads: int,
                          max_users: int, report: list[dict[str, Any]], seed: int = 17100) -> None:
    for i, fold in enumerate(FOLDS):
        rec = bank[fold]
        c1 = np.asarray(rec["p_hurdle"]) >= .5
        if i == 0:
            for n in ("ranker_safe", "ranker_balanced", "ranker_risk", "ranker_full"):
                rec[n] = np.asarray(rec["proxy"],dtype=np.float64).copy()
            continue

        # Build training from PAST fold predictions that were themselves OOT.
        Xs=[]; ys=[]; groups=[]; train_users=0
        for j in range(i):
            rr = bank[FOLDS[j]]
            mask = np.asarray(rr["p_hurdle"]) >= .5
            ids = np.flatnonzero(mask)
            if j == 0 and i > 1:
                # fold0 specialist predictions are fallbacks; keep only a smaller control sample.
                cap = max(8000, max_users//5)
            else:
                cap = max_users
            sub = deterministic_subset(np.asarray(rr["user_id"])[ids], min(cap,len(ids)), seed+100*i+j)
            ids = ids[sub]
            X, lab, grp, _ = ranker_rows(prev, rr, experts, idx=ids, include_labels=True)
            Xs.append(X); ys.append(lab); groups.append(grp); train_users += len(ids)
        Xtr=np.vstack(Xs); ytr=np.concatenate(ys); grp=np.concatenate(groups)
        model=make_ranker(seed+i,threads,n_estimators=220)
        model.fit(Xtr,ytr,group=grp)
        chosen, margin, kid = predict_ranker(prev,model,rec,experts,c1)

        m = margin[c1]
        q75=float(np.quantile(m,.75)) if len(m) else math.inf
        q50=float(np.quantile(m,.50)) if len(m) else math.inf
        risk=np.sqrt(np.clip(rec["risk_false_one"],0,1)*np.clip(rec["risk_over"],0,1))
        rthr=float(np.quantile(risk[c1],.65)) if c1.any() else 1.0

        safe=np.asarray(rec["proxy"],dtype=np.float64).copy(); act=c1&(margin>=q75)
        safe[act]=.35*safe[act]+.65*chosen[act]
        bal=np.asarray(rec["proxy"],dtype=np.float64).copy(); act2=c1&(margin>=q50)
        bal[act2]=.25*bal[act2]+.75*chosen[act2]
        rrisk=np.asarray(rec["proxy"],dtype=np.float64).copy(); act3=c1&(risk>=rthr)&(margin>=q50)
        rrisk[act3]=.20*rrisk[act3]+.80*chosen[act3]
        full=np.asarray(rec["proxy"],dtype=np.float64).copy(); full[c1]=chosen[c1]
        rec["ranker_safe"]=clip_z(safe);rec["ranker_balanced"]=clip_z(bal);rec["ranker_risk"]=clip_z(rrisk);rec["ranker_full"]=clip_z(full)

        # Simple top1 accuracy against realized oracle expert inside class1.
        ez=np.column_stack([rec[e] for e in experts]); ly=np.log1p(rec["y"])[:,None]
        oracle=np.argmin((ez-ly)**2,axis=1)
        top1=float(np.mean(kid[c1]==oracle[c1])) if c1.any() else float("nan")
        report.append(dict(fold=fold, train_users=train_users, n_experts=len(experts),
                           class1_users=int(c1.sum()), top1_oracle_accuracy=top1,
                           margin_q50=q50, margin_q75=q75,
                           safe_active=int((c1&(margin>=q75)).sum()),
                           balanced_active=int((c1&(margin>=q50)).sum()),
                           risk_active=int(act3.sum())))
        del Xtr,ytr,grp,model,chosen,margin,kid,ez,ly,oracle
        gc.collect()


CANDIDATES = ["proxy", *SPECIALISTS, "ranker_safe", "ranker_balanced", "ranker_risk", "ranker_full"]


def score_candidate(prev, bank, name: str) -> dict[str, Any]:
    scores=[]; deltas=[]; rows=[]
    for fold in FOLDS:
        rec=bank[fold]
        if name not in rec: raise KeyError(name)
        off,sc=prev.calibrate(rec["y"],rec[name]); boff,bsc=prev.calibrate(rec["y"],rec["proxy"])
        d=sc-bsc;scores.append(sc);deltas.append(d)
        rows.append(dict(candidate=name,fold=fold,rmsle=sc,offset=off,proxy_rmsle=bsc,delta_vs_proxy=d))
    return dict(name=name,wcv=weighted_cv(scores),wins=int(sum(d<0 for d in deltas)),
                latest_delta=float(deltas[-1]),worst_delta=float(max(deltas)),std_delta=float(np.std(deltas)),
                deltas=deltas,rows=rows)


def evaluate(prev, bank):
    ss=[];fr=[]
    for n in CANDIDATES:
        if not all(n in bank[f] for f in FOLDS): continue
        s=score_candidate(prev,bank,n);fr+=s.pop("rows");ss.append(s)
    ss.sort(key=lambda r:(r["wcv"],r["worst_delta"],r["std_delta"]))
    return ss,fr


def segment_report(prev, bank, names: list[str]) -> list[dict[str, Any]]:
    rows=[]
    for fold in FOLDS:
        rec=bank[fold]; y=np.asarray(rec["y"],dtype=np.float64); p=np.asarray(rec["p_hurdle"])
        c1=p>=.5
        rf=np.asarray(rec["risk_false_one"]); ro=np.asarray(rec["risk_over"])
        qf=float(np.quantile(rf[c1],.70)) if c1.any() else 1.0
        qo=float(np.quantile(ro[c1],.70)) if c1.any() else 1.0
        segs={"all":np.ones(len(y),bool),"class0":~c1,"class1":c1,
              "class1_high_false_one":c1&(rf>=qf),"class1_high_over":c1&(ro>=qo),
              "class1_false_one_and_over":c1&(rf>=qf)&(ro>=qo)}
        for n in names:
            off,_=prev.calibrate(y,rec[n]); z=np.maximum(np.asarray(rec[n])+off,0)
            ly=np.log1p(y)
            for sn,m in segs.items():
                if m.sum()==0: continue
                sc=float(np.sqrt(np.mean((ly[m]-z[m])**2)))
                rows.append(dict(fold=fold,candidate=n,segment=sn,n=int(m.sum()),share=float(m.mean()),rmsle=sc))
    return rows


def fit_final_error_detectors(prev, bank, test, threads: int, seed: int=18100):
    Xtr,ytr,ptr,fids,_=concat_folds(prev,bank,list(range(4)))
    Xt,_=base_matrix(prev,test)
    labs={k:[] for k in RISK_KEYS}
    for j in range(4):
        L=error_labels(bank[FOLDS[j]])
        for k in RISK_KEYS: labs[k].append(L[k])
    models={}
    for ki,k in enumerate(RISK_KEYS):
        yy=np.concatenate(labs[k]); rate=float(yy.mean())
        if rate<.005 or rate>.995:
            test[f"risk_{k}"]=np.full(len(test["user_id"]),rate,dtype=np.float64)
            continue
        clf=make_clf(seed+ki,threads,n_estimators=220);clf.fit(Xtr,yy,sample_weight=recency_weights(fids))
        test[f"risk_{k}"]=np.clip(clf.predict_proba(Xt)[:,1],EPS,1-EPS);models[k]=clf
    return models


def fit_final_specialists(prev, bank, test, threads: int, seed: int=19100):
    Xtr,ytr,ptr,fids,_=concat_folds(prev,bank,list(range(4)));Xt,_=base_matrix(prev,test)
    ztr=np.log1p(ytr);c1=ptr>=.5;wt=recency_weights(fids)
    base_hist=np.concatenate([bank[f]["proxy"] for f in FOLDS]);over_amt=base_hist-ztr
    outs={}
    reg=make_reg(seed,threads,"huber",n_estimators=280);reg.fit(Xtr[c1],ztr[c1],sample_weight=wt[c1]);outs["class1_direct"]=clip_z(reg.predict(Xt));del reg
    wg=wt*(1+2.5*((ytr<=0)&c1)+1.6*(over_amt>.75)+.5*(over_amt>1.5))
    reg=make_reg(seed+1,threads,"huber",n_estimators=300);reg.fit(Xtr[c1],ztr[c1],sample_weight=wg[c1]);outs["class1_over_guard"]=clip_z(reg.predict(Xt));del reg
    reg=make_reg(seed+2,threads,"quantile",alpha=.35,n_estimators=260);reg.fit(Xtr[c1],ztr[c1],sample_weight=wt[c1]);outs["class1_q35"]=clip_z(reg.predict(Xt));del reg
    # latest two folds
    Xr,yr,pr,fr,_=concat_folds(prev,bank,[2,3]);zr=np.log1p(yr);mr=pr>=.5;wr=recency_weights(fr)
    reg=make_reg(seed+3,threads,"huber",n_estimators=260);reg.fit(Xr[mr],zr[mr],sample_weight=wr[mr]);outs["class1_recent"]=clip_z(reg.predict(Xt));del reg
    yb=(ytr>0).astype(np.int8);clf=make_clf(seed+4,threads,240);clf.fit(Xtr[c1],yb[c1],sample_weight=wt[c1]);pocc=np.clip(clf.predict_proba(Xt)[:,1],EPS,1-EPS);del clf
    pos=c1&(ytr>0);reg=make_reg(seed+5,threads,"huber",n_estimators=240);reg.fit(Xtr[pos],ztr[pos],sample_weight=wt[pos]);mupos=clip_z(reg.predict(Xt));del reg
    outs["class1_occ"]=clip_z(pocc*mupos)
    c1t=np.asarray(test["p_hurdle"])>=.5
    for n,zz in outs.items():
        out=np.asarray(test["friend"],dtype=np.float64).copy();out[c1t]=zz[c1t];test[n]=clip_z(out)
    risk=np.sqrt(np.clip(test["risk_false_one"],0,1)*np.clip(test["risk_over"],0,1));rc=risk[c1t]
    thr=float(np.quantile(rc,.60)) if len(rc) else 1.;den=max(float(np.quantile(rc,.92)-thr),1e-6) if len(rc) else 1.
    a=np.zeros(len(risk));a[c1t]=np.clip((risk[c1t]-thr)/den,0,1)*.80
    target=.55*outs["class1_over_guard"]+.45*outs["class1_q35"]
    test["class1_riskblend"]=clip_z((1-a)*test["friend"]+a*target)
    del Xtr,Xt,Xr,ytr,ptr,fids,yr,pr,fr
    gc.collect()


def fit_final_ranker(prev, bank, test, experts: list[str], threads: int, max_users: int,
                     report: list[dict[str,Any]], seed: int=20100):
    # Use folds 1..3 preferentially: their specialist/risk predictions are genuinely OOT.
    Xs=[];ys=[];groups=[];nusers=0
    for j in (1,2,3):
        rec=bank[FOLDS[j]];mask=np.asarray(rec["p_hurdle"])>=.5;ids=np.flatnonzero(mask)
        sub=deterministic_subset(np.asarray(rec["user_id"])[ids],min(max_users,len(ids)),seed+j);ids=ids[sub]
        X,lab,grp,_=ranker_rows(prev,rec,experts,idx=ids,include_labels=True)
        Xs.append(X);ys.append(lab);groups.append(grp);nusers+=len(ids)
    Xtr=np.vstack(Xs);ytr=np.concatenate(ys);grp=np.concatenate(groups)
    model=make_ranker(seed,threads,240);model.fit(Xtr,ytr,group=grp)
    c1=np.asarray(test["p_hurdle"])>=.5
    # For deployment, the historical proxy expert is mapped to exact STRONGEST_CURRENT.
    # Temporarily replace proxy only for expert materialization; context features stay structural.
    proxy_struct=np.asarray(test["proxy"],dtype=np.float64).copy()
    chosen,margin,kid=predict_ranker(prev,model,test,experts,c1)
    proxy_index=experts.index("proxy")
    chosen[(c1)&(kid==proxy_index)]=np.asarray(test["friend"])[(c1)&(kid==proxy_index)]
    m=margin[c1];q75=float(np.quantile(m,.75));q50=float(np.quantile(m,.50))
    risk=np.sqrt(np.clip(test["risk_false_one"],0,1)*np.clip(test["risk_over"],0,1));rthr=float(np.quantile(risk[c1],.65))
    safe=np.asarray(test["friend"],dtype=np.float64).copy();a=c1&(margin>=q75);safe[a]=.35*safe[a]+.65*chosen[a]
    bal=np.asarray(test["friend"],dtype=np.float64).copy();a2=c1&(margin>=q50);bal[a2]=.25*bal[a2]+.75*chosen[a2]
    rr=np.asarray(test["friend"],dtype=np.float64).copy();a3=c1&(risk>=rthr)&(margin>=q50);rr[a3]=.20*rr[a3]+.80*chosen[a3]
    full=np.asarray(test["friend"],dtype=np.float64).copy();full[c1]=chosen[c1]
    test["ranker_safe"]=clip_z(safe);test["ranker_balanced"]=clip_z(bal);test["ranker_risk"]=clip_z(rr);test["ranker_full"]=clip_z(full)
    report.append(dict(stage="final",train_users=nusers,n_experts=len(experts),class1_users=int(c1.sum()),
                       margin_q50=q50,margin_q75=q75,risk_threshold=rthr,safe_active=int(a.sum()),balanced_active=int(a2.sum()),risk_active=int(a3.sum())))
    del Xtr,ytr,grp,model,chosen,margin,kid
    gc.collect()


def diversity(prev, a: np.ndarray, b: np.ndarray) -> dict[str,float]:
    aa,_=prev.level_calibrate_test(a);bb,_=prev.level_calibrate_test(b);d=aa-bb
    return dict(corr=float(np.corrcoef(aa,bb)[0,1]),std_delta=float(np.std(d)),mean_abs_delta=float(np.mean(np.abs(d))),
                p_abs_gt_005=float(np.mean(np.abs(d)>.05)),p_abs_gt_010=float(np.mean(np.abs(d)>.10)))


def select_three(prev, summaries: list[dict[str,Any]], test: dict[str,Any]) -> list[dict[str,Any]]:
    rows=[]
    for s in summaries:
        n=s["name"]
        if n=="proxy" or n not in test: continue
        d=diversity(prev,test[n],test["friend"]);r=dict(s);r.update({f"div_friend_{k}":v for k,v in d.items()})
        r["family"]="ranker" if n.startswith("ranker") else "specialist"
        # Emphasize latest fold and temporal stability over tiny mean differences.
        r["selection_cost"]=float(s["wcv"]+.35*max(0,s["worst_delta"])+.20*s["std_delta"]+.35*max(0,s["latest_delta"]))
        rows.append(r)
    rows.sort(key=lambda r:(r["selection_cost"],r["wcv"]))
    if len(rows)<3: raise RuntimeError("Fewer than three deployable candidates")

    selected=[]
    # 1. safest locally stable candidate.
    safe=[r for r in rows if r["wins"]>=2 and r["latest_delta"]<=0 and (r["div_friend_p_abs_gt_005"]>=.08 or r["div_friend_std_delta"]>=.015)]
    selected.append(safe[0] if safe else rows[0])
    # 2. force the other family if possible.
    other=[r for r in rows if r["name"]!=selected[0]["name"] and r["family"]!=selected[0]["family"]]
    selected.append(other[0] if other else next(r for r in rows if r["name"]!=selected[0]["name"]))
    # 3. maximize useful diversity subject to not being wildly worse than best.
    used={r["name"] for r in selected};best=rows[0]["wcv"]
    pool=[r for r in rows if r["name"] not in used and r["wcv"]<=best+.0035 and r["latest_delta"]<=.0025]
    if not pool: pool=[r for r in rows if r["name"] not in used]
    def divscore(r):
        vals=[]
        for x in selected:
            d=diversity(prev,test[r["name"]],test[x["name"]]);vals.append(d["std_delta"]+.35*d["p_abs_gt_005"])
        return min(vals) if vals else 0
    pool.sort(key=lambda r:(-divscore(r),r["selection_cost"]))
    selected.append(pool[0])
    return selected


def write_submissions(prev, ctx, selected, test):
    import pandas as pd
    sample=pd.read_csv(ctx.sample,usecols=["user_id"])
    uid=np.asarray(test["user_id"],dtype=np.int64);order=np.argsort(uid);su=uid[order]
    pos=np.searchsorted(su,sample.user_id.to_numpy(np.int64))
    if (pos>=len(uid)).any() or not np.array_equal(su[pos],sample.user_id.to_numpy(np.int64)):
        raise ValueError("test user_id mismatch")
    inv=order[pos];outs=[]
    labels=["SAFE","CLASS1","DIVERSE"]
    for i,(lab,s) in enumerate(zip(labels,selected),1):
        z,shift=prev.level_calibrate_test(test[s["name"]]);pred=np.maximum(np.expm1(z),0)
        df=sample.copy();df["predict"]=pred[inv];prev.validate_submission_frame(df,ctx.sample)
        path=ctx.submissions/f"submission_continue12h_{i}_{lab.lower()}_{s['name']}.csv"
        df.to_csv(path,index=False,float_format="%.6f")
        outs.append(dict(rank=i,label=lab,candidate=s["name"],path=str(path),sha256=prev.sha256(path),
                         validation_wcv=s["wcv"],wins=s["wins"],latest_delta=s["latest_delta"],level_shift=shift,
                         diversity_vs_friend={k:v for k,v in s.items() if k.startswith("div_friend_")}))
    return outs


def combine_previous_report(prev_work: Path, new_results: Path, manifest: dict[str,Any],
                            summaries, detector_rows, rank_rows, seg_rows, outs) -> None:
    lines=[]
    lines.append("E-CUP 2026 — ОБЩИЙ ОТЧЁТ: ПРЕДЫДУЩИЙ ЗАПУСК + 12H CONTINUATION\n")
    lines.append(f"Версия continuation: {VERSION}")
    lines.append(f"Предыдущая рабочая папка: {prev_work}")
    lines.append(f"Начало continuation: {manifest.get('started_at')}")
    lines.append(f"Завершение continuation: {manifest.get('finished_at')}")
    lines.append(f"Время continuation: {manifest.get('runtime_hours',float('nan')):.3f} ч")
    lines.append("\n=== 1. Что сохранилось от предыдущего запуска ===")
    counts=manifest["previous_checkpoint_counts"]["detail"]
    for k,v in counts.items(): lines.append(f"{k}: {len(v)}/4 folds -> {', '.join(v) if v else '-'}")
    old_prog=safe_read_json(prev_work/"results"/"progress.json")
    old_man=safe_read_json(prev_work/"results"/"RUN_MANIFEST.json")
    if old_prog: lines.append("Старый progress.json: "+json.dumps(old_prog,ensure_ascii=False))
    if old_man:
        lines.append("Старый RUN_MANIFEST найден; started_at="+str(old_man.get("started_at"))+", finished_at="+str(old_man.get("finished_at")))
    errp=prev_work/"results"/"errors.jsonl"
    if errp.exists():
        try:
            errs=errp.read_text(encoding="utf-8",errors="replace").strip().splitlines()
            lines.append(f"Старый errors.jsonl: {len(errs)} записей; последняя: {errs[-1][:700] if errs else '-'}")
        except Exception: pass
    repaired_cache = manifest.get("repaired_truncated_cache_files") or []
    repaired_ckpt = manifest.get("repaired_corrupt_checkpoints") or []
    lines.append("\n=== 1b. Восстановление после переполнения диска ===")
    lines.append(f"Удалено только повреждённых производных parquet-кэшей: {len(repaired_cache)}")
    lines.append(f"Удалено только нечитаемых NPZ checkpoints: {len(repaired_ckpt)}")
    if repaired_cache:
        lines.append("Повреждённые cache-файлы: " + ", ".join(Path(x).name for x in repaired_cache[:30]))
    if repaired_ckpt:
        lines.append("Повреждённые checkpoints: " + ", ".join(Path(x).name for x in repaired_ckpt[:30]))
    lines.append("\n=== 2. Что сделал continuation ===")
    lines.append("Новые neural OOF НЕ обучались. Использованы только полностью готовые старые seq42/etx42, если они существовали.")
    lines.append("Error-detectors применялись только как признаки маршрутизации; прямых magnitude corrections нет.")
    lines.append("Основная оптимизация ограничена classifier=1; class0 по умолчанию оставлен базовым.")
    lines.append("Экспертный выбор обучен одним LambdaRank ranker на историческом качестве экспертов.")
    lines.append("\n=== 3. Лучшие кандидаты continuation ===")
    for s in summaries[:12]:
        lines.append(f"{s['name']}: wCV={s['wcv']:.6f}, wins={s['wins']}/4, latest Δ={s['latest_delta']:+.6f}, worst Δ={s['worst_delta']:+.6f}")
    lines.append("\n=== 4. Error detectors (AUC по следующему временному fold) ===")
    for r in detector_rows:
        if np.isfinite(r.get("auc",np.nan)):
            lines.append(f"{r['fold']} {r['detector']}: AUC={r['auc']:.4f}, train_rate={r['train_positive_rate']:.3f}, val_rate={r['val_positive_rate']:.3f}")
    lines.append("\n=== 5. Expert ranker ===")
    for r in rank_rows: lines.append(json.dumps(r,ensure_ascii=False,default=_json_default))
    lines.append("\n=== 6. Три итоговых сабмита ===")
    for o in outs:
        lines.append(f"{o['rank']}. {o['label']} -> {o['candidate']} | local wCV={o['validation_wcv']:.6f} | wins={o['wins']}/4 | {Path(o['path']).name}")
    lines.append("\nПодробные таблицы: continuation_candidate_summary.csv, continuation_fold_scores.csv, error_detector_auc.csv, expert_ranker_diagnostics.csv, segment_scores.csv, combined_artifact_inventory.csv, FINAL_SELECTION.csv.")
    (new_results/"COMBINED_REPORT_RU.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")


def parse_args():
    ap=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--max-hours",type=float,default=11.5,help="Оставляет запас до 12-часового дедлайна")
    ap.add_argument("--reserve-hours",type=float,default=.75,help="Не тратить этот хвост на новое обучение")
    ap.add_argument("--threads",type=int,default=max(1,min(12,os.cpu_count() or 1)))
    ap.add_argument("--ranker-users",type=int,default=45000,help="Макс class1 users с каждого past fold для LambdaRank")
    ap.add_argument("--previous-work-dir",default=None,help="Явная папка старого _best_bas_...; иначе автоопределение")
    ap.add_argument("--previous-runner",default=None,help="Имя/путь предыдущего run_best_bas*.py")
    ap.add_argument("--new-work-dir",default="_best_bas_continue_12h")
    ap.add_argument("--row-frac",type=float,default=1.0,help="production=1.0; меньше только для отладки")
    ap.add_argument("--preflight-only",action="store_true")
    ap.add_argument("--self-test",action="store_true")
    return ap.parse_args()


def self_test():
    # Core rank relevance ordering and deterministic subset tests without teammate data.
    rng=np.random.default_rng(42);n=2000;E=5
    losses=rng.random((n,E));order=np.argsort(losses,axis=1);rel=np.zeros((n,E),np.int32);rel[np.arange(n)[:,None],order]=np.arange(E-1,-1,-1)[None,:]
    assert np.all(np.argmax(rel,axis=1)==np.argmin(losses,axis=1))
    uid=np.arange(10000,dtype=np.int64);a=deterministic_subset(uid,1234,7);b=deterministic_subset(uid,1234,7)
    assert len(a)==1234 and np.array_equal(np.sort(a),np.sort(b))
    print("CONTINUATION SELF-TEST OK")


def main():
    a=parse_args()
    if a.self_test:
        self_test();return
    base=Path(__file__).resolve().parent
    prev,prev_runner_path=import_previous_runner(base,a.previous_runner)
    package=prev.discover_package(base)
    if hasattr(prev,"ensure_dependencies"):
        prev.ensure_dependencies(package,False)
    raw,sample=prev.discover_raw_and_sample(base,package)
    new_work=(base/a.new_work_dir).resolve();new_results=new_work/"results";new_subs=new_work/"submissions"
    for p in (new_work,new_results,new_subs):p.mkdir(parents=True,exist_ok=True)
    prev_work=discover_previous_work(base,a.previous_work_dir,new_work)
    wall=WallBudget(a.max_hours,a.reserve_hours)

    # Context reads/writes old checkpoints/cache in place, but writes new reports/submissions separately.
    # Keeping ctx.work=prev_work is essential for reuse of the previous processed cache.
    dummy_budget=prev.Budget(started=time.time(),max_hours=a.max_hours,stop_new_hours=max(.1,a.max_hours-a.reserve_hours-1.0))
    ctx=prev.Context(base,package,package/"pipeline",raw,sample,prev_work,new_results,new_subs,prev_work/"checkpoints",dummy_budget)
    prev.configure_pipeline(ctx,a.threads)

    # A previous disk-full event can leave reproducible cache parquets or NPZ
    # checkpoints truncated while still existing.  Repair them BEFORE deciding
    # what is already complete, then make all new feature-cache writes atomic.
    bad_cache = repair_processed_cache(ctx)
    bad_ckpt = repair_corrupt_checkpoints(ctx)
    install_atomic_feature_cache(ctx)

    friend=prev.verify_friend_package(package)

    counts_before=previous_checkpoint_counts(prev,ctx)
    inventory=inventory_previous(prev_work)
    save_csv(new_results/"combined_artifact_inventory.csv",inventory)
    manifest=dict(version=VERSION,started_at=now_iso(),max_hours=a.max_hours,reserve_hours=a.reserve_hours,
                  previous_runner=str(prev_runner_path),previous_work=str(prev_work),package=str(package),raw=str(raw),sample=str(sample),
                  previous_checkpoint_counts=counts_before,friend_rebuild_max_log_error=friend.get("max_log_error"),
                  repaired_truncated_cache_files=bad_cache,repaired_corrupt_checkpoints=bad_ckpt)
    atomic_json(new_results/"CONTINUATION_MANIFEST.json",manifest)
    log("Reuse workdir:",prev_work)
    log("Core ready:",counts_before["core_ready"],"seq42 complete:",counts_before["seq42_complete"],"etx42 complete:",counts_before["etx42_complete"])
    log("STRONGEST_CURRENT rebuild max log error:",friend.get("max_log_error"))
    if a.preflight_only:
        log("PREFLIGHT ONLY: no training started")
        return

    runtime_rows=[];detector_rows=[];rank_rows=[]
    # Finish ONLY missing table infrastructure. No new neural training in emergency continuation.
    ensure_core(prev,ctx,wall,a.row_frac,runtime_rows)
    counts_after=previous_checkpoint_counts(prev,ctx)
    neural_complete={"seq42":counts_after["seq42_complete"],"etx42":counts_after["etx42_complete"]}
    log("Using existing neural OOF only:",neural_complete)

    bank=build_core_bank(prev,ctx,neural_complete)
    # No Phase14 q/meta/pairwise rerun. Full hurdle is only the occurrence/class split infrastructure.
    for f in FOLDS:
        rec=bank[f]
        rec["hurdle_temporal"]=np.asarray(rec["hurdle"],dtype=np.float64).copy()
        rec["p_hurdle"]=np.asarray(rec["p_hurdle"],dtype=np.float64)
        rec["mu"]=np.asarray(rec["mu"],dtype=np.float64)

    t=time.time();fit_overtime_error_detectors(prev,bank,a.threads,detector_rows);runtime_rows.append(dict(stage="error_detectors",hours=(time.time()-t)/3600));save_csv(new_results/"error_detector_auc.csv",detector_rows)
    t=time.time();fit_all_overtime_specialists(prev,bank,a.threads);runtime_rows.append(dict(stage="class1_specialists",hours=(time.time()-t)/3600))
    experts=expert_names(bank,neural_complete);log("Ranker experts:",experts)
    if not wall.can_start(.8,extra_reserve=2.0):
        raise RuntimeError(f"Недостаточно безопасного времени перед ranker/final: remaining={wall.remaining:.2f}h")
    t=time.time();add_ranker_candidates(prev,bank,experts,a.threads,a.ranker_users,rank_rows);runtime_rows.append(dict(stage="expert_ranker_oof",hours=(time.time()-t)/3600));save_csv(new_results/"expert_ranker_diagnostics.csv",rank_rows)

    summaries,fold_rows=evaluate(prev,bank);save_csv(new_results/"continuation_candidate_summary.csv",summaries);save_csv(new_results/"continuation_fold_scores.csv",fold_rows)
    seg=segment_report(prev,bank,[s["name"] for s in summaries]);save_csv(new_results/"segment_scores.csv",seg)
    log("Top continuation candidates:")
    for s in summaries[:10]:log(f" {s['name']:<26} wCV={s['wcv']:.6f} wins={s['wins']}/4 latest={s['latest_delta']:+.6f} worst={s['worst_delta']:+.6f}")

    # Reserve final deployable stage. build_test_record reuses hurdle_test.npz if already saved.
    if not wall.can_start(1.2,extra_reserve=.25):
        raise RuntimeError(f"До дедлайна осталось слишком мало безопасного времени для final refit: {wall.remaining:.2f}h")
    t=time.time();test=build_test_record_core(prev,ctx,friend,neural_complete);runtime_rows.append(dict(stage="build_test_record",hours=(time.time()-t)/3600))
    # Keep exact friend as production baseline; structural proxy remains a ranker feature.
    fit_final_error_detectors(prev,bank,test,a.threads)
    fit_final_specialists(prev,bank,test,a.threads)
    fit_final_ranker(prev,bank,test,experts,a.threads,a.ranker_users,rank_rows)
    save_csv(new_results/"expert_ranker_diagnostics.csv",rank_rows)

    selected=select_three(prev,summaries,test);outs=write_submissions(prev,ctx,selected,test);save_csv(new_results/"FINAL_SELECTION.csv",selected);save_csv(new_results/"runtime_continuation.csv",runtime_rows)
    div=[];names=[o["candidate"] for o in outs]+["friend"]
    for i,n1 in enumerate(names):
        for n2 in names[i+1:]:div.append(dict(a=n1,b=n2,**diversity(prev,test[n1],test[n2])))
    save_csv(new_results/"final_diversity.csv",div)

    manifest.update(dict(finished_at=now_iso(),runtime_hours=wall.elapsed,remaining_hours=wall.remaining,
                         checkpoint_counts_after=counts_after,neural_complete=neural_complete,ranker_experts=experts,
                         top_candidates=summaries[:12],selected_submissions=outs,
                         hard_limit_respected=wall.elapsed<a.max_hours))
    atomic_json(new_results/"CONTINUATION_MANIFEST.json",manifest)
    combine_previous_report(prev_work,new_results,manifest,summaries,detector_rows,rank_rows,seg,outs)
    log("DONE in",f"{wall.elapsed:.2f}h; remaining budget {wall.remaining:.2f}h")
    for o in outs:log("SUBMISSION",o["rank"],o["path"])
    log("COMMON REPORT:",new_results/"COMBINED_REPORT_RU.txt")


if __name__=="__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Old checkpoints were not deleted; continuation can be restarted.",file=sys.stderr)
        raise
    except Exception:
        traceback.print_exc()
        raise
