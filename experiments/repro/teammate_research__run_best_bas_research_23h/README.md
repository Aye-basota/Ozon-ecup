# !/usr/bin/env python3

## Catalogue metadata

- **Catalogue ID:** `teammate_research__run_best_bas_research_23h`
- **Namespace:** `teammate_research`
- **Experiment ID:** `run_best_bas_research_23h`
- **Original source:** `пайплайн сокомандника/research_scripts/run_best_bas_research_23h.py`
- **Source ref:** `working tree`
- **Source commit:** `cdf74c77108e3b731f9ecb4f4e8f7b198cbded66`
- **Kind:** teammate research runner
- **Model:** LightGBM, dilated TCN, sequence model, two-part / hurdle, blend
- **Features:** recency, occurrence features, gap/burst features, history-depth features
- **Preprocessing:** See preserved experiment card and frozen implementation
- **Validation:** about *new* signal.  It reuses the teammate's exact validation protocol, then
- **Known score:** 3) temporal occurrence and RMSLE-effective-q models.  q*=clip(z_true/mu,0,1)
- **Seed:** "Global blend validation uses the one-seed structural proxy; final blend retains the stronger exact friend AVG3 base.",
- **Postprocessing:** "All final candidates are shifted to teammate-proven level 2.3293 before expm1.",
- **Submission:** path=ctx.submissions/f"submission_best_bas_{k}_{lab.lower()}_{s['name']}.csv"
- **External data/artifacts:** Competition train.parquet and sample_submit.csv; additional artifacts are listed in the card
- **Reproducibility:** FULL when the inputs/checkpoints named by the preserved runner are available

## Reproduction

Run `python run.py` to inspect recovered commands and provenance. Use `python run.py --execute N` only after preparing the data/artifacts listed below.

## Preserved original documentation
# run_best_bas_research_23h

Original script: `пайплайн сокомандника/research_scripts/run_best_bas_research_23h.py`

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E-CUP 2026 / Track 3 — research runner for teammate STRONGEST_CURRENT.

Place this file in src/DL/best_bas (or any directory containing either the
unpacked submission_STRONGEST_CURRENT package or its ZIP) and run:

    python run_best_bas_research_23h.py

The runner is deliberately conservative about temporal leakage and aggressive
about *new* signal.  It reuses the teammate's exact validation protocol, then
checks four complementary families:

1) faithful table controls (CAP / UNC / DIST) and a new regularized two-part
   hurdle model P(y>0) * E[log1p(y)|y>0];
2) one-seed OOF replicas of ETX and TCN (only when CUDA + budget allow) so the
   meta models can see sequence/tabular disagreement without pretending that
   the archived test predictions are OOF;
3) temporal occurrence and RMSLE-effective-q models.  q*=clip(z_true/mu,0,1)
   is fitted with mu^2-aware weights, because in z-space the squared error of
   q*mu is mu^2*(q-q*)^2;
4) pairwise competence / learning-to-defer routing: for every expert a
   classifier estimates whether it beats the current proxy on squared z-error;
   only the most confident rows are softly moved toward a better expert.

Historical meta models NEVER use labels from the target fold or a later fold.
For fold i they are trained only on folds < i.  The first fold falls back to the
base expert.  For the future 2026-02-13 test, all four mature folds are allowed.

At the end the script writes THREE deliberately different submissions, not the
three numerically closest local winners.  Selection jointly considers:
  * calibrated 1:2:4:8 wCV;
  * number of fold wins and latest-fold sign;
  * stability of deltas;
  * distance from STRONGEST_CURRENT and from already selected submissions.

The exact teammate production submission is never emitted as one of the three.

Important teammate invariants kept intact:
  * no validation cutoff after 2025-10-16;
  * train target T+30 <= validation cutoff;
  * train panel = 1 block, validation/test panel = 3 blocks;
  * depth-clip 289 for production-like sequence inputs;
  * all model comparison is in z=log1p(pred), with per-fold optimal log shift;
  * test candidates are put on the same proven level 2.3293.

Runtime:
  default hard budget requested by the user is 23 h.  The runner stops starting
  optional expensive stages well before the wall-clock limit and always reserves
  time for final refit, reports and three valid CSV files.  Checkpoints make a
  restarted run resume instead of repeating completed folds.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import gc
import hashlib
import importlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
import warnings
import zipfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np

warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
warnings.filterwarnings("ignore", category=FutureWarning)

SCRIPT_VERSION = "best_bas_research_23h_v1_2026-08-21"
FOLDS = ("2025-09-04", "2025-09-18", "2025-10-02", "2025-10-16")
FOLD_WEIGHTS = np.asarray([1.0, 2.0, 4.0, 8.0], dtype=np.float64)
LEVEL = 2.3293
KNOWN_FRIEND_WCV = 1.74751
EPS = 1e-7

# Teammate's exact production recipe.
FRIEND_COMPONENTS = (
    ("S1-CAP", 0.100),
    ("S1-UNC", 0.200),
    ("S1-DIST", 0.250),
    ("SEQ-01", 0.075),
    ("SEQ-C289-S43", 0.075),
    ("SEQ-C289-S44", 0.075),
    ("ETX-01-S42-DCW", 0.075),
    ("ETX-01-S43-DCW", 0.075),
    ("ETX-01-S44-DCW", 0.075),
)

# Structurally matched one-seed OOF proxy.  Test counterpart uses the teammate's
# archived seed-42 production arrays, while the exact friend base still uses AVG3.
PROXY_COMPONENT_WEIGHTS = {
    "cap": 0.100,
    "unc": 0.200,
    "dist": 0.250,
    "seq42": 0.225,
    "etx42": 0.225,
}


# =============================================================================
# generic helpers
# =============================================================================

def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def log(*parts: Any) -> None:
    print(f"[{now_iso()}]", *parts, flush=True)


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    tmp.replace(path)


def _json_default(x: Any):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dt.date):
        return x.isoformat()
    raise TypeError(type(x).__name__)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ensure_finite(name: str, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if not np.isfinite(x).all():
        bad = int((~np.isfinite(x)).sum())
        raise FloatingPointError(f"{name}: {bad} NaN/inf")
    return x


def clip_z(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z = np.nan_to_num(z, nan=0.0, posinf=20.0, neginf=0.0)
    return np.clip(z, 0.0, 20.0)


def rmsle_z(y: np.ndarray, z: np.ndarray) -> float:
    ly = np.log1p(np.asarray(y, dtype=np.float64))
    zz = np.maximum(np.asarray(z, dtype=np.float64), 0.0)
    return float(np.sqrt(np.mean((ly - zz) ** 2)))


def calibrate(y: np.ndarray, z: np.ndarray, iters: int = 30) -> tuple[float, float]:
    """Exact fixed-point log shift used by the teammate validator."""
    ly = np.log1p(np.asarray(y, dtype=np.float64))
    zz = np.asarray(z, dtype=np.float64)
    d = float((ly - zz).mean())
    for _ in range(iters):
        active = zz + d > 0
        if not active.any():
            break
        dn = float((ly[active] - zz[active]).mean())
        if abs(dn - d) < 1e-12:
            d = dn
            break
        d = dn
    return d, rmsle_z(y, zz + d)


def weighted_cv(scores: Iterable[float]) -> float:
    a = np.asarray(list(scores), dtype=np.float64)
    if len(a) != 4:
        raise ValueError("wCV requires all four teammate folds")
    return float(np.dot(a, FOLD_WEIGHTS) / FOLD_WEIGHTS.sum())


def level_calibrate_test(z: np.ndarray, level: float = LEVEL) -> tuple[np.ndarray, float]:
    z = clip_z(z)
    d = float(level - z.mean())
    return np.maximum(z + d, 0.0), d


def rank01(x: np.ndarray) -> np.ndarray:
    """Stable [0,1] rank without scipy."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    out = np.empty(len(x), dtype=np.float64)
    if len(x) <= 1:
        out[:] = 1.0
        return out
    out[order] = np.arange(len(x), dtype=np.float64) / (len(x) - 1)
    return out


def fold_index(cutoff: str) -> int:
    return FOLDS.index(str(cutoff))


def date_obj(s: str) -> dt.date:
    return dt.date.fromisoformat(str(s))


@dataclasses.dataclass
class Budget:
    started: float
    max_hours: float
    stop_new_hours: float
    reserve_hours: float = 0.75

    @property
    def elapsed_hours(self) -> float:
        return (time.time() - self.started) / 3600.0

    @property
    def remaining_hours(self) -> float:
        return self.max_hours - self.elapsed_hours

    def can_start(self, estimate_hours: float, mandatory: bool = False) -> bool:
        if self.remaining_hours <= self.reserve_hours:
            return False
        if mandatory:
            return self.remaining_hours > estimate_hours + self.reserve_hours
        if self.elapsed_hours >= self.stop_new_hours:
            return False
        return self.remaining_hours > estimate_hours + self.reserve_hours


@dataclasses.dataclass
class Context:
    base_dir: Path
    package: Path
    pipeline: Path
    raw: Path
    sample: Path
    work: Path
    results: Path
    submissions: Path
    checkpoints: Path
    budget: Budget
    config: Any = None
    train_mod: Any = None
    features_mod: Any = None
    models_mod: Any = None
    data_mod: Any = None


# =============================================================================
# paths / package / dependencies
# =============================================================================

def _looks_like_package(p: Path) -> bool:
    return (p / "pipeline" / "src" / "config.py").exists() and (p / "artifacts" / "predictions").is_dir()


def discover_package(base: Path) -> Path:
    candidates = [
        base / "submission_STRONGEST_CURRENT",
        base,
        base / "best_bas" / "submission_STRONGEST_CURRENT",
    ]
    for p in candidates:
        if _looks_like_package(p):
            return p.resolve()

    zips = [
        base / "submission_STRONGEST_CURRENT_artifacts_2026-08-20.zip",
        *sorted(base.glob("*STRONGEST_CURRENT*.zip")),
    ]
    zips = [p for i, p in enumerate(zips) if p.exists() and p not in zips[:i]]
    if zips:
        out = base / "_best_bas_unpacked"
        out.mkdir(parents=True, exist_ok=True)
        marker = out / ".unzipped.ok"
        if not marker.exists():
            log("Распаковываю архив товарища:", zips[0])
            with zipfile.ZipFile(zips[0], "r") as zf:
                zf.extractall(out)
            marker.write_text(now_iso(), encoding="utf-8")
        for p in [out / "submission_STRONGEST_CURRENT", out]:
            if _looks_like_package(p):
                return p.resolve()
    raise FileNotFoundError(
        "Не найден submission_STRONGEST_CURRENT: положите распакованную папку или ZIP рядом со скриптом")


def discover_raw_and_sample(base: Path, package: Path) -> tuple[Path, Path]:
    roots = []
    for start in (base, package.parent, base.parent):
        p = start.resolve()
        for _ in range(6):
            if p not in roots:
                roots.append(p)
            if p.parent == p:
                break
            p = p.parent

    raw_names = [
        Path("train.parquet"), Path("data/train.parquet"), Path("data/raw/train.parquet"),
        Path("dataset/train.parquet"),
    ]
    sample_names = [
        Path("sample_submit.csv"), Path("sample_submission.csv"),
        Path("data/sample_submit.csv"), Path("data/sample_submission.csv"),
        Path("data/raw/sample_submit.csv"), Path("data/raw/sample_submission.csv"),
    ]
    raw = next((r / n for r in roots for n in raw_names if (r / n).exists()), None)
    sample = next((r / n for r in roots for n in sample_names if (r / n).exists()), None)
    if raw is None:
        raise FileNotFoundError("Не найден train.parquet рядом с best_bas или в родительских data[/raw]")
    if sample is None:
        # The package reference submission has exactly the right user order and is a valid fallback.
        fallback = package / "submission" / "submission_STRONGEST_CURRENT.csv"
        if fallback.exists():
            sample = fallback
            log("sample_submit.csv не найден; использую user_id-порядок reference submission")
        else:
            raise FileNotFoundError("Не найден sample_submit.csv")
    return raw.resolve(), sample.resolve()


def module_available(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def ensure_dependencies(package: Path, no_install: bool) -> None:
    required = {
        "polars": "polars>=1.43,<2",
        "pyarrow": "pyarrow",
        "pandas": "pandas>=2.2,<3",
        "lightgbm": "lightgbm>=4.6,<5",
        "sklearn": "scikit-learn>=1.5",
    }
    missing = [spec for mod, spec in required.items() if not module_available(mod)]
    if not missing:
        return
    if no_install:
        raise RuntimeError("Отсутствуют зависимости: " + ", ".join(missing))
    log("Устанавливаю отсутствующие зависимости:", ", ".join(missing))
    cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", *missing]
    subprocess.check_call(cmd)
    importlib.invalidate_caches()
    still = [mod for mod in required if not module_available(mod)]
    if still:
        raise RuntimeError("После pip всё ещё не импортируются: " + ", ".join(still))


def configure_pipeline(ctx: Context, threads: int) -> None:
    if str(ctx.pipeline) not in sys.path:
        sys.path.insert(0, str(ctx.pipeline))
    C = importlib.import_module("src.config")
    # Patch BEFORE importing data/features/train: those modules copy path constants at import time.
    C.RAW_PARQUET = ctx.raw
    C.SAMPLE_SUBMIT = ctx.sample
    C.DATA_PROCESSED = ctx.work / "cache" / "processed"
    C.ARTIFACTS = ctx.work / "pipeline_artifacts"
    C.EXPERIMENTS = ctx.work / "pipeline_experiments"
    C.SUBMISSIONS = ctx.submissions
    C.LGB_THREADS = int(threads)
    C.LGB_PARAMS = dict(C.LGB_PARAMS, num_threads=int(threads))
    for p in (C.DATA_PROCESSED, C.ARTIFACTS, C.EXPERIMENTS, C.SUBMISSIONS):
        Path(p).mkdir(parents=True, exist_ok=True)

    # Import after patching config paths.
    ctx.config = C
    ctx.data_mod = importlib.import_module("src.data")
    ctx.features_mod = importlib.import_module("src.features")
    ctx.models_mod = importlib.import_module("src.models")
    ctx.train_mod = importlib.import_module("src.train")


def verify_friend_package(package: Path) -> dict[str, Any]:
    pred_dir = package / "artifacts" / "predictions"
    ref_path = package / "submission" / "submission_STRONGEST_CURRENT.csv"
    weights = np.asarray([w for _, w in FRIEND_COMPONENTS], dtype=np.float64)
    assert abs(float(weights.sum()) - 1.0) < 1e-12
    uid_ref = None
    parts = []
    means = {}
    for name, _ in FRIEND_COMPONENTS:
        zp = pred_dir / f"ztest_{name}.npy"
        up = pred_dir / f"uid_{name}.npy"
        if not zp.exists() or not up.exists():
            raise FileNotFoundError(f"В архиве отсутствует production component {name}")
        z = ensure_finite(name, np.load(zp, mmap_mode="r"))
        u = np.load(up, mmap_mode="r")
        if len(z) != 250_000 or len(u) != 250_000:
            raise ValueError(f"{name}: ожидалось 250000 строк")
        if uid_ref is None:
            uid_ref = np.asarray(u).copy()
        elif not np.array_equal(u, uid_ref):
            raise ValueError(f"{name}: другой порядок user_id")
        parts.append(np.asarray(z, dtype=np.float64))
        means[name] = float(np.mean(z))
    mix = np.average(np.vstack(parts), axis=0, weights=weights)
    z_friend, delta = level_calibrate_test(mix)

    # Compare with the packaged CSV using pandas only if available; exact equality is not
    # required here because CSV rounding can move the 7th decimal.
    max_log_error = None
    if ref_path.exists():
        import pandas as pd
        ref = pd.read_csv(ref_path)
        if list(ref.columns)[:2] != ["user_id", "predict"] or len(ref) != 250_000:
            raise ValueError("reference submission has invalid schema")
        idx = {int(u): i for i, u in enumerate(uid_ref)}
        pos = np.fromiter((idx[int(u)] for u in ref.user_id.to_numpy()), dtype=np.int64, count=len(ref))
        p = np.maximum(np.expm1(z_friend[pos]), 0.0)
        max_log_error = float(np.max(np.abs(np.log1p(ref.predict.to_numpy(np.float64)) - np.log1p(p))))
        if max_log_error > 2e-6:
            raise AssertionError(f"Не удалось точно восстановить STRONGEST_CURRENT: max log error={max_log_error}")
    return dict(uid=uid_ref, z=z_friend, raw_mix=mix, delta=delta, means=means,
                max_log_error=max_log_error, ref_sha256=sha256(ref_path) if ref_path.exists() else None)


# =============================================================================
# checkpoints / aligned fold records
# =============================================================================

def fold_ckpt(ctx: Context, model: str, fold: str) -> Path:
    return ctx.checkpoints / "folds" / f"{model}__{fold}.npz"


def save_fold(path: Path, *, user_id: np.ndarray, y: np.ndarray, z: np.ndarray,
              p: np.ndarray | None = None, mu: np.ndarray | None = None,
              meta_raw: np.ndarray | None = None, meta_names: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    obj: dict[str, Any] = {
        "user_id": np.asarray(user_id, dtype=np.int64),
        "y": np.asarray(y, dtype=np.float32),
        "z": np.asarray(z, dtype=np.float32),
    }
    if p is not None:
        obj["p"] = np.asarray(p, dtype=np.float32)
    if mu is not None:
        obj["mu"] = np.asarray(mu, dtype=np.float32)
    if meta_raw is not None:
        obj["meta_raw"] = np.asarray(meta_raw, dtype=np.float32)
    if meta_names is not None:
        obj["meta_names"] = np.asarray(meta_names, dtype="U80")
    np.savez_compressed(path, **obj)


def load_fold(path: Path) -> dict[str, np.ndarray]:
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def validate_fold_record(d: dict[str, np.ndarray], name: str) -> None:
    n = len(d["user_id"])
    if n == 0 or len(d["y"]) != n or len(d["z"]) != n:
        raise ValueError(f"{name}: invalid fold lengths")
    if len(np.unique(d["user_id"])) != n:
        raise ValueError(f"{name}: duplicate user_id")
    ensure_finite(f"{name}.y", d["y"])
    ensure_finite(f"{name}.z", d["z"])
    if "p" in d:
        ensure_finite(f"{name}.p", d["p"])
    if "mu" in d:
        ensure_finite(f"{name}.mu", d["mu"])


def align_to_uid(d: dict[str, np.ndarray], uid: np.ndarray) -> dict[str, np.ndarray]:
    src = np.asarray(d["user_id"], dtype=np.int64)
    uid = np.asarray(uid, dtype=np.int64)
    if np.array_equal(src, uid):
        return d
    order = np.argsort(src)
    ss = src[order]
    pos = np.searchsorted(ss, uid)
    if (pos >= len(ss)).any() or not np.array_equal(ss[pos], uid):
        raise ValueError("user_id sets do not match")
    out = {k: (v[order[pos]] if len(v) == len(src) else v) for k, v in d.items()}
    return out


# =============================================================================
# table models — teammate controls + new two-part model
# =============================================================================

META_SUFFIXES = (
    "days_present", "days_search", "days_buy", "days_cart", "searches", "carts",
    "orders", "gmv", "gmv_max", "lgmv_mean", "lgmv_std", "aov", "gmv_per_day",
    "cart2ord", "srch2cart", "buyday_rate", "presence_rate",
)
META_WINDOWS = (7, 14, 30, 60, 90, 180)
META_OTHER = {
    "rec_any", "rec_search", "rec_cart", "rec_buy", "rec_cat", "weekend_share",
    "gap_mean", "gap_std", "gap_max", "buygap_mean", "buygap_std",
}


def choose_meta_features(feats: list[str], max_n: int = 72) -> list[str]:
    out = []
    for w in META_WINDOWS:
        for suf in META_SUFFIXES:
            name = f"w{w}_{suf}"
            if name in feats:
                out.append(name)
    out += [x for x in sorted(META_OTHER) if x in feats]
    # Stable recency-normalized features are allowed, but no unbounded all_/w365/lifetime/tenure.
    out += [x for x in feats if (x.startswith("rec_over_") or x.startswith("trend_") or x.startswith("dlog_"))]
    seen, unique = set(), []
    for x in out:
        if x not in seen:
            unique.append(x); seen.add(x)
    return unique[:max_n]


def variant_setup(ctx: Context, variant: str, row_frac: float = 1.0):
    T = ctx.train_mod
    if variant == "cap":
        return T.Setup(L=180, min_history=90, step=7, panel_blocks=3, train_blocks=1,
                       model="direct", rounds=600, row_frac=row_frac)
    if variant == "unc":
        return T.Setup(L=0, min_history=90, step=7, panel_blocks=3, train_blocks=1,
                       model="direct", rounds=600, row_frac=row_frac)
    if variant == "dist":
        return T.Setup(L=0, min_history=90, step=7, panel_blocks=3, train_blocks=1,
                       model="dist", rounds=250, norm_long=True, row_frac=row_frac)
    if variant == "hurdle":
        # New signal: two-part on the safer norm-long representation, deliberately
        # more regularized than the teammate's direct 127-leaf default.
        params = dict(learning_rate=0.035, num_leaves=63, min_data_in_leaf=260,
                      feature_fraction=0.78, bagging_fraction=0.88, bagging_freq=1,
                      lambda_l2=14.0, lambda_l1=1.5, max_bin=63)
        return T.Setup(L=0, min_history=90, step=7, panel_blocks=3, train_blocks=1,
                       model="two_part", rounds=420, norm_long=True, params=params,
                       row_frac=row_frac)
    raise KeyError(variant)


def train_table_fold(ctx: Context, variant: str, fold: str, row_frac: float = 1.0) -> dict[str, np.ndarray]:
    path = fold_ckpt(ctx, variant, fold)
    hurdle_snapshots_ready = all(fold_ckpt(ctx, f"hurdle{rr}", fold).exists() for rr in (240, 320))
    if path.exists() and (variant != "hurdle" or hurdle_snapshots_ready):
        d = load_fold(path); validate_fold_record(d, f"{variant}/{fold}")
        log("reuse", path.name)
        return d

    s = variant_setup(ctx, variant, row_frac=row_frac)
    V = date_obj(fold)
    tr_cuts = s.train_cutoffs(V)
    if not tr_cuts:
        raise RuntimeError(f"{variant}/{fold}: empty training cutoffs")
    F = ctx.features_mod
    T = ctx.train_mod
    M = ctx.models_mod

    Xv_frame, yv = T.xy(V, s)
    base_feats = F.feature_names(Xv_frame)
    feats = T.select_features(base_feats, s.drop_groups, s.keep_only)
    meta_names = choose_meta_features(feats) if variant == "cap" else []

    log(f"TABLE {variant}/{fold}: assembling {len(tr_cuts)} cutoffs, {len(feats)} features")
    Xtr, ytr, wtr = T.assemble(tr_cuts, s, feats, V)
    ntr = len(ytr)
    # Construct binned datasets before releasing the multi-GB dense matrix.
    dss = M.make_datasets(s.model, Xtr, ytr, None, s.params)
    Xtr = None
    gc.collect()
    if s.model == "direct":
        model = M.train_direct_ds(dss[0], s.params, s.rounds)
    elif s.model == "dist":
        model = M.train_dist_ds(dss, s.params, s.rounds)
    elif s.model == "two_part":
        model = M.train_two_part_ds(dss, s.params, s.rounds)
    else:
        raise KeyError(s.model)

    Av = F.to_np(Xv_frame, feats)
    p = mu = None
    if s.model == "two_part":
        clf, reg = model
        p = np.clip(clf.predict(Av), EPS, 1.0 - EPS)
        mu = np.maximum(reg.predict(Av), 0.0)
        z = p * mu
    elif s.model == "dist":
        z = M.predict_dist(model, Av)
    else:
        z = model.predict(Av)
    z = clip_z(z)
    meta_raw = F.to_np(Xv_frame, meta_names) if meta_names else None
    uid = Xv_frame["user_id"].to_numpy()
    save_fold(path, user_id=uid, y=yv, z=z, p=p, mu=mu,
              meta_raw=meta_raw, meta_names=meta_names if meta_names else None)
    if variant == "hurdle":
        clf, reg = model
        for rr in (240, 320):
            pr = np.clip(clf.predict(Av, num_iteration=rr), EPS, 1.0-EPS)
            mr = np.maximum(reg.predict(Av, num_iteration=rr), 0.0)
            save_fold(fold_ckpt(ctx, f"hurdle{rr}", fold), user_id=uid, y=yv,
                      z=clip_z(pr*mr), p=pr, mu=mr)
    del model, dss, Av, Xv_frame, ytr, wtr
    T._XY.clear(); gc.collect()
    d = load_fold(path)
    off, sc = calibrate(d["y"], d["z"])
    extra = f" pmean={float(d['p'].mean()):.4f}" if "p" in d else ""
    log(f"TABLE {variant}/{fold}: ntr={ntr:,} nval={len(uid):,} cal={sc:.6f} off={off:+.4f}{extra}")
    return d


def train_table_test_hurdle(ctx: Context) -> dict[str, np.ndarray]:
    path = ctx.checkpoints / "test" / "hurdle_test.npz"
    if path.exists():
        d = load_fold(path)
        required = {"user_id", "z", "p", "mu", "z240", "p240", "mu240", "z320", "p320", "mu320"}
        if required.issubset(d):
            validate_fold_record({"user_id": d["user_id"], "y": np.zeros(len(d["user_id"])), "z": d["z"]}, "hurdle_test")
            return d
        log("Old/incomplete hurdle_test checkpoint detected; rebuilding it.")
        path.unlink(missing_ok=True)
    s = variant_setup(ctx, "hurdle")
    F, T, M, C = ctx.features_mod, ctx.train_mod, ctx.models_mod, ctx.config
    Xt, _ = F.make_xy(C.CUTOFF_TEST, s.L, s.panel_blocks, with_target=False, norm_long=s.norm_long)
    feats = T.select_features(F.feature_names(Xt), s.drop_groups, s.keep_only)
    cuts = s.grid()
    log(f"FINAL HURDLE: {len(cuts)} cutoffs, {len(feats)} features")
    Xtr, ytr, wtr = T.assemble(cuts, s, feats)
    dss = M.make_datasets("two_part", Xtr, ytr, None, s.params)
    Xtr = None; gc.collect()
    clf, reg = M.train_two_part_ds(dss, s.params, s.rounds)
    At = F.to_np(Xt, feats)
    p = np.clip(clf.predict(At), EPS, 1.0 - EPS)
    mu = np.maximum(reg.predict(At), 0.0)
    z = clip_z(p * mu)
    snaps = {}
    for rr in (240, 320):
        pr = np.clip(clf.predict(At, num_iteration=rr), EPS, 1.0-EPS)
        mr = np.maximum(reg.predict(At, num_iteration=rr), 0.0)
        snaps[f"p{rr}"] = pr.astype(np.float32)
        snaps[f"mu{rr}"] = mr.astype(np.float32)
        snaps[f"z{rr}"] = clip_z(pr*mr).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, user_id=Xt["user_id"].to_numpy().astype(np.int64),
                        z=z.astype(np.float32), p=p.astype(np.float32), mu=mu.astype(np.float32), **snaps)
    del clf, reg, dss, At, Xt, ytr, wtr
    T._XY.clear(); gc.collect()
    return load_fold(path)


def build_test_meta_raw(ctx: Context) -> tuple[np.ndarray, np.ndarray, list[str]]:
    path = ctx.checkpoints / "test" / "meta_raw_test.npz"
    if path.exists():
        d = np.load(path, allow_pickle=False)
        return d["user_id"], d["X"], d["names"].astype(str).tolist()
    s = variant_setup(ctx, "cap")
    F, T, C = ctx.features_mod, ctx.train_mod, ctx.config
    Xt, _ = F.make_xy(C.CUTOFF_TEST, s.L, s.panel_blocks, with_target=False, norm_long=False)
    feats = T.select_features(F.feature_names(Xt), s.drop_groups, s.keep_only)
    names = choose_meta_features(feats)
    X = F.to_np(Xt, names)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, user_id=Xt["user_id"].to_numpy().astype(np.int64), X=X.astype(np.float32),
                        names=np.asarray(names, dtype="U80"))
    return Xt["user_id"].to_numpy(), X, names


# =============================================================================
# neural OOF replicas (one seed; archived test arrays are used at final inference)
# =============================================================================

def cuda_info() -> dict[str, Any]:
    try:
        import torch
        if not torch.cuda.is_available():
            return dict(available=False)
        prop = torch.cuda.get_device_properties(0)
        return dict(available=True, name=prop.name, total_gb=float(prop.total_memory / 2**30),
                    torch=torch.__version__)
    except Exception as exc:
        return dict(available=False, error=repr(exc))


def train_seq_fold(ctx: Context, fold: str, seed: int = 42) -> dict[str, np.ndarray]:
    name = f"seq{seed}"
    path = fold_ckpt(ctx, name, fold)
    if path.exists():
        d = load_fold(path); validate_fold_record(d, f"{name}/{fold}"); return d
    seq = importlib.import_module("src.seq")
    seq.build_panel(False)
    cfg = dict(seq.DEFAULT_CFG)
    cfg.update(seed=seed, epochs=4, compile=False)
    info = cuda_info()
    if info.get("available") and info.get("total_gb", 0) < 5.5:
        cfg["batch"] = 512
    V = date_obj(fold)
    log(f"TCN seed={seed} fold={fold} batch={cfg['batch']}")
    uv, z, yv, hist = seq.train_fold(V, cfg, curve=False, n_cutoffs=None, val_frac=1.0, ckpt=None)
    save_fold(path, user_id=uv, y=yv, z=clip_z(z))
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()
    return load_fold(path)


def train_etx_fold(ctx: Context, fold: str, seed: int = 42) -> dict[str, np.ndarray]:
    name = f"etx{seed}"
    path = fold_ckpt(ctx, name, fold)
    if path.exists():
        d = load_fold(path); validate_fold_record(d, f"{name}/{fold}"); return d
    seq = importlib.import_module("src.seq")
    seq.build_panel(False)
    etx = importlib.import_module("src.etx")
    etx.build_events(False)
    cfg = dict(etx.DEFAULT_CFG)
    cfg.update(seed=seed, epochs=4, compile=False)
    info = cuda_info()
    gb = float(info.get("total_gb", 0) or 0)
    if gb and gb < 6.0:
        cfg["batch"] = 256
    elif gb and gb < 8.0:
        cfg["batch"] = 384
    V = date_obj(fold)
    log(f"ETX seed={seed} fold={fold} batch={cfg['batch']}")
    try:
        uv, z, yv, hist, model, c = etx.train_fold(V, cfg, curve=False, n_cutoffs=None, val_frac=1.0, ckpt=None)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower() or cfg["batch"] <= 192:
            raise
        log("ETX CUDA OOM -> повтор с меньшим batch")
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        cfg["batch"] = max(192, cfg["batch"] // 2)
        uv, z, yv, hist, model, c = etx.train_fold(V, cfg, curve=False, n_cutoffs=None, val_frac=1.0, ckpt=None)
    save_fold(path, user_id=uv, y=yv, z=clip_z(z))
    del model
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()
    return load_fold(path)


def run_complete_neural_family(ctx: Context, family: str, budget: Budget) -> bool:
    if not cuda_info().get("available"):
        log(f"{family}: CUDA отсутствует, пропускаю дорогой OOF; табличная/meta ветка продолжится")
        return False
    fn = train_etx_fold if family == "etx42" else train_seq_fold
    done_times: list[float] = []
    for fold in FOLDS:
        p = fold_ckpt(ctx, family, fold)
        if p.exists():
            continue
        est = max(0.45, 1.45 * float(np.median(done_times))) if done_times else 0.9
        # We only keep a neural family if all four folds can plausibly finish.
        remaining_folds = sum(not fold_ckpt(ctx, family, f).exists() for f in FOLDS)
        total_est = est * remaining_folds
        if not budget.can_start(total_est, mandatory=False):
            log(f"{family}: недостаточно безопасного бюджета для полного 4-fold банка; пропуск")
            return False
        t0 = time.time()
        try:
            fn(ctx, fold, 42)
        except Exception as exc:
            log(f"{family}/{fold} FAILED:", repr(exc))
            append_error(ctx, "neural", family, fold, exc)
            return False
        done_times.append((time.time() - t0) / 3600.0)
    return all(fold_ckpt(ctx, family, f).exists() for f in FOLDS)


# =============================================================================
# OOF bank and meta features
# =============================================================================

def load_component_test(package: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    p = package / "artifacts" / "predictions"
    return np.load(p / f"uid_{name}.npy"), np.load(p / f"ztest_{name}.npy")


def build_fold_bank(ctx: Context, neural_complete: dict[str, bool]) -> dict[str, dict[str, Any]]:
    bank: dict[str, dict[str, Any]] = {}
    for fold in FOLDS:
        cap = load_fold(fold_ckpt(ctx, "cap", fold)); validate_fold_record(cap, f"cap/{fold}")
        uid = cap["user_id"].astype(np.int64)
        y = cap["y"].astype(np.float64)
        rec: dict[str, Any] = {
            "user_id": uid, "y": y,
            "cap": cap["z"].astype(np.float64),
            "meta_raw": cap["meta_raw"].astype(np.float32),
            "meta_names": cap["meta_names"].astype(str).tolist(),
        }
        for name in ("unc", "dist", "hurdle"):
            d = align_to_uid(load_fold(fold_ckpt(ctx, name, fold)), uid)
            if not np.allclose(d["y"], y, atol=1e-5, rtol=1e-6):
                raise AssertionError(f"target mismatch {name}/{fold}")
            rec[name] = d["z"].astype(np.float64)
            if name == "hurdle":
                rec["p_hurdle"] = d["p"].astype(np.float64)
                rec["mu"] = d["mu"].astype(np.float64)
        for rr in (240, 320):
            hd = align_to_uid(load_fold(fold_ckpt(ctx, f"hurdle{rr}", fold)), uid)
            rec[f"hurdle{rr}"] = hd["z"].astype(np.float64)
            rec[f"p_hurdle{rr}"] = hd["p"].astype(np.float64)
            rec[f"mu{rr}"] = hd["mu"].astype(np.float64)
        for fam in ("seq42", "etx42"):
            if neural_complete.get(fam):
                d = align_to_uid(load_fold(fold_ckpt(ctx, fam, fold)), uid)
                if not np.allclose(d["y"], y, atol=1e-5, rtol=1e-6):
                    raise AssertionError(f"target mismatch {fam}/{fold}")
                rec[fam] = d["z"].astype(np.float64)

        # Structurally matched proxy. Missing optional neural families are replaced by DIST
        # and weights are kept, so the proxy stays on the same scale and remains deterministic.
        parts = []
        for n, w in PROXY_COMPONENT_WEIGHTS.items():
            parts.append(w * rec.get(n, rec["dist"]))
        rec["proxy"] = clip_z(np.sum(parts, axis=0))
        bank[fold] = rec
    return bank


def select_temporal_hurdle(bank: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Choose 240/320/420 rounds only from previous folds; future uses all four."""
    variants = (("hurdle240", "p_hurdle240", "mu240"),
                ("hurdle320", "p_hurdle320", "mu320"),
                ("hurdle", "p_hurdle", "mu"))
    chosen = {}
    for i, fold in enumerate(FOLDS):
        if i == 0:
            best = variants[1]  # conservative middle snapshot before any history exists
        else:
            scored = []
            for v in variants:
                sc=[]
                for j in range(i):
                    f=FOLDS[j]; _,ss=calibrate(bank[f]["y"],bank[f][v[0]]); sc.append(ss)
                ww=FOLD_WEIGHTS[:i]; val=float(np.dot(np.asarray(sc),ww)/ww.sum())
                scored.append((val,v))
            best=min(scored,key=lambda x:x[0])[1]
        rec=bank[fold]; rec["hurdle_temporal"]=rec[best[0]].copy(); rec["p_hurdle"]=rec[best[1]].copy(); rec["mu"]=rec[best[2]].copy()
        chosen[fold]=best[0]
    return chosen


def future_hurdle_choice(bank: dict[str, dict[str, Any]]) -> str:
    choices=("hurdle240","hurdle320","hurdle")
    vals=[]
    for n in choices:
        sc=[]
        for f in FOLDS:
            _,ss=calibrate(bank[f]["y"],bank[f][n]);sc.append(ss)
        vals.append((weighted_cv(sc),n))
    return min(vals)[1]


def meta_matrix(rec: dict[str, Any], candidate_names: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    raw = np.asarray(rec["meta_raw"], dtype=np.float32)
    raw_names = [f"raw__{x}" for x in rec["meta_names"]]
    cols = [raw]
    names = list(raw_names)
    basic = ["cap", "unc", "dist", "hurdle", "hurdle_temporal", "proxy", "seq42", "etx42"]
    for n in basic:
        if n in rec:
            cols.append(np.asarray(rec[n], np.float32)[:, None]); names.append(f"z__{n}")
    p = np.asarray(rec["p_hurdle"], np.float32)
    mu = np.asarray(rec["mu"], np.float32)
    extras = np.column_stack([
        p, mu, 4.0 * p * (1.0 - p), np.log1p(np.maximum(mu, 0)),
    ]).astype(np.float32)
    cols.append(extras); names += ["p_hurdle", "mu", "p_uncertainty", "log1p_mu"]

    expert_z = [np.asarray(rec[n], np.float32) for n in ("cap", "unc", "dist", "hurdle", "hurdle_temporal", "seq42", "etx42") if n in rec]
    if expert_z:
        A = np.column_stack(expert_z)
        dis = np.column_stack([A.mean(1), A.std(1), A.max(1)-A.min(1), np.median(A, axis=1)]).astype(np.float32)
        cols.append(dis); names += ["expert_mean", "expert_std", "expert_range", "expert_median"]
    if candidate_names:
        for n in candidate_names:
            if n in rec:
                cols.append(np.asarray(rec[n], np.float32)[:, None]); names.append(f"candidate__{n}")
    X = np.column_stack(cols).astype(np.float32)
    # Keep missingness from raw features; replace only infinities.
    X[np.isposinf(X)] = np.nan
    X[np.isneginf(X)] = np.nan
    return X, names


def _make_meta_clf(seed: int):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=360, learning_rate=0.025, num_leaves=31, max_depth=7,
                          min_child_samples=220, subsample=0.9, subsample_freq=1,
                          colsample_bytree=0.82, reg_alpha=3.0, reg_lambda=20.0,
                          random_state=seed, n_jobs=max(1, min(12, os.cpu_count() or 1)), verbosity=-1)


def _make_meta_reg(seed: int):
    from lightgbm import LGBMRegressor
    return LGBMRegressor(n_estimators=400, learning_rate=0.025, num_leaves=31, max_depth=7,
                         min_child_samples=220, subsample=0.9, subsample_freq=1,
                         colsample_bytree=0.82, reg_alpha=3.0, reg_lambda=20.0,
                         objective="huber", random_state=seed,
                         n_jobs=max(1, min(12, os.cpu_count() or 1)), verbosity=-1)


def concat_history(bank: dict[str, dict[str, Any]], target_idx: int,
                   candidate_names: list[str] | None = None):
    Xs, ys, ps, mus, folds = [], [], [], [], []
    names = None
    for j in range(target_idx):
        f = FOLDS[j]; rec = bank[f]
        X, nm = meta_matrix(rec, candidate_names)
        if names is None:
            names = nm
        elif nm != names:
            raise AssertionError("meta feature schema changed between folds")
        Xs.append(X); ys.append(rec["y"]); ps.append(rec["p_hurdle"]); mus.append(rec["mu"])
        folds.append(np.full(len(rec["y"]), j, dtype=np.int8))
    if not Xs:
        return None
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(ps), np.concatenate(mus), np.concatenate(folds), names


def concat_history_range(bank: dict[str, dict[str, Any]], start_idx: int, end_idx: int,
                         candidate_names: list[str] | None = None):
    """Concatenate a contiguous PAST-only fold window [start_idx, end_idx)."""
    Xs, ys, ps, mus, folds = [], [], [], [], []
    names = None
    for j in range(max(0, start_idx), min(end_idx, len(FOLDS))):
        f = FOLDS[j]; rec = bank[f]
        X, nm = meta_matrix(rec, candidate_names)
        if names is None:
            names = nm
        elif nm != names:
            raise AssertionError("meta feature schema changed between folds")
        Xs.append(X); ys.append(rec["y"]); ps.append(rec["p_hurdle"]); mus.append(rec["mu"])
        folds.append(np.full(len(rec["y"]), j, dtype=np.int8))
    if not Xs:
        return None
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(ps), np.concatenate(mus), np.concatenate(folds), names


def train_temporal_meta(bank: dict[str, dict[str, Any]], seed: int = 4200) -> None:
    """Add occurrence/q candidates to every historical fold using PREVIOUS folds only."""
    for i, fold in enumerate(FOLDS):
        rec = bank[fold]
        Xtest, _ = meta_matrix(rec)
        if i == 0:
            # No mature previous primary fold: strict temporal fallback.
            rec["occ_meta"] = rec["hurdle_temporal"].copy()
            rec["occ_meta50"] = rec["hurdle_temporal"].copy()
            rec["q_global"] = rec["hurdle_temporal"].copy()
            rec["q_pband"] = rec["hurdle_temporal"].copy()
            rec["occ_recent2"] = rec["hurdle_temporal"].copy()
            rec["occ_recent2_50"] = rec["hurdle_temporal"].copy()
            rec["q_recent2"] = rec["hurdle_temporal"].copy()
            continue
        hist = concat_history(bank, i)
        assert hist is not None
        Xtr, ytr, ptr, mutr, _, _ = hist
        ybin = (ytr > 0).astype(np.int8)
        clf = _make_meta_clf(seed + i)
        clf.fit(Xtr, ybin)
        pmeta = np.clip(clf.predict_proba(Xtest)[:, 1], EPS, 1.0-EPS)
        rec["p_occ_meta"] = pmeta
        rec["occ_meta"] = clip_z(pmeta * rec["mu"])
        p50 = 0.5 * pmeta + 0.5 * rec["p_hurdle"]
        rec["occ_meta50"] = clip_z(p50 * rec["mu"])
        del clf

        # RMSLE-effective q.  For z=q*mu, squared z-risk is mu^2*(q-q*)^2.
        q = np.divide(np.log1p(ytr), np.maximum(mutr, 1e-4))
        q = np.clip(q, 0.0, 1.0)
        wq = np.maximum(mutr, 0.05) ** 2
        cap = float(np.quantile(wq, 0.97))
        wq = np.clip(wq, 0.05, max(cap, 0.05))
        wq = wq / max(float(np.mean(wq)), EPS)
        qreg = _make_meta_reg(seed + 100 + i)
        qreg.fit(Xtr, q, sample_weight=wq)
        qhat = np.clip(qreg.predict(Xtest), 0.0, 1.0)
        rec["q_global_score"] = qhat
        rec["q_global"] = clip_z(qhat * rec["mu"])

        # Local experts in broad p-bands; global q is a fallback for small cells.
        qlocal = qhat.copy()
        bands = np.linspace(0.0, 1.0, 6)
        for b, (lo, hi) in enumerate(zip(bands[:-1], bands[1:])):
            mtr = (ptr >= lo) & ((ptr < hi) if hi < 1.0 else (ptr <= hi))
            mt = (rec["p_hurdle"] >= lo) & ((rec["p_hurdle"] < hi) if hi < 1.0 else (rec["p_hurdle"] <= hi))
            if mtr.sum() < 8000 or mt.sum() == 0:
                continue
            local = _make_meta_reg(seed + 200 + 10*i + b)
            local.set_params(n_estimators=280, num_leaves=23, max_depth=6, min_child_samples=180)
            local.fit(Xtr[mtr], q[mtr], sample_weight=wq[mtr])
            qlocal[mt] = np.clip(local.predict(Xtest[mt]), 0.0, 1.0)
            del local
        rec["q_pband_score"] = qlocal
        rec["q_pband"] = clip_z(qlocal * rec["mu"])

        # Adaptive rolling-window counterpart: only the latest two mature folds.
        # This is deliberately cheap (one occurrence + one q model) and tests whether
        # recent regime information transfers better than pooling all previous folds.
        rh = concat_history_range(bank, max(0, i-2), i)
        if rh is not None:
            Xr, yr, pr, mur, _, _ = rh
            cr = _make_meta_clf(seed + 500 + i); cr.set_params(n_estimators=320, num_leaves=27, max_depth=6)
            cr.fit(Xr, (yr > 0).astype(np.int8))
            pr2 = np.clip(cr.predict_proba(Xtest)[:, 1], EPS, 1.0-EPS)
            rec["occ_recent2"] = clip_z(pr2 * rec["mu"])
            rec["occ_recent2_50"] = clip_z((0.5*pr2 + 0.5*rec["p_hurdle"]) * rec["mu"])
            qr = np.clip(np.log1p(yr) / np.maximum(mur, 1e-4), 0.0, 1.0)
            wr = np.maximum(mur, 0.05) ** 2; cwr=float(np.quantile(wr,0.97)); wr=np.clip(wr,0.05,max(cwr,0.05)); wr/=max(float(wr.mean()),EPS)
            rr = _make_meta_reg(seed + 600 + i); rr.set_params(n_estimators=340, num_leaves=27, max_depth=6)
            rr.fit(Xr, qr, sample_weight=wr)
            q2 = np.clip(rr.predict(Xtest), 0.0, 1.0)
            rec["q_recent2_score"] = q2; rec["q_recent2"] = clip_z(q2 * rec["mu"])
            del cr, rr, Xr, yr, pr, mur
        else:
            rec["occ_recent2"] = rec["occ_meta"].copy(); rec["occ_recent2_50"] = rec["occ_meta50"].copy(); rec["q_recent2"] = rec["q_global"].copy()

        del qreg, Xtr, ytr, ptr, mutr, Xtest
        gc.collect()


# =============================================================================
# pairwise competence router
# =============================================================================

ROUTER_EXPERTS = ("hurdle_temporal", "hurdle", "occ_meta50", "occ_recent2_50",
                  "q_global", "q_pband", "q_recent2", "dist", "seq42", "etx42")


def add_pairwise_router(bank: dict[str, dict[str, Any]], seed: int = 6100) -> None:
    available = [n for n in ROUTER_EXPERTS if all(n in bank[f] for f in FOLDS)]
    for i, fold in enumerate(FOLDS):
        rec = bank[fold]
        if i == 0 or not available:
            rec["router20"] = rec["proxy"].copy()
            rec["router35"] = rec["proxy"].copy()
            rec["router_ranksoft"] = rec["proxy"].copy()
            continue
        hist = concat_history(bank, i, candidate_names=available)
        assert hist is not None
        Xtr, ytr, _, _, _, _ = hist
        Xtest, _ = meta_matrix(rec, candidate_names=available)
        # Candidate arrays for the same history order.
        lytr = np.log1p(ytr)
        base_hist = np.concatenate([bank[FOLDS[j]]["proxy"] for j in range(i)])
        base_loss = (base_hist - lytr) ** 2
        prob_cols, expert_names = [], []
        for k, expert in enumerate(available):
            ez = np.concatenate([bank[FOLDS[j]][expert] for j in range(i)])
            gain = base_loss - (ez - lytr) ** 2
            # A small positive margin filters wins that are numerical noise only.
            margin = 0.0025
            lab = (gain > margin).astype(np.int8)
            rate = float(lab.mean())
            if rate < 0.03 or rate > 0.97:
                continue
            clf = _make_meta_clf(seed + 100*i + k)
            clf.set_params(n_estimators=280, num_leaves=23, max_depth=6)
            clf.fit(Xtr, lab)
            prob_cols.append(clf.predict_proba(Xtest)[:, 1])
            expert_names.append(expert)
            del clf
        if not prob_cols:
            rec["router20"] = rec["proxy"].copy(); rec["router35"] = rec["proxy"].copy(); rec["router_ranksoft"] = rec["proxy"].copy()
            continue
        P = np.column_stack(prob_cols)
        best_k = P.argmax(1)
        best_p = P[np.arange(len(P)), best_k]
        E = np.column_stack([rec[n] for n in expert_names])
        chosen = E[np.arange(len(E)), best_k]
        for frac, name in ((0.20, "router20"), (0.35, "router35")):
            threshold = float(np.quantile(best_p, 1.0-frac))
            active = (best_p >= threshold) & (best_p >= 0.52)
            out = rec["proxy"].copy()
            out[active] = 0.5 * rec["proxy"][active] + 0.5 * chosen[active]
            rec[name] = clip_z(out)
        # Rank-stable uncertainty gate: top 40% gets a smoothly increasing max 0.65 move.
        r = rank01(best_p)
        alpha = np.clip((r - 0.60) / 0.40, 0.0, 1.0) * 0.65
        alpha *= (best_p >= 0.50)
        rec["router_ranksoft"] = clip_z((1-alpha)*rec["proxy"] + alpha*chosen)
        del Xtr, Xtest, P, E
        gc.collect()


# =============================================================================
# scoring and candidate blend grid
# =============================================================================

def score_bank_candidate(bank: dict[str, dict[str, Any]], name: str, base: str = "proxy") -> dict[str, Any]:
    fold_scores, raw_scores, offsets, deltas = [], [], [], []
    rows = []
    for fold in FOLDS:
        r = bank[fold]; y = r["y"]; z = r[name]
        off, sc = calibrate(y, z)
        raw = rmsle_z(y, z)
        _, bsc = calibrate(y, r[base])
        delta = sc - bsc
        fold_scores.append(sc); raw_scores.append(raw); offsets.append(off); deltas.append(delta)
        rows.append(dict(candidate=name, fold=fold, rmsle=raw, rmsle_cal=sc, offset=off,
                         base_cal=bsc, delta_vs_proxy=delta))
    return dict(name=name, wcv=weighted_cv(fold_scores), fold_cal=fold_scores, fold_raw=raw_scores,
                offsets=offsets, deltas=deltas, wins=int(sum(d < 0 for d in deltas)),
                latest_delta=float(deltas[-1]), worst_delta=float(max(deltas)),
                std_delta=float(np.std(deltas)), rows=rows)


def add_global_blends(bank: dict[str, dict[str, Any]], source_names: list[str]) -> list[str]:
    made = []
    for src in source_names:
        for a in (0.25, 0.40, 0.55, 0.70):
            name = f"blend_{src}_a{int(round(100*a)):02d}"
            for fold in FOLDS:
                r = bank[fold]
                r[name] = clip_z((1-a)*r["proxy"] + a*r[src])
            made.append(name)
    return made


def candidate_family(name: str) -> str:
    root = name
    if root.startswith("blend_"):
        root = root[len("blend_"):].split("_a")[0]
    if root.startswith("router"):
        return "router"
    if root.startswith("q_"):
        return "q"
    if root.startswith("occ_") or root.startswith("hurdle"):
        return "occurrence"
    if root in ("seq42", "etx42"):
        return "sequence"
    return "other"


def evaluate_candidates(bank: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base_names = ["proxy", "hurdle240", "hurdle320", "hurdle", "hurdle_temporal",
                  "occ_meta", "occ_meta50", "occ_recent2", "occ_recent2_50",
                  "q_global", "q_pband", "q_recent2", "router20", "router35", "router_ranksoft"]
    for n in ("cap", "unc", "dist", "seq42", "etx42"):
        if all(n in bank[f] for f in FOLDS):
            base_names.append(n)
    blend_sources = [n for n in ("hurdle_temporal", "hurdle", "occ_meta50", "occ_recent2_50",
                                      "q_global", "q_pband", "q_recent2", "router20", "router35", "router_ranksoft")
                     if all(n in bank[f] for f in FOLDS)]
    blend_names = add_global_blends(bank, blend_sources)
    names = base_names + blend_names
    summaries, fold_rows = [], []
    for n in names:
        if not all(n in bank[f] for f in FOLDS):
            continue
        s = score_bank_candidate(bank, n)
        s["family"] = candidate_family(n)
        # Conservative status.  The proxy is only a one-seed reconstruction, so absolute
        # wCV is also reported against the teammate's exact known 1.74751.
        s["stable"] = bool(s["wins"] >= 3 and s["latest_delta"] <= 0 and
                           (s["wcv"] - score_bank_candidate(bank, "proxy")["wcv"]) <= 0.00015)
        summaries.append({k: v for k, v in s.items() if k != "rows"})
        fold_rows.extend(s["rows"])
    summaries.sort(key=lambda r: (r["wcv"], r["worst_delta"], r["std_delta"]))
    return summaries, fold_rows


# =============================================================================
# final future refit of meta models
# =============================================================================

def build_test_record(ctx: Context, friend: dict[str, Any], neural_complete: dict[str, bool]) -> dict[str, Any]:
    # Exact production table + structurally matched one-seed neural components from archive.
    uid_friend = friend["uid"].astype(np.int64)
    rec: dict[str, Any] = {"user_id": uid_friend}
    mapping = {"cap": "S1-CAP", "unc": "S1-UNC", "dist": "S1-DIST"}
    if neural_complete.get("seq42"):
        mapping["seq42"] = "SEQ-01"
    if neural_complete.get("etx42"):
        mapping["etx42"] = "ETX-01-S42-DCW"
    for key, comp in mapping.items():
        u, z = load_component_test(ctx.package, comp)
        tmp = {"user_id": u.astype(np.int64), "z": np.asarray(z, dtype=np.float64)}
        tmp = align_to_uid(tmp, uid_friend)
        rec[key] = clip_z(tmp["z"])
    h = align_to_uid(train_table_test_hurdle(ctx), uid_friend)
    rec["hurdle"] = clip_z(h["z"])
    rec["p_hurdle"] = np.clip(h["p"].astype(np.float64), EPS, 1-EPS)
    rec["mu"] = np.maximum(h["mu"].astype(np.float64), 0.0)
    for rr in (240, 320):
        rec[f"hurdle{rr}"] = clip_z(h[f"z{rr}"])
        rec[f"p_hurdle{rr}"] = np.clip(h[f"p{rr}"].astype(np.float64), EPS, 1-EPS)
        rec[f"mu{rr}"] = np.maximum(h[f"mu{rr}"].astype(np.float64), 0.0)
    u_meta, Xraw, names = build_test_meta_raw(ctx)
    md = align_to_uid({"user_id": u_meta.astype(np.int64), "X": Xraw}, uid_friend)
    rec["meta_raw"] = md["X"].astype(np.float32)
    rec["meta_names"] = names
    rec["friend"] = friend["z"].astype(np.float64)
    rec["proxy"] = clip_z(sum(w * rec.get(n, rec["dist"]) for n, w in PROXY_COMPONENT_WEIGHTS.items()))
    return rec


def concat_all_oof(bank: dict[str, dict[str, Any]], candidate_names: list[str] | None = None):
    Xs, ys, ps, mus = [], [], [], []
    names = None
    for f in FOLDS:
        X, nm = meta_matrix(bank[f], candidate_names)
        if names is None: names = nm
        elif names != nm: raise AssertionError("meta schema mismatch")
        Xs.append(X); ys.append(bank[f]["y"]); ps.append(bank[f]["p_hurdle"]); mus.append(bank[f]["mu"])
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(ps), np.concatenate(mus), names


def fit_final_meta(bank: dict[str, dict[str, Any]], test: dict[str, Any], seed: int = 8400) -> None:
    Xtr, ytr, ptr, mutr, _ = concat_all_oof(bank)
    Xt, _ = meta_matrix(test)
    ybin = (ytr > 0).astype(np.int8)
    clf = _make_meta_clf(seed)
    clf.fit(Xtr, ybin)
    pmeta = np.clip(clf.predict_proba(Xt)[:, 1], EPS, 1-EPS)
    test["p_occ_meta"] = pmeta
    test["occ_meta"] = clip_z(pmeta * test["mu"])
    test["occ_meta50"] = clip_z((0.5*pmeta + 0.5*test["p_hurdle"]) * test["mu"])
    del clf

    q = np.clip(np.log1p(ytr) / np.maximum(mutr, 1e-4), 0.0, 1.0)
    wq = np.maximum(mutr, 0.05) ** 2
    cap = float(np.quantile(wq, 0.97)); wq = np.clip(wq, 0.05, max(cap, 0.05)); wq /= max(wq.mean(), EPS)
    reg = _make_meta_reg(seed+1); reg.fit(Xtr, q, sample_weight=wq)
    qg = np.clip(reg.predict(Xt), 0.0, 1.0)
    test["q_global_score"] = qg; test["q_global"] = clip_z(qg * test["mu"])
    ql = qg.copy(); bands = np.linspace(0.0, 1.0, 6)
    for b, (lo, hi) in enumerate(zip(bands[:-1], bands[1:])):
        mtr = (ptr >= lo) & ((ptr < hi) if hi < 1 else (ptr <= hi))
        mt = (test["p_hurdle"] >= lo) & ((test["p_hurdle"] < hi) if hi < 1 else (test["p_hurdle"] <= hi))
        if mtr.sum() < 8000 or mt.sum() == 0: continue
        lr = _make_meta_reg(seed+10+b); lr.set_params(n_estimators=300, num_leaves=23, max_depth=6, min_child_samples=180)
        lr.fit(Xtr[mtr], q[mtr], sample_weight=wq[mtr]); ql[mt] = np.clip(lr.predict(Xt[mt]), 0.0, 1.0); del lr
    test["q_pband_score"] = ql; test["q_pband"] = clip_z(ql * test["mu"])

    # Future rolling-window models use only the latest two mature validation regimes.
    rh = concat_history_range(bank, len(FOLDS)-2, len(FOLDS))
    if rh is not None:
        Xr, yr, pr, mur, _, _ = rh
        cr = _make_meta_clf(seed+50); cr.set_params(n_estimators=320, num_leaves=27, max_depth=6)
        cr.fit(Xr, (yr>0).astype(np.int8)); pr2=np.clip(cr.predict_proba(Xt)[:,1],EPS,1-EPS)
        test["occ_recent2"] = clip_z(pr2*test["mu"]); test["occ_recent2_50"] = clip_z((.5*pr2+.5*test["p_hurdle"])*test["mu"])
        qr=np.clip(np.log1p(yr)/np.maximum(mur,1e-4),0,1); wr=np.maximum(mur,.05)**2; cwr=float(np.quantile(wr,.97)); wr=np.clip(wr,.05,max(cwr,.05)); wr/=max(float(wr.mean()),EPS)
        rr=_make_meta_reg(seed+60); rr.set_params(n_estimators=340,num_leaves=27,max_depth=6); rr.fit(Xr,qr,sample_weight=wr); q2=np.clip(rr.predict(Xt),0,1)
        test["q_recent2_score"] = q2; test["q_recent2"] = clip_z(q2*test["mu"])
        del cr, rr, Xr, yr, pr, mur
    else:
        test["occ_recent2"] = test["occ_meta"].copy(); test["occ_recent2_50"] = test["occ_meta50"].copy(); test["q_recent2"] = test["q_global"].copy()

    del reg, Xtr, Xt, ytr, ptr, mutr
    gc.collect()


def fit_final_router(bank: dict[str, dict[str, Any]], test: dict[str, Any], seed: int = 9200) -> None:
    available = [n for n in ROUTER_EXPERTS if all(n in bank[f] for f in FOLDS) and n in test]
    if not available:
        test["router20"] = test["proxy"].copy(); test["router35"] = test["proxy"].copy(); test["router_ranksoft"] = test["proxy"].copy(); return
    Xtr, ytr, _, _, _ = concat_all_oof(bank, candidate_names=available)
    Xt, _ = meta_matrix(test, candidate_names=available)
    ly = np.log1p(ytr)
    base = np.concatenate([bank[f]["proxy"] for f in FOLDS])
    base_loss = (base - ly)**2
    probs, names = [], []
    for k, n in enumerate(available):
        ez = np.concatenate([bank[f][n] for f in FOLDS])
        lab = (base_loss - (ez-ly)**2 > 0.0025).astype(np.int8)
        if lab.mean() < 0.03 or lab.mean() > 0.97: continue
        clf = _make_meta_clf(seed+k); clf.set_params(n_estimators=300, num_leaves=23, max_depth=6)
        clf.fit(Xtr, lab); probs.append(clf.predict_proba(Xt)[:,1]); names.append(n); del clf
    if not probs:
        test["router20"] = test["proxy"].copy(); test["router35"] = test["proxy"].copy(); test["router_ranksoft"] = test["proxy"].copy(); return
    P = np.column_stack(probs); k = P.argmax(1); bp = P[np.arange(len(P)), k]
    E = np.column_stack([test[n] for n in names]); chosen = E[np.arange(len(E)), k]
    for frac, nm in ((.20,"router20"),(.35,"router35")):
        thr=float(np.quantile(bp,1-frac)); act=(bp>=thr)&(bp>=.52); z=test["proxy"].copy(); z[act]=.5*z[act]+.5*chosen[act]; test[nm]=clip_z(z)
    r=rank01(bp); a=np.clip((r-.60)/.40,0,1)*.65; a*=bp>=.50
    test["router_ranksoft"] = clip_z((1-a)*test["proxy"]+a*chosen)
    del Xtr, Xt, P, E; gc.collect()


def materialize_test_blends(test: dict[str, Any], candidate_summaries: list[dict[str, Any]]) -> None:
    # Reconstruct exactly the same candidate keys used in OOF, but map the proxy part
    # to the *actual stronger friend* in final submission blends.  This is deliberate:
    # validation uses the structurally matched one-seed proxy; deployment keeps the
    # teammate's stronger AVG3 base wherever a candidate was a global blend.
    for s in candidate_summaries:
        n = s["name"]
        if n in test:
            continue
        if n.startswith("blend_"):
            body = n[len("blend_"):]
            src, aa = body.rsplit("_a", 1)
            a = int(aa) / 100.0
            if src in test:
                test[n] = clip_z((1-a)*test["friend"] + a*test[src])


# =============================================================================
# submission selection with explicit diversity
# =============================================================================

def diversity_stats(z: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    a, _ = level_calibrate_test(z); b, _ = level_calibrate_test(ref)
    d = a-b
    return dict(corr=float(np.corrcoef(a,b)[0,1]), var_delta=float(np.var(d)),
                std_delta=float(np.std(d)), mean_abs_delta=float(np.mean(np.abs(d))),
                p_abs_gt_002=float(np.mean(np.abs(d)>.02)), p_abs_gt_005=float(np.mean(np.abs(d)>.05)),
                p_abs_gt_010=float(np.mean(np.abs(d)>.10)))


def quality_penalty(s: dict[str, Any], proxy_wcv: float) -> float:
    # Lower is better. Penalize temporal brittleness more than a tiny mean gain.
    return float(s["wcv"] + 0.40*max(0.0, s["worst_delta"]) + 0.20*s["std_delta"] +
                 0.00020*max(0, 3-s["wins"]) + 0.00030*max(0.0, s["latest_delta"]))


def choose_three(summaries: list[dict[str, Any]], test: dict[str, Any]) -> list[dict[str, Any]]:
    proxy = next(s for s in summaries if s["name"] == "proxy")
    proxy_wcv = proxy["wcv"]
    eligible = []
    legacy_diagnostics = {"cap", "unc", "dist", "seq42", "etx42"}
    for s in summaries:
        n=s["name"]
        if n=="proxy" or n in legacy_diagnostics or n not in test: continue
        d=diversity_stats(test[n], test["friend"])
        row=dict(s); row.update({f"div_friend_{k}":v for k,v in d.items()}); row["quality_penalty"]=quality_penalty(s,proxy_wcv)
        # Meaningfully different, but not absurdly so.  The threshold is intentionally
        # modest in z-space: 0.05 is already ~5% multiplicative movement near typical z.
        row["meaningfully_different"] = bool(d["p_abs_gt_005"] >= .10 or d["std_delta"] >= .018)
        row["robust"] = bool(s["wins"]>=3 and s["latest_delta"]<=0 and s["wcv"] <= proxy_wcv+0.00015)
        eligible.append(row)
    if not eligible:
        raise RuntimeError("No final candidates available")
    eligible.sort(key=lambda r:(not r["robust"], r["quality_penalty"], r["wcv"]))

    selected=[]
    # First: safest robust improvement with non-trivial distance.
    first = next((r for r in eligible if r["robust"] and r["meaningfully_different"]), eligible[0])
    selected.append(first)

    # Second: prefer occurrence/q family different from first, with pairwise diversity.
    def pair_ok(r):
        if r["name"] in {x["name"] for x in selected}: return False
        if not r["meaningfully_different"]: return False
        for x in selected:
            dd=diversity_stats(test[r["name"]], test[x["name"]])
            if dd["p_abs_gt_005"] < .07 and dd["std_delta"] < .012:
                return False
        return True
    pref2 = [r for r in eligible if pair_ok(r) and r["family"] != first["family"]]
    selected.append(pref2[0] if pref2 else next(r for r in eligible if r["name"] not in {x["name"] for x in selected}))

    # Third: explicitly prefer router if it is not already represented.
    used={x["name"] for x in selected}
    pref3=[r for r in eligible if r["name"] not in used and pair_ok(r) and r["family"]=="router"]
    if not pref3:
        pref3=[r for r in eligible if r["name"] not in used and pair_ok(r)]
    if not pref3:
        pref3=[r for r in eligible if r["name"] not in used]
    if not pref3:
        raise RuntimeError("Fewer than three distinct final candidates")
    selected.append(pref3[0])
    return selected


def validate_submission_frame(df, sample_path: Path) -> None:
    import pandas as pd
    if list(df.columns) != ["user_id","predict"]: raise AssertionError(df.columns)
    if len(df)!=250_000 or df.user_id.nunique()!=250_000: raise AssertionError("submission row/user count")
    p=df.predict.to_numpy(np.float64)
    if not np.isfinite(p).all() or (p<0).any(): raise AssertionError("invalid predictions")
    sample=pd.read_csv(sample_path, usecols=lambda c:c=="user_id")
    if len(sample)!=250_000: raise AssertionError("sample rows !=250k")
    if not np.array_equal(df.user_id.to_numpy(), sample.user_id.to_numpy()): raise AssertionError("sample user order mismatch")


def write_submissions(ctx: Context, selected: list[dict[str, Any]], test: dict[str, Any]) -> list[dict[str, Any]]:
    import pandas as pd
    sample=pd.read_csv(ctx.sample)
    if "user_id" not in sample.columns:
        raise ValueError("sample_submit has no user_id")
    sample=sample[["user_id"]].copy()
    uid=np.asarray(test["user_id"], dtype=np.int64)
    order=np.argsort(uid); suid=uid[order]
    pos=np.searchsorted(suid, sample.user_id.to_numpy(np.int64))
    if (pos>=len(uid)).any() or not np.array_equal(suid[pos], sample.user_id.to_numpy(np.int64)):
        raise ValueError("test predictions do not cover sample user_id")
    inv=order[pos]
    labels=["SAFE", "SPECIALIST", "DIVERSE"]
    outs=[]
    for k,(lab,s) in enumerate(zip(labels,selected),1):
        z0=np.asarray(test[s["name"]], np.float64)
        z,delta=level_calibrate_test(z0)
        pred=np.maximum(np.expm1(z),0.0)
        out=sample.copy(); out["predict"]=pred[inv]
        validate_submission_frame(out,ctx.sample)
        path=ctx.submissions/f"submission_best_bas_{k}_{lab.lower()}_{s['name']}.csv"
        out.to_csv(path,index=False,float_format="%.6f")
        outs.append(dict(rank=k,label=lab,candidate=s["name"],path=str(path),sha256=sha256(path),
                         level_delta=delta,mean_z=float(z.mean()),mean_pred=float(pred.mean()),
                         validation_wcv=s["wcv"],wins=s["wins"],latest_delta=s["latest_delta"],
                         family=s["family"],diversity_vs_friend={k:v for k,v in s.items() if k.startswith("div_friend_")}))
    return outs


# =============================================================================
# reports / error log / preflight
# =============================================================================

def append_error(ctx: Context, stage: str, model: str, fold: str, exc: Exception) -> None:
    p=ctx.results/"errors.jsonl"; p.parent.mkdir(parents=True,exist_ok=True)
    row=dict(time=now_iso(),stage=stage,model=model,fold=fold,error=repr(exc),traceback=traceback.format_exc())
    with p.open("a",encoding="utf-8") as f:f.write(json.dumps(row,ensure_ascii=False)+"\n")


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import pandas as pd
    if not rows:
        path.write_text("",encoding="utf-8"); return
    flat=[]
    for r in rows:
        q={}
        for k,v in r.items():
            if isinstance(v,(list,dict,tuple,np.ndarray)): q[k]=json.dumps(v,ensure_ascii=False,default=_json_default)
            else:q[k]=v
        flat.append(q)
    pd.DataFrame(flat).to_csv(path,index=False)


def preflight_data(ctx: Context) -> dict[str, Any]:
    import polars as pl
    schema=pl.read_parquet_schema(ctx.raw)
    required={"event_date","user_id","searches","cat","to_cart","to_ord","gmv","gmv_search","gmv_cat"}
    missing=required-set(schema)
    if missing: raise ValueError(f"train.parquet missing {sorted(missing)}")
    # Metadata-only row count is cheap through lazy count.
    n=pl.scan_parquet(ctx.raw).select(pl.len()).collect().item()
    sample=pl.read_csv(ctx.sample,columns=["user_id"])
    if sample.height!=250_000 or sample["user_id"].n_unique()!=250_000: raise ValueError("invalid sample users")
    return dict(raw_rows=int(n),raw_columns=len(schema),sample_rows=sample.height,
                raw=str(ctx.raw),sample=str(ctx.sample),cuda=cuda_info())


def self_test() -> None:
    rng=np.random.default_rng(42); n=5000
    y=np.where(rng.random(n)<.55,np.expm1(rng.normal(3,1,n).clip(0)),0)
    z=np.maximum(np.log1p(y)+rng.normal(0,.8,n)+.15,0)
    d,sc=calibrate(y,z)
    assert np.isfinite(d) and np.isfinite(sc)
    zz,_=level_calibrate_test(z); assert (zz>=0).all()
    r=rank01(rng.normal(size=n)); assert r.min()==0 and r.max()==1
    assert abs(weighted_cv([1,2,3,4])-(1+4+12+32)/15)<1e-12
    # Synthetic competence gate must remain finite and bounded.
    p=rng.random(n); chosen=z+rng.normal(0,.1,n); rr=rank01(p); a=np.clip((rr-.6)/.4,0,1)*.65
    out=(1-a)*z+a*chosen; assert np.isfinite(out).all()
    print("SELF-TEST OK")


def parse_args():
    ap=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--max-hours",type=float,default=23.0)
    ap.add_argument("--stop-new-hours",type=float,default=21.0,
                    help="после этого времени не начинать новые optional heavy stages")
    ap.add_argument("--work-dir",default="_best_bas_research")
    ap.add_argument("--threads",type=int,default=max(1,min(12,os.cpu_count() or 1)))
    ap.add_argument("--no-install",action="store_true")
    ap.add_argument("--skip-neural-oof",action="store_true",
                    help="оставить только table/meta ветку; production neural test arrays всё равно используются в friend base")
    ap.add_argument("--preflight-only",action="store_true")
    ap.add_argument("--self-test",action="store_true")
    ap.add_argument("--row-frac",type=float,default=1.0,
                    help="только для отладки: hash-доля обучающих пользователей; production=1")
    return ap.parse_args()


def main() -> None:
    a=parse_args()
    if a.self_test:
        self_test(); return
    started=time.time(); base=Path(__file__).resolve().parent
    package=discover_package(base)
    ensure_dependencies(package,a.no_install)
    raw,sample=discover_raw_and_sample(base,package)
    work=(base/a.work_dir).resolve(); results=work/"results"; submissions=work/"submissions"; checkpoints=work/"checkpoints"
    for p in (work,results,submissions,checkpoints):p.mkdir(parents=True,exist_ok=True)
    budget=Budget(started=started,max_hours=a.max_hours,stop_new_hours=min(a.stop_new_hours,a.max_hours-1.0))
    ctx=Context(base,package,package/"pipeline",raw,sample,work,results,submissions,checkpoints,budget)
    configure_pipeline(ctx,a.threads)

    friend=verify_friend_package(package)
    pf=preflight_data(ctx)
    manifest=dict(script_version=SCRIPT_VERSION,started_at=now_iso(),package=str(package),preflight=pf,
                  friend_rebuild_max_log_error=friend["max_log_error"],friend_known_wcv=KNOWN_FRIEND_WCV,
                  max_hours=a.max_hours,stop_new_hours=budget.stop_new_hours,row_frac=a.row_frac)
    atomic_json(results/"RUN_MANIFEST.json",manifest)
    log("PREFLIGHT OK",json.dumps(pf,ensure_ascii=False))
    log(f"STRONGEST_CURRENT exact rebuild OK; public LB=1.6496571; known wCV={KNOWN_FRIEND_WCV:.5f}")
    if a.preflight_only:
        return

    # ------------------------------------------------------------------ A. table core
    runtime_rows=[]
    for variant in ("cap","unc","dist","hurdle"):
        fold_times=[]
        for fold in FOLDS:
            if fold_ckpt(ctx,variant,fold).exists(): continue
            est=max(.20,1.35*np.median(fold_times)) if fold_times else .55
            if not budget.can_start(est,mandatory=True):
                raise RuntimeError(f"Недостаточно бюджета для mandatory table {variant}/{fold}")
            t=time.time()
            try:train_table_fold(ctx,variant,fold,row_frac=a.row_frac)
            except Exception as exc:append_error(ctx,"table",variant,fold,exc);raise
            h=(time.time()-t)/3600;fold_times.append(h);runtime_rows.append(dict(stage="table",model=variant,fold=fold,hours=h))
            atomic_json(results/"progress.json",dict(stage="table",model=variant,fold=fold,elapsed_hours=budget.elapsed_hours))

    # ------------------------------------------------------------------ B. optional neural OOF replicas
    neural_complete={"etx42":False,"seq42":False}
    if not a.skip_neural_oof and a.row_frac==1.0:
        # ETX first: stronger one-seed standalone in the teammate experiments.
        for family in ("etx42","seq42"):
            t=time.time(); neural_complete[family]=run_complete_neural_family(ctx,family,budget)
            runtime_rows.append(dict(stage="neural_family",model=family,fold="all",hours=(time.time()-t)/3600,
                                     complete=neural_complete[family]))
    else:
        log("Neural OOF skipped by flag/debug row_frac")

    # ------------------------------------------------------------------ C. temporal meta / local specialists / router
    bank=build_fold_bank(ctx,neural_complete)
    hurdle_choices=select_temporal_hurdle(bank)
    proxy_sc=score_bank_candidate(bank,"proxy")
    log("Temporal hurdle round choices:", hurdle_choices)
    log("STRUCTURAL PROXY wCV",f"{proxy_sc['wcv']:.6f}","(exact teammate known",KNOWN_FRIEND_WCV,")")
    train_temporal_meta(bank)
    add_pairwise_router(bank)
    summaries,fold_rows=evaluate_candidates(bank)
    save_csv(results/"candidate_summary.csv",summaries);save_csv(results/"fold_scores.csv",fold_rows);save_csv(results/"runtime.csv",runtime_rows)
    log("Top local candidates:")
    for s in summaries[:12]:log(f"  {s['name']:<34} wCV={s['wcv']:.6f} wins={s['wins']}/4 latest={s['latest_delta']:+.6f} worst={s['worst_delta']:+.6f}")

    # ------------------------------------------------------------------ D. future full fit + test meta
    # Reserve this stage: without it there are no deployable new predictions.
    if not budget.can_start(0.5,mandatory=True):
        raise RuntimeError("Бюджет почти исчерпан до final refit; увеличьте --max-hours")
    test=build_test_record(ctx,friend,neural_complete)
    hv=future_hurdle_choice(bank)
    if hv == "hurdle240": pkey,mkey="p_hurdle240","mu240"
    elif hv == "hurdle320": pkey,mkey="p_hurdle320","mu320"
    else: pkey,mkey="p_hurdle","mu"
    test["hurdle_temporal"] = test[hv].copy()
    test["p_hurdle"] = test[pkey].copy(); test["mu"] = test[mkey].copy()
    log("Future hurdle round choice:", hv)
    fit_final_meta(bank,test)
    fit_final_router(bank,test)
    materialize_test_blends(test,summaries)

    # Save compact future candidate bank for reproducibility.
    bank_path=checkpoints/"test"/"final_candidate_bank.npz";bank_path.parent.mkdir(parents=True,exist_ok=True)
    keep=[n for n in [s["name"] for s in summaries]+["friend","proxy"] if n in test]
    np.savez_compressed(bank_path,user_id=test["user_id"].astype(np.int64),**{n:np.asarray(test[n],np.float32) for n in dict.fromkeys(keep)})

    # ------------------------------------------------------------------ E. diversity-aware 3 submissions
    selected=choose_three(summaries,test)
    outs=write_submissions(ctx,selected,test)
    save_csv(results/"FINAL_SELECTION.csv",selected)

    # Diversity matrix among final + friend.
    div_rows=[]; names=[o["candidate"] for o in outs]+["friend"]
    for i,n1 in enumerate(names):
        for n2 in names[i+1:]:
            d=diversity_stats(test[n1],test[n2]);div_rows.append(dict(a=n1,b=n2,**d))
    save_csv(results/"diversity_matrix.csv",div_rows)

    manifest.update(dict(finished_at=now_iso(),runtime_hours=budget.elapsed_hours,neural_complete=neural_complete,
                         proxy_wcv=proxy_sc["wcv"],hurdle_temporal_choices=hurdle_choices,future_hurdle_choice=hv,best_candidate=summaries[0],selected_submissions=outs,
                         hard_limit_respected=budget.elapsed_hours < a.max_hours,
                         notes=[
                             "Historical meta/routing uses previous primary folds only.",
                             "Exact STRONGEST_CURRENT is reconstructed only for future deployment; no archived test prediction is used as historical OOF.",
                             "Global blend validation uses the one-seed structural proxy; final blend retains the stronger exact friend AVG3 base.",
                             "All final candidates are shifted to teammate-proven level 2.3293 before expm1.",
                         ]))
    atomic_json(results/"RUN_MANIFEST.json",manifest)

    readme=[]
    readme.append("BEST_BAS RESEARCH — ИТОГ\n")
    readme.append(f"Версия: {SCRIPT_VERSION}")
    readme.append(f"Время: {budget.elapsed_hours:.3f} ч; лимит {a.max_hours:.2f} ч")
    readme.append(f"Точная база товарища: known wCV={KNOWN_FRIEND_WCV:.5f}, public LB=1.6496571")
    readme.append(f"Структурный OOF proxy этого запуска: wCV={proxy_sc['wcv']:.6f}")
    readme.append("\nТри файла выбраны не как top-3 подряд, а с ограничением на семейство и различие прогнозов:")
    for o in outs:
        readme.append(f"{o['rank']}. {o['label']}: {o['candidate']} | local wCV={o['validation_wcv']:.6f} | wins={o['wins']}/4 | {Path(o['path']).name}")
    readme.append("\nГлавные диагностические файлы: candidate_summary.csv, fold_scores.csv, diversity_matrix.csv, FINAL_SELECTION.csv, RUN_MANIFEST.json.")
    (results/"README_RESULTS_RU.txt").write_text("\n".join(readme)+"\n",encoding="utf-8")

    log("DONE. Three submissions:")
    for o in outs:log(" ",o["path"])
    log("Reports:",results)


if __name__=="__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user; completed checkpoints are reusable on restart.",file=sys.stderr)
        raise

```
