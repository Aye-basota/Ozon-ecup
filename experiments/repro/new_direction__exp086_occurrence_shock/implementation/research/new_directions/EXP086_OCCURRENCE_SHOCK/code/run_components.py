"""Exact teammate occurrence-component replay for EXP086.

The code deliberately mirrors the frozen source in
``continue_best_bas_final6h.py`` and its parent runners.  It writes only under
EXP086, is resumable per component, and never edits the historical repository.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
EXP = HERE.parent
OUT = EXP / "occurrence_components"
ROOT = HERE.parents[3]
OZON = Path(r"C:\Users\Admin\Desktop\OZON-E-CUP")
SOURCE = OZON / "пайплайн сокомандника" / "research_scripts"
EXP082 = ROOT / "research" / "new_directions" / "EXP082_PURGED_TEMPORAL_RESIDUAL"
PROD = EXP082 / "production_components"
FOLDS = tuple(dt.date.fromisoformat(x) for x in (
    "2025-07-03", "2025-08-07", "2025-09-11", "2025-10-16",
))
EPS = 1e-7

sys.path.insert(0, str(OZON))


@dataclass(frozen=True)
class OccCfg:
    name: str
    maxcuts: int
    tau: float
    rounds: int
    leaves: int
    min_leaf: int
    feature_mode: str = "all"
    feature_fraction: float = 0.82


# Byte-for-byte values from continue_best_bas_final6h.py::OCC_QUEUE.
OCC_QUEUE = (
    OccCfg("occ_r10_fast", 10, 55.0, 380, 31, 520, "all", 0.82),
    OccCfg("occ_r16_bal", 16, 100.0, 440, 47, 430, "all", 0.84),
    OccCfg("occ_r22_stable", 22, 180.0, 500, 31, 650, "all", 0.80),
    OccCfg("occ_r14_multiscale", 14, 85.0, 430, 47, 430, "multiscale", 0.90),
    OccCfg("occ_r18_wide", 18, 125.0, 470, 63, 420, "all", 0.76),
    OccCfg("occ_r24_multiscale", 24, 220.0, 520, 31, 700, "multiscale", 0.88),
    OccCfg("occ_r12_wide", 12, 70.0, 430, 79, 380, "all", 0.72),
    OccCfg("occ_r20_shallow", 20, 155.0, 500, 23, 760, "all", 0.90),
)
OCC_MAP = {x.name: x for x in OCC_QUEUE}

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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def component_path(kind: str, fold: dt.date, name: str | None = None) -> Path:
    stem = kind if name is None else name
    return OUT / f"{stem}__{fold.isoformat()}.npz"


def meta_path(path: Path) -> Path:
    return path.with_suffix(".json")


def save_once(path: Path, arrays: dict[str, np.ndarray], meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            if set(old.files) != set(arrays):
                raise FileExistsError(f"schema drift: {path}")
            for key, value in arrays.items():
                if not np.array_equal(old[key], value, equal_nan=True):
                    raise FileExistsError(f"content drift: {path}:{key}")
    else:
        temp = path.with_name(path.name + f".tmp.{os.getpid()}.npz")
        np.savez_compressed(temp, **arrays)
        os.replace(temp, path)
    payload = {**meta, "artifact": str(path), "sha256": sha256(path)}
    text = json.dumps(jsonable(payload), ensure_ascii=False, indent=2) + "\n"
    mp = meta_path(path)
    if mp.exists() and mp.read_text(encoding="utf-8") != text:
        raise FileExistsError(f"metadata drift: {mp}")
    if not mp.exists():
        mp.write_text(text, encoding="utf-8")


def load_prod(family: str, fold: dt.date) -> dict[str, np.ndarray]:
    path = PROD / f"{family}_{fold.isoformat()}.npz"
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def choose_meta_features(feats: list[str], max_n: int = 72) -> list[str]:
    out: list[str] = []
    for window in META_WINDOWS:
        for suffix in META_SUFFIXES:
            name = f"w{window}_{suffix}"
            if name in feats:
                out.append(name)
    out += [x for x in sorted(META_OTHER) if x in feats]
    out += [x for x in feats if x.startswith(("rec_over_", "trend_", "dlog_"))]
    unique: list[str] = []
    seen: set[str] = set()
    for name in out:
        if name not in seen:
            unique.append(name)
            seen.add(name)
    return unique[:max_n]


def multiscale_features(feats: list[str]) -> list[str]:
    wins = ("w7_", "w14_", "w30_", "w60_", "w90_", "w180_")
    prefix = ("rec_", "trend_", "dlog_", "gap_", "buygap_", "pt_")
    exact = {"weekend_share", "tenure_frac", "first_buy_frac", "gap_max_frac"}
    out = [x for x in feats if x.startswith(wins) or x.startswith(prefix) or x in exact]
    unique: list[str] = []
    seen: set[str] = set()
    for name in out:
        if name not in seen:
            unique.append(name)
            seen.add(name)
    return unique


def assert_alignment(fold: dt.date, uid: np.ndarray, y: np.ndarray) -> None:
    ref = load_prod("cap", fold)
    if not np.array_equal(uid.astype(np.int64), ref["user_id"].astype(np.int64)):
        raise AssertionError(f"user alignment failed: {fold}")
    if not np.allclose(np.asarray(y, np.float64), ref["target_y"], atol=1e-5, rtol=1e-7):
        raise AssertionError(f"target alignment failed: {fold}")


def build_meta_raw(fold: dt.date) -> None:
    """Rebuild the exact 72-column cap meta_raw view without training a model."""
    path = component_path("meta_raw", fold)
    if path.exists():
        print(f"resume: {path}", flush=True)
        return
    from src.data import load
    from src.features import feature_names, make_xy, to_np

    load()
    started = time.time()
    frame, y = make_xy(fold, L=180, n_blocks=3, with_target=True, norm_long=False)
    feats = feature_names(frame)
    names = choose_meta_features(feats)
    uid = frame["user_id"].to_numpy().astype(np.int64)
    assert_alignment(fold, uid, y)
    X = to_np(frame, names).astype(np.float32)
    save_once(path, {
        "user_id": uid, "y": np.asarray(y, np.float32), "X": X,
        "names": np.asarray(names, dtype="U80"),
    }, {
        "kind": "meta_raw", "cutoff": fold.isoformat(), "runtime_seconds": time.time() - started,
        "source_function": "run_best_bas_research_23h.py::choose_meta_features/build_test_meta_raw",
        "feature_count": len(names), "feature_names": names,
        "source_sha256": sha256(SOURCE / "run_best_bas_research_23h.py"),
        "target_safe": True, "config_changed": False,
    })


def train_hurdle(fold: dt.date, threads: int) -> None:
    """Rebuild the exact base hurdle supplying p_base and conditional mu."""
    path = component_path("hurdle", fold)
    if path.exists():
        print(f"resume: {path}", flush=True)
        return
    os.environ["LGB_THREADS"] = str(threads)
    from src.data import load
    from src.features import feature_names, to_np
    from src import models
    from src.train import Setup, assemble, select_features, xy, _XY

    load()
    params = dict(
        learning_rate=0.035, num_leaves=63, min_data_in_leaf=260,
        feature_fraction=0.78, bagging_fraction=0.88, bagging_freq=1,
        lambda_l2=14.0, lambda_l1=1.5, max_bin=63,
    )
    setup = Setup(L=0, min_history=90, step=7, panel_blocks=3, train_blocks=1,
                  model="two_part", rounds=420, norm_long=True, params=params)
    cuts = setup.train_cutoffs(fold)
    if not cuts or max(cuts) + dt.timedelta(days=30) > fold:
        raise AssertionError("hurdle target-availability rule failed")
    started = time.time()
    Xv, yv = xy(fold, setup)
    feats = select_features(feature_names(Xv), setup.drop_groups, setup.keep_only)
    X, y, weights = assemble(cuts, setup, feats, fold)
    n_train = len(y)
    # The frozen base runner passed None here; reproduce that semantics exactly.
    datasets = models.make_datasets("two_part", X, y, None, setup.params)
    del X
    _XY.clear()
    gc.collect()
    clf, reg = models.train_two_part_ds(datasets, setup.params, setup.rounds)
    A = to_np(Xv, feats)
    p = np.clip(clf.predict(A), EPS, 1.0 - EPS).astype(np.float32)
    mu = np.maximum(reg.predict(A), 0.0).astype(np.float32)
    uid = Xv["user_id"].to_numpy().astype(np.int64)
    assert_alignment(fold, uid, yv)
    save_once(path, {
        "user_id": uid, "y": np.asarray(yv, np.float32), "p": p, "mu": mu,
        "z": (p.astype(np.float64) * mu.astype(np.float64)).astype(np.float32),
    }, {
        "kind": "base_hurdle", "cutoff": fold.isoformat(),
        "target_end": str(fold + dt.timedelta(days=30)),
        "train_cutoffs": [str(x) for x in cuts],
        "max_train_target_end": str(max(cuts) + dt.timedelta(days=30)),
        "n_train": n_train, "n_validation": len(uid), "feature_count": len(feats),
        "feature_names": feats, "setup": setup.as_dict(), "seed": 42,
        "runtime_seconds": time.time() - started, "weights_intentionally_used": False,
        "source_function": "run_best_bas_research_23h.py::variant_setup/train_table_fold",
        "source_sha256": sha256(SOURCE / "run_best_bas_research_23h.py"),
        "target_safe": True, "config_changed": False,
    })
    del datasets, clf, reg, A, Xv, y, weights, p, mu
    _XY.clear()
    gc.collect()


def train_occurrence(fold: dt.date, name: str, threads: int) -> None:
    """Train one exact final6h raw occurrence probability head."""
    path = component_path("occ", fold, name)
    if path.exists():
        print(f"resume: {path}", flush=True)
        return
    cfg = OCC_MAP[name]
    os.environ["LGB_THREADS"] = str(threads)
    from src.data import load
    from src.features import feature_names, to_np
    from src import models
    from src.train import Setup, assemble, select_features, xy, _XY
    import lightgbm as lgb

    load()
    common = dict(
        learning_rate=0.035, num_leaves=63, min_data_in_leaf=220,
        feature_fraction=0.82, bagging_fraction=0.90, bagging_freq=1,
        lambda_l2=14.0, lambda_l1=1.0, max_bin=127,
    )
    setup = Setup(L=0, min_history=90, step=7, panel_blocks=3, train_blocks=1,
                  model="two_part", rounds=520, norm_long=True, weight_tau=105,
                  params=common)
    cuts = list(setup.train_cutoffs(fold))[-cfg.maxcuts:]
    if not cuts or max(cuts) + dt.timedelta(days=30) > fold:
        raise AssertionError("occurrence target-availability rule failed")
    started = time.time()
    Xv, yv = xy(fold, setup)
    feats = select_features(feature_names(Xv), setup.drop_groups, setup.keep_only)
    if cfg.feature_mode == "multiscale":
        selected = multiscale_features(feats)
        feats = selected if len(selected) >= 40 else feats
    X, y, weights = assemble(cuts, setup, feats, fold)
    n_train = len(y)
    params = models._params(dict(
        num_leaves=cfg.leaves, min_data_in_leaf=cfg.min_leaf,
        feature_fraction=cfg.feature_fraction, bagging_fraction=0.90,
        bagging_freq=1, lambda_l2=14.0, lambda_l1=1.0, max_bin=127,
        learning_rate=0.035, verbosity=-1, num_threads=threads,
    ), objective="binary", metric="binary_logloss")
    dataset = lgb.Dataset(X, (y > 0).astype(np.int8), weight=weights,
                          params=params, free_raw_data=True).construct()
    del X
    _XY.clear()
    gc.collect()
    model = lgb.train(params, dataset, num_boost_round=cfg.rounds)
    A = to_np(Xv, feats)
    p = np.clip(model.predict(A), EPS, 1.0 - EPS).astype(np.float32)
    uid = Xv["user_id"].to_numpy().astype(np.int64)
    assert_alignment(fold, uid, yv)
    save_once(path, {
        "user_id": uid, "y": np.asarray(yv, np.float32), "p": p,
    }, {
        "kind": "raw_occurrence_probability", "name": name, "cutoff": fold.isoformat(),
        "target": "1[GMV(T,T+30] > 0]", "target_end": str(fold + dt.timedelta(days=30)),
        "train_cutoffs": [str(x) for x in cuts],
        "max_train_target_end": str(max(cuts) + dt.timedelta(days=30)),
        "n_train": n_train, "n_validation": len(uid), "feature_count": len(feats),
        "feature_names": feats, "occ_config": asdict(cfg), "lgb_params": params,
        "seed": int(params.get("seed", 42)), "runtime_seconds": time.time() - started,
        "source_function": "continue_best_bas_final6h.py::train_occ_child",
        "source_sha256": sha256(SOURCE / "continue_best_bas_final6h.py"),
        "target_safe": True, "config_changed": False,
    })
    del dataset, model, A, Xv, y, weights, p
    _XY.clear()
    gc.collect()


def validate_all() -> None:
    missing: list[str] = []
    for fold in FOLDS:
        for kind in ("meta_raw", "hurdle"):
            if not component_path(kind, fold).exists():
                missing.append(str(component_path(kind, fold)))
        for cfg in OCC_QUEUE:
            if not component_path("occ", fold, cfg.name).exists():
                missing.append(str(component_path("occ", fold, cfg.name)))
    if missing:
        raise FileNotFoundError("missing components:\n" + "\n".join(missing))
    print("all EXP086 occurrence components present", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["meta_raw", "hurdle", "occ", "validate"])
    parser.add_argument("--fold", choices=[x.isoformat() for x in FOLDS])
    parser.add_argument("--name", choices=list(OCC_MAP))
    parser.add_argument("--threads", type=int, default=max(2, min(10, os.cpu_count() or 8)))
    args = parser.parse_args()
    if args.kind == "validate":
        validate_all()
        return
    if args.fold is None:
        parser.error("--fold is required")
    fold = dt.date.fromisoformat(args.fold)
    if args.kind == "meta_raw":
        build_meta_raw(fold)
    elif args.kind == "hurdle":
        train_hurdle(fold, args.threads)
    else:
        if args.name is None:
            parser.error("--name is required for occ")
        train_occurrence(fold, args.name, args.threads)


if __name__ == "__main__":
    main()
