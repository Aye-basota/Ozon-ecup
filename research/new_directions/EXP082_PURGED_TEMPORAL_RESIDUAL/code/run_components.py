"""Frozen production-component replay for EXP082.

This runner deliberately reuses the production repository's training functions.
It writes only EXP082-prefixed artifacts under the clean research repository and
never touches historical OOF/model files.  Each family/fold is independently
resumable so GPU runs stay bounded and auditable.
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
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
EXP = HERE.parent
OUT = EXP / "production_components"
OZON = Path(r"C:\Users\Admin\Desktop\OZON-E-CUP")
ART = OZON / "artifacts"
FOLDS = tuple(dt.date.fromisoformat(x) for x in (
    "2025-07-03", "2025-08-07", "2025-09-11", "2025-10-16"
))
CANONICAL = dt.date(2025, 10, 16)

sys.path.insert(0, str(OZON))


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def artifact_path(family: str, fold: dt.date) -> Path:
    return OUT / f"{family}_{fold.isoformat()}.npz"


def metadata_path(family: str, fold: dt.date) -> Path:
    return OUT / f"{family}_{fold.isoformat()}.json"


def save_once(family: str, fold: dt.date, arrays: dict[str, np.ndarray], meta: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = artifact_path(family, fold)
    if path.exists():
        old = np.load(path, allow_pickle=False)
        if set(old.files) != set(arrays):
            raise FileExistsError(f"schema drift: {path}")
        for key, value in arrays.items():
            if not np.array_equal(old[key], value, equal_nan=True):
                raise FileExistsError(f"content drift: {path}:{key}")
    else:
        np.savez_compressed(path, **arrays)
    meta = {**meta, "artifact": str(path), "sha256": sha256(path)}
    mp = metadata_path(family, fold)
    text = json.dumps(jsonable(meta), ensure_ascii=False, indent=2) + "\n"
    if mp.exists() and mp.read_text(encoding="utf-8") != text:
        raise FileExistsError(f"metadata drift: {mp}")
    if not mp.exists():
        mp.write_text(text, encoding="utf-8")


def align_canonical(name: str, fold: dt.date, field: str = "z") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(ART / f"oof_{name}.npz", allow_pickle=True)
    mask = np.asarray(data["cutoff"]).astype(str) == fold.isoformat()
    return (np.asarray(data["user_id"])[mask].astype(np.int64),
            np.asarray(data[field])[mask].astype(np.float64),
            np.asarray(data["y"])[mask].astype(np.float64))


def copy_canonical(family: str, fold: dt.date) -> None:
    source = {
        "cap": "S1-E03a", "unc": "S1-E02", "dist": "S1-DIST",
        "seq": "SEQ-01-S42", "etx": "ETX-01-S42",
    }[family]
    uid, z, y = align_canonical(source, fold)
    arrays: dict[str, np.ndarray] = {
        "user_id": uid, "cutoff": np.full(len(uid), fold.isoformat(), dtype="U10"),
        "target_y": y.astype(np.float64), "target_log": np.log1p(y),
        "z": z.astype(np.float64),
    }
    sources = [ART / f"oof_{source}.npz"]
    if family == "dist":
        pact = np.load(ART / f"PACT_dist_{fold.isoformat()}.npz", allow_pickle=False)
        if not np.array_equal(pact["user_id"], uid):
            raise AssertionError("canonical DIST p_act alignment failed")
        arrays["p_act"] = pact["p_act"].astype(np.float64)
        arrays["p0"] = pact["p0"].astype(np.float64)
        sources.append(ART / f"PACT_dist_{fold.isoformat()}.npz")
    save_once(family, fold, arrays, {
        "mode": "canonical frozen artifact reuse", "family": family,
        "cutoff": fold.isoformat(), "source_name": source,
        "source_paths": sources, "source_sha256": [sha256(p) for p in sources],
        "runtime_seconds": 0.0, "config_changed": False,
    })


def run_tabular(family: str, fold: dt.date) -> None:
    from src.config import SEED
    from src.data import load
    from src.features import feature_names, to_np
    from src.train import Setup, assemble, fit_free, infer, xy

    if fold == CANONICAL:
        copy_canonical(family, fold)
        return
    load()
    if family == "cap":
        setup = Setup(L=180, min_history=90, step=7, panel_blocks=3, train_blocks=1,
                      model="direct", rounds=600, params={"seed": SEED}, cutoffs="all")
    elif family == "unc":
        setup = Setup(L=0, min_history=90, step=7, panel_blocks=3, train_blocks=1,
                      model="direct", rounds=600, params={"seed": SEED}, cutoffs="all")
    elif family == "dist":
        setup = Setup(L=0, min_history=90, step=7, panel_blocks=3, train_blocks=1,
                      model="dist", rounds=250, params={"seed": SEED}, cutoffs="all",
                      norm_long=True)
    else:
        raise ValueError(family)

    t0 = time.time()
    train_cutoffs = setup.train_cutoffs(fold)
    if not train_cutoffs or max(train_cutoffs) + dt.timedelta(days=30) > fold:
        raise AssertionError("tabular target-availability rule failed")
    Xv_frame, yv = xy(fold, setup)
    feats = feature_names(Xv_frame)
    Xtr, ytr, wtr = assemble(train_cutoffs, setup, feats, fold)
    n_train = len(ytr)
    box = [Xtr]
    del Xtr
    model = fit_free(setup, box, ytr, None)
    Av = to_np(Xv_frame, feats)
    arrays: dict[str, np.ndarray] = {
        "user_id": Xv_frame["user_id"].to_numpy().astype(np.int64),
        "cutoff": np.full(len(yv), fold.isoformat(), dtype="U10"),
        "target_y": np.asarray(yv, np.float64),
        "target_log": np.log1p(np.asarray(yv, np.float64)),
    }
    if family == "dist":
        booster, centroids = model
        proba = booster.predict(Av, num_iteration=250)
        arrays["z"] = np.maximum(proba @ np.asarray(centroids), 0.0).astype(np.float64)
        arrays["p0"] = proba[:, 0].astype(np.float64)
        arrays["p_act"] = (1.0 - proba[:, 0]).astype(np.float64)
    else:
        arrays["z"] = np.maximum(infer(setup, model, Av), 0.0).astype(np.float64)
    elapsed = time.time() - t0
    save_once(family, fold, arrays, {
        "mode": "exact frozen pipeline replay", "family": family,
        "cutoff": fold.isoformat(), "target_end": str(fold + dt.timedelta(days=30)),
        "train_cutoffs": [str(x) for x in train_cutoffs],
        "max_train_target_end": str(max(train_cutoffs) + dt.timedelta(days=30)),
        "n_train": n_train, "n_validation": len(yv), "feature_count": len(feats),
        "feature_names": feats, "setup": setup.as_dict(), "seed": SEED,
        "runtime_seconds": elapsed, "config_changed": False,
    })
    del model, Av, box, ytr, wtr, Xv_frame
    gc.collect()


def run_seq(fold: dt.date) -> None:
    if fold == CANONICAL:
        copy_canonical("seq", fold)
        return
    from src import seq
    from src.config import SEED

    cfg = dict(seq.DEFAULT_CFG, hidden=64, blocks=8, kernel=3, dropout=0.10,
               batch=1024, chunk=256, lr=3e-3, wd=1e-2, epochs=4, warmup=300,
               seed=SEED, workers=3, compile=False, aug="none", depth_aug=0.0)
    t0 = time.time()
    uid, z, y, hist = seq.train_fold(fold, cfg, curve=False, ckpt=None)
    elapsed = time.time() - t0
    train_cutoffs = seq.fold_cutoffs(fold)
    save_once("seq", fold, {
        "user_id": uid.astype(np.int64),
        "cutoff": np.full(len(uid), fold.isoformat(), dtype="U10"),
        "target_y": y.astype(np.float64), "target_log": np.log1p(y.astype(np.float64)),
        "z": z.astype(np.float64),
    }, {
        "mode": "frozen single-seed production approximation", "family": "SEQ",
        "source_recipe": "SEQ-01-S42; EXP025/EXP026/EXP037", "cutoff": fold.isoformat(),
        "target_end": str(fold + dt.timedelta(days=30)),
        "train_cutoffs": [str(x) for x in train_cutoffs],
        "max_train_target_end": str(max(train_cutoffs) + dt.timedelta(days=30)),
        "config": cfg, "history": hist, "runtime_seconds": elapsed,
        "config_changed": False, "approximation": "single seed instead of AVG3",
    })
    import torch
    torch.cuda.empty_cache()


def run_etx(fold: dt.date) -> None:
    if fold == CANONICAL:
        copy_canonical("etx", fold)
        return
    from src import etx, seq
    from src.config import SEED

    cfg = dict(etx.DEFAULT_CFG, d_model=128, blocks=5, heads=8, head_dim=16,
               ffn=384, dropout=0.10, n_tok=192, batch=512, chunk=128,
               lr=1.5e-3, wd=1e-2, epochs=4, warmup=500, seed=SEED, compile=False)
    t0 = time.time()
    uid, z, y, hist, model, fitted = etx.train_fold(fold, cfg, curve=False, ckpt=None)
    elapsed = time.time() - t0
    train_cutoffs = seq.fold_cutoffs(fold)
    save_once("etx", fold, {
        "user_id": uid.astype(np.int64),
        "cutoff": np.full(len(uid), fold.isoformat(), dtype="U10"),
        "target_y": y.astype(np.float64), "target_log": np.log1p(y.astype(np.float64)),
        "z": z.astype(np.float64),
    }, {
        "mode": "frozen single-seed production approximation", "family": "ETX",
        "source_recipe": "ETX-01-S42; EXP036/EXP037", "cutoff": fold.isoformat(),
        "target_end": str(fold + dt.timedelta(days=30)),
        "train_cutoffs": [str(x) for x in train_cutoffs],
        "max_train_target_end": str(max(train_cutoffs) + dt.timedelta(days=30)),
        "config": fitted, "history": hist, "runtime_seconds": elapsed,
        "config_changed": False, "approximation": "single seed instead of AVG3",
    })
    del model
    import torch
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("family", choices=["cap", "unc", "dist", "seq", "etx"])
    parser.add_argument("--fold", required=True, choices=[x.isoformat() for x in FOLDS])
    args = parser.parse_args()
    fold = dt.date.fromisoformat(args.fold)
    existing = artifact_path(args.family, fold)
    if existing.exists():
        print(f"resume: {existing} already exists (sha256={sha256(existing)})", flush=True)
        return
    if args.family in {"cap", "unc", "dist"}:
        run_tabular(args.family, fold)
    elif args.family == "seq":
        run_seq(fold)
    else:
        run_etx(fold)
    print(f"done: {artifact_path(args.family, fold)}", flush=True)


if __name__ == "__main__":
    main()
