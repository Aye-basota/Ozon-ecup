"""EXP-046: controlled UNC/CAP rounds x seed refresh inside STRONGEST_CURRENT.

One command runs the registered protocol with resumable, isolated trajectories::

    python src/tabular_backbone_refresh.py

The runner never reads the test cutoff, never writes ``submissions/`` and never
overwrites an existing trajectory.  Analysis can be repeated without training::

    python src/tabular_backbone_refresh.py --analysis-only
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# ``python src/<script>.py`` is the repository convention.  Make that invocation
# equivalent to ``python -m src.<module>`` without relying on PYTHONPATH.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import models
from src.blend import aligned
from src.config import (ARTIFACTS, FOLD_WEIGHTS_S1, LGB_PARAMS, ROOT, SEED,
                        VAL_FOLDS_S1)
from src.data import load
from src.features import feature_names, to_np
from src.merge_oof import auc_positive
from src.report import evaluate
from src.train import Setup, _XY, assemble, xy


EXP_NUM = 46
EXP_ID = "TABULAR-BACKBONE-REFRESH"
PREFIX = "TBR_EXP046"
RUN_DIR = ARTIFACTS / PREFIX
RESULTS = ROOT / "research" / "strategies" / "results" / "TABULAR_BACKBONE_REFRESH"
ROUNDS = (200, 250, 300, 600)
PRIMARY_ROUND = 300
SEEDS = (SEED, SEED + 1, SEED + 2)
FOLDS = tuple(VAL_FOLDS_S1)
FOLD_LABELS = tuple(v.isoformat() for v in FOLDS)
FOLD_WEIGHTS = np.asarray(FOLD_WEIGHTS_S1, dtype=np.float64)
FOLD_WEIGHTS /= FOLD_WEIGHTS.sum()
COMPONENTS = ("UNC", "CAP")
HISTORICAL = {"UNC": "S1-E02", "CAP": "S1-E03a"}
FIXED_COMPONENTS = ("S1-DIST", "ETX-AVG3", "SEQ-AVG3")
ALL_HISTORICAL = ("S1-E03a", "S1-E02", *FIXED_COMPONENTS)
WEIGHTS = {"CAP": 0.10, "UNC": 0.20, "S1-DIST": 0.25,
           "ETX-AVG3": 0.225, "SEQ-AVG3": 0.225}
EXPECTED_H = {
    "fold_cal": (1.766883357, 1.760509577, 1.748629224, 1.741278566),
    "wcv": 1.747509863,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(a.dtype.str.encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2,
                              sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows for {path}")
    fields: list[str] = []
    for row in rows:
        fields.extend(k for k in row if k not in fields)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(jsonable(v), ensure_ascii=False)
                        if isinstance(v, (dict, list, tuple)) else v
                        for k, v in row.items()})


def component_setup(component: str, seed: int, vals=None) -> Setup:
    """The literal historical production recipe; only master seed may differ.

    Historical artifacts set LightGBM's master ``seed`` and leave subordinate
    streams to LightGBM's deterministic derivation.  Explicitly forcing every
    child seed to the same integer would be a different estimator and would
    violate the replay gate.
    """
    if component == "UNC":
        return Setup(L=None, min_history=90, step=7, panel_blocks=3,
                     train_blocks=1, model="direct", rounds=600,
                     params={"seed": int(seed)}, cutoffs="all", vals=vals,
                     norm_long=False)
    if component == "CAP":
        return Setup(L=180, min_history=90, step=7, panel_blocks=3,
                     train_blocks=1, model="direct", rounds=600,
                     params={"seed": int(seed)}, cutoffs="all", vals=vals,
                     norm_long=False)
    raise ValueError(component)


def feature_path(component: str) -> Path:
    return ARTIFACTS / f"feats_{HISTORICAL[component]}.txt"


def saved_features(component: str) -> list[str]:
    path = feature_path(component)
    feats = path.read_text(encoding="utf-8").splitlines()
    expected = 236 if component == "UNC" else 195
    assert len(feats) == expected, f"{component}: expected {expected} features, got {len(feats)}"
    assert len(feats) == len(set(feats)), f"{component}: duplicate features"
    return feats


def seed_recipe(seed: int) -> dict[str, Any]:
    p = dict(LGB_PARAMS)
    p["seed"] = int(seed)
    return p


def recipe_manifest(component: str, seed: int = SEED) -> dict[str, Any]:
    s = component_setup(component, seed)
    feats = saved_features(component)
    return {
        "component": component,
        "historical_artifact": HISTORICAL[component],
        "model_type": "LightGBM direct regression on log1p(y)",
        "setup": s.as_dict(),
        "feature_count": len(feats),
        "feature_order_sha256": sha256_array(np.asarray(feats, dtype="U")),
        "feature_file": str(feature_path(component).resolve()),
        "feature_file_sha256": sha256_file(feature_path(component)),
        "lightgbm_params": seed_recipe(seed),
        "seed_policy": {
            "master_seed": int(seed),
            "subordinate_streams": "LightGBM-derived from master seed; no explicit child override",
            "reason": "literal historical recipe required by replay gate",
        },
        "early_stopping": False,
        "snapshots": list(ROUNDS),
        "prediction_semantics": "max(booster.predict(X, num_iteration=k), 0), raw log1p-space",
        "target_semantics": "GMV sum on (cutoff, cutoff+30], trained as log1p(y)",
        "row_order": "cutoff chronological, user_id ascending within fold",
        "dtype": {"features": "float32", "saved_prediction": "float32"},
    }


def source_hashes() -> dict[str, str]:
    names = ("config.py", "features.py", "models.py", "train.py", "validation.py",
             "report.py", "blend.py", "tabular_backbone_refresh.py")
    return {name: sha256_file(ROOT / "src" / name) for name in names}


def environment_manifest() -> dict[str, Any]:
    import lightgbm
    import pandas
    import polars
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "lightgbm": lightgbm.__version__,
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "polars": polars.__version__,
        "lgb_threads": int(LGB_PARAMS["num_threads"]),
        "source_sha256": source_hashes(),
    }


def row_keys(cutoff: np.ndarray, user_id: np.ndarray) -> np.ndarray:
    return np.char.add(np.char.add(np.asarray(cutoff, dtype="U10"), "|"),
                       np.asarray(user_id).astype("U20"))


def historical_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return aligned(list(ALL_HISTORICAL))


def historical_baseline() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    Z, y, cut = historical_arrays()
    z = (WEIGHTS["CAP"] * Z[0].astype(np.float64)
         + WEIGHTS["UNC"] * Z[1].astype(np.float64)
         + WEIGHTS["S1-DIST"] * Z[2].astype(np.float64)
         + WEIGHTS["ETX-AVG3"] * Z[3].astype(np.float64)
         + WEIGHTS["SEQ-AVG3"] * Z[4].astype(np.float64))
    rep = evaluate(y, z, cut)
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-15
    assert max(abs(a - b) for a, b in zip(rep["fold_cal"], EXPECTED_H["fold_cal"])) < 5e-10
    assert abs(rep["wcv"] - EXPECTED_H["wcv"]) < 5e-10
    return z, y, cut, rep


def _prefix_smoke_audit() -> dict[str, Any]:
    """Reuse exp_017's independent-300 vs prefix-300 artifact-backed check."""
    a = ARTIFACTS / "oof_S1-ROUNDS-R300.npz"
    b = ARTIFACTS / "oof_S1-ROUNDSL-R300.npz"
    da, db = np.load(a, allow_pickle=False), np.load(b, allow_pickle=False)
    for key in ("user_id", "cutoff", "y", "z"):
        assert np.array_equal(da[key], db[key]), f"exp_017 prefix smoke drift: {key}"
    return {"status": "PASS_BITWISE", "historical_experiment": "exp_017",
            "prefix_artifact": str(a.resolve()), "independent_artifact": str(b.resolve()),
            "prefix_sha256": sha256_file(a), "independent_sha256": sha256_file(b),
            "prediction_sha256": sha256_array(da["z"]), "rounds": 300}


def phase0_audit() -> dict[str, Any]:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    z_h, y, cut, rep = historical_baseline()
    Z, _, _ = historical_arrays()
    components = []
    for name, z in zip(ALL_HISTORICAL, Z):
        path = ARTIFACTS / f"oof_{name}.npz"
        d = np.load(path, allow_pickle=False)
        keys = row_keys(d["cutoff"], d["user_id"])
        assert len(keys) == len(np.unique(keys)), f"{name}: duplicate row keys"
        components.append({
            "name": name, "path": str(path.resolve()), "file_sha256": sha256_file(path),
            "prediction_sha256": sha256_array(d["z"]),
            "row_keys_sha256": sha256_array(keys), "target_sha256": sha256_array(d["y"]),
            "n": len(d["z"]), "prediction_dtype": str(d["z"].dtype),
        })
    # Validate current default feature construction against frozen production order.
    current_features = {}
    for component in COMPONENTS:
        s = component_setup(component, SEED, vals=[FOLDS[-1]])
        Xv, yv = xy(FOLDS[-1], s)
        now = feature_names(Xv)
        frozen = saved_features(component)
        assert now == frozen, f"{component}: current feature order != frozen production order"
        assert np.all(np.diff(Xv["user_id"].to_numpy()) > 0), f"{component}: user order"
        current_features[component] = {
            "matches_production": True, "n_features": len(now),
            "feature_order_sha256": sha256_array(np.asarray(now, dtype="U")),
            "validation_feature_dtype": str(to_np(Xv, frozen).dtype),
            "target_dtype": str(np.asarray(yv).dtype),
        }
        _XY.clear()
    manifest = {
        "experiment": EXP_ID, "experiment_number": EXP_NUM, "prefix": PREFIX,
        "weights": WEIGHTS, "weight_sum": sum(WEIGHTS.values()),
        "historical_components": components,
        "recipes": {c: recipe_manifest(c) for c in COMPONENTS},
        "current_feature_audit": current_features,
        "historical_baseline": {
            "folds": list(FOLD_LABELS), "fold_sizes": rep["fold_sizes"],
            "fold_cal": rep["fold_cal"], "wcv": rep["wcv"], "mean_z": rep["mean_z"],
            "prediction_sha256": sha256_array(z_h),
            "row_keys_sha256": sha256_array(row_keys(cut, _historical_uid())),
            "target_sha256": sha256_array(y), "n": len(y),
            "reconstruction_tolerance": 5e-10, "status": "PASS_EXACT",
        },
        "cutoff_policy": {
            "validation_folds": list(FOLD_LABELS), "fold_weights": list(FOLD_WEIGHTS_S1),
            "train_rule": "T+30<=V", "train_grid_step_days": 7,
            "train_panel_blocks": 1, "validation_panel_blocks": 3,
            "test_data_or_predictions_read": False,
        },
        "prefix_prediction_smoke": _prefix_smoke_audit(),
        "environment": environment_manifest(),
    }
    write_json(RESULTS / "baseline_manifest.json", manifest)
    write_json(RESULTS / "reconstructed_baseline_metrics.json", rep)
    return manifest


def _historical_uid() -> np.ndarray:
    d = np.load(ARTIFACTS / "oof_S1-E03a.npz", allow_pickle=False)
    k = row_keys(d["cutoff"], d["user_id"])
    return d["user_id"][np.argsort(k)]


def trajectory_stem(component: str, seed: int, fold: dt.date) -> str:
    return f"{PREFIX}_{component}_S{seed}_V{fold.strftime('%m%d')}"


def trajectory_paths(component: str, seed: int, fold: dt.date) -> dict[str, Path]:
    stem = trajectory_stem(component, seed, fold)
    return {"npz": RUN_DIR / f"{stem}.npz", "model": RUN_DIR / f"{stem}.txt",
            "manifest": RUN_DIR / f"{stem}.json"}


def trajectory_complete(component: str, seed: int, fold: dt.date) -> bool:
    paths = trajectory_paths(component, seed, fold)
    present = {k: p.exists() for k, p in paths.items()}
    if not any(present.values()):
        return False
    if not all(present.values()):
        raise RuntimeError(f"partial trajectory exists and will not be overwritten: {paths}")
    meta = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert meta["recipe_sha256"] == recipe_signature(component, seed)
    assert meta["fold"] == fold.isoformat()
    assert meta["artifacts"]["npz_sha256"] == sha256_file(paths["npz"])
    assert meta["artifacts"]["model_sha256"] == sha256_file(paths["model"])
    return True


def recipe_signature(component: str, seed: int) -> str:
    core = {"recipe": recipe_manifest(component, seed), "folds": FOLD_LABELS,
            "rounds": ROUNDS, "lgb_version": environment_manifest()["lightgbm"]}
    return hashlib.sha256(json.dumps(jsonable(core), sort_keys=True).encode()).hexdigest()


def _resolved_seeds(booster) -> dict[str, int]:
    wanted = ("seed", "bagging_seed", "feature_fraction_seed", "data_random_seed")
    found: dict[str, int] = {}
    pattern = re.compile(r"^\[([^:]+):\s*(-?\d+)\]$")
    for line in booster.model_to_string().splitlines():
        m = pattern.match(line)
        if m and m.group(1) in wanted:
            found[m.group(1)] = int(m.group(2))
    assert set(found) == set(wanted), f"resolved seed audit incomplete: {found}"
    return found


def train_trajectory(component: str, seed: int, fold: dt.date) -> dict[str, Any]:
    if trajectory_complete(component, seed, fold):
        return json.loads(trajectory_paths(component, seed, fold)["manifest"].read_text(
            encoding="utf-8"))
    paths = trajectory_paths(component, seed, fold)
    if any(p.exists() for p in paths.values()):
        raise FileExistsError(f"refusing to overwrite {paths}")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    load()
    s = component_setup(component, seed, vals=[fold])
    feats = saved_features(component)
    Xcheck, _ = xy(fold, s)
    assert feature_names(Xcheck) == feats
    del Xcheck
    _XY.clear()
    cuts = s.train_cutoffs(fold)
    assert all(T + dt.timedelta(days=30) <= fold for T in cuts)
    Xtr, ytr, wtr = assemble(cuts, s, feats, fold)
    n_train, target_dtype = len(ytr), str(ytr.dtype)
    train_target_hash = sha256_array(np.asarray(ytr))
    params = seed_recipe(seed)
    ds = models.make_datasets("direct", Xtr, ytr, None, {"seed": seed})[0]
    del Xtr, ytr, wtr
    _XY.clear()
    gc.collect()
    booster = models.train_direct_ds(ds, {"seed": seed}, rounds=600)
    Xv, yv = xy(fold, s)
    uid = Xv["user_id"].to_numpy()
    assert np.all(np.diff(uid) > 0)
    Av = to_np(Xv, feats)
    arrays: dict[str, np.ndarray] = {
        "user_id": uid, "cutoff": np.full(len(uid), fold.isoformat(), dtype="U10"),
        "y": np.asarray(yv, dtype=np.float32),
    }
    snapshot_metrics = {}
    for rounds in ROUNDS:
        z = np.maximum(booster.predict(Av, num_iteration=rounds), 0.0).astype(np.float32)
        arrays[f"z_r{rounds}"] = z
        r = evaluate(arrays["y"], z, arrays["cutoff"])
        snapshot_metrics[str(rounds)] = {
            "rmsle_cal": r["fold_cal"][0], "offset": r["per_fold"][0]["offset"],
            "mean_z": r["mean_z"], "prediction_sha256": sha256_array(z),
        }
    paths["model"].parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(paths["model"]), num_iteration=600)
    np.savez_compressed(paths["npz"], **arrays)
    meta = {
        "experiment": EXP_ID, "prefix": PREFIX, "component": component,
        "seed": seed, "fold": fold.isoformat(), "recipe_sha256": recipe_signature(component, seed),
        "recipe": recipe_manifest(component, seed), "train_cutoffs": [v.isoformat() for v in cuts],
        "n_train": n_train, "n_validation": len(uid), "target_dtype_before_cast": target_dtype,
        "train_target_sha256": train_target_hash, "validation_target_sha256": sha256_array(arrays["y"]),
        "validation_row_keys_sha256": sha256_array(row_keys(arrays["cutoff"], uid)),
        "resolved_lightgbm_seeds": _resolved_seeds(booster),
        "only_config_difference_across_seeds": "master seed; child streams deterministically derived",
        "early_stopping": False, "snapshots": snapshot_metrics,
        "artifacts": {"npz": str(paths["npz"].resolve()), "npz_sha256": sha256_file(paths["npz"]),
                      "model": str(paths["model"].resolve()),
                      "model_sha256": sha256_file(paths["model"])},
        "runtime_s": time.time() - t0, "lightgbm_params": params,
    }
    write_json(paths["manifest"], meta)
    del booster, ds, Av, Xv, yv
    _XY.clear()
    gc.collect()
    return meta


def _historical_fold(component: str, fold: dt.date) -> dict[str, np.ndarray]:
    d = np.load(ARTIFACTS / f"oof_{HISTORICAL[component]}.npz", allow_pickle=False)
    mask = np.asarray(d["cutoff"], dtype="U10") == fold.isoformat()
    order = np.argsort(d["user_id"][mask])
    return {k: d[k][mask][order] for k in ("user_id", "z", "y", "cutoff")}


def load_trajectory(component: str, seed: int, fold: dt.date) -> dict[str, np.ndarray]:
    assert trajectory_complete(component, seed, fold)
    d = np.load(trajectory_paths(component, seed, fold)["npz"], allow_pickle=False)
    return {k: d[k] for k in d.files}


def replay_audit() -> dict[str, Any]:
    rows = []
    fold = FOLDS[-1]
    for component in COMPONENTS:
        old = _historical_fold(component, fold)
        new = load_trajectory(component, SEED, fold)
        assert np.array_equal(old["user_id"], new["user_id"])
        assert np.array_equal(old["y"], new["y"])
        z_old = old["z"].astype(np.float64)
        z_new = new["z_r600"].astype(np.float64)
        delta = z_new - z_old
        ro = evaluate(old["y"], z_old, old["cutoff"])
        rn = evaluate(new["y"], z_new, new["cutoff"])
        exact = np.array_equal(old["z"], new["z_r600"])
        row = {
            "component": component, "fold": fold.isoformat(), "exact_prediction_match": exact,
            "max_abs_delta_z": float(np.max(np.abs(delta))), "mae_delta_z": float(np.mean(np.abs(delta))),
            "var_delta_z": float(np.var(delta)), "pearson": float(np.corrcoef(z_old, z_new)[0, 1]),
            "old_prediction_sha256": sha256_array(old["z"]),
            "replay_prediction_sha256": sha256_array(new["z_r600"]),
            "old_rmsle_cal": ro["fold_cal"][0], "replay_rmsle_cal": rn["fold_cal"][0],
            "delta_rmsle_cal": rn["fold_cal"][0] - ro["fold_cal"][0],
        }
        numerical = (abs(row["delta_rmsle_cal"]) <= 1e-5
                     and row["var_delta_z"] <= 1e-10
                     and row["max_abs_delta_z"] <= 1e-5
                     and row["pearson"] >= 0.999999999)
        row["status"] = "PASS_BITWISE" if exact else ("PASS_NUMERICAL" if numerical else "FAIL")
        rows.append(row)
    passed = all(r["status"].startswith("PASS") for r in rows)
    out = {"status": "REPLAY_PASS" if passed else "REPLAY_FAIL", "components": rows,
           "thresholds": {"abs_delta_rmsle_cal": 1e-5, "var_delta_z": 1e-10,
                          "max_abs_delta_z": 1e-5, "pearson": 0.999999999}}
    write_json(RESULTS / "replay_audit.json", out)
    write_csv(RESULTS / "replay_audit.csv", rows)
    return out


def _launch_trajectory(component: str, seed: int, fold: dt.date) -> None:
    if trajectory_complete(component, seed, fold):
        return
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = str(seed)
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--train-one",
                    "--component", component, "--seed", str(seed),
                    "--fold", fold.isoformat()], cwd=ROOT, env=env, check=True)


def assemble_component(component: str, seed: int, rounds: int) -> dict[str, np.ndarray]:
    parts = [load_trajectory(component, seed, fold) for fold in FOLDS]
    out = {"user_id": np.concatenate([p["user_id"] for p in parts]),
           "cutoff": np.concatenate([p["cutoff"] for p in parts]),
           "y": np.concatenate([p["y"] for p in parts]),
           "z": np.concatenate([p[f"z_r{rounds}"] for p in parts])}
    keys = row_keys(out["cutoff"], out["user_id"])
    assert len(keys) == len(np.unique(keys))
    # ``src.blend.aligned`` is the historical canonical order and sorts the
    # string key ``cutoff + user_id``.  Fold artifacts themselves are numeric
    # user_id order, so explicitly apply the same canonical ordering here.
    order = np.argsort(keys)
    out = {name: value[order] for name, value in out.items()}
    return out


def save_npz_once(path: Path, **arrays: np.ndarray) -> str:
    if path.exists():
        old = np.load(path, allow_pickle=False)
        assert set(old.files) == set(arrays), f"artifact schema drift: {path}"
        for key, value in arrays.items():
            assert np.array_equal(old[key], value), f"artifact content drift: {path}:{key}"
    else:
        np.savez_compressed(path, **arrays)
    return sha256_file(path)


def merged_name(component: str, seed: int, rounds: int) -> Path:
    return RUN_DIR / f"{PREFIX}_{component}_S{seed}_R{rounds}.npz"


def metric_row(name: str, z: np.ndarray, y: np.ndarray, cut: np.ndarray) -> dict[str, Any]:
    r = evaluate(y, z, cut)
    return {"name": name, "wcv": r["wcv"], "fold_cal": r["fold_cal"],
            "fold_offsets": [v["offset"] for v in r["per_fold"]],
            "mean_z": r["mean_z"], "auc_positive": auc_positive(y, z),
            "prediction_sha256": sha256_array(np.asarray(z, dtype=np.float32))}


def prediction_diagnostics(name: str, z: np.ndarray, base_name: str, base: np.ndarray,
                           y: np.ndarray) -> dict[str, Any]:
    from scipy.stats import spearmanr
    z = np.asarray(z, dtype=np.float64)
    base = np.asarray(base, dtype=np.float64)
    dz = z - base
    residual = np.log1p(y) - base
    corr_delta_residual = (None if float(np.var(dz)) == 0.0
                           else float(np.corrcoef(dz, residual)[0, 1]))
    return {
        "candidate": name, "baseline": base_name, "var_delta_z": float(np.var(dz)),
        "max_abs_delta_z": float(np.max(np.abs(dz))), "mean_delta_z": float(np.mean(dz)),
        "pearson_prediction": float(np.corrcoef(z, base)[0, 1]),
        "spearman_prediction": float(spearmanr(z, base).statistic),
        "residual_correlation": float(np.corrcoef(np.log1p(y) - z, residual)[0, 1]),
        "corr_delta_with_baseline_residual": corr_delta_residual,
    }


def ensemble(fresh: dict[tuple[str, int, int], np.ndarray], fixed: dict[str, np.ndarray],
             unc: tuple[str, int], cap: tuple[str, int]) -> np.ndarray:
    def part(component: str, spec: tuple[str, int]) -> np.ndarray:
        mode, rounds = spec
        if mode == "OLD":
            return fixed[HISTORICAL[component]]
        if mode == "S42":
            return fresh[(component, SEED, rounds)]
        if mode == "AVG3":
            return np.mean(np.vstack([fresh[(component, s, rounds)] for s in SEEDS]), axis=0)
        raise ValueError(spec)
    return (0.20 * part("UNC", unc) + 0.10 * part("CAP", cap)
            + 0.25 * fixed["S1-DIST"] + 0.225 * fixed["ETX-AVG3"]
            + 0.225 * fixed["SEQ-AVG3"])


def _delta_row(candidate: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    fold = np.asarray(candidate["fold_cal"]) - np.asarray(base["fold_cal"])
    return {"delta_wcv": candidate["wcv"] - base["wcv"], "fold_delta": fold.tolist(),
            "improved_folds": int((fold < 0).sum()), "delta_2025_10_16": float(fold[-1])}


def _wcv_from_fold(values: np.ndarray) -> float:
    return float(np.dot(FOLD_WEIGHTS, np.asarray(values, dtype=np.float64)))


def _fold_not_essential(delta: np.ndarray) -> bool:
    """Gain remains negative after removing any single fold and renormalizing."""
    delta = np.asarray(delta, dtype=np.float64)
    for i in range(4):
        keep = np.arange(4) != i
        w = FOLD_WEIGHTS[keep] / FOLD_WEIGHTS[keep].sum()
        if float(np.dot(w, delta[keep])) >= 0:
            return False
    return True


def decide(metrics: dict[str, dict[str, Any]], replay: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    h, a, d = metrics["H"], metrics["A"], metrics["D"]
    drift = _delta_row(a, h)
    if abs(drift["delta_wcv"]) > 0.0001 or max(abs(v) for v in drift["fold_delta"]) > 0.0002:
        return "TECHNICAL_INCONCLUSIVE", {"reason": "A replay drift exceeds protocol bound",
                                           "a_minus_h": drift}
    ddh, dda = _delta_row(d, h), _delta_row(d, a)
    stable = _fold_not_essential(np.asarray(ddh["fold_delta"]))
    promote_d = (replay["status"] == "REPLAY_PASS" and ddh["delta_wcv"] <= -0.0005
                 and dda["delta_wcv"] < 0 and ddh["improved_folds"] >= 3
                 and ddh["delta_2025_10_16"] < 0 and stable)
    if promote_d:
        return "PROMOTE_COMBINED", {"d_minus_h": ddh, "d_minus_a": dda,
                                     "no_single_fold_creates_gain": stable}
    eligible = []
    for name in ("B", "C"):
        dh, da = _delta_row(metrics[name], h), _delta_row(metrics[name], a)
        if (dh["delta_wcv"] <= -0.0005 and da["delta_wcv"] < 0
                and dh["improved_folds"] >= 3 and dh["delta_2025_10_16"] < 0
                and _fold_not_essential(np.asarray(dh["fold_delta"]))):
            harmful_second = ((_delta_row(d, metrics[name])["delta_wcv"] > 0)
                              if name in ("B", "C") else False)
            if harmful_second:
                eligible.append(name)
    if len(eligible) == 1:
        return "PROMOTE_SINGLE_FACTOR", {"variant": eligible[0]}
    best = min(_delta_row(metrics[n], h)["delta_wcv"] for n in ("B", "C", "D"))
    any_unstable_good = any(_delta_row(metrics[n], h)["delta_wcv"] <= -0.0005
                            for n in ("B", "C", "D"))
    if (-0.0005 < best <= -0.0003) or any_unstable_good:
        return "WEAK_BORDERLINE", {"best_fixed_delta_to_h": best}
    return "REJECT", {"best_fixed_delta_to_h": best}


def analyze() -> dict[str, Any]:
    replay = replay_audit()
    assert replay["status"] == "REPLAY_PASS", "analysis forbidden before replay pass"
    z_h, y, cut, rep_h = historical_baseline()
    Zhist, yh, ch = historical_arrays()
    assert np.array_equal(y, yh) and np.array_equal(cut, ch)
    fixed = {name: Zhist[i].astype(np.float64) for i, name in enumerate(ALL_HISTORICAL)}
    fresh: dict[tuple[str, int, int], np.ndarray] = {}
    standalone_rows: list[dict[str, Any]] = []
    merged_artifacts = []
    for component in COMPONENTS:
        for seed in SEEDS:
            for rounds in ROUNDS:
                d = assemble_component(component, seed, rounds)
                assert np.array_equal(d["cutoff"], cut)
                assert np.array_equal(d["y"], y)
                path = merged_name(component, seed, rounds)
                file_hash = save_npz_once(path, **d)
                z = d["z"].astype(np.float64)
                fresh[(component, seed, rounds)] = z
                row = metric_row(f"{component}_S{seed}_R{rounds}", z, y, cut)
                row.update(component=component, seed=seed, rounds=rounds)
                standalone_rows.append(row)
                merged_artifacts.append({"component": component, "seed": seed,
                                         "rounds": rounds, "path": str(path.resolve()),
                                         "sha256": file_hash})
    # Every component shares exact canonical rows and targets by the assertions above.
    seed_rows = []
    avg3_metrics = []
    for component in COMPONENTS:
        for rounds in ROUNDS:
            z_avg = np.mean(np.vstack([fresh[(component, seed, rounds)] for seed in SEEDS]), axis=0)
            avg_path = RUN_DIR / f"{PREFIX}_{component}_AVG3_R{rounds}.npz"
            avg_hash = save_npz_once(avg_path, user_id=_historical_uid(), cutoff=cut,
                                     y=y.astype(np.float32), z=z_avg.astype(np.float32))
            avg_row = metric_row(f"{component}_AVG3_R{rounds}", z_avg, y, cut)
            avg_row.update(component=component, seed="AVG3", rounds=rounds,
                           artifact=str(avg_path.resolve()), artifact_sha256=avg_hash)
            avg3_metrics.append(avg_row)
            standalone_rows.append(avg_row)
            for scope in ("ALL", *FOLD_LABELS):
                mask = np.ones(len(y), dtype=bool) if scope == "ALL" else cut == scope
                seed_scores = []
                for seed in SEEDS:
                    z_seed = fresh[(component, seed, rounds)][mask]
                    r_seed = evaluate(y[mask], z_seed, cut[mask])
                    seed_scores.append(r_seed["wcv"] if scope == "ALL" else r_seed["fold_cal"][0])
                for i, s1 in enumerate(SEEDS):
                    for s2 in SEEDS[i + 1:]:
                        z1 = fresh[(component, s1, rounds)][mask]
                        z2 = fresh[(component, s2, rounds)][mask]
                        seed_rows.append({
                            "component": component, "rounds": rounds, "scope": scope,
                            "seed_a": s1, "seed_b": s2,
                            "var_delta_z": float(np.var(z1 - z2)),
                            "pearson_prediction": float(np.corrcoef(z1, z2)[0, 1]),
                            "standalone_seed_spread_std": float(np.std(seed_scores, ddof=1)),
                            "standalone_seed_spread_range": float(np.ptp(seed_scores)),
                        })
    recipes = {
        "H": (("OLD", 600), ("OLD", 600)),
        "A": (("S42", 600), ("S42", 600)),
        "B": (("AVG3", 600), ("AVG3", 600)),
        "C": (("S42", 300), ("S42", 300)),
        "D": (("AVG3", 300), ("AVG3", 300)),
    }
    z_variants = {name: (z_h if name == "H" else ensemble(fresh, fixed, unc, cap))
                  for name, (unc, cap) in recipes.items()}
    metrics = {name: metric_row(name, z, y, cut) for name, z in z_variants.items()}
    for name, z in z_variants.items():
        path = RUN_DIR / f"{PREFIX}_ENSEMBLE_{name}.npz"
        save_npz_once(path, user_id=_historical_uid(), cutoff=cut, y=y.astype(np.float32),
                      z=np.asarray(z, dtype=np.float32))
        metrics[name]["artifact"] = str(path.resolve())
        metrics[name]["artifact_sha256"] = sha256_file(path)
    ensemble_rows = []
    for name in recipes:
        row = {"variant": name, "recipe_unc": recipes[name][0], "recipe_cap": recipes[name][1],
               **metrics[name]}
        row.update({f"vs_H_{k}": v for k, v in _delta_row(metrics[name], metrics["H"]).items()})
        row.update({f"vs_A_{k}": v for k, v in _delta_row(metrics[name], metrics["A"]).items()})
        ensemble_rows.append(row)
    contrasts = []
    for candidate, base, label in (("B", "A", "AVG3_only_at_600"),
                                    ("C", "A", "rounds_only_seed42"),
                                    ("D", "B", "rounds_after_AVG3"),
                                    ("D", "C", "AVG3_at_300"),
                                    ("D", "A", "combined")):
        raw = prediction_diagnostics(candidate, z_variants[candidate], base,
                                     z_variants[base], y)
        row = {"contrast": f"{candidate}-{base}", "effect": label,
               **_delta_row(metrics[candidate], metrics[base]),
               "raw_var_delta_z": raw["var_delta_z"],
               "raw_max_abs_delta_z": raw["max_abs_delta_z"],
               "raw_mean_delta_z": raw["mean_delta_z"],
               "raw_pearson_prediction": raw["pearson_prediction"],
               "raw_residual_correlation": raw["residual_correlation"],
               "raw_corr_delta_with_baseline_residual": raw["corr_delta_with_baseline_residual"]}
        contrasts.append(row)
    interaction_fold = (np.asarray(metrics["D"]["fold_cal"])
                        - np.asarray(metrics["B"]["fold_cal"])
                        - np.asarray(metrics["C"]["fold_cal"])
                        + np.asarray(metrics["A"]["fold_cal"]))
    raw_interaction = (z_variants["D"] - z_variants["B"]
                       - z_variants["C"] + z_variants["A"])
    interaction = {"contrast": "D-B-C+A", "effect": "factor_interaction",
                   "delta_wcv": _wcv_from_fold(interaction_fold),
                   "fold_delta": interaction_fold.tolist(),
                   "raw_var_delta_z": float(np.var(raw_interaction)),
                   "raw_max_abs_delta_z": float(np.max(np.abs(raw_interaction))),
                   "raw_mean_delta_z": float(np.mean(raw_interaction)),
                   "raw_corr_with_a_residual": float(np.corrcoef(
                       raw_interaction, np.log1p(y) - z_variants["A"])[0, 1])}
    contrasts.append(interaction)
    attribution = []
    factor_specs = {"B": ("AVG3", 600), "C": ("S42", 300), "D": ("AVG3", 300)}
    for name, spec in factor_specs.items():
        z_unc = ensemble(fresh, fixed, spec, ("OLD", 600))
        z_cap = ensemble(fresh, fixed, ("OLD", 600), spec)
        z_both = z_variants[name]
        mu, mc, mb = (metric_row(f"{name}_{part}", z, y, cut)
                      for part, z in (("UNC_ONLY", z_unc), ("CAP_ONLY", z_cap), ("BOTH", z_both)))
        du, dc, db = (_delta_row(m, metrics["H"]) for m in (mu, mc, mb))
        attribution.append({
            "variant": name, "unc_only_delta_wcv": du["delta_wcv"],
            "cap_only_delta_wcv": dc["delta_wcv"], "both_delta_wcv": db["delta_wcv"],
            "replacement_interaction_wcv": db["delta_wcv"] - du["delta_wcv"] - dc["delta_wcv"],
            "unc_only_fold_delta": du["fold_delta"], "cap_only_fold_delta": dc["fold_delta"],
            "both_fold_delta": db["fold_delta"],
        })
    rounds_rows = []
    for rounds in ROUNDS:
        z = ensemble(fresh, fixed, ("AVG3", rounds), ("AVG3", rounds))
        m = metric_row(f"AVG3_R{rounds}", z, y, cut)
        rounds_rows.append({"rounds": rounds, **m})
    primary_curve = next(r for r in rounds_rows if r["rounds"] == PRIMARY_ROUND)
    for row in rounds_rows:
        row.update(_delta_row(row, primary_curve))
    pred_rows = []
    for name in ("A", "B", "C", "D"):
        pred_rows.append(prediction_diagnostics(name, z_variants[name], "H", z_h, y))
        if name != "A":
            pred_rows.append(prediction_diagnostics(name, z_variants[name], "A", z_variants["A"], y))
    verdict, decision_detail = decide(metrics, replay)
    # Frozen historical inputs must remain byte-identical after all training/analysis.
    base_manifest = json.loads((RESULTS / "baseline_manifest.json").read_text(encoding="utf-8"))
    unchanged = []
    for item in base_manifest["historical_components"]:
        actual = sha256_file(Path(item["path"]))
        assert actual == item["file_sha256"], f"historical artifact changed: {item['name']}"
        unchanged.append({"name": item["name"], "sha256": actual, "unchanged": True})
    summary = {
        "experiment": EXP_ID, "experiment_number": EXP_NUM, "prefix": PREFIX,
        "historical_baseline": metrics["H"], "replay": replay,
        "ensemble_metrics": metrics, "factorial_contrasts": contrasts,
        "component_attribution": attribution, "prediction_diagnostics": pred_rows,
        "rounds_diagnostics": rounds_rows, "seed_diagnostics": seed_rows,
        "component_avg3_metrics": avg3_metrics,
        "merged_oof_artifacts": merged_artifacts, "historical_artifacts_unchanged": unchanged,
        "primary_round": PRIMARY_ROUND, "diagnostic_rounds_not_eligible": [200, 250],
        "test_data_or_predictions_read": False, "submission_created": False,
        "verdict": verdict, "decision_detail": decision_detail,
        "production_experiment_allowed": verdict in ("PROMOTE_COMBINED", "PROMOTE_SINGLE_FACTOR"),
        "dist_avg3_next_experiment_basis": verdict == "WEAK_BORDERLINE",
    }
    write_csv(RESULTS / "standalone_metrics.csv", standalone_rows)
    write_csv(RESULTS / "seed_variance_diagnostics.csv", seed_rows)
    write_csv(RESULTS / "ensemble_folds.csv", ensemble_rows)
    write_csv(RESULTS / "factorial_contrasts.csv", contrasts)
    write_csv(RESULTS / "component_attribution.csv", attribution)
    write_csv(RESULTS / "rounds_diagnostics.csv", rounds_rows)
    write_csv(RESULTS / "prediction_diagnostics.csv", pred_rows)
    write_json(RESULTS / "summary.json", summary)
    return summary


def orchestrate() -> dict[str, Any]:
    phase0_audit()
    # Preregistered reproduction gate: only the latest fold and historical recipe.
    for component in COMPONENTS:
        _launch_trajectory(component, SEED, FOLDS[-1])
    replay = replay_audit()
    if replay["status"] != "REPLAY_PASS":
        raise RuntimeError("replay gate failed; full training matrix was not started")
    print("Replay gate passed", flush=True)
    for component in COMPONENTS:
        for seed in SEEDS:
            for fold in FOLDS:
                _launch_trajectory(component, seed, fold)
            print(f"{component} S{seed} complete", flush=True)
    print("OOF assembly complete", flush=True)
    summary = analyze()
    print("Analysis complete", flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-only", action="store_true")
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--train-one", action="store_true")
    ap.add_argument("--component", choices=COMPONENTS)
    ap.add_argument("--seed", type=int, choices=SEEDS)
    ap.add_argument("--fold", choices=FOLD_LABELS)
    args = ap.parse_args()
    if args.train_one:
        if args.component is None or args.seed is None or args.fold is None:
            ap.error("--train-one requires --component, --seed and --fold")
        train_trajectory(args.component, args.seed, dt.date.fromisoformat(args.fold))
        return
    if args.audit_only:
        phase0_audit()
        print("Baseline reconstructed", flush=True)
        return
    if args.analysis_only:
        analyze()
        print("Analysis complete", flush=True)
        return
    orchestrate()


if __name__ == "__main__":
    main()
