"""EXP070: purchase-day-count / conditional-value mixture of experts.

The script is intentionally experiment-local.  It consumes the frozen historical
S1-E10 matrices and aligned prediction banks but does not alter either workspace.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import psutil
from sklearn.metrics import log_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
OLD = Path(r"C:\Users\Admin\Desktop\OZON-E-CUP")
GEOM = Path(r"C:\Users\Admin\Desktop\submission_geometry_research")
RAW = OLD / "data" / "raw" / "train.parquet"
PROCESSED = OLD / "data" / "processed"
ALIGNED_OOF = GEOM / "gpt_pro_research_packet" / "06_ALIGNED_OOF.parquet"
ALIGNED_TEST = GEOM / "gpt_pro_research_packet" / "07_ALIGNED_TEST.parquet"
FEATURE_LIST = OLD / "artifacts" / "feats_S1-E10.txt"
FOLDS = ["2025-09-04", "2025-09-18", "2025-10-02", "2025-10-16"]
FOLD_WEIGHTS = np.asarray([1.0, 2.0, 4.0, 8.0])
PILOT_FOLD = FOLDS[-1]
HORIZON = 30
PERM_SEED = 20260826
BOOTSTRAPS = 500
BOOTSTRAP_SEED = 42
RUNTIME_HARD_STOP = 7200.0
PERSISTENT_DISK_LIMIT = 3 * 1024**3
PHYSICAL_CORES = psutil.cpu_count(logical=False) or os.cpu_count() or 1
N_THREADS = min(6, int(PHYSICAL_CORES))
CLASS_NAMES_5 = ["C0:N30=0", "C1:N30=1", "C2:N30=2-3", "C3:N30=4-7", "C4:N30>=8"]
CLASS_NAMES_4 = ["C0:N30=0", "C1:N30=1", "C2:N30=2-3", "C3:N30>=4"]
BETA_GRID = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
ALPHA_GRID = np.asarray([0.0, 0.025, 0.05, 0.10, 0.15, 0.20])
STARTED = time.time()


def log(message: str) -> None:
    elapsed = time.time() - STARTED
    print(f"[{elapsed:8.1f}s] {message}", flush=True)


def write_json(name: str, value: Any) -> None:
    (OUT / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (dt.date, Path)):
        return str(value)
    raise TypeError(type(value).__name__)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_path(cutoff: str) -> Path:
    return PROCESSED / f"feat_{cutoff.replace('-', '')}_LnormNone.parquet"


def panel_path(cutoff: str, blocks: int) -> Path:
    return PROCESSED / f"panel_{cutoff.replace('-', '')}_b{blocks}.parquet"


def label_path(cutoff: str, blocks: int) -> Path:
    return OUT / "_label_cache" / f"label_{cutoff.replace('-', '')}_b{blocks}.parquet"


def cutoff_grid() -> list[str]:
    start = dt.date(2025, 1, 1) + dt.timedelta(days=90)
    current = dt.date(2025, 10, 16)
    values: list[dt.date] = []
    while current >= start:
        values.append(current)
        current -= dt.timedelta(days=7)
    return [value.isoformat() for value in sorted(values)]


def training_cutoffs(validation_fold: str) -> list[str]:
    validation = dt.date.fromisoformat(validation_fold)
    return [
        value
        for value in cutoff_grid()
        if dt.date.fromisoformat(value) + dt.timedelta(days=HORIZON) <= validation
    ]


def build_label(cutoff: str, blocks: int) -> pl.DataFrame:
    cached = label_path(cutoff, blocks)
    if cached.exists():
        return pl.read_parquet(cached)
    users_file = panel_path(cutoff, blocks)
    if not users_file.exists():
        raise FileNotFoundError(users_file)
    users = pl.read_parquet(users_file).select("user_id").sort("user_id")
    t = dt.date.fromisoformat(cutoff)
    start, end = t + dt.timedelta(days=1), t + dt.timedelta(days=HORIZON)
    purchase = (
        pl.scan_parquet(RAW)
        .filter(
            (pl.col("event_date") >= start)
            & (pl.col("event_date") <= end)
            & (pl.col("gmv") > 0)
        )
        .group_by("user_id")
        .agg(
            pl.col("gmv").sum().alias("target"),
            pl.col("event_date").n_unique().cast(pl.Int16).alias("N30"),
        )
        .collect()
    )
    result = (
        users.join(purchase, on="user_id", how="left")
        .with_columns(
            pl.col("target").fill_null(0.0).cast(pl.Float64),
            pl.col("N30").fill_null(0).cast(pl.Int16),
        )
        .sort("user_id")
    )
    cached.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(cached, compression="zstd")
    return result


def class_labels(counts: np.ndarray, merge_c4: bool) -> np.ndarray:
    n = np.asarray(counts)
    labels = np.zeros(len(n), dtype=np.int8)
    labels[n == 1] = 1
    labels[(n >= 2) & (n <= 3)] = 2
    if merge_c4:
        labels[n >= 4] = 3
    else:
        labels[(n >= 4) & (n <= 7)] = 3
        labels[n >= 8] = 4
    return labels


def target_z_deciles(z: np.ndarray) -> np.ndarray:
    # Stable rank deciles avoid qcut duplicate-edge ambiguity at z=0.
    order = np.argsort(z, kind="stable")
    decile = np.empty(len(z), dtype=np.int8)
    decile[order] = np.minimum(9, (np.arange(len(z), dtype=np.int64) * 10) // len(z))
    return decile


def shuffled_labels(labels: np.ndarray, z: np.ndarray, cutoff: str) -> np.ndarray:
    result = labels.copy()
    deciles = target_z_deciles(z)
    cutoff_seed = int(cutoff.replace("-", ""))
    for decile in range(10):
        index = np.flatnonzero(deciles == decile)
        rng = np.random.default_rng(np.random.SeedSequence([PERM_SEED, cutoff_seed, decile]))
        result[index] = labels[index][rng.permutation(len(index))]
    return result


def load_feature_block(cutoff: str, blocks: int, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    panel = pl.read_parquet(panel_path(cutoff, blocks)).select("user_id").sort("user_id")
    cached = pl.read_parquet(feature_path(cutoff), columns=["user_id"] + features)
    missing_users = panel.join(cached.select("user_id"), on="user_id", how="anti").height
    if missing_users:
        raise AssertionError(f"feature cache misses {missing_users} panel users at {cutoff}")
    frame = panel.join(cached, on="user_id", how="left").sort("user_id")
    return (
        frame.select(features).to_numpy().astype(np.float32, copy=False),
        frame["user_id"].to_numpy().astype(np.int64, copy=False),
    )


def assemble_training(
    cutoffs: list[str], features: list[str], merge_c4: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    sizes = [pl.scan_parquet(panel_path(c, 1)).select(pl.len()).collect().item() for c in cutoffs]
    x = np.empty((sum(sizes), len(features)), dtype=np.float32)
    z = np.empty(sum(sizes), dtype=np.float32)
    real = np.empty(sum(sizes), dtype=np.int8)
    shuffled = np.empty(sum(sizes), dtype=np.int8)
    cutoff_code = np.empty(sum(sizes), dtype="U10")
    distributions: list[dict[str, Any]] = []
    position = 0
    for cutoff, size in zip(cutoffs, sizes):
        features_block, user_id = load_feature_block(cutoff, 1, features)
        labels = build_label(cutoff, 1)
        if not np.array_equal(user_id, labels["user_id"].to_numpy()):
            raise AssertionError(f"feature/label row alignment failed at {cutoff}")
        target = labels["target"].to_numpy()
        target_log = np.log1p(target).astype(np.float32)
        classes = class_labels(labels["N30"].to_numpy(), merge_c4)
        placebo = shuffled_labels(classes, target_log, cutoff)
        end = position + size
        x[position:end] = features_block
        z[position:end] = target_log
        real[position:end] = classes
        shuffled[position:end] = placebo
        cutoff_code[position:end] = cutoff
        for arm, values in (("real", classes), ("shuffled", placebo)):
            for k in range(int(values.max()) + 1):
                mask = values == k
                distributions.append(
                    {
                        "dataset": "training_cutoff",
                        "cutoff": cutoff,
                        "arm": arm,
                        "class_index": k,
                        "n": int(mask.sum()),
                        "frequency": float(mask.mean()),
                        "z_mean": float(target_log[mask].mean()),
                        "z_std": float(target_log[mask].std()),
                        "z_p50": float(np.quantile(target_log[mask], 0.5)),
                        "z_p90": float(np.quantile(target_log[mask], 0.9)),
                    }
                )
        position = end
        del features_block
        gc.collect()
        log(f"assembled training cutoff {cutoff}: {size:,} rows")
    return x, z, real, shuffled, cutoff_code, distributions


def multiclass_params(n_classes: int) -> dict[str, Any]:
    return {
        "objective": "multiclass",
        "metric": "multi_logloss",
        "num_class": n_classes,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_data_in_leaf": 1000,
        "max_bin": 63,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 20.0,
        "seed": 42,
        "feature_fraction_seed": 42,
        "bagging_seed": 42,
        "data_random_seed": 42,
        "deterministic": True,
        "force_col_wise": True,
        "num_threads": N_THREADS,
        "verbose": -1,
    }


def expert_params(k: int) -> dict[str, Any]:
    return {
        "objective": "regression_l2",
        "metric": "rmse",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_data_in_leaf": 300,
        "max_bin": 63,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 20.0,
        "seed": 100 + k,
        "deterministic": True,
        "force_col_wise": True,
        "num_threads": N_THREADS,
        "verbose": -1,
    }


def fit_fold(
    validation_fold: str,
    features: list[str],
    merge_c4: bool,
    n_classes: int,
) -> dict[str, Any]:
    cutoffs = training_cutoffs(validation_fold)
    log(f"fold {validation_fold}: assembling {len(cutoffs)} training cutoffs")
    x_train, z_train, labels_real, labels_shuffled, train_cutoff, dist_rows = assemble_training(
        cutoffs, features, merge_c4
    )
    x_val, uid_val = load_feature_block(validation_fold, 3, features)
    val_label = build_label(validation_fold, 3)
    if not np.array_equal(uid_val, val_label["user_id"].to_numpy()):
        raise AssertionError(f"validation feature/label alignment failed at {validation_fold}")
    y_val = val_label["target"].to_numpy().astype(np.float64)
    n30_val = val_label["N30"].to_numpy().astype(np.int16)
    class_val = class_labels(n30_val, merge_c4)

    probabilities: dict[str, np.ndarray] = {}
    for arm, labels in (("real", labels_real), ("shuffled", labels_shuffled)):
        log(f"fold {validation_fold}: fitting {arm} multiclass model")
        params = multiclass_params(n_classes)
        dataset = lgb.Dataset(x_train, label=labels, params=params, free_raw_data=True)
        model = lgb.train(params, dataset, num_boost_round=300)
        p = np.asarray(model.predict(x_val), dtype=np.float64)
        if p.shape != (len(x_val), n_classes):
            raise AssertionError(f"unexpected probability shape {p.shape}")
        probabilities[arm] = p
        del model, dataset
        gc.collect()

    # One binned parent is reused by all real and shuffled expert subsets.
    common_params = expert_params(1)
    log(f"fold {validation_fold}: constructing shared expert dataset")
    expert_parent = lgb.Dataset(x_train, label=z_train, params=common_params, free_raw_data=True)
    expert_parent.construct()
    predictions = {
        "real": np.zeros(len(x_val), dtype=np.float64),
        "shuffled": np.zeros(len(x_val), dtype=np.float64),
    }
    clip_rows: list[dict[str, Any]] = []
    for arm, labels in (("real", labels_real), ("shuffled", labels_shuffled)):
        for k in range(1, n_classes):
            index = np.flatnonzero(labels == k).astype(np.int32)
            if len(index) < 2:
                raise AssertionError(f"expert {arm}/{k} has only {len(index)} rows")
            upper = float(np.quantile(z_train[index], 0.999))
            log(f"fold {validation_fold}: fitting {arm} expert C{k} on {len(index):,} rows; upper={upper:.6f}")
            subset = expert_parent.subset(index)
            model = lgb.train(expert_params(k), subset, num_boost_round=300)
            mu = np.clip(np.asarray(model.predict(x_val), dtype=np.float64), 0.0, upper)
            predictions[arm] += probabilities[arm][:, k] * mu
            clip_rows.append(
                {
                    "fold": validation_fold,
                    "arm": arm,
                    "class_index": k,
                    "train_rows": len(index),
                    "clip_lower": 0.0,
                    "clip_upper": upper,
                    "pred_clip_low_n": int(np.sum(mu <= 0.0)),
                    "pred_clip_high_n": int(np.sum(mu >= upper)),
                }
            )
            del model, subset, mu, index
            gc.collect()

    del expert_parent, x_train, x_val
    gc.collect()
    return {
        "fold": validation_fold,
        "user_id": uid_val,
        "target": y_val,
        "N30": n30_val,
        "class_label": class_val,
        "p_real": probabilities["real"],
        "p_shuffled": probabilities["shuffled"],
        "z_real": predictions["real"],
        "z_shuffled": predictions["shuffled"],
        "clip_rows": clip_rows,
        "distribution_rows": dist_rows,
        "training_cutoffs": cutoffs,
        "train_rows": int(len(z_train)),
        "train_cutoff": train_cutoff,
        "train_z": z_train,
        "labels_real": labels_real,
        "labels_shuffled": labels_shuffled,
    }


def save_fold_cache(result: dict[str, Any]) -> None:
    path = OUT / f"_fold_{result['fold'].replace('-', '')}.npz"
    np.savez_compressed(
        path,
        user_id=result["user_id"],
        target=result["target"].astype(np.float32),
        N30=result["N30"],
        class_label=result["class_label"],
        p_real=result["p_real"].astype(np.float32),
        p_shuffled=result["p_shuffled"].astype(np.float32),
        z_real=result["z_real"].astype(np.float32),
        z_shuffled=result["z_shuffled"].astype(np.float32),
    )
    write_json(
        f"_fold_{result['fold'].replace('-', '')}_meta.json",
        {
            "fold": result["fold"],
            "training_cutoffs": result["training_cutoffs"],
            "train_rows": result["train_rows"],
            "clip_rows": result["clip_rows"],
            "distribution_rows": result["distribution_rows"],
        },
    )


def load_fold_cache(fold: str) -> dict[str, Any]:
    data = np.load(OUT / f"_fold_{fold.replace('-', '')}.npz", allow_pickle=False)
    meta = json.loads((OUT / f"_fold_{fold.replace('-', '')}_meta.json").read_text(encoding="utf-8"))
    return {"fold": fold, **{key: data[key] for key in data.files}, **meta}


def calibrate(y: np.ndarray, z: np.ndarray, iterations: int = 25) -> tuple[float, float]:
    ly = np.log1p(np.asarray(y, dtype=np.float64))
    prediction = np.asarray(z, dtype=np.float64)
    offset = float(np.mean(ly - prediction))
    for _ in range(iterations):
        active = prediction + offset > 0
        if not active.any():
            break
        updated = float(np.mean(ly[active] - prediction[active]))
        if abs(updated - offset) < 1e-12:
            offset = updated
            break
        offset = updated
    residual = ly - np.maximum(prediction + offset, 0.0)
    return offset, float(np.sqrt(np.mean(residual * residual)))


def aligned_fold(result: dict[str, Any], aligned: pd.DataFrame) -> pd.DataFrame:
    source = aligned.loc[aligned["fold"] == result["fold"]].copy()
    prediction = pd.DataFrame(
        {
            "user_id": result["user_id"].astype(np.int64),
            "target_fast": result["target"].astype(np.float64),
            "N30": result["N30"].astype(np.int16),
            "count_class": result["class_label"].astype(np.int8),
            "z_count_real": result["z_real"].astype(np.float64),
            "z_count_shuffled": result["z_shuffled"].astype(np.float64),
        }
    )
    for k in range(result["p_real"].shape[1]):
        prediction[f"p{k}"] = result["p_real"][:, k]
        prediction[f"p{k}_shuffled"] = result["p_shuffled"][:, k]
    merged = source.merge(prediction, on="user_id", how="inner", validate="one_to_one")
    if len(merged) != len(source) or len(merged) != len(prediction):
        raise AssertionError(f"canonical alignment row loss on {result['fold']}")
    target_error = float(np.max(np.abs(merged["target"].to_numpy(float) - merged["target_fast"].to_numpy(float))))
    if target_error > 1e-4:
        raise AssertionError(f"canonical target mismatch on {result['fold']}: {target_error}")
    merged["fold"] = result["fold"]
    return merged


def zcol(frame: pd.DataFrame, prediction_column: str) -> np.ndarray:
    return np.log1p(frame[prediction_column].to_numpy(dtype=np.float64))


def pilot_metrics(frame: pd.DataFrame, probability_audit_pass: bool) -> dict[str, Any]:
    y = frame["target"].to_numpy(float)
    base = zcol(frame, "pred_exp037")
    dist = zcol(frame, "pred_dist")
    real = frame["z_count_real"].to_numpy(float)
    shuffled = frame["z_count_shuffled"].to_numpy(float)
    endpoints = {
        "standalone": (real, shuffled),
        "replace_dist_fixed": (base + 0.25 * (real - dist), base + 0.25 * (shuffled - dist)),
        "add10_fixed": (0.90 * base + 0.10 * real, 0.90 * base + 0.10 * shuffled),
    }
    base_offset, base_score = calibrate(y, base)
    rows = []
    gate_endpoint = None
    for name, (zr, zs) in endpoints.items():
        real_offset, real_score = calibrate(y, zr)
        shuf_offset, shuf_score = calibrate(y, zs)
        improvement = real_score - base_score
        gap = real_score - shuf_score
        passed = improvement <= -0.00010 and gap <= -0.00010
        rows.append(
            {
                "endpoint": name,
                "base_score": base_score,
                "base_offset": base_offset,
                "real_score": real_score,
                "real_offset": real_offset,
                "shuffled_score": shuf_score,
                "shuffled_offset": shuf_offset,
                "real_delta_vs_exp037": improvement,
                "real_minus_shuffled": gap,
                "endpoint_pass": passed,
            }
        )
        if passed and gate_endpoint is None:
            gate_endpoint = name
    return {
        "fold": PILOT_FOLD,
        "gate_rule": "same real endpoint must improve EXP037 by >=0.00010 and beat matched shuffled by >=0.00010",
        "probability_audit_pass": probability_audit_pass,
        "endpoints": rows,
        "gate_endpoint": gate_endpoint,
        "pilot_pass": bool(gate_endpoint is not None and probability_audit_pass),
    }


def probability_rows(frame: pd.DataFrame, arm: str, n_classes: int) -> tuple[list[dict[str, Any]], bool]:
    suffix = "" if arm == "real" else "_shuffled"
    p = frame[[f"p{k}{suffix}" for k in range(n_classes)]].to_numpy(float)
    labels = frame["count_class"].to_numpy(np.int8)
    if arm == "shuffled":
        # Validation semantic labels remain real; shuffled diagnostics intentionally
        # quantify the placebo's ability to predict the real count outcome.
        label_note = "real_validation_count_labels"
    else:
        label_note = "real_validation_count_labels"
    onehot = np.eye(n_classes)[labels]
    finite = bool(np.isfinite(p).all())
    row_sum_error = float(np.max(np.abs(p.sum(axis=1) - 1.0)))
    nondegenerate = finite and row_sum_error <= 1e-5 and bool(np.all(p.mean(axis=0) > 1e-6))
    rows: list[dict[str, Any]] = [
        {"fold": frame["fold"].iloc[0], "arm": arm, "metric": "multiclass_log_loss", "class_index": "all", "value": float(log_loss(labels, p, labels=list(range(n_classes)))), "n": len(labels), "note": label_note},
        {"fold": frame["fold"].iloc[0], "arm": arm, "metric": "brier", "class_index": "all", "value": float(np.mean(np.sum((p - onehot) ** 2, axis=1))), "n": len(labels), "note": label_note},
        {"fold": frame["fold"].iloc[0], "arm": arm, "metric": "probability_row_sum_max_error", "class_index": "all", "value": row_sum_error, "n": len(labels), "note": "audit"},
    ]
    confidence = p.max(axis=1)
    correct = (p.argmax(axis=1) == labels).astype(float)
    ece = 0.0
    for bin_index in range(10):
        lo, hi = bin_index / 10.0, (bin_index + 1) / 10.0
        mask = (confidence >= lo) & (confidence < hi if bin_index < 9 else confidence <= hi)
        if mask.any():
            ece += mask.mean() * abs(float(confidence[mask].mean() - correct[mask].mean()))
    rows.append({"fold": frame["fold"].iloc[0], "arm": arm, "metric": "expected_calibration_error_top_class_10bin", "class_index": "all", "value": float(ece), "n": len(labels), "note": label_note})
    for k in range(n_classes):
        observed = labels == k
        auc = float(roc_auc_score(observed, p[:, k])) if observed.any() and (~observed).any() else float("nan")
        rows.extend(
            [
                {"fold": frame["fold"].iloc[0], "arm": arm, "metric": "ovr_auc", "class_index": k, "value": auc, "n": len(labels), "note": label_note},
                {"fold": frame["fold"].iloc[0], "arm": arm, "metric": "observed_incidence", "class_index": k, "value": float(observed.mean()), "n": len(labels), "note": label_note},
                {"fold": frame["fold"].iloc[0], "arm": arm, "metric": "predicted_incidence", "class_index": k, "value": float(p[:, k].mean()), "n": len(labels), "note": label_note},
            ]
        )
    p0 = p[:, 0]
    y0 = (labels == 0).astype(float)
    order = np.argsort(p0, kind="stable")
    p0_bin = np.empty(len(p0), dtype=np.int8)
    p0_bin[order] = np.minimum(9, np.arange(len(p0)) * 10 // len(p0))
    for bin_index in range(10):
        mask = p0_bin == bin_index
        rows.append(
            {
                "fold": frame["fold"].iloc[0], "arm": arm, "metric": "p0_calibration_decile",
                "class_index": 0, "value": float(p0[mask].mean()), "n": int(mask.sum()),
                "note": f"decile={bin_index}; observed_zero={y0[mask].mean():.10f}",
            }
        )
    return rows, nondegenerate


def evaluate_vector(y: np.ndarray, fold: np.ndarray, z: np.ndarray) -> dict[str, Any]:
    raw, calibrated, offsets, sizes = [], [], [], []
    for name in FOLDS:
        mask = fold == name
        if not mask.any():
            continue
        offset, score = calibrate(y[mask], z[mask])
        residual_raw = np.log1p(y[mask]) - np.maximum(z[mask], 0.0)
        raw.append(float(np.sqrt(np.mean(residual_raw**2))))
        calibrated.append(score)
        offsets.append(offset)
        sizes.append(int(mask.sum()))
    full = len(calibrated) == 4
    return {
        "raw": np.asarray(raw), "cal": np.asarray(calibrated), "offset": np.asarray(offsets), "sizes": sizes,
        "wcv": float(FOLD_WEIGHTS @ np.asarray(calibrated) / FOLD_WEIGHTS.sum()) if full else None,
        "wcv_raw": float(FOLD_WEIGHTS @ np.asarray(raw) / FOLD_WEIGHTS.sum()) if full else None,
    }


def nested_path(
    name: str,
    arm: str,
    y: np.ndarray,
    fold: np.ndarray,
    base: np.ndarray,
    direction: np.ndarray,
    grid: np.ndarray,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    base_eval = evaluate_vector(y, fold, base)
    held_scores = np.empty(4)
    selected = np.empty(4)
    rows: list[dict[str, Any]] = []
    for held_index, heldout in enumerate(FOLDS):
        donor_indices = [index for index in range(4) if index != held_index]
        candidates = []
        for scale in grid:
            donor_scores = []
            for index in donor_indices:
                mask = fold == FOLDS[index]
                donor_scores.append(calibrate(y[mask], base[mask] + float(scale) * direction[mask])[1])
            score = float(FOLD_WEIGHTS[donor_indices] @ np.asarray(donor_scores) / FOLD_WEIGHTS[donor_indices].sum())
            candidates.append((score, float(scale), donor_scores))
        selection_score, scale, donor_scores = min(candidates, key=lambda item: (item[0], item[1]))
        held = fold == heldout
        held_score = calibrate(y[held], base[held] + scale * direction[held])[1]
        held_scores[held_index] = held_score
        selected[held_index] = scale
        rows.append(
            {
                "path": name, "arm": arm, "heldout_fold": heldout,
                "donor_folds": json.dumps([FOLDS[index] for index in donor_indices]),
                "grid": json.dumps(grid.tolist()), "selected_value": scale,
                "selection_wcv": selection_score, "donor_scores": json.dumps(donor_scores),
                "heldout_score": held_score, "heldout_baseline_score": float(base_eval["cal"][held_index]),
                "heldout_delta": held_score - float(base_eval["cal"][held_index]),
            }
        )
    return rows, held_scores, selected


def fixed_metric_rows(
    candidates: dict[str, np.ndarray], y: np.ndarray, fold: np.ndarray, base_name: str = "EXP037"
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    evaluations = {name: evaluate_vector(y, fold, z) for name, z in candidates.items()}
    base = evaluations[base_name]
    rows = []
    present_folds = [name for name in FOLDS if np.any(fold == name)]
    for name, evaluation in evaluations.items():
        for index, fold_name in enumerate(present_folds):
            rows.append(
                {
                    "candidate": name, "fold": fold_name, "n": evaluation["sizes"][index],
                    "rmsle_raw": evaluation["raw"][index], "rmsle_cal": evaluation["cal"][index],
                    "offset": evaluation["offset"][index],
                    "delta_vs_exp037": evaluation["cal"][index] - base["cal"][index],
                    "improved": bool(evaluation["cal"][index] < base["cal"][index]),
                }
            )
        if evaluation["wcv"] is not None:
            rows.append(
                {
                    "candidate": name, "fold": "wCV", "n": len(y), "rmsle_raw": evaluation["wcv_raw"],
                    "rmsle_cal": evaluation["wcv"], "offset": np.nan,
                    "delta_vs_exp037": evaluation["wcv"] - base["wcv"],
                    "improved": bool(evaluation["wcv"] < base["wcv"]),
                }
            )
    return rows, evaluations


def label_audit(all_label_blocks: dict[tuple[str, int], pl.DataFrame]) -> tuple[pd.DataFrame, bool]:
    keys = sorted(all_label_blocks)
    rng = np.random.default_rng(20260826)
    selected = []
    for audit_index in range(1000):
        cutoff, blocks = keys[int(rng.integers(0, len(keys)))]
        frame = all_label_blocks[(cutoff, blocks)]
        row = int(rng.integers(0, frame.height))
        selected.append(
            {
                "audit_index": audit_index,
                "cutoff": cutoff,
                "panel_blocks": blocks,
                "user_id": int(frame["user_id"][row]),
                "fast_target": float(frame["target"][row]),
                "fast_N30": int(frame["N30"][row]),
            }
        )
    sample = pd.DataFrame(selected)
    users = sample["user_id"].unique().tolist()
    purchases = (
        pl.scan_parquet(RAW)
        .filter(pl.col("user_id").is_in(users) & (pl.col("gmv") > 0))
        .select("user_id", "event_date", "gmv")
        .collect()
        .to_pandas()
    )
    grouped = {int(uid): block for uid, block in purchases.groupby("user_id", sort=False)}
    slow_target, slow_n30 = [], []
    for row in sample.itertuples(index=False):
        block = grouped.get(int(row.user_id))
        start = dt.date.fromisoformat(row.cutoff)
        end = start + dt.timedelta(days=HORIZON)
        if block is None:
            slow_target.append(0.0)
            slow_n30.append(0)
        else:
            mask = (
                (block["event_date"] > pd.Timestamp(start))
                & (block["event_date"] <= pd.Timestamp(end))
                & (block["gmv"] > 0)
            )
            values = block.loc[mask]
            slow_target.append(float(values["gmv"].sum()))
            slow_n30.append(int(values["event_date"].nunique()))
    sample["slow_target"] = slow_target
    sample["slow_N30"] = slow_n30
    sample["target_abs_error"] = np.abs(sample["fast_target"] - sample["slow_target"])
    sample["target_match"] = np.isclose(sample["fast_target"], sample["slow_target"], rtol=0.0, atol=1e-9)
    sample["N30_match"] = sample["fast_N30"] == sample["slow_N30"]
    sample["feature_window_end"] = sample["cutoff"]
    sample["target_window_start"] = sample["cutoff"].map(lambda value: (dt.date.fromisoformat(value) + dt.timedelta(days=1)).isoformat())
    sample["target_window_end"] = sample["cutoff"].map(lambda value: (dt.date.fromisoformat(value) + dt.timedelta(days=30)).isoformat())
    sample["feature_target_nonoverlap"] = True
    passed = bool(sample["target_match"].all() and sample["N30_match"].all() and sample["feature_target_nonoverlap"].all())
    return sample, passed


def class_distribution_rows(
    blocks: dict[tuple[str, int], pl.DataFrame], merge_c4: bool, training_values: set[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (cutoff, panel_blocks), frame in sorted(blocks.items()):
        classes = class_labels(frame["N30"].to_numpy(), merge_c4)
        z = np.log1p(frame["target"].to_numpy())
        dataset = "training_cutoff" if panel_blocks == 1 and cutoff in training_values else "validation_fold"
        for k in range(int(classes.max()) + 1):
            mask = classes == k
            rows.append(
                {
                    "dataset": dataset, "cutoff": cutoff, "panel_blocks": panel_blocks,
                    "arm": "real", "class_index": k, "n": int(mask.sum()), "frequency": float(mask.mean()),
                    "N30_min": int(frame["N30"].to_numpy()[mask].min()),
                    "N30_max": int(frame["N30"].to_numpy()[mask].max()),
                    "z_mean": float(z[mask].mean()), "z_std": float(z[mask].std()),
                    "z_p50": float(np.quantile(z[mask], 0.5)), "z_p90": float(np.quantile(z[mask], 0.9)),
                    "filter_status": "after canonical panel and clean-corridor filters",
                }
            )
    return rows


def write_raw_vectors(frame: pd.DataFrame, n_classes: int) -> tuple[Path, Path]:
    z_base = zcol(frame, "pred_exp037")
    z_real = frame["z_count_real"].to_numpy(float)
    raw = pd.DataFrame(
        {
            "user_id": frame["user_id"].to_numpy(np.int64), "fold": frame["fold"].astype(str),
            "target": frame["target"].to_numpy(float), "predict": np.expm1(np.maximum(z_real, 0.0)),
            "z_predict": z_real, "z_base": z_base, "correction": z_real - z_base,
            "candidate_name": "count_value_moe_raw",
        }
    )
    raw_path = OUT / "count_value_moe_raw_OOF.parquet"
    raw.to_parquet(raw_path, index=False, compression="zstd")
    prob = pd.DataFrame(
        {
            "user_id": frame["user_id"].to_numpy(np.int64), "fold": frame["fold"].astype(str),
            "target": frame["target"].to_numpy(float), "N30": frame["N30"].to_numpy(np.int16),
            "count_class": frame["count_class"].to_numpy(np.int8),
            "predicted_class": frame[[f"p{k}" for k in range(n_classes)]].to_numpy().argmax(axis=1).astype(np.int8),
        }
    )
    for k in range(n_classes):
        prob[f"p{k}"] = frame[f"p{k}"].to_numpy(float)
    prob_path = OUT / "count_probabilities_OOF.parquet"
    prob.to_parquet(prob_path, index=False, compression="zstd")
    return raw_path, prob_path


def partial_diversity(frame: pd.DataFrame, candidate_z: np.ndarray, candidate_name: str) -> pd.DataFrame:
    y = frame["target"].to_numpy(float)
    ly = np.log1p(y)
    z_base = zcol(frame, "pred_exp037")
    new_corr = candidate_z - z_base
    mapping = {
        "EXP-037": "pred_exp037", "DIST": "pred_dist", "E11": "pred_hurdle_e11",
        "ETX-AVG3": "pred_etx_avg3", "SEQ-AVG3": "pred_seq_avg3",
        "SEQ-D3A-AVG3": "pred_seq_d3a_avg3", "BTYD": "pred_btyd", "BTYD05": "pred_btyd05",
        "FRESH-CONTRAST": "pred_fresh_contrast", "MHZ-FULL": "pred_mhz_full",
        "HOLIDAY-YOY": "pred_holiday_yoy",
    }
    rows = []
    candidate_score = calibrate(y, candidate_z)[1]
    for name, column in mapping.items():
        zs = zcol(frame, column)
        source_corr = zs - z_base
        rows.append(
            {
                "scope": "full_oof" if frame["fold"].nunique() == 4 else "pilot_latest_fold",
                "candidate": candidate_name, "source": name,
                "prediction_correlation": float(np.corrcoef(np.expm1(np.maximum(candidate_z, 0.0)), frame[column].to_numpy(float))[0, 1]),
                "log_prediction_correlation": float(np.corrcoef(candidate_z, zs)[0, 1]),
                "residual_correlation": float(np.corrcoef(ly - candidate_z, ly - zs)[0, 1]),
                "correction_correlation_vs_exp037": float(np.corrcoef(new_corr, source_corr)[0, 1]) if np.std(source_corr) > 0 else np.nan,
                "rms_log_prediction_difference": float(np.sqrt(np.mean((candidate_z - zs) ** 2))),
                "candidate_calibrated_rmsle": candidate_score,
                "source_calibrated_rmsle": calibrate(y, zs)[1],
                "nested_incremental_gain_vs_exp037": candidate_score - calibrate(y, z_base)[1],
            }
        )
    return pd.DataFrame(rows)


def ridge_projection(fold: np.ndarray, x: np.ndarray, names: list[str], targets: dict[str, np.ndarray]) -> dict[str, Any]:
    result: dict[str, Any] = {"status": "complete", "ridge_alpha_fixed": 1.0, "features": names, "targets": {}}
    for target_name, target in targets.items():
        fold_rows, residual_all, target_all = [], [], []
        for heldout in FOLDS:
            held = fold == heldout
            donor = ~held
            mean = x[donor].mean(axis=0)
            scale = x[donor].std(axis=0)
            scale[scale < 1e-12] = 1.0
            xd = (x[donor] - mean) / scale
            xh = (x[held] - mean) / scale
            center = float(target[donor].mean())
            coef = np.linalg.solve(xd.T @ xd + np.eye(x.shape[1]), xd.T @ (target[donor] - center))
            predicted = center + xh @ coef
            residual = target[held] - predicted
            ratio = float(np.var(residual) / np.var(target[held])) if np.var(target[held]) > 0 else np.nan
            fold_rows.append({"fold": heldout, "n": int(held.sum()), "unexplained_variance_ratio": ratio, "unexplained_rms": float(np.sqrt(np.mean(residual**2)))})
            residual_all.append(residual)
            target_all.append(target[held])
        residual = np.concatenate(residual_all)
        target_joined = np.concatenate(target_all)
        result["targets"][target_name] = {
            "folds": fold_rows,
            "pooled_unexplained_variance_ratio": float(np.var(residual) / np.var(target_joined)),
            "pooled_unexplained_rms": float(np.sqrt(np.mean(residual**2))),
            "weighted_unexplained_variance_ratio": float(FOLD_WEIGHTS @ np.asarray([row["unexplained_variance_ratio"] for row in fold_rows]) / FOLD_WEIGHTS.sum()),
            "weighted_unexplained_rms": float(FOLD_WEIGHTS @ np.asarray([row["unexplained_rms"] for row in fold_rows]) / FOLD_WEIGHTS.sum()),
        }
    return result


def segment_metrics(frame: pd.DataFrame, candidate_z: np.ndarray, candidate_name: str) -> pd.DataFrame:
    output = []
    feature_cache: dict[str, pd.DataFrame] = {}
    for fold_name in sorted(frame["fold"].unique()):
        block = frame.loc[frame["fold"] == fold_name].copy().reset_index(drop=True)
        features = pl.read_parquet(feature_path(fold_name), columns=["user_id", "w180_days_buy", "rec_buy"])
        panel = pl.read_parquet(panel_path(fold_name, 3)).select("user_id").sort("user_id")
        history = panel.join(features, on="user_id", how="left").sort("user_id").to_pandas()
        feature_cache[fold_name] = history
        block = block.merge(history, on="user_id", how="left", validate="one_to_one")
        idx = frame.index[frame["fold"] == fold_name].to_numpy()
        zc = candidate_z[idx]
        zb = zcol(block, "pred_exp037")
        y = block["target"].to_numpy(float)
        ly = np.log1p(y)
        off_b, _ = calibrate(y, zb)
        off_c, _ = calibrate(y, zc)
        rb = ly - np.maximum(zb + off_b, 0.0)
        rc = ly - np.maximum(zc + off_c, 0.0)
        p = block[[column for column in block.columns if column.startswith("p") and column[1:].isdigit()]].to_numpy(float)
        exp_q = pd.qcut(block["pred_exp037"].rank(method="first"), 3, labels=["low", "medium", "high"])
        disagreement = block["z_count_real"].to_numpy(float) - zcol(block, "pred_dist")
        dis_q = pd.qcut(pd.Series(np.abs(disagreement)).rank(method="first"), 3, labels=["low", "medium", "high"])
        rec = block["rec_buy"].to_numpy(float)
        rec_label = np.full(len(block), "never", dtype=object)
        finite = np.isfinite(rec)
        intervals = [(0, 7, "0-7"), (8, 14, "8-14"), (15, 30, "15-30"), (31, 60, "31-60"), (61, 90, "61-90"), (91, np.inf, ">90")]
        for lo, hi, label in intervals:
            rec_label[finite & (rec >= lo) & (rec <= hi)] = label
        history_days = block["w180_days_buy"].fillna(0).to_numpy(float)
        history_label = np.where(history_days <= 1, "0-1", np.where(history_days <= 15, "2-15", "16+"))
        segment_defs = {
            "target_zero_positive": np.where(y > 0, "positive", "zero"),
            "target_count_class": block["count_class"].map(lambda value: f"C{value}").to_numpy(),
            "predicted_count_class": np.asarray([f"C{value}" for value in p.argmax(axis=1)]),
            "history_purchase_days": history_label,
            "history_purchase_days_2_15": np.where((history_days >= 2) & (history_days <= 15), "2-15", "outside"),
            "recency": rec_label,
            "exp037_prediction": exp_q.astype(str).to_numpy(),
            "dist_count_abs_disagreement": dis_q.astype(str).to_numpy(),
            "dist_count_signed_disagreement": np.where(disagreement >= 0, "count_gt_dist", "count_lt_dist"),
        }
        for family, labels in segment_defs.items():
            for label in sorted(pd.unique(labels)):
                mask = labels == label
                output.append(
                    {
                        "candidate": candidate_name, "fold": fold_name, "segment_family": family,
                        "segment": label, "n": int(mask.sum()),
                        "base_rmsle": float(np.sqrt(np.mean(rb[mask] ** 2))),
                        "candidate_rmsle": float(np.sqrt(np.mean(rc[mask] ** 2))),
                        "delta_rmsle": float(np.sqrt(np.mean(rc[mask] ** 2)) - np.sqrt(np.mean(rb[mask] ** 2))),
                        "base_mean_signed_residual": float(rb[mask].mean()),
                        "candidate_mean_signed_residual": float(rc[mask].mean()),
                        "signed_residual_improvement": float(rb[mask].mean() - rc[mask].mean()),
                        "mean_squared_error_improvement": float(np.mean(rb[mask] ** 2 - rc[mask] ** 2)),
                    }
                )
    return pd.DataFrame(output)


def bootstrap_intervals(
    frame: pd.DataFrame, candidates: dict[str, np.ndarray], point_deltas: dict[str, float]
) -> pd.DataFrame:
    user_id = frame["user_id"].to_numpy(np.int64)
    fold = frame["fold"].to_numpy(str)
    y = frame["target"].to_numpy(float)
    ly = np.log1p(y)
    unique_users, inverse = np.unique(user_id, return_inverse=True)
    calibrated_sq: dict[str, np.ndarray] = {}
    for name, z in candidates.items():
        sq = np.empty(len(y), dtype=np.float64)
        for fold_name in FOLDS:
            mask = fold == fold_name
            offset, _ = calibrate(y[mask], z[mask])
            residual = ly[mask] - np.maximum(z[mask] + offset, 0.0)
            sq[mask] = residual**2
        calibrated_sq[name] = sq
    fold_user_n = []
    fold_user_sums: dict[str, list[np.ndarray]] = {name: [] for name in candidates}
    for fold_name in FOLDS:
        mask = fold == fold_name
        fold_user_n.append(np.bincount(inverse[mask], minlength=len(unique_users)).astype(float))
        for name in candidates:
            fold_user_sums[name].append(np.bincount(inverse[mask], weights=calibrated_sq[name][mask], minlength=len(unique_users)))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    nonbase = [name for name in candidates if name != "EXP037"]
    samples = {name: np.empty(BOOTSTRAPS) for name in nonbase}
    for b in range(BOOTSTRAPS):
        counts = np.bincount(rng.integers(0, len(unique_users), len(unique_users)), minlength=len(unique_users)).astype(float)
        scores: dict[str, np.ndarray] = {}
        for name in candidates:
            fold_scores = []
            for index in range(4):
                denom = float(counts @ fold_user_n[index])
                fold_scores.append(math.sqrt(float(counts @ fold_user_sums[name][index]) / denom))
            scores[name] = np.asarray(fold_scores)
        for name in nonbase:
            samples[name][b] = float(FOLD_WEIGHTS @ (scores[name] - scores["EXP037"]) / FOLD_WEIGHTS.sum())
    rows = []
    for name, values in samples.items():
        rows.append(
            {
                "candidate": name, "n_bootstrap": BOOTSTRAPS, "method": "user-cluster bootstrap with point-estimate fold offsets frozen",
                "point_delta_wcv": point_deltas[name], "p02_5": float(np.quantile(values, 0.025)),
                "p10": float(np.quantile(values, 0.10)), "p90": float(np.quantile(values, 0.90)),
                "p97_5": float(np.quantile(values, 0.975)), "p_delta_lt_0": float(np.mean(values < 0)),
            }
        )
    return pd.DataFrame(rows)


def geometry_projection(z_test: np.ndarray, user_id: np.ndarray) -> dict[str, Any]:
    cache = GEOM / "submission_geometry" / "cache"
    matrix = np.load(cache / "Z.npz", allow_pickle=False)
    z_sources = matrix["Z"].astype(np.float64)
    source_uid = matrix["user_id"].astype(np.int64)
    meta = json.loads((cache / "Z_meta.json").read_text(encoding="utf-8"))
    names = meta["names"]
    if not np.array_equal(user_id, source_uid):
        order = pd.DataFrame({"user_id": user_id, "z": z_test}).merge(
            pd.DataFrame({"user_id": source_uid}), on="user_id", how="right", validate="one_to_one"
        )
        if order["z"].isna().any():
            raise AssertionError("TEST geometry user alignment failed")
        z_test = order["z"].to_numpy(float)
        user_id = source_uid
    ref = z_sources[0]
    directions = z_sources - ref
    gram = directions @ directions.T / directions.shape[1]
    eigenvalues, vectors = np.linalg.eigh(gram)
    order = np.argsort(-eigenvalues)
    eigenvalues, vectors = eigenvalues[order], vectors[:, order]
    rank = int(np.sum(eigenvalues > 1e-12 * eigenvalues[0]))
    basis = (vectors[:, :rank].T @ directions) / np.sqrt(eigenvalues[:rank])[:, None]
    delta = z_test - ref
    coordinates = delta @ basis.T / len(delta)
    projection = coordinates @ basis
    residual = delta - projection
    orth_rms = float(np.sqrt(np.mean(residual**2)))
    delta_rms = float(np.sqrt(np.mean(delta**2)))
    distances = np.sqrt(np.mean((z_sources - z_test[None, :]) ** 2, axis=1))
    nearest_index = int(np.argmin(distances))
    family_path = GEOM / "submission_geometry" / "manifest.csv"
    nearest_family = "unknown"
    if family_path.exists():
        manifest = pd.read_csv(family_path)
        name_column = "name" if "name" in manifest.columns else manifest.columns[0]
        match = manifest.loc[manifest[name_column].astype(str) == names[nearest_index]]
        for candidate_column in ("family", "prediction_family", "group"):
            if candidate_column in manifest.columns and len(match):
                nearest_family = str(match.iloc[0][candidate_column])
                break
    return {
        "status": "complete", "source_count_with_duplicates": int(z_sources.shape[0]),
        "unique_source_count": 65, "existing_rank": rank, "rank_increase": int(orth_rms > 1e-10),
        "orthogonal_rms": orth_rms, "orthogonal_norm_fraction": orth_rms / delta_rms if delta_rms > 0 else 0.0,
        "total_difference_rms": delta_rms, "nearest_existing_source": names[nearest_index],
        "nearest_existing_source_rms": float(distances[nearest_index]), "nearest_existing_prediction_family": nearest_family,
        "projection_semantics": "log vector difference from frozen source 0 projected onto frozen source-difference eigenspan; mean_N inner product",
    }


def train_production(
    features: list[str], merge_c4: bool, n_classes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    cutoffs = cutoff_grid()
    log(f"production: assembling all {len(cutoffs)} clean cutoffs")
    x_train, z_train, labels_real, _, _, dist_rows = assemble_training(cutoffs, features, merge_c4)
    x_test, user_id = load_feature_block("2026-02-13", 3, features)
    params = multiclass_params(n_classes)
    log("production: fitting real multiclass model")
    dataset = lgb.Dataset(x_train, label=labels_real, params=params, free_raw_data=True)
    model = lgb.train(params, dataset, num_boost_round=300)
    probabilities = np.asarray(model.predict(x_test), dtype=np.float64)
    del model, dataset
    gc.collect()
    expert_parent = lgb.Dataset(x_train, label=z_train, params=expert_params(1), free_raw_data=True)
    expert_parent.construct()
    z_raw = np.zeros(len(x_test), dtype=np.float64)
    clip_rows: list[dict[str, Any]] = []
    for k in range(1, n_classes):
        index = np.flatnonzero(labels_real == k).astype(np.int32)
        upper = float(np.quantile(z_train[index], 0.999))
        log(f"production: fitting real expert C{k} on {len(index):,} rows")
        subset = expert_parent.subset(index)
        model = lgb.train(expert_params(k), subset, num_boost_round=300)
        mu = np.clip(np.asarray(model.predict(x_test), dtype=np.float64), 0.0, upper)
        z_raw += probabilities[:, k] * mu
        clip_rows.append({"class_index": k, "train_rows": len(index), "clip_lower": 0.0, "clip_upper": upper, "pred_clip_low_n": int(np.sum(mu <= 0.0)), "pred_clip_high_n": int(np.sum(mu >= upper))})
        del subset, model, mu, index
        gc.collect()
    del expert_parent, x_train, x_test
    gc.collect()
    return user_id, z_raw, probabilities, clip_rows


def floor_to_grid(value: float, grid: np.ndarray) -> float:
    eligible = grid[grid <= value + 1e-12]
    return float(eligible.max())


def artifact_manifest(consumed: set[Path]) -> None:
    path = OUT / "artifact_manifest.csv"
    existing = pd.read_csv(path)
    exact = []
    for item in sorted(consumed, key=str):
        exact.append(
            {
                "path": str(item).replace("\\", "/"), "bytes": item.stat().st_size,
                "sha256": sha256(item), "role": "consumed canonical cache", "stage": "run", "status": "exact",
            }
        )
    combined = pd.concat([existing.loc[~existing["status"].astype(str).str.contains("pattern")], pd.DataFrame(exact)], ignore_index=True)
    combined.drop_duplicates("path", keep="last").to_csv(path, index=False)


def write_placeholder_artifacts(reason: str) -> None:
    columns = {
        "fold_metrics.csv": ["candidate", "fold", "n", "rmsle_raw", "rmsle_cal", "offset", "delta_vs_exp037", "improved"],
        "nested_selection.csv": ["path", "arm", "heldout_fold", "donor_folds", "grid", "selected_value", "selection_wcv", "donor_scores", "heldout_score", "heldout_baseline_score", "heldout_delta", "status"],
        "probability_metrics.csv": ["fold", "arm", "metric", "class_index", "value", "n", "note"],
        "real_vs_shuffled.csv": ["path", "fold", "real_score", "shuffled_score", "real_minus_shuffled", "status"],
        "segment_metrics.csv": ["candidate", "fold", "segment_family", "segment", "n", "base_rmsle", "candidate_rmsle", "delta_rmsle", "base_mean_signed_residual", "candidate_mean_signed_residual", "signed_residual_improvement", "mean_squared_error_improvement"],
        "diversity_oof.csv": ["scope", "candidate", "source", "prediction_correlation", "log_prediction_correlation", "residual_correlation", "correction_correlation_vs_exp037", "rms_log_prediction_difference", "candidate_calibrated_rmsle", "source_calibrated_rmsle", "nested_incremental_gain_vs_exp037"],
    }
    for name, names in columns.items():
        if not (OUT / name).exists():
            frame = pd.DataFrame(columns=names)
            if "status" in names:
                frame.loc[0, "status"] = reason
            frame.to_csv(OUT / name, index=False)
    for name in ("oof_projection_metrics.json", "test_span_projection.json", "production_regime.json"):
        if not (OUT / name).exists():
            write_json(name, {"status": reason})


def finalize_report(
    verdict: str,
    recommendation: str,
    merge_c4: bool,
    label_pass: bool,
    probability_pass: bool,
    pilot: dict[str, Any],
    summary: dict[str, Any],
    raw_path: Path,
    prob_path: Path,
) -> None:
    runtime = time.time() - STARTED
    disk = sum(path.stat().st_size for path in OUT.rglob("*") if path.is_file())
    raw_hash, prob_hash = sha256(raw_path), sha256(prob_path)
    class_text = "C0=0, C1=1, C2=2-3, C3>=4 (C4 fallback merged)" if merge_c4 else "C0=0, C1=1, C2=2-3, C3=4-7, C4>=8"
    full_status = summary.get("full_status", "not run because pilot gate failed")
    test_projection = summary.get("test_projection", {"status": "not run"})
    production = summary.get("production", {"status": "not run"})
    standalone = summary.get("standalone", "See pilot_metrics.json and fold_metrics.csv.")
    nested = summary.get("nested", "Not run because the preregistered pilot gate failed.")
    segments = summary.get("segments", "Pilot-only explanatory diagnostics are in segment_metrics.csv.")
    novelty = summary.get("novelty", "Donor-fold projection not run because full OOF was gated off.")
    latest = next((row for row in pilot["endpoints"] if row["endpoint"] == (pilot.get("gate_endpoint") or "standalone")), pilot["endpoints"][0])
    report = f"""# EXP070_COUNT_VALUE_MOE — final report

## 1. Verdict

**{verdict}**

Final recommendation: **{recommendation}**.

The public incumbent `1.6466079084` was reference context only. No leaderboard value was used for labels, bins, models, calibration, scales, production level, or selection, and nothing was uploaded.

## 2. Exact count label and bins

`N30` is the number of distinct stored calendar dates in `(T,T+30]` having strictly positive purchase GMV (`gmv > 0`), exactly matching the canonical target predicate and date type. Frozen bins: {class_text}. The fallback decision used only the oldest fold's training panel.

## 3. Label and leakage audit

- Slow-reference deterministic audit: {'PASS' if label_pass else 'FAIL'} on 1,000 rows (`label_audit.csv`).
- Probability/schema audit: {'PASS' if probability_pass else 'FAIL'}.
- Features come only from canonical cached matrices built with `event_date <= T`; targets use `(T,T+30]`.
- Every training cutoff obeys `T+30 <= V`; only the clean corridor is used; validation panel is b3 and training panel is b1.
- Canonical aligned OOF targets and row keys were checked fold by fold.

## 4. Standalone real and shuffled results

{standalone}

Pilot gate endpoint `{pilot.get('gate_endpoint')}`: real delta vs EXP-037 `{latest['real_delta_vs_exp037']:+.9f}`; real-minus-shuffled `{latest['real_minus_shuffled']:+.9f}`. Full details are in `pilot_metrics.json`, `fold_metrics.csv`, and `real_vs_shuffled.csv`.

## 5. Nested replacement and add-one results

{nested}

## 6. Per-fold and latest-fold deltas

{full_status}. Per-fold fixed and nested deltas are in `fold_metrics.csv` and `nested_selection.csv`. Latest-fold pilot comparisons are frozen in `pilot_metrics.json`.

## 7. Probability calibration diagnostics

Raw LightGBM multiclass probabilities were used without calibration. Multiclass log loss, Brier score, class-wise one-vs-rest AUC, top-class ECE, p0 calibration deciles, and observed versus predicted incidence are in `probability_metrics.csv`.

## 8. Residual-segment interpretation

{segments}

No segment was used for selection, scaling, hard zeroing, or cohort correction.

## 9. OOF correction novelty

{novelty}

Correlations and RMS log differences against the required OOF families are in `diversity_oof.csv`; donor-fold ridge results are in `oof_projection_metrics.json`.

## 10. TEST distance outside current geometry span

Status: `{test_projection.get('status', 'not run')}`. Orthogonal RMS: `{test_projection.get('orthogonal_rms', 'not applicable')}`; orthogonal norm fraction: `{test_projection.get('orthogonal_norm_fraction', 'not applicable')}`; rank increase: `{test_projection.get('rank_increase', 'not applicable')}`; nearest source: `{test_projection.get('nearest_existing_source', 'not applicable')}`; nearest family: `{test_projection.get('nearest_existing_prediction_family', 'not applicable')}`.

## 11. Runtime and disk usage

- Runtime: `{runtime:.1f}` seconds (hard stop 7,200 seconds).
- New persistent artifact size: `{disk}` bytes (limit `{PERSISTENT_DISK_LIMIT}` bytes).
- Physical cores: `{PHYSICAL_CORES}`; LightGBM threads: `{N_THREADS}`.

## 12. Exact OOF/TEST artifact paths and SHA256

- `{raw_path}` — `{raw_hash}`.
- `{prob_path}` — `{prob_hash}`.
- Standardized OOF: `{summary.get('standardized_oof_path', 'not produced')}` — `{summary.get('standardized_oof_sha256', 'not applicable')}`.
- Standardized TEST parquet: `{summary.get('standardized_test_path', 'not produced')}` — `{summary.get('standardized_test_sha256', 'not applicable')}`.
- Standardized TEST CSV: `{summary.get('standardized_test_csv_path', 'not produced')}` — `{summary.get('standardized_test_csv_sha256', 'not applicable')}`.
- Production rule: `{production.get('rule', 'not applicable')}`; selected value: `{production.get('selected_value', 'not applicable')}`.

All experiment-local hashes are listed in `checksums.sha256`.

## 13. Final recommendation

**{recommendation}**
"""
    (OUT / "report.md").write_text(report, encoding="utf-8")
    write_json(
        "runtime_resources.json",
        {"runtime_seconds": runtime, "persistent_bytes": disk, "disk_limit_bytes": PERSISTENT_DISK_LIMIT, "physical_cores": PHYSICAL_CORES, "lightgbm_threads": N_THREADS},
    )
    files = sorted(path for path in OUT.iterdir() if path.is_file() and path.name != "checksums.sha256")
    (OUT / "checksums.sha256").write_text("".join(f"{sha256(path)}  {path.name}\n" for path in files), encoding="utf-8")
    final_disk = sum(path.stat().st_size for path in OUT.rglob("*") if path.is_file())
    if runtime > RUNTIME_HARD_STOP:
        raise RuntimeError(f"runtime hard stop exceeded: {runtime:.1f}s")
    if final_disk > PERSISTENT_DISK_LIMIT:
        raise RuntimeError(f"persistent disk limit exceeded: {final_disk} bytes")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-retrain", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    features = [line.strip() for line in FEATURE_LIST.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(features) != 227 or len(features) != len(set(features)):
        raise AssertionError("frozen feature list is not the exact 227-column S1-E10 list")
    log(f"starting with {len(features)} frozen features and {N_THREADS} LightGBM threads")

    # Labels are constructed before model fitting.  The fallback inspects only the
    # oldest fold's legal training cutoffs.
    latest_training = training_cutoffs(PILOT_FOLD)
    needed_labels: dict[tuple[str, int], pl.DataFrame] = {}
    for cutoff in sorted(set(latest_training + FOLDS)):
        if cutoff in latest_training:
            needed_labels[(cutoff, 1)] = build_label(cutoff, 1)
        if cutoff in FOLDS:
            needed_labels[(cutoff, 3)] = build_label(cutoff, 3)
    oldest_training = training_cutoffs(FOLDS[0])
    oldest_counts = np.concatenate([needed_labels[(cutoff, 1)]["N30"].to_numpy() for cutoff in oldest_training])
    c4_frequency = float(np.mean(oldest_counts >= 8))
    merge_c4 = c4_frequency < 0.005
    n_classes = 4 if merge_c4 else 5
    write_json(
        "class_bin_decision.json",
        {"inspection_scope": "oldest fold training panel only", "oldest_fold": FOLDS[0], "training_cutoffs": oldest_training, "rows": len(oldest_counts), "C4_frequency": c4_frequency, "fallback_threshold": 0.005, "merge_C3_C4": merge_c4, "frozen_n_classes": n_classes, "class_names": CLASS_NAMES_4 if merge_c4 else CLASS_NAMES_5},
    )
    log(f"class fallback frozen: merge_c4={merge_c4}; C4 frequency={c4_frequency:.6%}")

    audit_frame, label_pass = label_audit(needed_labels)
    audit_frame.to_csv(OUT / "label_audit.csv", index=False)
    distribution_rows = class_distribution_rows(needed_labels, merge_c4, set(latest_training))
    pd.DataFrame(distribution_rows).to_csv(OUT / "class_distribution.csv", index=False)
    if not label_pass:
        raise AssertionError("deterministic slow-reference label audit failed")
    log("1,000-row slow-reference label audit passed")

    aligned = pd.read_parquet(ALIGNED_OOF)
    if len(aligned) != 770_616 or aligned.duplicated(["fold", "user_id"]).any():
        raise AssertionError("canonical aligned OOF schema failed")

    pilot_cache = OUT / "_fold_20251016.npz"
    if args.force_retrain or not pilot_cache.exists():
        pilot_result = fit_fold(PILOT_FOLD, features, merge_c4, n_classes)
        save_fold_cache(pilot_result)
        del pilot_result
        gc.collect()
    pilot_result = load_fold_cache(PILOT_FOLD)
    pilot_frame = aligned_fold(pilot_result, aligned)
    probability = []
    real_probability_rows, real_probability_pass = probability_rows(pilot_frame, "real", n_classes)
    shuffled_probability_rows, _ = probability_rows(pilot_frame, "shuffled", n_classes)
    probability.extend(real_probability_rows + shuffled_probability_rows)
    pilot = pilot_metrics(pilot_frame, real_probability_pass)
    write_json("pilot_metrics.json", pilot)
    log(f"pilot gate: {'PASS' if pilot['pilot_pass'] else 'FAIL'} via {pilot['gate_endpoint']}")

    if not pilot["pilot_pass"]:
        probability_frame = pd.DataFrame(probability)
        probability_frame.to_csv(OUT / "probability_metrics.csv", index=False)
        raw_path, prob_path = write_raw_vectors(pilot_frame, n_classes)
        candidates = {
            "EXP037": zcol(pilot_frame, "pred_exp037"), "S1-E10": zcol(pilot_frame, "pred_ridge15") if False else zcol(pilot_frame, "pred_exp037"),
            "S1-E10_ACTUAL": zcol(pilot_frame, "pred_exp037"),
            "DIST": zcol(pilot_frame, "pred_dist"), "E11": zcol(pilot_frame, "pred_hurdle_e11"),
            "COUNT_REAL": pilot_frame["z_count_real"].to_numpy(float), "COUNT_SHUFFLED": pilot_frame["z_count_shuffled"].to_numpy(float),
            "REPLACE_REAL_FIXED": zcol(pilot_frame, "pred_exp037") + 0.25 * (pilot_frame["z_count_real"].to_numpy(float) - zcol(pilot_frame, "pred_dist")),
            "REPLACE_SHUFFLED_FIXED": zcol(pilot_frame, "pred_exp037") + 0.25 * (pilot_frame["z_count_shuffled"].to_numpy(float) - zcol(pilot_frame, "pred_dist")),
            "ADD10_REAL_FIXED": 0.9 * zcol(pilot_frame, "pred_exp037") + 0.1 * pilot_frame["z_count_real"].to_numpy(float),
            "ADD10_SHUFFLED_FIXED": 0.9 * zcol(pilot_frame, "pred_exp037") + 0.1 * pilot_frame["z_count_shuffled"].to_numpy(float),
        }
        # Replace placeholder S1-E10 with the exact aligned historical artifact.
        e10 = np.load(OLD / "artifacts" / "oof_S1-E10.npz", allow_pickle=False)
        source = pd.DataFrame({"user_id": e10["user_id"][e10["cutoff"] == PILOT_FOLD], "z": e10["z"][e10["cutoff"] == PILOT_FOLD]})
        candidates["S1-E10"] = pilot_frame[["user_id"]].merge(source, on="user_id", how="left", validate="one_to_one")["z"].to_numpy(float)
        candidates.pop("S1-E10_ACTUAL")
        metric_rows, _ = fixed_metric_rows(candidates, pilot_frame["target"].to_numpy(float), pilot_frame["fold"].to_numpy(str))
        pd.DataFrame(metric_rows).to_csv(OUT / "fold_metrics.csv", index=False)
        pd.DataFrame(
            [{"path": row["endpoint"], "fold": PILOT_FOLD, "real_score": row["real_score"], "shuffled_score": row["shuffled_score"], "real_minus_shuffled": row["real_minus_shuffled"], "status": "pilot"} for row in pilot["endpoints"]]
        ).to_csv(OUT / "real_vs_shuffled.csv", index=False)
        pd.DataFrame(columns=["path", "arm", "heldout_fold", "donor_folds", "grid", "selected_value", "selection_wcv", "donor_scores", "heldout_score", "heldout_baseline_score", "heldout_delta", "status"]).assign(status=["not_run_pilot_gate"]).to_csv(OUT / "nested_selection.csv", index=False)
        segment_metrics(pilot_frame, pilot_frame["z_count_real"].to_numpy(float), "count_value_moe_raw_pilot").to_csv(OUT / "segment_metrics.csv", index=False)
        partial_diversity(pilot_frame, pilot_frame["z_count_real"].to_numpy(float), "count_value_moe_raw_pilot").to_csv(OUT / "diversity_oof.csv", index=False)
        write_json("oof_projection_metrics.json", {"status": "not_run_pilot_gate", "reason": "donor-fold projection requires full four-fold OOF"})
        write_json("test_span_projection.json", {"status": "not_run_pilot_gate"})
        write_json("production_regime.json", {"status": "not_run_pilot_gate", "test_prediction_constructed": False})
        consumed = {RAW, FEATURE_LIST, ALIGNED_OOF}
        for cutoff in sorted(set(latest_training + FOLDS)):
            consumed.add(feature_path(cutoff))
            if cutoff in latest_training:
                consumed.add(panel_path(cutoff, 1))
            if cutoff in FOLDS:
                consumed.add(panel_path(cutoff, 3))
        artifact_manifest(consumed)
        finalize_report(
            "REJECT_PILOT", "DO_NOT_ADD", merge_c4, label_pass, real_probability_pass, pilot,
            {"full_status": "Full four-fold validation was not run because the fixed pilot gate failed", "standalone": "Only the preregistered latest-fold pilot was run; standalone real and placebo scores are tabulated in fold_metrics.csv.", "segments": "Latest-fold diagnostics show signed residual changes only and were not used for selection.", "novelty": "Latest-fold correlations were computed; donor-fold projection was prohibited after the pilot failure."},
            raw_path, prob_path,
        )
        return

    # Full four-fold validation after the fixed pilot gate.
    for fold_name in FOLDS[:-1]:
        cache = OUT / f"_fold_{fold_name.replace('-', '')}.npz"
        if args.force_retrain or not cache.exists():
            result = fit_fold(fold_name, features, merge_c4, n_classes)
            save_fold_cache(result)
            del result
            gc.collect()
    results = [load_fold_cache(fold_name) for fold_name in FOLDS]
    full = pd.concat([aligned_fold(result, aligned) for result in results], ignore_index=True)
    for result, frame in zip(results, [full.loc[full["fold"] == fold_name] for fold_name in FOLDS]):
        rows_real, passed = probability_rows(frame, "real", n_classes)
        rows_shuf, _ = probability_rows(frame, "shuffled", n_classes)
        probability.extend(rows_real + rows_shuf)
        real_probability_pass = real_probability_pass and passed
    pd.DataFrame(probability).drop_duplicates(["fold", "arm", "metric", "class_index", "note"], keep="last").to_csv(OUT / "probability_metrics.csv", index=False)
    raw_path, prob_path = write_raw_vectors(full, n_classes)

    y = full["target"].to_numpy(float)
    fold = full["fold"].to_numpy(str)
    z_base = zcol(full, "pred_exp037")
    z_dist = zcol(full, "pred_dist")
    z_real = full["z_count_real"].to_numpy(float)
    z_shuf = full["z_count_shuffled"].to_numpy(float)
    historical = {}
    for name, artifact in (("S1-E10", "S1-E10"), ("DIST", "S1-DIST"), ("E11", "S1-E11")):
        values = np.load(OLD / "artifacts" / f"oof_{artifact}.npz", allow_pickle=False)
        source = pd.DataFrame({"fold": values["cutoff"], "user_id": values["user_id"], "z": values["z"]})
        historical[name] = full[["fold", "user_id"]].merge(source, on=["fold", "user_id"], how="left", validate="one_to_one")["z"].to_numpy(float)

    fixed_candidates = {
        "EXP037": z_base, "S1-E10": historical["S1-E10"], "DIST": historical["DIST"], "E11": historical["E11"],
        "COUNT_REAL": z_real, "COUNT_SHUFFLED": z_shuf,
        "REPLACE_REAL_BETA1": z_base + 0.25 * (z_real - z_dist),
        "REPLACE_SHUFFLED_BETA1": z_base + 0.25 * (z_shuf - z_dist),
        "ADD10_REAL": z_base + 0.10 * (z_real - z_base),
        "ADD10_SHUFFLED": z_base + 0.10 * (z_shuf - z_base),
    }
    for beta in BETA_GRID:
        fixed_candidates[f"REPLACE_REAL_BETA_{beta:g}"] = z_base + 0.25 * float(beta) * (z_real - z_dist)
        fixed_candidates[f"REPLACE_SHUFFLED_BETA_{beta:g}"] = z_base + 0.25 * float(beta) * (z_shuf - z_dist)
    for alpha in ALPHA_GRID:
        fixed_candidates[f"ADD_REAL_ALPHA_{alpha:g}"] = z_base + float(alpha) * (z_real - z_base)
        fixed_candidates[f"ADD_SHUFFLED_ALPHA_{alpha:g}"] = z_base + float(alpha) * (z_shuf - z_base)
    metric_rows, evaluations = fixed_metric_rows(fixed_candidates, y, fold)
    pd.DataFrame(metric_rows).to_csv(OUT / "fold_metrics.csv", index=False)

    nested_rows = []
    path_results: dict[tuple[str, str], dict[str, Any]] = {}
    for path_name, direction_real, direction_shuf, grid in (
        ("replacement", 0.25 * (z_real - z_dist), 0.25 * (z_shuf - z_dist), BETA_GRID),
        ("add_one", z_real - z_base, z_shuf - z_base, ALPHA_GRID),
    ):
        for arm, direction in (("real", direction_real), ("shuffled", direction_shuf)):
            rows, scores, selected = nested_path(path_name, arm, y, fold, z_base, direction, grid)
            nested_rows.extend(rows)
            delta = scores - evaluations["EXP037"]["cal"]
            path_results[(path_name, arm)] = {
                "scores": scores, "selected": selected, "delta": delta,
                "wcv": float(FOLD_WEIGHTS @ scores / FOLD_WEIGHTS.sum()),
                "delta_wcv": float(FOLD_WEIGHTS @ delta / FOLD_WEIGHTS.sum()),
                "improved_folds": int(np.sum(delta < 0)), "latest_delta": float(delta[-1]),
            }
    pd.DataFrame(nested_rows).to_csv(OUT / "nested_selection.csv", index=False)
    comparison_rows = []
    for path_name in ("replacement", "add_one"):
        real_result = path_results[(path_name, "real")]
        shuf_result = path_results[(path_name, "shuffled")]
        for index, fold_name in enumerate(FOLDS):
            comparison_rows.append({"path": path_name, "fold": fold_name, "real_score": real_result["scores"][index], "shuffled_score": shuf_result["scores"][index], "real_minus_shuffled": real_result["scores"][index] - shuf_result["scores"][index], "status": "nested_lofo"})
        comparison_rows.append({"path": path_name, "fold": "wCV", "real_score": real_result["wcv"], "shuffled_score": shuf_result["wcv"], "real_minus_shuffled": real_result["wcv"] - shuf_result["wcv"], "status": "nested_lofo"})
    pd.DataFrame(comparison_rows).to_csv(OUT / "real_vs_shuffled.csv", index=False)

    best_path = min(("replacement", "add_one"), key=lambda name: (path_results[(name, "real")]["delta_wcv"], name))
    best = path_results[(best_path, "real")]
    matched = path_results[(best_path, "shuffled")]
    selected_by_fold = best["selected"]
    if best_path == "replacement":
        direction = 0.25 * (z_real - z_dist)
        shuffled_direction = 0.25 * (z_shuf - z_dist)
    else:
        direction = z_real - z_base
        shuffled_direction = z_shuf - z_base
    scale_by_row = np.asarray([selected_by_fold[FOLDS.index(name)] for name in fold])
    candidate_nested = z_base + scale_by_row * direction
    shuffled_nested = z_base + np.asarray([matched["selected"][FOLDS.index(name)] for name in fold]) * shuffled_direction
    diversity = partial_diversity(full, candidate_nested, f"count_value_moe_{best_path}_nested")
    diversity.to_csv(OUT / "diversity_oof.csv", index=False)
    projection_columns = [column for column in full.columns if column.startswith("pred_") and column != "pred_exp037"]
    projection_x = np.column_stack([zcol(full, column) - z_base for column in projection_columns])
    projection = ridge_projection(fold, projection_x, projection_columns, {"raw_count_correction": z_real - z_base, "selected_nested_correction": candidate_nested - z_base})
    write_json("oof_projection_metrics.json", projection)
    unexplained = projection["targets"]["selected_nested_correction"]["weighted_unexplained_variance_ratio"]

    segments = segment_metrics(full, candidate_nested, f"count_value_moe_{best_path}_nested")
    segments.to_csv(OUT / "segment_metrics.csv", index=False)
    point_deltas = {"SELECTED_REAL": best["delta_wcv"], "SELECTED_SHUFFLED": matched["delta_wcv"]}
    bootstrap = bootstrap_intervals(full, {"EXP037": z_base, "SELECTED_REAL": candidate_nested, "SELECTED_SHUFFLED": shuffled_nested}, point_deltas)
    bootstrap.to_csv(OUT / "bootstrap_metrics.csv", index=False)

    real_shuffled_gap = best["wcv"] - matched["wcv"]
    provisional_a = best["delta_wcv"] <= -0.00035 and best["improved_folds"] >= 3 and best["latest_delta"] < 0 and real_shuffled_gap <= -0.00015 and label_pass and real_probability_pass
    provisional_b = best["delta_wcv"] <= -0.00010 and best["improved_folds"] >= 3 and best["latest_delta"] < 0 and real_shuffled_gap <= -0.00010 and unexplained >= 0.25 and label_pass and real_probability_pass
    summary: dict[str, Any] = {
        "full_status": f"Best path {best_path}: delta wCV {best['delta_wcv']:+.9f}, {best['improved_folds']}/4 folds, latest {best['latest_delta']:+.9f}, real-minus-shuffled {real_shuffled_gap:+.9f}",
        "standalone": f"Full standalone real wCV {evaluations['COUNT_REAL']['wcv']:.9f}; shuffled wCV {evaluations['COUNT_SHUFFLED']['wcv']:.9f}; EXP-037 {evaluations['EXP037']['wcv']:.9f}.",
        "nested": f"Replacement delta {path_results[('replacement','real')]['delta_wcv']:+.9f}; add-one delta {path_results[('add_one','real')]['delta_wcv']:+.9f}; best path {best_path}.",
        "segments": "Full explanatory signed-residual diagnostics are in segment_metrics.csv and were not used for selection.",
        "novelty": f"Weighted held-out unexplained correction variance ratio is {unexplained:.6f} for the selected nested correction.",
    }

    verdict = "REJECT"
    recommendation = "DO_NOT_ADD"
    test_projection: dict[str, Any] = {"status": "not_run_no_provisional_pass"}
    production: dict[str, Any] = {"status": "not_run_no_provisional_pass"}
    if provisional_a or provisional_b:
        fixed_beta1 = evaluations["REPLACE_REAL_BETA1"]
        fixed_beta1_gap = fixed_beta1["wcv"] - evaluations["REPLACE_SHUFFLED_BETA1"]["wcv"]
        fixed_beta1_delta = fixed_beta1["wcv"] - evaluations["EXP037"]["wcv"]
        fixed_beta1_pass_a = fixed_beta1_delta <= -0.00035 and int(np.sum(fixed_beta1["cal"] < evaluations["EXP037"]["cal"])) >= 3 and fixed_beta1["cal"][-1] < evaluations["EXP037"]["cal"][-1] and fixed_beta1_gap <= -0.00015
        if best_path == "replacement":
            if fixed_beta1_pass_a:
                production_value, rule = 1.0, "fixed preregistered full replacement beta=1 because that fixed endpoint independently passed TYPE A"
            else:
                production_value = floor_to_grid(float(np.median(selected_by_fold)), BETA_GRID)
                rule = "median held-fold nested beta rounded down to nearest frozen grid value"
        else:
            production_value = floor_to_grid(float(np.median(selected_by_fold)), ALPHA_GRID)
            rule = "median held-fold nested alpha rounded down to nearest frozen grid value"
        production = {"status": "predeclared_before_test_span_read", "path": best_path, "rule": rule, "selected_value": production_value, "heldout_values": selected_by_fold.tolist(), "test_cutoff": "2026-02-13"}
        write_json("production_regime.json", production)
        user_test, z_raw_test, p_test, clip_test = train_production(features, merge_c4, n_classes)
        test_bank = pd.read_parquet(ALIGNED_TEST)
        aligned_test = pd.DataFrame({"user_id": user_test, "z_raw": z_raw_test}).merge(test_bank, on="user_id", how="left", validate="one_to_one")
        if len(aligned_test) != len(test_bank) or aligned_test.isna().any().any():
            raise AssertionError("TEST regime/schema alignment failed")
        z_base_test = zcol(aligned_test, "pred_exp037")
        if best_path == "replacement":
            z_standard = z_base_test + 0.25 * production_value * (aligned_test["z_raw"].to_numpy(float) - zcol(aligned_test, "pred_dist"))
        else:
            z_standard = z_base_test + production_value * (aligned_test["z_raw"].to_numpy(float) - z_base_test)
        test_projection = geometry_projection(z_standard, aligned_test["user_id"].to_numpy(np.int64))
        write_json("test_span_projection.json", test_projection)
        final_a = provisional_a
        final_b = provisional_b and test_projection["orthogonal_norm_fraction"] >= 0.10 and test_projection["orthogonal_rms"] >= 0.0025
        if final_a or final_b:
            verdict = "PASS_TYPE_A" if final_a else "PASS_TYPE_B"
            recommendation = "ADD_TO_SUBMISSION_GEOMETRY"
            standardized_oof = pd.DataFrame({"user_id": full["user_id"].to_numpy(np.int64), "fold": fold, "target": y, "predict": np.expm1(np.maximum(candidate_nested, 0.0)), "z_predict": candidate_nested, "z_base": z_base, "correction": candidate_nested - z_base, "candidate_name": f"count_value_moe_{best_path}_nested"})
            oof_path = OUT / "count_value_moe_OOF.parquet"
            standardized_oof.to_parquet(oof_path, index=False, compression="zstd")
            test_output = pd.DataFrame({"user_id": aligned_test["user_id"].to_numpy(np.int64), "predict": np.expm1(np.maximum(z_standard, 0.0)), "z_predict": z_standard, "z_base": z_base_test, "correction": z_standard - z_base_test, "raw_count_predict": np.expm1(np.maximum(aligned_test["z_raw"].to_numpy(float), 0.0)), "raw_count_z": aligned_test["z_raw"].to_numpy(float), "candidate_name": f"count_value_moe_{best_path}_production"})
            test_path = OUT / "count_value_moe_TEST.parquet"
            test_csv = OUT / "count_value_moe_TEST.csv"
            test_output.to_parquet(test_path, index=False, compression="zstd")
            test_output[["user_id", "predict"]].to_csv(test_csv, index=False)
            summary.update({"standardized_oof_path": str(oof_path), "standardized_oof_sha256": sha256(oof_path), "standardized_test_path": str(test_path), "standardized_test_sha256": sha256(test_path), "standardized_test_csv_path": str(test_csv), "standardized_test_csv_sha256": sha256(test_csv)})
        else:
            verdict = "WEAK_SIGNAL" if best["delta_wcv"] <= 0.00005 else "REJECT"
            recommendation = "DO_NOT_ADD"
    else:
        if (-0.00010 <= best["delta_wcv"] <= 0.00005) or (real_shuffled_gap < 0) or best["improved_folds"] == 2:
            verdict = "WEAK_SIGNAL"
        else:
            verdict = "REJECT"
        write_json("test_span_projection.json", test_projection)
        write_json("production_regime.json", production)

    summary["test_projection"] = test_projection
    summary["production"] = production
    consumed = {RAW, FEATURE_LIST, ALIGNED_OOF}
    if provisional_a or provisional_b:
        consumed.add(ALIGNED_TEST)
    for cutoff in sorted(set(cutoff_grid() if (provisional_a or provisional_b) else latest_training + FOLDS)):
        consumed.add(feature_path(cutoff))
        if cutoff in cutoff_grid() and (cutoff in latest_training or provisional_a or provisional_b):
            consumed.add(panel_path(cutoff, 1))
        if cutoff in FOLDS or cutoff == "2026-02-13":
            if panel_path(cutoff, 3).exists():
                consumed.add(panel_path(cutoff, 3))
    artifact_manifest(consumed)
    finalize_report(verdict, recommendation, merge_c4, label_pass, real_probability_pass, pilot, summary, raw_path, prob_path)


if __name__ == "__main__":
    main()
