"""Cheap occurrence-only residual overlay on the canonical EXP-037 OOF.

This is the explicitly permitted fallback for the missing teammate checkpoint
bank.  It performs one fixed LightGBM experiment: no feature search, no model
search and no blend-weight search.  The four OOF predictions are leave-one-
fold-out cross-fits; the TEST prediction is fit once on all four canonical
folds.  All source repositories are read-only.

Run from any directory:
    python analysis/unified_oof/scripts/run_occurrence_fallback.py
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl


SCRIPT = Path(__file__).resolve()
OUT_DIR = SCRIPT.parents[1]
ARTIFACT_DIR = OUT_DIR / "artifacts"
SOURCE_ROOT = Path(os.environ.get("OZON_SOURCE_ROOT", r"C:\Users\Admin\Desktop\OZON-E-CUP"))
ALIGNED_OOF = SOURCE_ROOT / "artifacts" / "RESDISC_053" / "aligned_oof.parquet"
TEST_FEATURES = SOURCE_ROOT / "data" / "processed" / "feat_20260213_L180.parquet"
STRONGEST_CSV = SOURCE_ROOT / "submissions" / "submission_STRONGEST_CURRENT.csv"

FOLDS = ("2025-09-04", "2025-09-18", "2025-10-02", "2025-10-16")
FOLD_WEIGHTS = np.asarray([1.0, 2.0, 4.0, 8.0], dtype=np.float64)
FOLD_WEIGHTS /= FOLD_WEIGHTS.sum()

# Locked before evaluation.  These columns contain purchase occurrence only:
# counts/days, recency, inter-purchase gaps and multi-scale count changes.
WINDOWS = (7, 14, 30, 60, 90, 180)
FEATURES = tuple(
    [f"w{w}_days_buy" for w in WINDOWS]
    + [f"w{w}_orders" for w in WINDOWS]
    + ["rec_buy", "buygap_mean", "buygap_std", "buygap_cv"]
    + [
        "dlog_buyd_7_14",
        "dlog_buyd_7_30",
        "dlog_buyd_14_30",
        "dlog_buyd_30_60",
        "dlog_buyd_30_90",
        "dlog_buyd_60_180",
    ]
)

# One deliberately conservative model.  num_boost_round is fixed; there is no
# early stopping on an outer fold and no hyperparameter selection.
PARAMS = {
    "objective": "regression_l2",
    "metric": "l2",
    "learning_rate": 0.025,
    "num_leaves": 15,
    "max_depth": 5,
    "min_data_in_leaf": 1200,
    "feature_fraction": 1.0,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l1": 5.0,
    "lambda_l2": 50.0,
    "min_gain_to_split": 1e-4,
    "max_bin": 127,
    "seed": 20260825,
    "feature_fraction_seed": 20260825,
    "bagging_seed": 20260825,
    "data_random_seed": 20260825,
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": max(2, min(8, os.cpu_count() or 8)),
    "verbosity": -1,
}
NUM_BOOST_ROUND = 120


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def finite_matrix(frame: pl.DataFrame) -> np.ndarray:
    x = frame.select(FEATURES).to_numpy().astype(np.float32, copy=False)
    # LightGBM handles NaN as missing.  Infinite ratios are not meaningful and
    # are converted to the same missing representation.
    x[~np.isfinite(x)] = np.nan
    return x


def training_weights(fold_idx: np.ndarray, mask: np.ndarray) -> np.ndarray:
    w = np.zeros(int(mask.sum()), dtype=np.float32)
    selected = fold_idx[mask]
    for j in range(4):
        m = selected == j
        if m.any():
            # Fold totals, rather than row counts, follow the project 1:2:4:8
            # temporal weighting.
            w[m] = float(FOLD_WEIGHTS[j] / m.sum())
    w *= len(w) / max(float(w.sum()), 1e-12)
    return w


def fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    weights: np.ndarray,
    x_eval: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    ds = lgb.Dataset(x_train, label=y_train, weight=weights, free_raw_data=False)
    model = lgb.train(PARAMS, ds, num_boost_round=NUM_BOOST_ROUND)
    pred = model.predict(x_eval, num_iteration=NUM_BOOST_ROUND).astype(np.float64)
    importance = dict(zip(FEATURES, model.feature_importance(importance_type="gain").astype(float)))
    return pred, importance


def main() -> None:
    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    required = (ALIGNED_OOF, TEST_FEATURES, STRONGEST_CSV)
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required existing artifacts: {missing}")

    cols = ["cutoff", "user_id", "fold", "y_true", "z_strong_raw", "z_strong_calibrated", "r_strong", *FEATURES]
    frame = pl.read_parquet(ALIGNED_OOF, columns=cols).sort(["cutoff", "user_id"])
    if frame.height != 770_616:
        raise AssertionError(f"Unexpected canonical OOF size: {frame.height}")
    if frame.select(pl.struct(["cutoff", "user_id"]).n_unique()).item() != frame.height:
        raise AssertionError("Duplicate canonical (cutoff,user_id) keys")
    cutoff = frame["cutoff"].to_numpy()
    if tuple(sorted(set(cutoff.tolist()))) != FOLDS:
        raise AssertionError("Canonical fold set mismatch")
    fold_idx = np.asarray([FOLDS.index(str(x)) for x in cutoff], dtype=np.int8)
    x = finite_matrix(frame)
    target = frame["r_strong"].to_numpy().astype(np.float64)
    if not np.isfinite(target).all():
        raise AssertionError("Non-finite residual target")

    delta_oof_raw = np.empty(frame.height, dtype=np.float64)
    fold_importance: dict[str, dict[str, float]] = {}
    for outer, fold in enumerate(FOLDS):
        val = fold_idx == outer
        train = ~val
        pred, importance = fit_predict(
            x[train], target[train], training_weights(fold_idx, train), x[val]
        )
        delta_oof_raw[val] = pred
        fold_importance[fold] = importance
        print(f"outer={fold} train={int(train.sum()):,} val={int(val.sum()):,}", flush=True)

    # Proper nested predictions used only for selecting one scalar correction
    # strength.  For outer=o and inner=i, both folds are excluded from model
    # fitting; therefore neither inner scoring nor the eventual outer scoring
    # can see the evaluated fold's labels.  This is validation infrastructure,
    # not an additional model/feature hypothesis.
    nested_inner_raw = np.full((4, frame.height), np.nan, dtype=np.float32)
    for outer, outer_fold in enumerate(FOLDS):
        for inner, inner_fold in enumerate(FOLDS):
            if inner == outer:
                continue
            val = fold_idx == inner
            train = (fold_idx != outer) & (fold_idx != inner)
            pred, _ = fit_predict(
                x[train], target[train], training_weights(fold_idx, train), x[val]
            )
            nested_inner_raw[outer, val] = pred.astype(np.float32)
        print(f"nested outer={outer_fold} complete", flush=True)

    delta_oof_centered = delta_oof_raw.copy()
    for outer in range(4):
        m = fold_idx == outer
        delta_oof_centered[m] -= float(delta_oof_centered[m].mean())

    test = pl.read_parquet(TEST_FEATURES, columns=["user_id", *FEATURES])
    friend = pl.read_csv(STRONGEST_CSV).select(["user_id", "predict"])
    if friend.height != 250_000 or friend["user_id"].n_unique() != 250_000:
        raise AssertionError("Invalid STRONGEST_CURRENT TEST keys")
    joined = friend.select("user_id").join(test, on="user_id", how="left", validate="1:1")
    if joined.height != friend.height or joined["user_id"].null_count() != 0:
        raise AssertionError("TEST feature alignment failed")
    x_test = finite_matrix(joined)
    delta_test_raw, test_importance = fit_predict(
        x, target, training_weights(fold_idx, np.ones(frame.height, dtype=bool)), x_test
    )
    delta_test_centered = delta_test_raw - float(delta_test_raw.mean())

    oof_path = ARTIFACT_DIR / "fallback_occ_overlay_oof.npz"
    np.savez_compressed(
        oof_path,
        cutoff=cutoff.astype("U10"),
        fold=fold_idx,
        user_id=frame["user_id"].to_numpy().astype(np.int64),
        y=frame["y_true"].to_numpy().astype(np.float64),
        z_strong=frame["z_strong_raw"].to_numpy().astype(np.float64),
        delta_raw=delta_oof_raw,
        delta_centered=delta_oof_centered,
        nested_inner_raw=nested_inner_raw,
    )
    test_path = ARTIFACT_DIR / "fallback_occ_overlay_test.npz"
    np.savez_compressed(
        test_path,
        user_id=friend["user_id"].to_numpy().astype(np.int64),
        delta_raw=delta_test_raw,
        delta_centered=delta_test_centered,
    )

    ly = np.log1p(frame["y_true"].to_numpy().astype(np.float64))
    base = frame["z_strong_raw"].to_numpy().astype(np.float64)

    def fold_score(z: np.ndarray, j: int) -> float:
        r = ly[fold_idx == j] - z[fold_idx == j]
        return float(np.std(r))

    base_scores = [fold_score(base, j) for j in range(4)]
    cand_scores = [fold_score(base + delta_oof_centered, j) for j in range(4)]
    summary = {
        "experiment": "OCC-OVERLAY-ON-CHAMPION",
        "protocol": "single locked occurrence-only LightGBM residual model; leave-one-fold-out cross-fit",
        "source_root_read_only": str(SOURCE_ROOT),
        "features": list(FEATURES),
        "feature_count": len(FEATURES),
        "params": PARAMS,
        "num_boost_round": NUM_BOOST_ROUND,
        "rows_oof": frame.height,
        "rows_test": friend.height,
        "folds": list(FOLDS),
        "base_fold_wcv_scores": base_scores,
        "candidate_fold_wcv_scores_fixed_unit_scale": cand_scores,
        "fold_deltas_fixed_unit_scale": [a - b for a, b in zip(cand_scores, base_scores)],
        "base_wcv": float(FOLD_WEIGHTS @ base_scores),
        "candidate_wcv_fixed_unit_scale": float(FOLD_WEIGHTS @ cand_scores),
        "delta_wcv_fixed_unit_scale": float(FOLD_WEIGHTS @ (np.asarray(cand_scores) - np.asarray(base_scores))),
        "delta_oof_rms": float(np.sqrt(np.mean(delta_oof_centered**2))),
        "delta_test_rms": float(np.sqrt(np.mean(delta_test_centered**2))),
        "delta_test_oof_variance_ratio": float(np.var(delta_test_centered) / max(np.var(delta_oof_centered), 1e-12)),
        "runtime_seconds": time.time() - started,
        "inputs": {
            str(ALIGNED_OOF): sha256(ALIGNED_OOF),
            str(TEST_FEATURES): sha256(TEST_FEATURES),
            str(STRONGEST_CSV): sha256(STRONGEST_CSV),
        },
        "outputs": {
            str(oof_path): sha256(oof_path),
            str(test_path): sha256(test_path),
        },
        "fold_feature_importance_gain": fold_importance,
        "test_feature_importance_gain": test_importance,
    }
    (ARTIFACT_DIR / "fallback_occ_overlay_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: summary[k] for k in (
        "runtime_seconds", "base_wcv", "candidate_wcv_fixed_unit_scale",
        "delta_wcv_fixed_unit_scale", "fold_deltas_fixed_unit_scale",
        "delta_test_oof_variance_ratio"
    )}, indent=2), flush=True)


if __name__ == "__main__":
    main()
