"""ZERO2D-SHRINK: OOF-only negative residual correction by amount and DIST p0.

The command uses only saved OOF predictions and the reproduced ``S1-DIST``
distribution head.  It trains no model::

    python src/zero2d_shrink.py

For every outer fold, p0 quantiles, calibrated residual means, isotonic mapping,
and eta selection use the other three folds only.  The held-out fold is touched
only once, when the already selected mapping and eta are scored.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.config import ARTIFACTS, FOLD_WEIGHTS_S1, SEED, VAL_FOLDS_S1
from src.validation import calibrate


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "research" / "strategies" / "results" / "ZERO2D_SHRINK"

LEVEL = 2.3293
AMOUNT_EDGES = np.asarray([1.0, 3.0, 10.0, 30.0, 50.0, 100.0])
AMOUNT_LABELS = ["[0, 1)", "[1, 3)", "[3, 10)", "[10, 30)", "[30, 50)",
                 "[50, 100)", "[100, +inf)"]
P0_QUANTILES = np.asarray([0.2, 0.4, 0.6, 0.8])
ETA_GRID = np.asarray([0.25, 0.50, 0.75, 1.00])
SHRINK_STRENGTH = 20_000
MIN_CELL_ROWS = 500

BASE_COMPONENTS = {
    "S1-E03a": 0.10,       # CAP
    "S1-E02": 0.20,        # UNC
    "S1-DIST": 0.25,
    "ETX-AVG3": 0.225,
    "SEQ-AVG3": 0.225,
}
BASE_TEST_COMPONENTS = {
    "S1-CAP": 0.10,
    "S1-UNC": 0.20,
    "S1-DIST": 0.25,
    "SEQ-01": 0.075,
    "SEQ-C289-S43": 0.075,
    "SEQ-C289-S44": 0.075,
    "ETX-01-S42-DCW": 0.075,
    "ETX-01-S43-DCW": 0.075,
    "ETX-01-S44-DCW": 0.075,
}
EXPECTED_BASE_FOLDS = np.asarray([1.766883, 1.760510, 1.748629, 1.741279])
EXPECTED_BASE_WCV = 1.74751


def _log(message: str) -> None:
    print(message, flush=True)


def _json_value(value):
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_value),
                    encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            cooked = {}
            for key, value in row.items():
                if isinstance(value, (list, tuple, dict, np.ndarray)):
                    cooked[key] = json.dumps(value, ensure_ascii=False, default=_json_value)
                else:
                    cooked[key] = value
            writer.writerow(cooked)


def amount_bins(prediction_amount: np.ndarray) -> np.ndarray:
    """Fixed left-closed amount bins from the registered protocol."""
    x = np.asarray(prediction_amount, dtype=float)
    if np.any(x < 0) or not np.all(np.isfinite(x)):
        raise AssertionError("prediction amounts must be finite and non-negative")
    return np.searchsorted(AMOUNT_EDGES, x, side="right").astype(np.int8)


def prediction_shape(z_base: np.ndarray, fold_index: np.ndarray,
                     level: float = LEVEL) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prediction-only fold normalization used solely for fixed amount bins."""
    z = np.asarray(z_base, float)
    fi = np.asarray(fold_index, int)
    means = np.asarray([z[fi == fold].mean() for fold in range(int(fi.max()) + 1)])
    z_shape = z - means[fi] + float(level)
    pred_shape = np.expm1(np.maximum(z_shape, 0.0))
    return z_shape, pred_shape, means


def calibrated_residuals(y: np.ndarray, z: np.ndarray,
                         fold_index: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-fold optimal offsets and registered residual ``ly-(z+offset)``."""
    y = np.asarray(y, float)
    z = np.asarray(z, float)
    fi = np.asarray(fold_index, int)
    residual = np.empty(len(y), float)
    offsets, scores = [], []
    for fold in range(int(fi.max()) + 1):
        mask = fi == fold
        offset, score = calibrate(y[mask], z[mask])
        residual[mask] = np.log1p(y[mask]) - (z[mask] + offset)
        offsets.append(offset)
        scores.append(score)
    return residual, np.asarray(offsets), np.asarray(scores)


def fit_p0_edges(p0: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    """p0 quintile boundaries; only explicitly supplied fitting rows are read."""
    p = np.asarray(p0, float)
    train = np.asarray(train_mask, bool)
    if not train.any():
        raise AssertionError("empty fitting set for p0 quantiles")
    edges = np.quantile(p[train], P0_QUANTILES)
    if not np.all(np.isfinite(edges)) or np.any(np.diff(edges) < 0):
        raise AssertionError("invalid p0 quantile edges")
    return edges


def assign_p0_bins(p0: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.asarray(edges, float), np.asarray(p0, float),
                           side="right").astype(np.int8)


def isotonic_negative(correction: np.ndarray, counts: np.ndarray,
                      weights: np.ndarray, min_cell_rows: int = MIN_CELL_ROWS) -> np.ndarray:
    """Weighted decreasing isotonic fit with exact zero for unsupported cells.

    A fixed zero at p0-bin j forces all lower-p0 bins through j to zero: with the
    upper bound zero this is the only way to satisfy both the support rule and
    ``c[j-1] >= c[j]``.  The supported suffix is then fit by weighted isotonic
    regression.
    """
    from sklearn.isotonic import IsotonicRegression

    raw = np.minimum(np.asarray(correction, float), 0.0)
    n = np.asarray(counts, int)
    w = np.asarray(weights, float)
    if not (raw.ndim == n.ndim == w.ndim == 1 and len(raw) == len(n) == len(w)):
        raise AssertionError("isotonic inputs must be equally-sized vectors")
    unsupported = np.flatnonzero(n < int(min_cell_rows))
    start = int(unsupported.max() + 1) if len(unsupported) else 0
    out = np.zeros_like(raw)
    if start < len(raw):
        x = np.arange(start, len(raw), dtype=float)
        sw = np.maximum(w[start:], np.finfo(float).tiny)
        if len(x) == 1:
            out[start] = min(raw[start], 0.0)
        else:
            iso = IsotonicRegression(increasing=False, y_max=0.0,
                                     out_of_bounds="clip")
            out[start:] = iso.fit_transform(x, raw[start:], sample_weight=sw)
    out[n < int(min_cell_rows)] = 0.0
    out = np.minimum(out, 0.0)
    if np.any(np.diff(out) > 1e-12):
        raise AssertionError("isotonic correction is not non-increasing in p0")
    return out


def fit_mapping(amount_bin: np.ndarray, p0: np.ndarray, residual: np.ndarray,
                row_weight: np.ndarray, train_mask: np.ndarray, *, two_dim: bool = True,
                strength: int = SHRINK_STRENGTH,
                min_cell_rows: int = MIN_CELL_ROWS) -> dict:
    """Fit one outer-fold mapping without reading any non-fitting residual."""
    amount = np.asarray(amount_bin, int)
    p = np.asarray(p0, float)
    r = np.asarray(residual, float)
    sw = np.asarray(row_weight, float)
    train = np.asarray(train_mask, bool)
    if not (len(amount) == len(p) == len(r) == len(sw) == len(train)):
        raise AssertionError("mapping arrays differ in length")
    edges = fit_p0_edges(p, train) if two_dim else np.asarray([], float)
    pbin = assign_p0_bins(p, edges) if two_dim else np.zeros(len(p), np.int8)
    nq = 5 if two_dim else 1
    counts = np.zeros((7, nq), np.int64)
    weight_sum = np.zeros((7, nq), float)
    c_raw = np.zeros((7, nq), float)
    c_shrunk = np.zeros((7, nq), float)

    for amount_id in range(7):
        for p0_id in range(nq):
            mask = train & (amount == amount_id) & (pbin == p0_id)
            n = int(mask.sum())
            counts[amount_id, p0_id] = n
            if not n:
                continue
            weight_sum[amount_id, p0_id] = float(sw[mask].sum())
            c_raw[amount_id, p0_id] = float(np.average(r[mask], weights=sw[mask]))
            if n >= min_cell_rows:
                reliability = n / (n + float(strength))
                c_shrunk[amount_id, p0_id] = min(reliability * c_raw[amount_id, p0_id], 0.0)

    correction = c_shrunk.copy()
    if two_dim:
        for amount_id in range(7):
            correction[amount_id] = isotonic_negative(
                c_shrunk[amount_id], counts[amount_id], weight_sum[amount_id], min_cell_rows)
    correction[counts < min_cell_rows] = 0.0
    if np.any(correction > 1e-12):
        raise AssertionError("positive correction is forbidden")
    if two_dim and np.any(np.diff(correction, axis=1) > 1e-12):
        raise AssertionError("correction is not monotone by p0")
    return {
        "edges": edges, "p0_bin": pbin, "counts": counts,
        "weight_sum": weight_sum, "c_raw": c_raw,
        "c_shrunk": c_shrunk, "correction": correction,
    }


def mapping_correction(mapping: dict, amount_bin: np.ndarray) -> np.ndarray:
    amount = np.asarray(amount_bin, int)
    pbin = np.asarray(mapping["p0_bin"], int)
    out = np.asarray(mapping["correction"], float)[amount, pbin]
    if np.any(out > 1e-12):
        raise AssertionError("positive correction is forbidden")
    return out


def apply_log_correction(z_base: np.ndarray, correction: np.ndarray,
                         eta: float) -> np.ndarray:
    """Apply correction in z=log1p(prediction), never in raw GMV space."""
    return (np.asarray(z_base, float) + np.asarray(eta, float)
            * np.asarray(correction, float))


def set_log_level(z: np.ndarray, level: float = LEVEL) -> np.ndarray:
    """Shift and floor log-predictions so their final mean is exactly ``level``."""
    raw = np.asarray(z, float)
    lo = -float(np.max(raw)) - float(level) - 1.0
    hi = float(level) - float(np.min(raw)) + 1.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if float(np.maximum(raw + mid, 0.0).mean()) < level:
            lo = mid
        else:
            hi = mid
    out = np.maximum(raw + (lo + hi) / 2.0, 0.0)
    if abs(float(out.mean()) - level) > 1e-11:
        raise AssertionError("final prediction level was not reached")
    return out


def assert_test_order(user_id: np.ndarray, sample_user_id: np.ndarray) -> None:
    if not np.array_equal(np.asarray(user_id), np.asarray(sample_user_id)):
        raise AssertionError("test order differs from sample submission")


def _weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    return float(np.average(np.asarray(x, float), weights=np.asarray(w, float)))


def _weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    order = np.argsort(x, kind="mergesort")
    cw = np.cumsum(w[order])
    return float(x[order[np.searchsorted(cw, cw[-1] / 2.0, side="left")]])


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(np.asarray(y) > 0, np.asarray(score, float)))


def _wavg(values: np.ndarray, indices: np.ndarray | None = None) -> float:
    weights = np.asarray(FOLD_WEIGHTS_S1, float)
    if indices is not None:
        weights = weights[np.asarray(indices, int)]
    return float(np.average(np.asarray(values, float), weights=weights))


def _load_component_oof() -> dict:
    canonical = None
    z_base = None
    z_dist = None
    for name, weight in BASE_COMPONENTS.items():
        data = np.load(ARTIFACTS / f"oof_{name}.npz", allow_pickle=False)
        order = np.lexsort((np.asarray(data["user_id"]), np.asarray(data["cutoff"], dtype="U10")))
        current = {
            "uid": np.asarray(data["user_id"])[order],
            "cutoff": np.asarray(data["cutoff"], dtype="U10")[order],
            "y": np.asarray(data["y"], float)[order],
            "z": np.asarray(data["z"], float)[order],
        }
        if canonical is None:
            canonical = current
            z_base = np.zeros(len(order), float)
        else:
            if not (np.array_equal(current["uid"], canonical["uid"])
                    and np.array_equal(current["cutoff"], canonical["cutoff"])):
                raise AssertionError(f"OOF keys differ for {name}")
            if not np.allclose(current["y"], canonical["y"], atol=1e-6):
                raise AssertionError(f"OOF targets differ for {name}")
        z_base += float(weight) * current["z"]
        if name == "S1-DIST":
            z_dist = current["z"].copy()
    assert canonical is not None and z_base is not None and z_dist is not None
    canonical["z_base"] = z_base
    canonical["z_dist"] = z_dist
    return canonical


def _load_p0(frame: dict) -> tuple[np.ndarray, dict]:
    p0 = np.empty(len(frame["y"]), float)
    audit = {"folds": {}, "p_act_identity": True, "order_exact": True}
    aucs = []
    for V in VAL_FOLDS_S1:
        fold = V.isoformat()
        mask = frame["cutoff"] == fold
        pact = np.load(ARTIFACTS / f"PACT_dist_{fold}.npz", allow_pickle=False)
        order = np.argsort(np.asarray(pact["user_id"]), kind="mergesort")
        uid = np.asarray(pact["user_id"])[order]
        if not np.array_equal(uid, frame["uid"][mask]):
            raise AssertionError(f"DIST p0 order differs from STRONGEST_CURRENT on {fold}")
        if not np.allclose(np.asarray(pact["y"], float)[order], frame["y"][mask], atol=1e-6):
            raise AssertionError(f"DIST p0 targets differ on {fold}")
        values = np.asarray(pact["p0"], float)[order]
        if not np.all((values >= 0.0) & (values <= 1.0)):
            raise AssertionError(f"p0 outside [0,1] on {fold}")
        if not np.array_equal(np.asarray(pact["p_act"], float),
                              1.0 - np.asarray(pact["p0"], float)):
            raise AssertionError(f"saved activity signal is not 1-p0 on {fold}")
        dz = float(np.max(np.abs(np.asarray(pact["z_ref"], float)[order]
                                      - frame["z_dist"][mask])))
        if dz > 1e-6:
            raise AssertionError(f"DIST reconstruction differs by {dz:g} on {fold}")
        p0[mask] = values
        auc = _auc(frame["y"][mask], 1.0 - values)
        aucs.append(auc)
        audit["folds"][fold] = {
            "n": int(mask.sum()), "p0_min": float(values.min()),
            "p0_max": float(values.max()), "mean_activity_signal": float((1-values).mean()),
            "activity_auc": auc, "max_abs_dist_z": dz,
        }
    audit["activity_auc_weighted"] = _wavg(np.asarray(aucs))
    return p0, audit


def _load_segments(frame: dict) -> tuple[np.ndarray, np.ndarray]:
    import polars as pl

    path = ROOT / "research" / "rmsle_diagnostics" / "fold_predictions.parquet"
    diagnostic = pl.read_parquet(path, columns=["cutoff", "user_id", "y", "rec_buy",
                                                 "w180_days_buy"])
    if diagnostic.select(pl.struct(["cutoff", "user_id"]).n_unique()).item() != diagnostic.height:
        raise AssertionError("duplicate keys in rmsle diagnostics frame")
    keys = pl.DataFrame({"_row": np.arange(len(frame["y"])), "cutoff": frame["cutoff"],
                         "user_id": frame["uid"]})
    joined = keys.join(diagnostic, on=["cutoff", "user_id"], how="left").sort("_row")
    if joined.height != len(frame["y"]) or joined["y"].null_count():
        raise AssertionError("segment diagnostics do not cover current OOF")
    if not np.allclose(joined["y"].to_numpy(), frame["y"], atol=1e-6):
        raise AssertionError("segment diagnostics targets differ from current OOF")
    return (joined["rec_buy"].to_numpy().astype(float),
            joined["w180_days_buy"].to_numpy().astype(float))


def load_frame() -> tuple[dict, dict]:
    frame = _load_component_oof()
    folds = [V.isoformat() for V in VAL_FOLDS_S1]
    if sorted(set(frame["cutoff"].tolist())) != folds:
        raise AssertionError("OOF fold set differs from the registered four folds")
    frame["fold_index"] = np.asarray([folds.index(value) for value in frame["cutoff"]], np.int8)
    frame["p0"], p0_audit = _load_p0(frame)
    frame["rec_buy"], frame["w180_days_buy"] = _load_segments(frame)
    counts = np.bincount(frame["fold_index"], minlength=4)
    frame["row_weight"] = (np.asarray(FOLD_WEIGHTS_S1, float)[frame["fold_index"]]
                           / counts[frame["fold_index"]])
    z_shape, pred_shape, fold_means = prediction_shape(frame["z_base"], frame["fold_index"])
    frame["z_shape"], frame["pred_shape"] = z_shape, pred_shape
    frame["amount_bin"] = amount_bins(pred_shape)
    frame["residual"], frame["base_offsets"], frame["base_scores"] = calibrated_residuals(
        frame["y"], frame["z_base"], frame["fold_index"])
    base_wcv = _wavg(frame["base_scores"])
    if np.max(np.abs(frame["base_scores"] - EXPECTED_BASE_FOLDS)) > 1e-6:
        raise AssertionError(f"STRONGEST_CURRENT folds not reproduced: {frame['base_scores']}")
    if abs(base_wcv - EXPECTED_BASE_WCV) > 1e-6:
        raise AssertionError(f"STRONGEST_CURRENT wCV not reproduced: {base_wcv}")
    audit = {
        "n": len(frame["y"]), "folds": folds, "fold_counts": counts,
        "base_components": BASE_COMPONENTS, "base_offsets": frame["base_offsets"],
        "base_scores": frame["base_scores"], "base_wcv": base_wcv,
        "expected_base_wcv": EXPECTED_BASE_WCV,
        "baseline_matches_reference": True, "fold_prediction_means": fold_means,
        "eta_zero_max_abs_difference": float(np.max(np.abs(
            apply_log_correction(frame["z_base"], np.ones(len(frame["y"])), 0.0)
            - frame["z_base"]))),
        "p0": p0_audit,
        "late_or_test_labels_used": False,
    }
    return frame, audit


def shuffled_p0(frame: dict) -> np.ndarray:
    """Deterministic seed-42 shuffle inside every fold x amount cell."""
    if SEED != 42:
        raise AssertionError("registered shuffled-p0 control requires config.SEED == 42")
    out = np.asarray(frame["p0"], float).copy()
    rng = np.random.default_rng(SEED)
    for fold in range(4):
        for amount_id in range(7):
            rows = np.flatnonzero((frame["fold_index"] == fold)
                                  & (frame["amount_bin"] == amount_id))
            out[rows] = rng.permutation(out[rows])
    return out


def _cell_rows(mapping: dict, outer_fold: str, method: str) -> list[dict]:
    rows = []
    edges = np.asarray(mapping["edges"], float)
    nq = mapping["correction"].shape[1]
    for amount_id in range(7):
        for p0_id in range(nq):
            n = int(mapping["counts"][amount_id, p0_id])
            lo = -math.inf if p0_id == 0 else float(edges[p0_id - 1])
            hi = math.inf if (not len(edges) or p0_id == len(edges)) else float(edges[p0_id])
            rows.append({
                "method": method, "outer_fold": outer_fold,
                "fitting_folds": [f for f in [V.isoformat() for V in VAL_FOLDS_S1]
                                  if f != outer_fold],
                "amount_bin": AMOUNT_LABELS[amount_id], "amount_bin_id": amount_id,
                "p0_bin": p0_id, "p0_lo": lo, "p0_hi": hi,
                "n_cell": n, "weight_sum": mapping["weight_sum"][amount_id, p0_id],
                "c_raw": mapping["c_raw"][amount_id, p0_id],
                "reliability": n / (n + SHRINK_STRENGTH) if n else 0.0,
                "c_pre_isotonic": mapping["c_shrunk"][amount_id, p0_id],
                "correction": mapping["correction"][amount_id, p0_id],
                "eligible": n >= MIN_CELL_ROWS,
            })
    return rows


def run_nested(frame: dict, method: str, p0_values: np.ndarray | None = None) -> dict:
    """Outer-fold honest mapping and eta selection for one registered method."""
    two_dim = method in {"ZERO2D", "SHUFFLED_P0"}
    p = np.asarray(frame["p0"] if p0_values is None else p0_values, float)
    fi = frame["fold_index"]
    folds = [V.isoformat() for V in VAL_FOLDS_S1]
    held_score = np.empty(4, float)
    held_eta = np.empty(4, float)
    held_correction = np.zeros(len(fi), float)
    nested_rows, cell_rows = [], []

    for held in range(4):
        outer = fi == held
        train = ~outer
        mapping = fit_mapping(frame["amount_bin"], p, frame["residual"],
                              frame["row_weight"], train, two_dim=two_dim)
        correction = mapping_correction(mapping, frame["amount_bin"])
        eta_curve = []
        curve_by_fold = []
        fitting = [fold for fold in range(4) if fold != held]
        for eta in ETA_GRID:
            fold_scores = []
            for fold in fitting:
                mask = fi == fold
                z_new = apply_log_correction(frame["z_base"][mask], correction[mask], eta)
                fold_scores.append(calibrate(frame["y"][mask], z_new)[1])
            eta_curve.append(_wavg(np.asarray(fold_scores), np.asarray(fitting)))
            curve_by_fold.append(fold_scores)
        eta_curve = np.asarray(eta_curve)
        best = float(eta_curve.min())
        tied = np.flatnonzero(np.isclose(eta_curve, best, rtol=0.0, atol=1e-12))
        selected_index = int(tied[0])
        eta = float(ETA_GRID[selected_index])
        z_outer = apply_log_correction(frame["z_base"][outer], correction[outer], eta)
        held_score[held] = calibrate(frame["y"][outer], z_outer)[1]
        held_eta[held] = eta
        held_correction[outer] = correction[outer]
        row = {
            "method": method, "outer_fold": folds[held],
            "fitting_folds": [folds[i] for i in fitting],
            "p0_edges": mapping["edges"], "selected_eta": eta,
            "selection_wcv": best, "base_rmsle": frame["base_scores"][held],
            "candidate_rmsle": held_score[held],
            "delta": held_score[held] - frame["base_scores"][held],
        }
        for eta_index, eta_value in enumerate(ETA_GRID):
            row[f"eta_{eta_value:.2f}_selection_wcv"] = eta_curve[eta_index]
            row[f"eta_{eta_value:.2f}_fitting_scores"] = curve_by_fold[eta_index]
        nested_rows.append(row)
        cell_rows.extend(_cell_rows(mapping, folds[held], method))

    delta = held_score - frame["base_scores"]
    result = {
        "method": method, "heldout_scores": held_score,
        "heldout_delta": delta, "heldout_eta": held_eta,
        "correction": held_correction,
        "z_honest": apply_log_correction(frame["z_base"], held_correction,
                                          held_eta[fi]),
        "base_wcv": _wavg(frame["base_scores"]),
        "candidate_wcv": _wavg(held_score), "delta_wcv": _wavg(delta),
        "improved_folds": int((delta < 0).sum()),
        "nested_rows": nested_rows, "cell_rows": cell_rows,
    }
    return result


def amount_diagnostics(frame: dict) -> tuple[list[dict], list[dict], np.ndarray]:
    fi = frame["fold_index"]
    ly = np.log1p(frame["y"])
    offsets = frame["base_offsets"][fi]
    z_cal = np.maximum(frame["z_base"] + offsets, 0.0)
    error = np.square(ly - z_cal)
    residual = ly - (frame["z_base"] + offsets)
    rows, zero_rows = [], []
    scopes = [(V.isoformat(), fi == i, np.ones(len(fi), float))
              for i, V in enumerate(VAL_FOLDS_S1)]
    scopes.append(("AGGREGATE", np.ones(len(fi), bool), frame["row_weight"]))
    for scope, scope_mask, scope_weight in scopes:
        weights = scope_weight[scope_mask]
        total_error = float(np.sum(weights * error[scope_mask]))
        zero_scope = scope_mask & (frame["y"] == 0)
        total_zero_error = float(np.sum(scope_weight[zero_scope] * error[zero_scope]))
        scope_mass = float(weights.sum())
        zero_mass = float(scope_weight[zero_scope].sum())
        for amount_id, label in enumerate(AMOUNT_LABELS):
            mask = scope_mask & (frame["amount_bin"] == amount_id)
            w = scope_weight[mask]
            if not mask.any():
                continue
            zero = mask & (frame["y"] == 0)
            positive = mask & (frame["y"] > 0)
            rows.append({
                "fold": scope, "amount_bin": label, "n": int(mask.sum()),
                "user_share": float(w.sum() / scope_mass),
                "actual_zero_rate": float(scope_weight[zero].sum() / w.sum()),
                "mean_y_true": _weighted_mean(frame["y"][mask], w),
                "median_y_true": _weighted_median(frame["y"][mask], w),
                "mean_prediction": _weighted_mean(frame["pred_shape"][mask], w),
                "median_prediction": _weighted_median(frame["pred_shape"][mask], w),
                "mean_p0": _weighted_mean(frame["p0"][mask], w),
                "rmsle": math.sqrt(_weighted_mean(error[mask], w)),
                "error_share": float(np.sum(w * error[mask]) / total_error),
                "zero_row_error_share": float(np.sum(scope_weight[zero] * error[zero]) / total_error),
                "positive_row_error_share": float(np.sum(scope_weight[positive] * error[positive]) / total_error),
                "mean_calibrated_residual": _weighted_mean(residual[mask], w),
            })
            wz = scope_weight[zero]
            if zero.any():
                zero_rows.append({
                    "fold": scope, "amount_bin": label, "n_zero": int(zero.sum()),
                    "share_of_all_zero_users": float(wz.sum() / zero_mass),
                    "mean_prediction": _weighted_mean(frame["pred_shape"][zero], wz),
                    "mean_squared_log_error": _weighted_mean(error[zero], wz),
                    "share_of_total_zero_error": float(np.sum(wz * error[zero]) / total_zero_error),
                    "share_of_total_error": float(np.sum(wz * error[zero]) / total_error),
                })

    # Diagnostic-only p0 quintiles, defined independently inside each fold.
    p0_diag_bin = np.empty(len(fi), np.int8)
    for fold in range(4):
        mask = fi == fold
        p0_diag_bin[mask] = assign_p0_bins(frame["p0"][mask],
                                           np.quantile(frame["p0"][mask], P0_QUANTILES))
    return rows, zero_rows, p0_diag_bin


def _calibrated_predictions(frame: dict, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fi = frame["fold_index"]
    out = np.empty(len(fi), float)
    offsets, scores = [], []
    for fold in range(4):
        mask = fi == fold
        offset, score = calibrate(frame["y"][mask], np.asarray(z)[mask])
        out[mask] = np.maximum(np.asarray(z)[mask] + offset, 0.0)
        offsets.append(offset)
        scores.append(score)
    return out, np.asarray(offsets), np.asarray(scores)


def error_decomposition(frame: dict, zero2d: dict) -> list[dict]:
    rows = []
    fi = frame["fold_index"]
    ly = np.log1p(frame["y"])
    for model, z_raw in [("BASE", frame["z_base"]), ("ZERO2D", zero2d["z_honest"])]:
        z_cal, offsets, scores = _calibrated_predictions(frame, z_raw)
        fold_rows = []
        for fold, V in enumerate(VAL_FOLDS_S1):
            mask = fi == fold
            zero = mask & (frame["y"] == 0)
            positive = mask & (frame["y"] > 0)
            err = np.square(ly[mask] - z_cal[mask])
            n = int(mask.sum())
            row = {
                "model": model, "fold": V.isoformat(), "n": n,
                "optimal_log_shift": offsets[fold], "rmsle_all": scores[fold],
                "mse_contribution_y0": float(np.square(ly[zero] - z_cal[zero]).sum() / n),
                "mse_contribution_y_positive": float(np.square(ly[positive] - z_cal[positive]).sum() / n),
                "rmsle_zero_rows": float(np.sqrt(np.mean(np.square(ly[zero] - z_cal[zero])))),
                "rmsle_positive_rows": float(np.sqrt(np.mean(np.square(ly[positive] - z_cal[positive])))),
                "auc_y_positive": _auc(frame["y"][mask], np.asarray(z_raw)[mask]),
            }
            rows.append(row)
            fold_rows.append(row)
        aggregate = {"model": model, "fold": "AGGREGATE", "n": len(fi),
                     "optimal_log_shift": float("nan")}
        for column in ["rmsle_all", "mse_contribution_y0", "mse_contribution_y_positive",
                       "rmsle_zero_rows", "rmsle_positive_rows", "auc_y_positive"]:
            aggregate[column] = _wavg(np.asarray([row[column] for row in fold_rows]))
        rows.append(aggregate)
    return rows


def segment_diagnostics(frame: dict, zero2d: dict) -> list[dict]:
    fi = frame["fold_index"]
    base_cal, _, _ = _calibrated_predictions(frame, frame["z_base"])
    new_cal, _, _ = _calibrated_predictions(frame, zero2d["z_honest"])
    ly = np.log1p(frame["y"])
    rec, buy = frame["rec_buy"], frame["w180_days_buy"]
    segments = {
        "rec_buy 15–60": np.isfinite(rec) & (rec >= 15) & (rec <= 60),
        "w180_days_buy 0–1": np.isfinite(buy) & (buy >= 0) & (buy <= 1),
        "w180_days_buy 2–15": np.isfinite(buy) & (buy >= 2) & (buy <= 15),
        "w180_days_buy >=16": np.isfinite(buy) & (buy >= 16),
        "never purchased": ~np.isfinite(rec),
    }
    rows = []
    for name, segment in segments.items():
        fold_rows = []
        for fold, V in enumerate(VAL_FOLDS_S1):
            scope = segment & (fi == fold)
            base_error = np.square(ly[scope] - base_cal[scope])
            new_error = np.square(ly[scope] - new_cal[scope])
            row = {
                "fold": V.isoformat(), "segment": name, "n": int(scope.sum()),
                "user_share": float(scope.sum() / np.sum(fi == fold)),
                "base_rmsle": float(np.sqrt(base_error.mean())),
                "zero2d_rmsle": float(np.sqrt(new_error.mean())),
                "delta_rmsle": float(np.sqrt(new_error.mean()) - np.sqrt(base_error.mean())),
                "base_mse": float(base_error.mean()), "zero2d_mse": float(new_error.mean()),
                "delta_mse": float(new_error.mean() - base_error.mean()),
            }
            rows.append(row)
            fold_rows.append(row)
        aggregate = {"fold": "AGGREGATE", "segment": name,
                     "n": int(segment.sum()), "user_share": _wavg(np.asarray(
                         [row["user_share"] for row in fold_rows]))}
        for column in ["base_rmsle", "zero2d_rmsle", "delta_rmsle",
                       "base_mse", "zero2d_mse", "delta_mse"]:
            aggregate[column] = _wavg(np.asarray([row[column] for row in fold_rows]))
        rows.append(aggregate)
    return rows


def hard_zero_diagnostics(frame: dict) -> list[dict]:
    fi = frame["fold_index"]
    ly = np.log1p(frame["y"])
    base_cal, _, base_scores = _calibrated_predictions(frame, frame["z_base"])
    rows = []
    for fraction in (0.05, 0.10):
        fold_rows = []
        for fold, V in enumerate(VAL_FOLDS_S1):
            scope_rows = np.flatnonzero(fi == fold)
            k = int(math.ceil(fraction * len(scope_rows)))
            top = scope_rows[np.argsort(frame["p0"][scope_rows], kind="mergesort")[-k:]]
            z_new = frame["z_base"][scope_rows].copy()
            selected = np.isin(scope_rows, top, assume_unique=True)
            z_new[selected] = 0.0
            offset, score = calibrate(frame["y"][scope_rows], z_new)
            z_new_cal = np.maximum(z_new + offset, 0.0)
            zero = frame["y"][scope_rows] == 0
            positive = ~zero
            base_err = np.square(ly[scope_rows] - base_cal[scope_rows])
            new_err = np.square(ly[scope_rows] - z_new_cal)
            row = {
                "method": f"HARD_ZERO_TOP_{int(fraction*100)}PCT", "fold": V.isoformat(),
                "fraction": fraction, "n_zeroed": k,
                "base_rmsle": base_scores[fold], "rmsle_all": score,
                "delta_rmsle": score - base_scores[fold],
                "rmsle_actual_zeros": float(np.sqrt(new_err[zero].mean())),
                "rmsle_positives": float(np.sqrt(new_err[positive].mean())),
                "zero_mse_contribution_base": float(base_err[zero].sum()/len(scope_rows)),
                "zero_mse_contribution_new": float(new_err[zero].sum()/len(scope_rows)),
                "positive_mse_contribution_base": float(base_err[positive].sum()/len(scope_rows)),
                "positive_mse_contribution_new": float(new_err[positive].sum()/len(scope_rows)),
                "share_improved_zero_error": float((base_err[zero].sum()-new_err[zero].sum())
                                                    / base_err[zero].sum()),
                "share_lost_positive_error": float((new_err[positive].sum()-base_err[positive].sum())
                                                   / base_err[positive].sum()),
            }
            rows.append(row)
            fold_rows.append(row)
        aggregate = {"method": fold_rows[0]["method"], "fold": "AGGREGATE",
                     "fraction": fraction, "n_zeroed": sum(r["n_zeroed"] for r in fold_rows)}
        for column in ["base_rmsle", "rmsle_all", "delta_rmsle", "rmsle_actual_zeros",
                       "rmsle_positives", "zero_mse_contribution_base",
                       "zero_mse_contribution_new", "positive_mse_contribution_base",
                       "positive_mse_contribution_new"]:
            aggregate[column] = _wavg(np.asarray([row[column] for row in fold_rows]))
        aggregate["share_improved_zero_error"] = (
            (aggregate["zero_mse_contribution_base"]-aggregate["zero_mse_contribution_new"])
            / aggregate["zero_mse_contribution_base"])
        aggregate["share_lost_positive_error"] = (
            (aggregate["positive_mse_contribution_new"]
             - aggregate["positive_mse_contribution_base"])
            / aggregate["positive_mse_contribution_base"])
        rows.append(aggregate)
    return rows


def fit_production_mapping(frame: dict) -> dict:
    """Full-four-fold mapping and production eta after the nested result exists."""
    train = np.ones(len(frame["y"]), bool)
    mapping = fit_mapping(frame["amount_bin"], frame["p0"], frame["residual"],
                          frame["row_weight"], train, two_dim=True)
    correction = mapping_correction(mapping, frame["amount_bin"])
    curve, fold_curve = [], []
    for eta in ETA_GRID:
        scores = []
        for fold in range(4):
            mask = frame["fold_index"] == fold
            z_new = apply_log_correction(frame["z_base"][mask], correction[mask], eta)
            scores.append(calibrate(frame["y"][mask], z_new)[1])
        fold_curve.append(scores)
        curve.append(_wavg(np.asarray(scores)))
    curve = np.asarray(curve)
    best = float(curve.min())
    tied = np.flatnonzero(np.isclose(curve, best, rtol=0.0, atol=1e-12))
    index = int(tied[0])
    return {"mapping": mapping, "correction_oof": correction,
            "eta": float(ETA_GRID[index]), "eta_curve_wcv": curve,
            "eta_curve_folds": np.asarray(fold_curve),
            "selected_wcv": best}


def infer_dist_p0_test(resume: bool = True, allow_same_recipe_mismatch: bool = False) -> dict:
    """Reproduce the production S1-DIST head and retain its test p0 column.

    The original test artifact was built through ``src.predict.train_full``.
    Keep that exact execution path here: unlike the OOF extraction helper it
    deliberately retains the dense matrix while LightGBM constructs/trains its
    Dataset.  A same-recipe rebuild mismatch is persisted with full diagnostics
    before it can stop the caller; the explicit LB-probe override may use it,
    but normal production must not.
    """
    path = ARTIFACTS / "ZERO2D_DIST_test.npz"
    if resume and path.exists():
        _log(f"loading cached production DIST probabilities: {path.name}")
        return dict(np.load(path, allow_pickle=False))

    import datetime as dt
    from src.data import load
    from src.features import feature_names, to_np
    from src.predict import train_full
    from src.train import Setup, xy

    started = time.time()
    setup = Setup(L=0, min_history=90, step=7, panel_blocks=3, train_blocks=1,
                  model="dist", rounds=250, params={"seed": SEED}, norm_long=True,
                  vals=list(VAL_FOLDS_S1))
    cuts = setup.grid()
    if len(cuts) != 29 or cuts[-1] != dt.date(2025, 10, 16):
        raise AssertionError(f"production DIST grid differs: {len(cuts)} cuts, last={cuts[-1]}")
    load()
    Xt, _ = xy(dt.date(2026, 2, 13), setup, with_target=False)
    feats = feature_names(Xt)
    saved_feats = (ARTIFACTS / "feats_S1-DIST-A.txt").read_text(
        encoding="utf-8").splitlines()
    if feats != saved_feats:
        raise AssertionError("production DIST features differ from saved S1-DIST")
    uid = Xt["user_id"].to_numpy()
    At = to_np(Xt, feats)
    _log(f"production DIST: assembling {len(cuts)} clean cutoffs, {len(feats)} features")
    trained = train_full(setup, feats, ["dist"])
    _, (booster, centroids) = trained["dist"]
    _log(f"production DIST: fitted in {time.time()-started:.0f}s; predicting test p0")
    if not np.array_equal(Xt["user_id"].to_numpy(), uid):
        raise AssertionError("test panel order changed during DIST training")
    probabilities = booster.predict(At, num_iteration=250)
    p0 = np.asarray(probabilities[:, 0], float)
    z = np.asarray(probabilities, float) @ np.asarray(centroids, float)
    del At, probabilities, booster, Xt, trained
    gc.collect()
    if not np.all((p0 >= 0.0) & (p0 <= 1.0)):
        raise AssertionError("production DIST p0 outside [0,1]")

    uid_ref = np.load(ARTIFACTS / "uid_S1-DIST.npy")
    z_ref = np.load(ARTIFACTS / "ztest_S1-DIST.npy").astype(float)
    if not np.array_equal(uid, uid_ref):
        raise AssertionError("production DIST test order differs from saved S1-DIST")
    dz = z - z_ref
    max_abs_dz = float(np.max(np.abs(dz)))
    mean_abs_dz = float(np.mean(np.abs(dz)))
    rmse_dz = float(np.sqrt(np.mean(np.square(dz))))
    correlation_z = float(np.corrcoef(z, z_ref)[0, 1])
    reference_reproduced = bool(max_abs_dz <= 1e-6)
    np.savez_compressed(path, user_id=uid, p0=p0, p_act=1.0-p0, z=z,
                        z_ref=z_ref, centroids=np.asarray(centroids),
                        max_abs_dz=max_abs_dz, mean_abs_dz=mean_abs_dz,
                        rmse_dz=rmse_dz, correlation_z=correlation_z,
                        mean_z=float(z.mean()), mean_z_ref=float(z_ref.mean()),
                        reference_reproduced=reference_reproduced, rounds=250,
                        cuts=np.asarray([str(cut) for cut in cuts], dtype="U10"),
                        runtime_s=time.time()-started)
    _log(f"production DIST saved: {path.name}; max|dz|={max_abs_dz:.3e}, "
         f"MAE(dz)={mean_abs_dz:.3e}, corr={correlation_z:.9f}, "
         f"mean p0={p0.mean():.6f}")
    if not reference_reproduced and not allow_same_recipe_mismatch:
        raise AssertionError(f"production DIST head not reproduced: max|dz|={max_abs_dz:g}")
    if not reference_reproduced:
        _log("WARNING: using same-recipe DIST rebuild under explicit LB-probe override")
    return dict(np.load(path, allow_pickle=False))


def load_strongest_test() -> tuple[np.ndarray, np.ndarray]:
    uid = np.load(ARTIFACTS / "uid_S1-CAP.npy")
    z = np.zeros(len(uid), float)
    for name, weight in BASE_TEST_COMPONENTS.items():
        component_uid = np.load(ARTIFACTS / f"uid_{name}.npy")
        if not np.array_equal(component_uid, uid):
            raise AssertionError(f"STRONGEST_CURRENT test order differs for {name}")
        z += float(weight) * np.load(ARTIFACTS / f"ztest_{name}.npy").astype(float)
    return uid, z


def _align_values(uid: np.ndarray, values: np.ndarray,
                  target_uid: np.ndarray) -> np.ndarray:
    order = np.argsort(np.asarray(uid), kind="mergesort")
    sorted_uid = np.asarray(uid)[order]
    positions = np.searchsorted(sorted_uid, np.asarray(target_uid))
    if np.any(positions == len(sorted_uid)) or not np.array_equal(
            sorted_uid[positions], np.asarray(target_uid)):
        raise AssertionError("user_id sets differ during test alignment")
    return np.asarray(values)[order[positions]]


def test_regime_audit(frame: dict, production: dict, test_amount: np.ndarray,
                      test_p0: np.ndarray, test_p0_bin: np.ndarray,
                      test_correction: np.ndarray) -> dict:
    mapping = production["mapping"]
    oof_amount = frame["amount_bin"].astype(int)
    oof_p0_bin = mapping["p0_bin"].astype(int)
    eta = float(production["eta"])
    oof_correction = eta * production["correction_oof"]
    test_correction = np.asarray(test_correction, float)
    oof_cell = oof_amount * 5 + oof_p0_bin
    test_cell = np.asarray(test_amount, int) * 5 + np.asarray(test_p0_bin, int)
    oof_cell_share = np.bincount(oof_cell, minlength=35) / len(oof_cell)
    test_cell_share = np.bincount(test_cell, minlength=35) / len(test_cell)
    count_flat = mapping["counts"].reshape(-1)
    empty = count_flat[test_cell] == 0
    unsupported = count_flat[test_cell] < MIN_CELL_ROWS
    variance_oof = float(np.var(oof_correction))
    variance_test = float(np.var(test_correction))
    variance_ratio = variance_test / variance_oof if variance_oof > 0 else float("inf")
    quantiles = [0.01, 0.05, 0.50, 0.95, 0.99]
    p0_min, p0_max = float(frame["p0"].min()), float(frame["p0"].max())
    outside_p0 = (np.asarray(test_p0) < p0_min) | (np.asarray(test_p0) > p0_max)
    max_cell_shift = float(np.max(np.abs(test_cell_share - oof_cell_share)))
    passed = (0.5 <= variance_ratio <= 1.5
              and float(empty.mean()) <= 0.001
              and float(unsupported.mean()) <= 0.01
              and float(outside_p0.mean()) <= 0.01
              and max_cell_shift <= 0.05)
    return {
        "status": "PASS" if passed else "WARN_OUTSIDE_OOF_SUPPORT",
        "eta": eta,
        "amount_bin_share_oof": np.bincount(oof_amount, minlength=7) / len(oof_amount),
        "amount_bin_share_test": np.bincount(test_amount, minlength=7) / len(test_amount),
        "p0_bin_share_oof": np.bincount(oof_p0_bin, minlength=5) / len(oof_p0_bin),
        "p0_bin_share_test": np.bincount(test_p0_bin, minlength=5) / len(test_p0_bin),
        "cell_share_oof": oof_cell_share.reshape(7, 5),
        "cell_share_test": test_cell_share.reshape(7, 5),
        "max_abs_cell_share_shift": max_cell_shift,
        "fraction_empty_cell": float(empty.mean()),
        "fraction_unsupported_lt500": float(unsupported.mean()),
        "fraction_p0_outside_oof_minmax": float(outside_p0.mean()),
        "variance_correction_oof": variance_oof,
        "variance_correction_test": variance_test,
        "variance_test_over_oof": variance_ratio,
        "correction_test_quantiles": dict(zip(
            ["p01", "p05", "p50", "p95", "p99"],
            np.quantile(test_correction, quantiles))),
        "fraction_nonzero_correction_test": float(np.mean(test_correction != 0.0)),
        "fraction_nonzero_correction_oof": float(np.mean(oof_correction != 0.0)),
    }


def run_production(force_submission: bool = True) -> dict:
    """Build the explicitly requested LB probe after the registered OOF REJECT."""
    import polars as pl
    from src.config import SUBMISSIONS
    from src.data import sample_submit
    from src.submit import check_submission

    RESULTS.mkdir(parents=True, exist_ok=True)
    frame, audit = load_frame()
    production = fit_production_mapping(frame)
    _write_csv(RESULTS / "production_cells.csv",
               _cell_rows(production["mapping"], "PRODUCTION_ALL_OOF", "ZERO2D"))
    _log("full-OOF eta curve: " + " / ".join(
        f"{eta:.2f}={score:.9f}" for eta, score in zip(ETA_GRID, production["eta_curve_wcv"])))
    _log(f"production eta={production['eta']:.2f}")

    dist = infer_dist_p0_test(resume=True, allow_same_recipe_mismatch=force_submission)
    uid, z_test_raw = load_strongest_test()
    if not np.array_equal(uid, dist["user_id"]):
        raise AssertionError("DIST p0 test order differs from STRONGEST_CURRENT")
    z_test_level = set_log_level(z_test_raw, LEVEL)
    test_amount = amount_bins(np.expm1(z_test_level))
    test_p0 = np.asarray(dist["p0"], float)
    test_p0_bin = assign_p0_bins(test_p0, production["mapping"]["edges"])
    test_c = production["mapping"]["correction"][test_amount, test_p0_bin]
    scaled_c = float(production["eta"]) * test_c
    z_corrected = apply_log_correction(z_test_level, test_c, production["eta"])
    z_final = set_log_level(z_corrected, LEVEL)
    regime = test_regime_audit(frame, production, test_amount, test_p0,
                               test_p0_bin, scaled_c)
    regime.update({
        "validation_status": "REJECT_OVERRIDDEN_BY_EXPLICIT_USER_REQUEST",
        "dist_test_max_abs_dz": float(dist["max_abs_dz"]),
        "dist_test_mean_abs_dz": float(dist["mean_abs_dz"]),
        "dist_test_rmse_dz": float(dist["rmse_dz"]),
        "dist_test_correlation_z": float(dist["correlation_z"]),
        "dist_test_mean_z_rebuild": float(dist["mean_z"]),
        "dist_test_mean_z_reference": float(dist["mean_z_ref"]),
        "dist_reference_reproduced": bool(dist["reference_reproduced"]),
        "mean_z_strongest_raw": float(z_test_raw.mean()),
        "mean_z_before_correction_after_level": float(z_test_level.mean()),
        "mean_z_after_correction_before_relevel": float(z_corrected.mean()),
        "mean_z_final": float(z_final.mean()),
        "full_oof_eta_curve": dict(zip([f"{eta:.2f}" for eta in ETA_GRID],
                                        production["eta_curve_wcv"])),
        "submission_forced_after_reject": bool(force_submission),
    })
    _write_json(RESULTS / "test_regime.json", regime)
    if regime["status"] != "PASS" and not force_submission:
        raise AssertionError("test regime is outside OOF support; submission blocked")

    sample = sample_submit()
    sample_uid = sample["user_id"].to_numpy()
    z_ordered = _align_values(uid, z_final, sample_uid)
    p0_ordered = _align_values(uid, test_p0, sample_uid)
    amount_ordered = _align_values(uid, test_amount, sample_uid)
    p0_bin_ordered = _align_values(uid, test_p0_bin, sample_uid)
    correction_ordered = _align_values(uid, scaled_c, sample_uid)
    assert_test_order(sample_uid, sample_uid)
    prediction = np.expm1(z_ordered)
    submission = pl.DataFrame({"user_id": sample_uid,
                               "predict": prediction.astype(np.float64)})
    check_submission(submission)
    if abs(float(np.log1p(prediction).mean()) - LEVEL) > 1e-11:
        raise AssertionError("submission final log level differs from 2.3293")
    out = SUBMISSIONS / "submission_ZERO2D_SHRINK.csv"
    submission.write_csv(out, float_precision=10)
    roundtrip = pl.read_csv(out)
    check_submission(roundtrip)
    roundtrip_level = float(np.log1p(roundtrip["predict"].to_numpy()).mean())
    if abs(roundtrip_level - LEVEL) > 1e-9:
        raise AssertionError(f"serialized submission level differs: {roundtrip_level}")
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    np.save(ARTIFACTS / "ztest_ZERO2D_SHRINK.npy", z_ordered)
    np.save(ARTIFACTS / "uid_ZERO2D_SHRINK.npy", sample_uid)
    np.savez_compressed(
        ARTIFACTS / "ZERO2D_SHRINK_test.npz", user_id=sample_uid,
        p0=p0_ordered, amount_bin=amount_ordered, p0_bin=p0_bin_ordered,
        correction=correction_ordered, z_final=z_ordered,
        eta=production["eta"], p0_edges=production["mapping"]["edges"])
    production_summary = {
        "submission": str(out), "sha256": digest, "rows": submission.height,
        "eta": production["eta"], "final_level": roundtrip_level,
        "zero_fraction": float(np.mean(roundtrip["predict"].to_numpy() == 0)),
        "min_prediction": float(roundtrip["predict"].min()),
        "max_prediction": float(roundtrip["predict"].max()),
        "regime_status": regime["status"], "forced_after_validation_reject": True,
        "dist_reference_reproduced": regime["dist_reference_reproduced"],
        "dist_test_mean_abs_dz": regime["dist_test_mean_abs_dz"],
        "dist_test_correlation_z": regime["dist_test_correlation_z"],
    }
    _write_json(RESULTS / "production_summary.json", production_summary)
    summary_path = RESULTS / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["production"] = production_summary
        summary["test_regime"] = regime
        _write_json(summary_path, summary)
    _log(f"submission saved: {out}; regime={regime['status']}; sha256={digest}")
    return production_summary


def _controls_rows(frame: dict, results: list[dict], hard_rows: list[dict]) -> list[dict]:
    rows = []
    folds = [V.isoformat() for V in VAL_FOLDS_S1]
    for result in results:
        for fold in range(4):
            rows.append({
                "method": result["method"], "fold": folds[fold],
                "selected_eta": result["heldout_eta"][fold],
                "base_rmsle": frame["base_scores"][fold],
                "candidate_rmsle": result["heldout_scores"][fold],
                "delta": result["heldout_delta"][fold],
            })
        rows.append({
            "method": result["method"], "fold": "AGGREGATE",
            "selected_eta": list(result["heldout_eta"]),
            "base_rmsle": result["base_wcv"], "candidate_rmsle": result["candidate_wcv"],
            "delta": result["delta_wcv"], "improved_folds": result["improved_folds"],
        })
    rows.extend(row for row in hard_rows if row["fold"] == "AGGREGATE")
    return rows


def decision_gate(zero2d: dict, amount_only: dict, shuffle: dict,
                  decomposition: list[dict]) -> dict:
    aggregate = {(row["model"], row["fold"]): row for row in decomposition}
    base = aggregate[("BASE", "AGGREGATE")]
    candidate = aggregate[("ZERO2D", "AGGREGATE")]
    zero_gain = base["mse_contribution_y0"] - candidate["mse_contribution_y0"]
    positive_loss = candidate["mse_contribution_y_positive"] - base["mse_contribution_y_positive"]
    positive_not_cancel = zero_gain > 0 and positive_loss < zero_gain
    checks = {
        "delta_wcv_le_minus_0.0005": zero2d["delta_wcv"] <= -0.0005,
        "at_least_3_of_4_folds": zero2d["improved_folds"] >= 3,
        "fold_2025_10_16_improves": zero2d["heldout_delta"][-1] < 0,
        "beats_amount_only_by_0.0002": (
            zero2d["candidate_wcv"] <= amount_only["candidate_wcv"] - 0.0002),
        "shuffled_p0_not_better_than_minus_0.0001": shuffle["delta_wcv"] >= -0.0001,
        "zero_gain_not_cancelled_by_positive_loss": positive_not_cancel,
        "not_only_global_level": float(np.var(zero2d["correction"])) > 1e-12,
    }
    reasons = [name for name, passed in checks.items() if not passed]
    accepted = all(checks.values())
    strong = (accepted and zero2d["delta_wcv"] <= -0.0010
              and zero2d["improved_folds"] == 4
              and np.all(zero2d["heldout_eta"] == zero2d["heldout_eta"][0]))
    return {
        "status": "STRONG ACCEPT" if strong else ("ACCEPT" if accepted else "REJECT"),
        "checks": checks, "failed_checks": reasons,
        "zero_mse_gain": zero_gain, "positive_mse_loss": positive_loss,
        "positive_loss_over_zero_gain": (positive_loss / zero_gain if zero_gain > 0 else None),
    }


def make_plots(frame: dict, amount_rows: list[dict], zero_rows: list[dict],
               p0_diag_bin: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    aggregate_amount = [row for row in amount_rows if row["fold"] == "AGGREGATE"]
    aggregate_zero = [row for row in zero_rows if row["fold"] == "AGGREGATE"]
    labels = [row["amount_bin"] for row in aggregate_amount]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, [row["share_of_total_zero_error"] for row in aggregate_zero], color="#c44e52")
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel("share of all zero-row squared log error")
    ax.set_title("Future-zero error contribution by prediction amount")
    fig.tight_layout()
    fig.savefig(RESULTS / "zero_error_share_by_amount.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, [row["actual_zero_rate"] for row in aggregate_amount], color="#4c72b0")
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel("actual zero rate")
    ax.set_ylim(0, 1)
    ax.set_title("Future-zero rate by prediction amount")
    fig.tight_layout()
    fig.savefig(RESULTS / "zero_rate_by_amount.png", dpi=160)
    plt.close(fig)

    matrix = np.full((7, 5), np.nan)
    for amount_id in range(7):
        for p0_id in range(5):
            mask = ((frame["amount_bin"] == amount_id) & (p0_diag_bin == p0_id))
            if mask.any():
                matrix[amount_id, p0_id] = _weighted_mean(
                    frame["residual"][mask], frame["row_weight"][mask])
    vmax = float(np.nanmax(np.abs(matrix)))
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(5), ["Q1", "Q2", "Q3", "Q4", "Q5"])
    ax.set_yticks(range(7), AMOUNT_LABELS)
    ax.set_xlabel("p0 quintile (within fold; higher = more likely zero)")
    ax.set_ylabel("prediction amount bin")
    ax.set_title("Mean calibrated residual ly - (z_base + fold shift)")
    fig.colorbar(image, ax=ax, label="mean residual")
    fig.tight_layout()
    fig.savefig(RESULTS / "residual_by_amount_and_p0.png", dpi=160)
    plt.close(fig)


def run() -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    _log("loading current OOF components, DIST p0, and cutoff-safe diagnostics")
    frame, audit = load_frame()
    _write_json(RESULTS / "audit.json", audit)
    _log("baseline: " + " / ".join(f"{value:.9f}" for value in frame["base_scores"])
         + f"; wCV={_wavg(frame['base_scores']):.9f}")

    amount_rows, zero_rows, p0_diag_bin = amount_diagnostics(frame)
    _write_csv(RESULTS / "amount_diagnostics.csv", amount_rows)
    _write_csv(RESULTS / "zero_error_by_amount.csv", zero_rows)
    make_plots(frame, amount_rows, zero_rows, p0_diag_bin)

    _log("running honest outer-fold ZERO2D")
    zero2d = run_nested(frame, "ZERO2D")
    _log("running AMOUNT-ONLY control")
    amount_only = run_nested(frame, "AMOUNT_ONLY")
    _log("running shuffled-p0 control")
    shuffle = run_nested(frame, "SHUFFLED_P0", shuffled_p0(frame))

    nested_rows = zero2d["nested_rows"] + [{
        "method": "ZERO2D", "outer_fold": "AGGREGATE",
        "selected_eta": zero2d["heldout_eta"], "base_rmsle": zero2d["base_wcv"],
        "candidate_rmsle": zero2d["candidate_wcv"], "delta": zero2d["delta_wcv"],
        "improved_folds": zero2d["improved_folds"],
    }]
    _write_csv(RESULTS / "nested_lofo.csv", nested_rows)
    _write_csv(RESULTS / "zero2d_cells.csv", zero2d["cell_rows"])

    decomposition = error_decomposition(frame, zero2d)
    segments = segment_diagnostics(frame, zero2d)
    hard_rows = hard_zero_diagnostics(frame)
    controls = _controls_rows(frame, [zero2d, amount_only, shuffle], hard_rows)
    _write_csv(RESULTS / "controls.csv", controls)
    _write_csv(RESULTS / "hard_zero.csv", hard_rows)
    _write_csv(RESULTS / "error_decomposition.csv", decomposition)
    _write_csv(RESULTS / "segments.csv", segments)

    gate = decision_gate(zero2d, amount_only, shuffle, decomposition)
    test_regime = {
        "status": "NOT_RUN_VALIDATION_REJECT" if gate["status"] == "REJECT" else "PENDING",
        "validation_status": gate["status"],
        "submission_created": False,
        "reason": ("Production mapping, DIST p0 test inference, regime audit, and submission are "
                   "forbidden after validation REJECT." if gate["status"] == "REJECT" else
                   "Validation accepted; production stage must now be run."),
    }
    _write_json(RESULTS / "test_regime.json", test_regime)

    summary = {
        "baseline": {"fold_scores": frame["base_scores"], "wcv": zero2d["base_wcv"]},
        "zero2d": {key: zero2d[key] for key in ["heldout_scores", "heldout_delta",
                                                  "heldout_eta", "candidate_wcv",
                                                  "delta_wcv", "improved_folds"]},
        "amount_only": {key: amount_only[key] for key in ["heldout_scores", "heldout_delta",
                                                            "heldout_eta", "candidate_wcv",
                                                            "delta_wcv", "improved_folds"]},
        "shuffled_p0": {key: shuffle[key] for key in ["heldout_scores", "heldout_delta",
                                                        "heldout_eta", "candidate_wcv",
                                                        "delta_wcv", "improved_folds"]},
        "gate": gate, "test_regime": test_regime,
        "config": {"amount_edges": AMOUNT_EDGES, "p0_quantiles": P0_QUANTILES,
                   "eta_grid": ETA_GRID, "strength": SHRINK_STRENGTH,
                   "min_cell_rows": MIN_CELL_ROWS, "fold_weights": FOLD_WEIGHTS_S1,
                   "shuffle_seed_from_config": SEED, "level": LEVEL},
    }
    _write_json(RESULTS / "summary.json", summary)
    _log(f"ZERO2D delta={zero2d['delta_wcv']:+.9f}, "
         f"AMOUNT_ONLY={amount_only['delta_wcv']:+.9f}, "
         f"SHUFFLED_P0={shuffle['delta_wcv']:+.9f}; {gate['status']}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--production", action="store_true",
                        help="build explicitly requested LB submission after OOF analysis")
    parser.add_argument("--no-resume-dist", action="store_true",
                        help="retrain production DIST even if its p0 artifact exists")
    args = parser.parse_args()
    if args.production:
        if args.no_resume_dist:
            p = ARTIFACTS / "ZERO2D_DIST_test.npz"
            if p.exists():
                raise AssertionError("refusing to overwrite cached DIST artifact; move it explicitly")
        run_production(force_submission=True)
    else:
        run()
