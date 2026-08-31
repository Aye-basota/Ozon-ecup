"""EXP083: information-loss discovery above the production-like span.

This is deliberately an oracle/low-capacity diagnostic, not a model search.
It reuses the audited EXP080 production residual and full 40-component
historical span, then tests:

* fine multi-horizon future shape (oracle-only);
* channel-specific funnel geometry lost by the canonical tabular aggregates;
* within-day funnel coherence/Jensen statistics;
* novelty of requested temporal-phase descriptors against prior experiments.

No TEST inference, direct GMV model, residual learner, or leaderboard input is
used.  Fully purged observable evidence is the sole available transition
2025-09-04 -> 2025-10-16 (the source 30-day label is known by the destination
cutoff).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EXP080 = ROOT / "research" / "new_directions" / "EXP080_ORACLE_GAP_ATTRIBUTION"
EXP075 = ROOT / "research" / "new_directions" / "EXP075_OUT_OF_SPAN_RESIDUAL_SIGNALS"
EXP077 = ROOT / "research" / "new_directions" / "EXP077_FORWARD_STACK"
FOLDS = ("2025-09-04", "2025-09-18", "2025-10-02", "2025-10-16")
FOLD_WEIGHTS = np.asarray([1.0, 2.0, 4.0, 8.0], np.float64)
SOURCE_FOLD = FOLDS[0]
PURGED_FOLD = FOLDS[-1]
REQUIRED_MSE = 0.004909273595109509
BOOTSTRAP_REPS = 1000
BOOTSTRAP_SEED = 20260829
EPS = 1e-12


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


e80 = load_module(EXP080 / "run_oracle.py", "exp083_e80")
sys.path.insert(0, str(EXP080))
e80obs = load_module(EXP080 / "run_observable.py", "exp083_e80obs")
exp077 = load_module(EXP077 / "run_exp077.py", "exp083_exp077")


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, np.float64)
    return float(np.sqrt(np.mean(x * x)))


def corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, np.float64) - float(np.mean(x))
    y = np.asarray(y, np.float64) - float(np.mean(y))
    den = math.sqrt(float(x @ x) * float(y @ y))
    return 0.0 if den <= 1e-300 else float(x @ y / den)


def safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    num, den = np.asarray(num, np.float64), np.asarray(den, np.float64)
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


def binary_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, np.float64), 0.0, 1.0)
    out = np.zeros_like(p)
    middle = (p > 0) & (p < 1)
    out[middle] = -(p[middle] * np.log(p[middle])
                    + (1 - p[middle]) * np.log1p(-p[middle]))
    return out


def quantile_design(X: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """Tie-preserving cohort quantile bins, fixed width for every fold."""
    X = np.asarray(X, np.float64)
    if X.ndim == 1:
        X = X[:, None]
    out = []
    probs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    for j in range(X.shape[1]):
        x = np.nan_to_num(X[:, j], nan=0.0, posinf=1e12, neginf=-1e12)
        edges = np.unique(np.quantile(x, probs))
        label = np.searchsorted(edges, x, side="right")
        for k in range(n_bins):
            out.append((label == k).astype(np.float64))
    return np.column_stack(out) if out else np.empty((len(X), 0), np.float64)


def fixed_label_design(labels: np.ndarray, n_values: int) -> np.ndarray:
    labels = np.asarray(labels, np.int64)
    return np.column_stack([(labels == k).astype(np.float64) for k in range(n_values)])


def count_bucket(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return np.select([x == 0, x == 1, x == 2, x == 3, x <= 5], [0, 1, 2, 3, 4], default=5).astype(np.int8)


def event_bucket(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return np.select([x == 0, x <= 3, x <= 7, x <= 14], [0, 1, 2, 3], default=4).astype(np.int8)


def moments(U: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return U.T @ U / len(r), U.T @ r / len(r)


def beta_from(U: np.ndarray, r: np.ndarray) -> np.ndarray:
    G, b = moments(U, r)
    return np.linalg.pinv(G, rcond=1e-9) @ b


def correction_metrics(q: np.ndarray, r: np.ndarray) -> dict[str, float]:
    delta = (r - q) ** 2 - r ** 2
    return {
        "rho": corr(q, r) if np.any(q) else 0.0,
        "Delta_MSE": float(np.mean(delta)),
        "Delta_RMSLE": rms(r - q) - rms(r),
        "correction_RMS": rms(q),
        "oracle_gain_at_validation": max(0.0, 2 * float(np.mean(q * r)) - float(np.mean(q * q))),
    }


def poisson_bootstrap(delta: np.ndarray, reps: int, seed: int) -> dict[str, Any]:
    delta = np.asarray(delta, np.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty(reps, np.float64)
    for start in range(0, reps, 20):
        n = min(20, reps - start)
        weights = rng.poisson(1.0, size=(n, len(delta))).astype(np.float32)
        draws[start:start + n] = (weights @ delta) / np.maximum(weights.sum(axis=1), 1.0)
    return {
        "reps": reps,
        "seed": seed,
        "method": "user-level Poisson bootstrap (one row per user in the purged fold)",
        "P_gain": float(np.mean(draws < 0)),
        "CI95_Delta_MSE": np.quantile(draws, [0.025, 0.975]).tolist(),
        "mean_Delta_MSE": float(draws.mean()),
    }


def load_context() -> dict[str, Any]:
    work = np.load(EXP080 / "oracle_working_arrays.npz", allow_pickle=True)
    canon = pd.read_parquet(EXP075 / "clean_forward_predictions.parquet")
    canon["cutoff"] = canon.cutoff.astype(str)
    cut = work["cutoff"].astype(str)
    uid = work["user_id"].astype(np.int64)
    z_current = work["z_current"].astype(np.float64)
    z_match = work["z_match"].astype(np.float64)
    d_post = work["d_exp075_postspan"].astype(np.float64)
    residual = work["residual_current"].astype(np.float64)
    target_log = work["target_log"].astype(np.float64)
    masks = {f: cut == f for f in FOLDS}
    Z, bank_audit = exp077.load_reference_bank(canon)
    bases = {
        f: np.column_stack([
            np.ones(masks[f].sum()), z_current[masks[f]], z_match[masks[f]],
            Z[masks[f]], d_post[masks[f]],
        ]) for f in FOLDS
    }
    return {
        "uid": uid, "cut": cut, "z_current": z_current, "z_match": z_match,
        "d_post": d_post, "residual": residual, "target_log": target_log,
        "masks": masks, "Z": Z, "bases": bases, "bank_audit": bank_audit,
    }


def future_fine(ids: np.ndarray, cutoff: str, uid_all: np.ndarray,
                panel: np.ndarray, gmv: np.ndarray) -> dict[str, np.ndarray]:
    base = e80.future_arrays(ids, cutoff, uid_all, panel, gmv)
    idx = np.searchsorted(uid_all, ids)
    d = int((np.datetime64(cutoff) - e80.DATA_START).astype("timedelta64[D]").astype(int))
    daily = np.asarray(gmv[idx, d + 1:d + 31], np.float64)
    y1 = daily[:, :1].sum(axis=1)
    y3 = daily[:, :3].sum(axis=1)
    y7 = daily[:, :7].sum(axis=1)
    y14 = daily[:, :14].sum(axis=1)
    y30 = daily.sum(axis=1)
    return {**base, "y1": y1, "y3": y3, "y7": y7, "y14": y14, "y30": y30,
            "i1": y1, "i2_3": y3 - y1, "i4_7": y7 - y3,
            "i8_14": y14 - y7, "i15_30": y30 - y14}


def oracle_controls(fut: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    positive = fut["y30"] > 0
    avg_value = safe_ratio(fut["y30"], np.maximum(fut["purchase_days"], 1))
    value_label = np.zeros(len(positive), np.int8)
    if np.any(positive):
        q = np.unique(np.quantile(np.log1p(avg_value[positive]), [0.2, 0.4, 0.6, 0.8]))
        value_label[positive] = 1 + np.searchsorted(q, np.log1p(avg_value[positive]), side="right")
    controls = np.column_stack([
        fixed_label_design(positive.astype(np.int8), 2),
        fixed_label_design(count_bucket(fut["purchase_days"]), 6),
        fixed_label_design(event_bucket(fut["event_days"]), 5),
        fixed_label_design(count_bucket(fut["order_items"]), 6),
        fixed_label_design(value_label, 6),
    ])
    coarse_parts = np.column_stack([fut["y7"], fut["y14"] - fut["y7"], fut["y30"] - fut["y14"]])
    pattern = ((coarse_parts[:, 0] > 0).astype(np.int8)
               + 2 * (coarse_parts[:, 1] > 0).astype(np.int8)
               + 4 * (coarse_parts[:, 2] > 0).astype(np.int8))
    coarse = np.column_stack([
        fixed_label_design(pattern, 8),
        quantile_design(safe_ratio(fut["y7"], fut["y30"]), 5),
        quantile_design(safe_ratio(fut["y30"] - fut["y14"], fut["y30"]), 5),
    ])
    return controls, coarse


def multi_horizon_analysis(ctx: dict[str, Any], uid_all: np.ndarray,
                           panel: np.ndarray, gmv: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], np.ndarray]:
    individual_rows: list[dict[str, Any]] = []
    joint_rows: list[dict[str, Any]] = []
    latest_shape_q = np.zeros(ctx["masks"][PURGED_FOLD].sum(), np.float64)
    names_value = ["Y1", "Y3", "Y7", "Y14"]
    names_inc = ["I1", "I2_3", "I4_7", "I8_14", "I15_30"]
    fold_summary = []
    for fold in FOLDS:
        m = ctx["masks"][fold]
        ids, r, B = ctx["uid"][m], ctx["residual"][m], ctx["bases"][fold]
        fut = future_fine(ids, fold, uid_all, panel, gmv)
        value_map = {"Y1": fut["y1"], "Y3": fut["y3"], "Y7": fut["y7"], "Y14": fut["y14"]}
        inc_map = {"I1": fut["i1"], "I2_3": fut["i2_3"], "I4_7": fut["i4_7"],
                   "I8_14": fut["i8_14"], "I15_30": fut["i15_30"]}
        controls, coarse = oracle_controls(fut)
        for name, raw in {**value_map, **inc_map}.items():
            x = np.log1p(raw)
            g0 = e80.gain_from_design(r, x)
            gs = e80.gain_from_design(r, x, B)
            gc = e80.gain_from_design(r, x, np.column_stack([B, controls]))
            individual_rows.append({
                "cutoff": fold, "candidate": name, "kind": "cumulative" if name.startswith("Y") else "increment",
                "raw_rho": corr(x, r), "raw_oracle_gain": g0["gain"],
                "after_span_rho": gs["rho"], "after_span_oracle_gain": gs["gain"],
                "after_count_value_activity_gain": gc["gain"],
            })
        Dcum = np.column_stack([np.log1p(value_map[n]) for n in names_value])
        Dinc = np.column_stack([np.log1p(inc_map[n]) for n in names_inc])
        ratio = np.column_stack([
            safe_ratio(fut["y1"], fut["y7"]), safe_ratio(fut["y3"], fut["y14"]),
            safe_ratio(fut["y7"], fut["y30"]), safe_ratio(fut["i2_3"], fut["y3"]),
            safe_ratio(fut["i4_7"], fut["y7"]), safe_ratio(fut["i8_14"], fut["y14"]),
            safe_ratio(fut["i15_30"], fut["y30"]),
            np.log1p(safe_ratio(fut["i15_30"], np.full(len(r), 16.0)))
            - np.log1p(safe_ratio(fut["y7"], np.full(len(r), 7.0))),
        ])
        fine_pattern = ((fut["i1"] > 0).astype(np.int8)
                        + 2 * (fut["i2_3"] > 0).astype(np.int8)
                        + 4 * (fut["i4_7"] > 0).astype(np.int8)
                        + 8 * (fut["i8_14"] > 0).astype(np.int8)
                        + 16 * (fut["i15_30"] > 0).astype(np.int8))
        Dshape = np.column_stack([quantile_design(ratio, 5), fixed_label_design(fine_pattern, 32)])
        for mechanism, D in [("cumulative_Y1_Y14", Dcum), ("increments_I1_I15_30", Dinc),
                             ("fine_future_shape", Dshape)]:
            raw = e80.gain_from_design(r, D)
            span = e80.gain_from_design(r, D, B)
            conditioned = e80.gain_from_design(r, D, np.column_stack([B, controls]))
            distinct = e80.gain_from_design(r, D, np.column_stack([B, controls, coarse]))
            joint_rows.append({
                "cutoff": fold, "mechanism": mechanism, "columns": D.shape[1],
                "raw_joint_gain": raw["gain"], "raw_joint_rho": raw["rho"],
                "after_span_joint_gain": span["gain"], "after_span_joint_rho": span["rho"],
                "after_count_value_activity_gain": conditioned["gain"],
                "after_EXP080_coarse_timing_gain": distinct["gain"],
            })
            if fold == PURGED_FOLD and mechanism == "fine_future_shape":
                latest_shape_q = distinct["correction"]
        fold_summary.append({"cutoff": fold, "target_parity_max": float(np.max(np.abs(fut["y30"] - np.expm1(ctx["target_log"][m]))))})

    indiv = pd.DataFrame(individual_rows)
    joint = pd.DataFrame(joint_rows)
    for frame, keys in [(indiv, ["candidate", "kind"]), (joint, ["mechanism"] )]:
        agg = []
        for group_key, part in frame.groupby(keys, sort=False):
            part = part.set_index("cutoff").loc[list(FOLDS)]
            row: dict[str, Any] = {"cutoff": "weighted_1_2_4_8"}
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            row.update(dict(zip(keys, group_key)))
            for col in [c for c in part.columns if c not in keys]:
                row[col] = float(np.average(part[col].to_numpy(np.float64), weights=FOLD_WEIGHTS))
            agg.append(row)
        if agg:
            frame2 = pd.concat([frame, pd.DataFrame(agg)], ignore_index=True)
            if frame is indiv:
                indiv = frame2
            else:
                joint = frame2
    weighted_shape = joint[(joint.cutoff == "weighted_1_2_4_8") & (joint.mechanism == "fine_future_shape")].iloc[0]
    gate = {
        "literal_after_span_shape_oracle": float(weighted_shape.after_span_joint_gain),
        "after_count_value_activity_shape_oracle": float(weighted_shape.after_count_value_activity_gain),
        "distinct_from_EXP080_coarse_timing_oracle": float(weighted_shape.after_EXP080_coarse_timing_gain),
        "gate_threshold": 0.001,
        "passes_literal_after_span": bool(weighted_shape.after_span_joint_gain >= 0.001),
        "passes_novelty_conditioned_gate": bool(weighted_shape.after_EXP080_coarse_timing_gain >= 0.001),
        "models_authorized": bool(weighted_shape.after_span_joint_gain >= 0.001
                                   and weighted_shape.after_EXP080_coarse_timing_gain >= 0.001),
        "old_MHZ_relation": "MHZ used 7/14/21/30/45/60 hazard/count heads; this oracle adds 1/3-day and fine increment shares but conditions on EXP080's coarse timing block.",
        "fold_target_audit": fold_summary,
    }
    return indiv, joint, gate, latest_shape_q


PANEL_CHANNELS = list(e80.PANEL_CHANNELS)
LOG_CHANNELS = {
    "searches", "search_to_cart", "search_to_ord", "cat_to_cart", "cat_to_ord",
    "to_cart", "to_ord", "gmv_search", "gmv_cat", "gmv",
}


def decode(raw: np.ndarray, name: str) -> np.ndarray:
    x = raw[..., PANEL_CHANNELS.index(name)].astype(np.float64)
    return np.expm1(x) if name in LOG_CHANNELS else x


FUNNEL_NAMES = [
    "search_ord_rate30", "search_cart_rate30", "cat_ord_per_catday30", "cat_cart_per_catday30",
    "search_order_share30", "search_cart_share30", "search_aov30", "cat_aov30",
    "order_conversion_disagreement30", "cart_conversion_disagreement30",
    "stage_efficiency_disagreement30", "order_channel_entropy30",
    "search_ord_elasticity7_90", "cat_ord_elasticity7_90",
    "order_share_change7_90", "gmv_share_change7_90",
]

COHERENCE_NAMES = [
    "search_completion_day_rate30", "cat_completion_day_rate30",
    "search_cart_completion_day_rate30", "cat_cart_completion_day_rate30",
    "search_daily_conversion_jensen30", "search_daily_conversion_std90",
    "cat_order_burst_cv90", "dual_channel_order_day_share90",
]


def aggregate_window(arrays: dict[str, np.ndarray], w: int) -> dict[str, np.ndarray]:
    return {k: v[:, -w:].sum(axis=1) for k, v in arrays.items()}


def history_information_features(ids: np.ndarray, cutoff: str, uid_all: np.ndarray,
                                 panel: np.ndarray, chunk: int = 12000) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    idx = np.searchsorted(uid_all, ids)
    d = int((np.datetime64(cutoff) - e80.DATA_START).astype("timedelta64[D]").astype(int))
    funnel = np.empty((len(ids), len(FUNNEL_NAMES)), np.float32)
    coherence = np.empty((len(ids), len(COHERENCE_NAMES)), np.float32)
    parity_searches = []
    for lo in range(0, len(ids), chunk):
        hi = min(len(ids), lo + chunk)
        raw = np.asarray(panel[idx[lo:hi], d - 89:d + 1, :], np.float32)
        a = {name: decode(raw, name) for name in [
            "cat", "searches", "search_to_cart", "search_to_ord", "cat_to_cart", "cat_to_ord",
            "to_cart", "to_ord", "gmv_search", "gmv_cat", "gmv",
        ]}
        s7, s30, s90 = aggregate_window(a, 7), aggregate_window(a, 30), aggregate_window(a, 90)
        def r(s: dict[str, np.ndarray], num: str, den: str) -> np.ndarray:
            return safe_ratio(s[num], s[den])
        so30, sc30 = r(s30, "search_to_ord", "searches"), r(s30, "search_to_cart", "searches")
        co30, cc30 = r(s30, "cat_to_ord", "cat"), r(s30, "cat_to_cart", "cat")
        so7, so90 = r(s7, "search_to_ord", "searches"), r(s90, "search_to_ord", "searches")
        co7, co90 = r(s7, "cat_to_ord", "cat"), r(s90, "cat_to_ord", "cat")
        order_share30 = safe_ratio(s30["search_to_ord"], s30["to_ord"])
        cart_share30 = safe_ratio(s30["search_to_cart"], s30["to_cart"])
        order_share7 = safe_ratio(s7["search_to_ord"], s7["to_ord"])
        order_share90 = safe_ratio(s90["search_to_ord"], s90["to_ord"])
        gmv_share7 = safe_ratio(s7["gmv_search"], s7["gmv"])
        gmv_share90 = safe_ratio(s90["gmv_search"], s90["gmv"])
        f = np.column_stack([
            np.log1p(so30), np.log1p(sc30), np.log1p(co30), np.log1p(cc30),
            order_share30, cart_share30,
            np.log1p(safe_ratio(s30["gmv_search"], s30["search_to_ord"])),
            np.log1p(safe_ratio(s30["gmv_cat"], s30["cat_to_ord"])),
            np.log1p(so30) - np.log1p(co30), np.log1p(sc30) - np.log1p(cc30),
            np.log1p(safe_ratio(s30["search_to_ord"], s30["search_to_cart"]))
            - np.log1p(safe_ratio(s30["cat_to_ord"], s30["cat_to_cart"])),
            binary_entropy(order_share30),
            np.log1p(so7) - np.log1p(so90), np.log1p(co7) - np.log1p(co90),
            order_share7 - order_share90, gmv_share7 - gmv_share90,
        ])
        search_day = a["searches"][:, -30:] > 0
        cat_day = a["cat"][:, -30:] > 0
        s_cart_day = a["search_to_cart"][:, -30:] > 0
        c_cart_day = a["cat_to_cart"][:, -30:] > 0
        s_ord_day = a["search_to_ord"][:, -30:] > 0
        c_ord_day = a["cat_to_ord"][:, -30:] > 0
        daily_s_rate30 = safe_ratio(a["search_to_ord"][:, -30:], a["searches"][:, -30:])
        daily_s_rate90 = safe_ratio(a["search_to_ord"], a["searches"])
        cat90 = a["cat"] > 0
        cat_ord90 = a["cat_to_ord"]
        cat_mean90 = safe_ratio((cat_ord90 * cat90).sum(axis=1), cat90.sum(axis=1))
        cat_var90 = safe_ratio((((cat_ord90 - cat_mean90[:, None]) ** 2) * cat90).sum(axis=1), cat90.sum(axis=1))
        order_day90 = a["to_ord"] > 0
        dual90 = (a["search_to_ord"] > 0) & (a["cat_to_ord"] > 0)
        c = np.column_stack([
            safe_ratio(s_ord_day.sum(axis=1), search_day.sum(axis=1)),
            safe_ratio(c_ord_day.sum(axis=1), cat_day.sum(axis=1)),
            safe_ratio((s_ord_day & s_cart_day).sum(axis=1), s_cart_day.sum(axis=1)),
            safe_ratio((c_ord_day & c_cart_day).sum(axis=1), c_cart_day.sum(axis=1)),
            safe_ratio((daily_s_rate30 * search_day).sum(axis=1), search_day.sum(axis=1)) - so30,
            np.sqrt(np.maximum(safe_ratio((((daily_s_rate90 - safe_ratio(
                (daily_s_rate90 * (a["searches"] > 0)).sum(axis=1),
                (a["searches"] > 0).sum(axis=1))[:, None]) ** 2)
                * (a["searches"] > 0)).sum(axis=1), (a["searches"] > 0).sum(axis=1)), 0.0)),
            np.sqrt(np.maximum(cat_var90, 0.0)) / (cat_mean90 + 1e-6),
            safe_ratio(dual90.sum(axis=1), order_day90.sum(axis=1)),
        ])
        funnel[lo:hi] = np.nan_to_num(f, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
        coherence[lo:hi] = np.nan_to_num(c, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
        parity_searches.append(s30["searches"])
    # Exact decoding audit against the canonical aggregate feature.
    feat = pd.read_parquet(e80.PROCESSED / f"feat_{cutoff.replace('-', '')}_LnormNone.parquet",
                           columns=["user_id", "w30_searches"]).sort_values("user_id")
    source_ids = feat.user_id.to_numpy(np.int64)
    pos = np.searchsorted(source_ids, ids)
    expected = feat.w30_searches.to_numpy(np.float64)[pos]
    actual = np.concatenate(parity_searches)
    audit = {"w30_searches_max_abs_decode_error": float(np.max(np.abs(actual - expected))),
             "finite_funnel": float(np.mean(np.isfinite(funnel))),
             "finite_coherence": float(np.mean(np.isfinite(coherence)))}
    return funnel, coherence, audit


LEVEL_FEATURES = [
    f"w{w}_{name}" for w in (7, 30, 90)
    for name in ("days_present", "days_search", "days_buy", "days_cart", "searches", "carts", "orders", "gmv", "gmv_cat")
] + ["rec_any", "rec_search", "rec_cart", "rec_buy", "rec_cat"]


def strong_level_matrix(fold: str, ids: np.ndarray) -> np.ndarray:
    X, _ = e80obs.load_feature_matrix(fold, ids)
    idx = [e80obs.FEATURES.index(name) for name in LEVEL_FEATURES]
    return X[:, idx].astype(np.float64)


def evaluate_observable_block(name: str, feature_names: list[str], per_fold_X: dict[str, np.ndarray],
                              ctx: dict[str, Any], seed: int) -> tuple[pd.DataFrame, dict[str, Any], np.ndarray]:
    fold_data: dict[str, dict[str, np.ndarray]] = {}
    individual = []
    for fold in FOLDS:
        m = ctx["masks"][fold]
        ids, r, z = ctx["uid"][m], ctx["residual"][m], ctx["z_current"][m]
        X = per_fold_X[fold].astype(np.float64)
        levels = strong_level_matrix(fold, ids)
        Bstrong = np.column_stack([np.ones(len(ids)), z, levels])
        Bspan = ctx["bases"][fold]
        Bdecisive = np.column_stack([Bspan, levels])
        D = quantile_design(X, 5)
        graw = e80.gain_from_design(r, D)
        gstrong = e80.gain_from_design(r, D, Bstrong)
        gspan = e80.gain_from_design(r, D, Bspan)
        gdec = e80.gain_from_design(r, D, Bdecisive)
        U = e80.project_out_matrix(D, Bdecisive)
        fold_data[fold] = {"U": U, "r": r, "uid": ids, "q_opt": gdec["correction"]}
        for j, fname in enumerate(feature_names):
            x = X[:, j]
            u0 = e80.project_out_matrix(x, np.ones((len(x), 1)))[:, 0]
            us = e80.project_out_matrix(x, Bstrong)[:, 0]
            up = e80.project_out_matrix(x, Bspan)[:, 0]
            ud = e80.project_out_matrix(x, Bdecisive)[:, 0]
            Dj = D[:, j * 5:(j + 1) * 5]
            Uj = U[:, j * 5:(j + 1) * 5]
            gj = e80.gain_from_design(r, Uj)
            individual.append({
                "mechanism": name, "cutoff": fold, "candidate": fname,
                "raw_rho": corr(u0, r), "rho_after_strong_level_conditioning": corr(us, r),
                "rho_after_production_span": corr(up, r),
                "rho_after_span_plus_levels": corr(ud, r),
                "optimistic_binned_headroom": gj["gain"],
            })
        fold_data[fold].update({
            "raw_gain": np.asarray([graw["gain"], graw["rho"]]),
            "strong_gain": np.asarray([gstrong["gain"], gstrong["rho"]]),
            "span_gain": np.asarray([gspan["gain"], gspan["rho"]]),
            "decisive_gain": np.asarray([gdec["gain"], gdec["rho"]]),
        })
    beta = beta_from(fold_data[SOURCE_FOLD]["U"], fold_data[SOURCE_FOLD]["r"])
    Uv, rv = fold_data[PURGED_FOLD]["U"], fold_data[PURGED_FOLD]["r"]
    qv = Uv @ beta
    met = correction_metrics(qv, rv)
    delta = (rv - qv) ** 2 - rv ** 2
    boot = poisson_bootstrap(delta, BOOTSTRAP_REPS, seed)
    weights = FOLD_WEIGHTS
    def wstat(key: str, pos: int) -> float:
        return float(np.average([fold_data[f][key][pos] for f in FOLDS], weights=weights))
    summary = {
        "mechanism": name, "feature_count": len(feature_names), "binned_design_columns": int(fold_data[SOURCE_FOLD]["U"].shape[1]),
        "raw_oracle_headroom": wstat("raw_gain", 0), "raw_oracle_rho": wstat("raw_gain", 1),
        "after_strong_level_oracle_headroom": wstat("strong_gain", 0),
        "after_strong_level_oracle_rho": wstat("strong_gain", 1),
        "after_production_span_oracle_headroom": wstat("span_gain", 0),
        "after_production_span_oracle_rho": wstat("span_gain", 1),
        "after_span_plus_levels_oracle_headroom": wstat("decisive_gain", 0),
        "after_span_plus_levels_oracle_rho": wstat("decisive_gain", 1),
        "purged_train_fold": SOURCE_FOLD, "purged_validation_fold": PURGED_FOLD,
        "source_target_end": str(np.datetime64(SOURCE_FOLD) + np.timedelta64(30, "D")),
        "source_label_available_at_validation": bool(np.datetime64(SOURCE_FOLD) + np.timedelta64(30, "D") <= np.datetime64(PURGED_FOLD)),
        "purged_rho": met["rho"], "purged_Delta_MSE": met["Delta_MSE"],
        "purged_Delta_RMSLE": met["Delta_RMSLE"], "purged_correction_RMS": met["correction_RMS"],
        "purged_P_gain": boot["P_gain"], "purged_CI95_Delta_MSE": boot["CI95_Delta_MSE"],
        "oracle_gate_pass": bool(wstat("decisive_gain", 0) >= 0.001),
        "observable_gate_pass": bool((met["rho"] >= 0.020 or met["Delta_MSE"] <= -0.001) and boot["P_gain"] >= 0.95),
        "full_model_authorized": bool(wstat("decisive_gain", 0) >= 0.001 and met["rho"] >= 0.015
                                      and (met["rho"] >= 0.020 or met["Delta_MSE"] <= -0.001)
                                      and boot["P_gain"] >= 0.95),
        "bootstrap": boot,
    }
    return pd.DataFrame(individual), summary, fold_data[PURGED_FOLD]["q_opt"]


def aggregate_individual(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    numeric = [c for c in frame.columns if c not in ("mechanism", "cutoff", "candidate")]
    for (mechanism, candidate), part in frame.groupby(["mechanism", "candidate"], sort=False):
        p = part.set_index("cutoff").loc[list(FOLDS)]
        row = {"mechanism": mechanism, "cutoff": "weighted_1_2_4_8", "candidate": candidate}
        for col in numeric:
            row[col] = float(np.average(p[col].to_numpy(np.float64), weights=FOLD_WEIGHTS))
        rows.append(row)
    return pd.concat([frame, pd.DataFrame(rows)], ignore_index=True)


def feature_audit_table() -> pd.DataFrame:
    rows = [
        ("Daily presence vs absent day", "present/buy/ponly masks; days_present/presence_only; sequence mask", "TABULAR, SEQ, ETX, DIST/CAP", "None material; absent and zero row kept distinct"),
        ("Total activity levels", "7/14/30/60/90/180/365 sums, days, recencies, trends", "TABULAR, DIST/CAP/UNC, SEQ/ETX", "Fine within-window distribution lost in tabular but available to sequence models"),
        ("Search and catalog funnel counts", "Only total carts/orders ratios; raw channel levels in SEQ/ETX", "TABULAR + SEQ/ETX", "Channel-specific normalized conversion and cross-channel disagreement absent from canonical 227"),
        ("GMV search/catalog composition", "gmv_cat share by window; channel raw sequence; EXP052 Shapley heads", "TABULAR, SEQ/ETX, CHANNEL-SHAPLEY", "Channel AOV/conditional conversion geometry not explicit; future contribution supervision already rejected"),
        ("Daily event order", "A1/A2, SEQ/ETX; explicit OPEN-FUNNEL and EVENT-ORDER", "SEQ/ETX/A1/A2", "Requested transition/lag phase descriptors already tested: SKIP_DUPLICATE"),
        ("Bursts and gaps", "recency, gap mean/std/CV; explicit threshold-3 episode summaries", "TABULAR/BTYD + EXP054", "Burst duration/density/open gap already tested: SKIP_DUPLICATE"),
        ("Multi-horizon future state", "MHZ 7/14/21/30/45/60 hazard/count heads; EXP080 coarse 7/14/30 timing oracle", "MHZ, DIST, EXP080", "Y1/Y3 and fine early-share oracle not previously isolated after count/value/coarse timing"),
        ("Calendar/platform phase", "DOW sequence channels, Holiday-YoY, platform detrend", "SEQ/ETX/HOLIDAY-YOY", "No new external exposure/promo/inventory channel in raw schema"),
        ("Relational identity", "user_id alignment only; behavioral prototypes", "EXP081", "No item/category/order/session identifiers: relation cannot be recovered"),
        ("Within-day conversion distribution", "Raw daily channel counts available; aggregate ratios and binary transition states only", "SEQ/ETX implicit", "Mean-of-ratios, dispersion and Jensen gap absent explicitly; tested as one other mechanism"),
    ]
    return pd.DataFrame(rows, columns=["raw_information", "existing_transformations_features", "model_families_using_it", "potentially_lost_information"])


def novelty_table() -> pd.DataFrame:
    rows = [
        ("Fine Y1/Y3/Y7/Y14 future shape", "MHZ + EXP080", "PARTIAL_NOVELTY", "1/3-day shares are new; must survive EXP080 count/value and coarse timing conditioning"),
        ("Channel-specific conversion geometry", "exp_002, exp_025, EXP052", "PARTIAL_NOVELTY", "Generic total ratios and channel target split exist; fixed search-vs-cat conditional geometry is not in canonical 227"),
        ("Time since last burst / duration / density", "EXP054", "SKIP_DUPLICATE", "Explicit threshold-3 burst/gap summaries failed preflight"),
        ("Inter-event gap compression/expansion", "EXP054 + gap CV + EXP075", "SKIP_DUPLICATE", "Episode gap ratios and generic temporal residual representations already tested"),
        ("Channel transition imbalance", "EXP064", "SKIP_DUPLICATE", "Fixed daily-state transition directions selected zero correction"),
        ("Purchase-after-search lag/open intent", "EXP061", "SKIP_DUPLICATE", "Open funnel after last order selected zero correction"),
        ("Recent-vs-long activity phase", "227 trends/BLOCK4/EXP080", "SKIP_DUPLICATE", "Generic slope/trend/acceleration and state bins are closed"),
        ("Within-day funnel coherence", "No exact prior; SEQ/ETX implicit", "NOVEL_EXPLICIT_INVARIANT", "Mean-of-ratios/Jensen/dispersion not in canonical tabular construction"),
    ]
    return pd.DataFrame(rows, columns=["candidate", "closest_prior", "novelty_verdict", "reason"])


def main() -> None:
    t0 = time.time()
    HERE.mkdir(parents=True, exist_ok=True)
    ctx = load_context()
    uid_all = np.load(e80.UID_PATH, mmap_mode="r")
    panel = np.load(e80.PANEL_PATH, mmap_mode="r")
    gmv = np.load(e80.GMV_PATH, mmap_mode="r")

    feature_audit_table().to_csv(HERE / "feature_information_audit.csv", index=False)
    novelty_table().to_csv(HERE / "novelty_audit.csv", index=False)

    mh_ind, mh_joint, mh_gate, mh_q = multi_horizon_analysis(ctx, uid_all, panel, gmv)
    mh_ind.to_csv(HERE / "multi_horizon_individual_oracle.csv", index=False)
    mh_joint.to_csv(HERE / "multi_horizon_joint_oracle.csv", index=False)
    write_json(HERE / "multi_horizon_gate.json", mh_gate)

    funnel_X: dict[str, np.ndarray] = {}
    coherence_X: dict[str, np.ndarray] = {}
    decode_audit = []
    for fold in FOLDS:
        ids = ctx["uid"][ctx["masks"][fold]]
        funnel_X[fold], coherence_X[fold], aud = history_information_features(ids, fold, uid_all, panel)
        decode_audit.append({"cutoff": fold, "rows": len(ids), **aud})

    fun_ind, fun_summary, fun_q = evaluate_observable_block(
        "funnel_channel_geometry", FUNNEL_NAMES, funnel_X, ctx, BOOTSTRAP_SEED)
    coh_ind, coh_summary, coh_q = evaluate_observable_block(
        "within_day_funnel_coherence", COHERENCE_NAMES, coherence_X, ctx, BOOTSTRAP_SEED + 1)
    individual = aggregate_individual(pd.concat([fun_ind, coh_ind], ignore_index=True))
    individual.to_csv(HERE / "observable_candidate_metrics.csv", index=False)
    summaries = [fun_summary, coh_summary]
    write_json(HERE / "observable_mechanism_metrics.json", summaries)

    direction_names = ["multi_horizon_shape_oracle_only", "funnel_geometry_oracle_map", "within_day_coherence_oracle_map"]
    directions = [mh_q, fun_q, coh_q]
    corr_rows = []
    for i, ni in enumerate(direction_names):
        for j, nj in enumerate(direction_names):
            corr_rows.append({"direction_i": ni, "direction_j": nj, "correlation": corr(directions[i], directions[j])})
    pd.DataFrame(corr_rows).to_csv(HERE / "candidate_direction_correlations.csv", index=False)

    ranking = pd.DataFrame([
        {
            "Mechanism": "Fine multi-horizon future shape",
            "Oracle_MSE": mh_gate["distinct_from_EXP080_coarse_timing_oracle"],
            "Observable_MSE": 0.0,
            "Purged_rho": np.nan,
            "Expected_gain": 0.0,
            "rho_squared": np.nan,
            "fraction_remaining_gap": mh_gate["distinct_from_EXP080_coarse_timing_oracle"] / REQUIRED_MSE,
            "Verdict": "REJECT_DUPLICATE_TIMING" if not mh_gate["models_authorized"] else "PROMISING",
        },
        {
            "Mechanism": "Funnel/channel geometry",
            "Oracle_MSE": fun_summary["after_span_plus_levels_oracle_headroom"],
            "Observable_MSE": max(0.0, -fun_summary["purged_Delta_MSE"]),
            "Purged_rho": fun_summary["purged_rho"],
            "Expected_gain": fun_summary["purged_Delta_MSE"],
            "rho_squared": fun_summary["purged_rho"] ** 2,
            "fraction_remaining_gap": max(0.0, -fun_summary["purged_Delta_MSE"]) / REQUIRED_MSE,
            "Verdict": "PROMISING" if fun_summary["full_model_authorized"] else "REJECT_GATE",
        },
        {
            "Mechanism": "Temporal phase descriptors",
            "Oracle_MSE": 0.0,
            "Observable_MSE": 0.0,
            "Purged_rho": np.nan,
            "Expected_gain": 0.0,
            "rho_squared": np.nan,
            "fraction_remaining_gap": 0.0,
            "Verdict": "SKIP_DUPLICATE_EXP054_061_064",
        },
        {
            "Mechanism": "Within-day funnel coherence",
            "Oracle_MSE": coh_summary["after_span_plus_levels_oracle_headroom"],
            "Observable_MSE": max(0.0, -coh_summary["purged_Delta_MSE"]),
            "Purged_rho": coh_summary["purged_rho"],
            "Expected_gain": coh_summary["purged_Delta_MSE"],
            "rho_squared": coh_summary["purged_rho"] ** 2,
            "fraction_remaining_gap": max(0.0, -coh_summary["purged_Delta_MSE"]) / REQUIRED_MSE,
            "Verdict": "PROMISING" if coh_summary["full_model_authorized"] else "REJECT_GATE",
        },
    ])
    ranking.to_csv(HERE / "mathematical_ranking.csv", index=False)

    audit = {
        "experiment": "EXP083_INFORMATION_LOSS_DISCOVERY",
        "phase": "oracle_and_target_free_binned_diagnostics_only",
        "runtime_seconds": time.time() - t0,
        "raw_sha256": sha256(e80.RAW),
        "canonical_oof_sha256": sha256(EXP075 / "clean_forward_predictions.parquet"),
        "rows": int(len(ctx["uid"])),
        "folds": FOLDS,
        "fold_weights": FOLD_WEIGHTS,
        "production_basis": "[1,z_current,z_match,40 clean OOF components,EXP075_postspan] per fold",
        "production_bank_columns": 40,
        "production_residual_RMSLE_weighted": float(np.average(
            [rms(ctx["residual"][ctx["masks"][f]]) for f in FOLDS], weights=FOLD_WEIGHTS)),
        "purged_transition": {"source": SOURCE_FOLD, "source_target_end": str(np.datetime64(SOURCE_FOLD) + np.timedelta64(30, "D")),
                              "validation": PURGED_FOLD, "label_available": True},
        "decode_audit": decode_audit,
        "target_derived_values_used_as_features": False,
        "target_derived_values_used_for_oracle_only": True,
        "models_trained": 0,
        "full_experiment_authorized": bool(mh_gate["models_authorized"] or fun_summary["full_model_authorized"]
                                           or coh_summary["full_model_authorized"]),
        "test_inference_run": False,
        "submission_created": False,
        "leaderboard_used_for_selection": False,
    }
    write_json(HERE / "audit.json", audit)
    print(json.dumps(jsonable({"multi_horizon": mh_gate, "funnel": fun_summary,
                               "coherence": coh_summary, "ranking": ranking.to_dict("records"),
                               "runtime_seconds": audit["runtime_seconds"]}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
