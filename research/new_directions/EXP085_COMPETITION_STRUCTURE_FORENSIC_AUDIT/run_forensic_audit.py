"""Competition-structure forensic audit.

No predictive model is trained.  The script reconstructs the audited EXP080
production-like residual/basis, scans exact raw identities, measures panel and
eligibility geometry, and tests fixed target-free structural transforms.  A
single purged transfer (2025-09-04 -> 2025-10-16) is used because the source
30-day labels are fully available before the recipient cutoff.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.neighbors import NearestNeighbors


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OZON = Path(r"C:\Users\Admin\Desktop\OZON-E-CUP")
RAW = OZON / "data" / "raw" / "train.parquet"
SAMPLE = OZON / "data" / "raw" / "sample_submit.csv"
PROCESSED = OZON / "data" / "processed"
ART = OZON / "artifacts"
E75 = ROOT / "research" / "new_directions" / "EXP075_OUT_OF_SPAN_RESIDUAL_SIGNALS"
E76 = ROOT / "research" / "new_directions" / "EXP076_STRONG_BASELINE_VALIDATION_CHANNEL"
E77 = ROOT / "research" / "new_directions" / "EXP077_FORWARD_STACK"
E80 = ROOT / "research" / "new_directions" / "EXP080_ORACLE_GAP_ATTRIBUTION"

FOLDS = ["2025-09-04", "2025-09-18", "2025-10-02", "2025-10-16"]
FOLD_WEIGHT = dict(zip(FOLDS, [1.0, 2.0, 4.0, 8.0]))
CURRENT = 1.646143314225527
TARGET = 1.6446514942
ANCHOR = 1.6461597403364463
DATA_START = np.datetime64("2025-01-01")
A_JOINT = (0.7462560852846633, 0.6466415684754089)
PANEL_CHANNELS = [
    "present", "cat", "buy", "ponly", "searches", "search_to_cart",
    "search_to_ord", "cat_to_cart", "cat_to_ord", "to_cart", "to_ord",
    "gmv_search", "gmv_cat", "gmv",
]


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


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, np.float64)
    return float(np.sqrt(np.mean(x * x)))


def corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, np.float64) - float(np.mean(x))
    y = np.asarray(y, np.float64) - float(np.mean(y))
    den = math.sqrt(float(x @ x) * float(y @ y))
    return 0.0 if den <= 1e-300 else float(x @ y / den)


def project_out(U: np.ndarray, B: np.ndarray) -> np.ndarray:
    U = np.asarray(U, np.float64)
    if U.ndim == 1:
        U = U[:, None]
    coef, *_ = np.linalg.lstsq(B, U, rcond=1e-10)
    out = U - B @ coef
    coef2, *_ = np.linalg.lstsq(B, out, rcond=1e-10)
    return out - B @ coef2


def gain(r: np.ndarray, D: np.ndarray, B: np.ndarray | None = None) -> dict[str, Any]:
    r = np.asarray(r, np.float64)
    D = np.asarray(D, np.float64)
    if D.ndim == 1:
        D = D[:, None]
    U = D if B is None else project_out(D, B)
    G = U.T @ U / len(r)
    b = U.T @ r / len(r)
    beta = np.linalg.pinv(G, rcond=1e-10) @ b
    q = U @ beta
    g = float(2 * np.mean(q * r) - np.mean(q * q))
    rank = int(np.linalg.matrix_rank(G, tol=max(float(np.max(np.diag(G), initial=0)), 1.0) * 1e-10))
    null_bias = rank * float(np.mean(r * r)) / len(r)
    return {
        "gain": max(g, 0.0),
        "debiased_gain": max(g - null_bias, 0.0),
        "null_df_bias": null_bias,
        "rho": corr(q, r),
        "rank": rank,
        "rms_correction": rms(q),
        "beta": beta,
        "U": U,
        "q": q,
    }


def rank01(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(np.asarray(x, np.float64), nan=-1e30, posinf=1e30, neginf=-1e30)
    order = np.argsort(x, kind="mergesort")
    out = np.empty(len(x), np.float64)
    out[order] = (np.arange(len(x), dtype=np.float64) + 0.5) / len(x)
    return out


def quantile_label(x: np.ndarray, bins: int) -> np.ndarray:
    return np.minimum((rank01(x) * bins).astype(np.int16), bins - 1)


def one_hot(labels: np.ndarray, levels: int) -> np.ndarray:
    labels = np.asarray(labels, np.int64)
    return np.eye(levels, dtype=np.float64)[labels]


def fixed_bucket(x: np.ndarray, edges: list[float]) -> np.ndarray:
    return np.digitize(np.asarray(x, np.float64), np.asarray(edges, np.float64), right=True).astype(np.int16)


def align_state(cutoff: str, ids: np.ndarray) -> pd.DataFrame:
    cols = [
        "user_id", "w30_gmv", "w90_gmv", "w30_days_present", "w90_days_present",
        "w90_days_buy", "rec_buy", "rec_any", "w30_cat_gmv_share", "weekend_share",
        "tenure_frac", "gap_cv", "w30_searches", "w30_orders",
    ]
    path = PROCESSED / f"feat_{cutoff.replace('-', '')}_LnormNone.parquet"
    frame = pd.read_parquet(path, columns=cols).sort_values("user_id")
    src = frame.user_id.to_numpy(np.int64)
    pos = np.searchsorted(src, ids)
    if pos.max(initial=0) >= len(src) or not np.array_equal(src[pos], ids):
        raise AssertionError(f"state alignment failed for {cutoff}")
    return frame.iloc[pos].reset_index(drop=True)


def reconstruct_production() -> dict[str, Any]:
    exp077 = load_module("exp077_forensic", E77 / "run_exp077.py")
    canon = pd.read_parquet(E75 / "clean_forward_predictions.parquet")
    a2 = pd.read_parquet(E75 / "a2_clean_forward_predictions.parquet")
    canon["cutoff"] = canon.cutoff.astype(str)
    a2["cutoff"] = a2.cutoff.astype(str)
    key = pd.MultiIndex.from_frame(canon[["cutoff", "user_id"]])
    a2 = a2.set_index(["cutoff", "user_id"]).reindex(key)
    if a2.u_raw_A2.isna().any():
        raise AssertionError("A2 alignment failed")
    y = canon.target_log.to_numpy(np.float64)
    uid = canon.user_id.to_numpy(np.int64)
    cut = canon.cutoff.to_numpy()
    masks = {f: cut == f for f in FOLDS}
    Z, bank_audit = exp077.load_reference_bank(canon)
    shares_src = json.loads((E76 / "out" / "s10_alpha_composition.json").read_text(encoding="utf-8"))
    shares0 = shares_src["SUBMIT_ORTH_ALPHA"]["shares"]
    shares = {"SEQ": shares0["SEQ"], "ETX": shares0["ETX"],
              "TABULAR": shares0["TAB"], "BTYD_OTHER": shares0["BTYD"]}
    z_match, _ = exp077.build_composition_proxy(Z, exp077.REFERENCE_BANK, y, masks, shares)
    raw_joint = (A_JOINT[0] * canon.u_raw_365.to_numpy(np.float64)
                 + A_JOINT[1] * a2.u_raw_A2.to_numpy(np.float64))
    z_current = np.empty(len(y), np.float64)
    bases: dict[str, np.ndarray] = {}
    for f in FOLDS:
        m = masks[f]
        B0 = np.column_stack([np.ones(m.sum()), z_match[m], Z[m]])
        up = project_out(raw_joint[m], B0)[:, 0]
        zc = np.maximum(z_match[m] + up, 0.0)
        z_current[m] = zc
        bases[f] = np.column_stack([np.ones(m.sum()), zc, z_match[m], Z[m], up])
    primary = np.load(E80 / "oracle_working_arrays.npz", allow_pickle=True)
    parity = {
        "uid_exact": bool(np.array_equal(uid, primary["user_id"])),
        "cutoff_exact": bool(np.array_equal(cut, primary["cutoff"])),
        "target_max_abs": float(np.max(np.abs(y - primary["target_log"]))),
        "z_current_max_abs": float(np.max(np.abs(z_current - primary["z_current"]))),
        "residual_max_abs": float(np.max(np.abs((y - z_current) - primary["residual_current"]))),
        "bank_columns": int(Z.shape[1]),
        "bank_rows": int(len(bank_audit)),
    }
    return {"canon": canon, "uid": uid, "cut": cut, "y": y, "z": z_current,
            "r": y - z_current, "masks": masks, "bases": bases, "parity": parity}


def scan_raw_identities() -> tuple[dict[str, Any], pd.DataFrame]:
    pf = pq.ParquetFile(RAW)
    counters: dict[str, int] = {
        "rows": 0,
        "gmv_sum_mismatch": 0,
        "to_ord_sum_mismatch": 0,
        "to_cart_sum_mismatch": 0,
        "search_indicator_mismatch": 0,
        "has_search_to_cart_mismatch": 0,
        "has_search_to_ord_mismatch": 0,
        "has_cat_to_cart_mismatch": 0,
        "has_cat_to_ord_mismatch": 0,
        "gmv_positive_without_order": 0,
        "order_without_positive_gmv": 0,
        "all_informational_fields_zero": 0,
        "negative_count_rows": 0,
        "negative_gmv_rows": 0,
    }
    max_gmv_error = 0.0
    raw_fields = [
        "event_date", "user_id", "search", "cat", "has_search_to_cart",
        "has_search_to_ord", "has_cat_to_cart", "has_cat_to_ord", "search_to_cart",
        "search_to_ord", "cat_to_cart", "cat_to_ord", "gmv_search", "gmv_cat",
        "to_cart", "to_ord", "gmv", "searches",
    ]
    count_cols = ["search", "cat", "has_search_to_cart", "has_search_to_ord",
                  "has_cat_to_cart", "has_cat_to_ord", "search_to_cart", "search_to_ord",
                  "cat_to_cart", "cat_to_ord", "to_cart", "to_ord", "searches"]
    info_cols = ["cat", "searches", "search_to_cart", "search_to_ord", "cat_to_cart",
                 "cat_to_ord", "gmv_search", "gmv_cat"]
    for batch in pf.iter_batches(batch_size=750_000, columns=raw_fields):
        b = batch.to_pydict()
        a = {k: np.asarray(v) for k, v in b.items()}
        n = len(a["user_id"])
        counters["rows"] += n
        err = np.abs(a["gmv"] - (a["gmv_search"] + a["gmv_cat"]))
        max_gmv_error = max(max_gmv_error, float(err.max(initial=0)))
        counters["gmv_sum_mismatch"] += int(np.sum(err > 1e-10))
        counters["to_ord_sum_mismatch"] += int(np.sum(a["to_ord"] != a["search_to_ord"] + a["cat_to_ord"]))
        counters["to_cart_sum_mismatch"] += int(np.sum(a["to_cart"] != a["search_to_cart"] + a["cat_to_cart"]))
        counters["search_indicator_mismatch"] += int(np.sum(a["search"] != (a["searches"] > 0)))
        for flag, value in [("has_search_to_cart", "search_to_cart"),
                            ("has_search_to_ord", "search_to_ord"),
                            ("has_cat_to_cart", "cat_to_cart"),
                            ("has_cat_to_ord", "cat_to_ord")]:
            counters[f"{flag}_mismatch"] += int(np.sum(a[flag] != (a[value] > 0)))
        counters["gmv_positive_without_order"] += int(np.sum((a["gmv"] > 0) & (a["to_ord"] <= 0)))
        counters["order_without_positive_gmv"] += int(np.sum((a["to_ord"] > 0) & (a["gmv"] <= 0)))
        zero = np.ones(n, bool)
        for c in info_cols:
            zero &= a[c] == 0
        counters["all_informational_fields_zero"] += int(zero.sum())
        neg_count = np.zeros(n, bool)
        for c in count_cols:
            neg_count |= a[c] < 0
        counters["negative_count_rows"] += int(neg_count.sum())
        counters["negative_gmv_rows"] += int(np.sum((a["gmv"] < 0) | (a["gmv_search"] < 0) | (a["gmv_cat"] < 0)))
    counters["max_abs_gmv_identity_error"] = max_gmv_error
    schema = pd.DataFrame({
        "raw_field": pf.schema_arrow.names,
        "dtype": [str(x.type) for x in pf.schema_arrow],
    })
    usage = {
        "event_date": ("YES", "cutoff/window alignment", "calendar position and missing-day geometry"),
        "user_id": ("ALIGNMENT_ONLY", "row key, cross-fit/bootstrap", "ID rank/density if assignment is structured"),
        "search": ("YES", "presence/search-day sequence", "none: exact searches>0 identity"),
        "cat": ("YES", "catalog-day activity", "catalog browsing state not reducible to downstream counts"),
        "has_search_to_cart": ("YES_REDUNDANT", "sequence/raw channel variants", "none: exact count>0 identity"),
        "has_search_to_ord": ("YES_REDUNDANT", "sequence/raw channel variants", "none: exact count>0 identity"),
        "has_cat_to_cart": ("YES_REDUNDANT", "sequence/raw channel variants", "none: exact count>0 identity"),
        "has_cat_to_ord": ("YES_REDUNDANT", "sequence/raw channel variants", "none: exact count>0 identity"),
        "search_to_cart": ("YES", "window sums and raw sequences", "conditional channel geometry already gated"),
        "search_to_ord": ("YES", "window sums and raw sequences", "conditional channel geometry already gated"),
        "cat_to_cart": ("YES", "window sums and raw sequences", "conditional channel geometry already gated"),
        "cat_to_ord": ("YES", "window sums and raw sequences", "conditional channel geometry already gated"),
        "gmv_search": ("YES", "channel GMV windows/sequences", "component reconciliation only if separate heads add signal"),
        "gmv_cat": ("YES", "channel GMV windows/sequences", "component reconciliation only if separate heads add signal"),
        "to_cart": ("YES_REDUNDANT", "total cart windows/sequences", "none: exact channel sum"),
        "to_ord": ("YES_REDUNDANT", "total order windows/sequences", "none: exact channel sum"),
        "gmv": ("YES_REDUNDANT_TARGET", "window GMV and target sum", "none: exact channel sum"),
        "searches": ("YES", "window sums and sequences", "within-day distribution tested in EXP083"),
    }
    schema["already_used"] = schema.raw_field.map(lambda x: usage[x][0])
    schema["how"] = schema.raw_field.map(lambda x: usage[x][1])
    schema["potential_missed_information"] = schema.raw_field.map(lambda x: usage[x][2])
    return counters, schema


def panel_reconstruction(uid_all: np.ndarray, panel: np.ndarray) -> tuple[dict[str, Any], pd.DataFrame]:
    present_ix = PANEL_CHANNELS.index("present")
    n, days = panel.shape[:2]
    first = np.empty(n, np.int16)
    last = np.empty(n, np.int16)
    rows = np.empty(n, np.int16)
    for lo in range(0, n, 10_000):
        hi = min(lo + 10_000, n)
        p = np.asarray(panel[lo:hi, :, present_ix]) > 0
        first[lo:hi] = np.argmax(p, axis=1)
        last[lo:hi] = days - 1 - np.argmax(p[:, ::-1], axis=1)
        rows[lo:hi] = p.sum(axis=1)
    sample = pd.read_csv(SAMPLE)
    test_blocks = []
    for lo, hi in [(days - 90, days - 60), (days - 60, days - 30), (days - 30, days)]:
        counts = np.empty(n, np.int16)
        for a in range(0, n, 20_000):
            b = min(a + 20_000, n)
            counts[a:b] = (np.asarray(panel[a:b, lo:hi, present_ix]) > 0).sum(axis=1)
        test_blocks.append(counts)
    eligible = np.logical_and.reduce([x > 0 for x in test_blocks])
    growth_cutoffs = [
        "2025-04-03", "2025-05-01", "2025-06-01", "2025-07-01", "2025-08-01",
        "2025-09-01", "2025-10-01", "2025-11-01", "2025-12-01", "2026-01-01",
        "2026-02-13",
    ]
    growth = []
    for c in growth_cutoffs:
        d = int((np.datetime64(c) - DATA_START).astype("timedelta64[D]").astype(int))
        cnt = []
        for lo, hi in [(d - 89, d - 59), (d - 59, d - 29), (d - 29, d + 1)]:
            arr = np.empty(n, np.int16)
            for a in range(0, n, 20_000):
                b = min(a + 20_000, n)
                arr[a:b] = (np.asarray(panel[a:b, lo:hi, present_ix]) > 0).sum(axis=1)
            cnt.append(arr)
        e = np.logical_and.reduce([x > 0 for x in cnt])
        growth.append({"cutoff": c, "eligible_users": int(e.sum()),
                       "share_of_final_250k": float(e.mean()),
                       "median_min_block_days": float(np.median(np.minimum.reduce(cnt)[e]))})
    start_dates = DATA_START + first.astype("timedelta64[D]")
    end_dates = DATA_START + last.astype("timedelta64[D]")
    out = {
        "rows": 30_631_006,
        "users": int(n),
        "calendar_days": int(days),
        "dense_user_days": int(n * days),
        "observed_row_fraction": float(30_631_006 / (n * days)),
        "sample_rows": int(len(sample)),
        "sample_sorted_by_user_id": bool(sample.user_id.is_monotonic_increasing),
        "sample_equals_sorted_raw_users": bool(np.array_equal(sample.user_id.to_numpy(np.int64), uid_all)),
        "uid_min": int(uid_all.min()),
        "uid_max": int(uid_all.max()),
        "uid_missing_integer_gaps": int(np.sum(np.diff(uid_all) > 1)),
        "test_eligibility_blocks": ["2025-11-16..2025-12-15", "2025-12-16..2026-01-14", "2026-01-15..2026-02-13"],
        "eligible_test_users": int(eligible.sum()),
        "all_users_pass_test_rule": bool(eligible.all()),
        "test_block_days_quantiles": [
            {"block": i + 1, "q": np.quantile(x, [0, .01, .1, .5, .9, .99, 1]).tolist()}
            for i, x in enumerate(test_blocks)
        ],
        "first_observed_date_quantiles": [str(x) for x in np.quantile(first, [0, .01, .1, .5, .9, .99, 1]).astype(int).astype("timedelta64[D]") + DATA_START],
        "last_observed_date_quantiles": [str(x) for x in np.quantile(last, [0, .01, .1, .5, .9, .99, 1]).astype(int).astype("timedelta64[D]") + DATA_START],
        "observed_days_per_user_quantiles": np.quantile(rows, [0, .01, .1, .5, .9, .99, 1]).tolist(),
        "first_date_is_data_start_share": float(np.mean(first == 0)),
        "last_date_is_cutoff_share": float(np.mean(last == days - 1)),
        "first_month_counts": pd.Series(start_dates.astype("datetime64[M]").astype(str)).value_counts().sort_index().to_dict(),
        "last_month_counts": pd.Series(end_dates.astype("datetime64[M]").astype(str)).value_counts().sort_index().to_dict(),
    }
    return out, pd.DataFrame(growth)


def eligibility_arrays(ids: np.ndarray, cutoff: str, uid_all: np.ndarray,
                       panel: np.ndarray) -> dict[str, np.ndarray]:
    idx = np.searchsorted(uid_all, ids)
    if not np.array_equal(uid_all[idx], ids):
        raise AssertionError("panel uid alignment failed")
    d = int((np.datetime64(cutoff) - DATA_START).astype("timedelta64[D]").astype(int))
    p = np.asarray(panel[idx, d - 89:d + 1, PANEL_CHANNELS.index("present")]) > 0
    blocks = p.reshape(len(ids), 3, 30)
    days = blocks.sum(axis=2)
    # Recency from each block end; eligibility guarantees at least one row.
    last_pos = 29 - np.argmax(blocks[:, :, ::-1], axis=2)
    end_gap = 29 - last_pos
    first_pos = np.argmax(blocks, axis=2)
    return {
        "block_days": days,
        "min_days": days.min(axis=1),
        "max_days": days.max(axis=1),
        "imbalance": (days.max(axis=1) - days.min(axis=1)) / np.maximum(days.sum(axis=1), 1),
        "max_block_end_gap": end_gap.max(axis=1),
        "max_block_start_gap": first_pos.max(axis=1),
        "new_old_ratio": (days[:, 2] + 0.5) / (days[:, 0] + 0.5),
    }


def future_continuation(ids: np.ndarray, cutoff: str, uid_all: np.ndarray,
                        panel: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(uid_all, ids)
    d = int((np.datetime64(cutoff) - DATA_START).astype("timedelta64[D]").astype(int))
    p = np.asarray(panel[idx, d + 31:d + 121, PANEL_CHANNELS.index("present")]) > 0
    return (p.reshape(len(ids), 3, 30).sum(axis=2) > 0).sum(axis=1).astype(np.int8)


def structural_designs(ids: np.ndarray, z: np.ndarray, state: pd.DataFrame,
                       elig: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    scalar: dict[str, np.ndarray] = {}
    design: dict[str, np.ndarray] = {}

    uid_rank = rank01(ids)
    design["uid_rank_20"] = one_hot(np.minimum((uid_rank * 20).astype(int), 19), 20)
    scalar["uid_rank_20"] = uid_rank
    design["uid_mod_10"] = one_hot(ids % 10, 10)
    scalar["uid_mod_10"] = (ids % 10).astype(float)
    gap = np.empty(len(ids), np.float64)
    gap[0] = ids[1] - ids[0]
    gap[-1] = ids[-1] - ids[-2]
    gap[1:-1] = (ids[2:] - ids[:-2]) / 2
    design["uid_local_gap_decile"] = one_hot(quantile_label(gap, 10), 10)
    scalar["uid_local_gap_decile"] = gap

    min_lab = fixed_bucket(elig["min_days"], [1, 2, 4, 7])
    gap_lab = fixed_bucket(elig["max_block_end_gap"], [0, 2, 6, 13])
    design["eligibility_min_days"] = one_hot(min_lab, 5)
    scalar["eligibility_min_days"] = elig["min_days"]
    design["eligibility_end_gap"] = one_hot(gap_lab, 5)
    scalar["eligibility_end_gap"] = elig["max_block_end_gap"]
    joint = min_lab * 5 + gap_lab
    design["eligibility_joint_geometry"] = one_hot(joint, 25)
    scalar["eligibility_joint_geometry"] = elig["min_days"] - elig["max_block_end_gap"] / 30
    design["eligibility_imbalance_decile"] = one_hot(quantile_label(elig["imbalance"], 10), 10)
    scalar["eligibility_imbalance_decile"] = elig["imbalance"]
    design["eligibility_new_old_ratio_decile"] = one_hot(quantile_label(elig["new_old_ratio"], 10), 10)
    scalar["eligibility_new_old_ratio_decile"] = elig["new_old_ratio"]

    rank_features = {
        "cohort_rank_w90_gmv": state.w90_gmv.to_numpy(float),
        "cohort_rank_activity": state.w90_days_present.to_numpy(float),
        "cohort_rank_recency": -state.rec_buy.to_numpy(float),
        "cohort_rank_channel_mix": state.w30_cat_gmv_share.to_numpy(float),
        "cohort_rank_baseline": z,
        "availability_tenure_rank": state.tenure_frac.to_numpy(float),
    }
    for name, values in rank_features.items():
        lab = quantile_label(values, 10)
        design[name] = one_hot(lab, 10)
        scalar[name] = rank01(values)
    rg = quantile_label(state.w90_gmv.to_numpy(float), 5)
    rr = quantile_label(-state.rec_buy.to_numpy(float), 5)
    design["cohort_joint_gmv_recency_rank"] = one_hot(rg * 5 + rr, 25)
    scalar["cohort_joint_gmv_recency_rank"] = rg + rr / 5

    # Whole-cohort target-free local density in a rank-normalized state space.
    X = np.column_stack([
        rank01(state.w30_gmv.to_numpy(float)),
        rank01(state.w90_days_present.to_numpy(float)),
        rank01(-state.rec_buy.to_numpy(float)),
        rank01(state.w30_cat_gmv_share.to_numpy(float)),
        rank01(z),
    ])
    nn = NearestNeighbors(n_neighbors=11, algorithm="kd_tree", leaf_size=64, n_jobs=-1)
    nn.fit(X)
    dist, _ = nn.kneighbors(X, return_distance=True)
    density = -np.log(np.maximum(dist[:, -1], 1e-12))
    design["cohort_knn_density_rank"] = one_hot(quantile_label(density, 10), 10)
    scalar["cohort_knn_density_rank"] = density
    return design, scalar


def calendar_metrics(prod: dict[str, Any], uid_all: np.ndarray,
                     gmv: np.ndarray) -> dict[str, Any]:
    fold_rows = []
    proxy_store: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    dates = pd.date_range("2025-01-01", periods=gmv.shape[1], freq="D")
    dow = dates.dayofweek.to_numpy()
    for f in FOLDS:
        m = prod["masks"][f]
        ids = prod["uid"][m]
        idx = np.searchsorted(uid_all, ids)
        d = int((np.datetime64(f) - DATA_START).astype("timedelta64[D]").astype(int))
        y_shift = np.asarray(gmv[idx, d + 2:d + 32], np.float64).sum(axis=1)
        target_shift = np.log1p(y_shift)
        r_shift = target_shift - prod["z"][m]
        delta_actual = target_shift - prod["y"][m]
        B = prod["bases"][f]
        oracle = gain(r_shift, delta_actual, B)
        h0 = max(0, d - 364)
        hist = np.asarray(gmv[idx, h0:d + 1], np.float64)
        hdow = dow[h0:d + 1]
        fri = hist[:, hdow == 4].mean(axis=1)
        sun = hist[:, hdow == 6].mean(axis=1)
        pred_y = np.expm1(prod["z"][m])
        proxy = np.log1p(np.maximum(pred_y + sun - fri, 0)) - prod["z"][m]
        observable = gain(r_shift, proxy, B)
        proxy_store[f] = (proxy[:, None], r_shift, B)
        fold_rows.append({
            "cutoff": f,
            "original_target_start": str(np.datetime64(f) + np.timedelta64(1, "D")),
            "shifted_target_start": str(np.datetime64(f) + np.timedelta64(2, "D")),
            "actual_delta_mean": float(delta_actual.mean()),
            "actual_delta_rms": rms(delta_actual),
            "actual_delta_corr_original_residual": corr(delta_actual, prod["r"][m]),
            "calendar_oracle_after_span": oracle["gain"],
            "calendar_oracle_debiased": oracle["debiased_gain"],
            "observable_proxy_rho": observable["rho"],
            "observable_proxy_headroom": observable["gain"],
            "observable_proxy_debiased": observable["debiased_gain"],
            "mean_sunday_minus_friday_gmv": float(np.mean(sun - fri)),
            "aggregate_shifted_vs_original_gmv_ratio": float(y_shift.sum() / prod["canon"].loc[m, "target_y30"].sum()),
        })
    w = np.array([FOLD_WEIGHT[f] for f in FOLDS], float)
    agg = {"cutoff": "weighted_1_2_4_8"}
    for k in fold_rows[0]:
        if k not in ("cutoff", "original_target_start", "shifted_target_start"):
            agg[k] = float(np.average([x[k] for x in fold_rows], weights=w))
    donor_D, donor_r, donor_B = proxy_store["2025-09-04"]
    donor_fit = gain(donor_r, donor_D, donor_B)
    val_D, val_r, val_B = proxy_store["2025-10-16"]
    q = project_out(val_D, val_B) @ donor_fit["beta"]
    purged = {
        "source": "2025-09-04",
        "source_target_end": "2025-10-04",
        "recipient": "2025-10-16",
        "rho": corr(q, val_r),
        "correction_rms": rms(q),
        "Delta_MSE": float(np.mean((val_r - q) ** 2 - val_r ** 2)),
    }
    target_dates = pd.date_range("2026-02-14", "2026-03-15", freq="D")
    counts = target_dates.day_name().value_counts().reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    ).to_dict()
    return {
        "test_window": ["2026-02-14", "2026-03-15"],
        "days": int(len(target_dates)),
        "weekday_counts": counts,
        "weekend_days": int(np.sum(target_dates.dayofweek >= 5)),
        "special_dates_requested": {
            "2026-02-23": str(pd.Timestamp("2026-02-23").day_name()),
            "2026-03-08": str(pd.Timestamp("2026-03-08").day_name()),
        },
        "training_fold_target_composition": "all four canonical cutoffs are Thursday; their targets start Friday and contain five Fridays plus five Saturdays",
        "test_composition_difference": "test cutoff is Friday; target starts Saturday and replaces the fifth Friday by a fifth Sunday",
        "folds": fold_rows + [agg],
        "purged_proxy_transfer": purged,
    }


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    required = CURRENT ** 2 - TARGET ** 2
    rho = math.sqrt(required / CURRENT ** 2)
    scenario = {
        "current_RMSLE": CURRENT,
        "target_RMSLE": TARGET,
        "Delta_RMSLE": TARGET - CURRENT,
        "current_MSE": CURRENT ** 2,
        "target_MSE": TARGET ** 2,
        "Delta_MSE": TARGET ** 2 - CURRENT ** 2,
        "required_MSE_gain": required,
        "required_independent_rho": rho,
        "required_explained_residual_variance": rho ** 2,
        "optimal_total_correction_rms": math.sqrt(required),
        "required_total_covariance_at_optimum": required,
        "scenarios": {
            "one_direction": {"rho_each": rho, "norm_each": CURRENT * rho, "covariance_each": required},
            "two_independent_equal": {"rho_each": rho / math.sqrt(2), "norm_each": CURRENT * rho / math.sqrt(2), "covariance_each": required / 2},
            "three_independent_equal": {"rho_each": rho / math.sqrt(3), "norm_each": CURRENT * rho / math.sqrt(3), "covariance_each": required / 3},
        },
    }
    write_json(HERE / "gap_math.json", scenario)

    raw_identity, schema = scan_raw_identities()
    write_json(HERE / "raw_identity_audit.json", raw_identity)
    schema.to_csv(HERE / "raw_schema_audit.csv", index=False)

    uid_all = np.load(PROCESSED / "seq_uid_v1.npy", mmap_mode="r")
    panel = np.load(PROCESSED / "seq_panel_v1.npy", mmap_mode="r")
    gmv = np.load(PROCESSED / "seq_gmv_v1.npy", mmap_mode="r")
    panel_audit, growth = panel_reconstruction(uid_all, panel)
    write_json(HERE / "dataset_reconstruction.json", panel_audit)
    growth.to_csv(HERE / "panel_growth.csv", index=False)

    prod = reconstruct_production()
    write_json(HERE / "production_reconstruction.json", prod["parity"])

    designs: dict[str, dict[str, np.ndarray]] = {}
    scalars: dict[str, dict[str, np.ndarray]] = {}
    selection_rows = []
    future_rows = []
    for f in FOLDS:
        m = prod["masks"][f]
        ids = prod["uid"][m]
        elig = eligibility_arrays(ids, f, uid_all, panel)
        state = align_state(f, ids)
        d, s = structural_designs(ids, prod["z"][m], state, elig)
        designs[f] = d
        scalars[f] = s
        y30 = prod["canon"].loc[m, "target_y30"].to_numpy(np.float64)
        bucket = fixed_bucket(elig["min_days"], [1, 2, 4, 7])
        for b in range(5):
            q = bucket == b
            selection_rows.append({
                "cutoff": f, "eligibility_min_days_bucket": ["1", "2", "3-4", "5-7", "8+"][b],
                "n": int(q.sum()), "share": float(q.mean()),
                "P_target_positive": float(np.mean(y30[q] > 0)),
                "E_target_GMV": float(np.mean(y30[q])),
                "E_target_log": float(np.mean(prod["y"][m][q])),
                "E_production_residual": float(np.mean(prod["r"][m][q])),
                "E_max_block_end_gap": float(np.mean(elig["max_block_end_gap"][q])),
            })
        k = future_continuation(ids, f, uid_all, panel)
        Dk = one_hot(k, 4)
        gk = gain(prod["r"][m], Dk, prod["bases"][f])
        future_rows.append({
            "cutoff": f, "k0": float(np.mean(k == 0)), "k1": float(np.mean(k == 1)),
            "k2": float(np.mean(k == 2)), "k3": float(np.mean(k == 3)),
            "future_continuation_oracle_headroom": gk["gain"],
            "future_continuation_oracle_debiased": gk["debiased_gain"],
            "future_continuation_rho": gk["rho"],
        })
    pd.DataFrame(selection_rows).to_csv(HERE / "selection_boundary_outcomes.csv", index=False)
    pd.DataFrame(future_rows).to_csv(HERE / "future_survivorship_oracle.csv", index=False)

    candidate_rows = []
    candidate_names = list(designs[FOLDS[0]])
    for name in candidate_names:
        per = []
        for f in FOLDS:
            m = prod["masks"][f]
            fit = gain(prod["r"][m], designs[f][name], prod["bases"][f])
            row = {"candidate": name, "cutoff": f, "n": int(m.sum()),
                   "oracle_headroom": fit["gain"], "oracle_headroom_debiased": fit["debiased_gain"],
                   "null_df_bias": fit["null_df_bias"], "oracle_rho": fit["rho"],
                   "rank": fit["rank"], "oracle_correction_rms": fit["rms_correction"],
                   "raw_scalar_corr_residual": corr(scalars[f][name], prod["r"][m])}
            candidate_rows.append(row)
            per.append(row)
        weights = np.array([FOLD_WEIGHT[f] for f in FOLDS], float)
        agg = {"candidate": name, "cutoff": "weighted_1_2_4_8", "n": int(sum(x["n"] for x in per))}
        for c in ["oracle_headroom", "oracle_headroom_debiased", "null_df_bias", "oracle_rho",
                  "rank", "oracle_correction_rms", "raw_scalar_corr_residual"]:
            agg[c] = float(np.average([x[c] for x in per], weights=weights))
        donor = gain(prod["r"][prod["masks"]["2025-09-04"]], designs["2025-09-04"][name],
                     prod["bases"]["2025-09-04"])
        mv = prod["masks"]["2025-10-16"]
        Uv = project_out(designs["2025-10-16"][name], prod["bases"]["2025-10-16"])
        qv = Uv @ donor["beta"]
        rv = prod["r"][mv]
        agg["purged_source"] = "2025-09-04"
        agg["purged_recipient"] = "2025-10-16"
        agg["purged_rho"] = corr(qv, rv)
        agg["purged_correction_rms"] = rms(qv)
        agg["purged_Delta_MSE"] = float(np.mean((rv - qv) ** 2 - rv ** 2))
        agg["oracle_gate_ge_0_001"] = bool(agg["oracle_headroom_debiased"] >= 0.001)
        agg["observable_gate"] = bool(abs(agg["purged_rho"]) >= 0.020 or agg["purged_Delta_MSE"] <= -0.001)
        candidate_rows.append(agg)
    candidates = pd.DataFrame(candidate_rows)
    candidates.to_csv(HERE / "structural_candidate_metrics.csv", index=False)

    cal = calendar_metrics(prod, uid_all, gmv)
    write_json(HERE / "calendar_audit.json", cal)

    observed_exp075_gain = ANCHOR ** 2 - CURRENT ** 2
    geometry = {
        "scores": {
            "best_pre_geometry_single": 1.64921756224069,
            "geometry_v2": 1.6467120249048954,
            "geometry_next_best": 1.646607908393441,
            "public_EB": 1.6463246740442117,
            "ORTH_ALPHA_anchor": ANCHOR,
            "EXP075_joint_current": CURRENT,
            "forensic_target": TARGET,
        },
        "EXP075_observed_Delta_RMSLE": CURRENT - ANCHOR,
        "EXP075_observed_MSE_gain": observed_exp075_gain,
        "EXP075_expected_nested_MSE_gain": 0.00439191,
        "EXP075_observed_fraction_of_expected": observed_exp075_gain / 0.00439191,
        "geometry_evidence": {
            "broad_directions_net_signal_MSE": 0.001670,
            "intermediate_directions_net_signal_MSE": 0.004854,
            "narrow_seed_scale_directions_net_signal_MSE": -0.002167,
            "public_to_full_transfer_ratio_v2_estimate": 0.47,
            "public_direction_noise_sd": 0.00660,
            "loo_unexpected_positive_alignment": [
                "HOLIDAY-YOY (-2.28 sd)", "baseline_hgbr (-1.91 sd)",
                "S04-A (-1.76 sd)", "candidate_e11mix (-1.70 sd)", "submission_v2 (-1.52 sd)",
            ],
        },
        "no_new_probes_created": True,
    }
    write_json(HERE / "submission_geometry_evidence.json", geometry)

    summary = {
        "gap": scenario,
        "production_parity": prod["parity"],
        "raw_identity": raw_identity,
        "panel": panel_audit,
        "max_structural_oracle_debiased": candidates[candidates.cutoff == "weighted_1_2_4_8"]
            .sort_values("oracle_headroom_debiased", ascending=False).head(5)
            [["candidate", "oracle_headroom_debiased", "purged_rho", "purged_Delta_MSE"]].to_dict("records"),
        "calendar": cal,
        "submission_geometry": geometry,
        "models_trained": 0,
        "submissions_created": 0,
    }
    write_json(HERE / "audit_summary.json", summary)
    print(json.dumps(jsonable({
        "production_parity": prod["parity"],
        "top_structural": summary["max_structural_oracle_debiased"],
        "calendar_purged": cal["purged_proxy_transfer"],
        "raw_identity": raw_identity,
    }), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
