"""EXP080 phase 1: target-oracle gap attribution, with no model training.

The script rebuilds the production-matched historical baseline from primary OOF
artifacts, adds the exact EXP075 joint direction at its deployed unit amplitude,
and measures target-derived oracle headroom only after projecting candidate
designs out of the full production span.  It never loads forbidden activity
labels as features and never trains a predictive model.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OZON = Path(r"C:\Users\Admin\Desktop\OZON-E-CUP")
ART = OZON / "artifacts"
PROCESSED = OZON / "data" / "processed"
RAW = OZON / "data" / "raw" / "train.parquet"
E75 = ROOT / "research" / "new_directions" / "EXP075_OUT_OF_SPAN_RESIDUAL_SIGNALS"
E76 = ROOT / "research" / "new_directions" / "EXP076_STRONG_BASELINE_VALIDATION_CHANNEL"
E77 = ROOT / "research" / "new_directions" / "EXP077_FORWARD_STACK"

FOLDS = ["2025-09-04", "2025-09-18", "2025-10-02", "2025-10-16"]
FOLD_WEIGHT = dict(zip(FOLDS, [1.0, 2.0, 4.0, 8.0]))
CURRENT_RMSLE = 1.646143314225527
TARGET_RMSLE = 1.6446514942
A_JOINT = (0.7462560852846633, 0.6466415684754089)
DATA_START = np.datetime64("2025-01-01")

PANEL_PATH = PROCESSED / "seq_panel_v1.npy"
GMV_PATH = PROCESSED / "seq_gmv_v1.npy"
UID_PATH = PROCESSED / "seq_uid_v1.npy"
PANEL_CHANNELS = [
    "present", "cat", "buy", "ponly", "searches", "search_to_cart",
    "search_to_ord", "cat_to_cart", "cat_to_ord", "to_cart", "to_ord",
    "gmv_search", "gmv_cat", "gmv",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


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


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, np.float64)
    return float(np.sqrt(np.mean(x * x)))


def corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, np.float64) - float(np.mean(x))
    y = np.asarray(y, np.float64) - float(np.mean(y))
    den = math.sqrt(float(x @ x) * float(y @ y))
    return 0.0 if den <= 1e-300 else float(x @ y / den)


def load_exp077():
    spec = importlib.util.spec_from_file_location("exp077_primary", E77 / "run_exp077.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load EXP077 reconstruction module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def project_out_matrix(U: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Two-pass least-squares projection, matching the established SBVC."""
    U = np.asarray(U, np.float64)
    if U.ndim == 1:
        U = U[:, None]
    coef, *_ = np.linalg.lstsq(B, U, rcond=1e-10)
    out = U - B @ coef
    coef2, *_ = np.linalg.lstsq(B, out, rcond=1e-10)
    out -= B @ coef2
    return out


def design_from_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    values = np.unique(labels)
    return np.column_stack([(labels == value).astype(np.float64) for value in values])


def percentile_bins(values: np.ndarray, n_bins: int, mask: np.ndarray | None = None,
                    zero_label: int = -1) -> np.ndarray:
    values = np.asarray(values, np.float64)
    use = np.ones(len(values), bool) if mask is None else np.asarray(mask, bool)
    out = np.full(len(values), zero_label, dtype=np.int16)
    idx = np.flatnonzero(use)
    if len(idx) == 0:
        return out
    order = idx[np.argsort(values[idx], kind="mergesort")]
    ranks = np.arange(len(order), dtype=np.int64)
    out[order] = np.minimum(n_bins - 1, ranks * n_bins // len(order)).astype(np.int16)
    return out


def count_bucket(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    out = np.zeros(len(x), np.int8)
    out[x == 1] = 1
    out[x == 2] = 2
    out[x == 3] = 3
    out[(x >= 4) & (x <= 5)] = 4
    out[x >= 6] = 5
    return out


def event_bucket(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    out = np.zeros(len(x), np.int8)
    out[(x >= 1) & (x <= 3)] = 1
    out[(x >= 4) & (x <= 7)] = 2
    out[(x >= 8) & (x <= 14)] = 3
    out[x >= 15] = 4
    return out


def gain_from_design(r: np.ndarray, D: np.ndarray, B: np.ndarray | None = None) -> dict[str, Any]:
    r = np.asarray(r, np.float64)
    D = np.asarray(D, np.float64)
    if D.ndim == 1:
        D = D[:, None]
    U = D if B is None else project_out_matrix(D, B)
    G = U.T @ U / len(r)
    b = U.T @ r / len(r)
    beta = np.linalg.pinv(G, rcond=1e-10) @ b
    q = U @ beta
    gain = float(2.0 * np.mean(q * r) - np.mean(q * q))
    return {
        "gain": max(gain, 0.0),
        "rho": corr(q, r),
        "rank": int(np.linalg.matrix_rank(G, tol=max(float(np.max(np.diag(G), initial=0.0)), 1.0) * 1e-10)),
        "rms_correction": rms(q),
        "beta": beta,
        "correction": q,
        "U": U,
    }


def incremental_gain(r: np.ndarray, block: np.ndarray, B: np.ndarray,
                     previous: np.ndarray | None = None) -> dict[str, Any]:
    base = B if previous is None or previous.shape[1] == 0 else np.column_stack([B, previous])
    return gain_from_design(r, block, base)


def signal_r2_from_basis(signal: np.ndarray, B: np.ndarray) -> float:
    """Fraction of centered oracle-label energy linearly represented by B."""
    signal = np.asarray(signal, np.float64)
    if signal.ndim == 1:
        signal = signal[:, None]
    centered = signal - signal.mean(axis=0, keepdims=True)
    total = float(np.sum(centered * centered))
    if total <= 1e-300:
        return 0.0
    residual = project_out_matrix(signal, B)
    return float(1.0 - np.sum(residual * residual) / total)


def weighted_mean(rows: list[dict[str, Any]], key: str, weights_key: str = "cutoff") -> float:
    vals, weights = [], []
    for row in rows:
        if key in row and np.isfinite(row[key]):
            vals.append(float(row[key]))
            weights.append(FOLD_WEIGHT[str(row[weights_key])])
    return float(np.average(vals, weights=weights))


def future_arrays(user_ids: np.ndarray, cutoff: str, uid_all: np.ndarray,
                  panel: np.ndarray, gmv: np.ndarray) -> dict[str, np.ndarray]:
    idx = np.searchsorted(uid_all, np.asarray(user_ids, np.int64))
    if idx.max(initial=0) >= len(uid_all) or not np.array_equal(uid_all[idx], user_ids):
        raise AssertionError("unknown or misaligned user id")
    d = int((np.datetime64(cutoff) - DATA_START).astype("timedelta64[D]").astype(int))
    future_gmv = np.asarray(gmv[idx, d + 1:d + 31], dtype=np.float64)
    y7 = future_gmv[:, :7].sum(axis=1)
    y14 = future_gmv[:, :14].sum(axis=1)
    y30 = future_gmv.sum(axis=1)
    purchase_days = (future_gmv > 0).sum(axis=1).astype(np.int16)
    present = np.asarray(panel[idx, d + 1:d + 31, PANEL_CHANNELS.index("present")], dtype=np.float32)
    event_days = (present > 0).sum(axis=1).astype(np.int16)
    to_ord_log = np.asarray(panel[idx, d + 1:d + 31, PANEL_CHANNELS.index("to_ord")], dtype=np.float32)
    order_items = np.rint(np.expm1(to_ord_log)).sum(axis=1).astype(np.int32)
    return {
        "y7": y7,
        "y14": y14,
        "y30": y30,
        "purchase_days": purchase_days,
        "event_days": event_days,
        "order_items": order_items,
    }


def align_npz(path: Path, uid_field: str, value_field: str, ids: np.ndarray) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    uid = np.asarray(data[uid_field], np.int64)
    val = np.asarray(data[value_field], np.float64)
    order = np.argsort(uid)
    pos = np.searchsorted(uid[order], ids)
    if pos.max(initial=0) >= len(uid) or not np.array_equal(uid[order][pos], ids):
        raise AssertionError(f"alignment failed: {path.name}:{value_field}")
    return val[order][pos]


def existing_activity_signals(cutoff: str, ids: np.ndarray) -> dict[str, np.ndarray]:
    pact = align_npz(ART / f"PACT_dist_{cutoff}.npz", "user_id", "p_act", ids)
    block_q = align_npz(ART / f"BLOCK4_SAF_fold_{cutoff.replace('-', '')}.npz", "uid", "q", ids)
    pmf_path = ART / "BTYD_STABLE_EXP051" / f"pmf_{cutoff.replace('-', '')}.npz"
    pmf = np.load(pmf_path, allow_pickle=False)
    puid = np.asarray(pmf["user_id"], np.int64)
    q = np.asarray(pmf["q"], np.float64)
    order = np.argsort(puid)
    pos = np.searchsorted(puid[order], ids)
    if pos.max(initial=0) >= len(puid) or not np.array_equal(puid[order][pos], ids):
        raise AssertionError(f"BTYD PMF alignment failed: {cutoff}")
    q = q[order][pos]
    k = np.arange(q.shape[1], dtype=np.float64)
    return {
        "dist_p_act": pact,
        "block4_q_event": block_q,
        "btyd_p_act": 1.0 - q[:, 0],
        "btyd_expected_count": q @ k,
    }


def load_state_features(cutoff: str, ids: np.ndarray) -> pd.DataFrame:
    columns = [
        "user_id", "rec_buy", "w90_days_buy", "w30_days_present", "tenure_frac",
        "w30_gmv", "w30_cat_gmv_share", "w30_days_search", "w30_days_cart",
        "w30_searches", "w30_carts", "w30_orders",
    ]
    path = PROCESSED / f"feat_{cutoff.replace('-', '')}_LnormNone.parquet"
    frame = pd.read_parquet(path, columns=columns).sort_values("user_id")
    source_ids = frame.user_id.to_numpy(np.int64)
    pos = np.searchsorted(source_ids, ids)
    if pos.max(initial=0) >= len(frame) or not np.array_equal(source_ids[pos], ids):
        raise AssertionError(f"state feature alignment failed: {cutoff}")
    return frame.iloc[pos].reset_index(drop=True)


def segment_labels(state: pd.DataFrame, z_current: np.ndarray,
                   p_act: np.ndarray) -> dict[str, tuple[np.ndarray, list[str]]]:
    rec = state.rec_buy.to_numpy(np.float64)
    rec_lab = np.select([rec <= 7, rec <= 30, rec <= 90], [0, 1, 2], default=3).astype(np.int8)
    freq = state.w90_days_buy.to_numpy(np.float64)
    freq_lab = np.select([freq == 0, freq == 1, freq <= 3, freq <= 7], [0, 1, 2, 3], default=4).astype(np.int8)
    intensity = state.w30_days_present.to_numpy(np.float64)
    int_lab = np.select([intensity <= 3, intensity <= 7, intensity <= 15], [0, 1, 2], default=3).astype(np.int8)
    tenure_lab = percentile_bins(state.tenure_frac.to_numpy(np.float64), 4, zero_label=0)
    pred_lab = percentile_bins(z_current, 10, zero_label=0)
    pact_lab = percentile_bins(p_act, 5, zero_label=0)
    gmv30 = state.w30_gmv.to_numpy(np.float64)
    cat_share = state.w30_cat_gmv_share.to_numpy(np.float64)
    channel = np.zeros(len(state), np.int8)
    buy = gmv30 > 0
    channel[buy & (cat_share < 0.25)] = 1
    channel[buy & (cat_share >= 0.25) & (cat_share < 0.75)] = 2
    channel[buy & (cat_share >= 0.75)] = 3
    return {
        "recency": (rec_lab, ["0-7", "8-30", "31-90", ">90/never"]),
        "purchase_frequency": (freq_lab, ["0", "1", "2-3", "4-7", "8+"]),
        "activity_intensity": (int_lab, ["1-3", "4-7", "8-15", "16+"]),
        "tenure_quartile": (tenure_lab, ["Q1", "Q2", "Q3", "Q4"]),
        "baseline_prediction_decile": (pred_lab, [f"D{i}" for i in range(1, 11)]),
        "zero_probability_quintile": (pact_lab, [f"Q{i}" for i in range(1, 6)]),
        "recent_channel_mix": (channel, ["no_recent_gmv", "search-heavy", "mixed", "catalog-heavy"]),
    }


def segment_rows(cutoff: str, name: str, labels: np.ndarray, label_names: list[str],
                 r: np.ndarray, B: np.ndarray) -> list[dict[str, Any]]:
    total_sse = float(np.sum(r * r))
    design = design_from_labels(labels)
    whole = gain_from_design(r, design, B)
    rows = []
    for value in np.unique(labels):
        m = labels == value
        rr = r[m]
        indicator = m.astype(np.float64)
        one = gain_from_design(r, indicator, B)
        rows.append({
            "cutoff": cutoff,
            "segment": name,
            "bucket": label_names[int(value)] if int(value) < len(label_names) else str(value),
            "population_share": float(np.mean(m)),
            "mse_share": float(np.sum(rr * rr) / total_sse),
            "mean_signed_residual": float(np.mean(rr)),
            "residual_variance": float(np.var(rr)),
            "oracle_indicator_intercept_gain": one["gain"],
            "whole_segmentation_after_span_gain": whole["gain"],
        })
    return rows


def indicator_gain(r: np.ndarray, m: np.ndarray, B: np.ndarray) -> float:
    return gain_from_design(r, np.asarray(m, np.float64), B)["gain"]


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    exp077 = load_exp077()
    canon = pd.read_parquet(E75 / "clean_forward_predictions.parquet")
    a2 = pd.read_parquet(E75 / "a2_clean_forward_predictions.parquet")
    canon["cutoff"] = canon.cutoff.astype(str)
    a2["cutoff"] = a2.cutoff.astype(str)
    if canon.duplicated(["cutoff", "user_id"]).any():
        raise AssertionError("duplicate canonical keys")
    key = pd.MultiIndex.from_frame(canon[["cutoff", "user_id"]])
    a2_keyed = a2.set_index(["cutoff", "user_id"]).reindex(key)
    if a2_keyed.u_raw_A2.isna().any():
        raise AssertionError("A2 alignment failed")

    y = canon.target_log.to_numpy(np.float64)
    uid = canon.user_id.to_numpy(np.int64)
    cut = canon.cutoff.to_numpy()
    masks = {fold: cut == fold for fold in FOLDS}
    Z, bank_audit = exp077.load_reference_bank(canon)
    shares_src = json.loads((E76 / "out" / "s10_alpha_composition.json").read_text(encoding="utf-8"))
    shares0 = shares_src["SUBMIT_ORTH_ALPHA"]["shares"]
    shares = {"SEQ": shares0["SEQ"], "ETX": shares0["ETX"],
              "TABULAR": shares0["TAB"], "BTYD_OTHER": shares0["BTYD"]}
    z_match, _ = exp077.build_composition_proxy(Z, exp077.REFERENCE_BANK, y, masks, shares)

    u_joint_raw = (A_JOINT[0] * canon.u_raw_365.to_numpy(np.float64)
                   + A_JOINT[1] * a2_keyed.u_raw_A2.to_numpy(np.float64))
    z_current = np.empty(len(y), np.float64)
    d_post = np.empty(len(y), np.float64)
    fold_bases: dict[str, np.ndarray] = {}
    baseline_rows = []
    published = pd.read_csv(E76 / "out" / "s12_sbvc_folds.csv").set_index("cutoff")
    parity_diffs = []
    for fold in FOLDS:
        m = masks[fold]
        B0 = np.column_stack([np.ones(m.sum()), z_match[m], Z[m]])
        up = project_out_matrix(u_joint_raw[m], B0)[:, 0]
        d_post[m] = up
        preclip = z_match[m] + up
        zc = np.maximum(preclip, 0.0)
        z_current[m] = zc
        Bprod = np.column_stack([np.ones(m.sum()), zc, z_match[m], Z[m], up])
        fold_bases[fold] = Bprod
        b = float(np.mean(up * (y[m] - z_match[m])))
        g = float(np.mean(up * up))
        parity_diffs += [abs(b - float(published.loc[fold, "b"])), abs(g - float(published.loc[fold, "G"]))]
        baseline_rows.append({
            "cutoff": fold,
            "n": int(m.sum()),
            "composition_proxy_RMSLE": rms(y[m] - z_match[m]),
            "current_proxy_RMSLE": rms(y[m] - zc),
            "unit_EXP075_Delta_MSE": float(np.mean((y[m] - zc) ** 2 - (y[m] - z_match[m]) ** 2)),
            "clip_count": int(np.sum(preclip < 0)),
            "rms_EXP075_post_span": rms(up),
        })
    r_current = y - z_current
    weighted_mse = float(np.average([np.mean(r_current[masks[f]] ** 2) for f in FOLDS],
                                    weights=[FOLD_WEIGHT[f] for f in FOLDS]))
    weighted_rmsle = float(np.average([rms(r_current[masks[f]]) for f in FOLDS],
                                      weights=[FOLD_WEIGHT[f] for f in FOLDS]))

    uid_all = np.load(UID_PATH, mmap_mode="r")
    panel = np.load(PANEL_PATH, mmap_mode="r")
    gmv = np.load(GMV_PATH, mmap_mode="r")

    mechanism_rows: list[dict[str, Any]] = []
    segment_detail: list[dict[str, Any]] = []
    tail_detail: list[dict[str, Any]] = []
    horizon_detail: list[dict[str, Any]] = []
    oracle_fold_designs: dict[str, np.ndarray] = {}
    observable_seed: dict[str, dict[str, np.ndarray]] = {}
    target_audit = []

    for fold in FOLDS:
        m = masks[fold]
        ids = uid[m]
        r = r_current[m]
        B = fold_bases[fold]
        fut = future_arrays(ids, fold, uid_all, panel, gmv)
        max_y_error = float(np.max(np.abs(fut["y30"] - canon.loc[m, "target_y30"].to_numpy(np.float64))))
        max_log_error = float(np.max(np.abs(np.log1p(fut["y30"]) - y[m])))
        target_audit.append({"cutoff": fold, "max_y30_error": max_y_error,
                             "max_target_log_error": max_log_error})
        if max_y_error > 1e-8 or max_log_error > 1e-10:
            raise AssertionError(f"future target parity failed: {fold}")

        positive = fut["y30"] > 0
        zero = ~positive
        Dzero = design_from_labels(positive.astype(np.int8))
        raw_zero = gain_from_design(r, Dzero)
        span_zero = gain_from_design(r, Dzero, B)
        existing = existing_activity_signals(fold, ids)
        Ezero = np.column_stack([existing[k] for k in ["dist_p_act", "block4_q_event", "btyd_p_act"]])
        stricter_zero = incremental_gain(r, Dzero, B, Ezero)
        mechanism_rows.append({
            "cutoff": fold, "component": "zero_positive",
            "population_share": 1.0,
            "mse_share": 1.0,
            "oracle_removable_mse": raw_zero["gain"],
            "after_span_headroom": span_zero["gain"],
            "after_existing_hurdle_headroom": stricter_zero["gain"],
            "oracle_label_R2_in_production_span": signal_r2_from_basis(positive.astype(np.float64), B),
            "oracle_label_R2_with_explicit_hurdle_signals": signal_r2_from_basis(
                positive.astype(np.float64), np.column_stack([B, Ezero])),
            "oracle_rho_after_span": span_zero["rho"],
            "positive_share": float(np.mean(positive)),
            "zero_mse_share": float(np.sum(r[zero] ** 2) / np.sum(r ** 2)),
            "positive_mse_share": float(np.sum(r[positive] ** 2) / np.sum(r ** 2)),
        })

        pos_share = float(np.mean(positive))
        rp = r[positive]
        Bp = B[positive]
        purchase_label = count_bucket(fut["purchase_days"][positive])
        event_label = event_bucket(fut["event_days"][positive])
        order_label = count_bucket(fut["order_items"][positive])
        count_design = design_from_labels(purchase_label)
        for name, labels in [("purchase_days_count", purchase_label),
                             ("event_days_count", event_label),
                             ("order_items_count", order_label)]:
            D = design_from_labels(labels)
            raw = gain_from_design(rp, D)
            span = gain_from_design(rp, D, Bp)
            mechanism_rows.append({
                "cutoff": fold, "component": name,
                "population_share": pos_share,
                "mse_share": float(np.sum(rp ** 2) / np.sum(r ** 2)),
                "oracle_removable_mse": pos_share * raw["gain"],
                "after_span_headroom": pos_share * span["gain"],
                "oracle_rho_after_span": span["rho"],
                "oracle_label_R2_in_production_span": signal_r2_from_basis(
                    np.log1p(np.asarray(labels, np.float64)), Bp),
                "conditional_relation_corr": corr(np.log1p(np.asarray(labels, np.float64)), rp),
            })

        avg_value = fut["y30"][positive] / np.maximum(fut["purchase_days"][positive], 1)
        value_label = percentile_bins(np.log1p(avg_value), 5, zero_label=0)
        value_design = design_from_labels(value_label)
        count_only = gain_from_design(rp, count_design, Bp)
        count_value = gain_from_design(rp, np.column_stack([count_design, value_design]), Bp)
        monetary_gain = max(count_value["gain"] - count_only["gain"], 0.0)
        mechanism_rows.append({
            "cutoff": fold, "component": "conditional_monetary_value",
            "population_share": pos_share,
            "mse_share": float(np.sum(rp ** 2) / np.sum(r ** 2)),
            "oracle_removable_mse": pos_share * max(
                gain_from_design(rp, np.column_stack([count_design, value_design]))["gain"]
                - gain_from_design(rp, count_design)["gain"], 0.0),
            "after_span_headroom": pos_share * monetary_gain,
            "oracle_rho_after_span": count_value["rho"],
            "oracle_label_R2_in_production_span": signal_r2_from_basis(np.log1p(avg_value), Bp),
            "conditional_relation_corr": corr(np.log1p(avg_value), rp),
            "joint_count_value_after_span": pos_share * count_value["gain"],
        })

        yparts = np.column_stack([fut["y7"], fut["y14"] - fut["y7"], fut["y30"] - fut["y14"]])
        active_bits = (yparts > 0).astype(np.int8)
        pattern = active_bits[:, 0] + 2 * active_bits[:, 1] + 4 * active_bits[:, 2]
        Dpattern = design_from_labels(pattern)
        raw_pattern = gain_from_design(r, Dpattern)
        span_pattern = gain_from_design(r, Dpattern, B)
        shares_h = np.zeros_like(yparts)
        shares_h[positive] = yparts[positive] / fut["y30"][positive, None]
        early_q = percentile_bins(shares_h[:, 0], 5, mask=positive, zero_label=-1)
        late_q = percentile_bins(shares_h[:, 2], 5, mask=positive, zero_label=-1)
        timing_pos = np.column_stack([design_from_labels(pattern[positive]),
                                      design_from_labels(early_q[positive]),
                                      design_from_labels(late_q[positive])])
        count_value_design = np.column_stack([count_design, value_design])
        cv_gain = gain_from_design(rp, count_value_design, Bp)["gain"]
        cvt_gain = gain_from_design(rp, np.column_stack([count_value_design, timing_pos]), Bp)["gain"]
        timing_increment = max(cvt_gain - cv_gain, 0.0)
        mechanism_rows.append({
            "cutoff": fold, "component": "horizon_distribution",
            "population_share": 1.0,
            "mse_share": 1.0,
            "oracle_removable_mse": raw_pattern["gain"],
            "after_span_headroom": span_pattern["gain"],
            "after_count_value_headroom": pos_share * timing_increment,
            "oracle_rho_after_span": span_pattern["rho"],
            "oracle_label_R2_in_production_span": signal_r2_from_basis(Dpattern, B),
        })

        log_alloc = shares_h * y[m, None]
        for j, name in enumerate(["days_1_7", "days_8_14", "days_15_30"]):
            comp = log_alloc[:, j]
            q = gain_from_design(r, comp, B)
            horizon_detail.append({
                "cutoff": fold, "horizon": name,
                "gmv_share": float(np.sum(yparts[:, j]) / max(np.sum(fut["y30"]), 1e-300)),
                "target_log_allocation_share": float(np.sum(comp) / max(np.sum(y[m]), 1e-300)),
                "corr_with_residual": corr(comp, r),
                "after_span_linear_gain": q["gain"],
            })

        # Joint target oracle: count, conditional amount, and timing composition.
        count_all = design_from_labels(count_bucket(fut["purchase_days"]))
        value_all_label = percentile_bins(
            np.divide(fut["y30"], np.maximum(fut["purchase_days"], 1)), 5,
            mask=positive, zero_label=-1,
        )
        value_all = design_from_labels(value_all_label)
        oracle_design = np.column_stack([
            count_all,
            value_all,
            Dpattern,
            design_from_labels(early_q),
            design_from_labels(late_q),
        ])
        oracle_fold_designs[fold] = oracle_design

        state = load_state_features(fold, ids)
        segments = segment_labels(state, z_current[m], existing["dist_p_act"])
        for name, (labels, names) in segments.items():
            segment_detail.extend(segment_rows(fold, name, labels, names, r, B))

        total_sse = float(np.sum(r * r))
        for tail_name, score in [("top_target_gmv", fut["y30"]),
                                 ("top_baseline_prediction", z_current[m]),
                                 ("largest_absolute_residual", np.abs(r))]:
            for pct in (1, 5, 10):
                threshold = float(np.quantile(score, 1.0 - pct / 100.0))
                tail = score >= threshold
                tail_detail.append({
                    "cutoff": fold, "tail": tail_name, "nominal_top_pct": pct,
                    "population_share": float(np.mean(tail)),
                    "mse_share": float(np.sum(r[tail] ** 2) / total_sse),
                    "mean_signed_residual": float(np.mean(r[tail])),
                    "residual_variance": float(np.var(r[tail])),
                    "oracle_indicator_intercept_gain": indicator_gain(r, tail, B),
                    "threshold": threshold,
                })

        observable_seed[fold] = {
            "user_id": ids,
            "residual": r,
            "z_current": z_current[m],
            "target_positive": positive.astype(np.int8),
            "target_purchase_days": fut["purchase_days"],
            "target_event_days": fut["event_days"],
            "target_order_items": fut["order_items"],
            "target_avg_value": np.divide(fut["y30"], np.maximum(fut["purchase_days"], 1)),
            "target_early_share": shares_h[:, 0],
            "target_late_share": shares_h[:, 2],
            **existing,
        }

    # Aggregate oracle rows using the established 1:2:4:8 fold weighting.
    mechanism = pd.DataFrame(mechanism_rows)
    agg_rows = []
    numeric = [c for c in mechanism.columns if c not in ("cutoff", "component")]
    for component, part in mechanism.groupby("component", sort=False):
        row: dict[str, Any] = {"cutoff": "weighted_1_2_4_8", "component": component}
        weights = np.asarray([FOLD_WEIGHT[x] for x in part.cutoff], np.float64)
        for col in numeric:
            vals = part[col].to_numpy(np.float64)
            ok = np.isfinite(vals)
            row[col] = float(np.average(vals[ok], weights=weights[ok])) if ok.any() else np.nan
        agg_rows.append(row)
    mechanism = pd.concat([mechanism, pd.DataFrame(agg_rows)], ignore_index=True)

    # Fold-level joint structural oracle and identity-oracle upper bound.
    joint_oracle_rows = []
    for fold in FOLDS:
        m = masks[fold]
        r = r_current[m]
        structural = gain_from_design(r, oracle_fold_designs[fold], fold_bases[fold])
        identity = gain_from_design(r, y[m], fold_bases[fold])
        joint_oracle_rows.append({
            "cutoff": fold,
            "structural_oracle_after_span": structural["gain"],
            "structural_oracle_rho": structural["rho"],
            "identity_oracle_after_span": identity["gain"],
            "residual_mse": float(np.mean(r * r)),
        })
    jo = pd.DataFrame(joint_oracle_rows)
    w = np.asarray([FOLD_WEIGHT[f] for f in FOLDS], np.float64)
    jo.loc[len(jo)] = {
        "cutoff": "weighted_1_2_4_8",
        **{c: float(np.average(jo[c], weights=w)) for c in jo.columns if c != "cutoff"},
    }

    segment_frame = pd.DataFrame(segment_detail)
    segment_agg = []
    for (seg, bucket), part in segment_frame.groupby(["segment", "bucket"], sort=False):
        weights = np.asarray([FOLD_WEIGHT[x] for x in part.cutoff], np.float64)
        row = {"cutoff": "weighted_1_2_4_8", "segment": seg, "bucket": bucket}
        for col in [c for c in part.columns if c not in ("cutoff", "segment", "bucket")]:
            row[col] = float(np.average(part[col], weights=weights))
        segment_agg.append(row)
    segment_frame = pd.concat([segment_frame, pd.DataFrame(segment_agg)], ignore_index=True)

    tail_frame = pd.DataFrame(tail_detail)
    tail_agg = []
    for (tail, pct), part in tail_frame.groupby(["tail", "nominal_top_pct"], sort=False):
        weights = np.asarray([FOLD_WEIGHT[x] for x in part.cutoff], np.float64)
        row = {"cutoff": "weighted_1_2_4_8", "tail": tail, "nominal_top_pct": pct}
        for col in [c for c in part.columns if c not in ("cutoff", "tail", "nominal_top_pct")]:
            row[col] = float(np.average(part[col], weights=weights))
        tail_agg.append(row)
    tail_frame = pd.concat([tail_frame, pd.DataFrame(tail_agg)], ignore_index=True)

    horizon_frame = pd.DataFrame(horizon_detail)
    horizon_agg = []
    for horizon, part in horizon_frame.groupby("horizon", sort=False):
        weights = np.asarray([FOLD_WEIGHT[x] for x in part.cutoff], np.float64)
        row = {"cutoff": "weighted_1_2_4_8", "horizon": horizon}
        for col in [c for c in part.columns if c not in ("cutoff", "horizon")]:
            row[col] = float(np.average(part[col], weights=weights))
        horizon_agg.append(row)
    horizon_frame = pd.concat([horizon_frame, pd.DataFrame(horizon_agg)], ignore_index=True)

    required_mse = CURRENT_RMSLE ** 2 - TARGET_RMSLE ** 2
    gap_math = {
        "current_RMSLE": CURRENT_RMSLE,
        "target_RMSLE": TARGET_RMSLE,
        "required_Delta_RMSLE_improvement": CURRENT_RMSLE - TARGET_RMSLE,
        "current_MSE": CURRENT_RMSLE ** 2,
        "target_MSE": TARGET_RMSLE ** 2,
        "required_Delta_MSE_gain": required_mse,
        "required_independent_rho": math.sqrt(required_mse) / CURRENT_RMSLE,
        "required_explained_residual_variance_rho2": required_mse / CURRENT_RMSLE ** 2,
        "equivalent_global_intercept_abs_log_error": math.sqrt(required_mse),
        "required_MSE_reduction_per_user": required_mse,
        "required_total_SSE_reduction_for_250k_users": required_mse * 250_000,
        "fraction_current_residual_MSE_to_explain": required_mse / CURRENT_RMSLE ** 2,
    }
    gate = {}
    weighted_mech = mechanism[mechanism.cutoff == "weighted_1_2_4_8"].set_index("component")
    for component in weighted_mech.index:
        headroom = float(weighted_mech.loc[component, "after_span_headroom"])
        gate[component] = {
            "oracle_incremental_Delta_MSE": headroom,
            "passes_oracle_0_001_gate": headroom >= 0.0010,
        }

    audit = {
        "phase": "oracle_only_no_training",
        "models_trained": 0,
        "forbidden_activity_feature_loaded": False,
        "raw_sha256": sha256(RAW),
        "canonical_oof_sha256": sha256(E75 / "clean_forward_predictions.parquet"),
        "a2_oof_sha256": sha256(E75 / "a2_clean_forward_predictions.parquet"),
        "current_submission_sha256": sha256(ROOT / "submissions" / "SUBMIT_EXP075_JOINT_A1_365_A2.csv"),
        "orth_alpha_sha256": sha256(Path(r"C:\Users\Admin\Downloads\SUBMIT_ORTH_ALPHA.csv")),
        "production_bank_columns": int(Z.shape[1]),
        "canonical_rows": int(len(canon)),
        "folds": FOLDS,
        "fold_weights": FOLD_WEIGHT,
        "target_parity": target_audit,
        "EXP076_b_G_max_abs_parity_error": max(parity_diffs),
        "weighted_production_like_current_RMSLE": weighted_rmsle,
        "weighted_production_like_current_MSE": weighted_mse,
        "production_baseline_definition": (
            "composition-matched SUBMIT_ORTH_ALPHA proxy + foldwise EXP075 joint raw correction "
            "projected out of [1,z_match,40-component OOF bank], unit deployed amplitude, z>=0 clip"
        ),
        "projection_basis": "[1,z_current,z_match,40 clean OOF components,EXP075_postspan] per fold",
        "oracle_gate": gate,
        "bank_audit_rows": int(len(bank_audit)),
    }

    pd.DataFrame(baseline_rows).to_csv(HERE / "baseline_folds.csv", index=False)
    mechanism.to_csv(HERE / "oracle_components.csv", index=False)
    segment_frame.to_csv(HERE / "segment_attribution.csv", index=False)
    tail_frame.to_csv(HERE / "tail_attribution.csv", index=False)
    horizon_frame.to_csv(HERE / "horizon_attribution.csv", index=False)
    jo.to_csv(HERE / "joint_oracle.csv", index=False)
    bank_audit.to_csv(HERE / "production_bank_audit.csv", index=False)
    write_json(HERE / "gap_math.json", gap_math)
    write_json(HERE / "oracle_audit.json", audit)
    np.savez_compressed(
        HERE / "oracle_working_arrays.npz",
        user_id=uid,
        cutoff=cut,
        target_log=y,
        z_match=z_match,
        z_current=z_current,
        residual_current=r_current,
        d_exp075_postspan=d_post,
    )
    seed_arrays = {}
    for fold, values in observable_seed.items():
        tag = fold.replace("-", "")
        for name, value in values.items():
            seed_arrays[f"{tag}__{name}"] = value
    np.savez_compressed(HERE / "observable_seed_arrays.npz", **seed_arrays)
    print(json.dumps(jsonable({
        "gap_math": gap_math,
        "baseline": {"wRMSLE": weighted_rmsle, "wMSE": weighted_mse},
        "oracle_gate": gate,
        "joint_oracle": jo.iloc[-1].to_dict(),
    }), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
