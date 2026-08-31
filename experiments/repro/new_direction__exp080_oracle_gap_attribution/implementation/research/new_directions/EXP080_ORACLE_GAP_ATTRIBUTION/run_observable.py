"""EXP080 phase 2: cheap target-free predictability tests for gated mechanisms.

Only the three mechanisms that passed the phase-1 oracle gate are evaluated:
activity, future count, and conditional monetary value.  Models are fixed ridge
regressions trained on four historical snapshots whose 30-day targets end before
each validation cutoff.  No direct GMV30 model, ensemble reweighting, target-
derived eligibility signal, or leaderboard tuning is used.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

import run_oracle as oracle


HERE = Path(__file__).resolve().parent
FOLDS = oracle.FOLDS
FOLD_WEIGHT = oracle.FOLD_WEIGHT
TRAIN_LAGS = [77, 63, 49, 35]
RIDGE_ALPHA = 100.0
BOOTSTRAP_REPS = 1000
BOOTSTRAP_SEED = 20260828

WINDOWS = [7, 30, 90, 180]
WINDOW_METRICS = [
    "days_present", "days_search", "days_buy", "days_cart", "days_presence_only",
    "searches", "carts", "orders", "gmv", "gmv_cat", "gmv_max", "lgmv_mean",
    "lgmv_std", "aov", "buyday_rate", "presence_rate", "ponly_share",
    "cat_gmv_share", "gmv30eq",
]
FEATURES = [f"w{w}_{metric}" for w in WINDOWS for metric in WINDOW_METRICS] + [
    "rec_any", "rec_search", "rec_cart", "rec_buy", "rec_cat", "gap_mean", "gap_std",
    "buygap_mean", "buygap_std", "weekend_share", "tenure_frac", "first_buy_frac",
    "gap_max_frac", "rec_over_buygap", "rec_over_gap", "gap_cv", "buygap_cv",
    "trend_gmv_7_30", "trend_pres_7_30", "trend_srch_7_30", "dlog_gmv_7_30",
    "dlog_buyd_7_30", "trend_gmv_30_90", "trend_pres_30_90", "trend_srch_30_90",
    "dlog_gmv_30_90", "dlog_buyd_30_90", "trend_gmv_60_180",
    "trend_pres_60_180", "trend_srch_60_180", "dlog_gmv_60_180", "dlog_buyd_60_180",
]


def date_minus(value: str, days: int) -> str:
    return str((np.datetime64(value) - np.timedelta64(days, "D")).astype("datetime64[D]"))


def eligible_ids(cutoff: str) -> np.ndarray:
    path = oracle.PROCESSED / f"panel_{cutoff.replace('-', '')}_b3.parquet"
    return pd.read_parquet(path, columns=["user_id"]).user_id.to_numpy(np.int64)


def load_feature_matrix(cutoff: str, ids: np.ndarray) -> tuple[np.ndarray, float]:
    path = oracle.PROCESSED / f"feat_{cutoff.replace('-', '')}_LnormNone.parquet"
    frame = pd.read_parquet(path, columns=["user_id", *FEATURES]).sort_values("user_id")
    source_ids = frame.user_id.to_numpy(np.int64)
    pos = np.searchsorted(source_ids, ids)
    if pos.max(initial=0) >= len(frame) or not np.array_equal(source_ids[pos], ids):
        raise AssertionError(f"feature alignment failed: {cutoff}")
    aligned = frame.iloc[pos]
    X = aligned[FEATURES].to_numpy(np.float32)
    finite_before = float(np.mean(np.isfinite(X)))
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    # Fixed robust compression; no target or fold-specific tuning.
    X = np.sign(X) * np.log1p(np.abs(X))
    return X.astype(np.float32, copy=False), finite_before


def build_dataset(cutoff: str, ids: np.ndarray, uid_all: np.ndarray,
                  panel: np.ndarray, gmv: np.ndarray) -> dict[str, Any]:
    X, finite_before = load_feature_matrix(cutoff, ids)
    fut = oracle.future_arrays(ids, cutoff, uid_all, panel, gmv)
    positive = fut["y30"] > 0
    avg_value = np.divide(fut["y30"], np.maximum(fut["purchase_days"], 1))
    # Target-free feature parity against the primary GMV mmap.
    rows = np.searchsorted(uid_all, ids)
    d = int((np.datetime64(cutoff) - oracle.DATA_START).astype("timedelta64[D]").astype(int))
    hist30 = np.asarray(gmv[rows, max(0, d - 29):d + 1], np.float64).sum(axis=1)
    raw_feature = pd.read_parquet(
        oracle.PROCESSED / f"feat_{cutoff.replace('-', '')}_LnormNone.parquet",
        columns=["user_id", "w30_gmv"],
    ).sort_values("user_id")
    sid = raw_feature.user_id.to_numpy(np.int64)
    spos = np.searchsorted(sid, ids)
    feature_gmv = raw_feature.w30_gmv.to_numpy(np.float64)[spos]
    return {
        "cutoff": cutoff,
        "user_id": ids,
        "X": X,
        "positive": positive.astype(np.float64),
        "log_purchase_days": np.log1p(fut["purchase_days"].astype(np.float64)),
        "log_event_days": np.log1p(fut["event_days"].astype(np.float64)),
        "log_order_items": np.log1p(fut["order_items"].astype(np.float64)),
        "log_avg_value": np.log1p(avg_value),
        "feature_finite_fraction_before_fill": finite_before,
        "w30_gmv_max_abs_parity_error": float(np.max(np.abs(feature_gmv - hist30))),
    }


def standardize(train: np.ndarray, val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, dtype=np.float64)
    std = train.std(axis=0, dtype=np.float64)
    std[std < 1e-6] = 1.0
    return ((train - mean) / std).astype(np.float32), ((val - mean) / std).astype(np.float32)


def fit_ridge_predictions(train_sets: list[dict[str, Any]], val: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    Xtr = np.concatenate([x["X"] for x in train_sets], axis=0)
    Xv = val["X"]
    Xtr, Xv = standardize(Xtr, Xv)
    y_all = np.column_stack([
        np.concatenate([x["positive"] for x in train_sets]),
        np.concatenate([x["log_purchase_days"] for x in train_sets]),
        np.concatenate([x["log_event_days"] for x in train_sets]),
        np.concatenate([x["log_order_items"] for x in train_sets]),
    ])
    all_model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True, solver="lsqr", tol=1e-4, max_iter=1000)
    all_model.fit(Xtr, y_all)
    pred_all = all_model.predict(Xv)

    pos_parts = [x["positive"] > 0 for x in train_sets]
    Xpos = np.concatenate([x["X"][m] for x, m in zip(train_sets, pos_parts)], axis=0)
    # Reuse the all-row standardization by applying it before positive selection.
    mean = np.concatenate([x["X"] for x in train_sets], axis=0).mean(axis=0, dtype=np.float64)
    std = np.concatenate([x["X"] for x in train_sets], axis=0).std(axis=0, dtype=np.float64)
    std[std < 1e-6] = 1.0
    Xpos = ((Xpos - mean) / std).astype(np.float32)
    ypos = np.concatenate([x["log_avg_value"][m] for x, m in zip(train_sets, pos_parts)])
    value_model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True, solver="lsqr", tol=1e-4, max_iter=1000)
    value_model.fit(Xpos, ypos)
    pred_value = value_model.predict(Xv)
    pred = {
        "ridge_p_active": np.clip(pred_all[:, 0], 0.0, 1.0),
        "ridge_log_purchase_days": pred_all[:, 1],
        "ridge_log_event_days": pred_all[:, 2],
        "ridge_log_order_items": pred_all[:, 3],
        "ridge_log_cond_value": pred_value,
    }
    pos = val["positive"] > 0
    diagnostics = {
        "train_rows": int(len(Xtr)),
        "train_positive_rows": int(len(Xpos)),
        "validation_rows": int(len(Xv)),
        "validation_positive_rows": int(np.sum(pos)),
        "activity_AUC": float(roc_auc_score(val["positive"], pred["ridge_p_active"])),
        "purchase_count_corr_all": oracle.corr(pred["ridge_log_purchase_days"], val["log_purchase_days"]),
        "event_count_corr_all": oracle.corr(pred["ridge_log_event_days"], val["log_event_days"]),
        "order_count_corr_all": oracle.corr(pred["ridge_log_order_items"], val["log_order_items"]),
        "conditional_value_corr_positive": oracle.corr(pred_value[pos], val["log_avg_value"][pos]),
    }
    return pred, diagnostics


def stable_segment_design(cutoff: str, ids: np.ndarray, z_current: np.ndarray,
                          p_act: np.ndarray) -> tuple[np.ndarray, list[str]]:
    state = oracle.load_state_features(cutoff, ids)
    groups = oracle.segment_labels(state, z_current, p_act)
    columns, names = [], []
    for seg, (labels, label_names) in groups.items():
        # Drop the first bucket; the production basis already has an intercept.
        for value in range(1, len(label_names)):
            columns.append((labels == value).astype(np.float64))
            names.append(f"{seg}:{label_names[value]}")
    return np.column_stack(columns), names


def block_local_metrics(r: np.ndarray, primary: np.ndarray, block: np.ndarray,
                        z_current: np.ndarray, Bprod: np.ndarray,
                        previous: np.ndarray | None) -> dict[str, Any]:
    ones = np.ones((len(r), 1), np.float64)
    raw_primary = oracle.project_out_matrix(primary, ones)[:, 0]
    strong_primary = oracle.project_out_matrix(primary, np.column_stack([ones, z_current]))[:, 0]
    post_primary = oracle.project_out_matrix(primary, Bprod)[:, 0]
    inc = oracle.incremental_gain(r, block, Bprod, previous)
    return {
        "rho_raw": oracle.corr(raw_primary, r),
        "rho_after_strong_baseline": oracle.corr(strong_primary, r),
        "rho_after_projection": oracle.corr(post_primary, r),
        "incremental_multiple_rho": inc["rho"],
        "observable_optimal_incremental_gain": inc["gain"],
        "incremental_rank": inc["rank"],
    }


def moments(U: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return U.T @ U / len(r), U.T @ r / len(r)


def nested_for_columns(per_fold: dict[str, dict[str, np.ndarray]], n_cols: int) -> tuple[
        pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    rows = []
    deltas: dict[str, np.ndarray] = {}
    corrections: dict[str, np.ndarray] = {}
    past_G: list[np.ndarray] = []
    past_b: list[np.ndarray] = []
    past_w: list[float] = []
    for k, fold in enumerate(FOLDS):
        U = per_fold[fold]["U"][:, :n_cols]
        r = per_fold[fold]["r"]
        G, b = moments(U, r)
        if k == 0:
            beta = np.zeros(n_cols, np.float64)
            source = "zero_no_prior_validation_fold"
        else:
            Gp = sum(w * x for w, x in zip(past_w, past_G))
            bp = sum(w * x for w, x in zip(past_w, past_b))
            beta = np.linalg.pinv(Gp, rcond=1e-8) @ bp
            source = "strictly_earlier_clean_validation_folds"
        q = U @ beta
        delta = (r - q) ** 2 - r ** 2
        deltas[fold] = delta
        corrections[fold] = q
        rows.append({
            "cutoff": fold,
            "n_columns": n_cols,
            "coefficient_source": source,
            "nested_Delta_MSE": float(np.mean(delta)),
            "nested_Delta_RMSLE": float(np.sqrt(np.mean((r - q) ** 2)) - np.sqrt(np.mean(r ** 2))),
            "nested_rho": oracle.corr(q, r) if np.any(q) else 0.0,
            "coefficient_l2": float(np.sqrt(beta @ beta)),
        })
        past_G.append(G)
        past_b.append(b)
        past_w.append(FOLD_WEIGHT[fold])
    return pd.DataFrame(rows), deltas, corrections


def cluster_bootstrap_delta(per_fold: dict[str, dict[str, np.ndarray]],
                            deltas: dict[str, np.ndarray], reps: int,
                            seed: int) -> dict[str, Any]:
    uid = np.concatenate([per_fold[f]["uid"] for f in FOLDS])
    all_delta = np.concatenate([deltas[f] for f in FOLDS])
    fold_index = np.concatenate([np.full(len(deltas[f]), i, np.int8) for i, f in enumerate(FOLDS)])
    unique, inv = np.unique(uid, return_inverse=True)
    fold_n = np.bincount(fold_index, minlength=len(FOLDS)).astype(np.float64)
    fw = np.asarray([FOLD_WEIGHT[f] for f in FOLDS], np.float64)
    row_w = fw[fold_index] / fold_n[fold_index] / fw.sum()
    cluster_w = np.bincount(inv, weights=row_w, minlength=len(unique))
    cluster_d = np.bincount(inv, weights=row_w * all_delta, minlength=len(unique))
    rng = np.random.default_rng(seed)
    draws = np.empty(reps, np.float64)
    for start in range(0, reps, 10):
        n = min(10, reps - start)
        count = rng.poisson(1.0, size=(n, len(unique))).astype(np.float64)
        draws[start:start + n] = (count @ cluster_d) / np.maximum(count @ cluster_w, 1e-300)
    return {
        "replicates": reps,
        "seed": seed,
        "unit": "user_id cluster across all folds",
        "CI95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "P_Delta_MSE_lt_0": float(np.mean(draws < 0)),
        "draw_mean": float(np.mean(draws)),
    }


def main() -> None:
    t0 = time.time()
    work = np.load(HERE / "oracle_working_arrays.npz", allow_pickle=True)
    seeds = np.load(HERE / "observable_seed_arrays.npz", allow_pickle=False)
    canon = pd.read_parquet(oracle.E75 / "clean_forward_predictions.parquet")
    canon["cutoff"] = canon.cutoff.astype(str)
    y = work["target_log"].astype(np.float64)
    uid = work["user_id"].astype(np.int64)
    cut = work["cutoff"].astype(str)
    z_match = work["z_match"].astype(np.float64)
    z_current = work["z_current"].astype(np.float64)
    d_post = work["d_exp075_postspan"].astype(np.float64)
    residual = work["residual_current"].astype(np.float64)
    masks = {fold: cut == fold for fold in FOLDS}

    exp077 = oracle.load_exp077()
    Z, _ = exp077.load_reference_bank(canon)
    fold_bases = {
        fold: np.column_stack([
            np.ones(masks[fold].sum()), z_current[masks[fold]], z_match[masks[fold]],
            Z[masks[fold]], d_post[masks[fold]],
        ]) for fold in FOLDS
    }

    uid_all = np.load(oracle.UID_PATH, mmap_mode="r")
    panel = np.load(oracle.PANEL_PATH, mmap_mode="r")
    gmv = np.load(oracle.GMV_PATH, mmap_mode="r")
    cache: dict[str, dict[str, Any]] = {}
    data_audit = []

    def get_dataset(cutoff: str, ids: np.ndarray | None = None) -> dict[str, Any]:
        if cutoff in cache:
            return cache[cutoff]
        chosen = eligible_ids(cutoff) if ids is None else ids
        ds = build_dataset(cutoff, chosen, uid_all, panel, gmv)
        cache[cutoff] = ds
        data_audit.append({
            "cutoff": cutoff,
            "rows": len(chosen),
            "feature_finite_fraction_before_fill": ds["feature_finite_fraction_before_fill"],
            "w30_gmv_max_abs_parity_error": ds["w30_gmv_max_abs_parity_error"],
        })
        return ds

    prediction_rows = []
    model_diagnostics = []
    per_fold_blocks: dict[str, dict[str, np.ndarray]] = {}
    local_rows = []
    segment_names_ref: list[str] | None = None

    for fold in FOLDS:
        m = masks[fold]
        ids = uid[m]
        val = get_dataset(fold, ids)
        train_cutoffs = [date_minus(fold, lag) for lag in TRAIN_LAGS]
        for train_cutoff in train_cutoffs:
            if np.datetime64(train_cutoff) + np.timedelta64(30, "D") > np.datetime64(fold):
                raise AssertionError("training target crosses validation cutoff")
        train_sets = [get_dataset(x) for x in train_cutoffs]
        pred, diag = fit_ridge_predictions(train_sets, val)
        diag.update({"cutoff": fold, "train_cutoffs": train_cutoffs,
                     "max_training_target_end": str(max(np.datetime64(x) + np.timedelta64(30, "D") for x in train_cutoffs))})
        model_diagnostics.append(diag)

        tag = fold.replace("-", "")
        existing = {
            "dist_p_act": seeds[f"{tag}__dist_p_act"].astype(np.float64),
            "block4_q_event": seeds[f"{tag}__block4_q_event"].astype(np.float64),
            "btyd_p_act": seeds[f"{tag}__btyd_p_act"].astype(np.float64),
            "btyd_expected_count": seeds[f"{tag}__btyd_expected_count"].astype(np.float64),
        }
        activity = np.column_stack([
            pred["ridge_p_active"], existing["dist_p_act"],
            existing["block4_q_event"], existing["btyd_p_act"],
        ])
        count = np.column_stack([
            pred["ridge_log_purchase_days"], pred["ridge_log_event_days"],
            pred["ridge_log_order_items"], existing["btyd_expected_count"],
        ])
        value = pred["ridge_log_cond_value"][:, None]
        segments, segment_names = stable_segment_design(fold, ids, z_current[m], existing["dist_p_act"])
        if segment_names_ref is None:
            segment_names_ref = segment_names
        elif segment_names != segment_names_ref:
            raise AssertionError("segment design drift")
        blocks = {"activity": activity, "count": count, "monetary": value,
                  "user_state_segments": segments}
        primary = {
            "activity": pred["ridge_p_active"],
            "count": pred["ridge_log_purchase_days"],
            "monetary": pred["ridge_log_cond_value"],
            "user_state_segments": segments[:, 0],
        }
        previous = np.empty((len(ids), 0), np.float64)
        for name in ["activity", "count", "monetary", "user_state_segments"]:
            met = block_local_metrics(residual[m], primary[name], blocks[name], z_current[m],
                                      fold_bases[fold], previous)
            local_rows.append({"cutoff": fold, "component": name, **met})
            previous = np.column_stack([previous, blocks[name]])

        Uraw = np.column_stack([activity, count, value, segments])
        U = oracle.project_out_matrix(Uraw, fold_bases[fold])
        per_fold_blocks[fold] = {"uid": ids, "r": residual[m], "U": U}
        pred_frame = pd.DataFrame({"user_id": ids, "cutoff": fold, **pred})
        prediction_rows.append(pred_frame)

    local = pd.DataFrame(local_rows)
    weights = np.asarray([FOLD_WEIGHT[f] for f in FOLDS], np.float64)
    local_agg = []
    for component, part in local.groupby("component", sort=False):
        part = part.set_index("cutoff").loc[FOLDS]
        row: dict[str, Any] = {"cutoff": "weighted_1_2_4_8", "component": component}
        for col in [c for c in part.columns if c != "component"]:
            row[col] = float(np.average(part[col].to_numpy(np.float64), weights=weights))
        row["latest_rho_after_projection"] = float(part.loc[FOLDS[-1], "rho_after_projection"])
        row["latest_incremental_multiple_rho"] = float(part.loc[FOLDS[-1], "incremental_multiple_rho"])
        local_agg.append(row)
    local = pd.concat([local, pd.DataFrame(local_agg)], ignore_index=True)

    block_widths = {"activity": 4, "count": 4, "monetary": 1,
                    "user_state_segments": len(segment_names_ref or [])}
    cumulative_widths = {}
    total = 0
    nested_frames = []
    cumulative_deltas: dict[str, dict[str, np.ndarray]] = {}
    cumulative_corrections: dict[str, dict[str, np.ndarray]] = {}
    for component in ["activity", "count", "monetary", "user_state_segments"]:
        total += block_widths[component]
        cumulative_widths[component] = total
        nf, deltas, corrections = nested_for_columns(per_fold_blocks, total)
        nf["through_component"] = component
        nested_frames.append(nf)
        cumulative_deltas[component] = deltas
        cumulative_corrections[component] = corrections
    nested = pd.concat(nested_frames, ignore_index=True)

    # Marginal forward deltas respect covariance: cumulative(k)-cumulative(k-1).
    marginal_rows = []
    marginal_deltas: dict[str, dict[str, np.ndarray]] = {}
    marginal_corrections: dict[str, dict[str, np.ndarray]] = {}
    previous_name = None
    for component in ["activity", "count", "monetary", "user_state_segments"]:
        marginal_deltas[component] = {}
        marginal_corrections[component] = {}
        for fold in FOLDS:
            current_delta = cumulative_deltas[component][fold]
            prior_delta = np.zeros_like(current_delta) if previous_name is None else cumulative_deltas[previous_name][fold]
            d = current_delta - prior_delta
            current_q = cumulative_corrections[component][fold]
            prior_q = np.zeros_like(current_q) if previous_name is None else cumulative_corrections[previous_name][fold]
            q = current_q - prior_q
            marginal_deltas[component][fold] = d
            marginal_corrections[component][fold] = q
            marginal_rows.append({"cutoff": fold, "component": component,
                                  "nested_marginal_Delta_MSE": float(np.mean(d)),
                                  "nested_marginal_rho": oracle.corr(q, per_fold_blocks[fold]["r"]) if np.any(q) else 0.0})
        previous_name = component
    marginal = pd.DataFrame(marginal_rows)

    bootstrap = {}
    for i, component in enumerate(["activity", "count", "monetary", "user_state_segments"]):
        bootstrap[component] = cluster_bootstrap_delta(
            per_fold_blocks, marginal_deltas[component], BOOTSTRAP_REPS, BOOTSTRAP_SEED + i)
    bootstrap["joint_all"] = cluster_bootstrap_delta(
        per_fold_blocks, cumulative_deltas["user_state_segments"], BOOTSTRAP_REPS, BOOTSTRAP_SEED + 10)

    # Gate denominators are deliberately non-additive oracle classes.
    oracle_components = pd.read_csv(HERE / "oracle_components.csv")
    o = oracle_components[oracle_components.cutoff == "weighted_1_2_4_8"].set_index("component")
    oracle_denominator = {
        "activity": float(o.loc["zero_positive", "after_span_headroom"]),
        "count": float(max(o.loc["purchase_days_count", "after_span_headroom"],
                           o.loc["event_days_count", "after_span_headroom"],
                           o.loc["order_items_count", "after_span_headroom"])),
        "monetary": float(o.loc["conditional_monetary_value", "after_span_headroom"]),
        "user_state_segments": float(pd.read_csv(HERE / "segment_attribution.csv")
                                     .query("cutoff == 'weighted_1_2_4_8'")
                                     .groupby("segment").whole_segmentation_after_span_gain.first().sum()),
    }
    weighted_local = local[local.cutoff == "weighted_1_2_4_8"].set_index("component")
    # For the combined state specialist, the correct oracle class is the joint
    # 29-dummy design after the three structural observable blocks, not the sum
    # of seven overlapping one-segmentation gains.
    oracle_denominator["user_state_segments"] = float(
        weighted_local.loc["user_state_segments", "observable_optimal_incremental_gain"])
    weighted_marginal = marginal[marginal.cutoff != "weighted_1_2_4_8"].copy()
    gate_rows = []
    for component in ["activity", "count", "monetary", "user_state_segments"]:
        optimistic_gain = float(weighted_local.loc[component, "observable_optimal_incremental_gain"])
        part_m = weighted_marginal[weighted_marginal.component == component].set_index("cutoff").loc[FOLDS]
        nested_delta = float(np.average(part_m.nested_marginal_Delta_MSE, weights=weights))
        observable_gain = max(0.0, -nested_delta)
        fraction = observable_gain / max(oracle_denominator[component], 1e-300)
        latest_rho = float(part_m.loc[FOLDS[-1], "nested_marginal_rho"])
        oracle_pass = oracle_denominator[component] >= 0.0010
        predict_pass = fraction >= 0.25 or latest_rho >= 0.020
        gate_rows.append({
            "component": component,
            "oracle_headroom": oracle_denominator[component],
            "observable_headroom": observable_gain,
            "optimistic_heldout_feature_space_headroom": optimistic_gain,
            "observable_fraction_of_oracle": fraction,
            "weighted_rho_after_projection": float(weighted_local.loc[component, "rho_after_projection"]),
            "latest_clean_nested_incremental_rho": latest_rho,
            "oracle_gate_ge_0_001": oracle_pass,
            "predictability_gate": predict_pass,
            "verdict": "GO" if oracle_pass and predict_pass else (
                "REJECT_ORACLE_LT_0.001" if not oracle_pass else "REJECT_UNOBSERVABLE"
            ),
        })
    gates = pd.DataFrame(gate_rows)

    marginal_agg = []
    for component, part in marginal.groupby("component", sort=False):
        p = part.set_index("cutoff").loc[FOLDS]
        point = float(np.average(p.nested_marginal_Delta_MSE, weights=weights))
        marginal_agg.append({
            "cutoff": "weighted_1_2_4_8", "component": component,
            "nested_marginal_Delta_MSE": point,
            "nested_marginal_Delta_RMSLE_equivalent": point / (2 * oracle.CURRENT_RMSLE),
            "nested_marginal_rho": float(np.average(p.nested_marginal_rho, weights=weights)),
        })
    marginal = pd.concat([marginal, pd.DataFrame(marginal_agg)], ignore_index=True)

    joint_optimal = 0.0
    joint_rho = 0.0
    for fold in FOLDS:
        U = per_fold_blocks[fold]["U"]
        r = per_fold_blocks[fold]["r"]
        q = oracle.gain_from_design(r, U)
        joint_optimal += FOLD_WEIGHT[fold] * q["gain"] / sum(FOLD_WEIGHT.values())
        joint_rho += FOLD_WEIGHT[fold] * q["rho"] / sum(FOLD_WEIGHT.values())
    joint_nested_delta = float(np.average(
        [np.mean(cumulative_deltas["user_state_segments"][f]) for f in FOLDS], weights=weights))
    joint_boot = bootstrap["joint_all"]
    robust_forward_headroom = max(0.0, -float(joint_boot["CI95"][1]))
    attainability = {
        "required_MSE_gain": oracle.CURRENT_RMSLE ** 2 - oracle.TARGET_RMSLE ** 2,
        "theoretical_structural_oracle_headroom": float(
            pd.read_csv(HERE / "joint_oracle.csv").query("cutoff == 'weighted_1_2_4_8'")
            .structural_oracle_after_span.iloc[0]),
        "identity_oracle_headroom": float(
            pd.read_csv(HERE / "joint_oracle.csv").query("cutoff == 'weighted_1_2_4_8'")
            .identity_oracle_after_span.iloc[0]),
        "observable_joint_optimal_headroom": joint_optimal,
        "observable_joint_optimal_rho": joint_rho,
        "joint_nested_forward_Delta_MSE": joint_nested_delta,
        "joint_nested_forward_headroom_point": max(0.0, -joint_nested_delta),
        "robust_forward_headroom_95pct_lower_bound": robust_forward_headroom,
        "joint_nested_bootstrap": joint_boot,
        "attainability": (
            "YES" if robust_forward_headroom >= oracle.CURRENT_RMSLE ** 2 - oracle.TARGET_RMSLE ** 2
            else ("PARTIAL" if max(0.0, -joint_nested_delta) >= 0.25 * (oracle.CURRENT_RMSLE ** 2 - oracle.TARGET_RMSLE ** 2)
                  else "NO_EVIDENCE")
        ),
    }

    audit = {
        "phase": "target_free_observable_predictability",
        "models": "fixed multi-output Ridge; no target/log-GMV30 direct model",
        "ridge_alpha": RIDGE_ALPHA,
        "features": FEATURES,
        "feature_count": len(FEATURES),
        "feature_source": "cutoff-safe feat_*_LnormNone joined to raw-verified eligible IDs",
        "training_protocol": "for each validation cutoff train on cutoff-[77,63,49,35] days; all 30d targets end before validation cutoff",
        "forbidden_activity_feature_loaded": False,
        "leaderboard_used_for_selection": False,
        "direct_GMV30_model_trained": False,
        "data_audit": data_audit,
        "model_diagnostics": model_diagnostics,
        "runtime_seconds": time.time() - t0,
    }

    pd.concat(prediction_rows, ignore_index=True).to_parquet(HERE / "observable_predictions.parquet", index=False)
    local.to_csv(HERE / "observable_fold_metrics.csv", index=False)
    nested.to_csv(HERE / "observable_nested_cumulative.csv", index=False)
    marginal.to_csv(HERE / "observable_nested_marginal.csv", index=False)
    gates.to_csv(HERE / "mechanism_gates.csv", index=False)
    oracle.write_json(HERE / "observable_bootstrap.json", bootstrap)
    oracle.write_json(HERE / "attainability.json", attainability)
    oracle.write_json(HERE / "observable_audit.json", audit)
    print(json.dumps(oracle.jsonable({
        "gates": gates.to_dict("records"),
        "attainability": attainability,
        "runtime_seconds": audit["runtime_seconds"],
    }), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
