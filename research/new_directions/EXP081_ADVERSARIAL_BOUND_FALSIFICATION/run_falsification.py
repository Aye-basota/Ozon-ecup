"""EXP081: adversarial falsification of the EXP080 observable-information bound.

The primary unit is the canonical clean OOF row.  All candidate inputs are
available at the row cutoff.  The script distinguishes:

* fixed target-free bases with coefficients estimated on earlier clean folds;
* user-cross-fitted nonlinear diagnostics (an optimistic conditional test);
* ordered-forward models trained on earlier canonical folds;
* a genuinely label-availability-purged latest-fold check.

No leaderboard value is used for model selection and no TEST target exists.
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

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import roc_auc_score


HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parents[1]
ROOT = HERE.parents[2]
EXP080 = RESEARCH / "new_directions" / "EXP080_ORACLE_GAP_ATTRIBUTION"
EXP077 = RESEARCH / "new_directions" / "EXP077_FORWARD_STACK"
EXP075 = RESEARCH / "new_directions" / "EXP075_OUT_OF_SPAN_RESIDUAL_SIGNALS"
OZON = Path(r"C:\Users\Admin\Desktop\OZON-E-CUP")
PROCESSED = OZON / "data" / "processed"

sys.path.insert(0, str(EXP080))
import run_oracle as e80  # noqa: E402
import run_observable as e80obs  # noqa: E402


FOLDS = e80.FOLDS
FOLD_WEIGHT = e80.FOLD_WEIGHT
WEIGHTS = np.asarray([FOLD_WEIGHT[f] for f in FOLDS], np.float64)
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20260828
EPS = 1e-12

LGBM_CONFIGS: dict[str, dict[str, Any]] = {
    "A_depth3": {
        "n_estimators": 180, "learning_rate": 0.035, "max_depth": 3,
        "num_leaves": 8, "min_child_samples": 1500, "subsample": 0.80,
        "subsample_freq": 1, "colsample_bytree": 0.72, "reg_alpha": 3.0,
        "reg_lambda": 30.0, "max_bin": 127,
    },
    "B_depth5": {
        "n_estimators": 240, "learning_rate": 0.025, "max_depth": 5,
        "num_leaves": 20, "min_child_samples": 1200, "subsample": 0.80,
        "subsample_freq": 1, "colsample_bytree": 0.65, "reg_alpha": 5.0,
        "reg_lambda": 50.0, "max_bin": 127,
    },
}


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


def write_json(name: str, value: Any) -> None:
    (HERE / name).write_text(
        json.dumps(jsonable(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rank01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    out = np.empty(len(x), np.float32)
    out[order] = (np.arange(len(x), dtype=np.float32) + 0.5) / len(x)
    return out


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    return e80.corr(np.asarray(x, np.float64), np.asarray(y, np.float64))


def align_frame(frame: pd.DataFrame, ids: np.ndarray, cutoff_col: str = "cutoff") -> pd.DataFrame:
    if cutoff_col in frame.columns:
        raise ValueError("filter cutoff before align_frame")
    frame = frame.sort_values("user_id")
    source = frame.user_id.to_numpy(np.int64)
    pos = np.searchsorted(source, ids)
    if pos.max(initial=0) >= len(source) or not np.array_equal(source[pos], ids):
        raise AssertionError("frame alignment failed")
    return frame.iloc[pos].reset_index(drop=True)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def weighted_row_mean(values: dict[str, float]) -> float:
    return float(np.average([values[f] for f in FOLDS], weights=WEIGHTS))


def cluster_bootstrap(deltas: dict[str, np.ndarray], uids: dict[str, np.ndarray],
                      reps: int, seed: int) -> dict[str, Any]:
    uid = np.concatenate([uids[f] for f in FOLDS])
    delta = np.concatenate([deltas[f] for f in FOLDS])
    fold_index = np.concatenate([
        np.full(len(deltas[f]), i, np.int8) for i, f in enumerate(FOLDS)
    ])
    unique, inv = np.unique(uid, return_inverse=True)
    fold_n = np.bincount(fold_index, minlength=len(FOLDS)).astype(np.float64)
    row_w = WEIGHTS[fold_index] / fold_n[fold_index] / WEIGHTS.sum()
    cluster_w = np.bincount(inv, weights=row_w, minlength=len(unique))
    cluster_d = np.bincount(inv, weights=row_w * delta, minlength=len(unique))
    rng = np.random.default_rng(seed)
    draws = np.empty(reps, np.float64)
    for start in range(0, reps, 10):
        n = min(10, reps - start)
        count = rng.poisson(1.0, size=(n, len(unique))).astype(np.float64)
        draws[start:start + n] = (count @ cluster_d) / np.maximum(count @ cluster_w, EPS)
    return {
        "replicates": reps,
        "seed": seed,
        "CI95_Delta_MSE": np.quantile(draws, [0.025, 0.975]).tolist(),
        "P_Delta_MSE_lt_0": float(np.mean(draws < 0)),
        "draw_mean_Delta_MSE": float(draws.mean()),
    }


def basis_forward(name: str, raw_by_fold: dict[str, np.ndarray], residual: dict[str, np.ndarray],
                  bases: dict[str, np.ndarray], uids: dict[str, np.ndarray],
                  seed: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Project a fixed target-free basis and fit its coefficients on earlier folds."""
    U: dict[str, np.ndarray] = {}
    G: dict[str, np.ndarray] = {}
    b: dict[str, np.ndarray] = {}
    optimistic: dict[str, float] = {}
    for fold in FOLDS:
        U[fold] = e80.project_out_matrix(raw_by_fold[fold], bases[fold])
        G[fold] = U[fold].T @ U[fold] / len(U[fold])
        b[fold] = U[fold].T @ residual[fold] / len(U[fold])
        optimistic[fold] = e80.gain_from_design(residual[fold], U[fold])["gain"]

    corrections: dict[str, np.ndarray] = {}
    deltas: dict[str, np.ndarray] = {}
    fold_rows = []
    past_G: list[np.ndarray] = []
    past_b: list[np.ndarray] = []
    past_w: list[float] = []
    for k, fold in enumerate(FOLDS):
        if k == 0:
            beta = np.zeros(U[fold].shape[1], np.float64)
        else:
            gp = sum(w * x for w, x in zip(past_w, past_G))
            bp = sum(w * x for w, x in zip(past_w, past_b))
            beta = np.linalg.pinv(gp, rcond=1e-8) @ bp
        q = U[fold] @ beta
        d = (residual[fold] - q) ** 2 - residual[fold] ** 2
        corrections[fold] = q
        deltas[fold] = d
        fold_rows.append({
            "cutoff": fold, "Delta_MSE": float(d.mean()),
            "rho": safe_corr(q, residual[fold]) if np.any(q) else 0.0,
            "optimistic_gain": optimistic[fold], "rank": int(np.linalg.matrix_rank(G[fold])),
            "correction_rms": e80.rms(q), "coefficient_l2": float(np.linalg.norm(beta)),
        })
        past_G.append(G[fold])
        past_b.append(b[fold])
        past_w.append(FOLD_WEIGHT[fold])

    point = weighted_row_mean({r["cutoff"]: r["Delta_MSE"] for r in fold_rows})
    boot = cluster_bootstrap(deltas, uids, BOOTSTRAP_REPS, seed)
    latest = fold_rows[-1]
    result = {
        "candidate": name,
        "protocol": "earlier-clean-fold coefficients on fixed target-free basis (EXP080-comparable)",
        "columns": int(next(iter(U.values())).shape[1]),
        "folds": fold_rows,
        "weighted_Delta_MSE": point,
        "strict_forward_headroom": max(0.0, -point),
        "optimistic_headroom": weighted_row_mean(optimistic),
        "latest_clean_post_projection_rho": latest["rho"],
        "positive_sign_folds": int(sum(r["Delta_MSE"] < 0 for r in fold_rows)),
        "bootstrap": boot,
    }
    result["passes_gate"] = bool(
        (result["latest_clean_post_projection_rho"] >= 0.020 or point <= -0.0010)
        and boot["P_Delta_MSE_lt_0"] >= 0.95
        and result["positive_sign_folds"] >= 3
    )
    return result, U


def vector_forward(name: str, raw_by_fold: dict[str, np.ndarray], residual: dict[str, np.ndarray],
                   bases: dict[str, np.ndarray], uids: dict[str, np.ndarray], seed: int,
                   scalar_calibration: bool = True) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Project one prediction vector, optionally calibrating a scalar on earlier folds."""
    U = {f: e80.project_out_matrix(raw_by_fold[f], bases[f])[:, 0] for f in FOLDS}
    corrections: dict[str, np.ndarray] = {}
    deltas: dict[str, np.ndarray] = {}
    fold_rows = []
    past_g: list[float] = []
    past_b: list[float] = []
    past_w: list[float] = []
    for k, fold in enumerate(FOLDS):
        if scalar_calibration:
            if k == 0:
                alpha = 0.0
            else:
                alpha = sum(w * x for w, x in zip(past_w, past_b)) / max(
                    sum(w * x for w, x in zip(past_w, past_g)), EPS
                )
        else:
            alpha = 1.0
        q = alpha * U[fold]
        d = (residual[fold] - q) ** 2 - residual[fold] ** 2
        corrections[fold] = q
        deltas[fold] = d
        raw_rho = safe_corr(U[fold], residual[fold])
        g = float(np.mean(U[fold] ** 2))
        bb = float(np.mean(U[fold] * residual[fold]))
        fold_rows.append({
            "cutoff": fold, "Delta_MSE": float(d.mean()),
            "rho": safe_corr(q, residual[fold]) if np.any(q) else 0.0,
            "raw_post_projection_rho": raw_rho, "alpha": alpha,
            "raw_prediction_rms": e80.rms(raw_by_fold[fold]),
            "post_projection_rms": e80.rms(U[fold]),
            "optimistic_scalar_gain": max(0.0, bb * bb / max(g, EPS)),
        })
        past_g.append(g)
        past_b.append(bb)
        past_w.append(FOLD_WEIGHT[fold])
    point = weighted_row_mean({r["cutoff"]: r["Delta_MSE"] for r in fold_rows})
    boot = cluster_bootstrap(deltas, uids, BOOTSTRAP_REPS, seed)
    result = {
        "candidate": name,
        "protocol": "post-projection scalar from earlier clean folds" if scalar_calibration else "unit direct residual prediction",
        "folds": fold_rows,
        "weighted_Delta_MSE": point,
        "strict_forward_headroom": max(0.0, -point),
        "optimistic_headroom": weighted_row_mean({
            r["cutoff"]: r["optimistic_scalar_gain"] for r in fold_rows
        }),
        "latest_clean_post_projection_rho": fold_rows[-1]["rho"],
        "latest_raw_post_projection_rho": fold_rows[-1]["raw_post_projection_rho"],
        "positive_sign_folds": int(sum(r["Delta_MSE"] < 0 for r in fold_rows)),
        "bootstrap": boot,
    }
    result["passes_gate"] = bool(
        (result["latest_clean_post_projection_rho"] >= 0.020 or point <= -0.0010)
        and boot["P_Delta_MSE_lt_0"] >= 0.95
        and result["positive_sign_folds"] >= 3
    )
    return result, U


def fit_regressor(X: np.ndarray, y: np.ndarray, config: dict[str, Any], seed: int):
    model = lgb.LGBMRegressor(
        objective="regression_l2", random_state=seed, n_jobs=-1, verbosity=-1,
        deterministic=True, force_col_wise=True, **config,
    )
    model.fit(X, y)
    return model


def fit_classifier(X: np.ndarray, y: np.ndarray, seed: int):
    model = lgb.LGBMClassifier(
        objective="binary", n_estimators=140, learning_rate=0.035, max_depth=3,
        num_leaves=8, min_child_samples=1500, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.70, reg_alpha=3.0, reg_lambda=30.0, max_bin=127,
        random_state=seed, n_jobs=-1, verbosity=-1, deterministic=True,
        force_col_wise=True,
    )
    model.fit(X, y)
    return model


def user_crossfit_regression(X: np.ndarray, y: np.ndarray, uid: np.ndarray,
                             config: dict[str, Any], seed: int) -> np.ndarray:
    side = ((uid * 2654435761 + 101) & 1).astype(np.int8)
    pred = np.empty(len(y), np.float32)
    for hold in (0, 1):
        tr = side != hold
        va = side == hold
        model = fit_regressor(X[tr], y[tr], config, seed + hold)
        pred[va] = model.predict(X[va]).astype(np.float32)
    return pred


def user_crossfit_tail(X: np.ndarray, residual: np.ndarray, uid: np.ndarray,
                       pct: int, seed: int) -> tuple[np.ndarray, np.ndarray, float]:
    side = ((uid * 2654435761 + 101) & 1).astype(np.int8)
    pred = np.empty(len(residual), np.float32)
    truth = np.zeros(len(residual), np.int8)
    full_threshold = float(np.quantile(np.abs(residual), 1.0 - pct / 100.0))
    truth[np.abs(residual) >= full_threshold] = 1
    for hold in (0, 1):
        tr = side != hold
        va = side == hold
        threshold = float(np.quantile(np.abs(residual[tr]), 1.0 - pct / 100.0))
        ytr = (np.abs(residual[tr]) >= threshold).astype(np.int8)
        model = fit_classifier(X[tr], ytr, seed + hold)
        pred[va] = model.predict_proba(X[va])[:, 1].astype(np.float32)
    return pred, truth, float(roc_auc_score(truth, pred))


def disagreement_features(Z: np.ndarray, names: list[str], z_current: np.ndarray,
                          structural: np.ndarray, structural_names: list[str]) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    family = np.asarray([
        "SEQ" if n.startswith("SEQ-") else "ETX" if n.startswith("ETX-") else
        "TAB" if n.startswith(("S1-", "RIDGE", "HOLIDAY", "MHZ", "S04-", "GAP-", "SAMPLE-"))
        else "OTHER" for n in names
    ])
    fam_mean = {f: Z[:, family == f].mean(axis=1) for f in ["SEQ", "ETX", "TAB", "OTHER"]}
    q10 = np.quantile(Z, 0.10, axis=1)
    q90 = np.quantile(Z, 0.90, axis=1)
    median = np.median(Z, axis=1)
    mean = Z.mean(axis=1)
    std = Z.std(axis=1)
    mad = np.median(np.abs(Z - median[:, None]), axis=1)
    model_q10 = np.quantile(Z, 0.10, axis=0)
    model_q90 = np.quantile(Z, 0.90, axis=0)
    frac_low = (Z <= model_q10[None, :]).mean(axis=1)
    frac_high = (Z >= model_q90[None, :]).mean(axis=1)
    dist = Z[:, names.index("S1-DIST")]
    cols = [
        mean, std, q90 - q10, mad,
        fam_mean["SEQ"] - fam_mean["TAB"],
        fam_mean["ETX"] - fam_mean["TAB"],
        dist - fam_mean["TAB"], fam_mean["OTHER"] - fam_mean["TAB"],
        np.abs(rank01(fam_mean["SEQ"]) - rank01(fam_mean["TAB"])),
        np.abs(rank01(fam_mean["ETX"]) - rank01(fam_mean["TAB"])),
        frac_high, frac_low,
        np.max(Z, axis=1) - mean, mean - np.min(Z, axis=1),
    ]
    out_names = [
        "pred_mean", "pred_std", "robust_spread_q90_q10", "pred_mad",
        "seq_vs_tab", "etx_vs_tab", "dist_vs_tab", "other_vs_tab",
        "rank_seq_tab_abs", "rank_etx_tab_abs", "fraction_unusually_high",
        "fraction_unusually_low", "max_minus_mean", "mean_minus_min",
    ]
    extra = {"family_" + k.lower(): v for k, v in fam_mean.items()}
    extra.update({"dist": dist, "z_current": z_current})
    return np.column_stack(cols).astype(np.float32), out_names, extra


def interaction_features(dis: np.ndarray, dis_names: list[str], extra: dict[str, np.ndarray],
                         structural: np.ndarray, structural_names: list[str]) -> tuple[np.ndarray, list[str]]:
    d = {n: dis[:, i].astype(np.float64) for i, n in enumerate(dis_names)}
    s = {n: structural[:, i].astype(np.float64) for i, n in enumerate(structural_names)}
    z = extra["z_current"].astype(np.float64)
    pact = s["ridge_p_active"]
    count = s["ridge_log_purchase_days"]
    value = s["ridge_log_cond_value"]
    seq, etx, tab, other = [extra[f"family_{x}"].astype(np.float64) for x in ["seq", "etx", "tab", "other"]]
    cols = [
        z * pact, z * count, z * value,
        seq * tab, etx * tab, other * tab,
        (seq - tab) * pact, (etx - tab) * pact,
        (seq - tab) * count, (etx - tab) * count,
        d["pred_std"] * z, d["pred_std"] * pact,
        d["seq_vs_tab"] * d["etx_vs_tab"],
        rank01(z) * rank01(pact), rank01(z) * rank01(count),
        rank01(d["pred_std"]) * rank01(pact),
        (seq - tab) ** 2, (etx - tab) ** 2,
    ]
    names = [
        "z_x_pactive", "z_x_count", "z_x_value", "seq_x_tab", "etx_x_tab", "other_x_tab",
        "seqtab_x_pactive", "etxtab_x_pactive", "seqtab_x_count", "etxtab_x_count",
        "std_x_z", "std_x_pactive", "seqtab_x_etxtab", "rank_z_x_rank_pactive",
        "rank_z_x_rank_count", "rank_std_x_rank_pactive", "seqtab_squared", "etxtab_squared",
    ]
    return np.column_stack(cols).astype(np.float32), names


def feature_importance_frame(models: list[Any], names: list[str], candidate: str) -> pd.DataFrame:
    gains = np.vstack([m.booster_.feature_importance(importance_type="gain") for m in models])
    mean = gains.mean(axis=0)
    total = max(mean.sum(), EPS)
    order = np.argsort(mean)[::-1]
    return pd.DataFrame({
        "candidate": candidate,
        "feature": np.asarray(names)[order],
        "mean_gain": mean[order],
        "gain_share": mean[order] / total,
    })


def main() -> None:
    t0 = time.time()
    exp077 = load_module(EXP077 / "run_exp077.py", "exp077_for_exp081")
    canon = pd.read_parquet(EXP075 / "clean_forward_predictions.parquet")
    canon["cutoff"] = canon.cutoff.astype(str)
    work = np.load(EXP080 / "oracle_working_arrays.npz", allow_pickle=True)
    obs = pd.read_parquet(EXP080 / "observable_predictions.parquet")
    obs["cutoff"] = obs.cutoff.astype(str)
    seeds = np.load(EXP080 / "observable_seed_arrays.npz", allow_pickle=False)
    Z, bank_audit = exp077.load_reference_bank(canon)
    names = list(exp077.REFERENCE_BANK)

    uid_all = work["user_id"].astype(np.int64)
    cut_all = work["cutoff"].astype(str)
    residual_all = work["residual_current"].astype(np.float64)
    z_current_all = work["z_current"].astype(np.float64)
    z_match_all = work["z_match"].astype(np.float64)
    dpost_all = work["d_exp075_postspan"].astype(np.float64)
    y_all = work["target_log"].astype(np.float64)

    # Independent EXP080 arithmetic reproduction.
    required = e80.CURRENT_RMSLE ** 2 - e80.TARGET_RMSLE ** 2
    gap_repro = {
        "current_RMSLE": e80.CURRENT_RMSLE,
        "target_RMSLE": e80.TARGET_RMSLE,
        "required_Delta_MSE": required,
        "required_independent_rho": math.sqrt(required) / e80.CURRENT_RMSLE,
        "reported_required_Delta_MSE": json.loads((EXP080 / "gap_math.json").read_text())["required_Delta_MSE_gain"],
    }

    masks = {f: cut_all == f for f in FOLDS}
    uid = {f: uid_all[masks[f]] for f in FOLDS}
    residual = {f: residual_all[masks[f]] for f in FOLDS}
    y = {f: y_all[masks[f]] for f in FOLDS}
    z_current = {f: z_current_all[masks[f]] for f in FOLDS}
    Zf = {f: Z[masks[f]] for f in FOLDS}
    deploy_idx = [names.index(n) for n in exp077.DEPLOY_BANK]
    full_bases = {
        f: np.column_stack([
            np.ones(masks[f].sum()), z_current_all[masks[f]], z_match_all[masks[f]],
            Z[masks[f]], dpost_all[masks[f]],
        ]) for f in FOLDS
    }
    deploy_bases = {
        f: np.column_stack([
            np.ones(masks[f].sum()), z_current_all[masks[f]], z_match_all[masks[f]],
            Z[masks[f]][:, deploy_idx], dpost_all[masks[f]],
        ]) for f in FOLDS
    }
    current_bases = {f: np.column_stack([np.ones(masks[f].sum()), z_current[f]]) for f in FOLDS}

    structural_names = [
        "ridge_p_active", "ridge_log_purchase_days", "ridge_log_event_days",
        "ridge_log_order_items", "ridge_log_cond_value", "dist_p_act",
        "block4_q_event", "btyd_p_act", "btyd_expected_count",
    ]
    state_names = list(e80obs.FEATURES)
    selected_state = [
        "w30_days_present", "w30_days_buy", "w30_searches", "w30_carts",
        "w30_orders", "w30_gmv", "w30_cat_gmv_share", "w90_days_buy",
        "w90_gmv", "rec_any", "rec_buy", "tenure_frac",
    ]

    fold_data: dict[str, dict[str, Any]] = {}
    disagreement_detail = []
    for f in FOLDS:
        ids = uid[f]
        Xstate, finite = e80obs.load_feature_matrix(f, ids)
        op = obs[obs.cutoff == f].drop(columns="cutoff")
        op = align_frame(op, ids)
        tag = f.replace("-", "")
        struct = np.column_stack([
            op[n].to_numpy(np.float32) for n in structural_names[:5]
        ] + [
            seeds[f"{tag}__{n}"].astype(np.float32) for n in structural_names[5:]
        ])
        dis, dis_names, extra = disagreement_features(Zf[f], names, z_current[f], struct, structural_names)
        inter, inter_names = interaction_features(dis, dis_names, extra, struct, structural_names)
        state_idx = [state_names.index(n) for n in selected_state]
        cluster_state = Xstate[:, state_idx].astype(np.float32)

        cohort_cols = [
            rank01(z_current[f]), rank01(extra["family_seq"]), rank01(extra["family_etx"]),
            rank01(extra["family_tab"]), rank01(dis[:, dis_names.index("pred_std")]),
            rank01(struct[:, structural_names.index("ridge_p_active")]),
            rank01(struct[:, structural_names.index("ridge_log_purchase_days")]),
            rank01(Xstate[:, state_names.index("rec_buy")]),
        ]
        cohort_names = [
            "cohort_rank_z", "cohort_rank_seq", "cohort_rank_etx", "cohort_rank_tab",
            "cohort_rank_disagreement_std", "cohort_rank_pactive", "cohort_rank_count",
            "cohort_rank_recency",
        ]
        km_density = MiniBatchKMeans(
            n_clusters=32, batch_size=8192, n_init=3, max_iter=100, random_state=20260828
        ).fit(cluster_state)
        density_label = km_density.predict(cluster_state)
        density_dist = km_density.transform(cluster_state).min(axis=1)
        density_size = np.bincount(density_label, minlength=32)[density_label] / len(density_label)
        cohort_cols += [np.log1p(density_dist), np.log(np.maximum(density_size, EPS))]
        cohort_names += ["cohort_density_log_distance", "cohort_cluster_log_share"]
        cohort = np.column_stack(cohort_cols).astype(np.float32)

        for j, n in enumerate(dis_names):
            raw_rho = safe_corr(dis[:, j], residual[f])
            post = e80.project_out_matrix(dis[:, j], full_bases[f])[:, 0]
            disagreement_detail.append({
                "cutoff": f, "feature": n, "rho_raw": raw_rho,
                "rho_after_full_span": safe_corr(post, residual[f]),
            })
        fold_data[f] = {
            "Xstate": Xstate, "struct": struct, "dis": dis, "inter": inter,
            "cohort": cohort, "cluster_state": cluster_state, "finite": finite,
            "dis_names": dis_names, "inter_names": inter_names, "cohort_names": cohort_names,
        }

    # Fixed target-free bases: direct answer to projection/nonlinear/disagreement claims.
    candidates: dict[str, Any] = {}
    interaction_raw = {f: fold_data[f]["inter"] for f in FOLDS}
    disagreement_raw = {f: fold_data[f]["dis"] for f in FOLDS}
    transductive_raw = {f: fold_data[f]["cohort"] for f in FOLDS}
    candidates["fixed_interactions_full_span"], _ = basis_forward(
        "fixed_interactions_full_span", interaction_raw, residual, full_bases, uid, BOOTSTRAP_SEED + 1
    )
    candidates["fixed_interactions_deployable_span"], _ = basis_forward(
        "fixed_interactions_deployable_span", interaction_raw, residual, deploy_bases, uid, BOOTSTRAP_SEED + 2
    )
    candidates["fixed_interactions_current_only"], _ = basis_forward(
        "fixed_interactions_current_only", interaction_raw, residual, current_bases, uid, BOOTSTRAP_SEED + 3
    )
    candidates["disagreement_full_span"], _ = basis_forward(
        "disagreement_full_span", disagreement_raw, residual, full_bases, uid, BOOTSTRAP_SEED + 4
    )
    candidates["transductive_rank_density_full_span"], _ = basis_forward(
        "transductive_rank_density_full_span", transductive_raw, residual, full_bases, uid, BOOTSTRAP_SEED + 5
    )

    # Stable target-free latent state (K=4 fixed, not selected by validation score).
    first = FOLDS[0]
    km4 = MiniBatchKMeans(n_clusters=4, batch_size=8192, n_init=10, max_iter=150,
                          random_state=20260828).fit(fold_data[first]["cluster_state"])
    cluster4: dict[str, np.ndarray] = {}
    mixture_raw: dict[str, np.ndarray] = {}
    mixture_rows = []
    family_groups = {
        "SEQ": [i for i, n in enumerate(names) if n.startswith("SEQ-")],
        "ETX": [i for i, n in enumerate(names) if n.startswith("ETX-")],
        "TAB": [i for i, n in enumerate(names) if n.startswith(("S1-", "RIDGE", "HOLIDAY", "MHZ", "S04-", "GAP-", "SAMPLE-"))],
        "OTHER": [i for i, n in enumerate(names) if not n.startswith(("SEQ-", "ETX-", "S1-", "RIDGE", "HOLIDAY", "MHZ", "S04-", "GAP-", "SAMPLE-"))],
    }
    for f in FOLDS:
        lab = km4.predict(fold_data[f]["cluster_state"])
        cluster4[f] = lab
        mixture_raw[f] = np.column_stack([(lab == k).astype(np.float32) for k in range(1, 4)])
        fam = {k: Zf[f][:, idx].mean(axis=1) for k, idx in family_groups.items()}
        for k in range(4):
            m = lab == k
            family_mse = {n: float(np.mean((y[f][m] - q[m]) ** 2)) for n, q in fam.items()}
            mixture_rows.append({
                "cutoff": f, "state": k, "share": float(m.mean()),
                "signed_residual": float(residual[f][m].mean()),
                "MSE_contribution": float(np.sum(residual[f][m] ** 2) / np.sum(residual[f] ** 2)),
                "mean_disagreement_std": float(fold_data[f]["dis"][m, 1].mean()),
                "best_production_family": min(family_mse, key=family_mse.get),
                **{f"MSE_{n}": v for n, v in family_mse.items()},
            })
    candidates["latent_state_k4_full_span"], _ = basis_forward(
        "latent_state_k4_full_span", mixture_raw, residual, full_bases, uid, BOOTSTRAP_SEED + 6
    )

    # Relational/cohort prototype: 128 behavioral prototypes, never user_id memorization.
    km128 = MiniBatchKMeans(n_clusters=128, batch_size=8192, n_init=3, max_iter=150,
                            random_state=20260829).fit(fold_data[first]["cluster_state"])
    proto_lab = {f: km128.predict(fold_data[f]["cluster_state"]) for f in FOLDS}
    proto_pred: dict[str, np.ndarray] = {}
    for k, f in enumerate(FOLDS):
        if k == 0:
            proto_pred[f] = np.zeros(len(uid[f]), np.float64)
            continue
        sums = np.zeros(128, np.float64)
        counts = np.zeros(128, np.float64)
        for prior in FOLDS[:k]:
            sums += FOLD_WEIGHT[prior] * np.bincount(proto_lab[prior], weights=residual[prior], minlength=128)
            counts += FOLD_WEIGHT[prior] * np.bincount(proto_lab[prior], minlength=128)
        means = sums / np.maximum(counts, 1.0)
        proto_pred[f] = means[proto_lab[f]]
    candidates["behavioral_prototype_k128_full_span"], _ = vector_forward(
        "behavioral_prototype_k128_full_span", proto_pred, residual, full_bases, uid,
        BOOTSTRAP_SEED + 7, scalar_calibration=False,
    )
    purged_proto_latest = np.bincount(proto_lab[first], weights=residual[first], minlength=128) / np.maximum(
        np.bincount(proto_lab[first], minlength=128), 1
    )
    purged_proto_q = purged_proto_latest[proto_lab[FOLDS[-1]]]
    purged_proto_u = e80.project_out_matrix(purged_proto_q, full_bases[FOLDS[-1]])[:, 0]
    purged_proto_delta = (residual[FOLDS[-1]] - purged_proto_u) ** 2 - residual[FOLDS[-1]] ** 2
    purged_proto = {
        "train_cutoff": first, "train_target_end": str(np.datetime64(first) + np.timedelta64(30, "D")),
        "validation_cutoff": FOLDS[-1], "label_available_at_validation": True,
        "rho": safe_corr(purged_proto_u, residual[FOLDS[-1]]),
        "Delta_MSE": float(purged_proto_delta.mean()), "post_projection_rms": e80.rms(purged_proto_u),
    }

    # Full allowed feature space for the diagnostic learner.
    feature_names = [
        *names, "z_current", *structural_names, *state_names,
        *fold_data[first]["dis_names"], *fold_data[first]["inter_names"],
        *fold_data[first]["cohort_names"],
    ]
    X: dict[str, np.ndarray] = {}
    for f in FOLDS:
        X[f] = np.column_stack([
            Zf[f], z_current[f], fold_data[f]["struct"], fold_data[f]["Xstate"],
            fold_data[f]["dis"], fold_data[f]["inter"], fold_data[f]["cohort"],
        ]).astype(np.float32)
        if X[f].shape[1] != len(feature_names) or not np.isfinite(X[f]).all():
            raise AssertionError(f"diagnostic feature matrix failed: {f}")

    crossfit_predictions: dict[str, dict[str, np.ndarray]] = {cfg: {} for cfg in LGBM_CONFIGS}
    for ci, (cfg_name, config) in enumerate(LGBM_CONFIGS.items()):
        for fi, f in enumerate(FOLDS):
            crossfit_predictions[cfg_name][f] = user_crossfit_regression(
                X[f], residual[f], uid[f], config, 20260900 + 20 * ci + fi
            )
        candidates[f"lgbm_{cfg_name}_user_crossfit_full_span"], _ = vector_forward(
            f"lgbm_{cfg_name}_user_crossfit_full_span", crossfit_predictions[cfg_name],
            residual, full_bases, uid, BOOTSTRAP_SEED + 20 + ci, scalar_calibration=True,
        )

    ensemble_cf = {f: 0.5 * (crossfit_predictions["A_depth3"][f] + crossfit_predictions["B_depth5"][f]) for f in FOLDS}
    candidates["lgbm_ensemble_user_crossfit_full_span"], _ = vector_forward(
        "lgbm_ensemble_user_crossfit_full_span", ensemble_cf, residual, full_bases, uid,
        BOOTSTRAP_SEED + 22, scalar_calibration=True,
    )

    # Ordered-forward nonlinear learner: prior canonical folds only.  This is
    # compared to EXP080 but explicitly audited for label-availability overlap.
    ordered: dict[str, np.ndarray] = {}
    ordered_audit = []
    for k, f in enumerate(FOLDS):
        if k == 0:
            ordered[f] = np.zeros(len(uid[f]), np.float32)
            ordered_audit.append({"cutoff": f, "train_folds": [], "target_available": True})
            continue
        train_folds = FOLDS[:k]
        Xtr = np.concatenate([X[p] for p in train_folds])
        ytr = np.concatenate([residual[p] for p in train_folds])
        model = fit_regressor(Xtr, ytr, LGBM_CONFIGS["A_depth3"], 20261000 + k)
        ordered[f] = model.predict(X[f]).astype(np.float32)
        max_end = max(np.datetime64(p) + np.timedelta64(30, "D") for p in train_folds)
        ordered_audit.append({
            "cutoff": f, "train_folds": train_folds, "max_train_target_end": str(max_end),
            "target_available": bool(max_end <= np.datetime64(f)),
        })
    candidates["lgbm_A_ordered_previous_folds_full_span"], _ = vector_forward(
        "lgbm_A_ordered_previous_folds_full_span", ordered, residual, full_bases, uid,
        BOOTSTRAP_SEED + 23, scalar_calibration=False,
    )

    # Truly purged nonlinear latest-fold check: only 2025-09-04 labels are
    # available by 2025-10-16 among the canonical four folds.
    purge_model = fit_regressor(X[first], residual[first], LGBM_CONFIGS["A_depth3"], 20261099)
    purge_raw = purge_model.predict(X[FOLDS[-1]])
    purge_u = e80.project_out_matrix(purge_raw, full_bases[FOLDS[-1]])[:, 0]
    purge_delta = (residual[FOLDS[-1]] - purge_u) ** 2 - residual[FOLDS[-1]] ** 2
    purged_lgbm = {
        "train_cutoff": first, "train_target_end": str(np.datetime64(first) + np.timedelta64(30, "D")),
        "validation_cutoff": FOLDS[-1], "label_available_at_validation": True,
        "rho": safe_corr(purge_u, residual[FOLDS[-1]]), "Delta_MSE": float(purge_delta.mean()),
        "raw_prediction_rms": e80.rms(purge_raw), "post_projection_rms": e80.rms(purge_u),
    }

    # Tail precursor and routed specialist.  The cross-fitted residual learner
    # supplies the signed correction; the classifier only routes its amplitude.
    tail_rows = []
    for ti, pct in enumerate([5, 10, 20]):
        probs: dict[str, np.ndarray] = {}
        for fi, f in enumerate(FOLDS):
            p, truth, auc = user_crossfit_tail(X[f], residual[f], uid[f], pct, 20262000 + ti * 20 + fi)
            probs[f] = p
            tail_rows.append({
                "cutoff": f, "top_abs_residual_pct": pct, "AUC": auc,
                "mean_predicted_probability": float(p.mean()),
                "true_share": float(truth.mean()),
            })
        routed = {f: ensemble_cf[f] * probs[f] for f in FOLDS}
        result, _ = vector_forward(
            f"tail_routed_{pct}pct_full_span", routed, residual, full_bases, uid,
            BOOTSTRAP_SEED + 30 + ti, scalar_calibration=True,
        )
        candidates[f"tail_routed_{pct}pct_full_span"] = result

    # Reproduce EXP080 38-column observable basis from its primary predictions.
    repro_raw: dict[str, np.ndarray] = {}
    segment_names_ref = None
    for f in FOLDS:
        ids = uid[f]
        op = align_frame(obs[obs.cutoff == f].drop(columns="cutoff"), ids)
        tag = f.replace("-", "")
        existing = {
            "dist_p_act": seeds[f"{tag}__dist_p_act"].astype(np.float64),
            "block4_q_event": seeds[f"{tag}__block4_q_event"].astype(np.float64),
            "btyd_p_act": seeds[f"{tag}__btyd_p_act"].astype(np.float64),
            "btyd_expected_count": seeds[f"{tag}__btyd_expected_count"].astype(np.float64),
        }
        activity = np.column_stack([
            op.ridge_p_active, existing["dist_p_act"], existing["block4_q_event"], existing["btyd_p_act"]
        ])
        count = np.column_stack([
            op.ridge_log_purchase_days, op.ridge_log_event_days, op.ridge_log_order_items,
            existing["btyd_expected_count"],
        ])
        seg, seg_names = e80obs.stable_segment_design(f, ids, z_current[f], existing["dist_p_act"])
        if segment_names_ref is None:
            segment_names_ref = seg_names
        if seg_names != segment_names_ref:
            raise AssertionError("segment drift")
        repro_raw[f] = np.column_stack([activity, count, op.ridge_log_cond_value.to_numpy()[:, None], seg])
    exp080_repro, _ = basis_forward(
        "EXP080_joint_38_reproduction", repro_raw, residual, full_bases, uid, 20260838
    )
    reported = json.loads((EXP080 / "attainability.json").read_text(encoding="utf-8"))
    exp080_repro["reported"] = reported
    exp080_repro["absolute_differences"] = {
        "optimistic_headroom": abs(exp080_repro["optimistic_headroom"] - reported["observable_joint_optimal_headroom"]),
        "strict_forward_point": abs(exp080_repro["strict_forward_headroom"] - reported["joint_nested_forward_headroom_point"]),
        "robust_lower": abs(max(0.0, -exp080_repro["bootstrap"]["CI95_Delta_MSE"][1]) - reported["robust_forward_headroom_95pct_lower_bound"]),
    }

    # Per-interaction raw/post-projection measurements and small-basis headroom.
    interaction_detail = []
    for f in FOLDS:
        for j, n in enumerate(fold_data[f]["inter_names"]):
            raw = fold_data[f]["inter"][:, j]
            post = e80.project_out_matrix(raw, full_bases[f])[:, 0]
            interaction_detail.append({
                "cutoff": f, "feature": n, "rho_raw": safe_corr(raw, residual[f]),
                "rho_after_full_span": safe_corr(post, residual[f]),
                "single_feature_optimal_gain": e80.gain_from_design(residual[f], post)["gain"],
            })

    # TEST-cohort disagreement shift for the exact 16 deployable counterparts.
    sample = pd.read_csv(exp077.SAMPLE_PATH)
    sample_uid = sample.user_id.to_numpy(np.int64)
    Ztest = np.column_stack([exp077.load_test_component(n, sample_uid) for n in exp077.DEPLOY_BANK])
    zlatest = Zf[FOLDS[-1]][:, deploy_idx]
    test_shift_rows = []
    for metric, hist, test in [
        ("mean", zlatest.mean(axis=1), Ztest.mean(axis=1)),
        ("std", zlatest.std(axis=1), Ztest.std(axis=1)),
        ("q90_q10", np.quantile(zlatest, .9, axis=1) - np.quantile(zlatest, .1, axis=1),
         np.quantile(Ztest, .9, axis=1) - np.quantile(Ztest, .1, axis=1)),
    ]:
        ks = ks_2samp(hist, test)
        test_shift_rows.append({
            "metric": metric, "historical_latest_mean": float(hist.mean()),
            "historical_latest_std": float(hist.std()), "TEST_mean": float(test.mean()),
            "TEST_std": float(test.std()), "KS_statistic": float(ks.statistic), "KS_pvalue": float(ks.pvalue),
        })

    # Primary metric summary and revised headroom use only gate-valid mechanisms.
    summary_rows = []
    for name, c in candidates.items():
        summary_rows.append({
            "candidate": name, "protocol": c["protocol"],
            "weighted_Delta_MSE": c["weighted_Delta_MSE"],
            "strict_forward_headroom": c["strict_forward_headroom"],
            "optimistic_headroom": c["optimistic_headroom"],
            "latest_clean_post_projection_rho": c["latest_clean_post_projection_rho"],
            "positive_sign_folds": c["positive_sign_folds"],
            "P_Delta_MSE_lt_0": c["bootstrap"]["P_Delta_MSE_lt_0"],
            "CI_low": c["bootstrap"]["CI95_Delta_MSE"][0],
            "CI_high": c["bootstrap"]["CI95_Delta_MSE"][1],
            "passes_gate": c["passes_gate"],
        })
    summary = pd.DataFrame(summary_rows).sort_values("weighted_Delta_MSE")
    passed = summary[summary.passes_gate]
    revised = {
        "required_Delta_MSE": required,
        "EXP080_optimistic_headroom": reported["observable_joint_optimal_headroom"],
        "EXP080_strict_forward_headroom": reported["joint_nested_forward_headroom_point"],
        "EXP080_robust_95pct_headroom": reported["robust_forward_headroom_95pct_lower_bound"],
        "new_optimistic_headroom": float(summary.optimistic_headroom.max()),
        "new_strict_forward_headroom": float(summary.strict_forward_headroom.max()),
        "new_robust_95pct_headroom": float(max(0.0, -summary.CI_high.min())),
        "gate_passing_candidates": passed.candidate.tolist(),
    }

    # Raw schema is deliberately recorded: there is no product/category/entity relation.
    raw_schema = pd.read_parquet(e80.RAW).dtypes.astype(str).to_dict()
    relational_audit = {
        "raw_rows": 30_631_006, "raw_users": 250_000, "columns": raw_schema,
        "legitimate_entity_or_category_identifier_present": False,
        "available_relational_channel": "behavioral-fingerprint cohorts only",
        "user_id_used_as_model_feature": False,
        "user_id_use": "alignment, user-disjoint cross-fitting, and clustered bootstrap only",
        "purged_prototype_latest": purged_proto,
    }

    pd.DataFrame(disagreement_detail).to_csv(HERE / "disagreement_feature_metrics.csv", index=False)
    pd.DataFrame(interaction_detail).to_csv(HERE / "interaction_feature_metrics.csv", index=False)
    pd.DataFrame(mixture_rows).to_csv(HERE / "mixture_state_metrics.csv", index=False)
    pd.DataFrame(tail_rows).to_csv(HERE / "tail_classifier_metrics.csv", index=False)
    pd.DataFrame(test_shift_rows).to_csv(HERE / "test_disagreement_shift.csv", index=False)
    summary.to_csv(HERE / "candidate_summary.csv", index=False)
    bank_audit.to_csv(HERE / "production_bank_audit.csv", index=False)
    write_json("exp080_reproduction.json", {"gap": gap_repro, "observable": exp080_repro})
    write_json("candidate_details.json", candidates)
    write_json("temporal_audit.json", {
        "EXP080_comparable_ordered": ordered_audit,
        "purged_latest_LGBM": purged_lgbm,
        "purged_latest_prototype": purged_proto,
        "note": "Adjacent 14-day fold labels are not available at the next cutoff; only 2025-09-04 is available by 2025-10-16.",
    })
    write_json("relational_schema_audit.json", relational_audit)
    write_json("revised_headroom.json", revised)
    write_json("config.json", {
        "experiment": "EXP081_ADVERSARIAL_BOUND_FALSIFICATION",
        "folds": FOLDS, "fold_weights": FOLD_WEIGHT, "bootstrap_reps": BOOTSTRAP_REPS,
        "models": LGBM_CONFIGS, "tail_percentages": [5, 10, 20],
        "mixture_states": 4, "behavioral_prototypes": 128,
        "projection_primary": "[1,z_current,z_match,40 clean OOF components,EXP075_postspan]",
        "projection_sensitivity": ["16 exact OOF/TEST deployable components", "[1,z_current] current-only"],
        "feature_count": len(feature_names), "feature_names": feature_names,
        "leaderboard_used_for_selection": False,
    })
    write_json("audit.json", {
        "runtime_seconds": time.time() - t0,
        "raw_sha256": sha256(e80.RAW),
        "canonical_oof_sha256": sha256(EXP075 / "clean_forward_predictions.parquet"),
        "EXP080_working_arrays_sha256": sha256(EXP080 / "oracle_working_arrays.npz"),
        "observable_predictions_sha256": sha256(EXP080 / "observable_predictions.parquet"),
        "rows": len(canon), "feature_rows_by_fold": {f: len(X[f]) for f in FOLDS},
        "finite_fraction_before_EXP080_fill": {f: fold_data[f]["finite"] for f in FOLDS},
        "forbidden_activity_feature_loaded": False,
        "target_or_user_id_as_feature": False,
        "TEST_target_or_leaderboard_used": False,
    })
    print(summary.to_string(index=False))
    print(json.dumps(jsonable({"revised": revised, "purged_lgbm": purged_lgbm}), indent=2))


if __name__ == "__main__":
    main()
