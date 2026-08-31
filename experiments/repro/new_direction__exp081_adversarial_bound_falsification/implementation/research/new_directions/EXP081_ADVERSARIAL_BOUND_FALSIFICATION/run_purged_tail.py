"""Purged latest-fold tail-routing supplement for EXP081.

The only canonical residual labels observable by 2025-10-16 are those from
2025-09-04 (target ends 2025-10-04).  This script trains on that fold only and
uses the fixed 2025-10-16 cohort as an unlabeled TEST analogue.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import roc_auc_score


HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parents[1]
EXP080 = RESEARCH / "new_directions" / "EXP080_ORACLE_GAP_ATTRIBUTION"
EXP077 = RESEARCH / "new_directions" / "EXP077_FORWARD_STACK"
EXP075 = RESEARCH / "new_directions" / "EXP075_OUT_OF_SPAN_RESIDUAL_SIGNALS"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXP080))

import run_falsification as rf  # noqa: E402
import run_oracle as e80  # noqa: E402
import run_observable as e80obs  # noqa: E402


FIRST = "2025-09-04"
LATEST = "2025-10-16"


def build_fold(fold: str, canon: pd.DataFrame, work: np.lib.npyio.NpzFile,
               obs: pd.DataFrame, seeds: np.lib.npyio.NpzFile,
               Z: np.ndarray, names: list[str]) -> dict:
    cut = work["cutoff"].astype(str)
    m = cut == fold
    ids = work["user_id"].astype(np.int64)[m]
    residual = work["residual_current"].astype(np.float64)[m]
    z_current = work["z_current"].astype(np.float64)[m]
    Xstate, _ = e80obs.load_feature_matrix(fold, ids)
    op = rf.align_frame(obs[obs.cutoff == fold].drop(columns="cutoff"), ids)
    structural_names = [
        "ridge_p_active", "ridge_log_purchase_days", "ridge_log_event_days",
        "ridge_log_order_items", "ridge_log_cond_value", "dist_p_act",
        "block4_q_event", "btyd_p_act", "btyd_expected_count",
    ]
    tag = fold.replace("-", "")
    struct = np.column_stack([
        op[n].to_numpy(np.float32) for n in structural_names[:5]
    ] + [seeds[f"{tag}__{n}"].astype(np.float32) for n in structural_names[5:]])
    Zf = Z[m]
    dis, dis_names, extra = rf.disagreement_features(Zf, names, z_current, struct, structural_names)
    inter, inter_names = rf.interaction_features(dis, dis_names, extra, struct, structural_names)
    state_names = list(e80obs.FEATURES)
    selected_state = [
        "w30_days_present", "w30_days_buy", "w30_searches", "w30_carts",
        "w30_orders", "w30_gmv", "w30_cat_gmv_share", "w90_days_buy",
        "w90_gmv", "rec_any", "rec_buy", "tenure_frac",
    ]
    cluster_state = Xstate[:, [state_names.index(n) for n in selected_state]]
    cohort_cols = [
        rf.rank01(z_current), rf.rank01(extra["family_seq"]), rf.rank01(extra["family_etx"]),
        rf.rank01(extra["family_tab"]), rf.rank01(dis[:, dis_names.index("pred_std")]),
        rf.rank01(struct[:, structural_names.index("ridge_p_active")]),
        rf.rank01(struct[:, structural_names.index("ridge_log_purchase_days")]),
        rf.rank01(Xstate[:, state_names.index("rec_buy")]),
    ]
    km = MiniBatchKMeans(n_clusters=32, batch_size=8192, n_init=3, max_iter=100,
                         random_state=20260828).fit(cluster_state)
    label = km.predict(cluster_state)
    dist = km.transform(cluster_state).min(axis=1)
    share = np.bincount(label, minlength=32)[label] / len(label)
    cohort_cols += [np.log1p(dist), np.log(np.maximum(share, 1e-12))]
    cohort = np.column_stack(cohort_cols).astype(np.float32)
    X = np.column_stack([Zf, z_current, struct, Xstate, dis, inter, cohort]).astype(np.float32)
    B = np.column_stack([
        np.ones(m.sum()), work["z_current"].astype(np.float64)[m],
        work["z_match"].astype(np.float64)[m], Zf,
        work["d_exp075_postspan"].astype(np.float64)[m],
    ])
    return {
        "uid": ids, "residual": residual, "X": X, "B": B,
        "dis": dis, "inter": inter, "cohort": cohort,
        "cluster_state": cluster_state,
    }


def bootstrap_latest(delta: np.ndarray, seed: int, reps: int = 2000) -> dict:
    rng = np.random.default_rng(seed)
    draws = np.empty(reps, np.float64)
    n = len(delta)
    for i in range(reps):
        # Poisson bootstrap avoids materializing a resampled row index.
        w = rng.poisson(1.0, n)
        draws[i] = float(w @ delta / max(w.sum(), 1))
    return {
        "CI95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "P_Delta_MSE_lt_0": float(np.mean(draws < 0)),
        "mean": float(draws.mean()),
    }


def main() -> None:
    exp077 = rf.load_module(EXP077 / "run_exp077.py", "exp077_for_exp081_purged")
    canon = pd.read_parquet(EXP075 / "clean_forward_predictions.parquet")
    canon["cutoff"] = canon.cutoff.astype(str)
    work = np.load(EXP080 / "oracle_working_arrays.npz", allow_pickle=True)
    obs = pd.read_parquet(EXP080 / "observable_predictions.parquet")
    obs["cutoff"] = obs.cutoff.astype(str)
    seeds = np.load(EXP080 / "observable_seed_arrays.npz", allow_pickle=False)
    Z, _ = exp077.load_reference_bank(canon)
    names = list(exp077.REFERENCE_BANK)
    train = build_fold(FIRST, canon, work, obs, seeds, Z, names)
    val = build_fold(LATEST, canon, work, obs, seeds, Z, names)

    q_train_cf = rf.user_crossfit_regression(
        train["X"], train["residual"], train["uid"], rf.LGBM_CONFIGS["A_depth3"], 20263000
    )
    reg = rf.fit_regressor(train["X"], train["residual"], rf.LGBM_CONFIGS["A_depth3"], 20263010)
    q_val_raw = reg.predict(val["X"])

    rows = []
    for pct in [5, 10, 20]:
        p_train_cf, train_truth, train_auc = rf.user_crossfit_tail(
            train["X"], train["residual"], train["uid"], pct, 20263100 + pct
        )
        threshold = float(np.quantile(np.abs(train["residual"]), 1.0 - pct / 100.0))
        ytail = (np.abs(train["residual"]) >= threshold).astype(np.int8)
        clf = rf.fit_classifier(train["X"], ytail, 20263200 + pct)
        p_val = clf.predict_proba(val["X"])[:, 1]
        val_threshold = float(np.quantile(np.abs(val["residual"]), 1.0 - pct / 100.0))
        val_truth = (np.abs(val["residual"]) >= val_threshold).astype(np.int8)

        u_train = e80.project_out_matrix(q_train_cf * p_train_cf, train["B"])[:, 0]
        alpha = float(np.mean(u_train * train["residual"]) / max(np.mean(u_train * u_train), 1e-12))
        u_val = e80.project_out_matrix(q_val_raw * p_val, val["B"])[:, 0]
        q = alpha * u_val
        delta = (val["residual"] - q) ** 2 - val["residual"] ** 2
        rows.append({
            "top_abs_residual_pct": pct,
            "train_crossfit_AUC": train_auc,
            "purged_latest_AUC": float(roc_auc_score(val_truth, p_val)),
            "train_crossfit_scalar": alpha,
            "purged_latest_post_projection_rho": rf.safe_corr(q, val["residual"]),
            "purged_latest_Delta_MSE": float(delta.mean()),
            "purged_latest_post_projection_rms": e80.rms(q),
            **{f"bootstrap_{k}": v for k, v in bootstrap_latest(delta, 20263300 + pct).items()},
        })

    # Also calibrate the ungated residual learner from honest first-fold OOF.
    u0 = e80.project_out_matrix(q_train_cf, train["B"])[:, 0]
    alpha0 = float(np.mean(u0 * train["residual"]) / max(np.mean(u0 * u0), 1e-12))
    u1 = e80.project_out_matrix(q_val_raw, val["B"])[:, 0]
    q1 = alpha0 * u1
    d1 = (val["residual"] - q1) ** 2 - val["residual"] ** 2
    audit = {
        "train_cutoff": FIRST,
        "train_target_end": str(np.datetime64(FIRST) + np.timedelta64(30, "D")),
        "validation_cutoff": LATEST,
        "label_available": True,
        "ungated": {
            "calibration": alpha0,
            "rho": rf.safe_corr(q1, val["residual"]),
            "Delta_MSE": float(d1.mean()),
            "bootstrap": bootstrap_latest(d1, 20263400),
        },
    }

    # Purged versions of the fixed bases.  Coefficients are learned only from
    # the fully observed first fold and applied unchanged to the latest fold.
    fixed_rows = []
    fixed_bases = {
        "interactions": (train["inter"], val["inter"]),
        "disagreement": (train["dis"], val["dis"]),
        "transductive_rank_density": (train["cohort"], val["cohort"]),
    }
    km4 = MiniBatchKMeans(n_clusters=4, batch_size=8192, n_init=10, max_iter=150,
                          random_state=20260828).fit(train["cluster_state"])
    lt = km4.predict(train["cluster_state"])
    lv = km4.predict(val["cluster_state"])
    fixed_bases["latent_state_k4"] = (
        np.column_stack([(lt == k).astype(np.float32) for k in range(1, 4)]),
        np.column_stack([(lv == k).astype(np.float32) for k in range(1, 4)]),
    )
    for j, (name, (d0, d1raw)) in enumerate(fixed_bases.items()):
        u0b = e80.project_out_matrix(d0, train["B"])
        u1b = e80.project_out_matrix(d1raw, val["B"])
        g = u0b.T @ u0b / len(u0b)
        b = u0b.T @ train["residual"] / len(u0b)
        beta = np.linalg.pinv(g, rcond=1e-8) @ b
        q = u1b @ beta
        delta = (val["residual"] - q) ** 2 - val["residual"] ** 2
        fixed_rows.append({
            "basis": name,
            "purged_latest_rho": rf.safe_corr(q, val["residual"]),
            "purged_latest_Delta_MSE": float(delta.mean()),
            "purged_latest_correction_rms": e80.rms(q),
            "positive": bool(delta.mean() < 0),
            **{f"bootstrap_{k}": v for k, v in bootstrap_latest(delta, 20263500 + j).items()},
        })
    pd.DataFrame(rows).to_csv(HERE / "purged_tail_metrics.csv", index=False)
    pd.DataFrame(fixed_rows).to_csv(HERE / "purged_fixed_basis_metrics.csv", index=False)
    (HERE / "purged_tail_audit.json").write_text(
        json.dumps(rf.jsonable(audit), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(fixed_rows).to_string(index=False))
    print(json.dumps(rf.jsonable(audit), indent=2))


if __name__ == "__main__":
    main()
