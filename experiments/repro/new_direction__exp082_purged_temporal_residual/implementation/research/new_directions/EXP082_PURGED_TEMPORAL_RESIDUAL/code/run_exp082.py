"""EXP082 fully-purged temporal residual validation.

The nonlinear learner configurations are frozen from EXP081.  The new-fold
feature matrix is the reproducible core-production subset of the EXP081 feature
space: five frozen production components, the production blend, six structural
predictions, all 108 RFM/state features, and the same disagreement, interaction,
and cohort-relative transforms.  Missing non-production research-bank members
are never imputed or distilled.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import itertools
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans


HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[2]
OZON = Path(r"C:\Users\Admin\Desktop\OZON-E-CUP")
COMP = EXP / "production_components"
EXP075 = ROOT / "research" / "new_directions" / "EXP075_OUT_OF_SPAN_RESIDUAL_SIGNALS"
EXP076 = ROOT / "research" / "new_directions" / "EXP076_STRONG_BASELINE_VALIDATION_CHANNEL"
EXP077 = ROOT / "research" / "new_directions" / "EXP077_FORWARD_STACK"
EXP080 = ROOT / "research" / "new_directions" / "EXP080_ORACLE_GAP_ATTRIBUTION"
EXP081 = ROOT / "research" / "new_directions" / "EXP081_ADVERSARIAL_BOUND_FALSIFICATION"

FOLDS = ("2025-07-03", "2025-08-07", "2025-09-11", "2025-10-16")
TRANSITIONS = FOLDS[1:]
FOLD_WEIGHT = dict(zip(FOLDS, (1.0, 2.0, 4.0, 8.0)))
PRODUCTION_WEIGHTS = {"cap": 0.10, "unc": 0.20, "dist": 0.25, "seq": 0.225, "etx": 0.225}
CURRENT_RMSLE = 1.646143314225527
TARGET_RMSLE = 1.6446514942
REQUIRED_MSE = CURRENT_RMSLE ** 2 - TARGET_RMSLE ** 2
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


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


e80 = load_module(EXP080 / "run_oracle.py", "exp082_e80")
sys.path.insert(0, str(EXP080))
e80obs = load_module(EXP080 / "run_observable.py", "exp082_e80obs")
rf = load_module(EXP081 / "run_falsification.py", "exp082_rf")
exp077 = load_module(EXP077 / "run_exp077.py", "exp082_exp077")


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


def corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, np.float64) - float(np.mean(x))
    y = np.asarray(y, np.float64) - float(np.mean(y))
    den = math.sqrt(float(x @ x) * float(y @ y))
    return 0.0 if den <= 1e-300 else float(x @ y / den)


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, np.float64)
    return float(np.sqrt(np.mean(x * x)))


def rank01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    out = np.empty(len(x), np.float32)
    out[order] = (np.arange(len(x), dtype=np.float32) + 0.5) / len(x)
    return out


def project_two_pass(raw: np.ndarray, basis: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    raw = np.asarray(raw, np.float64)
    centered = raw - float(np.mean(raw))
    coef1, *_ = np.linalg.lstsq(basis, centered, rcond=1e-10)
    first = centered - basis @ coef1
    coef2, *_ = np.linalg.lstsq(basis, first, rcond=1e-10)
    removed2 = basis @ coef2
    out = first - removed2
    return out, {
        "RMS_u_raw": rms(raw), "RMS_u_centered": rms(centered),
        "RMS_u_perp": rms(out),
        "perp_fraction": rms(out) / max(rms(centered), EPS),
        "second_pass_projection_error_RMS": rms(removed2),
        "second_pass_relative_error": rms(removed2) / max(rms(out), EPS),
    }


def fit_regressor(X: np.ndarray, y: np.ndarray, config: dict[str, Any], seed: int):
    model = lgb.LGBMRegressor(
        objective="regression", random_state=seed, n_jobs=-1, verbosity=-1,
        deterministic=True, force_col_wise=True, **config,
    )
    model.fit(X, y)
    return model


def user_crossfit(X: np.ndarray, y: np.ndarray, uid: np.ndarray,
                  config: dict[str, Any], seed: int) -> np.ndarray:
    side = ((uid.astype(np.uint64) * np.uint64(2654435761) + np.uint64(101)) & np.uint64(1)).astype(np.int8)
    pred = np.empty(len(y), np.float32)
    for hold in (0, 1):
        tr, va = side != hold, side == hold
        model = fit_regressor(X[tr], y[tr], config, seed + hold)
        pred[va] = model.predict(X[va]).astype(np.float32)
    return pred


def load_components(fold: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    ref_uid = ref_y = None
    columns = []
    for family in PRODUCTION_WEIGHTS:
        path = COMP / f"{family}_{fold}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        part = np.load(path, allow_pickle=False)
        uid = part["user_id"].astype(np.int64)
        y = part["target_log"].astype(np.float64)
        if ref_uid is None:
            ref_uid, ref_y = uid, y
        elif not np.array_equal(uid, ref_uid) or not np.allclose(y, ref_y, atol=1e-10, rtol=0):
            raise AssertionError(f"component alignment/target parity failed: {fold}/{family}")
        z = part["z"].astype(np.float64)
        data[family] = z
        columns.append(z)
        if family == "dist":
            data["dist_p_act"] = part["p_act"].astype(np.float64)
    data["uid"], data["target_log"] = ref_uid, ref_y
    data["components"] = np.column_stack(columns).astype(np.float32)
    data["baseline"] = sum(PRODUCTION_WEIGHTS[k] * data[k] for k in PRODUCTION_WEIGHTS)
    data["residual"] = data["target_log"] - data["baseline"]
    data["basis"] = np.column_stack([np.ones(len(ref_uid)), data["components"], data["baseline"]])
    return data


def disagreement_features(Z: np.ndarray, z_current: np.ndarray) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    cap, unc, dist, seq, etx = (Z[:, i].astype(np.float64) for i in range(5))
    tab = (cap + unc + dist) / 3.0
    other = z_current.astype(np.float64)
    q10, q90 = np.quantile(Z, 0.10, axis=1), np.quantile(Z, 0.90, axis=1)
    median, mean, std = np.median(Z, axis=1), Z.mean(axis=1), Z.std(axis=1)
    mad = np.median(np.abs(Z - median[:, None]), axis=1)
    model_q10, model_q90 = np.quantile(Z, 0.10, axis=0), np.quantile(Z, 0.90, axis=0)
    frac_low = (Z <= model_q10[None, :]).mean(axis=1)
    frac_high = (Z >= model_q90[None, :]).mean(axis=1)
    cols = [
        mean, std, q90 - q10, mad, seq - tab, etx - tab, dist - tab, other - tab,
        np.abs(rank01(seq) - rank01(tab)), np.abs(rank01(etx) - rank01(tab)),
        frac_high, frac_low, np.max(Z, axis=1) - mean, mean - np.min(Z, axis=1),
    ]
    names = [
        "pred_mean", "pred_std", "robust_spread_q90_q10", "pred_mad",
        "seq_vs_tab", "etx_vs_tab", "dist_vs_tab", "other_vs_tab",
        "rank_seq_tab_abs", "rank_etx_tab_abs", "fraction_unusually_high",
        "fraction_unusually_low", "max_minus_mean", "mean_minus_min",
    ]
    return np.column_stack(cols).astype(np.float32), names, {
        "family_seq": seq, "family_etx": etx, "family_tab": tab,
        "family_other": other, "dist": dist, "z_current": z_current,
    }


def interaction_features(dis: np.ndarray, dis_names: list[str], extra: dict[str, np.ndarray],
                         structural: np.ndarray, structural_names: list[str]) -> tuple[np.ndarray, list[str]]:
    d = {n: dis[:, i].astype(np.float64) for i, n in enumerate(dis_names)}
    s = {n: structural[:, i].astype(np.float64) for i, n in enumerate(structural_names)}
    z, pact = extra["z_current"].astype(np.float64), s["ridge_p_active"]
    count, value = s["ridge_log_purchase_days"], s["ridge_log_cond_value"]
    seq, etx, tab, other = (extra[f"family_{x}"].astype(np.float64) for x in ("seq", "etx", "tab", "other"))
    cols = [
        z * pact, z * count, z * value, seq * tab, etx * tab, other * tab,
        (seq - tab) * pact, (etx - tab) * pact, (seq - tab) * count,
        (etx - tab) * count, d["pred_std"] * z, d["pred_std"] * pact,
        d["seq_vs_tab"] * d["etx_vs_tab"], rank01(z) * rank01(pact),
        rank01(z) * rank01(count), rank01(d["pred_std"]) * rank01(pact),
        (seq - tab) ** 2, (etx - tab) ** 2,
    ]
    names = [
        "z_x_pactive", "z_x_count", "z_x_value", "seq_x_tab", "etx_x_tab", "other_x_tab",
        "seqtab_x_pactive", "etxtab_x_pactive", "seqtab_x_count", "etxtab_x_count",
        "std_x_z", "std_x_pactive", "seqtab_x_etxtab", "rank_z_x_rank_pactive",
        "rank_z_x_rank_count", "rank_std_x_rank_pactive", "seqtab_squared", "etxtab_squared",
    ]
    return np.column_stack(cols).astype(np.float32), names


def align_frame(frame: pd.DataFrame, ids: np.ndarray) -> pd.DataFrame:
    frame = frame.sort_values("user_id")
    source = frame.user_id.to_numpy(np.int64)
    pos = np.searchsorted(source, ids)
    if pos.max(initial=0) >= len(source) or not np.array_equal(source[pos], ids):
        raise AssertionError("frame alignment failed")
    return frame.iloc[pos].reset_index(drop=True)


def structural_predictions(fold: str, ids: np.ndarray,
                           uid_all: np.ndarray, panel: np.ndarray, gmv: np.ndarray,
                           saved_obs: pd.DataFrame) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    names = ["ridge_p_active", "ridge_log_purchase_days", "ridge_log_event_days",
             "ridge_log_order_items", "ridge_log_cond_value", "dist_p_act"]
    if fold == "2025-10-16":
        frame = align_frame(saved_obs[saved_obs.cutoff == fold].drop(columns="cutoff"), ids)
        pred = {name: frame[name].to_numpy(np.float64) for name in names[:5]}
        audit = {"source": "frozen EXP080 observable_predictions.parquet", "target_safe": True}
    else:
        val = e80obs.build_dataset(fold, ids, uid_all, panel, gmv)
        train_cutoffs = [e80obs.date_minus(fold, lag) for lag in e80obs.TRAIN_LAGS]
        for train_cutoff in train_cutoffs:
            if np.datetime64(train_cutoff) + np.timedelta64(30, "D") > np.datetime64(fold):
                raise AssertionError("structural training target crosses validation cutoff")
        train_sets = [e80obs.build_dataset(x, e80obs.eligible_ids(x), uid_all, panel, gmv)
                      for x in train_cutoffs]
        pred, diagnostics = e80obs.fit_ridge_predictions(train_sets, val)
        audit = {"source": "exact EXP080 fixed Ridge replay", "train_cutoffs": train_cutoffs,
                 "max_train_target_end": str(max(np.datetime64(x) + np.timedelta64(30, "D")
                                                    for x in train_cutoffs)),
                 "target_safe": True, "diagnostics": diagnostics}
    return np.column_stack([pred[name] for name in names[:5]]).astype(np.float32), names, audit


def build_fold_features(fold: str, data: dict[str, Any], uid_all: np.ndarray,
                        panel: np.ndarray, gmv: np.ndarray, saved_obs: pd.DataFrame) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    ids = data["uid"]
    Xstate, finite = e80obs.load_feature_matrix(fold, ids)
    struct5, structural_names, struct_audit = structural_predictions(
        fold, ids, uid_all, panel, gmv, saved_obs)
    struct = np.column_stack([struct5, data["dist_p_act"]]).astype(np.float32)
    dis, dis_names, extra = disagreement_features(data["components"], data["baseline"])
    inter, inter_names = interaction_features(dis, dis_names, extra, struct, structural_names)

    state_names = list(e80obs.FEATURES)
    selected_state = ["w30_days_present", "w30_days_buy", "w30_searches", "w30_carts",
                      "w30_orders", "w30_gmv", "w30_cat_gmv_share", "w90_days_buy",
                      "w90_gmv", "rec_any", "rec_buy", "tenure_frac"]
    cluster_state = Xstate[:, [state_names.index(n) for n in selected_state]]
    cohort_cols = [
        rank01(data["baseline"]), rank01(extra["family_seq"]), rank01(extra["family_etx"]),
        rank01(extra["family_tab"]), rank01(dis[:, dis_names.index("pred_std")]),
        rank01(struct[:, structural_names.index("ridge_p_active")]),
        rank01(struct[:, structural_names.index("ridge_log_purchase_days")]),
        rank01(Xstate[:, state_names.index("rec_buy")]),
    ]
    cohort_names = ["cohort_rank_z", "cohort_rank_seq", "cohort_rank_etx", "cohort_rank_tab",
                    "cohort_rank_disagreement_std", "cohort_rank_pactive", "cohort_rank_count",
                    "cohort_rank_recency"]
    km = MiniBatchKMeans(n_clusters=32, batch_size=8192, n_init=3, max_iter=100,
                         random_state=20260828).fit(cluster_state)
    label = km.predict(cluster_state)
    distance = km.transform(cluster_state).min(axis=1)
    share = np.bincount(label, minlength=32)[label] / len(label)
    cohort_cols += [np.log1p(distance), np.log(np.maximum(share, EPS))]
    cohort_names += ["cohort_density_log_distance", "cohort_cluster_log_share"]
    cohort = np.column_stack(cohort_cols).astype(np.float32)

    component_names = ["S1-E03a", "S1-E02", "S1-DIST", "SEQ-01-S42", "ETX-01-S42"]
    names = [*component_names, "z_current", *structural_names, *state_names,
             *dis_names, *inter_names, *cohort_names]
    X = np.column_stack([data["components"], data["baseline"], struct, Xstate, dis, inter, cohort]).astype(np.float32)
    if X.shape[1] != len(names) or not np.isfinite(X).all():
        raise AssertionError(f"feature matrix failed: {fold}/{X.shape}")
    return X, names, {"finite_before_fill": finite, "structural": struct_audit,
                      "feature_count": len(names), "excluded_exp081_features": 38,
                      "reason": "35 non-production research-bank predictions and 3 unavailable auxiliary structural predictions were not imputed"}


def baseline_fidelity_audit(data_latest: dict[str, Any]) -> dict[str, Any]:
    canon = pd.read_parquet(EXP075 / "clean_forward_predictions.parquet")
    canon["cutoff"] = canon.cutoff.astype(str)
    m = canon.cutoff.eq("2025-10-16").to_numpy()
    ids = canon.user_id.to_numpy(np.int64)[m]
    if not np.array_equal(ids, data_latest["uid"]):
        raise AssertionError("fidelity alignment failed")
    y = canon.target_log.to_numpy(np.float64)[m]

    def align(name: str) -> np.ndarray:
        d = np.load(OZON / "artifacts" / f"oof_{name}.npz", allow_pickle=True)
        mm = d["cutoff"].astype(str) == "2025-10-16"
        su, sv = d["user_id"][mm].astype(np.int64), d["z"][mm].astype(np.float64)
        order = np.argsort(su)
        pos = np.searchsorted(su[order], ids)
        if not np.array_equal(su[order][pos], ids):
            raise AssertionError(name)
        return sv[order][pos]

    z_single = data_latest["baseline"]
    z_avg3 = (0.10 * align("S1-E03a") + 0.20 * align("S1-E02") + 0.25 * align("S1-DIST")
              + 0.225 * align("SEQ-AVG3") + 0.225 * align("ETX-AVG3"))
    work = np.load(EXP080 / "oracle_working_arrays.npz", allow_pickle=True)
    wm = work["cutoff"].astype(str) == "2025-10-16"
    if not np.array_equal(work["user_id"][wm].astype(np.int64), ids):
        raise AssertionError("EXP080 work alignment")
    z_match = work["z_match"][wm].astype(np.float64)

    a2 = pd.read_parquet(EXP075 / "a2_clean_forward_predictions.parquet")
    a2["cutoff"] = a2.cutoff.astype(str)
    a2 = a2.set_index(["cutoff", "user_id"])
    key = pd.MultiIndex.from_arrays([np.full(len(ids), "2025-10-16"), ids])
    u_exp075 = (0.7462560852846633 * canon.loc[m, "u_raw_365"].to_numpy(np.float64)
                + 0.6466415684754089 * a2.loc[key, "u_raw_A2"].to_numpy(np.float64))

    candidates = {"EXP075_joint": u_exp075}
    # Exact saved-period EXP081 corrections are deterministically reproduced
    # from the frozen 200-column canonical feature builder.
    work2 = np.load(EXP080 / "oracle_working_arrays.npz", allow_pickle=True)
    obs = pd.read_parquet(EXP080 / "observable_predictions.parquet")
    obs["cutoff"] = obs.cutoff.astype(str)
    seeds = np.load(EXP080 / "observable_seed_arrays.npz", allow_pickle=False)
    Z, _ = exp077.load_reference_bank(canon)
    purged = load_module(EXP081 / "run_purged_tail.py", "exp082_purged_builder")
    built = purged.build_fold("2025-10-16", canon, work2, obs, seeds, Z, list(exp077.REFERENCE_BANK))
    qa = rf.user_crossfit_regression(built["X"], built["residual"], built["uid"], LGBM_CONFIGS["A_depth3"], 20260903)
    qb = rf.user_crossfit_regression(built["X"], built["residual"], built["uid"], LGBM_CONFIGS["B_depth5"], 20260923)
    candidates.update({"EXP081_A": qa, "EXP081_B": qb, "EXP081_AB_mean": 0.5 * (qa + qb)})

    # Fidelity concerns residual geometry after the same full production-span
    # removal used by EXP081, not a fresh two-column [1, baseline] projection.
    # Project each saved/reproduced diagnostic correction once through the
    # canonical full span, then compare its rho against alternative baselines.
    full_basis = np.column_stack([
        np.ones(len(ids)), work2["z_current"][wm].astype(np.float64), z_match,
        Z[m].astype(np.float64), work2["d_exp075_postspan"][wm].astype(np.float64),
    ])
    candidate_q = {name: project_two_pass(u, full_basis)[0] for name, u in candidates.items()}

    def candidate_rho(q: np.ndarray, z: np.ndarray) -> float:
        return corr(q, y - z)

    rows = []
    for name, q in candidate_q.items():
        rs, rm = candidate_rho(q, z_single), candidate_rho(q, z_match)
        rows.append({"candidate": name, "rho_rebuilt_single_seed": rs,
                     "rho_composition_matched_EXP076": rm, "absolute_difference": abs(rs - rm),
                     "passes_0_003": abs(rs - rm) <= 0.003})
    # Before authorizing costly historical extra-seed replays, enumerate frozen
    # canonical S42/S43/S44 averages and find the smallest seed composition that
    # passes both fidelity gates. This is baseline fidelity repair, not model search.
    seq_seed = {s: align(f"SEQ-01-S{s}") for s in (42, 43, 44)}
    etx_seed = {s: align(f"ETX-01-S{s}") for s in (42, 43, 44)}
    tab = 0.10 * align("S1-E03a") + 0.20 * align("S1-E02") + 0.25 * align("S1-DIST")
    rho_reference = {name: candidate_rho(q, z_match) for name, q in candidate_q.items()}
    variant_rows = []
    subsets = [combo for size in (1, 2, 3) for combo in itertools.combinations((42, 43, 44), size)]
    for seq_subset in subsets:
        for etx_subset in subsets:
            z_variant = (tab
                         + 0.225 * np.mean([seq_seed[s] for s in seq_subset], axis=0)
                         + 0.225 * np.mean([etx_seed[s] for s in etx_subset], axis=0))
            diffs = {name: abs(candidate_rho(q, z_variant) - rho_reference[name])
                     for name, q in candidate_q.items()}
            correlation = corr(z_variant, z_avg3)
            variant_rows.append({
                "seq_seeds": "+".join(map(str, seq_subset)),
                "etx_seeds": "+".join(map(str, etx_subset)),
                "seed_models_total": len(seq_subset) + len(etx_subset),
                "additional_historical_seed_families":
                    int(any(s != 42 for s in seq_subset)) + int(any(s != 42 for s in etx_subset)),
                "corr_vs_exact_AVG3": correlation,
                "max_candidate_rho_difference": max(diffs.values()),
                **{f"difference_{name}": value for name, value in diffs.items()},
                "fidelity_pass": correlation >= 0.995 and max(diffs.values()) <= 0.003,
            })
    variant_rows.sort(key=lambda row: (
        not row["fidelity_pass"], row["seed_models_total"],
        row["additional_historical_seed_families"], row["max_candidate_rho_difference"],
    ))
    correlation = corr(z_single, z_avg3)
    return {
        "canonical_cutoff": "2025-10-16", "corr_rebuilt_vs_exact_AVG3": correlation,
        "corr_gate_0_995": correlation >= 0.995,
        "corr_rebuilt_vs_composition_matched": corr(z_single, z_match),
        "RMS_rebuilt_minus_AVG3": rms(z_single - z_avg3),
        "candidate_rho_rows": rows,
        "surrogate_variant_rows": variant_rows,
        "recommended_fidelity_repair": variant_rows[0],
        "fidelity_pass": correlation >= 0.995 and all(x["passes_0_003"] for x in rows),
    }


def fit_amplitude(raw_cf: dict[str, np.ndarray], prior_folds: list[str],
                  fold_data: dict[str, dict[str, Any]]) -> tuple[float, list[dict[str, float]]]:
    rows, numerator, denominator = [], 0.0, 0.0
    for f in prior_folds:
        u, _ = project_two_pass(raw_cf[f], fold_data[f]["basis"])
        b, G = float(np.mean(u * fold_data[f]["residual"])), float(np.mean(u * u))
        w = FOLD_WEIGHT[f]
        numerator += w * b
        denominator += w * G
        rows.append({"cutoff": f, "b": b, "G": G, "oracle_amplitude": b / max(G, EPS), "weight": w})
    return numerator / max(denominator, EPS), rows


def bootstrap(deltas: dict[str, np.ndarray], uids: dict[str, np.ndarray], seed: int) -> dict[str, Any]:
    uid = np.concatenate([uids[f] for f in TRANSITIONS])
    delta = np.concatenate([deltas[f] for f in TRANSITIONS])
    fi = np.concatenate([np.full(len(deltas[f]), i, np.int8) for i, f in enumerate(TRANSITIONS)])
    unique, inv = np.unique(uid, return_inverse=True)
    fold_n = np.bincount(fi, minlength=len(TRANSITIONS)).astype(np.float64)
    fw = np.asarray([FOLD_WEIGHT[f] for f in TRANSITIONS], np.float64)
    row_w = fw[fi] / fold_n[fi] / fw.sum()
    cluster_w = np.bincount(inv, weights=row_w, minlength=len(unique))
    cluster_d = np.bincount(inv, weights=row_w * delta, minlength=len(unique))
    rng = np.random.default_rng(seed)
    draws = np.empty(BOOTSTRAP_REPS, np.float64)
    for start in range(0, BOOTSTRAP_REPS, 10):
        n = min(10, BOOTSTRAP_REPS - start)
        count = rng.poisson(1.0, size=(n, len(unique))).astype(np.float64)
        draws[start:start + n] = (count @ cluster_d) / np.maximum(count @ cluster_w, EPS)
    return {"replicates": BOOTSTRAP_REPS, "seed": seed, "unit": "user_id cluster across transitions",
            "CI95_Delta_MSE": np.quantile(draws, [0.025, 0.975]).tolist(),
            "P_Delta_MSE_lt_0": float(np.mean(draws < 0)), "draw_mean": float(draws.mean())}


def evaluate_candidate(candidate: str, config_names: list[str], fold_data: dict[str, dict[str, Any]],
                       seed_base: int) -> tuple[dict[str, Any], pd.DataFrame]:
    fold_rows, pred_frames, deltas, uids = [], [], {}, {}
    for k, fold in enumerate(TRANSITIONS, start=1):
        prior = list(FOLDS[:k])
        Xtr = np.concatenate([fold_data[f]["X"] for f in prior])
        ytr = np.concatenate([fold_data[f]["residual"] for f in prior])
        uidtr = np.concatenate([fold_data[f]["uid"] for f in prior])
        raw_cf_by_model: dict[str, dict[str, np.ndarray]] = {}
        raw_val_by_model: dict[str, np.ndarray] = {}
        for ci, cfg_name in enumerate(config_names):
            cfg = LGBM_CONFIGS[cfg_name]
            raw_cf = user_crossfit(Xtr, ytr, uidtr, cfg, seed_base + 100 * k + 20 * ci)
            raw_cf_by_model[cfg_name] = {}
            offset = 0
            for f in prior:
                n = len(fold_data[f]["uid"])
                raw_cf_by_model[cfg_name][f] = raw_cf[offset:offset + n]
                offset += n
            model = fit_regressor(Xtr, ytr, cfg, seed_base + 100 * k + 20 * ci + 10)
            raw_val_by_model[cfg_name] = model.predict(fold_data[fold]["X"]).astype(np.float32)
        raw_cf_mean = {f: np.mean(np.vstack([raw_cf_by_model[c][f] for c in config_names]), axis=0)
                       for f in prior}
        raw_val = np.mean(np.vstack([raw_val_by_model[c] for c in config_names]), axis=0)
        amplitude, amp_rows = fit_amplitude(raw_cf_mean, prior, fold_data)
        u_perp, projection = project_two_pass(raw_val, fold_data[fold]["basis"])
        correction = amplitude * u_perp
        residual = fold_data[fold]["residual"]
        delta = (residual - correction) ** 2 - residual ** 2
        b, G = float(np.mean(u_perp * residual)), float(np.mean(u_perp * u_perp))
        strong_basis = np.column_stack([np.ones(len(residual)), fold_data[fold]["baseline"]])
        u_strong, _ = project_two_pass(raw_val, strong_basis)
        row = {
            "candidate": candidate, "cutoff": fold, "train_folds": prior,
            "max_train_target_end": str(max(np.datetime64(f) + np.timedelta64(30, "D") for f in prior)),
            "label_available": bool(max(np.datetime64(f) + np.timedelta64(30, "D") for f in prior) <= np.datetime64(fold)),
            "rho_raw": corr(raw_val, fold_data[fold]["target_log"]),
            "rho_vs_strong_residual": corr(u_strong, residual),
            "rho_post_projection": corr(u_perp, residual),
            "rho_deployed_correction": corr(correction, residual) if np.any(correction) else 0.0,
            "b": b, "G": G, "oracle_amplitude": b / max(G, EPS),
            "deployable_amplitude": amplitude, "amplitude_training": amp_rows,
            "Delta_MSE": float(np.mean(delta)),
            "Delta_RMSLE": rms(residual - correction) - rms(residual),
            "baseline_RMSLE": rms(residual), **projection,
        }
        fold_rows.append(row)
        deltas[fold], uids[fold] = delta, fold_data[fold]["uid"]
        pred_frames.append(pd.DataFrame({
            "candidate": candidate, "user_id": fold_data[fold]["uid"], "cutoff": fold,
            "target_log": fold_data[fold]["target_log"], "z_production_like": fold_data[fold]["baseline"],
            "residual": residual, "u_raw": raw_val, "u_perp": u_perp,
            "deployable_amplitude": amplitude, "correction": correction, "delta_mse": delta,
        }))
    weights = np.asarray([FOLD_WEIGHT[f] for f in TRANSITIONS], np.float64)
    weighted_rho = float(np.average([r["rho_post_projection"] for r in fold_rows], weights=weights))
    weighted_delta = float(np.average([r["Delta_MSE"] for r in fold_rows], weights=weights))
    weighted_drmsle = float(np.average([r["Delta_RMSLE"] for r in fold_rows], weights=weights))
    leave_one_out = []
    for omitted_idx, omitted_fold in enumerate(TRANSITIONS):
        keep = np.arange(len(TRANSITIONS)) != omitted_idx
        kept_weights = weights[keep]
        kept_rows = [row for i, row in enumerate(fold_rows) if keep[i]]
        leave_one_out.append({
            "omitted_transition": omitted_fold,
            "weighted_rho": float(np.average(
                [row["rho_post_projection"] for row in kept_rows], weights=kept_weights)),
            "weighted_Delta_MSE": float(np.average(
                [row["Delta_MSE"] for row in kept_rows], weights=kept_weights)),
            "weighted_Delta_RMSLE": float(np.average(
                [row["Delta_RMSLE"] for row in kept_rows], weights=kept_weights)),
        })
    boot = bootstrap(deltas, uids, seed_base + 900)
    result = {"candidate": candidate, "configs": config_names, "folds": fold_rows,
              "weighted_purged_post_projection_rho": weighted_rho,
              "latest_purged_post_projection_rho": fold_rows[-1]["rho_post_projection"],
              "positive_rho_transitions": int(sum(r["rho_post_projection"] > 0 for r in fold_rows)),
              "nested_Delta_MSE": weighted_delta, "nested_Delta_RMSLE": weighted_drmsle,
              "leave_one_transition_out": leave_one_out, "bootstrap": boot}
    return result, pd.concat(pred_frames, ignore_index=True)


def main() -> None:
    t0 = time.time()
    EXP.mkdir(parents=True, exist_ok=True)
    fold_data = {f: load_components(f) for f in FOLDS}
    component_audit = []
    for fold in FOLDS:
        for family in PRODUCTION_WEIGHTS:
            meta_path = COMP / f"{family}_{fold}.json"
            artifact_path = COMP / f"{family}_{fold}.npz"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            recorded_end = meta.get("max_train_target_end")
            temporal_proof = "recorded by exact replay"
            if recorded_end is None and meta.get("mode") == "canonical frozen artifact reuse":
                recorded_end = "2025-10-11"
                temporal_proof = "inferred from canonical OOF recipe: last legal train cutoff 2025-09-11"
            row = {
                "family": family, "cutoff": fold, "mode": meta.get("mode"),
                "max_train_target_end": recorded_end, "temporal_proof": temporal_proof,
                "target_safe": recorded_end is not None and np.datetime64(recorded_end) <= np.datetime64(fold),
                "runtime_seconds": float(meta.get("runtime_seconds", 0.0)),
                "runtime_lt_6h": float(meta.get("runtime_seconds", 0.0)) < 6 * 3600,
                "config_changed": bool(meta.get("config_changed", False)),
                "recorded_sha256": meta.get("sha256"), "actual_sha256": sha256(artifact_path),
            }
            row["sha256_match"] = row["recorded_sha256"] == row["actual_sha256"]
            component_audit.append(row)
    component_frame = pd.DataFrame(component_audit)
    component_frame.to_csv(EXP / "production_component_audit.csv", index=False)
    if not (component_frame.target_safe.all() and component_frame.runtime_lt_6h.all()
            and (~component_frame.config_changed).all() and component_frame.sha256_match.all()):
        raise AssertionError("production component audit failed")
    leakage_rows = []
    for i, fold in enumerate(FOLDS):
        end = np.datetime64(fold) + np.timedelta64(30, "D")
        previous_end = None if i == 0 else np.datetime64(FOLDS[i - 1]) + np.timedelta64(30, "D")
        leakage_rows.append({"cutoff": fold, "target_start": str(np.datetime64(fold) + np.timedelta64(1, "D")),
                             "target_end": str(end), "n": len(fold_data[fold]["uid"]),
                             "spacing_from_previous_days": None if i == 0 else int((np.datetime64(fold) - np.datetime64(FOLDS[i - 1])).astype(int)),
                             "previous_target_known": True if i == 0 else bool(previous_end <= np.datetime64(fold)),
                             "final_outside_survivorship_interval": bool(end <= np.datetime64("2025-11-15"))})
    if len(FOLDS) < 4 or not all(x["previous_target_known"] for x in leakage_rows) or not all(x["final_outside_survivorship_interval"] for x in leakage_rows):
        raise AssertionError("BLOCKED_INSUFFICIENT_PURGED_FOLDS")
    pd.DataFrame(leakage_rows).to_csv(EXP / "fold_definitions.csv", index=False)
    write_json(EXP / "leakage_assertions.json", {"status": "PASS", "rows": leakage_rows,
                                                 "rule": "target_end(previous_fold) <= current_cutoff",
                                                 "same_period_targets_used_as_temporal_evidence": False})

    fidelity = baseline_fidelity_audit(fold_data["2025-10-16"])
    write_json(EXP / "baseline_fidelity_audit.json", fidelity)
    pd.DataFrame(fidelity["candidate_rho_rows"]).to_csv(EXP / "baseline_fidelity_candidate_rho.csv", index=False)
    pd.DataFrame(fidelity["surrogate_variant_rows"]).to_csv(EXP / "baseline_fidelity_seed_variants.csv", index=False)
    if not fidelity["fidelity_pass"]:
        raise RuntimeError("BASELINE_FIDELITY_FAIL")

    uid_all = np.load(e80.UID_PATH, mmap_mode="r")
    panel = np.load(e80.PANEL_PATH, mmap_mode="r")
    gmv = np.load(e80.GMV_PATH, mmap_mode="r")
    saved_obs = pd.read_parquet(EXP080 / "observable_predictions.parquet")
    saved_obs["cutoff"] = saved_obs.cutoff.astype(str)
    feature_audit = []
    feature_names_ref = None
    for fold in FOLDS:
        X, names, audit = build_fold_features(fold, fold_data[fold], uid_all, panel, gmv, saved_obs)
        if feature_names_ref is None:
            feature_names_ref = names
        elif names != feature_names_ref:
            raise AssertionError("feature-name drift")
        fold_data[fold]["X"] = X
        feature_audit.append({"cutoff": fold, **audit})

    exp081_config = json.loads((EXP081 / "config.json").read_text(encoding="utf-8"))
    exp081_feature_names = list(exp081_config["feature_names"])
    reproduced_feature_names = list(feature_names_ref or [])
    feature_fidelity = {
        "pass": reproduced_feature_names == exp081_feature_names,
        "exp081_feature_count": len(exp081_feature_names),
        "reproduced_feature_count": len(reproduced_feature_names),
        "missing_feature_names": [name for name in exp081_feature_names if name not in reproduced_feature_names],
        "unexpected_feature_names": [name for name in reproduced_feature_names if name not in exp081_feature_names],
        "reason": "The full 40-model canonical prediction bank and three auxiliary structural predictions have no saved values on the requested historical cutoffs.",
        "derived_disagreement_formula_match": False,
        "future_canonical_predictions_reused": False,
        "imputation_or_distillation_used": False,
    }
    write_json(EXP / "residual_feature_fidelity_gate.json", feature_fidelity)
    config = {"experiment": "EXP082_PURGED_TEMPORAL_RESIDUAL", "folds": FOLDS,
              "fold_weights": FOLD_WEIGHT, "production_weights": PRODUCTION_WEIGHTS,
              "residual_models": LGBM_CONFIGS, "feature_count": len(feature_names_ref or []),
              "feature_names": feature_names_ref, "exp081_original_feature_count": 200,
              "feature_reproduction": "162-column core-production subset; no imputation/distillation",
              "residual_feature_fidelity_pass": feature_fidelity["pass"],
              "bootstrap_reps": BOOTSTRAP_REPS, "leaderboard_used": False,
              "same_period_target_used_as_primary_temporal_evidence": False}
    write_json(EXP / "config.json", config)
    write_json(EXP / "feature_reproduction_audit.json", feature_audit)

    results, prediction_frames = {}, []
    for i, (name, cfgs) in enumerate((("LightGBM_A", ["A_depth3"]),
                                      ("LightGBM_B", ["B_depth5"]),
                                      ("LightGBM_AB_mean", ["A_depth3", "B_depth5"]))):
        result, pred = evaluate_candidate(name, cfgs, fold_data, 20270000 + i * 10000)
        results[name] = result
        prediction_frames.append(pred)
    write_json(EXP / "purged_results.json", results)
    metric_frame = pd.DataFrame([
        {k: v for k, v in row.items() if k not in ("train_folds", "amplitude_training")}
        for result in results.values() for row in result["folds"]
    ])
    metric_frame.to_csv(EXP / "purged_fold_metrics.csv", index=False)
    projection_columns = [
        "candidate", "cutoff", "RMS_u_raw", "RMS_u_centered", "RMS_u_perp",
        "perp_fraction", "second_pass_projection_error_RMS",
        "second_pass_relative_error", "rho_post_projection",
    ]
    metric_frame[projection_columns].to_csv(EXP / "projection_diagnostics.csv", index=False)
    write_json(EXP / "bootstrap.json", {
        name: {"bootstrap": result["bootstrap"],
               "leave_one_transition_out": result["leave_one_transition_out"]}
        for name, result in results.items()
    })
    pd.concat(prediction_frames, ignore_index=True).to_parquet(EXP / "residual_oof_predictions.parquet", index=False)

    primary = results["LightGBM_AB_mean"]
    boot = primary["bootstrap"]
    strong = (primary["weighted_purged_post_projection_rho"] >= 0.020
              and primary["latest_purged_post_projection_rho"] >= 0.020
              and primary["positive_rho_transitions"] == 3
              and boot["P_Delta_MSE_lt_0"] >= 0.95
              and primary["nested_Delta_MSE"] < 0)
    weighted_rho = primary["weighted_purged_post_projection_rho"]
    if strong:
        statistical_verdict = "STRONG_GO"
    elif 0.015 <= weighted_rho < 0.020 and primary["positive_rho_transitions"] == 3:
        statistical_verdict = "PARTIAL"
    else:
        statistical_verdict = "FINAL_NO_EVIDENCE"
    residual_feature_fidelity_pass = feature_fidelity["pass"]
    verdict = statistical_verdict if residual_feature_fidelity_pass else "BLOCKED"

    residual_mse = float(np.average([np.mean(fold_data[f]["residual"] ** 2) for f in TRANSITIONS],
                                    weights=[FOLD_WEIGHT[f] for f in TRANSITIONS]))
    rho2_headroom = weighted_rho ** 2 * residual_mse
    robust_gain = max(0.0, -float(boot["CI95_Delta_MSE"][1]))
    headroom = {"required_Delta_MSE_gain": REQUIRED_MSE,
                "weighted_purged_rho": weighted_rho,
                "achieved_purged_rho_squared": weighted_rho ** 2,
                "weighted_production_residual_MSE": residual_mse,
                "correlation_only_MSE_headroom": rho2_headroom,
                "correlation_only_fraction_of_gap": rho2_headroom / REQUIRED_MSE,
                "nested_point_MSE_gain": max(0.0, -primary["nested_Delta_MSE"]),
                "nested_point_fraction_of_gap": max(0.0, -primary["nested_Delta_MSE"]) / REQUIRED_MSE,
                "robust_95pct_MSE_headroom": robust_gain,
                "robust_95pct_fraction_of_gap": robust_gain / REQUIRED_MSE}
    write_json(EXP / "mathematical_headroom.json", headroom)

    old_summary = pd.read_csv(EXP081 / "candidate_summary.csv")
    old_details = json.loads((EXP081 / "candidate_details.json").read_text(encoding="utf-8"))
    same = old_summary[old_summary.candidate.eq("lgbm_A_depth3_user_crossfit_full_span")].iloc[0]
    ordered = old_summary[old_summary.candidate.eq("lgbm_A_ordered_previous_folds_full_span")].iloc[0]
    same_detail = old_details["lgbm_A_depth3_user_crossfit_full_span"]
    ordered_detail = old_details["lgbm_A_ordered_previous_folds_full_span"]
    old_weights = np.asarray([1.0, 2.0, 4.0, 8.0])
    same_rho = float(np.average([x["raw_post_projection_rho"] for x in same_detail["folds"]], weights=old_weights))
    ordered_rho = float(np.average([x["raw_post_projection_rho"] for x in ordered_detail["folds"]], weights=old_weights))
    new_a = results["LightGBM_A"]
    protocol = pd.DataFrame([
        {"Protocol": "A. same-period user-disjoint (EXP081)", "candidate": "LightGBM A / full-200",
         "rho": same_rho, "latest_rho": same.latest_clean_post_projection_rho,
         "Delta_MSE": same.weighted_Delta_MSE, "P_gain": same.P_Delta_MSE_lt_0},
        {"Protocol": "B. old ordered canonical (EXP081)", "candidate": "LightGBM A / full-200",
         "rho": ordered_rho, "latest_rho": ordered.latest_clean_post_projection_rho,
         "Delta_MSE": ordered.weighted_Delta_MSE, "P_gain": ordered.P_Delta_MSE_lt_0},
        {"Protocol": "C. fully purged 35-day (EXP082)", "candidate": "LightGBM A / core-162",
         "rho": new_a["weighted_purged_post_projection_rho"],
         "latest_rho": new_a["latest_purged_post_projection_rho"],
         "Delta_MSE": new_a["nested_Delta_MSE"], "P_gain": new_a["bootstrap"]["P_Delta_MSE_lt_0"]},
    ])
    protocol.to_csv(EXP / "protocol_comparison.csv", index=False)

    verdict_payload = {"verdict": verdict,
                       "statistical_verdict_core_subset": statistical_verdict,
                       "blocker": None if residual_feature_fidelity_pass else
                           "BLOCKED_RESIDUAL_FEATURE_FIDELITY: frozen EXP081 used 200 features; only 162 cutoff-reproducible columns are available on the requested folds",
                       "primary_candidate": "LightGBM_AB_mean",
                       "criteria": {"weighted_rho_ge_0_020": weighted_rho >= 0.020,
                                    "latest_rho_ge_0_020": primary["latest_purged_post_projection_rho"] >= 0.020,
                                    "positive_all_three": primary["positive_rho_transitions"] == 3,
                                    "P_gain_ge_0_95": boot["P_Delta_MSE_lt_0"] >= 0.95,
                                    "nested_Delta_MSE_lt_0": primary["nested_Delta_MSE"] < 0},
                       "submission_created": False,
                       "residual_feature_fidelity_pass": residual_feature_fidelity_pass,
                       "submission_reason": "created only after exact-fidelity STRONG_GO" if strong else "STRONG_GO gate not passed",
                       "runtime_seconds": time.time() - t0}
    write_json(EXP / "verdict.json", verdict_payload)
    print(json.dumps(jsonable({"verdict": verdict_payload, "primary": primary, "headroom": headroom}),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
