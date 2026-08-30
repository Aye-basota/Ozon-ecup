"""Build exact walk-forward occurrence outputs and evaluate EXP086 shock signals."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = HERE.parents[3]
COMP = EXP / "occurrence_components"
EXP082 = ROOT / "research" / "new_directions" / "EXP082_PURGED_TEMPORAL_RESIDUAL"
PROD = EXP082 / "production_components"
PROCESSED = Path(r"C:\Users\Admin\Desktop\OZON-E-CUP\data\processed")
TEAM = Path(r"C:\Users\Admin\Desktop\OZON-E-CUP\пайплайн сокомандника")
SOURCE = TEAM / "research_scripts"
FOLDS = ("2025-07-03", "2025-08-07", "2025-09-11", "2025-10-16")
FOLD_WEIGHT = {fold: float(2 ** i) for i, fold in enumerate(FOLDS)}
TRANSITIONS = FOLDS[1:]
EPS = 1e-12
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20260829
OCC_NAMES = (
    "occ_r10_fast", "occ_r16_bal", "occ_r22_stable", "occ_r14_multiscale",
    "occ_r18_wide", "occ_r24_multiscale", "occ_r12_wide", "occ_r20_shallow",
)
PROD_WEIGHTS = {"cap": 0.10, "unc": 0.20, "dist": 0.25, "seq": 0.225, "etx": 0.225}
TABLE_WEIGHTS = {"cap": 0.10 / 0.55, "unc": 0.20 / 0.55, "dist": 0.25 / 0.55}


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
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    sx, sy = float(x.std()), float(y.std())
    if sx < EPS or sy < EPS:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2)))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def project_two_pass(raw: np.ndarray, basis: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    raw = np.asarray(raw, np.float64)
    centered = raw - raw.mean(axis=0, keepdims=True)
    coef1, *_ = np.linalg.lstsq(basis, centered, rcond=1e-10)
    first = centered - basis @ coef1
    coef2, *_ = np.linalg.lstsq(basis, first, rcond=1e-10)
    removed2 = basis @ coef2
    out = first - removed2
    raw_rms = np.sqrt(np.mean(raw ** 2, axis=0))
    centered_rms = np.sqrt(np.mean(centered ** 2, axis=0))
    perp_rms = np.sqrt(np.mean(out ** 2, axis=0))
    second = np.sqrt(np.mean(removed2 ** 2, axis=0))
    return out, {
        "RMS_raw_max": float(np.max(raw_rms)),
        "RMS_centered_max": float(np.max(centered_rms)),
        "RMS_perp_max": float(np.max(perp_rms)),
        "perp_fraction_min": float(np.min(perp_rms / np.maximum(centered_rms, EPS))),
        "second_pass_projection_error_RMS_max": float(np.max(second)),
        "second_pass_relative_error_max": float(np.max(second / np.maximum(perp_rms, EPS))),
    }


def make_occ_meta(seed: int, leaves: int = 31) -> lgb.LGBMClassifier:
    # Exact continue_best_bas_final6h.py::make_occ_meta configuration.
    return lgb.LGBMClassifier(
        n_estimators=420, learning_rate=0.03, num_leaves=leaves, max_depth=-1,
        min_child_samples=450, subsample=0.88, colsample_bytree=0.78,
        reg_lambda=18.0, reg_alpha=1.2, max_bin=127, random_state=seed,
        n_jobs=max(2, min(10, os.cpu_count() or 8)), verbosity=-1,
    )


def make_risk_model(seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=320, learning_rate=0.035, num_leaves=23, min_child_samples=550,
        subsample=0.90, colsample_bytree=0.72, reg_lambda=22.0, reg_alpha=1.5,
        max_bin=127, random_state=seed, n_jobs=max(2, min(10, os.cpu_count() or 8)),
        verbosity=-1,
    )


def load_fold(fold: str) -> dict[str, Any]:
    prod: dict[str, np.ndarray] = {}
    ref_uid = ref_target = None
    components = []
    for family in PROD_WEIGHTS:
        part = load_npz(PROD / f"{family}_{fold}.npz")
        uid = part["user_id"].astype(np.int64)
        target = part["target_log"].astype(np.float64)
        if ref_uid is None:
            ref_uid, ref_target = uid, target
        elif not np.array_equal(uid, ref_uid) or not np.allclose(target, ref_target, atol=1e-10, rtol=0):
            raise AssertionError(f"production alignment failed: {fold}/{family}")
        prod[family] = part["z"].astype(np.float64)
        components.append(prod[family])
    hurdle = load_npz(COMP / f"hurdle__{fold}.npz")
    meta = load_npz(COMP / f"meta_raw__{fold}.npz")
    if not np.array_equal(hurdle["user_id"], ref_uid) or not np.array_equal(meta["user_id"], ref_uid):
        raise AssertionError(f"occurrence alignment failed: {fold}")
    occ = {}
    for name in OCC_NAMES:
        part = load_npz(COMP / f"{name}__{fold}.npz")
        if not np.array_equal(part["user_id"], ref_uid):
            raise AssertionError(f"raw occurrence alignment failed: {fold}/{name}")
        occ[name] = np.clip(part["p"].astype(np.float64), 1e-7, 1 - 1e-7)
    baseline = sum(PROD_WEIGHTS[name] * prod[name] for name in PROD_WEIGHTS)
    table_core = sum(TABLE_WEIGHTS[name] * prod[name] for name in TABLE_WEIGHTS)
    Z = np.column_stack(components)
    basis = np.column_stack([np.ones(len(ref_uid)), Z, baseline])
    feature_path = PROCESSED / f"feat_{fold.replace('-', '')}_LnormNone.parquet"
    feature = pd.read_parquet(
        feature_path,
        columns=["user_id", "w30_gmv", "rec_buy", "w90_days_buy", "w30_days_present"],
    ).sort_values("user_id")
    source_uid = feature.user_id.to_numpy(np.int64)
    pos = np.searchsorted(source_uid, ref_uid)
    if pos.max(initial=0) >= len(source_uid) or not np.array_equal(source_uid[pos], ref_uid):
        raise AssertionError(f"state feature alignment failed: {fold}")
    feature = feature.iloc[pos].reset_index(drop=True)
    p_base = np.clip(hurdle["p"].astype(np.float64), 1e-7, 1 - 1e-7)
    mu = np.maximum(hurdle["mu"].astype(np.float64), 0)
    return {
        "uid": ref_uid, "target": ref_target, "prod": prod, "components": Z,
        "baseline": baseline, "residual": ref_target - baseline, "basis": basis,
        "table_core": table_core,
        "p_base": p_base, "mu": mu,
        "meta_raw": meta["X"].astype(np.float32), "meta_names": meta["names"].astype(str),
        # Exact teammate base-hurdle output is the weak comparator.  It is not
        # used as primary evidence; the five-family EXP082 blend remains the
        # production-like residual throughout the audit.
        "occ": occ, "weak_z": p_base * mu,
        "state": feature,
    }


def occ_meta_features(record: dict[str, Any], names: tuple[str, ...] = OCC_NAMES) -> np.ndarray:
    p = record["p_base"]
    cols: list[np.ndarray] = [logit(p), p, np.log1p(record["mu"]), record["table_core"]]
    for name in names:
        cols.extend([logit(record["occ"][name]), record["occ"][name] - p])
    raw = record["meta_raw"]
    if raw.shape[1] > 96:
        raw = raw[:, :96]
    return np.column_stack([raw, *cols]).astype(np.float32)


def risk_features(record: dict[str, Any]) -> np.ndarray:
    X = occ_meta_features(record)
    p, mu, table = record["p_base"], record["mu"], record["table_core"]
    extra = np.column_stack([table - p * mu, p * mu, p * (1 - p)]).astype(np.float32)
    return np.column_stack([X, extra]).astype(np.float32)


def risk_labels(record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    zero = record["target"] <= 0
    false_one = (zero & (record["p_base"] >= 0.5)).astype(np.int8)
    severe_over = ((record["table_core"] - record["target"]) > 1.0).astype(np.int8)
    return false_one, severe_over


def build_walk_forward_outputs(data: dict[str, dict[str, Any]]) -> None:
    """Exact all8/power1.7/leaves31 meta and exact risk gate."""
    for i, fold in enumerate(FOLDS):
        current = data[fold]
        if i == 0:
            current["p_meta"] = current["p_base"].copy()
            current["risk"] = np.full(len(current["uid"]), 0.5, np.float64)
            continue
        Xs, ys, ws, yfs, yos = [], [], [], [], []
        for j in range(i):
            prior = data[FOLDS[j]]
            X = occ_meta_features(prior)
            weight = np.full(len(prior["uid"]), FOLD_WEIGHT[FOLDS[j]] ** 1.7, np.float32)
            yf, yo = risk_labels(prior)
            Xs.append(X)
            ys.append((prior["target"] > 0).astype(np.int8))
            ws.append(weight)
            yfs.append(yf)
            yos.append(yo)
        X = np.vstack(Xs)
        y = np.concatenate(ys)
        w = np.concatenate(ws)
        model = make_occ_meta(7100 + i, leaves=31)
        model.fit(X, y, sample_weight=w)
        current["p_meta"] = np.clip(
            model.predict_proba(occ_meta_features(current))[:, 1], 1e-7, 1 - 1e-7,
        )
        del model, y
        # Exact risk layer uses the same features plus three risk-specific columns.
        XR = np.vstack([risk_features(data[FOLDS[j]]) for j in range(i)])
        yf, yo = np.concatenate(yfs), np.concatenate(yos)
        mf, mo = make_risk_model(8100 + i), make_risk_model(8200 + i)
        mf.fit(XR, yf, sample_weight=w)
        mo.fit(XR, yo, sample_weight=w)
        Xt = risk_features(current)
        pf, po = mf.predict_proba(Xt)[:, 1], mo.predict_proba(Xt)[:, 1]
        current["risk"] = np.sqrt(np.clip(pf * po, 0, 1))
        del X, XR, Xs, ys, ws, yfs, yos, w, yf, yo, mf, mo, Xt, pf, po


def build_signals(record: dict[str, Any]) -> dict[str, np.ndarray]:
    p0, mu = record["p_base"], record["mu"]
    praw, pmeta, risk = record["occ"]["occ_r10_fast"], record["p_meta"], record["risk"]
    raw = (praw - p0) * mu
    meta = (pmeta - p0) * mu
    disagreement = (pmeta - praw) * mu
    trust = np.where(meta < 0, 0.30 + 0.70 * risk, 0.70 + 0.30 * (1 - risk))
    return {
        "raw_X3_intensity_shock": raw,
        "meta_B_intensity_shock": meta,
        "raw_meta_disagreement": disagreement,
        "meta_risk_weighted_shock": trust * meta,
    }


def weighted_moments(
    projected: dict[str, dict[str, np.ndarray]], data: dict[str, dict[str, Any]],
    folds: tuple[str, ...], names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    G = np.zeros((len(names), len(names)), np.float64)
    b = np.zeros(len(names), np.float64)
    total = 0.0
    for fold in folds:
        U = np.column_stack([projected[fold][name] for name in names])
        r = data[fold]["residual"]
        weight = FOLD_WEIGHT[fold]
        G += weight * (U.T @ U / len(U))
        b += weight * (U.T @ r / len(U))
        total += weight
    return G / total, b / total


def solve_moments(G: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.pinv(G, rcond=1e-10) @ b


def bootstrap_nested(
    delta: dict[str, np.ndarray], uids: dict[str, np.ndarray], reps: int = BOOTSTRAP_REPS,
) -> dict[str, Any]:
    all_uid = np.unique(np.concatenate([uids[fold] for fold in TRANSITIONS]))
    index = {int(uid): i for i, uid in enumerate(all_uid)}
    numerator = np.zeros(len(all_uid), np.float64)
    denominator = np.zeros(len(all_uid), np.float64)
    for fold in TRANSITIONS:
        ids = uids[fold]
        pos = np.fromiter((index[int(uid)] for uid in ids), dtype=np.int64, count=len(ids))
        row_weight = FOLD_WEIGHT[fold] / len(ids)
        np.add.at(numerator, pos, row_weight * delta[fold])
        np.add.at(denominator, pos, row_weight)
    point = float(numerator.sum() / denominator.sum())
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(reps, np.float64)
    batch = 20
    for start in range(0, reps, batch):
        stop = min(reps, start + batch)
        q = rng.poisson(1.0, size=(stop - start, len(all_uid))).astype(np.float32)
        draws[start:stop] = (q @ numerator) / np.maximum(q @ denominator, EPS)
    return {
        "method": "Poisson cluster bootstrap by user_id across purged transitions",
        "seed": BOOTSTRAP_SEED, "reps": reps, "point_delta_mse": point,
        "ci95": np.quantile(draws, [0.025, 0.975]),
        "p_gain": float(np.mean(draws < 0)),
        "draw_mean": float(draws.mean()), "draw_sd": float(draws.std(ddof=1)),
    }


def segment_label(values: np.ndarray, kind: str, baseline: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(values, np.float64)
    if kind == "baseline_decile":
        rank = pd.Series(values).rank(method="first", pct=True).to_numpy()
        return np.minimum((rank * 10).astype(int), 9).astype(str)
    if kind == "recency":
        labels = np.asarray(pd.cut(values, [-np.inf, 7, 30, 90, np.inf], labels=["0-7", "8-30", "31-90", ">90/never"]).astype(str))
        labels[~np.isfinite(values)] = ">90/never"
        return labels
    if kind == "frequency":
        return pd.cut(values, [-np.inf, 0, 1, 3, 7, np.inf], labels=["0", "1", "2-3", "4-7", "8+"]).astype(str)
    if kind == "recent_activity":
        return pd.cut(values, [-np.inf, 5, 15, 25, np.inf], labels=["0-5", "6-15", "16-25", "26-30"]).astype(str)
    raise KeyError(kind)


def build_segments(data: dict[str, dict[str, Any]], nested: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    specs = {
        "baseline_prediction_decile": ("baseline_decile", "baseline"),
        "recency": ("recency", "rec_buy"),
        "purchase_frequency_90d": ("frequency", "w90_days_buy"),
        "recent_activity_30d": ("recent_activity", "w30_days_present"),
    }
    for dimension, (kind, column) in specs.items():
        labels_by_fold = {}
        unique = set()
        for fold in TRANSITIONS:
            values = data[fold]["baseline"] if column == "baseline" else data[fold]["state"][column].to_numpy(np.float64)
            labels = segment_label(values, kind)
            labels_by_fold[fold] = labels
            unique.update(labels.tolist())
        for label in sorted(unique):
            num = den = 0.0
            count = 0
            mean_corr_num = 0.0
            mean_res_num = 0.0
            for fold in TRANSITIONS:
                mask = labels_by_fold[fold] == label
                if not np.any(mask):
                    continue
                r = data[fold]["residual"][mask]
                d = nested[fold][mask]
                weight = FOLD_WEIGHT[fold]
                num += weight * float(np.mean((r - d) ** 2 - r ** 2))
                mean_corr_num += weight * float(np.mean(d))
                mean_res_num += weight * float(np.mean(r))
                den += weight
                count += int(mask.sum())
            rows.append({
                "dimension": dimension, "segment": label, "n_rows": count,
                "weighted_mean_correction": mean_corr_num / den,
                "weighted_mean_residual": mean_res_num / den,
                "weighted_delta_mse": num / den,
            })
    return pd.DataFrame(rows)


def provenance_outputs() -> None:
    latest = TEAM / "latest"
    final6 = TEAM / "review_bundles" / "final6h_REVIEW_BUNDLE_20260823_204823_extracted"
    extra = TEAM / "review_bundles" / "extra90_REVIEW_BUNDLE_20260823_222555_extracted"
    rows = [
        {
            "artifact": "8 raw occurrence probability heads (including occ_r10_fast)",
            "pipeline": "continue_best_bas_final6h.py::train_occ_child; frozen OCC_QUEUE",
            "target": "1[GMV next 30d > 0]",
            "features": "6 heads x 227 normalized-long features; 2 multiscale heads x 202",
            "training_dates": "last 10/12/14/16/18/20/22/24 eligible weekly cutoffs; T_train+30<=T_val",
            "TEST_path": "source TEST child is exact; not executed in EXP086 because gate is NO_GO",
            "reproducible": "YES for primary fold outputs and source TEST path",
        },
        {
            "artifact": "base hurdle p_base, mu",
            "pipeline": "run_best_bas_research_23h.py::recent_hurdle (two_part)",
            "target": "binary occurrence plus positive-case log1p(GMV30)",
            "features": "227 normalized-long cutoff-safe table features",
            "training_dates": "all eligible weekly cutoffs; T_train+30<=T_val",
            "TEST_path": "source TEST path is exact; not executed in EXP086 because gate is NO_GO",
            "reproducible": "YES",
        },
        {
            "artifact": "meta_raw state",
            "pipeline": "choose_meta_features/build_test_meta_raw",
            "target": "none (cutoff-state matrix)",
            "features": "first deterministic 72 recent aggregate features",
            "training_dates": "state at each validation cutoff",
            "TEST_path": "source TEST path is exact; not executed in EXP086 because gate is NO_GO",
            "reproducible": "YES",
        },
        {
            "artifact": "all8 meta occurrence probability (core of occ_meta_B)",
            "pipeline": "walk_meta_occ(all8,power=1.7,leaves=31)",
            "target": "1[GMV next 30d > 0]",
            "features": "72 meta_raw + p_base/mu/table_core + 8 raw p/logit deltas",
            "training_dates": "walk-forward earlier fully-purged folds only",
            "TEST_path": "final_meta_occ(seed=7900); not executed in EXP086 because gate is NO_GO",
            "reproducible": "YES",
        },
        {
            "artifact": "false-one / severe-over risk trust output",
            "pipeline": "walk_risk_gate(all8,power=1.7)",
            "target": "false-one and severe-over binary error indicators",
            "features": "meta-occurrence feature matrix + table_core-p*mu, p*mu, p*(1-p)",
            "training_dates": "walk-forward earlier fully-purged folds only",
            "TEST_path": "final_risk_gate(seeds 8901/8902); not executed in EXP086 because gate is NO_GO",
            "reproducible": "YES",
        },
        {
            "artifact": "EXP086 primary intermediate predictions",
            "pipeline": "exact raw/meta/risk replay before downstream submission overlay",
            "target": "all primary incidence and reliability outputs",
            "features": "p_base, mu, 8 raw p, p_meta, risk, four shock candidates and d_perp",
            "training_dates": "2025-07-03, 2025-08-07, 2025-09-11, 2025-10-16",
            "TEST_path": str(EXP / "occurrence_predictions.parquet"),
            "reproducible": "YES",
        },
        {
            "artifact": "occ_raw_X3.csv",
            "pipeline": "xraw_occ_r10_fast_adapt__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85",
            "target": "downstream 30d log-GMV candidate using raw occurrence correction",
            "features": "occ_r10_fast p + base hurdle p/mu + stable Ridge/greedy table anchor",
            "training_dates": "historical old 4 folds 2025-09-04..2025-10-16",
            "TEST_path": str(latest / "components" / "occ_raw_X3.csv"),
            "reproducible": "TEST YES; full historical downstream OOF NO (missing bank)",
        },
        {
            "artifact": "occ_meta_B.csv",
            "pipeline": "metaocc_l31_risk__blend_ridge_recentpow1p7_s075__greedy35_finalizable_pr85",
            "target": "meta 30d occurrence + false-one/severe-over trust gate",
            "features": "all 8 occurrence heads + meta_raw + hurdle state + stable table anchor",
            "training_dates": "historical old 4 folds 2025-09-04..2025-10-16",
            "TEST_path": str(latest / "components" / "occ_meta_B.csv"),
            "reproducible": "TEST YES; full historical downstream OOF NO (missing bank)",
        },
        {
            "artifact": "latest.csv",
            "pipeline": "0.12 friend + 0.16 occ_meta_B + 0.72 occ_raw_X3 in log1p; clip z>=0",
            "target": "submission blend",
            "features": "three TEST submission vectors",
            "training_dates": "none at final blend stage",
            "TEST_path": str(latest / "latest.csv"),
            "reproducible": "YES",
        },
    ]
    pd.DataFrame(rows).to_csv(EXP / "occurrence_pipeline_reconstruction.csv", index=False)
    detail = {
        "source_files": {
            str(SOURCE / "continue_best_bas_final6h.py"): sha256(SOURCE / "continue_best_bas_final6h.py"),
            str(SOURCE / "materialize_final6h_extra90m.py"): sha256(SOURCE / "materialize_final6h_extra90m.py"),
            str(latest / "rebuild_latest.py"): sha256(latest / "rebuild_latest.py"),
        },
        "test_artifacts": {
            str(latest / "components" / "occ_raw_X3.csv"): sha256(latest / "components" / "occ_raw_X3.csv"),
            str(latest / "components" / "occ_meta_B.csv"): sha256(latest / "components" / "occ_meta_B.csv"),
            str(latest / "latest.csv"): sha256(latest / "latest.csv"),
        },
        "review_evidence": [
            str(final6 / "results" / "RUN_MANIFEST.json"),
            str(final6 / "results" / "OCCURRENCE_BRANCH_VALIDATION.csv"),
            str(extra / "results" / "RUN_MANIFEST.json"),
            str(extra / "results" / "ALL_EXTRA90_VALIDATION.csv"),
        ],
        "relation": {
            "occ_raw_X3": "raw occ_r10_fast adaptive overlay over the frozen Ridge/greedy anchor",
            "occ_meta_B": "all8 walk-forward meta-occurrence leaves31 plus risk gate over the same anchor",
            "latest": "0.12 friend + 0.16 occ_meta_B + 0.72 occ_raw_X3 in log1p space",
        },
    }
    write_json(EXP / "provenance.json", detail)
    fold_rows = []
    for cutoff in FOLDS:
        start = pd.Timestamp(cutoff) + pd.Timedelta(days=1)
        end = start + pd.Timedelta(days=29)
        fold_rows.append({
            "cutoff": cutoff, "target_start": start.date().isoformat(),
            "target_end": end.date().isoformat(), "spacing_days": 35,
            "prior_target_fully_known": "N/A" if cutoff == FOLDS[0] else True,
            "primary_evidence": cutoff != FOLDS[0],
        })
    pd.DataFrame(fold_rows).to_csv(EXP / "fold_definitions.csv", index=False)
    write_json(EXP / "config.json", {
        "experiment": "EXP086_OCCURRENCE_SHOCK",
        "folds": FOLDS, "purged_transitions": TRANSITIONS,
        "fold_weights": FOLD_WEIGHT, "production_weights": PROD_WEIGHTS,
        "occurrence_heads": OCC_NAMES, "meta": {"power": 1.7, "leaves": 31},
        "projection": "remove mean; least-squares project out [1, cap, unc, dist, seq, etx, production blend]; repeat projection",
        "bootstrap": {"method": "Poisson cluster by user_id", "reps": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED},
        "leaderboard_used": False,
    })


def novelty_outputs() -> None:
    rows = [
        ("EXP063 occurrence revisit", "single S1-E11 two-part GMV member", "eight incidence-only heads and their relative disagreement with p_base; no direct GMV donor blend", "NOVEL"),
        ("EVENT-ORDER", "ordered daily funnel-state transitions", "bagged temporal incidence estimators over aggregate state, not event-order tokens", "NOVEL"),
        ("OPEN-FUNNEL", "Search/Cart after last purchase", "broad 227-feature incidence ensemble; no unresolved-funnel handcrafted state", "NOVEL"),
        ("BURST/GAP", "fixed episode/gap summaries", "recency/capacity disagreement across independently trained incidence heads", "NOVEL"),
        ("BTYD", "parametric common-origin repeat-purchase process", "nonparametric LightGBM incidence with recent cutoff cap and meta disagreement", "NOVEL"),
        ("MHZ hazard", "multi-horizon hazard/count heads", "only exact 30d incidence; novelty is ensemble disagreement versus own hurdle state, not extra horizons", "NOVEL"),
        ("generic count/value", "future count bins and conditional value", "no future count/value target; occurrence probability only, mu is frozen state scaling", "NOVEL"),
        ("EXP075 temporal trajectory", "position-specific daily/weekly path shape", "same aggregate feature universe but a different supervised object: relative next-30d incidence shock", "PARTIALLY_OVERLAPPING_FEATURES_NEW_TARGET_GEOMETRY"),
    ]
    pd.DataFrame(rows, columns=["prior_experiment", "prior_mechanism", "incremental_information", "verdict"]).to_csv(
        EXP / "novelty_audit.csv", index=False,
    )


def main() -> None:
    provenance_outputs()
    novelty_outputs()
    data = {fold: load_fold(fold) for fold in FOLDS}
    build_walk_forward_outputs(data)

    signals = {fold: build_signals(data[fold]) for fold in FOLDS}
    names = tuple(next(iter(signals.values())).keys())
    projected: dict[str, dict[str, np.ndarray]] = {}
    projection_rows = []
    prediction_frames = []
    for fold in FOLDS:
        raw_matrix = np.column_stack([signals[fold][name] for name in names])
        perp_matrix, diag = project_two_pass(raw_matrix, data[fold]["basis"])
        projected[fold] = {name: perp_matrix[:, i] for i, name in enumerate(names)}
        for i, name in enumerate(names):
            centered = raw_matrix[:, i] - raw_matrix[:, i].mean()
            projection_rows.append({
                "cutoff": fold, "candidate": name,
                "RMS_raw": rms(raw_matrix[:, i]), "RMS_centered": rms(centered),
                "RMS_perp": rms(perp_matrix[:, i]),
                "perp_fraction": rms(perp_matrix[:, i]) / max(rms(centered), EPS),
                "second_pass_projection_error_RMS_max": diag["second_pass_projection_error_RMS_max"],
                "mean_raw": float(raw_matrix[:, i].mean()),
                "mean_perp": float(perp_matrix[:, i].mean()),
            })
        frame = pd.DataFrame({
            "user_id": data[fold]["uid"], "cutoff": fold,
            "target_log": data[fold]["target"].astype(np.float32),
            "z_production_like": data[fold]["baseline"].astype(np.float32),
            "p_base": data[fold]["p_base"].astype(np.float32),
            "mu": data[fold]["mu"].astype(np.float32),
            "p_meta_all8_l31": data[fold]["p_meta"].astype(np.float32),
            "risk_gate": data[fold]["risk"].astype(np.float32),
        })
        for occ_name in OCC_NAMES:
            frame[f"p_{occ_name}"] = data[fold]["occ"][occ_name].astype(np.float32)
        for name in names:
            frame[name] = signals[fold][name].astype(np.float32)
            frame[f"{name}_perp"] = projected[fold][name].astype(np.float32)
        prediction_frames.append(frame)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_parquet(EXP / "occurrence_predictions.parquet", index=False, compression="zstd")
    pd.DataFrame(projection_rows).to_csv(EXP / "projection_diagnostics.csv", index=False)

    # Scalar candidate audit, including the mandatory b/G arithmetic.
    candidate_rows = []
    overlap_rows = []
    scalar_nested: dict[str, dict[str, np.ndarray]] = {name: {} for name in names}
    for i, fold in enumerate(FOLDS):
        for name in names:
            raw = signals[fold][name] - np.mean(signals[fold][name])
            d = projected[fold][name]
            r = data[fold]["residual"]
            b = float(np.mean(d * r))
            G = float(np.mean(d * d))
            amplitude = b / G if G > EPS else 0.0
            candidate_rows.append({
                "cutoff": fold, "candidate": name,
                "rho_vs_weak_residual": corr(raw, data[fold]["target"] - data[fold]["weak_z"]),
                "rho_vs_strong_residual": corr(raw, r),
                "rho_post_span": corr(d, r), "b": b, "G": G,
                "oracle_amplitude": amplitude,
                "oracle_mse_gain": -(b * b / G) if G > EPS else 0.0,
                "is_purged_transition": i > 0,
            })
            if i > 0:
                Gp, bp = weighted_moments(projected, data, FOLDS[:i], (name,))
                amp = float(solve_moments(Gp, bp)[0])
                scalar_nested[name][fold] = amp * d
        if i > 0:
            # Joint coefficients are fitted only on fully available previous folds.
            joint_names = ("raw_X3_intensity_shock", "meta_B_intensity_shock")
            Gp, bp = weighted_moments(projected, data, FOLDS[:i], joint_names)
            coef = solve_moments(Gp, bp)
            rawU = np.column_stack([signals[fold][name] - np.mean(signals[fold][name]) for name in joint_names])
            U = np.column_stack([projected[fold][name] for name in joint_names])
            raw_joint = rawU @ coef
            perp_joint = U @ coef
            overlap_rows.append({
                "cutoff": fold, "candidate": "joint_raw_meta_nested",
                "rho_vs_weak_residual": corr(raw_joint, data[fold]["target"] - data[fold]["weak_z"]),
                "rho_vs_strong_residual": corr(raw_joint, data[fold]["residual"]),
                "rho_post_span": corr(perp_joint, data[fold]["residual"]),
                "coef_raw": float(coef[0]), "coef_meta": float(coef[1]),
            })
    candidate_frame = pd.DataFrame(candidate_rows)
    candidate_frame.to_csv(EXP / "candidate_signal_audit.csv", index=False)
    pd.DataFrame(overlap_rows).to_csv(EXP / "production_overlap.csv", index=False)
    aggregate_candidates = {}
    for name in names:
        part = candidate_frame[(candidate_frame.candidate == name) & candidate_frame.is_purged_transition]
        w = np.asarray([FOLD_WEIGHT[x] for x in part.cutoff], np.float64)
        aggregate_candidates[name] = {
            "weighted_rho_vs_weak_residual": float(np.average(part.rho_vs_weak_residual, weights=w)),
            "weighted_rho_vs_strong_residual": float(np.average(part.rho_vs_strong_residual, weights=w)),
            "weighted_rho_post_span": float(np.average(part.rho_post_span, weights=w)),
            "latest_rho_post_span": float(part.iloc[-1].rho_post_span),
            "positive_transitions": int((part.rho_post_span > 0).sum()),
        }
    write_json(EXP / "candidate_aggregate.json", aggregate_candidates)

    # Joint covariance, same-fold oracle headroom, and deployable nested application.
    joint_names = ("raw_X3_intensity_shock", "meta_B_intensity_shock")
    joint_rows = []
    nested: dict[str, np.ndarray] = {}
    delta: dict[str, np.ndarray] = {}
    for i, fold in enumerate(FOLDS[1:], start=1):
        U = np.column_stack([projected[fold][name] for name in joint_names])
        r = data[fold]["residual"]
        G = U.T @ U / len(U)
        b = U.T @ r / len(U)
        oracle_coef = solve_moments(G, b)
        oracle_direction = U @ oracle_coef
        past_G, past_b = weighted_moments(projected, data, FOLDS[:i], joint_names)
        deploy_coef = solve_moments(past_G, past_b)
        correction = U @ deploy_coef
        nested[fold] = correction
        delta[fold] = (r - correction) ** 2 - r ** 2
        raw_rho = corr(U[:, 0], r)
        meta_rho = corr(U[:, 1], r)
        joint_rows.append({
            "cutoff": fold, "train_folds": "+".join(FOLDS[:i]),
            "raw_rho": raw_rho, "meta_rho": meta_rho,
            "joint_oracle_rho": corr(oracle_direction, r),
            "post_span_nested_rho": corr(correction, r),
            "G_raw_raw": float(G[0, 0]), "G_raw_meta": float(G[0, 1]),
            "G_meta_meta": float(G[1, 1]), "b_raw": float(b[0]), "b_meta": float(b[1]),
            "oracle_coef_raw": float(oracle_coef[0]), "oracle_coef_meta": float(oracle_coef[1]),
            "deployed_coef_raw": float(deploy_coef[0]), "deployed_coef_meta": float(deploy_coef[1]),
            "baseline_mse": float(np.mean(r ** 2)),
            "corrected_mse": float(np.mean((r - correction) ** 2)),
            "delta_mse": float(np.mean(delta[fold])),
            "delta_rmsle": rms(r - correction) - rms(r),
            "mean_correction": float(np.mean(correction)), "rms_correction": rms(correction),
        })
    joint_frame = pd.DataFrame(joint_rows)
    joint_frame.to_csv(EXP / "raw_meta_joint_metrics.csv", index=False)

    weights = np.asarray([FOLD_WEIGHT[f] for f in TRANSITIONS], np.float64)
    weighted_rho = float(np.average(joint_frame.post_span_nested_rho, weights=weights))
    baseline_mse = float(np.average(joint_frame.baseline_mse, weights=weights))
    corrected_mse = float(np.average(joint_frame.corrected_mse, weights=weights))
    nested_delta_mse = corrected_mse - baseline_mse
    nested_delta_rmsle = math.sqrt(corrected_mse) - math.sqrt(baseline_mse)
    boot = bootstrap_nested(delta, {fold: data[fold]["uid"] for fold in TRANSITIONS})
    write_json(EXP / "bootstrap.json", boot)

    pooled_G, pooled_b = weighted_moments(projected, data, TRANSITIONS, joint_names)
    pooled_coef = solve_moments(pooled_G, pooled_b)
    pooled_gain = float(pooled_b @ pooled_coef)
    raw_only_gain = float(pooled_b[0] ** 2 / pooled_G[0, 0]) if pooled_G[0, 0] > EPS else 0.0
    meta_only_gain = float(pooled_b[1] ** 2 / pooled_G[1, 1]) if pooled_G[1, 1] > EPS else 0.0
    covariance = {
        "G": pooled_G, "b": pooled_b, "a_oracle": pooled_coef,
        "joint_oracle_mse_gain": -pooled_gain,
        "raw_only_oracle_mse_gain": -raw_only_gain,
        "meta_only_oracle_mse_gain": -meta_only_gain,
        "conditional_raw_gain_given_meta": pooled_gain - meta_only_gain,
        "conditional_meta_gain_given_raw": pooled_gain - raw_only_gain,
        "condition_number": float(np.linalg.cond(pooled_G)),
    }
    write_json(EXP / "joint_covariance.json", covariance)

    latest_rho = float(joint_frame.iloc[-1].post_span_nested_rho)
    positives = int((joint_frame.post_span_nested_rho > 0).sum())
    promising = weighted_rho >= 0.015 and latest_rho > 0 and positives >= 2
    strong = (
        weighted_rho >= 0.020 and latest_rho >= 0.020 and positives == 3
        and boot["p_gain"] >= 0.95 and nested_delta_mse < 0
    )
    breakthrough = weighted_rho >= 0.025
    verdict = "STRONG_GO" if strong else "PROMISING" if promising else "NO_GO"
    gap = 0.004909273595110
    signed_nested_gain = -nested_delta_mse
    summary = {
        "verdict": verdict, "breakthrough": breakthrough,
        "weighted_purged_post_span_rho": weighted_rho,
        "latest_post_span_rho": latest_rho, "positive_transitions": positives,
        "nested_delta_mse": nested_delta_mse, "nested_delta_rmsle": nested_delta_rmsle,
        "bootstrap_ci95_delta_mse": boot["ci95"], "p_gain": boot["p_gain"],
        "required_gap_mse": gap,
        "occurrence_nested_gain": signed_nested_gain,
        "fraction_of_gap_signed": signed_nested_gain / gap,
        "fraction_of_gap": max(0.0, signed_nested_gain) / gap,
        "test_inference_authorized": strong,
        "leaderboard_used": False,
    }
    write_json(EXP / "verdict.json", summary)
    write_json(EXP / "mathematical_headroom.json", {
        "required_gap_mse": gap,
        "deployable_nested_delta_mse": nested_delta_mse,
        "deployable_nested_gain": signed_nested_gain,
        "deployable_fraction_of_gap_signed": signed_nested_gain / gap,
        "deployable_fraction_of_gap_closeable": max(0.0, signed_nested_gain) / gap,
        "pooled_same_fold_oracle_gain": pooled_gain,
        "pooled_same_fold_oracle_fraction_of_gap": pooled_gain / gap,
        "note": "Same-fold oracle is mathematical headroom only; it is not deployable evidence.",
    })

    segment = build_segments(data, nested)
    segment.to_csv(EXP / "segment_diagnostics.csv", index=False)

    manifest_rows = []
    for path in sorted(EXP.rglob("*")):
        if (
            path.is_file() and "__pycache__" not in path.parts
            and path.name not in {"SHA256SUMS.txt", "artifact_manifest.csv"}
        ):
            manifest_rows.append({"path": str(path.relative_to(EXP)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(manifest_rows).to_csv(EXP / "artifact_manifest.csv", index=False)
    (EXP / "SHA256SUMS.txt").write_text(
        "".join(f"{row['sha256']}  {row['path']}\n" for row in manifest_rows), encoding="utf-8",
    )
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
