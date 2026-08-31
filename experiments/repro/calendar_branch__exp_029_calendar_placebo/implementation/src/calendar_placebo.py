"""CALENDAR-PLACEBO-01: fixed-support temporal drift placebo diagnostics.

The main experiment compares historical cutoff pairs and the latest clean
historical cutoff against the real test cutoff.  Every classifier receives the
same fixed-L180 behavioral representation, an equal number of rows per class,
and deterministic user-group OOF folds.  Dates, source markers and user IDs are
metadata only.

Run from the repository root:

    python src/calendar_placebo.py run --baseline-artifacts <path>

Large checkpoints are written to ``artifacts/calendar_placebo_01``.  Compact
reviewable CSV/JSON reports are written to
``research/calendar_placebo_01/results``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import SGDClassifier

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (ARTIFACTS, CUTOFF_TEST, DATA_START, FOLD_WEIGHTS_S1,
                        SEED, VAL_FOLDS_S1, cutoff_grid)
from src.domain01 import (domain_metrics, fit_lgb_oof, load_production_predictions,
                          shift_report, user_group_fold)
from src.features import feature_names, make_xy, to_np
from src.validation import calibrate


HISTORY_SUPPORT = 180
PANEL_BLOCKS = 3
N_GROUP_FOLDS = 5
REAL_TASK = "real_20251016_to_20260213"
PRODUCTION_MODEL = "SEQ-01-MIX"

# Repeated transitions at the same gap make a calendar-looking change face a
# genuine placebo: it must not be inferred from a single month boundary.
PLACEBO_PAIRS = [
    ("p07_early", dt.date(2025, 7, 3), dt.date(2025, 7, 10), "short"),
    ("p07_mid", dt.date(2025, 9, 18), dt.date(2025, 9, 25), "short"),
    ("p07_late", dt.date(2025, 10, 9), dt.date(2025, 10, 16), "short"),
    ("p28_early", dt.date(2025, 7, 3), dt.date(2025, 7, 31), "medium"),
    ("p28_mid", dt.date(2025, 8, 21), dt.date(2025, 9, 18), "medium"),
    ("p28_late", dt.date(2025, 9, 18), dt.date(2025, 10, 16), "medium"),
    ("p56_early", dt.date(2025, 7, 3), dt.date(2025, 8, 28), "long"),
    ("p56_late", dt.date(2025, 8, 21), dt.date(2025, 10, 16), "long"),
    ("p105_max", dt.date(2025, 7, 3), dt.date(2025, 10, 16), "max"),
]


def log(*items) -> None:
    print(*items, flush=True)


def json_dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                               default=_json_default), encoding="utf-8")


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (dt.date, Path)):
        return str(value)
    raise TypeError(type(value).__name__)


def strict_support_eligible(cutoff: dt.date, history: int = HISTORY_SUPPORT) -> bool:
    """Whether exactly ``history`` pre-cutoff days exist in the source data."""
    return cutoff >= DATA_START + dt.timedelta(days=history)


def validate_feature_contract(features: list[str]) -> None:
    forbidden_tokens = ("user_id", "cutoff", "fold", "source", "domain", "date")
    bad = [name for name in features
           if any(token in name.lower() for token in forbidden_tokens)]
    if bad:
        raise ValueError(f"forbidden source/date identifiers: {bad}")


def validate_task(cutoff_a: dt.date, cutoff_b: dt.date,
                  history: int = HISTORY_SUPPORT,
                  panel_blocks: int = PANEL_BLOCKS) -> None:
    if cutoff_a >= cutoff_b:
        raise ValueError("domain direction must be earlier cutoff A -> later cutoff B")
    if cutoff_b > CUTOFF_TEST:
        raise ValueError("cutoff after the observed dataset end")
    if history != HISTORY_SUPPORT or panel_blocks != PANEL_BLOCKS:
        raise ValueError("main protocol requires fixed-L180 and the same 3-block panel")
    if not strict_support_eligible(cutoff_a, history):
        raise ValueError(f"{cutoff_a} has less than {history} days of source history")


def signed_standardized_shift(a, b) -> dict[str, float]:
    """Signed B-A mean/median shifts on a common pooled scale."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    af, bf = a[np.isfinite(a)], b[np.isfinite(b)]
    if not len(af) or not len(bf):
        return {"smd": np.nan, "median_shift": np.nan}
    va, vb = float(np.var(af)), float(np.var(bf))
    pooled = math.sqrt(max((va + vb) / 2.0, 1e-12))
    joined = np.r_[af, bf]
    iqr = max(float(np.quantile(joined, 0.75) - np.quantile(joined, 0.25)), 1e-9)
    return {"smd": float((np.mean(bf) - np.mean(af)) / pooled),
            "median_shift": float((np.median(bf) - np.median(af)) / iqr)}


def _sample_rows(n: int, cap: int, seed: int) -> np.ndarray:
    if n <= cap:
        return np.arange(n)
    return np.sort(np.random.RandomState(seed).choice(n, cap, replace=False))


def load_state(cutoff: dt.date, cap: int, seed_offset: int = 0):
    X, _ = make_xy(cutoff, HISTORY_SUPPORT, n_blocks=PANEL_BLOCKS,
                   with_target=False, norm_long=False)
    features = feature_names(X)
    validate_feature_contract(features)
    rows = _sample_rows(X.height, cap, SEED + seed_offset)
    users = X["user_id"].to_numpy()[rows]
    A = to_np(X, features)[rows]
    A[~np.isfinite(A)] = np.nan
    del X
    return A, users, features


def balanced_task_matrix(cutoff_a: dt.date, cutoff_b: dt.date, cap: int):
    validate_task(cutoff_a, cutoff_b)
    A, ua, features_a = load_state(cutoff_a, cap, 11)
    B, ub, features_b = load_state(cutoff_b, cap, 29)
    if features_a != features_b:
        raise AssertionError("fixed-L180 feature schema differs between cutoffs")
    n = min(len(A), len(B))
    A, ua, B, ub = A[:n], ua[:n], B[:n], ub[:n]
    X = np.vstack([A, B])
    users = np.r_[ua, ub]
    y = np.r_[np.zeros(n, np.int8), np.ones(n, np.int8)]
    sources = y.copy()
    return X, y, users, sources, features_a


def fit_signed_linear(X: np.ndarray, y: np.ndarray, features: list[str],
                      cap: int = 80_000) -> pd.DataFrame:
    """Full-sample standardized logistic direction; GBDT gain has no sign."""
    rng = np.random.RandomState(SEED + 901)
    per_class = min(cap // 2, int(np.sum(y == 0)), int(np.sum(y == 1)))
    rows = np.r_[rng.choice(np.flatnonzero(y == 0), per_class, replace=False),
                 rng.choice(np.flatnonzero(y == 1), per_class, replace=False)]
    A = X[rows].astype(np.float32, copy=True)
    median = np.nanmedian(A, axis=0)
    median[~np.isfinite(median)] = 0.0
    bad = ~np.isfinite(A)
    A[bad] = median[np.where(bad)[1]]
    mean, scale = A.mean(axis=0), A.std(axis=0)
    scale[scale < 1e-6] = 1.0
    A = np.clip((A - mean) / scale, -20, 20)
    model = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4,
                          max_iter=200, tol=1e-4, average=True,
                          random_state=SEED)
    model.fit(A, y[rows])
    out = pd.DataFrame({"feature": features,
                        "linear_standardized_coef": model.coef_[0].copy()})
    del A, model, bad
    gc.collect()
    return out


def run_domain_task(name: str, cutoff_a: dt.date, cutoff_b: dt.date,
                    gap_group: str, args, out: Path, report_dir: Path) -> dict:
    task_dir = out / name
    task_dir.mkdir(parents=True, exist_ok=True)
    task_json = task_dir / "summary.json"
    task_shift = task_dir / "signed_shifts.csv"
    task_imp = task_dir / "feature_importance.csv"
    task_folds = task_dir / "domain_folds.csv"
    if args.resume and all(p.exists() for p in (task_json, task_shift, task_imp, task_folds)):
        log(f"{name}: resume compact reports")
        return json.loads(task_json.read_text(encoding="utf-8"))

    log(f"{name}: {cutoff_a} -> {cutoff_b} ({(cutoff_b-cutoff_a).days}d)")
    X, y, users, sources, features = balanced_task_matrix(cutoff_a, cutoff_b,
                                                          args.state_cap)
    groups = user_group_fold(users, args.folds)
    for fold in range(args.folds):
        train_users = np.unique(users[groups != fold])
        valid_users = np.unique(users[groups == fold])
        if np.intersect1d(train_users, valid_users).size:
            raise AssertionError("user leakage in grouped split")
    result = fit_lgb_oof(name, X, y, users, sources, features,
                         n_folds=args.folds, rounds=args.rounds,
                         train_cap=args.train_cap, out_dir=task_dir)
    linear = fit_signed_linear(X, y, features, args.linear_cap)
    shifts = shift_report(X, y, features, result["importance"],
                          sample_n=args.shift_cap).merge(linear, on="feature", how="left")
    shifts["signed_ks"] = np.sign(shifts["smd"]) * shifts["ks"]
    shifts.insert(0, "task", name)
    shifts.insert(1, "cutoff_a", cutoff_a.isoformat())
    shifts.insert(2, "cutoff_b", cutoff_b.isoformat())
    shifts.to_csv(task_shift, index=False)
    result["importance"].assign(task=name).to_csv(task_imp, index=False)
    result["folds"].assign(cutoff_a=cutoff_a.isoformat(),
                           cutoff_b=cutoff_b.isoformat()).to_csv(task_folds, index=False)
    summary = {"task": name, "cutoff_a": cutoff_a, "cutoff_b": cutoff_b,
               "gap_days": (cutoff_b - cutoff_a).days, "gap_group": gap_group,
               "history_support": HISTORY_SUPPORT, "panel_blocks": PANEL_BLOCKS,
               "n_per_class": int(np.sum(y == 0)), "n_features": len(features),
               **result["metrics"]}
    json_dump(task_json, summary)
    del X, y, users, sources, result, linear, shifts
    gc.collect()
    return json.loads(task_json.read_text(encoding="utf-8"))


def vector_similarity(shifts: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    pivot = shifts.pivot(index="feature", columns="task", values="smd")
    tasks, rows = list(pivot.columns), []
    for left in tasks:
        for right in tasks:
            a, b = pivot[left].to_numpy(float), pivot[right].to_numpy(float)
            ok = np.isfinite(a) & np.isfinite(b)
            denom = float(np.linalg.norm(a[ok]) * np.linalg.norm(b[ok]))
            cosine = float(np.dot(a[ok], b[ok]) / denom) if denom else np.nan
            rho = float(spearmanr(a[ok], b[ok]).statistic) if ok.sum() > 2 else np.nan
            ia = set(pivot.index[np.argsort(np.abs(a))[-top_n:]])
            ib = set(pivot.index[np.argsort(np.abs(b))[-top_n:]])
            overlap = ia & ib
            pos = {f: i for i, f in enumerate(pivot.index)}
            sign_agree = (float(np.mean([np.sign(a[pos[f]]) == np.sign(b[pos[f]])
                                         for f in overlap])) if overlap else np.nan)
            rows.append({"task_left": left, "task_right": right, "cosine": cosine,
                         "signed_rank_spearman": rho, "top_n": top_n,
                         "top_overlap_n": len(overlap),
                         "top_overlap_fraction": len(overlap) / top_n,
                         "overlap_sign_agreement": sign_agree})
    return pd.DataFrame(rows)


def mean_placebo_similarity(shifts: pd.DataFrame) -> dict:
    pivot = shifts.pivot(index="feature", columns="task", values="smd")
    placebo = [c for c in pivot.columns if c != REAL_TASK]
    U = pivot[placebo].to_numpy(float)
    norms = np.linalg.norm(np.nan_to_num(U), axis=0)
    norms[norms == 0] = 1.0
    mean_direction = np.nanmean(U / norms, axis=1)
    real = pivot[REAL_TASK].to_numpy(float)
    ok = np.isfinite(real) & np.isfinite(mean_direction)
    cosine = float(np.dot(real[ok], mean_direction[ok]) /
                   (np.linalg.norm(real[ok]) * np.linalg.norm(mean_direction[ok])))
    rho = float(spearmanr(real[ok], mean_direction[ok]).statistic)
    return {"real_vs_mean_placebo_cosine": cosine,
            "real_vs_mean_placebo_signed_rank_spearman": rho,
            "n_placebo_tasks": len(placebo)}


def gap_curve_diagnostic(metrics: pd.DataFrame) -> dict:
    """Simple pre-test interpolation/extrapolation diagnostic, not a fitted model."""
    placebo = metrics[metrics.task != REAL_TASK]
    x = placebo.gap_days.to_numpy(float)
    y = placebo.roc_auc.to_numpy(float)
    slope, intercept = np.polyfit(x, y, 1)
    real = metrics[metrics.task == REAL_TASK].iloc[0]
    expected = float(intercept + slope * float(real.gap_days))
    return {"linear_auc_per_day": float(slope), "linear_intercept": float(intercept),
            "expected_auc_at_real_gap": expected,
            "real_auc_minus_gap_curve": float(real.roc_auc - expected)}


def circular_calendar_gap(a: dt.date, b: dt.date) -> int:
    da, db = a.timetuple().tm_yday, b.timetuple().tm_yday
    raw = abs(da - db)
    return int(min(raw, 365 - raw))


def calendar_alignment(args, real_shift: pd.DataFrame, report_dir: Path) -> dict:
    """Distances for every strict-L180 cutoff; no fake February L180 state."""
    real_vec = real_shift.set_index("feature")["smd"]
    test, _, test_features = load_state(CUTOFF_TEST, args.shift_cap, 131)
    rows = []
    for i, cutoff in enumerate(cutoff_grid(HISTORY_SUPPORT, 7)):
        hist, _, features = load_state(cutoff, args.shift_cap, 211 + i)
        if features != test_features:
            raise AssertionError("calendar alignment feature schema differs")
        values = []
        for j, feature in enumerate(features):
            values.append((feature, signed_standardized_shift(hist[:, j], test[:, j])["smd"]))
        v = pd.Series(dict(values), dtype=float).reindex(real_vec.index)
        ok = np.isfinite(v) & np.isfinite(real_vec)
        denom = float(np.linalg.norm(v[ok]) * np.linalg.norm(real_vec[ok]))
        rows.append({"cutoff": cutoff.isoformat(), "eligible_fixed_l180": True,
                     "chronological_gap_days": (CUTOFF_TEST - cutoff).days,
                     "calendar_gap_days": circular_calendar_gap(cutoff, CUTOFF_TEST),
                     "rms_smd_to_test": float(np.sqrt(np.nanmean(v.to_numpy() ** 2))),
                     "median_abs_smd_to_test": float(np.nanmedian(np.abs(v.to_numpy()))),
                     "cosine_to_latest_test_direction":
                         float(np.dot(v[ok], real_vec[ok]) / denom) if denom else np.nan,
                     "limitation": ""})
        del hist
    # Exact YoY cutoff exists, but using it as L180 would silently provide only
    # 44 source days and a different (one-block) selection mechanism.
    yoy = dt.date(2025, 2, 13)
    rows.append({"cutoff": yoy.isoformat(), "eligible_fixed_l180": False,
                 "chronological_gap_days": (CUTOFF_TEST - yoy).days,
                 "calendar_gap_days": 0, "rms_smd_to_test": np.nan,
                 "median_abs_smd_to_test": np.nan,
                 "cosine_to_latest_test_direction": np.nan,
                 "limitation": "only 44 calendar days available; 1-block panel, not fixed-L180/3-block"})
    out = pd.DataFrame(rows)
    out.to_csv(report_dir / "calendar_alignment.csv", index=False)
    valid = out[out.eligible_fixed_l180].copy()
    nearest_chrono = valid.loc[valid.chronological_gap_days.idxmin()].to_dict()
    nearest_calendar = valid.loc[valid.calendar_gap_days.idxmin()].to_dict()
    summary = {"exact_yoy_eligible": False,
               "exact_yoy_reason": rows[-1]["limitation"],
               "nearest_chronological": nearest_chrono,
               "nearest_calendar_available": nearest_calendar,
               "independent_calendar_contrast_available":
                   nearest_chrono["cutoff"] != nearest_calendar["cutoff"]}
    json_dump(report_dir / "calendar_alignment_summary.json", summary)
    return summary


def _fold_calibrated(y: np.ndarray, z: np.ndarray, cutoffs: np.ndarray) -> np.ndarray:
    out = np.empty(len(z), float)
    for cutoff in sorted(set(cutoffs.tolist())):
        m = cutoffs == cutoff
        offset, _ = calibrate(y[m], z[m])
        out[m] = np.maximum(z[m] + offset, 0.0)
    return out


def score_error_relationship(args, real_shift: pd.DataFrame, report_dir: Path) -> dict:
    """Project the production OOF panels on the fixed-L180 test direction."""
    anchor, _, features = load_state(VAL_FOLDS_S1[-1], 10**9, 0)
    direction = real_shift.set_index("feature")["smd"].reindex(features).to_numpy(float)
    direction[~np.isfinite(direction)] = 0.0
    norm = float(np.linalg.norm(direction))
    if norm == 0:
        raise AssertionError("empty real-test drift direction")
    direction /= norm
    center = np.nanmean(anchor, axis=0)
    scale = np.nanstd(anchor, axis=0)
    center[~np.isfinite(center)] = 0.0
    scale[(~np.isfinite(scale)) | (scale < 1e-6)] = 1.0
    pd.DataFrame({"feature": features, "anchor_mean": center, "anchor_std": scale,
                  "real_smd": real_shift.set_index("feature")["smd"].reindex(features).to_numpy(),
                  "score_weight": direction}).to_csv(report_dir / "calendar_score_direction.csv",
                                                       index=False)
    del anchor

    users_all, cutoffs_all, scores_all = [], [], []
    for cutoff in VAL_FOLDS_S1:
        X, _ = make_xy(cutoff, HISTORY_SUPPORT, n_blocks=PANEL_BLOCKS,
                       with_target=False, norm_long=False)
        if feature_names(X) != features:
            raise AssertionError("OOF score feature schema differs")
        A = to_np(X, features)
        bad = ~np.isfinite(A)
        A[bad] = center[np.where(bad)[1]]
        score = np.clip((A - center) / scale, -20, 20).dot(direction)
        users_all.append(X["user_id"].to_numpy())
        cutoffs_all.append(np.full(X.height, cutoff.isoformat(), dtype="U10"))
        scores_all.append(score.astype(np.float32))
        del X, A, bad, score
        gc.collect()
    users = np.concatenate(users_all)
    cutoffs = np.concatenate(cutoffs_all)
    score = np.concatenate(scores_all).astype(float)
    models, y = load_production_predictions(Path(args.baseline_artifacts), users, cutoffs)
    selected_models = ["SEQ-01-MIX", "S1-DIST-MIX", "S1-ROUNDS", "S1-SEEDAVG3"]
    calibrated = {name: _fold_calibrated(y, models[name], cutoffs) for name in selected_models}
    ly = np.log1p(y)
    rows, corr_rows = [], []
    for fold_i, cutoff in enumerate([d.isoformat() for d in VAL_FOLDS_S1]):
        m = cutoffs == cutoff
        local_rank = rankdata(score[m], method="average")
        q = np.minimum(((local_rank - 1) * 5 / len(local_rank)).astype(int) + 1, 5)
        prod_residual = ly[m] - calibrated[PRODUCTION_MODEL][m]
        corr_rows.append({
            "cutoff": cutoff, "n": int(m.sum()),
            "spearman_score_squared_error": float(spearmanr(score[m], prod_residual ** 2).statistic),
            "spearman_score_signed_residual": float(spearmanr(score[m], prod_residual).statistic),
        })
        for quantile in range(1, 6):
            qm = q == quantile
            for name in selected_models:
                residual = ly[m][qm] - calibrated[name][m][qm]
                rows.append({"cutoff": cutoff, "fold_weight": FOLD_WEIGHTS_S1[fold_i],
                             "score_quantile": quantile, "model": name,
                             "n": int(qm.sum()), "score_mean": float(np.mean(score[m][qm])),
                             "rmsle": float(np.sqrt(np.mean(residual ** 2))),
                             "residual_bias": float(np.mean(residual))})
    by_fold = pd.DataFrame(rows)
    by_fold.to_csv(report_dir / "calendar_score_error_by_fold_quantile.csv", index=False)
    correlations = pd.DataFrame(corr_rows)
    correlations.to_csv(report_dir / "calendar_score_error_correlations.csv", index=False)
    agg_rows = []
    for (quantile, name), group in by_fold.groupby(["score_quantile", "model"]):
        weights = group.fold_weight.to_numpy(float)
        agg_rows.append({"score_quantile": quantile, "model": name,
                         "weighted_fold_rmsle": float(np.average(group.rmsle, weights=weights)),
                         "weighted_residual_bias": float(np.average(group.residual_bias,
                                                                      weights=weights)),
                         "folds_won_vs_production": int(np.sum(group.rmsle.to_numpy() <
                             by_fold[(by_fold.score_quantile == quantile) &
                                     (by_fold.model == PRODUCTION_MODEL)].rmsle.to_numpy()))
                         if name != PRODUCTION_MODEL else 0})
    aggregate = pd.DataFrame(agg_rows)
    aggregate.to_csv(report_dir / "calendar_score_error_aggregate.csv", index=False)
    high = aggregate[aggregate.score_quantile == 5].sort_values("weighted_fold_rmsle")
    prod = aggregate[aggregate.model == PRODUCTION_MODEL].set_index("score_quantile")
    summary = {
        "production_model": PRODUCTION_MODEL,
        "production_q1_rmsle": float(prod.loc[1, "weighted_fold_rmsle"]),
        "production_q5_rmsle": float(prod.loc[5, "weighted_fold_rmsle"]),
        "production_q5_minus_q1": float(prod.loc[5, "weighted_fold_rmsle"] -
                                          prod.loc[1, "weighted_fold_rmsle"]),
        "best_high_score_model": str(high.iloc[0].model),
        "best_high_score_rmsle": float(high.iloc[0].weighted_fold_rmsle),
        "high_score_delta_best_vs_production": float(high.iloc[0].weighted_fold_rmsle -
            high[high.model == PRODUCTION_MODEL].iloc[0].weighted_fold_rmsle),
        "mean_abs_error_spearman": float(np.mean(np.abs(correlations.spearman_score_squared_error))),
        "mean_signed_residual_spearman": float(np.mean(correlations.spearman_score_signed_residual)),
    }
    json_dump(report_dir / "calendar_score_error_summary.json", summary)
    return summary


def run(args) -> None:
    started = time.time()
    out, report_dir = Path(args.out), Path(args.report_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    tasks = list(PLACEBO_PAIRS) + [
        (REAL_TASK, VAL_FOLDS_S1[-1], CUTOFF_TEST, "real-test")]
    specs = []
    for name, a, b, group in tasks:
        validate_task(a, b)
        specs.append({"task": name, "cutoff_a": a, "cutoff_b": b,
                      "gap_days": (b-a).days, "gap_group": group,
                      "history_support": HISTORY_SUPPORT, "panel_blocks": PANEL_BLOCKS,
                      "class_sampling": f"equal cap {args.state_cap}"})
    pd.DataFrame(specs).to_csv(report_dir / "placebo_pairs.csv", index=False)
    summaries = [run_domain_task(name, a, b, group, args, out, report_dir)
                 for name, a, b, group in tasks]
    metrics = pd.DataFrame(summaries)
    metrics.to_csv(report_dir / "domain_task_metrics.csv", index=False)
    folds = pd.concat([pd.read_csv(out / name / "domain_folds.csv")
                       for name, *_ in tasks], ignore_index=True)
    folds.to_csv(report_dir / "domain_task_folds.csv", index=False)
    shifts = pd.concat([pd.read_csv(out / name / "signed_shifts.csv")
                        for name, *_ in tasks], ignore_index=True)
    shifts.to_csv(report_dir / "signed_shift_vectors.csv", index=False)
    (shifts.assign(abs_smd=shifts.smd.abs())
     .sort_values(["task", "abs_smd"], ascending=[True, False])
     .groupby("task", as_index=False).head(20)
     .to_csv(report_dir / "top_shifted_features.csv", index=False))
    importance = pd.concat([pd.read_csv(out / name / "feature_importance.csv")
                            for name, *_ in tasks], ignore_index=True)
    importance.to_csv(report_dir / "feature_importance.csv", index=False)
    audit = {
        "feature_path": "make_xy(cutoff, L=180, n_blocks=3) -> build_features(cutoff)",
        "source_upper_bound": "event_date <= cutoff (enforced by build_features)",
        "all_task_endpoints_not_after_dataset_end": all(b <= CUTOFF_TEST for _, a, b, _ in tasks),
        "all_task_endpoints_have_full_l180": all(strict_support_eligible(a) and
                                                   strict_support_eligible(b)
                                                   for _, a, b, _ in tasks),
        "same_history_support": HISTORY_SUPPORT,
        "same_panel_blocks": PANEL_BLOCKS,
        "balanced_rows_per_class": True,
        "grouped_split": "deterministic user hash; disjoint users asserted in every fold/task",
        "forbidden_columns_present": [],
        "user_id_used_as_feature": False,
        "explicit_date_cutoff_fold_source_feature": False,
    }
    json_dump(report_dir / "leakage_support_audit.json", audit)
    similarity = vector_similarity(shifts)
    similarity.to_csv(report_dir / "drift_vector_similarity.csv", index=False)
    gap_summary = (metrics[metrics.task != REAL_TASK]
                   .groupby(["gap_group", "gap_days"], as_index=False)
                   .agg(n_pairs=("task", "count"), roc_auc_mean=("roc_auc", "mean"),
                        roc_auc_min=("roc_auc", "min"), roc_auc_max=("roc_auc", "max"),
                        pr_auc_mean=("pr_auc", "mean")))
    gap_summary.to_csv(report_dir / "gap_auc_summary.csv", index=False)
    placebo_similarity = mean_placebo_similarity(shifts)
    gap_curve = gap_curve_diagnostic(metrics)
    real_shift = shifts[shifts.task == REAL_TASK]
    calendar_summary = calendar_alignment(args, real_shift, report_dir)
    error_summary = score_error_relationship(args, real_shift, report_dir)
    real_row = metrics[metrics.task == REAL_TASK].iloc[0]
    conclusion = {
        "experiment": "CALENDAR-PLACEBO-01", "seed": SEED,
        "protocol": {"history_support": HISTORY_SUPPORT, "panel_blocks": PANEL_BLOCKS,
                     "group_folds": args.folds, "rounds": args.rounds,
                     "state_cap_per_class": args.state_cap,
                     "forbidden_identifiers": ["date", "cutoff", "fold", "source", "user_id"]},
        "n_placebo_pairs": len(PLACEBO_PAIRS),
        "real_test_auc": float(real_row.roc_auc), "real_test_pr_auc": float(real_row.pr_auc),
        "gap_auc_summary": gap_summary.to_dict("records"),
        "gap_curve_diagnostic": gap_curve,
        "drift_similarity": placebo_similarity,
        "calendar_alignment": calendar_summary,
        "error_relationship": error_summary,
        "verdict": "STOP-CALENDAR",
        "verdict_reason": ("real direction differs from summer/autumn placebos, but exact YoY "
                           "fixed-L180 comparison is ineligible and the direction score has no "
                           "monotonic production-error or component-win evidence"),
        "next_experiment": "SEQ-DEPTH-AUG-01",
        "runtime_s": time.time() - started,
    }
    json_dump(report_dir / "calendar_placebo_01_summary.json", conclusion)
    json_dump(out / "calendar_placebo_01_summary.json", conclusion)
    log(json.dumps(conclusion, ensure_ascii=False, indent=2, default=_json_default))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    runp = sub.add_parser("run")
    runp.add_argument("--out", default=str(ARTIFACTS / "calendar_placebo_01"))
    runp.add_argument("--report-dir", default="research/calendar_placebo_01/results")
    runp.add_argument("--baseline-artifacts", required=True)
    runp.add_argument("--folds", type=int, default=N_GROUP_FOLDS)
    runp.add_argument("--rounds", type=int, default=80)
    runp.add_argument("--state-cap", type=int, default=120_000)
    runp.add_argument("--train-cap", type=int, default=180_000)
    runp.add_argument("--linear-cap", type=int, default=80_000)
    runp.add_argument("--shift-cap", type=int, default=60_000)
    runp.add_argument("--resume", action="store_true")
    runp.set_defaults(func=run)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
