"""DOMAIN-01: test-like validation and covariate/domain-shift diagnostics.

Primary domain task:
  class 0 = observations on the four production validation cutoffs;
  class 1 = observations on the real test cutoff.

The classifier sees only the feature columns produced by the production
``build_features(cutoff_date)`` path.  ``user_id``, cutoff/fold identifiers and
dataset-source columns are metadata and are never passed to a model.  OOF folds
are deterministic user groups: every state of one user is held out together.

Commands (from the repository root):

  python src/domain01.py diagnose --baseline-artifacts artifacts/source_main
  python src/domain01.py adapt --baseline-artifacts artifacts/source_main

Large row-level artifacts and model files go to ``artifacts/domain_01``
(gitignored).  Compact, reviewable reports go to ``research/domain_01/results``.
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
from typing import Iterable

import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             roc_auc_score)

# ``python src/domain01.py`` is the repository's required one-command style.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (ARTIFACTS, CUTOFF_TEST, FOLD_WEIGHTS_S1, LGB_PARAMS, SEED,
                        VAL_FOLDS_S1, cutoff_grid)
from src.features import feature_names, make_xy, panel_users, to_np
from src.validation import calibrate


TEST_CUTOFF = CUTOFF_TEST
DOMAIN_CUTOFFS = list(VAL_FOLDS_S1) + [TEST_CUTOFF]
FOLD_NAMES = [d.isoformat() for d in VAL_FOLDS_S1]
SEED_FLOOR = 0.00712
CURRENT_MIX = {
    "S1-E10": 0.15,
    "S1-E02": 0.20,
    "S1-E03a": 0.10,
    "S1-DIST": 0.25,
    "SEQ-01-S42": 0.30,
}
DIST_MIX = {"S1-E10": 0.15, "S1-E02": 0.30, "S1-E03a": 0.10, "S1-DIST": 0.45}
RAW_DEPTH_FEATURES = [
    "tenure", "first_buy_age", "all_days_present", "all_days_buy", "all_orders",
    "all_searches", "gap_mean", "gap_std", "gap_max", "buygap_mean", "buygap_std",
    "rec_any", "rec_search", "rec_cart", "rec_buy", "rec_cat", "w365_days_present",
    "w365_days_buy", "w365_searches", "trend_pres_90_365",
]


def log(*items) -> None:
    print(*items, flush=True)


def json_dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
                    encoding="utf-8")


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


# --------------------------------------------------------------------------- pure helpers

def user_group_fold(user_ids, n_folds: int = 5) -> np.ndarray:
    """Stable user-level split; the same user is never in train and validation."""
    u = np.asarray(user_ids, dtype=np.uint64)
    # SplitMix64 finalizer: unlike uid % k, it does not inherit structure from IDs.
    x = u + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    x = x ^ (x >> np.uint64(31))
    return (x % np.uint64(n_folds)).astype(np.int8)


def density_ratio(prob_test, train_positive_prior) -> np.ndarray:
    """p_test(x)/p_hist(x) from domain posterior and the classifier prior."""
    p = np.clip(np.asarray(prob_test, float), 1e-5, 1.0 - 1e-5)
    prior = np.clip(np.asarray(train_positive_prior, float), 1e-5, 1.0 - 1e-5)
    return (p / (1.0 - p)) * ((1.0 - prior) / prior)


def clipped_importance_weights(ratio, temperature: float = 1.0, low: float = 0.25,
                               high: float = 4.0, normalize: bool = True) -> np.ndarray:
    if not (temperature > 0 and 0 < low <= high):
        raise ValueError("temperature/clip must be positive and low <= high")
    raw = np.power(np.maximum(np.asarray(ratio, float), 1e-12), temperature)
    if not normalize:
        return np.clip(raw, low, high)
    # Find one scale for which clipping and mean-one normalization both hold.
    # Post-hoc division would violate the requested clip when overlap is weak.
    left, right = 0.0, 1.0
    while float(np.mean(np.clip(raw * right, low, high))) < 1.0:
        right *= 2.0
    for _ in range(60):
        middle = (left + right) / 2.0
        if float(np.mean(np.clip(raw * middle, low, high))) < 1.0:
            left = middle
        else:
            right = middle
    return np.clip(raw * ((left + right) / 2.0), low, high)


def effective_sample_size(weights) -> float:
    w = np.asarray(weights, float)
    return float(w.sum() ** 2 / max(float(np.dot(w, w)), 1e-12))


def weighted_rmsle_z(y, z, weights) -> float:
    ly = np.log1p(np.asarray(y, float))
    pred = np.maximum(np.asarray(z, float), 0.0)
    w = np.asarray(weights, float)
    return float(np.sqrt(np.average((ly - pred) ** 2, weights=w)))


def weighted_calibrate(y, z, weights, iters: int = 30) -> tuple[float, float]:
    """Weighted counterpart of ``validation.calibrate`` without changing shared code."""
    ly = np.log1p(np.asarray(y, float))
    z = np.asarray(z, float)
    w = np.asarray(weights, float)
    d = float(np.average(ly - z, weights=w))
    for _ in range(iters):
        active = z + d > 0
        if not active.any():
            break
        new = float(np.average(ly[active] - z[active], weights=w[active]))
        if abs(new - d) < 1e-12:
            d = new
            break
        d = new
    return d, weighted_rmsle_z(y, np.maximum(z + d, 0.0), w)


def population_stability_index(hist, test, bins: int = 10) -> float:
    """Quantile-bin PSI with a separate missing-value bin."""
    a = np.asarray(hist, float)
    b = np.asarray(test, float)
    af, bf = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(af) == 0 or len(bf) == 0:
        return float("nan")
    cuts = np.unique(np.quantile(af, np.linspace(0, 1, bins + 1)[1:-1]))
    ca = np.bincount(np.searchsorted(cuts, af, side="right"), minlength=len(cuts) + 1)
    cb = np.bincount(np.searchsorted(cuts, bf, side="right"), minlength=len(cuts) + 1)
    ca = np.r_[ca, len(a) - len(af)].astype(float)
    cb = np.r_[cb, len(b) - len(bf)].astype(float)
    pa = np.maximum(ca / max(ca.sum(), 1.0), 1e-6)
    pb = np.maximum(cb / max(cb.sum(), 1.0), 1e-6)
    return float(np.sum((pb - pa) * np.log(pb / pa)))


def is_depth_proxy(name: str) -> bool:
    """Production columns that mainly encode recency/coverage/observation density."""
    exact = {"tenure_frac", "first_buy_frac", "gap_max_frac", "weekend_share"}
    prefixes = ("rec_", "gap_", "buygap_", "trend_pres_")
    tokens = ("days_present", "days_presence_only", "presence_rate", "ponly_share")
    return name in exact or name.startswith(prefixes) or any(t in name for t in tokens)


def domain_feature_sets(features: Iterable[str]) -> dict[str, list[str]]:
    features = list(features)
    forbidden = [f for f in features if any(t in f.lower() for t in ("user_id", "cutoff", "fold"))]
    if forbidden:
        raise ValueError(f"forbidden source features: {forbidden}")
    depth = [f for f in features if is_depth_proxy(f)]
    # Avoid the 365-day availability discontinuity in the behavioral ablation.
    behavior = [f for f in features if f not in depth and not f.startswith("w365")
                and not f.endswith("_365")]
    return {"all": features, "production_depth": depth, "behavioral": behavior}


def _normalize_within_cutoff(weights: np.ndarray, cutoffs: np.ndarray) -> np.ndarray:
    out = np.asarray(weights, float).copy()
    for cutoff in np.unique(cutoffs):
        m = cutoffs == cutoff
        out[m] /= max(float(out[m].mean()), 1e-12)
    return out


# --------------------------------------------------------------------------- domain data

def load_domain_matrix(L=None, norm_long: bool = True, selected: list[str] | None = None):
    """Load observation-level matrices through the exact production feature path."""
    sizes = [panel_users(cutoff, 3).height for cutoff in DOMAIN_CUTOFFS]
    total = sum(sizes)
    matrix = users = sources = labels = None
    names, start = None, 0
    for source, (cutoff, size) in enumerate(zip(DOMAIN_CUTOFFS, sizes)):
        X, _ = make_xy(cutoff, L, n_blocks=3, with_target=False, norm_long=norm_long)
        if X.height != size:
            raise AssertionError(f"panel size changed at {cutoff}: {X.height} != {size}")
        current = feature_names(X)
        if selected is not None:
            missing = sorted(set(selected) - set(current))
            if missing:
                raise ValueError(f"{cutoff}: missing diagnostic features {missing}")
            current = list(selected)
        if names is None:
            names = current
            matrix = np.empty((total, len(names)), np.float32)
            users = np.empty(total, np.int64)
            sources = np.empty(total, np.int8)
            labels = np.empty(total, np.int8)
        elif current != names:
            raise AssertionError(f"feature schema differs at {cutoff}")
        A = to_np(X, names)
        A[~np.isfinite(A)] = np.nan
        stop = start + size
        matrix[start:stop] = A
        users[start:stop] = X["user_id"].to_numpy()
        sources[start:stop] = source
        labels[start:stop] = int(cutoff == TEST_CUTOFF)
        start = stop
        log(f"  loaded {cutoff}: {X.height:,} rows x {len(names)} features")
        del X, A
    assert matrix is not None and users is not None and sources is not None and labels is not None
    return matrix, users, sources, labels, list(names or [])


def _sample_train_indices(mask: np.ndarray, sources: np.ndarray, cap: int,
                          seed: int) -> np.ndarray:
    idx = np.flatnonzero(mask)
    if len(idx) <= cap:
        return idx
    rng = np.random.RandomState(seed)
    picked = []
    counts = {int(s): int(np.sum(sources[idx] == s)) for s in np.unique(sources[idx])}
    remaining = cap
    ordered = sorted(counts)
    for j, source in enumerate(ordered):
        pool = idx[sources[idx] == source]
        take = remaining if j == len(ordered) - 1 else int(round(cap * len(pool) / len(idx)))
        take = min(take, len(pool))
        picked.append(rng.choice(pool, take, replace=False))
        remaining -= take
    return np.sort(np.concatenate(picked))


def _selected(A: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    return A[np.ix_(rows, cols)]


def _predict_chunks(model, X: np.ndarray, rows: np.ndarray, cols: np.ndarray,
                    chunk: int = 100_000) -> np.ndarray:
    out = np.empty(len(rows), np.float32)
    for start in range(0, len(rows), chunk):
        r = rows[start:start + chunk]
        out[start:start + len(r)] = model.predict(_selected(X, r, cols))
    return out


def calibration_table(y, p, bins: int = 10) -> pd.DataFrame:
    y, p = np.asarray(y, int), np.asarray(p, float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ids = np.minimum(np.searchsorted(edges, p, side="right") - 1, bins - 1)
    rows = []
    for i in range(bins):
        m = ids == i
        rows.append({"bin": i, "lo": edges[i], "hi": edges[i + 1], "n": int(m.sum()),
                     "mean_pred": float(p[m].mean()) if m.any() else np.nan,
                     "observed_rate": float(y[m].mean()) if m.any() else np.nan})
    return pd.DataFrame(rows)


def domain_metrics(y, p) -> dict:
    y, p = np.asarray(y, int), np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    cal = calibration_table(y, p)
    valid = cal[cal.n > 0]
    ece = float(np.sum(valid.n * np.abs(valid.mean_pred - valid.observed_rate)) / len(y))
    return {
        "n": int(len(y)), "positive_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)), "ece10": ece,
        "mean_probability": float(p.mean()),
    }


def _lgb_params() -> dict:
    p = dict(LGB_PARAMS)
    p.update(objective="binary", metric="auc", learning_rate=0.05, num_leaves=31,
             min_data_in_leaf=500, feature_fraction=0.8, bagging_fraction=0.8,
             bagging_freq=1, lambda_l2=10.0, seed=SEED, verbose=-1)
    return p


def fit_lgb_oof(name: str, X: np.ndarray, y: np.ndarray, users: np.ndarray,
                sources: np.ndarray, features: list[str], *, n_folds: int,
                rounds: int, train_cap: int, out_dir: Path,
                save_full: bool = False, permutation: bool = False,
                full_train_cap: int = 180_000) -> dict:
    import lightgbm as lgb

    groups = user_group_fold(users, n_folds)
    cols = np.arange(len(features), dtype=int)
    pred = np.empty(len(y), np.float32)
    priors = np.empty(len(y), np.float32)
    fold_rows, importance_rows, permutation_rows = [], [], []
    for fold in range(n_folds):
        valid = np.flatnonzero(groups == fold)
        train_mask = groups != fold
        train = _sample_train_indices(train_mask, sources, train_cap, SEED + fold)
        if np.intersect1d(np.unique(users[train]), np.unique(users[valid])).size:
            raise AssertionError("user leakage in domain split")
        prior = float(y[train].mean())
        A = _selected(X, train, cols)
        ds = lgb.Dataset(A, label=y[train], params=_lgb_params()).construct()
        del A
        model = lgb.train(_lgb_params(), ds, num_boost_round=rounds)
        pv = _predict_chunks(model, X, valid, cols)
        pred[valid], priors[valid] = pv, prior
        metrics = domain_metrics(y[valid], pv)
        fold_rows.append({"model": name, "group_fold": fold, "train_n": len(train),
                          "train_positive_rate": prior, **metrics})
        gain = model.feature_importance("gain")
        split = model.feature_importance("split")
        importance_rows.extend({"model": name, "group_fold": fold, "feature": feature,
                                "gain": float(g), "split": int(s)}
                               for feature, g, s in zip(features, gain, split))
        if permutation and fold == 0:
            rng = np.random.RandomState(SEED)
            take = rng.choice(valid, min(30_000, len(valid)), replace=False)
            top = np.argsort(gain)[::-1][:15]
            P = _selected(X, take, cols)
            base_auc = roc_auc_score(y[take], model.predict(P))
            for j in top:
                original = P[:, j].copy()
                rng.shuffle(P[:, j])
                auc = roc_auc_score(y[take], model.predict(P))
                permutation_rows.append({"model": name, "feature": features[j],
                                         "base_auc": base_auc, "permuted_auc": auc,
                                         "auc_drop": base_auc - auc})
                P[:, j] = original
            del P
        log(f"  {name} group-fold {fold}: AUC={metrics['roc_auc']:.6f}, "
            f"PR={metrics['pr_auc']:.6f}, train={len(train):,}, valid={len(valid):,}")
        del ds, model, pv
        gc.collect()

    imp = pd.DataFrame(importance_rows)
    agg = (imp.groupby(["model", "feature"], as_index=False)
           .agg(gain=("gain", "mean"), split=("split", "mean")))
    agg["gain_share"] = agg.gain / max(float(agg.gain.sum()), 1e-12)
    agg = agg.sort_values("gain", ascending=False)
    # Persist strict OOF before the optional full fit, whose memory peak is higher.
    np.savez_compressed(out_dir / f"checkpoint_{name}.npz", p=pred, prior=priors)
    pd.DataFrame(fold_rows).to_csv(out_dir / f"checkpoint_{name}_folds.csv", index=False)
    agg.to_csv(out_dir / f"checkpoint_{name}_importance.csv", index=False)

    full_model_path, full_prior = None, None
    if save_full:
        train = _sample_train_indices(np.ones(len(y), bool), sources,
                                      min(train_cap, full_train_cap), SEED + 100)
        full_prior = float(y[train].mean())
        A = _selected(X, train, cols)
        ds = lgb.Dataset(A, label=y[train], params=_lgb_params()).construct()
        del A
        model = lgb.train(_lgb_params(), ds, num_boost_round=rounds)
        full_model_path = out_dir / f"model_{name}.txt"
        model.save_model(str(full_model_path))
        del ds, model
        gc.collect()

    return {
        "name": name, "p": pred, "prior": priors,
        "metrics": domain_metrics(y, pred), "folds": pd.DataFrame(fold_rows),
        "importance": agg, "permutation": pd.DataFrame(permutation_rows),
        "full_model": full_model_path, "full_prior": full_prior,
    }


def fit_linear_oof(X: np.ndarray, y: np.ndarray, users: np.ndarray, sources: np.ndarray,
                   features: list[str], *, n_folds: int, train_cap: int) -> dict:
    groups = user_group_fold(users, n_folds)
    pred = np.empty(len(y), np.float32)
    priors = np.empty(len(y), np.float32)
    coefs, rows = [], []
    for fold in range(n_folds):
        valid = np.flatnonzero(groups == fold)
        train = _sample_train_indices(groups != fold, sources, train_cap, SEED + 300 + fold)
        A = X[train].astype(np.float32, copy=True)
        median = np.nanmedian(A, axis=0)
        median[~np.isfinite(median)] = 0.0
        bad = ~np.isfinite(A)
        A[bad] = median[np.where(bad)[1]]
        mean, scale = A.mean(axis=0), A.std(axis=0)
        scale[scale < 1e-6] = 1.0
        A = np.clip((A - mean) / scale, -20, 20).astype(np.float32)
        model = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4,
                              max_iter=200, tol=1e-4, average=True, random_state=SEED)
        model.fit(A, y[train])
        del A, bad
        pv = np.empty(len(valid), np.float32)
        for start in range(0, len(valid), 100_000):
            ridx = valid[start:start + 100_000]
            B = X[ridx].astype(np.float32, copy=True)
            bad = ~np.isfinite(B)
            B[bad] = median[np.where(bad)[1]]
            B = np.clip((B - mean) / scale, -20, 20).astype(np.float32)
            pv[start:start + len(ridx)] = model.predict_proba(B)[:, 1]
            del B, bad
        pred[valid] = pv
        priors[valid] = float(y[train].mean())
        met = domain_metrics(y[valid], pv)
        rows.append({"model": "D0-linear", "group_fold": fold, "train_n": len(train), **met})
        coefs.append(np.abs(model.coef_[0]))
        log(f"  D0-linear group-fold {fold}: AUC={met['roc_auc']:.6f}, PR={met['pr_auc']:.6f}")
        del model, pv
        gc.collect()
    importance = pd.DataFrame({"model": "D0-linear", "feature": features,
                               "abs_standardized_coef": np.mean(coefs, axis=0)})
    importance = importance.sort_values("abs_standardized_coef", ascending=False)
    return {"name": "D0-linear", "p": pred, "prior": priors,
            "metrics": domain_metrics(y, pred), "folds": pd.DataFrame(rows),
            "importance": importance}


# --------------------------------------------------------------------------- shift reports

def shift_report(X: np.ndarray, y: np.ndarray, features: list[str], importance: pd.DataFrame,
                 sample_n: int = 100_000) -> pd.DataFrame:
    from scipy.stats import ks_2samp

    rng = np.random.RandomState(SEED)
    hist = np.flatnonzero(y == 0)
    test = np.flatnonzero(y == 1)
    hist = rng.choice(hist, min(sample_n, len(hist)), replace=False)
    test = rng.choice(test, min(sample_n, len(test)), replace=False)
    rows = []
    for j, feature in enumerate(features):
        a, b = X[hist, j].astype(float), X[test, j].astype(float)
        af, bf = a[np.isfinite(a)], b[np.isfinite(b)]
        ma = float(np.mean(af)) if len(af) else np.nan
        mb = float(np.mean(bf)) if len(bf) else np.nan
        va = float(np.var(af)) if len(af) else np.nan
        vb = float(np.var(bf)) if len(bf) else np.nan
        denom = math.sqrt(max((va + vb) / 2, 1e-12)) if np.isfinite(va + vb) else np.nan
        qa = np.quantile(af, [0.1, 0.25, 0.5, 0.75, 0.9]) if len(af) else [np.nan] * 5
        qb = np.quantile(bf, [0.1, 0.25, 0.5, 0.75, 0.9]) if len(bf) else [np.nan] * 5
        rows.append({
            "feature": feature, "hist_mean": ma, "test_mean": mb,
            "smd": (mb - ma) / denom if denom and np.isfinite(denom) else np.nan,
            "ks": float(ks_2samp(af, bf).statistic) if len(af) and len(bf) else np.nan,
            "psi": population_stability_index(a, b),
            "hist_missing": float(1 - len(af) / len(a)),
            "test_missing": float(1 - len(bf) / len(b)),
            "hist_q10": qa[0], "test_q10": qb[0], "hist_q50": qa[2], "test_q50": qb[2],
            "hist_q90": qa[4], "test_q90": qb[4],
            "median_delta_iqr": (qb[2] - qa[2]) / max(float(qa[3] - qa[1]), 1e-9),
        })
    out = pd.DataFrame(rows).merge(importance.drop(columns=["model"], errors="ignore"),
                                   on="feature", how="left")
    return out.sort_values(["gain", "psi"], ascending=False, na_position="last")


def _load_aligned_oof(path: Path, users: np.ndarray, cutoffs: np.ndarray,
                      reference_y: np.ndarray | None = None):
    d = np.load(path, allow_pickle=False)
    if np.array_equal(d["user_id"], users) and np.array_equal(d["cutoff"], cutoffs):
        order = np.arange(len(users))
    else:
        order = np.lexsort((d["user_id"], d["cutoff"]))
        expected = np.lexsort((users, cutoffs))
        if not (np.array_equal(d["user_id"][order], users[expected])
                and np.array_equal(d["cutoff"][order], cutoffs[expected])):
            raise AssertionError(f"OOF keys do not align: {path}")
        # Callers use the production chronological/user-sorted order.
        inverse_expected = np.empty(len(expected), int)
        inverse_expected[expected] = np.arange(len(expected))
        order = order[inverse_expected]
    aligned = {name: d[name][order] for name in d.files}
    if reference_y is not None and not np.allclose(aligned["y"], reference_y, rtol=0, atol=1e-6):
        raise AssertionError(f"OOF targets do not align: {path}")
    return aligned


def load_production_predictions(base_dir: Path, users: np.ndarray, cutoffs: np.ndarray):
    needed = ["S1-E10", "S1-E02", "S1-E03a", "S1-DIST", "S1-ROUNDS",
              "S1-SEEDAVG3", "SEQ-01-S42"]
    out, y = {}, None
    for name in needed:
        d = _load_aligned_oof(base_dir / f"oof_{name}.npz", users, cutoffs, y)
        if y is None:
            y = d["y"].astype(float)
        out[name] = d["z"].astype(float)
    out["S1-DIST-MIX"] = sum(w * out[name] for name, w in DIST_MIX.items())
    out["SEQ-01-MIX"] = sum(w * out[name] for name, w in CURRENT_MIX.items())
    out["SEQ-MIX-ROUNDS-CONTROL"] = (out["SEQ-01-MIX"]
                                     + CURRENT_MIX["S1-E10"] * (out["S1-ROUNDS"] - out["S1-E10"]))
    return out, np.asarray(y, float)


def weighted_wcv(y, z, cutoffs, row_weights) -> tuple[float, list[float], list[float]]:
    scores, offsets = [], []
    for cutoff in FOLD_NAMES:
        m = cutoffs == cutoff
        offset, score = weighted_calibrate(y[m], z[m], row_weights[m])
        scores.append(score)
        offsets.append(offset)
    return float(np.average(scores, weights=FOLD_WEIGHTS_S1)), scores, offsets


def weight_schemes(p_primary, prior_primary, p_behavior, prior_behavior, cutoffs):
    specs = {
        "primary_t05_clip4": (p_primary, prior_primary, 0.5, 0.25, 4.0),
        "primary_t10_clip4": (p_primary, prior_primary, 1.0, 0.25, 4.0),
        "primary_t10_clip10": (p_primary, prior_primary, 1.0, 0.10, 10.0),
        "behavior_t05_clip4": (p_behavior, prior_behavior, 0.5, 0.25, 4.0),
        "behavior_t10_clip4": (p_behavior, prior_behavior, 1.0, 0.25, 4.0),
    }
    out = {"ordinary": np.ones(len(cutoffs), float)}
    for name, (p, prior, temp, low, high) in specs.items():
        ratio = density_ratio(p, prior)
        out[name] = np.empty(len(cutoffs), float)
        for cutoff in np.unique(cutoffs):
            m = cutoffs == cutoff
            out[name][m] = clipped_importance_weights(ratio[m], temp, low, high)
    return out


def model_ranking_report(models: dict[str, np.ndarray], y, cutoffs, schemes: dict[str, np.ndarray]):
    rows, fold_rows = [], []
    for scheme, weights in schemes.items():
        for name, z in models.items():
            score, per, offsets = weighted_wcv(y, z, cutoffs, weights)
            rows.append({"scheme": scheme, "model": name, "weighted_wcv": score})
            fold_rows.extend({"scheme": scheme, "model": name, "cutoff": cutoff,
                              "rmsle_cal": sc, "offset": off}
                             for cutoff, sc, off in zip(FOLD_NAMES, per, offsets))
    table = pd.DataFrame(rows)
    table["rank"] = table.groupby("scheme").weighted_wcv.rank(method="min")
    ordinary = table[table.scheme == "ordinary"][["model", "weighted_wcv"]]
    ordinary = ordinary.rename(columns={"weighted_wcv": "ordinary_wcv"})
    table = table.merge(ordinary, on="model", how="left")
    table["delta_vs_ordinary"] = table.weighted_wcv - table.ordinary_wcv
    return table.sort_values(["scheme", "rank", "model"]), pd.DataFrame(fold_rows)


def _feature_column(X: np.ndarray, features: list[str], name: str, hist: np.ndarray):
    return X[hist, features.index(name)].astype(float) if name in features else None


def cutoff_diagnostics(feature_values: dict[str, np.ndarray], hist, hist_cutoffs, p,
                       models, y) -> pd.DataFrame:
    prod = models["SEQ-01-MIX"]
    rows = []
    for cutoff in FOLD_NAMES:
        local = hist_cutoffs == cutoff
        absolute = hist[local]
        _, sc = weighted_calibrate(y[local], prod[local], np.ones(local.sum()))
        rec = feature_values.get("rec_buy")
        buy = feature_values.get("w180_days_buy")
        present = feature_values.get("w30_days_present")
        tenure = feature_values.get("tenure_frac")
        rec = rec[absolute] if rec is not None else None
        buy = buy[absolute] if buy is not None else None
        present = present[absolute] if present is not None else None
        tenure = tenure[absolute] if tenure is not None else None
        rows.append({
            "cutoff": cutoff, "n": int(local.sum()), "mean_p_test_like": float(p[local].mean()),
            "median_p_test_like": float(np.median(p[local])),
            "q90_p_test_like": float(np.quantile(p[local], .9)),
            "production_rmsle_cal": sc, "target_zero_rate": float(np.mean(y[local] == 0)),
            "rec_buy_median": float(np.nanmedian(rec)) if rec is not None else np.nan,
            "inactive_or_no_buy_rate": float(np.mean(~np.isfinite(rec) | (rec >= 90))) if rec is not None else np.nan,
            "zero_buy_days_180_rate": float(np.mean(np.nan_to_num(buy) == 0)) if buy is not None else np.nan,
            "mean_days_present_30": float(np.nanmean(present)) if present is not None else np.nan,
            "mean_tenure_frac": float(np.nanmean(tenure)) if tenure is not None else np.nan,
        })
    return pd.DataFrame(rows)


def load_lgb_checkpoint(name: str, y: np.ndarray, sources: np.ndarray, features: list[str],
                        out: Path, report_dir: Path, full_cap: int = 180_000) -> dict:
    """Resume an identical diagnostic run after a later reporting/memory failure."""
    checkpoint = np.load(out / f"checkpoint_{name}.npz", allow_pickle=False)
    folds = pd.read_csv(out / f"checkpoint_{name}_folds.csv")
    importance = pd.read_csv(out / f"checkpoint_{name}_importance.csv")
    permutation_path = report_dir / "domain_permutation_importance.csv"
    permutation = (pd.read_csv(permutation_path)
                   if name == "D1-production" and permutation_path.exists() else pd.DataFrame())
    full_model = out / f"model_{name}.txt"
    if not full_model.exists():
        full_model = None
        full_prior = None
    else:
        idx = _sample_train_indices(np.ones(len(y), bool), sources, full_cap, SEED + 100)
        full_prior = float(y[idx].mean())
    log(f"  resumed {name}: AUC={domain_metrics(y, checkpoint['p'])['roc_auc']:.6f}")
    return {"name": name, "p": checkpoint["p"], "prior": checkpoint["prior"],
            "metrics": domain_metrics(y, checkpoint["p"]), "folds": folds,
            "importance": importance, "permutation": permutation,
            "full_model": full_model, "full_prior": full_prior}


def error_relationships(models, y, cutoffs, p) -> tuple[pd.DataFrame, pd.DataFrame]:
    from scipy.stats import spearmanr

    ly = np.log1p(y)
    calibrated = {}
    for name, z in models.items():
        zc = np.empty_like(z)
        for cutoff in FOLD_NAMES:
            m = cutoffs == cutoff
            off, _ = calibrate(y[m], z[m])
            zc[m] = np.maximum(z[m] + off, 0.0)
        calibrated[name] = zc
    base_loss = (ly - calibrated["SEQ-01-MIX"]) ** 2
    rows = [{"model": "SEQ-01-MIX", "quantity": "squared_error",
             "spearman_with_p_test_like": float(spearmanr(p, base_loss).statistic)}]
    for name, zc in calibrated.items():
        if name == "SEQ-01-MIX":
            continue
        delta = (ly - zc) ** 2 - base_loss
        rows.append({"model": name, "quantity": "loss_delta_vs_production",
                     "spearman_with_p_test_like": float(spearmanr(p, delta).statistic),
                     "mean_loss_delta": float(delta.mean())})
    order = np.argsort(p, kind="stable")
    decile = np.empty(len(p), np.int8)
    decile[order] = np.minimum(np.arange(len(p)) * 10 // len(p), 9)
    bins = []
    for d in range(10):
        m = decile == d
        row = {"p_decile": d + 1, "n": int(m.sum()), "mean_p": float(p[m].mean()),
               "production_rmsle": float(np.sqrt(base_loss[m].mean()))}
        for name, zc in calibrated.items():
            if name != "SEQ-01-MIX":
                row[f"delta_{name}"] = float(np.sqrt(np.mean((ly[m] - zc[m]) ** 2))
                                               - row["production_rmsle"])
        bins.append(row)
    return pd.DataFrame(rows), pd.DataFrame(bins)


# --------------------------------------------------------------------------- diagnose command

def diagnose(args) -> None:
    started = time.time()
    out = Path(args.out)
    report_dir = Path(args.report_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    log("DOMAIN-01 diagnose: loading primary production representation")
    X, users, sources, domain_y, features = load_domain_matrix(L=None, norm_long=True)
    sets = domain_feature_sets(features)
    groups = user_group_fold(users, args.folds)
    audit = {
        "n_rows": len(domain_y), "n_users": int(len(np.unique(users))),
        "n_historical": int(np.sum(domain_y == 0)), "n_test": int(np.sum(domain_y == 1)),
        "positive_rate": float(domain_y.mean()), "n_features": len(features),
        "group_folds": args.folds, "group_fold_sizes": np.bincount(groups, minlength=args.folds),
        "forbidden_columns_present": [], "feature_path": "make_xy -> build_features",
        "production_representation": {"L": None, "norm_long": True, "panel_blocks": 3},
        "explicit_cutoff_or_fold_feature": False, "user_id_used_as_feature": False,
    }
    json_dump(out / "leakage_audit.json", audit)

    d0 = fit_linear_oof(X, domain_y, users, sources, features, n_folds=args.folds,
                        train_cap=args.linear_cap)
    behavior_cols = np.array([features.index(f) for f in sets["behavioral"]], int)
    if args.resume and (out / "checkpoint_D1-production.npz").exists():
        d1 = load_lgb_checkpoint("D1-production", domain_y, sources, features, out, report_dir)
    else:
        d1 = fit_lgb_oof("D1-production", X, domain_y, users, sources, features,
                         n_folds=args.folds, rounds=args.rounds, train_cap=args.train_cap,
                         out_dir=out, save_full=True, permutation=True)
    if args.resume and (out / "checkpoint_D1-behavior.npz").exists():
        behavior = load_lgb_checkpoint("D1-behavior", domain_y, sources,
                                       sets["behavioral"], out, report_dir)
    else:
        behavior = fit_lgb_oof("D1-behavior", X[:, behavior_cols], domain_y, users, sources,
                               sets["behavioral"], n_folds=args.folds,
                               rounds=args.ablation_rounds, train_cap=args.ablation_cap,
                               out_dir=out, save_full=True)
    if args.resume and (out / "checkpoint_D1-missingness.npz").exists():
        missing = load_lgb_checkpoint("D1-missingness", domain_y, sources,
                                      [f"missing__{f}" for f in features], out, report_dir)
    else:
        missing = fit_lgb_oof("D1-missingness", np.isnan(X).astype(np.float32), domain_y,
                              users, sources, [f"missing__{f}" for f in features],
                              n_folds=args.folds, rounds=max(30, args.ablation_rounds // 2),
                              train_cap=args.ablation_cap, out_dir=out)

    shift = shift_report(X, domain_y, features, d1["importance"])
    shift.to_csv(report_dir / "feature_shift_report.csv", index=False)
    d1["importance"].to_csv(report_dir / "domain_feature_importance.csv", index=False)
    d1["permutation"].to_csv(report_dir / "domain_permutation_importance.csv", index=False)
    d0["importance"].to_csv(report_dir / "linear_feature_importance.csv", index=False)
    segment_names = ["rec_buy", "w180_days_buy", "w30_days_present", "tenure_frac"]
    segment_values = {name: X[:, features.index(name)].copy()
                      for name in segment_names if name in features}
    del X
    gc.collect()

    # Raw user-level history depth: diagnostic only, never used for primary p_test_like.
    log("DOMAIN-01 diagnose: raw history-depth ablation")
    if args.resume and (out / "checkpoint_D1-depth-raw.npz").exists():
        depth_features = list(RAW_DEPTH_FEATURES)
        depth = load_lgb_checkpoint("D1-depth-raw", domain_y, sources, depth_features,
                                    out, report_dir)
    else:
        Xdepth, u2, s2, y2, depth_features = load_domain_matrix(
            L=None, norm_long=False, selected=RAW_DEPTH_FEATURES)
        if not (np.array_equal(users, u2) and np.array_equal(sources, s2)
                and np.array_equal(domain_y, y2)):
            raise AssertionError("raw-depth representation rows differ")
        depth = fit_lgb_oof("D1-depth-raw", Xdepth, domain_y, users, sources, depth_features,
                            n_folds=args.folds, rounds=args.ablation_rounds,
                            train_cap=args.ablation_cap, out_dir=out)
        del Xdepth, u2, s2, y2
    gc.collect()

    # Fixed L=180 removes available-history depth while retaining all bounded behavior.
    log("DOMAIN-01 diagnose: fixed-L180 ablation")
    fixed_checkpoint = out / "checkpoint_D1-fixed-L180.npz"
    if args.resume and fixed_checkpoint.exists():
        importance180 = pd.read_csv(out / "checkpoint_D1-fixed-L180_importance.csv")
        features180 = importance180.feature.tolist()
        fixed = load_lgb_checkpoint("D1-fixed-L180", domain_y, sources, features180,
                                    out, report_dir)
    else:
        X180, u3, s3, y3, features180 = load_domain_matrix(L=180, norm_long=False)
        if not (np.array_equal(users, u3) and np.array_equal(sources, s3)
                and np.array_equal(domain_y, y3)):
            raise AssertionError("L180 representation rows differ")
        fixed = fit_lgb_oof("D1-fixed-L180", X180, domain_y, users, sources, features180,
                            n_folds=args.folds, rounds=args.ablation_rounds,
                            train_cap=args.ablation_cap, out_dir=out)
        del X180, u3, s3, y3
    gc.collect()

    classifiers = [d0, d1, behavior, depth, fixed, missing]
    metrics_rows, fold_frames = [], []
    for result in classifiers:
        metrics_rows.append({"model": result["name"], **result["metrics"]})
        fold_frames.append(result["folds"])
        calibration_table(domain_y, result["p"]).assign(model=result["name"]).to_csv(
            report_dir / f"calibration_{result['name']}.csv", index=False)
    metrics = pd.DataFrame(metrics_rows).sort_values("roc_auc", ascending=False)
    metrics.to_csv(report_dir / "domain_classifier_metrics.csv", index=False)
    pd.concat(fold_frames, ignore_index=True).to_csv(report_dir / "domain_classifier_folds.csv",
                                                     index=False)

    source_dates = np.array([d.isoformat() for d in DOMAIN_CUTOFFS], dtype="U10")[sources]
    hist = np.flatnonzero(domain_y == 0)
    test = np.flatnonzero(domain_y == 1)
    hist_cutoffs = source_dates[hist]
    schemes = weight_schemes(d1["p"][hist], d1["prior"][hist],
                             behavior["p"][hist], behavior["prior"][hist], hist_cutoffs)
    weight_rows = []
    for name, w in schemes.items():
        for cutoff in FOLD_NAMES + ["ALL"]:
            m = np.ones(len(w), bool) if cutoff == "ALL" else hist_cutoffs == cutoff
            weight_rows.append({"scheme": name, "cutoff": cutoff, "n": int(m.sum()),
                                "mean": float(w[m].mean()), "std": float(w[m].std()),
                                "min": float(w[m].min()), "max": float(w[m].max()),
                                "neff": effective_sample_size(w[m]),
                                "neff_fraction": effective_sample_size(w[m]) / m.sum()})
    pd.DataFrame(weight_rows).to_csv(report_dir / "importance_weight_diagnostics.csv", index=False)

    models, target_y = load_production_predictions(Path(args.baseline_artifacts), users[hist],
                                                    hist_cutoffs)
    ranking, ranking_folds = model_ranking_report(models, target_y, hist_cutoffs, schemes)
    ranking.to_csv(report_dir / "weighted_cv_comparison.csv", index=False)
    ranking_folds.to_csv(report_dir / "weighted_cv_folds.csv", index=False)
    cutoff_diag = cutoff_diagnostics(segment_values, hist, hist_cutoffs, d1["p"][hist],
                                     models, target_y)
    cutoff_diag.to_csv(report_dir / "cutoff_test_likeness.csv", index=False)
    relation, deciles = error_relationships(models, target_y, hist_cutoffs, d1["p"][hist])
    relation.to_csv(report_dir / "error_test_likeness_correlations.csv", index=False)
    deciles.to_csv(report_dir / "test_likeness_deciles.csv", index=False)

    ratio_primary = density_ratio(d1["p"], d1["prior"])
    ratio_behavior = density_ratio(behavior["p"], behavior["prior"])
    all_frame = pl.DataFrame({
        "user_id": users, "source_cutoff": source_dates, "domain_test": domain_y,
        "group_fold": groups, "p_d0_linear": d0["p"], "p_test_like": d1["p"],
        "p_behavior": behavior["p"], "p_depth_raw": depth["p"],
        "p_fixed_l180": fixed["p"], "p_missingness": missing["p"],
        "density_ratio_primary": ratio_primary.astype(np.float32),
        "density_ratio_behavior": ratio_behavior.astype(np.float32),
    })
    all_frame.write_parquet(out / "domain_oof_probabilities.parquet", compression="zstd")
    hist_payload = all_frame.filter(pl.col("domain_test") == 0)
    for name, w in schemes.items():
        if name != "ordinary":
            hist_payload = hist_payload.with_columns(pl.Series(f"weight_{name}", w.astype(np.float32)))
    hist_payload.write_parquet(out / "historical_test_likeness.parquet", compression="zstd")
    all_frame.filter(pl.col("domain_test") == 1).write_parquet(
        out / "test_domain_probabilities.parquet", compression="zstd")

    meta = {
        "experiment": "DOMAIN-01", "seed": SEED, "runtime_s": time.time() - started,
        "features": features, "behavior_features": sets["behavioral"],
        "depth_features": depth_features, "fixed_l180_features": features180,
        "models": {r["name"]: r["metrics"] for r in classifiers},
        "full_models": {
            "primary": {"path": str(d1["full_model"]), "prior": d1["full_prior"],
                        "features": features},
            "behavior": {"path": str(behavior["full_model"]), "prior": behavior["full_prior"],
                         "features": sets["behavioral"]},
        },
        "primary_probability": "strict user-group OOF D1-production",
        "weighting": "odds/prior_odds; clipped and normalized within validation cutoff",
        "baseline_artifacts": str(Path(args.baseline_artifacts).resolve()),
    }
    json_dump(out / "domain_01_metrics.json", meta)
    # Compact copy is tracked; the large artifact directory remains gitignored.
    json_dump(report_dir / "domain_01_summary.json", meta)
    log(f"DOMAIN-01 diagnose complete in {time.time() - started:.0f}s")
    log(metrics.to_string(index=False))
    log("\nCutoff test-likeness:\n", cutoff_diag.to_string(index=False))
    log("\nWeighted ranking:\n", ranking.to_string(index=False))


# --------------------------------------------------------------------------- adaptation command

def _load_domain_model(out: Path, model_name: str):
    import lightgbm as lgb

    meta = json.loads((out / "domain_01_metrics.json").read_text(encoding="utf-8"))
    info = meta["full_models"][model_name]
    path = Path(info["path"])
    if not path.is_absolute():
        path = out / path.name
    return lgb.Booster(model_file=str(path)), list(info["features"]), float(info["prior"])


def _frame_matrix(frame: pl.DataFrame, features: list[str]) -> np.ndarray:
    """Float32 matrix without ``features.to_np``'s unconditional second copy."""
    return frame.select(features).to_numpy().astype(np.float32, copy=False)


def _predict_frame(model, frame: pl.DataFrame, features: list[str], chunk: int = 25_000):
    pred = np.empty(frame.height, np.float32)
    for start in range(0, frame.height, chunk):
        block = _frame_matrix(frame.slice(start, chunk), features)
        pred[start:start + len(block)] = model.predict(block)
        del block
    return pred


def _assemble_adapt(cuts: list[dt.date], all_features: list[str], domain_model,
                    domain_features: list[str], prior: float, temperature: float,
                    clip_low: float, clip_high: float, matrix_path: Path):
    sizes = [panel_users(cutoff, 1).height for cutoff in cuts]
    total = sum(sizes)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    if matrix_path.exists():
        matrix_path.unlink()
    Xtr = np.memmap(matrix_path, mode="w+", dtype=np.float32,
                    shape=(total, len(all_features)))
    ytr = np.empty(total, np.float32)
    p = np.empty(total, np.float32)
    cut = np.empty(total, dtype="U10")
    start = 0
    for cutoff, size in zip(cuts, sizes):
        X, y = make_xy(cutoff, None, n_blocks=1, with_target=True, norm_long=True)
        if feature_names(X) != all_features:
            raise AssertionError(f"feature schema changed at {cutoff}")
        if len(y) != size:
            raise AssertionError(f"panel size changed at {cutoff}: {len(y)} != {size}")
        pdomain = _predict_frame(domain_model, X, domain_features)
        stop = start + size
        for row_start in range(0, size, 25_000):
            A = _frame_matrix(X.slice(row_start, 25_000), all_features)
            A[~np.isfinite(A)] = np.nan
            n = len(A)
            Xtr[start + row_start:start + row_start + n] = A
            del A
        ytr[start:stop] = y
        p[start:stop] = pdomain
        cut[start:stop] = cutoff.isoformat()
        start = stop
        log(f"    train cutoff {cutoff}: {len(y):,} rows, mean p={np.mean(pdomain):.6f}")
        del X, pdomain
    Xtr.flush()
    ratio = density_ratio(p, prior)
    weights = clipped_importance_weights(ratio, temperature, clip_low, clip_high)
    rows = []
    for cutoff in sorted(set(cut.tolist())) + ["ALL"]:
        m = np.ones(len(cut), bool) if cutoff == "ALL" else cut == cutoff
        rows.append({"train_cutoff": cutoff, "n": int(m.sum()),
                     "mean_p_test_like": float(p[m].mean()),
                     "mean_weight": float(weights[m].mean()),
                     "neff_fraction": effective_sample_size(weights[m]) / m.sum()})
    return Xtr, ytr, weights.astype(np.float32), pd.DataFrame(rows)


def _adapt_weight_diagnostics(cuts: list[dt.date], domain_model, domain_features: list[str],
                              prior: float, temperature: float, clip_low: float,
                              clip_high: float) -> pd.DataFrame:
    """Rebuild weight summaries without allocating the production train matrix."""
    probs, sizes = [], []
    for cutoff in cuts:
        X, _ = make_xy(cutoff, None, n_blocks=1, with_target=False, norm_long=True)
        probs.append(_predict_frame(domain_model, X, domain_features))
        sizes.append(X.height)
        del X
    p = np.concatenate(probs)
    weights = clipped_importance_weights(density_ratio(p, prior), temperature,
                                         clip_low, clip_high)
    rows, start = [], 0
    for cutoff, size in zip(cuts, sizes):
        sl = slice(start, start + size)
        rows.append({"train_cutoff": cutoff.isoformat(), "n": size,
                     "mean_p_test_like": float(p[sl].mean()),
                     "mean_weight": float(weights[sl].mean()),
                     "neff_fraction": effective_sample_size(weights[sl]) / size})
        start += size
    rows.append({"train_cutoff": "ALL", "n": len(p),
                 "mean_p_test_like": float(p.mean()), "mean_weight": float(weights.mean()),
                 "neff_fraction": effective_sample_size(weights) / len(weights)})
    return pd.DataFrame(rows)


def _fold_calibrated_predictions(y, z, cutoffs):
    out = np.empty(len(z), float)
    for cutoff in FOLD_NAMES:
        m = cutoffs == cutoff
        d, _ = calibrate(y[m], z[m])
        out[m] = np.maximum(z[m] + d, 0.0)
    return out


def _lofo_replacement(y, base, delta, cutoffs, max_weight: float = 0.15):
    grid = np.round(np.arange(0.0, max_weight + 1e-9, 0.025), 3)
    chosen, heldout_rows, heldout_scores = [], [], []
    for heldout, cutoff in enumerate(FOLD_NAMES):
        train_folds = [i for i in range(4) if i != heldout]
        scores = []
        for weight in grid:
            per = []
            for i in train_folds:
                m = cutoffs == FOLD_NAMES[i]
                per.append(weighted_calibrate(y[m], base[m] + weight * delta[m],
                                              np.ones(m.sum()))[1])
            scores.append(np.average(per, weights=np.asarray(FOLD_WEIGHTS_S1)[train_folds]))
        best = int(np.argmin(scores))
        weight = float(grid[best])
        m = cutoffs == cutoff
        score = weighted_calibrate(y[m], base[m] + weight * delta[m], np.ones(m.sum()))[1]
        base_score = weighted_calibrate(y[m], base[m], np.ones(m.sum()))[1]
        chosen.append(weight); heldout_scores.append(score)
        heldout_rows.append({"heldout_cutoff": cutoff, "selected_weight": weight,
                             "heldout_rmsle": score, "base_rmsle": base_score,
                             "delta": score - base_score})
    base_per = [r["base_rmsle"] for r in heldout_rows]
    return {
        "selected_weights": chosen,
        "lofo_wcv": float(np.average(heldout_scores, weights=FOLD_WEIGHTS_S1)),
        "base_wcv": float(np.average(base_per, weights=FOLD_WEIGHTS_S1)),
        "delta_wcv": float(np.average(heldout_scores, weights=FOLD_WEIGHTS_S1)
                           - np.average(base_per, weights=FOLD_WEIGHTS_S1)),
        "rows": pd.DataFrame(heldout_rows),
    }


def adaptation_reports(out: Path, report_dir: Path, baseline_artifacts: Path,
                       weight_scheme: str) -> dict:
    d = np.load(out / "oof_DOMAIN-01-DIRECT.npz", allow_pickle=False)
    users, cutoffs, z_domain, y = d["user_id"], d["cutoff"], d["z"].astype(float), d["y"].astype(float)
    models, ref_y = load_production_predictions(baseline_artifacts, users, cutoffs)
    if not np.allclose(y, ref_y, atol=1e-6, rtol=0):
        raise AssertionError("adaptation targets differ from production OOF")
    hist_like = pl.read_parquet(out / "historical_test_likeness.parquet")
    if not (np.array_equal(hist_like["user_id"].to_numpy(), users)
            and np.array_equal(hist_like["source_cutoff"].to_numpy(), cutoffs)):
        raise AssertionError("test-likeness rows do not align with adaptation OOF")
    ordinary = np.ones(len(y))
    testlike = hist_like[f"weight_{weight_scheme}"].to_numpy()
    base = models["S1-ROUNDS"]
    rows = []
    for scheme, w in [("ordinary", ordinary), (weight_scheme, testlike)]:
        for name, z in [("S1-ROUNDS", base), ("DOMAIN-01-DIRECT", z_domain)]:
            score, per, _ = weighted_wcv(y, z, cutoffs, w)
            rows.append({"scheme": scheme, "model": name, "wcv": score,
                         "folds": per})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(report_dir / "adaptation_metrics.csv", index=False)

    zprod = models["SEQ-01-MIX"]
    zcontrol = models["SEQ-MIX-ROUNDS-CONTROL"]
    slot_weight = CURRENT_MIX["S1-E10"]
    zcandidate = zcontrol + slot_weight * (z_domain - base)
    mix_rows = []
    for scheme, w in [("ordinary", ordinary), (weight_scheme, testlike)]:
        for name, z in [("SEQ-01-MIX", zprod), ("ROUNDS-slot-control", zcontrol),
                        ("DOMAIN-full-slot", zcandidate)]:
            score, per, _ = weighted_wcv(y, z, cutoffs, w)
            mix_rows.append({"scheme": scheme, "model": name, "wcv": score, "folds": per})
    pd.DataFrame(mix_rows).to_csv(report_dir / "adaptation_mix.csv", index=False)

    ly = np.log1p(y)
    zdc = _fold_calibrated_predictions(y, z_domain, cutoffs)
    zbc = _fold_calibrated_predictions(y, base, cutoffs)
    zpc = _fold_calibrated_predictions(y, zprod, cutoffs)
    var_delta = float(np.var(z_domain - base))
    diversity = pd.DataFrame([{
        "comparison": "DOMAIN-01-DIRECT vs S1-ROUNDS",
        "var_prediction_delta": var_delta, "seed_floor": SEED_FLOOR,
        "seed_floor_ratio": var_delta / SEED_FLOOR,
        "residual_corr_with_same_recipe_base": float(np.corrcoef(ly - zdc, ly - zbc)[0, 1]),
        "residual_corr_with_production": float(np.corrcoef(ly - zdc, ly - zpc)[0, 1]),
    }])
    diversity.to_csv(report_dir / "adaptation_diversity.csv", index=False)

    lofo = _lofo_replacement(y, zcontrol, z_domain - base, cutoffs, max_weight=slot_weight)
    lofo["rows"].to_csv(report_dir / "adaptation_lofo.csv", index=False)
    summary = {
        "standalone": rows, "mix": mix_rows, "diversity": diversity.iloc[0].to_dict(),
        "lofo": {k: v for k, v in lofo.items() if k != "rows"},
        "weight_scheme": weight_scheme,
    }
    json_dump(out / "adaptation_metrics.json", summary)
    json_dump(report_dir / "adaptation_summary.json", summary)
    return summary


def adapt(args) -> None:
    import lightgbm as lgb
    from src import models as production_models

    started = time.time()
    out, report_dir = Path(args.out), Path(args.report_dir)
    out.mkdir(parents=True, exist_ok=True); report_dir.mkdir(parents=True, exist_ok=True)
    model, domain_features, prior = _load_domain_model(out, args.domain_model)
    all_features = feature_names(make_xy(VAL_FOLDS_S1[-1], None, n_blocks=3,
                                         with_target=False, norm_long=True)[0])
    if not set(domain_features).issubset(all_features):
        raise AssertionError("domain model expects unavailable production features")
    oof_u, oof_c, oof_z, oof_y, weight_frames = [], [], [], [], []
    params = {"seed": SEED}
    for V in VAL_FOLDS_S1:
        part_path = out / f"oof_DOMAIN-01-DIRECT_{V.strftime('%Y%m%d')}.npz"
        model_path = out / f"model_DOMAIN-01-DIRECT_{V.strftime('%Y%m%d')}.txt"
        weight_path = out / f"adapt_weights_{V.strftime('%Y%m%d')}.csv"
        if args.resume and part_path.exists() and not weight_path.exists():
            cuts = [T for T in cutoff_grid(90, 7) if T + dt.timedelta(days=30) <= V]
            diag = _adapt_weight_diagnostics(cuts, model, domain_features, prior,
                                             args.temperature, args.clip_low, args.clip_high)
            diag.insert(0, "validation_cutoff", V.isoformat())
            diag.to_csv(weight_path, index=False)
        if args.resume and part_path.exists():
            part = np.load(part_path, allow_pickle=False)
            oof_u.append(part["user_id"]); oof_c.append(part["cutoff"])
            oof_z.append(part["z"]); oof_y.append(part["y"])
            log(f"DOMAIN-01 adapt fold {V}: resumed saved OOF part")
            continue
        if args.resume and model_path.exists():
            fitted = lgb.Booster(model_file=str(model_path))
            Xv, yv = make_xy(V, None, n_blocks=3, with_target=True, norm_long=True)
            z = np.maximum(_predict_frame(fitted, Xv, all_features), 0.0)
            uid = Xv["user_id"].to_numpy()
            cut = np.full(len(yv), V.isoformat(), dtype="U10")
            np.savez_compressed(part_path, user_id=uid, cutoff=cut,
                                z=z.astype(np.float32), y=np.asarray(yv, np.float32))
            oof_u.append(uid); oof_c.append(cut); oof_z.append(z); oof_y.append(yv)
            log(f"DOMAIN-01 adapt fold {V}: resumed model, cal={calibrate(yv, z)[1]:.6f}")
            del fitted, Xv, yv, z
            gc.collect()
            continue
        cuts = [T for T in cutoff_grid(90, 7) if T + dt.timedelta(days=30) <= V]
        log(f"DOMAIN-01 adapt fold {V}: {len(cuts)} train cutoffs")
        mmap_dir = Path(args.mmap_dir) if args.mmap_dir else out
        Xtr, ytr, wtr, weight_diag = _assemble_adapt(
            cuts, all_features, model, domain_features, prior, args.temperature,
            args.clip_low, args.clip_high,
            mmap_dir / f"codex_domain_01_train_{V.strftime('%Y%m%d')}.mmap")
        weight_diag.insert(0, "validation_cutoff", V.isoformat())
        weight_frames.append(weight_diag)
        weight_diag.to_csv(weight_path, index=False)
        ds = production_models.make_datasets("direct", Xtr, ytr, wtr, params)[0]
        del Xtr
        mmap_path = mmap_dir / f"codex_domain_01_train_{V.strftime('%Y%m%d')}.mmap"
        if mmap_path.exists():
            mmap_path.unlink()
        gc.collect()
        fitted = production_models.train_direct_ds(ds, params, args.main_rounds)
        fitted.save_model(str(model_path))
        Xv, yv = make_xy(V, None, n_blocks=3, with_target=True, norm_long=True)
        z = np.maximum(_predict_frame(fitted, Xv, all_features), 0.0)
        uid = Xv["user_id"].to_numpy()
        cut = np.full(len(yv), V.isoformat(), dtype="U10")
        np.savez_compressed(part_path, user_id=uid, cutoff=cut,
                            z=z.astype(np.float32), y=np.asarray(yv, np.float32))
        oof_u.append(uid)
        oof_c.append(cut)
        oof_z.append(z); oof_y.append(yv)
        log(f"  fold {V}: n={len(yv):,}, neff={effective_sample_size(wtr)/len(wtr):.3f}, "
            f"cal={calibrate(yv, z)[1]:.6f}")
        del ytr, wtr, ds, fitted, Xv, yv, z
        gc.collect()
    np.savez_compressed(out / "oof_DOMAIN-01-DIRECT.npz",
                        user_id=np.concatenate(oof_u), cutoff=np.concatenate(oof_c),
                        z=np.concatenate(oof_z).astype(np.float32),
                        y=np.concatenate(oof_y).astype(np.float32))
    saved_weight_frames = [pd.read_csv(p) for p in sorted(out.glob("adapt_weights_*.csv"))]
    if saved_weight_frames:
        pd.concat(saved_weight_frames, ignore_index=True).to_csv(
            report_dir / "adaptation_train_weights.csv", index=False)
    summary = adaptation_reports(out, report_dir, Path(args.baseline_artifacts), args.weight_scheme)
    log(f"DOMAIN-01 adaptation complete in {time.time() - started:.0f}s")
    log(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", default=str(ARTIFACTS / "domain_01"))
    common.add_argument("--report-dir", default="research/domain_01/results")
    common.add_argument("--baseline-artifacts", default=str(ARTIFACTS))

    d = sub.add_parser("diagnose", parents=[common])
    d.add_argument("--folds", type=int, default=5)
    d.add_argument("--rounds", type=int, default=120)
    d.add_argument("--ablation-rounds", type=int, default=80)
    d.add_argument("--train-cap", type=int, default=450_000)
    d.add_argument("--ablation-cap", type=int, default=300_000)
    d.add_argument("--linear-cap", type=int, default=150_000)
    d.add_argument("--resume", action="store_true",
                   help="reuse strict-OOF checkpoints from an identical interrupted run")
    d.set_defaults(func=diagnose)

    a = sub.add_parser("adapt", parents=[common])
    a.add_argument("--domain-model", choices=["primary", "behavior"], default="behavior")
    a.add_argument("--temperature", type=float, default=0.5)
    a.add_argument("--clip-low", type=float, default=0.25)
    a.add_argument("--clip-high", type=float, default=4.0)
    a.add_argument("--weight-scheme", default="behavior_t05_clip4")
    a.add_argument("--main-rounds", type=int, default=300)
    a.add_argument("--mmap-dir", default=None,
                   help="directory for one temporary train matrix; useful on a low-RAM host")
    a.add_argument("--resume", action="store_true", help="reuse completed fold model/OOF parts")
    a.set_defaults(func=adapt)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
