"""BLOCK4-SAF: selection-aware residual correction from the last known blocks.

One command performs the invariant audit, four project folds, both controls,
the production cross-fit and the test-regime audit::

    python src/block4_saf.py

The script is deliberately separate from the production trainers.  It reuses
``build_features`` and the saved ``STRONGEST_CURRENT`` components, and writes
only BLOCK4_SAF-prefixed result artifacts.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (ARTIFACTS, CORRIDOR_END, CUTOFF_TEST, DATA_END, DATA_PROCESSED,
                        EXPERIMENTS, FOLD_WEIGHTS_S1, HISTORY_L, LGB_THREADS, SEED,
                        SUBMISSIONS, TARGET_DAYS, VAL_FOLDS_S1, cutoff_grid)
from src.data import sample_submit
from src.features import (build_features, feature_names, features_cached, panel_users,
                          to_np)
from src.validation import calibrate

PREFIX = "BLOCK4_SAF"
RESULTS = Path("research/strategies/results") / PREFIX
RESULTS.mkdir(parents=True, exist_ok=True)

PROD_V = dt.date(2026, 2, 13)
PROD_C = dt.date(2025, 12, 15)
PROD_F = dt.date(2026, 1, 14)
L_STAR = 2.3293
SEEDS = tuple(SEED + i for i in range(3))
ALPHAS = np.asarray([0.25, 0.50, 0.75, 1.00, 1.25], dtype=float)
BLOCK_CACHE_VERSION = 1
ROUNDS = 200

MODEL_PARAMS = dict(
    learning_rate=0.03,
    num_leaves=63,
    min_data_in_leaf=500,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=10.0,
    max_bin=63,
)

BASE_OOF = {
    "S1-E03a": 0.10,
    "S1-E02": 0.20,
    "S1-DIST": 0.25,
    "ETX-AVG3": 0.225,
    "SEQ-AVG3": 0.225,
}
BASE_TEST = {
    "S1-CAP": 0.10,
    "S1-UNC": 0.20,
    "S1-DIST": 0.25,
    "SEQ-01": 0.075,
    "SEQ-C289-S43": 0.075,
    "SEQ-C289-S44": 0.075,
    "ETX-01-S42-DCW": 0.075,
    "ETX-01-S43-DCW": 0.075,
    "ETX-01-S44-DCW": 0.075,
}

T0 = time.time()


def log(*parts) -> None:
    print(f"[{time.time() - T0:7.0f}s]", *parts, flush=True)


def json_dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8")


def window(T: dt.date, days: int = TARGET_DAYS) -> tuple[dt.date, dt.date]:
    """Closed date bounds implementing the mathematical interval ``(T,T+days]``."""
    return T + dt.timedelta(days=1), T + dt.timedelta(days=days)


def splitmix_group(user_ids) -> np.ndarray:
    """Stable global user split: ``splitmix64(user_id) & 1``."""
    h = np.asarray(user_ids, dtype=np.uint64).copy()
    with np.errstate(over="ignore"):
        h += np.uint64(0x9E3779B97F4A7C15)
        h ^= h >> np.uint64(30)
        h *= np.uint64(0xBF58476D1CE4E5B9)
        h ^= h >> np.uint64(27)
        h *= np.uint64(0x94D049BB133111EB)
        h ^= h >> np.uint64(31)
    return (h & np.uint64(1)).astype(np.int8)


def _future_path(T: dt.date) -> Path:
    return DATA_PROCESSED / f"{PREFIX}_future_{T:%Y%m%d}.parquet"


def future_all(T: dt.date) -> pl.DataFrame:
    """Any-row activity and positive-GMV target for every observed future user."""
    p = _future_path(T)
    if p.exists():
        return pl.read_parquet(p)
    if T + dt.timedelta(days=TARGET_DAYS) > DATA_END:
        raise ValueError(f"target {T} is not fully observed")
    from src.data import load
    a, b = window(T)
    out = (load().lazy()
           .filter((pl.col("event_date") >= a) & (pl.col("event_date") <= b))
           .group_by("user_id")
           .agg([
               pl.lit(1, dtype=pl.Int8).first().alias("activity"),
               pl.when(pl.col("gmv") > 0).then(pl.col("gmv")).otherwise(0.0).sum().alias("y"),
           ]).collect().sort("user_id"))
    out.write_parquet(p)
    return out


def future_labels(T: dt.date, users: pl.DataFrame) -> pl.DataFrame:
    return (users.select("user_id").join(future_all(T), on="user_id", how="left")
            .with_columns([
                pl.col("activity").fill_null(0).cast(pl.Int8),
                pl.col("y").fill_null(0.0),
            ]).sort("user_id"))


def _block_feature_path(T: dt.date) -> Path:
    return DATA_PROCESSED / f"feat_{PREFIX}_v{BLOCK_CACHE_VERSION}_{T:%Y%m%d}_L180.parquet"


def block_features(T: dt.date) -> pl.DataFrame:
    """S1-CAP/L=180 plus the opt-in non-overlapping block columns."""
    p = _block_feature_path(T)
    if p.exists():
        return pl.read_parquet(p)
    base = features_cached(T, HISTORY_L, False)
    out = build_features(T, HISTORY_L, False, block_features=True, base_features=base)
    out.write_parquet(p)
    return out


def users_features(T: dt.date, users: pl.DataFrame) -> pl.DataFrame:
    f = block_features(T)
    missing = users.select("user_id").join(f.select("user_id"), on="user_id", how="anti")
    if missing.height:
        raise AssertionError(f"{T}: {missing.height} panel users have no cutoff-safe history")
    return users.select("user_id").join(f, on="user_id", how="left").sort("user_id")


def segment_features(T: dt.date, users: pl.DataFrame) -> pl.DataFrame:
    """All-history recency plus the existing 180-day purchase count for diagnostics."""
    f = features_cached(T, None, False).select(["user_id", "w180_days_buy", "rec_buy"])
    out = users.select("user_id").join(f, on="user_id", how="left").sort("user_id")
    if out.height != users.height:
        raise AssertionError(f"{T}: segment feature alignment failed")
    return out


def canonical_features(frame: pl.DataFrame | None = None) -> list[str]:
    if frame is None:
        frame = block_features(VAL_FOLDS_S1[-1])
    feats = feature_names(frame)
    if "user_id" in feats or len(feats) != len(set(feats)):
        raise AssertionError("user_id or duplicate name in model features")
    cap_path = ARTIFACTS / "feats_S1-E03a.txt"
    if cap_path.exists():
        cap = cap_path.read_text(encoding="utf-8").splitlines()
        if feats[:len(cap)] != cap:
            raise AssertionError("BLOCK4 base is not the saved S1-CAP/L=180 feature set")
    return feats


def release_raw() -> None:
    from src import data
    data._CACHE.pop("df", None)
    gc.collect()


def clean_q_cutoffs(V: dt.date | None) -> list[dt.date]:
    cuts = cutoff_grid(90, 7)
    if V is not None:
        cuts = [T for T in cuts if T + dt.timedelta(days=TARGET_DAYS) <= V]
    for T in cuts:
        assert T <= CORRIDOR_END
        assert T + dt.timedelta(days=TARGET_DAYS) <= (V or DATA_END)
    return cuts


def required_feature_dates() -> list[dt.date]:
    dates = set(cutoff_grid(90, 7))
    for V in VAL_FOLDS_S1:
        dates |= {V, V - dt.timedelta(days=60), V - dt.timedelta(days=30)}
    dates |= {PROD_C, PROD_F, PROD_V}
    return sorted(dates)


def required_future_dates() -> list[dt.date]:
    dates = set(cutoff_grid(90, 7))
    for V in VAL_FOLDS_S1:
        dates |= {V, V - dt.timedelta(days=60), V - dt.timedelta(days=30)}
    dates |= {dt.date(2025, 11, 15), PROD_C, PROD_F}
    return sorted(T for T in dates if T + dt.timedelta(days=30) <= DATA_END)


def warm() -> None:
    log("warming future labels")
    for i, T in enumerate(required_future_dates(), 1):
        future_all(T)
        if i % 5 == 0:
            log(f"  future {i}/{len(required_future_dates())}: {T}")
    log("warming BLOCK4 feature profile")
    dates = required_feature_dates()
    for i, T in enumerate(dates, 1):
        f = block_features(T)
        if i == 1:
            log(f"  feature width={f.width - 1}")
        if i % 4 == 0 or i == len(dates):
            log(f"  features {i}/{len(dates)}: {T}")
    release_raw()


def run_audit() -> dict:
    """Fail-fast audit of every assumption that makes the experiment meaningful."""
    audit: dict[str, object] = {
        "status": "PASS",
        "activity_definition": "any daily-log row in (T,T+30]",
        "target_interval": "(T,T+30] = dates T+1 through T+30 inclusive",
        "production": {"V": PROD_V, "C": PROD_C, "F": PROD_F},
    }
    if (PROD_V, PROD_C, PROD_F) != (CUTOFF_TEST, CUTOFF_TEST - dt.timedelta(days=60),
                                    CUTOFF_TEST - dt.timedelta(days=30)):
        raise AssertionError("production geometry mismatch")

    test_users = panel_users(PROD_V, 3)
    sample = sample_submit().select("user_id")
    audit["test_panel"] = {
        "n": test_users.height,
        "unique": test_users["user_id"].n_unique(),
        "same_users_as_sample": set(test_users["user_id"].to_list()) == set(sample["user_id"].to_list()),
    }
    if audit["test_panel"] != {"n": 250000, "unique": 250000, "same_users_as_sample": True}:
        raise AssertionError(f"test panel invariant failed: {audit['test_panel']}")

    purchase_violations = []
    for T in required_future_dates():
        d = future_all(T)
        bad = d.filter((pl.col("y") > 0) & (pl.col("activity") != 1)).height
        purchase_violations.append({"cutoff": str(T), "violations": bad})
        if bad:
            raise AssertionError(f"{T}: purchase=>activity violations={bad}")
    audit["purchase_implies_activity"] = {
        "cutoffs_checked": len(purchase_violations),
        "violations": sum(x["violations"] for x in purchase_violations),
    }

    late = {}
    for T in (dt.date(2025, 11, 15), PROD_C, PROD_F):
        lab = future_labels(T, test_users)
        late[str(T)] = {"n": lab.height, "active": int(lab["activity"].sum())}
        if late[str(T)] != {"n": 250000, "active": 250000}:
            raise AssertionError(f"late guaranteed activity failed: {T} {late[str(T)]}")
    audit["late_activity"] = late

    panels = {}
    for V in VAL_FOLDS_S1:
        C, F = V - dt.timedelta(days=60), V - dt.timedelta(days=30)
        u = panel_users(V, 3)
        a2 = int(future_labels(C, u)["activity"].sum())
        a3 = int(future_labels(F, u)["activity"].sum())
        panels[str(V)] = {"n": u.height, "active_B2": a2, "active_B3": a3}
        if a2 != u.height or a3 != u.height:
            raise AssertionError(f"{V}: P_V is not active in B2/B3")
    audit["validation_panels"] = panels

    feats = canonical_features()
    q_fold_max = {str(V): str(max(clean_q_cutoffs(V))) for V in VAL_FOLDS_S1}
    q_prod = clean_q_cutoffs(None)
    audit["model_features"] = {
        "n": len(feats), "user_id_present": "user_id" in feats,
        "calendar_features": [c for c in feats if c.startswith("cutoff_")],
        "base_cap_features": len((ARTIFACTS / "feats_S1-E03a.txt").read_text(
            encoding="utf-8").splitlines()),
        "block_features": sum(c.startswith("block") for c in feats),
    }
    audit["q_training"] = {
        "fold_last_cutoffs": q_fold_max,
        "production_last_cutoff": str(max(q_prod)),
        "late_rows": sum(T > CORRIDOR_END for T in q_prod),
    }
    if "user_id" in feats or audit["model_features"]["calendar_features"]:
        raise AssertionError("forbidden identifier/calendar feature")
    if audit["q_training"]["late_rows"]:
        raise AssertionError("late rows entered q")

    # The executable tests mutate rows after T and pin the boundary behavior.
    audit["future_feature_lookup"] = {
        "implementation": "build_features filters event_date <= cutoff; block raw scan also <= cutoff",
        "verified_by": ["test_block_features_ignore_future_rows",
                        "test_default_feature_pipeline_is_backward_compatible"],
        "violations": 0,
    }
    json_dump(RESULTS / "audit.json", audit)
    return audit


def lgb_params(objective: str, seed: int) -> dict:
    p = dict(MODEL_PARAMS)
    p.update(objective=objective,
             metric="binary_logloss" if objective == "binary" else "rmse",
             seed=seed, bagging_seed=seed, feature_fraction_seed=seed,
             data_random_seed=seed, num_threads=LGB_THREADS, verbose=-1,
             force_row_wise=True)
    return p


def make_dataset(X: np.ndarray, y: np.ndarray, objective: str, feats: list[str],
                 seed: int = SEED):
    import lightgbm as lgb
    p = lgb_params(objective, seed)
    ds = lgb.Dataset(X, label=y, feature_name=feats, params=p, free_raw_data=True)
    ds.construct()
    return ds


def make_seed_datasets(X: np.ndarray, y: np.ndarray, objective: str, feats: list[str],
                       seeds=SEEDS) -> list:
    """One constructed Dataset per seed; LightGBM pins data_random_seed at construction."""
    return [make_dataset(X, y, objective, feats, seed) for seed in seeds]


def fit_predict(ds, Xpred: np.ndarray, objective: str, seeds=SEEDS) -> np.ndarray:
    import lightgbm as lgb
    out = []
    for seed in seeds:
        m = lgb.train(lgb_params(objective, seed), ds, num_boost_round=ROUNDS)
        out.append(np.asarray(m.predict(Xpred), dtype=np.float64))
        del m
    return np.vstack(out)


def assemble_q(cuts: list[dt.date], feats: list[str]) -> tuple[np.ndarray, np.ndarray, dict]:
    sizes = [panel_users(T, 2).height for T in cuts]
    X = np.empty((sum(sizes), len(feats)), dtype=np.float32)
    y = np.empty(sum(sizes), dtype=np.int8)
    pos, per_cut = 0, []
    for i, (T, n) in enumerate(zip(cuts, sizes), 1):
        u = panel_users(T, 2)
        xf = users_features(T, u)
        lab = future_labels(T, u)
        if not np.array_equal(xf["user_id"].to_numpy(), lab["user_id"].to_numpy()):
            raise AssertionError(f"{T}: q feature/label order mismatch")
        X[pos:pos + n] = to_np(xf, feats)
        y[pos:pos + n] = lab["activity"].to_numpy()
        per_cut.append({"cutoff": str(T), "rows": n, "activity_rate": float(y[pos:pos+n].mean())})
        pos += n
        if i % 4 == 0 or i == len(cuts):
            log(f"    q matrix {i}/{len(cuts)} cutoffs, {pos:,} rows")
        del xf, lab
    return X, y, {"rows": len(y), "cutoffs": per_cut,
                  "activity_rate": float(y.mean()), "last_cutoff": str(max(cuts))}


def train_q(V: dt.date | None, Xpred: np.ndarray, feats: list[str]) -> tuple[np.ndarray, np.ndarray, dict]:
    cuts = clean_q_cutoffs(V)
    log(f"  q: assembling {len(cuts)} clean cutoffs")
    X, y, meta = assemble_q(cuts, feats)
    release_raw()
    log(f"  q: LightGBM dataset {X.shape[0]:,} x {X.shape[1]}")
    datasets = make_seed_datasets(X, y, "binary", feats)
    del X, y
    gc.collect()
    pred = np.vstack([fit_predict(ds, Xpred, "binary", (seed,))[0]
                      for ds, seed in zip(datasets, SEEDS)])
    del datasets
    gc.collect()
    return pred.mean(axis=0), pred, meta


def shuffle_within_bins(z: np.ndarray, w180: np.ndarray, rec: np.ndarray,
                        V: dt.date, donor_group: int) -> np.ndarray:
    wb = np.where(w180 <= 1, 0, np.where(w180 <= 15, 1, 2))
    # Missing recency means no prior purchase and is conservatively included in 61+.
    rb = np.where(np.isnan(rec) | (rec >= 61), 2, np.where(rec <= 14, 0, 1))
    strata = wb * 3 + rb
    rng = np.random.default_rng(np.random.SeedSequence([SEED, V.toordinal(), donor_group]))
    out = np.asarray(z, dtype=float).copy()
    for s in range(9):
        idx = np.flatnonzero(strata == s)
        if len(idx) > 1:
            out[idx] = out[rng.permutation(idx)]
    return out


def conditional_crossfit(V: dt.date, users: pl.DataFrame, Xc: np.ndarray, Xf: np.ndarray,
                         Xv: np.ndarray, zc: np.ndarray, zf: np.ndarray,
                         feats: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    groups = splitmix_group(users["user_id"].to_numpy())
    if set(np.unique(groups)) != {0, 1}:
        raise AssertionError("degenerate user split")
    n = len(groups)
    nu_c = np.empty((len(SEEDS), n), dtype=np.float64)
    nu_f = np.empty_like(nu_c)
    nu_shuf = np.empty_like(nu_c)
    iw = feats.index("w180_days_buy")
    ir = feats.index("rec_buy")
    rows = []
    for recipient in (0, 1):
        donor = 1 - recipient
        tr = groups == donor
        va = groups == recipient
        if np.any(tr & va):
            raise AssertionError("cross-fit groups overlap")
        zfs = shuffle_within_bins(zf[tr], Xf[tr, iw], Xf[tr, ir], V, donor)
        ds_c = make_seed_datasets(Xc[tr], zc[tr], "regression", feats)
        ds_f = make_seed_datasets(Xf[tr], zf[tr], "regression", feats)
        ds_s = make_seed_datasets(Xf[tr], zfs, "regression", feats)
        for si, seed in enumerate(SEEDS):
            nu_c[si, va] = fit_predict(ds_c[si], Xv[va], "regression", (seed,))[0]
            nu_f[si, va] = fit_predict(ds_f[si], Xv[va], "regression", (seed,))[0]
            nu_shuf[si, va] = fit_predict(ds_s[si], Xv[va], "regression", (seed,))[0]
        rows.append({"recipient_group": recipient, "donor_group": donor,
                     "train_rows_nu_C": int(tr.sum()), "train_rows_nu_F": int(tr.sum()),
                     "prediction_rows": int(va.sum()), "overlap": int(np.sum(tr & va))})
        del ds_c, ds_f, ds_s, zfs
        gc.collect()
    return nu_c, nu_f, nu_shuf, {"groups": rows,
                                         "group0": int((groups == 0).sum()),
                                         "group1": int((groups == 1).sum())}


def _strongest_fold(V: dt.date, uid: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.zeros(len(uid), dtype=np.float64)
    for name, weight in BASE_OOF.items():
        d = np.load(ARTIFACTS / f"oof_{name}.npz", allow_pickle=False)
        m = np.asarray(d["cutoff"], dtype="U10") == V.isoformat()
        u = np.asarray(d["user_id"])[m]
        order = np.argsort(u)
        if not np.array_equal(u[order], uid):
            raise AssertionError(f"{name} OOF users differ on {V}")
        yy = np.asarray(d["y"], float)[m][order]
        if not np.allclose(yy, y, atol=1e-6):
            raise AssertionError(f"{name} OOF targets differ on {V}")
        out += weight * np.asarray(d["z"], float)[m][order]
    return out


def _strongest_test() -> tuple[np.ndarray, np.ndarray]:
    uid = np.load(ARTIFACTS / "uid_S1-CAP.npy")
    z = np.zeros(len(uid), dtype=np.float64)
    for name, weight in BASE_TEST.items():
        u = np.load(ARTIFACTS / f"uid_{name}.npy")
        if not np.array_equal(u, uid):
            raise AssertionError(f"test uid mismatch for {name}")
        z += weight * np.load(ARTIFACTS / f"ztest_{name}.npy")
    return uid, z


def fold_raw_path(V: dt.date) -> Path:
    return ARTIFACTS / f"{PREFIX}_fold_{V:%Y%m%d}.npz"


def run_fold(V: dt.date, feats: list[str], resume: bool = True) -> dict:
    p = fold_raw_path(V)
    if resume and p.exists():
        log(f"fold {V}: loading cached raw predictions")
        return dict(np.load(p, allow_pickle=False))
    C, F = V - dt.timedelta(days=60), V - dt.timedelta(days=30)
    log(f"fold {V}: C={C}, F={F}")
    users = panel_users(V, 3).sort("user_id")
    fc, ff, fv = (users_features(T, users) for T in (C, F, V))
    Xc, Xf, Xv = (to_np(x, feats) for x in (fc, ff, fv))
    lc, lf, lv = (future_labels(T, users) for T in (C, F, V))
    if int(lc["activity"].sum()) != users.height or int(lf["activity"].sum()) != users.height:
        raise AssertionError(f"{V}: conditional rows are not all A=1")
    zc, zf, y = np.log1p(lc["y"].to_numpy()), np.log1p(lf["y"].to_numpy()), lv["y"].to_numpy()
    activity = lv["activity"].to_numpy().astype(np.int8)
    q, q_seed, q_meta = train_q(V, Xv, feats)
    log(f"  q ready: mean={q.mean():.4f}")
    nu_c, nu_f, nu_shuf, cross = conditional_crossfit(V, users, Xc, Xf, Xv, zc, zf, feats)
    z_base = _strongest_fold(V, users["user_id"].to_numpy(), y)
    seg = segment_features(V, users)
    np.savez_compressed(
        p, uid=users["user_id"].to_numpy(), y=y.astype(np.float32), activity=activity,
        z_base=z_base.astype(np.float32), q=q.astype(np.float32), q_seed=q_seed.astype(np.float32),
        nu_c=nu_c.astype(np.float32), nu_f=nu_f.astype(np.float32),
        nu_shuf=nu_shuf.astype(np.float32),
        w180=seg["w180_days_buy"].to_numpy().astype(np.float32),
        rec=seg["rec_buy"].to_numpy().astype(np.float32),
    )
    json_dump(RESULTS / f"fold_{V:%Y%m%d}_train.json", {"q": q_meta, "crossfit": cross,
                                                          "C": C, "F": F, "V": V})
    log(f"fold {V}: saved {p.name}")
    del Xc, Xf, Xv, fc, ff, fv
    gc.collect()
    return dict(np.load(p, allow_pickle=False))


def auc(y, score) -> float:
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def q_metrics(activity: np.ndarray, q: np.ndarray) -> dict:
    from sklearn.metrics import brier_score_loss, log_loss
    return {"auc": auc(activity, q),
            "logloss": float(log_loss(activity, np.clip(q, 1e-8, 1 - 1e-8))),
            "brier": float(brier_score_loss(activity, q))}


def preprocess_by_fold(raws: list[np.ndarray]) -> tuple[list[np.ndarray], tuple[float, float], dict]:
    all_raw = np.concatenate(raws)
    lo, hi = np.quantile(all_raw, [0.005, 0.995])
    out, details = [], {}
    for V, raw in zip(VAL_FOLDS_S1, raws):
        clipped = np.clip(raw, lo, hi)
        centered = clipped - clipped.mean()
        out.append(centered)
        details[str(V)] = {"raw_mean": float(raw.mean()), "clipped_mean": float(clipped.mean()),
                           "centered_mean": float(centered.mean()),
                           "clipped_share": float(np.mean((raw < lo) | (raw > hi)))}
    return out, (float(lo), float(hi)), details


def fold_score(y: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    off, score = calibrate(y, z)
    return float(off), float(score)


def lofo(deltas: list[np.ndarray], folds: list[dict]) -> dict:
    weights = np.asarray(FOLD_WEIGHTS_S1, float)
    base_scores = np.asarray([fold_score(d["y"], d["z_base"])[1] for d in folds])
    curves = np.empty((len(ALPHAS), len(folds)), dtype=float)
    offsets = np.empty_like(curves)
    for ai, alpha in enumerate(ALPHAS):
        for fi, (d, delta) in enumerate(zip(folds, deltas)):
            offsets[ai, fi], curves[ai, fi] = fold_score(d["y"], d["z_base"] + alpha * delta)
    held_scores, held_alpha, held_offsets = [], [], []
    for h in range(len(folds)):
        keep = np.asarray([i for i in range(len(folds)) if i != h])
        wk = weights[keep] / weights[keep].sum()
        best = int(np.argmin(curves[:, keep] @ wk))
        held_scores.append(float(curves[best, h]))
        held_alpha.append(float(ALPHAS[best]))
        held_offsets.append(float(offsets[best, h]))
    held_scores = np.asarray(held_scores)
    held_delta = held_scores - base_scores
    w = weights / weights.sum()
    return {
        "base_scores": base_scores.tolist(), "alpha_curve": curves.tolist(),
        "heldout_scores": held_scores.tolist(), "heldout_delta": held_delta.tolist(),
        "heldout_alpha": held_alpha, "heldout_offsets": held_offsets,
        "delta_wcv": float(w @ held_delta), "base_wcv": float(w @ base_scores),
        "candidate_wcv": float(w @ held_scores), "improved_folds": int((held_delta < 0).sum()),
        "alpha_range": float(max(held_alpha) - min(held_alpha)),
    }


def weighted_median(values: list[float]) -> float:
    v = np.asarray(values, float)
    w = np.asarray(FOLD_WEIGHTS_S1, float)
    o = np.argsort(v)
    return float(v[o][np.searchsorted(np.cumsum(w[o]), w.sum() / 2.0, side="left")])


def segment_masks(w180: np.ndarray, rec: np.ndarray) -> dict[str, np.ndarray]:
    known = ~np.isnan(rec)
    return {
        "rec_buy 15-60": known & (rec >= 15) & (rec <= 60),
        "w180_days_buy 2-15": (w180 >= 2) & (w180 <= 15),
        "intersection": known & (rec >= 15) & (rec <= 60) & (w180 >= 2) & (w180 <= 15),
        "w180_days_buy 0-1": w180 <= 1,
        "w180_days_buy 16+": w180 >= 16,
        "never purchased": ~known,
    }


def segment_diagnostics(folds: list[dict], deltas: list[np.ndarray], main: dict) -> list[dict]:
    rows = []
    for name in segment_masks(folds[0]["w180"], folds[0]["rec"]):
        base_fold, new_fold, ns = [], [], []
        for fi, (d, delta) in enumerate(zip(folds, deltas)):
            m = segment_masks(d["w180"], d["rec"])[name]
            off_b = fold_score(d["y"], d["z_base"])[0]
            alpha = main["heldout_alpha"][fi]
            znew = d["z_base"] + alpha * delta
            off_n = main["heldout_offsets"][fi]
            ly = np.log1p(d["y"])
            base_fold.append(float(np.sqrt(np.mean((ly[m] - np.maximum(d["z_base"][m] + off_b, 0)) ** 2))))
            new_fold.append(float(np.sqrt(np.mean((ly[m] - np.maximum(znew[m] + off_n, 0)) ** 2))))
            ns.append(int(m.sum()))
        w = np.asarray(FOLD_WEIGHTS_S1, float); w /= w.sum()
        rows.append({"segment": name, "n_by_fold": ns,
                     "base_rmsle": float(w @ base_fold), "new_rmsle": float(w @ new_fold),
                     "delta": float(w @ (np.asarray(new_fold) - np.asarray(base_fold)))})
    return rows


def analyze_validation(folds: list[dict]) -> tuple[dict, list[np.ndarray], tuple[float, float]]:
    main_raw = [d["q"] * (d["nu_f"].mean(axis=0) - d["nu_c"].mean(axis=0)) for d in folds]
    shuf_raw = [d["q"] * (d["nu_shuf"].mean(axis=0) - d["nu_c"].mean(axis=0)) for d in folds]
    seed_raw = [[d["q"] * (d["nu_f"][si] - d["nu_c"][si]) for d in folds]
                for si in range(len(SEEDS))]
    deltas, bounds, prep = preprocess_by_fold(main_raw)
    shuf_delta, shuf_bounds, shuf_prep = preprocess_by_fold(shuf_raw)
    seed_delta, seed_bounds = [], []
    for raw in seed_raw:
        d, b, _ = preprocess_by_fold(raw)
        seed_delta.append(d); seed_bounds.append(b)

    main = lofo(deltas, folds)
    shuffle = lofo(shuf_delta, folds)
    seed_control = [dict(seed=seed, **lofo(ds, folds)) for seed, ds in zip(SEEDS, seed_delta)]
    qrows, diag_rows = [], []
    for fi, (V, d, raw, delta) in enumerate(zip(VAL_FOLDS_S1, folds, main_raw, deltas)):
        qm = q_metrics(d["activity"], d["q"])
        active = d["activity"] == 1
        nu_c = d["nu_c"].mean(axis=0); nu_f = d["nu_f"].mean(axis=0)
        alpha = main["heldout_alpha"][fi]
        znew = d["z_base"] + alpha * delta
        ly = np.log1p(d["y"])
        residual = ly - d["z_base"]
        qrows.append({"fold": str(V), **qm})
        diag_rows.append({
            "fold": str(V),
            "baseline_rmsle": main["base_scores"][fi],
            "honest_rmsle": main["heldout_scores"][fi],
            "honest_delta": main["heldout_delta"][fi],
            "heldout_alpha": alpha,
            "purchase_auc_base": auc(d["y"] > 0, d["z_base"]),
            "purchase_auc_new": auc(d["y"] > 0, znew),
            "conditional_rmse_nu_C_A1": float(np.sqrt(np.mean((ly[active] - nu_c[active]) ** 2))),
            "conditional_rmse_nu_F_A1": float(np.sqrt(np.mean((ly[active] - nu_f[active]) ** 2))),
            "var_delta": float(np.var(delta)),
            "corr_delta_residual": float(np.corrcoef(delta, residual)[0, 1]),
            "corr_residuals": float(np.corrcoef(residual, ly - znew)[0, 1]),
            "corr_conditional_residuals_A1": float(np.corrcoef(
                ly[active] - nu_c[active], ly[active] - nu_f[active])[0, 1]),
            "share_abs_delta_gt_0.1": float(np.mean(np.abs(delta) > 0.1)),
            "share_abs_delta_gt_0.25": float(np.mean(np.abs(delta) > 0.25)),
            "share_abs_delta_gt_0.5": float(np.mean(np.abs(delta) > 0.5)),
            **q_metrics(d["activity"], d["q"]),
        })

    segments = segment_diagnostics(folds, deltas, main)
    oof = {
        "uid": np.concatenate([d["uid"] for d in folds]),
        "cutoff": np.concatenate([np.full(len(d["uid"]), str(V)) for V, d in zip(VAL_FOLDS_S1, folds)]),
        "y": np.concatenate([d["y"] for d in folds]),
        "activity": np.concatenate([d["activity"] for d in folds]),
        "z_base": np.concatenate([d["z_base"] for d in folds]),
        "q": np.concatenate([d["q"] for d in folds]),
        "nu_c": np.concatenate([d["nu_c"].mean(axis=0) for d in folds]),
        "nu_f": np.concatenate([d["nu_f"].mean(axis=0) for d in folds]),
        "delta_raw": np.concatenate(main_raw),
        "delta": np.concatenate(deltas),
        "delta_shuf": np.concatenate(shuf_delta),
    }
    held_alpha_rows = np.concatenate([np.full(len(d["uid"]), a) for d, a in zip(folds, main["heldout_alpha"])])
    oof["z_new_honest"] = oof["z_base"] + held_alpha_rows * oof["delta"]
    np.savez_compressed(ARTIFACTS / f"oof_{PREFIX}.npz", **oof)

    report = {
        "main_lofo": main, "shuffle_lofo": shuffle,
        "seed_control": seed_control, "winsor_bounds": bounds,
        "shuffle_bounds": shuf_bounds, "seed_bounds": seed_bounds,
        "preprocess": prep, "shuffle_preprocess": shuf_prep,
        "q_metrics": qrows, "fold_diagnostics": diag_rows, "segments": segments,
        "alpha_production": weighted_median(main["heldout_alpha"]),
    }
    json_dump(RESULTS / "validation.json", report)
    pl.DataFrame(qrows).write_csv(RESULTS / "q_metrics.csv")
    pl.DataFrame(diag_rows).write_csv(RESULTS / "fold_diagnostics.csv")
    alpha_csv = []
    for ai, alpha in enumerate(ALPHAS):
        row = {"alpha": float(alpha)}
        for fi, V in enumerate(VAL_FOLDS_S1):
            score = main["alpha_curve"][ai][fi]
            row[f"rmsle_{V:%Y%m%d}"] = score
            row[f"delta_{V:%Y%m%d}"] = score - main["base_scores"][fi]
        w = np.asarray(FOLD_WEIGHTS_S1, float); w /= w.sum()
        row["delta_wcv_insample"] = float(w @ np.asarray(
            [row[f"delta_{V:%Y%m%d}"] for V in VAL_FOLDS_S1]))
        alpha_csv.append(row)
    pl.DataFrame(alpha_csv).write_csv(RESULTS / "alpha_curve.csv")
    seg_csv = []
    for r in segments:
        flat = {k: v for k, v in r.items() if k != "n_by_fold"}
        flat.update({f"n_{V:%Y%m%d}": n for V, n in zip(VAL_FOLDS_S1, r["n_by_fold"])})
        seg_csv.append(flat)
    pl.DataFrame(seg_csv).write_csv(RESULTS / "segments.csv")
    seed_csv = []
    for r in seed_control:
        flat = {"seed": r["seed"], "delta_wcv": r["delta_wcv"],
                "candidate_wcv": r["candidate_wcv"], "improved_folds": r["improved_folds"],
                "alpha_range": r["alpha_range"]}
        for V, d, a in zip(VAL_FOLDS_S1, r["heldout_delta"], r["heldout_alpha"]):
            flat[f"delta_{V:%Y%m%d}"] = d
            flat[f"alpha_{V:%Y%m%d}"] = a
        seed_csv.append(flat)
    pl.DataFrame(seed_csv).write_csv(RESULTS / "seed_control.csv")
    return report, deltas, bounds


def production_raw_path() -> Path:
    return ARTIFACTS / f"{PREFIX}_test_raw.npz"


def run_production(feats: list[str], resume: bool = True) -> dict:
    p = production_raw_path()
    if resume and p.exists():
        log("production: loading cached raw predictions")
        return dict(np.load(p, allow_pickle=False))
    log(f"production: V={PROD_V}, C={PROD_C}, F={PROD_F}")
    users = panel_users(PROD_V, 3).sort("user_id")
    fc, ff, fv = (users_features(T, users) for T in (PROD_C, PROD_F, PROD_V))
    Xc, Xf, Xv = (to_np(x, feats) for x in (fc, ff, fv))
    lc, lf = future_labels(PROD_C, users), future_labels(PROD_F, users)
    if int(lc["activity"].sum()) != 250000 or int(lf["activity"].sum()) != 250000:
        raise AssertionError("production conditional population is not fully active")
    zc, zf = np.log1p(lc["y"].to_numpy()), np.log1p(lf["y"].to_numpy())
    q, q_seed, q_meta = train_q(None, Xv, feats)
    nu_c, nu_f, nu_shuf, cross = conditional_crossfit(PROD_V, users, Xc, Xf, Xv, zc, zf, feats)
    uid_base, z_base = _strongest_test()
    if not np.array_equal(uid_base, users["user_id"].to_numpy()):
        raise AssertionError("production BLOCK4/base user order mismatch")
    seg = segment_features(PROD_V, users)
    np.savez_compressed(p, uid=uid_base, z_base=z_base.astype(np.float32),
                        q=q.astype(np.float32), q_seed=q_seed.astype(np.float32),
                        nu_c=nu_c.astype(np.float32), nu_f=nu_f.astype(np.float32),
                        nu_shuf=nu_shuf.astype(np.float32),
                        w180=seg["w180_days_buy"].to_numpy().astype(np.float32),
                        rec=seg["rec_buy"].to_numpy().astype(np.float32))
    json_dump(RESULTS / "production_train.json", {"q": q_meta, "crossfit": cross,
                                                    "C": PROD_C, "F": PROD_F, "V": PROD_V})
    log(f"production saved: {p.name}")
    return dict(np.load(p, allow_pickle=False))


def quantiles(x: np.ndarray) -> dict[str, float]:
    ps = [0, .005, .01, .05, .25, .5, .75, .95, .99, .995, 1]
    return {f"q{p:g}": float(v) for p, v in zip(ps, np.quantile(x, ps))}


def regime_audit(prod: dict, validation: dict) -> tuple[dict, np.ndarray, np.ndarray]:
    oof = np.load(ARTIFACTS / f"oof_{PREFIX}.npz", allow_pickle=False)
    raw_oof = np.asarray(oof["delta_raw"], float)
    raw_test = prod["q"] * (prod["nu_f"].mean(axis=0) - prod["nu_c"].mean(axis=0))
    lo, hi = validation["winsor_bounds"]
    clipped = np.clip(raw_test, lo, hi)
    delta_test = clipped - clipped.mean()
    var_oof, var_test = float(np.var(raw_oof)), float(np.var(raw_test))
    ratio = var_test / var_oof if var_oof > 0 else float("inf")
    std_ratio = float(np.std(raw_test) / np.std(raw_oof)) if var_oof > 0 else float("inf")
    clip_share = float(np.mean((raw_test < lo) | (raw_test > hi)))
    segrows = []
    for name, m in segment_masks(prod["w180"], prod["rec"]).items():
        segrows.append({"segment": name, "n": int(m.sum()), "mean": float(raw_test[m].mean()),
                        "std": float(raw_test[m].std()), "var": float(raw_test[m].var()),
                        **quantiles(raw_test[m])})
    passed = bool(0.5 <= ratio <= 1.5 and clip_share <= 0.05)
    report = {
        "var_oof": var_oof, "var_test": var_test, "var_ratio": ratio,
        "std_oof": float(np.std(raw_oof)), "std_test": float(np.std(raw_test)),
        "std_ratio": std_ratio, "oof_quantiles": quantiles(raw_oof),
        "test_quantiles": quantiles(raw_test), "winsor_bounds": [lo, hi],
        "test_clipped_share": clip_share, "segments": segrows,
        "pass": passed,
        "rule": "0.5<=Var_test/Var_oof<=1.5 and test clipped share<=5%",
    }
    json_dump(RESULTS / "test_regime.json", report)
    pl.DataFrame(segrows).write_csv(RESULTS / "test_delta_segments.csv")
    return report, raw_test, delta_test


def decide(validation: dict, regime: dict) -> tuple[str, list[str]]:
    main, shuf = validation["main_lofo"], validation["shuffle_lofo"]
    d = float(main["delta_wcv"])
    folds = np.asarray(main["heldout_delta"])
    reasons = []
    accept = (d <= -0.0012 and np.all(folds < 0) and folds[-1] <= -0.0007
              and shuf["delta_wcv"] > -0.0002 and main["alpha_range"] <= 0.5
              and regime["pass"])
    if accept:
        return "ACCEPT", ["all ACCEPT gates passed"]
    if shuf["delta_wcv"] <= -0.0002:
        reasons.append(f"shuffle gain {shuf['delta_wcv']:+.6f} is too strong")
    if not regime["pass"]:
        reasons.append(f"test regime failed (variance ratio {regime['var_ratio']:.3f})")
    if folds[-1] >= 0:
        reasons.append("fold 2025-10-16 did not improve")
    cont = (-0.0012 < d <= -0.0005 and int((folds < 0).sum()) >= 3 and folds[-1] < 0
            and shuf["delta_wcv"] > -0.0002 and regime["pass"])
    if cont:
        return "CONTINUE", reasons or ["CONTINUE magnitude/stability gate passed"]
    if d > -0.0005:
        reasons.append(f"honest LOFO gain {d:+.6f} is below the -0.0005 floor")
    if int((folds < 0).sum()) < 3:
        reasons.append(f"only {int((folds < 0).sum())}/4 folds improve")
    return "REJECT", reasons or ["ACCEPT/CONTINUE gates not met"]


def level_shift(z: np.ndarray, target: float) -> float:
    lo, hi = -5.0, 5.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if float(np.maximum(z + mid, 0).mean()) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def build_submission(prod: dict, delta_test: np.ndarray, alpha: float) -> dict:
    z_raw = np.asarray(prod["z_base"], float) + alpha * delta_test
    shift = level_shift(z_raw, L_STAR)
    z = np.maximum(z_raw + shift, 0.0)
    pred = np.expm1(z)
    uid = np.asarray(prod["uid"])
    frame = pl.DataFrame({"user_id": uid, "predict": pred})
    order = sample_submit().select("user_id").with_row_index("_order")
    frame = frame.join(order, on="user_id", how="inner").sort("_order").drop("_order")
    if frame.height != 250000 or frame["user_id"].n_unique() != 250000:
        raise AssertionError("submission row/unique invariant")
    if frame["user_id"].to_list() != sample_submit()["user_id"].to_list():
        raise AssertionError("submission order invariant")
    p = frame["predict"].to_numpy()
    if not np.isfinite(p).all() or np.any(p < 0):
        raise AssertionError("submission contains invalid predictions")
    out = SUBMISSIONS / f"{PREFIX}_submission.csv"
    frame.write_csv(out, float_precision=6)
    disk = pl.read_csv(out)
    zdisk = np.log1p(disk["predict"].to_numpy())
    # Reorder expected z by the same explicit sample order, without relying on uid sort.
    expected = pl.DataFrame({"user_id": uid, "z": z}).join(
        order, on="user_id", how="inner").sort("_order")["z"].to_numpy()
    recon = float(np.max(np.abs(zdisk - expected)))
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    stats = {
        "path": str(out), "sha256": sha, "n": frame.height,
        "unique_users": frame["user_id"].n_unique(), "mean_log1p": float(zdisk.mean()),
        "min": float(p.min()), "max": float(p.max()), "zeros": float(np.mean(p == 0)),
        "nan_inf": int((~np.isfinite(p)).sum()), "negative": int((p < 0).sum()),
        "level_shift": float(shift), "alpha": float(alpha), "reconstruction_max_abs": recon,
        "corr_with_strongest": float(np.corrcoef(prod["z_base"], z)[0, 1]),
        "var_change": float(np.var(z - prod["z_base"])),
    }
    json_dump(RESULTS / "submission.json", stats)
    return stats


def save_test_predictions(prod: dict, raw_test: np.ndarray, delta_test: np.ndarray,
                          validation: dict) -> None:
    alpha = validation["alpha_production"]
    znew = np.asarray(prod["z_base"], float) + alpha * delta_test
    np.savez_compressed(
        ARTIFACTS / f"test_{PREFIX}.npz", uid=prod["uid"], z_base=prod["z_base"],
        q=prod["q"], nu_c=prod["nu_c"].mean(axis=0), nu_f=prod["nu_f"].mean(axis=0),
        delta_raw=raw_test.astype(np.float32), delta=delta_test.astype(np.float32),
        alpha=np.asarray(alpha), z_new=znew.astype(np.float32))


def write_config(feats: list[str]) -> None:
    json_dump(RESULTS / "config.json", {
        "prefix": PREFIX, "feature_profile": "S1-CAP/L=180 + opt-in 30d block dynamics",
        "n_features": len(feats), "feature_names": feats, "seeds": SEEDS,
        "model_params": MODEL_PARAMS, "rounds": ROUNDS, "alphas": ALPHAS.tolist(),
        "folds": VAL_FOLDS_S1, "fold_weights": FOLD_WEIGHTS_S1,
        "production": {"V": PROD_V, "C": PROD_C, "F": PROD_F}, "level": L_STAR,
        "strongest_oof": BASE_OOF, "strongest_test": BASE_TEST,
    })


def run(resume: bool = True) -> dict:
    audit = run_audit()
    log("audit PASS")
    warm()
    feats = canonical_features()
    write_config(feats)
    folds = [run_fold(V, feats, resume) for V in VAL_FOLDS_S1]
    validation, _, _ = analyze_validation(folds)
    log(f"honest LOFO {validation['main_lofo']['delta_wcv']:+.6f}; "
        f"shuffle {validation['shuffle_lofo']['delta_wcv']:+.6f}")
    prod = run_production(feats, resume)
    regime, raw_test, delta_test = regime_audit(prod, validation)
    save_test_predictions(prod, raw_test, delta_test, validation)
    verdict, reasons = decide(validation, regime)
    submission = None
    if verdict == "ACCEPT":
        submission = build_submission(prod, delta_test, validation["alpha_production"])
    summary = {
        "verdict": verdict, "reasons": reasons,
        "honest_lofo_delta_wcv": validation["main_lofo"]["delta_wcv"],
        "honest_fold_deltas": validation["main_lofo"]["heldout_delta"],
        "heldout_alphas": validation["main_lofo"]["heldout_alpha"],
        "alpha_range": validation["main_lofo"]["alpha_range"],
        "alpha_production": validation["alpha_production"],
        "shuffle_delta_wcv": validation["shuffle_lofo"]["delta_wcv"],
        "test_var_ratio": regime["var_ratio"], "test_regime_pass": regime["pass"],
        "submission": submission, "audit": audit["status"],
    }
    json_dump(RESULTS / "summary.json", summary)
    log(f"FINAL VERDICT: {verdict} — {'; '.join(reasons)}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", nargs="?", default="run", choices=["run", "audit", "warm"])
    ap.add_argument("--no-resume", action="store_true")
    a = ap.parse_args()
    if a.command == "audit":
        print(json.dumps(run_audit(), ensure_ascii=False, indent=2, default=str))
    elif a.command == "warm":
        warm()
    else:
        run(not a.no_resume)


if __name__ == "__main__":
    main()
