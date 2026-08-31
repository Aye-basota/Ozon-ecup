"""RENEWAL-01 / Next-Purchase Clock.

Компактный recurrent-event эксперимент: восстанавливает интервалы между
purchase-days (строго ``gmv > 0``), оценивает условную вероятность завершения
текущего интервала в следующие 30 дней и проверяет её в штатной S1-схеме.

R0 — дискретный Kaplan–Meier с правым цензурированием текущего интервала и
     beta-shrinkage индивидуальной эмпирической вероятности к cohort curve.
R1 — небольшой binary LightGBM только на renewal/clock-признаках.

Все признаки строятся исключительно через ``build_features(cutoff_date)`` и
фильтруют события условием ``event_date <= cutoff_date``. Search/Catalog clocks
корректны, потому что в сырых данных ``gmv = gmv_search + gmv_cat``.

Полный запуск одной командой из корня репозитория::

    python src/renewal.py --baseline-artifacts artifacts
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

# Project style requires ``python src/<script>.py``.  In that invocation Python
# puts ``src/`` rather than the repository root on sys.path.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (ARTIFACTS, CORRIDOR_END, CUTOFF_TEST, DATA_START,
                        FOLD_WEIGHTS_S1, LGB_PARAMS, RAW_PARQUET, SEED, TARGET_DAYS,
                        VAL_FOLDS_S1, cutoff_grid)
from src.data import sample_submit
from src.features import panel_users
from src.report import evaluate, save_report
from src.validation import get_folds

EXP_ID = "RENEWAL-01"
CACHE = ARTIFACTS / "renewal_01_cache"
ROUNDS = 180
SEEDS = (SEED, SEED + 1, SEED + 2)
SHRINKAGES = (2.0, 5.0, 10.0, 20.0)
PRIMARY_SHRINKAGE = 10.0
EPS = 1e-6
SEED_FLOOR = 0.00712

T0 = time.time()
_EVENT_CACHE: dict[str, pl.DataFrame] = {}


def say(*items) -> None:
    print(f"[{time.time() - T0:7.1f}s]", *items, flush=True)


def _tag(T: dt.date) -> str:
    return T.strftime("%Y%m%d")


def purchase_events(channel: str = "all") -> pl.DataFrame:
    """Purchase-day sequence with a backward-looking gap column.

    The gap is safe to precompute on the full raw table: it is a difference to
    the *previous* event. ``build_features`` still removes rows after cutoff.
    """
    if channel in _EVENT_CACHE:
        return _EVENT_CACHE[channel]
    if "base" not in _EVENT_CACHE:
        # Read only the five columns needed by this experiment.  Using the
        # shared ``data.load()`` would retain all 13 columns / 30.6M rows and
        # needlessly compete for memory with LightGBM and the parallel SEQ run.
        base = (pl.scan_parquet(RAW_PARQUET).filter(pl.col("gmv") > 0)
                .select("user_id", "event_date", "gmv", "gmv_search", "gmv_cat")
                .sort(["user_id", "event_date"]).collect())
        _EVENT_CACHE["base"] = base
    base = _EVENT_CACHE["base"]
    if channel == "all":
        events = base
    elif channel == "search":
        events = base.filter(pl.col("gmv_search") > 0)
    elif channel == "catalog":
        events = base.filter(pl.col("gmv_cat") > 0)
    else:
        raise ValueError(channel)
    events = (events.sort(["user_id", "event_date"])
              .with_columns(gap=pl.col("event_date").diff().over("user_id")
                            .dt.total_days().cast(pl.Float32)))
    _EVENT_CACHE[channel] = events
    return events


def audit_purchase_definition() -> dict:
    """Prove that channel purchase clocks follow exactly from raw columns."""
    b = purchase_events("all")
    r = b.select([
        pl.len().alias("purchase_days"),
        (pl.col("gmv_search") > 0).sum().alias("search_purchase_days"),
        (pl.col("gmv_cat") > 0).sum().alias("catalog_purchase_days"),
        ((pl.col("gmv_search") > 0) & (pl.col("gmv_cat") > 0)).sum().alias("both_days"),
        (pl.col("gmv") - pl.col("gmv_search") - pl.col("gmv_cat")).abs()
        .max().alias("max_component_error"),
        ((pl.col("gmv") - pl.col("gmv_search") - pl.col("gmv_cat")).abs() > 1e-8)
        .sum().alias("component_mismatches"),
    ]).to_dicts()[0]
    assert r["component_mismatches"] == 0
    return r


def _last_gap_expr(k: int, name: str) -> pl.Expr:
    return (pl.col("gap").drop_nulls().reverse().get(k - 1, null_on_oob=True)
            .alias(name))


def _clock_table(events: pl.DataFrame, T: dt.date, prefix: str,
                 full: bool) -> pl.DataFrame:
    """Aggregate one recurrent-event sequence at ``T`` without future rows."""
    e = events.filter(pl.col("event_date") <= T)
    if e.is_empty():
        return pl.DataFrame({"user_id": []}, schema={"user_id": pl.Int64})
    e = e.with_columns(
        _rec=(pl.lit(T) - pl.col("event_date").max().over("user_id"))
        .dt.total_days().cast(pl.Float32))

    p = prefix
    expr: list[pl.Expr] = [
        pl.len().alias(f"{p}n_events"),
        pl.col("_rec").first().alias(f"{p}recency"),
        pl.col("gap").drop_nulls().mean().alias(f"{p}gap_mean"),
        pl.col("gap").drop_nulls().median().alias(f"{p}gap_median"),
        pl.col("gap").drop_nulls().std().alias(f"{p}gap_std"),
        _last_gap_expr(1, f"{p}gap_last1"),
        _last_gap_expr(2, f"{p}gap_last2"),
        _last_gap_expr(3, f"{p}gap_last3"),
    ]
    if full:
        expr += [
            _last_gap_expr(4, f"{p}gap_last4"),
            _last_gap_expr(5, f"{p}gap_last5"),
            pl.col("gap").drop_nulls().quantile(0.10).alias(f"{p}gap_q10"),
            pl.col("gap").drop_nulls().quantile(0.25).alias(f"{p}gap_q25"),
            pl.col("gap").drop_nulls().quantile(0.75).alias(f"{p}gap_q75"),
            pl.col("gap").drop_nulls().quantile(0.90).alias(f"{p}gap_q90"),
            pl.col("gap").drop_nulls().min().alias(f"{p}gap_min"),
            pl.col("gap").drop_nulls().max().alias(f"{p}gap_max"),
            (pl.col("gap") > pl.col("_rec")).sum().alias(f"{p}risk_at_recency"),
            ((pl.col("gap") > pl.col("_rec")) &
             (pl.col("gap") <= pl.col("_rec") + TARGET_DAYS))
            .sum().alias(f"{p}ends_next30"),
        ]
        for d in (7, 14, 30, 60, 90):
            tol = max(2, int(round(0.20 * d)))
            expr.append(((pl.col("gap") >= d - tol) & (pl.col("gap") <= d + tol))
                        .sum().alias(f"{p}near_{d}"))
    out = e.group_by("user_id", maintain_order=True).agg(expr)

    nint = (pl.col(f"{p}n_events") - 1).clip(lower_bound=0).cast(pl.Float32)
    mean = pl.col(f"{p}gap_mean")
    med = pl.col(f"{p}gap_median")
    std = pl.col(f"{p}gap_std")
    rec = pl.col(f"{p}recency")
    g1, g2, g3 = (pl.col(f"{p}gap_last{i}") for i in range(1, 4))
    derived = [
        nint.alias(f"{p}n_intervals"),
        (std / (mean + EPS)).alias(f"{p}gap_cv"),
        (1.0 / (1.0 + std / (mean + EPS))).alias(f"{p}regularity"),
        ((std - mean) / (std + mean + EPS)).alias(f"{p}burstiness"),
        ((0.50 * g1.fill_null(0.0) + 0.30 * g2.fill_null(0.0) +
          0.20 * g3.fill_null(0.0)) /
         (0.50 * g1.is_not_null().cast(pl.Float32) +
          0.30 * g2.is_not_null().cast(pl.Float32) +
          0.20 * g3.is_not_null().cast(pl.Float32) + EPS)).alias(f"{p}gap_ewma"),
        (rec / (med + EPS)).alias(f"{p}rec_over_median"),
        (rec / (mean + EPS)).alias(f"{p}rec_over_mean"),
        (rec - med).alias(f"{p}rec_minus_median"),
        (rec + g1).alias(f"{p}since_prev_purchase"),
        (rec + g1 + g2).alias(f"{p}since_third_purchase"),
    ]
    out = out.with_columns(derived)
    out = out.with_columns(
        (pl.col(f"{p}recency") / (pl.col(f"{p}gap_ewma") + EPS))
        .alias(f"{p}rec_over_ewma"))

    if full:
        g4, g5 = pl.col(f"{p}gap_last4"), pl.col(f"{p}gap_last5")
        out = out.with_columns([
            (pl.col(f"{p}gap_q75") - pl.col(f"{p}gap_q25")).alias(f"{p}gap_iqr"),
            ((1.5 * g1 + 0.5 * g2 - 0.5 * g3 - 1.5 * g4) / 5.0)
            .alias(f"{p}gap_trend"),
            (((g1 + g2) / 2.0) / ((g3 + g4 + g5) / 3.0 + EPS))
            .alias(f"{p}recent_old_ratio"),
        ])
        for d in (7, 14, 30, 60, 90):
            out = out.with_columns(
                (pl.col(f"{p}near_{d}") / (pl.col(f"{p}n_intervals") + EPS))
                .alias(f"{p}share_near_{d}"))
    return out


def _build_features_from_events(T: dt.date, all_users: pl.DataFrame,
                                all_events: pl.DataFrame,
                                search_events: pl.DataFrame,
                                catalog_events: pl.DataFrame) -> pl.DataFrame:
    """Pure implementation used by ``build_features`` and anti-leak tests."""
    avail = float((T - DATA_START).days + 1)
    a = _clock_table(all_events, T, "clk_", full=True)
    s = _clock_table(search_events, T, "sclk_", full=False)
    c = _clock_table(catalog_events, T, "cclk_", full=False)
    f = (all_users.select("user_id").unique().sort("user_id")
         .join(a, on="user_id", how="left")
         .join(s, on="user_id", how="left")
         .join(c, on="user_id", how="left"))
    count_cols = [x for x in f.columns if x.endswith(("n_events", "n_intervals"))
                  or "near_" in x or x.endswith(("risk_at_recency", "ends_next30"))]
    f = f.with_columns([pl.col(x).fill_null(0) for x in count_cols])
    for p in ("clk_", "sclk_", "cclk_"):
        f = f.with_columns(pl.col(f"{p}recency").fill_null(avail).alias(f"{p}recency"))
    f = f.with_columns([
        (pl.col("clk_n_events") == 0).cast(pl.Float32).alias("clk_cold_0"),
        (pl.col("clk_n_events") == 1).cast(pl.Float32).alias("clk_cold_1"),
        (pl.col("clk_n_events") == 2).cast(pl.Float32).alias("clk_cold_2"),
        (pl.col("clk_n_events") >= 3).cast(pl.Float32).alias("clk_hist_3plus"),
        (pl.col("sclk_n_events") / (pl.col("clk_n_events") + EPS)).alias("clk_search_share"),
        (pl.col("cclk_n_events") / (pl.col("clk_n_events") + EPS)).alias("clk_catalog_share"),
        (pl.col("sclk_recency") - pl.col("cclk_recency")).alias("clk_channel_rec_diff"),
        pl.lit(avail).cast(pl.Float32).alias("clk_available_days"),
    ])
    numeric = [x for x in f.columns if x != "user_id"]
    return f.with_columns([pl.col(x).cast(pl.Float32) for x in numeric]).sort("user_id")


def build_features(cutoff_date: dt.date) -> pl.DataFrame:
    """Build only purchase-timing features from history ``<= cutoff_date``."""
    # Only purchasers need an aggregated row.  ``make_frame`` performs a left
    # join to the exact panel and materializes the zero-purchase cold start.
    users = (purchase_events("all").lazy()
             .filter(pl.col("event_date") <= cutoff_date)
             .select("user_id").unique().collect())
    return _build_features_from_events(cutoff_date, users, purchase_events("all"),
                                       purchase_events("search"),
                                       purchase_events("catalog"))


def features_cached(T: dt.date) -> pl.DataFrame:
    """Compatibility wrapper; intentionally memory-only.

    The host volume is nearly full and the raw recurrent aggregation takes
    under a second after the event table is in memory.  Persisting 30 copies of
    the same histories would cost ~0.6 GB without changing the experiment.
    """
    return build_features(T)


def feature_names() -> list[str]:
    """Stable R1 feature order (obtained from one cached frame)."""
    f = features_cached(VAL_FOLDS_S1[-1])
    return [x for x in f.columns if x != "user_id"]


def _complete_panel_defaults(f: pl.DataFrame, T: dt.date) -> pl.DataFrame:
    """Materialize the no-purchase state after the panel left join."""
    avail = float((T - DATA_START).days + 1)
    count_cols = [x for x in f.columns if x.endswith(("n_events", "n_intervals"))
                  or "near_" in x or x.endswith(("risk_at_recency", "ends_next30"))]
    f = f.with_columns([pl.col(x).fill_null(0) for x in count_cols])
    for p in ("clk_", "sclk_", "cclk_"):
        f = f.with_columns(pl.col(f"{p}recency").fill_null(avail).alias(f"{p}recency"))
    return f.with_columns([
        (pl.col("clk_n_events") == 0).cast(pl.Float32).alias("clk_cold_0"),
        (pl.col("clk_n_events") == 1).cast(pl.Float32).alias("clk_cold_1"),
        (pl.col("clk_n_events") == 2).cast(pl.Float32).alias("clk_cold_2"),
        (pl.col("clk_n_events") >= 3).cast(pl.Float32).alias("clk_hist_3plus"),
        (pl.col("sclk_n_events") / (pl.col("clk_n_events") + EPS)).alias("clk_search_share"),
        (pl.col("cclk_n_events") / (pl.col("clk_n_events") + EPS)).alias("clk_catalog_share"),
        (pl.col("sclk_recency") - pl.col("cclk_recency")).alias("clk_channel_rec_diff"),
        pl.lit(avail).cast(pl.Float32).alias("clk_available_days"),
    ])


def _target_from_raw(T: dt.date, users: pl.DataFrame) -> pl.Series:
    """Same purchase target as ``features.target`` without loading 13 raw columns."""
    a, b = T + dt.timedelta(days=1), T + dt.timedelta(days=TARGET_DAYS)
    y = (pl.scan_parquet(RAW_PARQUET)
         .filter((pl.col("event_date") >= a) & (pl.col("event_date") <= b) &
                 (pl.col("gmv") > 0))
         .select("user_id", "gmv").group_by("user_id")
         .agg(pl.col("gmv").sum().alias("y")).collect())
    return (users.select("user_id").join(y, on="user_id", how="left")
            .with_columns(pl.col("y").fill_null(0.0)).sort("user_id"))["y"]


def make_frame(T: dt.date, blocks: int, with_target: bool = True) -> pl.DataFrame:
    """Panel -> ``build_features(T)`` -> target, matching production semantics."""
    u = panel_users(T, blocks)
    f = u.join(features_cached(T), on="user_id", how="left").sort("user_id")
    f = _complete_panel_defaults(f, T)
    if with_target:
        y = _target_from_raw(T, u)
        f = f.with_columns([y.alias("y"), (y > 0).cast(pl.Int8).alias("y_buy")])
    return f


def _to_matrix(frame: pl.DataFrame, feats: list[str]) -> np.ndarray:
    return frame.select(feats).to_numpy().astype(np.float32)


def assemble(cuts: list[dt.date], feats: list[str]) -> tuple[np.ndarray, np.ndarray]:
    sizes = [panel_users(T, 1).height for T in cuts]
    X = np.empty((sum(sizes), len(feats)), np.float32)
    y = np.empty(sum(sizes), np.int8)
    pos = 0
    for T, n in zip(cuts, sizes):
        f = make_frame(T, 1, True)
        assert f.height == n
        X[pos:pos+n] = _to_matrix(f, feats)
        y[pos:pos+n] = f["y_buy"].to_numpy()
        pos += n
    return X, y


def lgb_params(seed: int) -> dict:
    p = dict(LGB_PARAMS)
    p.update(objective="binary", metric="binary_logloss", learning_rate=0.05,
             num_leaves=31, min_data_in_leaf=500, feature_fraction=0.85,
             lambda_l2=10.0, seed=int(seed), verbosity=-1)
    return p


def cohort_code(median_gap: np.ndarray) -> np.ndarray:
    """Cohorts by already observed typical interval; missing -> global curve."""
    x = np.asarray(median_gap, float)
    out = np.full(len(x), -1, np.int8)
    ok = np.isfinite(x)
    out[ok] = np.digitize(x[ok], [14.0, 30.0, 60.0, 90.0]).astype(np.int8)
    return out


def km_curve(durations: np.ndarray, observed: np.ndarray, max_day: int) -> tuple[np.ndarray, np.ndarray]:
    """Discrete Kaplan–Meier S(t) and risk count, with durations in days."""
    d = np.clip(np.asarray(durations, int), 0, max_day)
    o = np.asarray(observed, bool)
    total = np.bincount(d, minlength=max_day + 1)
    events = np.bincount(d[o], minlength=max_day + 1)
    risk = np.cumsum(total[::-1])[::-1].astype(float)
    surv = np.ones(max_day + 1, float)
    s = 1.0
    for day in range(1, max_day + 1):
        if risk[day] > 0:
            s *= max(0.0, 1.0 - events[day] / risk[day])
        surv[day] = s
    return surv, risk


@dataclass
class R0Model:
    curves: dict[int, np.ndarray]
    risks: dict[int, np.ndarray]
    cold0_prior: float
    shrinkage: float
    limit: str

    def _cohort_probability(self, recency: np.ndarray, cohort: np.ndarray) -> np.ndarray:
        max_day = len(self.curves[-1]) - 1
        r = np.clip(np.asarray(recency, int), 0, max_day)
        r30 = np.clip(r + TARGET_DAYS, 0, max_day)
        out = np.zeros(len(r), float)
        for code in np.unique(cohort):
            m = cohort == code
            use = int(code) if int(code) in self.curves else -1
            curve, risk = self.curves[use], self.risks[use]
            # Sparse cohort tails fall back to the global KM curve.
            sparse = risk[r[m]] < 100
            p = 1.0 - curve[r30[m]] / np.maximum(curve[r[m]], 1e-9)
            if sparse.any() and use != -1:
                g, _ = self.curves[-1], self.risks[-1]
                p[sparse] = 1.0 - g[r30[m][sparse]] / np.maximum(g[r[m][sparse]], 1e-9)
            out[m] = np.clip(p, 0.0, 1.0)
        return out

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        n = frame["clk_n_events"].to_numpy()
        rec = frame["clk_recency"].to_numpy()
        med = frame["clk_gap_median"].to_numpy()
        risk_i = frame["clk_risk_at_recency"].to_numpy().astype(float)
        end_i = frame["clk_ends_next30"].to_numpy().astype(float)
        prior = self._cohort_probability(rec, cohort_code(med))
        out = (end_i + self.shrinkage * prior) / (risk_i + self.shrinkage)
        out[n == 0] = self.cold0_prior
        return np.clip(out, EPS, 1.0 - EPS).astype(np.float32)


def fit_r0(limit: dt.date, train_cuts: list[dt.date], shrinkage: float) -> R0Model:
    """Fit fold-specific recurrent-event distributions using history <= limit."""
    e = purchase_events("all").filter(pl.col("event_date") <= limit)
    users = e.group_by("user_id").agg([
        pl.col("gap").drop_nulls().median().alias("median_gap"),
        pl.col("event_date").max().alias("last_date"),
    ])
    gaps = (e.filter(pl.col("gap").is_not_null())
            .join(users.select("user_id", "median_gap"), on="user_id", how="left")
            .select("user_id", "gap", "median_gap"))
    cens = users.with_columns([
        (pl.lit(limit) - pl.col("last_date")).dt.total_days().cast(pl.Float32).alias("gap"),
        pl.lit(False).alias("observed"),
    ]).select("user_id", "gap", "median_gap", "observed")
    complete = gaps.with_columns(pl.lit(True).alias("observed"))
    samples = pl.concat([complete, cens], how="vertical")
    med = samples["median_gap"].to_numpy()
    codes = cohort_code(med)
    duration = samples["gap"].to_numpy()
    observed = samples["observed"].to_numpy()
    max_day = max(540, int(np.nanmax(duration)) + TARGET_DAYS + 1)
    curves, risks = {}, {}
    curves[-1], risks[-1] = km_curve(duration, observed, max_day)
    for code in range(5):
        m = codes == code
        if m.sum() >= 500:
            curves[code], risks[code] = km_curve(duration[m], observed[m], max_day)

    # First-purchase process is not a renewal interval: estimate only its cold
    # fallback from train labels, with Beta(1,1) smoothing.
    n0 = pos0 = 0
    for T in train_cuts:
        f = make_frame(T, 1, True)
        m = f["clk_n_events"].to_numpy() == 0
        n0 += int(m.sum())
        pos0 += int(f["y_buy"].to_numpy()[m].sum())
    cold = (pos0 + 1.0) / (n0 + 2.0)
    return R0Model(curves, risks, float(cold), float(shrinkage), str(limit))


def platt_crossfit(raw: np.ndarray, y: np.ndarray, cut: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """LOFO Platt calibration; each row is calibrated without its fold labels."""
    raw = np.clip(np.asarray(raw, float), EPS, 1 - EPS)
    x = logit(raw).reshape(-1, 1)
    out = np.empty(len(raw), float)
    info = []
    for fold in sorted(set(cut.tolist())):
        va = cut == fold
        tr = ~va
        m = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200)
        m.fit(x[tr], y[tr])
        out[va] = m.predict_proba(x[va])[:, 1]
        info.append(dict(fold=fold, intercept=float(m.intercept_[0]), slope=float(m.coef_[0, 0])))
    return np.clip(out, EPS, 1-EPS).astype(np.float32), info


def fit_platt_all(raw: np.ndarray, y: np.ndarray) -> LogisticRegression:
    x = logit(np.clip(np.asarray(raw, float), EPS, 1-EPS)).reshape(-1, 1)
    return LogisticRegression(C=1.0, solver="lbfgs", max_iter=200).fit(x, y)


def _save_primary_oof(out: dict[str, np.ndarray]) -> None:
    np.savez_compressed(ARTIFACTS / "oof_RENEWAL-01.npz", **out)


def run_cv(n_seeds: int = 3) -> dict[str, np.ndarray]:
    feats = feature_names()
    folds = get_folds(min_history=90)
    uid_all: list[np.ndarray] = []
    cut_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []
    ygmv_all: list[np.ndarray] = []
    p_seeds_all: list[list[np.ndarray]] = [[] for _ in range(n_seeds)]
    p_r0_all: dict[float, list[np.ndarray]] = {a: [] for a in SHRINKAGES}
    imp = np.zeros(len(feats), float)

    for fi, (tr_cuts, V) in enumerate(folds):
        vf = make_frame(V, 3, True)
        Xv = _to_matrix(vf, feats)
        yv = vf["y_buy"].to_numpy().astype(np.int8)
        uid_all.append(vf["user_id"].to_numpy())
        cut_all.append(np.full(vf.height, V.isoformat(), dtype="U10"))
        y_all.append(yv)
        ygmv_all.append(vf["y"].to_numpy().astype(np.float32))

        paths = [ARTIFACTS / f"model_RENEWAL-01_{_tag(V)}_s{seed}.txt"
                 for seed in SEEDS[:n_seeds]]
        missing = [p for p in paths if not p.exists()]
        ds = None
        if missing:
            say(f"fold {V}: assemble {len(tr_cuts)} train cutoffs; resume missing {len(missing)}")
            Xtr, ytr = assemble(tr_cuts, feats)
            ds = lgb.Dataset(Xtr, label=ytr, feature_name=feats,
                             params=lgb_params(SEEDS[0]))
            ds.construct()
            del Xtr, ytr
            gc.collect()
        else:
            say(f"fold {V}: resume from {len(paths)} saved models")
        for si, seed in enumerate(SEEDS[:n_seeds]):
            path = paths[si]
            if path.exists():
                model = lgb.Booster(model_file=str(path))
                say(f"fold {V}: R1 seed {seed} loaded")
            else:
                assert ds is not None
                say(f"fold {V}: R1 seed {seed} training")
                model = lgb.train(lgb_params(seed), ds, num_boost_round=ROUNDS)
                model.save_model(str(path))
            p_seeds_all[si].append(model.predict(Xv).astype(np.float32))
            imp += model.feature_importance("gain")
            del model
        del ds, Xv
        gc.collect()

        r0 = fit_r0(max(tr_cuts), tr_cuts, PRIMARY_SHRINKAGE)
        for alpha in SHRINKAGES:
            variant = R0Model(r0.curves, r0.risks, r0.cold0_prior, alpha, r0.limit)
            p_r0_all[alpha].append(variant.predict(vf))
        say(f"fold {V}: done")

    uid = np.concatenate(uid_all)
    cut = np.concatenate(cut_all)
    y = np.concatenate(y_all)
    ygmv = np.concatenate(ygmv_all)
    p_seed = np.vstack([np.concatenate(x) for x in p_seeds_all])
    p_raw = p_seed.mean(axis=0)
    p_clock, platt = platt_crossfit(p_raw, y, cut)
    p_r0 = np.concatenate(p_r0_all[PRIMARY_SHRINKAGE])
    out: dict[str, np.ndarray] = {
        "user_id": uid, "cutoff": cut, "y": ygmv, "y_buy": y,
        "p_clock_30": p_clock, "p_r1_raw": p_raw.astype(np.float32),
        "p_r0": p_r0.astype(np.float32), "p_r1_seeds": p_seed.astype(np.float32),
    }
    for alpha in SHRINKAGES:
        out[f"p_r0_a{int(alpha)}"] = np.concatenate(p_r0_all[alpha]).astype(np.float32)
    _save_primary_oof(out)
    order = np.argsort(-imp)
    (ARTIFACTS / "renewal_01_importance.csv").write_text(
        "feature,gain\n" + "\n".join(f"{feats[i]},{imp[i]:.8g}" for i in order),
        encoding="utf-8")
    (ARTIFACTS / "renewal_01_platt.json").write_text(
        json.dumps(platt, indent=1, ensure_ascii=False), encoding="utf-8")
    return out


def run_test(oof: dict[str, np.ndarray], n_seeds: int = 3) -> dict[str, np.ndarray]:
    feats = feature_names()
    cuts = cutoff_grid(90)
    say(f"test: assemble {len(cuts)} labeled cutoffs")
    Xtr, ytr = assemble(cuts, feats)
    tf = make_frame(CUTOFF_TEST, 3, False)
    Xt = _to_matrix(tf, feats)
    ds = lgb.Dataset(Xtr, label=ytr, feature_name=feats, params=lgb_params(SEEDS[0]))
    ds.construct()
    del Xtr, ytr
    gc.collect()
    ps = []
    for seed in SEEDS[:n_seeds]:
        say(f"test: R1 seed {seed}")
        model = lgb.train(lgb_params(seed), ds, num_boost_round=ROUNDS)
        ps.append(model.predict(Xt).astype(np.float32))
        model.save_model(str(ARTIFACTS / f"model_RENEWAL-01_TEST_s{seed}.txt"))
        del model
    raw = np.mean(ps, axis=0)
    calibrator = fit_platt_all(oof["p_r1_raw"], oof["y_buy"])
    p_clock = calibrator.predict_proba(logit(np.clip(raw, EPS, 1-EPS)).reshape(-1, 1))[:, 1]
    r0 = fit_r0(CORRIDOR_END, cuts, PRIMARY_SHRINKAGE)
    p_r0 = r0.predict(tf)
    ss = sample_submit()["user_id"].to_numpy()
    uid = tf["user_id"].to_numpy()
    assert np.array_equal(uid, ss), "test renewal predictions must follow sample_submit"
    out = {
        "user_id": uid, "p_clock_30": p_clock.astype(np.float32),
        "p_r1_raw": raw.astype(np.float32), "p_r0": p_r0.astype(np.float32),
        "p_r1_seeds": np.vstack(ps).astype(np.float32),
        "clk_n_events": tf["clk_n_events"].to_numpy().astype(np.float32),
        "clk_n_intervals": tf["clk_n_intervals"].to_numpy().astype(np.float32),
        "clk_recency": tf["clk_recency"].to_numpy().astype(np.float32),
        "clk_rec_over_median": tf["clk_rec_over_median"].to_numpy().astype(np.float32),
        "clk_regularity": tf["clk_regularity"].to_numpy().astype(np.float32),
    }
    np.savez_compressed(ARTIFACTS / "test_RENEWAL-01.npz", **out)
    return out


def warm_cache() -> None:
    cuts = sorted(set(cutoff_grid(90) + VAL_FOLDS_S1 + [CUTOFF_TEST]))
    for i, T in enumerate(cuts, 1):
        f = features_cached(T)
        say(f"features {i:02d}/{len(cuts)} {T}: {f.height:,} x {f.width-1}"
            + " checked (memory-only)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-artifacts", type=Path, default=ARTIFACTS,
                    help="каталог актуальных S1/DIST/SEQ OOF и mhz_val_*.npz")
    ap.add_argument("--stage", choices=["all", "cache", "cv", "test", "evaluate"],
                    default="all")
    ap.add_argument("--seeds", type=int, choices=[1, 2, 3], default=3)
    a = ap.parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    audit = audit_purchase_definition()
    (ARTIFACTS / "renewal_01_purchase_audit.json").write_text(
        json.dumps(audit, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    say("purchase audit", audit)

    if a.stage in ("all", "cache"):
        warm_cache()
        if a.stage == "cache":
            return
    if a.stage in ("all", "cv"):
        oof = run_cv(a.seeds)
        if a.stage == "cv":
            return
    else:
        d = np.load(ARTIFACTS / "oof_RENEWAL-01.npz", allow_pickle=False)
        oof = {k: d[k] for k in d.files}
    if a.stage in ("all", "test"):
        run_test(oof, a.seeds)
        if a.stage == "test":
            return
    if a.stage in ("all", "evaluate"):
        from src.renewal_eval import evaluate_experiment
        report = evaluate_experiment(oof, Path(a.baseline_artifacts))
        save_report(EXP_ID, report["rmsle_report"], extra=report["summary"])
        say("complete", report["summary"])


if __name__ == "__main__":
    main()
