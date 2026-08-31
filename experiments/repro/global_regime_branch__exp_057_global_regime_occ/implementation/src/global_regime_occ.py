"""EXP-057 GLOBAL-REGIME-OCC-RANK: cutoff-safe global monetization regime and
user movement relative to the population, as extra inputs to the fixed
occ_r10_fast recent-occurrence classifier.

Causal question
---------------
Does cutoff-safe *platform* state plus the user cross-sectional *trajectory*
carry occurrence signal that the user-local occ_r10_fast recipe does not
already contain?

Nothing here retrains CAP/UNC/DIST/SEQ/ETX or touches the downstream blend.
The only moving part is the occurrence feature matrix.

Design notes that matter when reading results
---------------------------------------------
* Global columns are CONSTANT inside one cutoff.  Within a validation fold every
  row shares them, so they cannot rank users directly; they act only through
  between-cutoff reweighting during training and through the user x global
  interactions of build_interactions.  The percentile trajectory of
  build_user_relative is the part that varies within a cutoff, which is why the
  preregistered hypothesis names it primary and the level percentile control.
* Many global columns that are constant per cutoff are also a perfect cutoff
  fingerprint.  That is what PLACEBO_OCC controls for: a cyclically shifted
  global vector is an equally good fingerprint but carries the wrong regime, so
  REAL - PLACEBO isolates regime information from cutoff identity.

Run:
    python src/global_regime_occ.py --stage global-state
    python src/global_regime_occ.py --stage support-audit
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from time import perf_counter as _time
from typing import Sequence

import numpy as np
import polars as pl

from src.config import ARTIFACTS, CORRIDOR_END, CUTOFF_TEST, cutoff_grid
from src.data import load

RUN = "GLOBAL_REGIME_OCC_EXP057"
OUT = ARTIFACTS / RUN

FOLDS = [dt.date(2025, 9, 4), dt.date(2025, 9, 18), dt.date(2025, 10, 2), dt.date(2025, 10, 16)]
FOLD_WEIGHTS = np.array([1.0, 2.0, 4.0, 8.0])

# --- preregistered feature space, frozen before any result is read -------------
GLOBAL_WINDOWS = (7, 14, 30, 60, 90)
GLOBAL_METRICS = ("users_row", "users_active", "users_buy", "users_search", "users_cart",
                  "searches", "cat", "carts", "orders", "gmv",
                  "buyer_rate", "order_rate", "conv_search_cart", "conv_cart_order",
                  "gmv_per_buyer", "gmv_per_order", "orders_per_buyer")
DYNAMIC_WINDOWS = (7, 14, 30)
# metrics compared between the last window and the window immediately before it
DYNAMIC_METRICS = ("users_row", "users_active", "users_buy", "orders", "gmv",
                   "searches", "carts", "buyer_rate", "gmv_per_buyer", "conv_cart_order")
# per-user block metrics that get a cross-sectional percentile
RANK_METRICS = ("gmv", "orders", "buy_days", "carts", "searches", "active_days")
BLOCK_DAYS = 30
EPS = 1e-9


def log(*a):
    print(*a, flush=True)


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()


# ---------------------------------------------------------------- global state
def _window_frame(df: pl.LazyFrame, hi: dt.date, days: int) -> pl.LazyFrame:
    """Rows in the half-open window (hi - days, hi].  Never looks past hi."""
    lo = hi - dt.timedelta(days=days)
    return df.filter((pl.col("event_date") > lo) & (pl.col("event_date") <= hi))


_ACTIVE = ((pl.col("searches") > 0) | (pl.col("cat") > 0)
           | (pl.col("to_cart") > 0) | (pl.col("to_ord") > 0))


def _block_totals(df: pl.LazyFrame, hi: dt.date, days: int) -> dict[str, float]:
    """Platform totals over the fixed competition universe = every user with a raw
    row in the window.  Pure aggregation of event_date <= hi, so cutoff-safe."""
    w = _window_frame(df, hi, days)
    agg = w.select([
        pl.col("user_id").n_unique().alias("users_row"),
        pl.col("user_id").filter(_ACTIVE).n_unique().alias("users_active"),
        pl.col("user_id").filter(pl.col("gmv") > 0).n_unique().alias("users_buy"),
        pl.col("user_id").filter(pl.col("searches") > 0).n_unique().alias("users_search"),
        pl.col("user_id").filter(pl.col("to_cart") > 0).n_unique().alias("users_cart"),
        pl.col("searches").sum().alias("searches"),
        pl.col("cat").sum().alias("cat"),
        pl.col("to_cart").sum().alias("carts"),
        pl.col("to_ord").sum().alias("orders"),
        pl.col("gmv").sum().alias("gmv"),
    ]).collect().to_dicts()[0]
    t = {k: float(v if v is not None else 0.0) for k, v in agg.items()}
    t["buyer_rate"] = t["users_buy"] / max(t["users_row"], 1.0)
    t["order_rate"] = t["orders"] / max(t["users_row"], 1.0)
    t["conv_search_cart"] = t["carts"] / max(t["searches"], EPS)
    t["conv_cart_order"] = t["orders"] / max(t["carts"], EPS)
    t["gmv_per_buyer"] = t["gmv"] / max(t["users_buy"], 1.0)
    t["gmv_per_order"] = t["gmv"] / max(t["orders"], EPS)
    t["orders_per_buyer"] = t["orders"] / max(t["users_buy"], 1.0)
    return t


def build_global_state(T: dt.date, df: pl.LazyFrame | None = None) -> dict[str, float]:
    """Level + preregistered dynamics of platform state at cutoff T.

    Contains no cutoff index, no date, no fold identifier: only quantities a
    forecaster standing at T could compute from history."""
    if df is None:
        df = load().lazy()
    out: dict[str, float] = {}
    for w in GLOBAL_WINDOWS:
        for k, v in _block_totals(df, T, w).items():
            out[f"g_w{w}_{k}"] = v
    for w in DYNAMIC_WINDOWS:
        last = _block_totals(df, T, w)
        prev = _block_totals(df, T - dt.timedelta(days=w), w)
        for m in DYNAMIC_METRICS:
            out[f"g_d{w}_dlog_{m}"] = float(np.log1p(max(last[m], 0.0))
                                            - np.log1p(max(prev[m], 0.0)))
    return out


def global_feature_names() -> list[str]:
    names = [f"g_w{w}_{k}" for w in GLOBAL_WINDOWS for k in GLOBAL_METRICS]
    names += [f"g_d{w}_dlog_{m}" for w in DYNAMIC_WINDOWS for m in DYNAMIC_METRICS]
    return names


def global_state_table(cutoffs: Sequence[dt.date]) -> pl.DataFrame:
    df = load().lazy()
    rows = []
    for T in cutoffs:
        s = build_global_state(T, df)
        rows.append({"cutoff": T.isoformat(), **s})
        log(f"  global state {T}  gmv30={s['g_w30_gmv']:,.0f}  "
            f"dlog30_gmv={s['g_d30_dlog_gmv']:+.4f}  buyer_rate30={s['g_w30_buyer_rate']:.4f}")
    return pl.DataFrame(rows).select(["cutoff"] + global_feature_names())


# ------------------------------------------------------------ user-relative state
def _user_block(df: pl.LazyFrame, hi: dt.date, days: int) -> pl.DataFrame:
    w = _window_frame(df, hi, days)
    return (w.group_by("user_id").agg([
        pl.col("gmv").sum().alias("gmv"),
        pl.col("to_ord").sum().alias("orders"),
        (pl.col("gmv") > 0).sum().alias("buy_days"),
        pl.col("to_cart").sum().alias("carts"),
        pl.col("searches").sum().alias("searches"),
        pl.len().alias("active_days"),
    ]).collect())


def percentile(values: np.ndarray) -> np.ndarray:
    """Deterministic average-rank percentile in (0, 1].

    Average ranks map the huge zero mass to one shared value instead of an
    arbitrary order-dependent spread, and a stable argsort makes repeated runs
    bitwise identical."""
    v = np.asarray(values, dtype=np.float64)
    n = v.size
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    order = np.argsort(v, kind="stable")
    sv = v[order]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sv[j] == sv[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * ((i + 1) + j)  # average of 1-based ranks
        i = j
    return ranks / float(n)


def user_blocks(T: dt.date, users: np.ndarray,
                df: pl.LazyFrame | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Current and previous 30d per-user blocks, aligned to `users` order."""
    if df is None:
        df = load().lazy()
    base = pl.DataFrame({"user_id": np.asarray(users, dtype=np.int64)}).sort("user_id")
    cur = base.join(_user_block(df, T, BLOCK_DAYS), on="user_id", how="left").sort("user_id")
    prev = base.join(_user_block(df, T - dt.timedelta(days=BLOCK_DAYS), BLOCK_DAYS),
                     on="user_id", how="left").sort("user_id")
    fill = [pl.col(m).fill_null(0).cast(pl.Float64) for m in RANK_METRICS]
    return cur.with_columns(fill), prev.with_columns(fill)


def build_user_relative(T: dt.date, users: np.ndarray,
                        df: pl.LazyFrame | None = None) -> pl.DataFrame:
    """Cross-sectional percentile of each user inside the population at T, for the
    current 30d block and the previous one, plus the trajectory delta.

    The percentile is taken over `users`, the cutoff own population, so it is a
    rank inside the same cross-section the model scores."""
    cur, prev = user_blocks(T, users, df)
    out: dict[str, np.ndarray] = {"user_id": cur["user_id"].to_numpy()}
    for tag, blk in (("cur", cur), ("prev", prev)):
        for m in RANK_METRICS:
            out[f"u_pct_{tag}_{m}"] = percentile(blk[m].to_numpy())
    for m in RANK_METRICS:
        out[f"u_pct_delta_{m}"] = out[f"u_pct_cur_{m}"] - out[f"u_pct_prev_{m}"]
    return pl.DataFrame(out)


def user_relative_feature_names() -> list[str]:
    return ([f"u_pct_cur_{m}" for m in RANK_METRICS]
            + [f"u_pct_prev_{m}" for m in RANK_METRICS]
            + [f"u_pct_delta_{m}" for m in RANK_METRICS])


# ------------------------------------------------------------------ interactions
INTERACTION_NAMES = (
    "x_user_gmv_over_platform_per_active",
    "x_user_order_rate_over_platform",
    "x_user_conv_over_platform",
    "x_pctdelta_gmv_X_rec_buy",
    "x_pctdelta_gmv_X_buy_days",
    "x_global_decline_X_rec_any",
    "x_global_decline_X_user_trend",
)


def build_interactions(cur: pl.DataFrame, prev: pl.DataFrame, pct: pl.DataFrame,
                       glob: dict[str, float], rec_buy: np.ndarray,
                       rec_any: np.ndarray) -> dict[str, np.ndarray]:
    """Seven preregistered user x platform crosses. No sweep, no search."""
    c = {m: cur[m].to_numpy().astype(np.float64) for m in RANK_METRICS}
    p_gmv = prev["gmv"].to_numpy().astype(np.float64)
    plat_gmv_per_active = glob["g_w30_gmv"] / max(glob["g_w30_users_active"], 1.0)
    plat_order_rate = glob["g_w30_orders"] / max(glob["g_w30_users_active"], 1.0)
    plat_conv = glob["g_w30_conv_search_cart"]
    decline = glob["g_d30_dlog_gmv"]
    d_gmv = pct["u_pct_delta_gmv"].to_numpy().astype(np.float64)
    rb = np.nan_to_num(np.asarray(rec_buy, dtype=np.float64), nan=400.0)
    ra = np.nan_to_num(np.asarray(rec_any, dtype=np.float64), nan=400.0)
    out = {
        "x_user_gmv_over_platform_per_active":
            np.log1p(c["gmv"]) - np.log1p(plat_gmv_per_active),
        "x_user_order_rate_over_platform":
            np.log1p(c["orders"] / np.maximum(c["active_days"], 1.0)) - np.log1p(plat_order_rate),
        "x_user_conv_over_platform":
            np.log1p(c["carts"] / np.maximum(c["searches"], 1.0)) - np.log1p(plat_conv),
        "x_pctdelta_gmv_X_rec_buy": d_gmv * np.log1p(rb),
        "x_pctdelta_gmv_X_buy_days": d_gmv * c["buy_days"],
        "x_global_decline_X_rec_any": decline * np.log1p(ra),
        "x_global_decline_X_user_trend": decline * (np.log1p(c["gmv"]) - np.log1p(p_gmv)),
    }
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def all_new_feature_names() -> list[str]:
    return global_feature_names() + user_relative_feature_names() + list(INTERACTION_NAMES)


# ---------------------------------------------------------------------- placebo
# Architecture-matched negative control.  Same columns, same marginal
# distributions, same missingness, same row count; only the user <-> regime
# correspondence is destroyed.
def placebo_global_map(cutoffs: Sequence[dt.date],
                       real: dict[dt.date, dict[str, float]]) -> dict[dt.date, dict[str, float]]:
    """Cyclic shift by one eligible cutoff backwards.

    Cutoff i receives the global vector of cutoff i-1 (the first receives the
    last).  Every global value that appears in the real arm still appears in the
    placebo arm exactly once, so scale and support are identical, but each cutoff
    is told the wrong regime."""
    cuts = list(cutoffs)
    if len(cuts) < 2:
        raise ValueError("placebo needs at least two eligible cutoffs")
    return {c: real[cuts[(i - 1) % len(cuts)]] for i, c in enumerate(cuts)}


def recency_bucket(rec_buy: np.ndarray) -> np.ndarray:
    """Fixed recency strata used by the placebo permutation."""
    rb = np.nan_to_num(np.asarray(rec_buy, dtype=np.float64), nan=400.0)
    return np.digitize(rb, [7.0, 15.0, 30.0, 60.0, 120.0, 240.0]).astype(np.int64)


def decile(values: np.ndarray) -> np.ndarray:
    """Deterministic decile of a value inside its own cross-section."""
    p = percentile(values)
    return np.clip((p * 10.0).astype(np.int64), 0, 9)


def placebo_permute(block: np.ndarray, strata: np.ndarray, seed: int) -> np.ndarray:
    """Permute whole rows of `block` inside each stratum.

    Rows move together, so every within-stratum marginal and the joint
    distribution among the permuted columns are preserved exactly; only the link
    to the user identity (and hence to the user local features and the label) is
    broken."""
    block = np.asarray(block, dtype=np.float64)
    strata = np.asarray(strata, dtype=np.int64)
    if block.shape[0] != strata.shape[0]:
        raise ValueError("row count mismatch between block and strata")
    out = np.empty_like(block)
    rng = np.random.default_rng(seed)
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        out[idx] = block[rng.permutation(idx)]
    return out


def placebo_strata(cutoff_code: int, gmv_cur: np.ndarray, rec_buy: np.ndarray) -> np.ndarray:
    """cutoff x recent-GMV decile x recency bucket, as required by the contract."""
    return (int(cutoff_code) * 100 + decile(gmv_cur) * 10 + recency_bucket(rec_buy)).astype(np.int64)


def write_manifest(path: Path) -> dict:
    man = {
        "run": RUN,
        "global_windows": list(GLOBAL_WINDOWS),
        "global_metrics": list(GLOBAL_METRICS),
        "dynamic_windows": list(DYNAMIC_WINDOWS),
        "dynamic_metrics": list(DYNAMIC_METRICS),
        "rank_metrics": list(RANK_METRICS),
        "block_days": BLOCK_DAYS,
        "global_features": global_feature_names(),
        "user_relative_features": user_relative_feature_names(),
        "interactions": list(INTERACTION_NAMES),
        "n_global": len(global_feature_names()),
        "n_user_relative": len(user_relative_feature_names()),
        "n_interactions": len(INTERACTION_NAMES),
        "n_total": len(all_new_feature_names()),
        "forbidden": ["cutoff index", "cutoff date", "fold id",
                      "any event after cutoff", "target window (T, T+30]"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(man, indent=2, ensure_ascii=False), encoding="utf-8")
    return man


# ----------------------------------------------------------------- occurrence arms
# Exact occ_r10_fast recipe, transcribed from the teammate's
# continue_best_bas_final6h.py: OccCfg("occ_r10_fast", 10, 55.0, 380, 31, 520, "all", .82)
# on top of the fixedstack "recent_hurdle" expert spec.
OCC_MAXCUTS = 10
OCC_TAU = 55.0
OCC_ROUNDS = 380
OCC_LEAVES = 31
OCC_MIN_LEAF = 520
OCC_FEATURE_FRACTION = 0.82
HURDLE_TAU = 105.0
HURDLE_ROUNDS = 520
COMMON_PARAMS = dict(learning_rate=.035, num_leaves=63, min_data_in_leaf=220,
                     feature_fraction=.82, bagging_fraction=.90, bagging_freq=1,
                     lambda_l2=14., lambda_l1=1., max_bin=127)
ARMS = ("BASE", "GLOBAL", "PLACEBO")


def occ_setup(tau: float = OCC_TAU):
    from src.train import Setup
    return Setup(L=0, min_history=90, step=7, panel_blocks=3, train_blocks=1,
                 model="two_part", rounds=HURDLE_ROUNDS, norm_long=True,
                 weight_tau=tau, params=dict(COMMON_PARAMS))


def occ_params(threads: int = 10) -> dict:
    return dict(objective="binary", metric="binary_logloss", learning_rate=.035,
                num_leaves=OCC_LEAVES, min_data_in_leaf=OCC_MIN_LEAF,
                feature_fraction=OCC_FEATURE_FRACTION, bagging_fraction=.90, bagging_freq=1,
                lambda_l2=14., lambda_l1=1., max_bin=127, verbosity=-1,
                num_threads=threads, seed=42)


def _cutoff_code(T: dt.date) -> int:
    return int(T.strftime("%Y%m%d"))


def new_block(T: dt.date, Xb, arm: str, glob_map: dict[dt.date, dict[str, float]],
              df: pl.LazyFrame) -> np.ndarray:
    """The 140 new columns for one cutoff, in `all_new_feature_names()` order.

    `Xb` is the cutoff's own feature frame, so the percentile is a rank inside
    exactly the cross-section that the model scores at `T`."""
    users = Xb["user_id"].to_numpy().astype(np.int64)
    n = users.size
    cur, prev = user_blocks(T, users, df)
    pctf = build_user_relative(T, users, df)
    pct_names = user_relative_feature_names()
    pct = np.column_stack([pctf[c].to_numpy() for c in pct_names])
    glob = glob_map[T]
    rec_buy = Xb["rec_buy"].to_numpy().astype(np.float64)
    rec_any = Xb["rec_any"].to_numpy().astype(np.float64)

    if arm == "PLACEBO":
        strata = placebo_strata(_cutoff_code(T), cur["gmv"].to_numpy(), rec_buy)
        pct = placebo_permute(pct, strata, seed=int(_cutoff_code(T)) % 2_147_483_647)
        pctf = pl.DataFrame({c: pct[:, i] for i, c in enumerate(pct_names)})

    inter = build_interactions(cur, prev, pctf, glob, rec_buy, rec_any)
    gvec = np.array([glob[k] for k in global_feature_names()], dtype=np.float64)
    block = np.empty((n, len(all_new_feature_names())), dtype=np.float32)
    block[:, :gvec.size] = gvec.astype(np.float32)
    block[:, gvec.size:gvec.size + pct.shape[1]] = pct.astype(np.float32)
    off = gvec.size + pct.shape[1]
    for j, name in enumerate(INTERACTION_NAMES):
        block[:, off + j] = inter[name].astype(np.float32)
    return np.nan_to_num(block, nan=0.0, posinf=20.0, neginf=-20.0)


def assemble_augmented(cuts, s, feats, V, arm, glob_map, df):
    """Mirror of train.assemble with the new block appended per cutoff.

    With arm=None it must be bitwise identical to train.assemble; that equality is
    asserted by test_base_arm_reproduces_plain_assemble."""
    from src.features import to_np
    from src.train import block_rows, xy
    extra = 0 if arm is None else len(all_new_feature_names())
    sizes = [block_rows(T, s) for T in cuts]
    X = np.empty((sum(sizes), len(feats) + extra), np.float32)
    ys, ws, i = [], [], 0
    for T, n in zip(cuts, sizes):
        Xb, y = xy(T, s, blocks=s.train_blocks)
        A = to_np(Xb, feats)
        assert A.shape[0] == n, f"{T}: rows {A.shape[0]} vs panel {n}"
        X[i:i + n, :len(feats)] = A
        if extra:
            X[i:i + n, len(feats):] = new_block(T, Xb, arm, glob_map, df)
        i += n
        del A
        ys.append(y)
        w = float(np.exp(-((V - T).days) / s.weight_tau)) if (s.weight_tau and V) else 1.0
        ws.append(np.full(n, w, np.float32))
    return X, np.concatenate(ys), np.concatenate(ws)


def load_global_maps() -> tuple[dict, dict, list[dt.date]]:
    tab = pl.read_parquet(OUT / "global_state.parquet")
    names = global_feature_names()
    cuts = [dt.date.fromisoformat(c) for c in tab["cutoff"].to_list()]
    real = {c: {n: float(tab[n][i]) for n in names} for i, c in enumerate(cuts)}
    eligible = [c for c in cuts if c <= CORRIDOR_END]
    plac = placebo_global_map(eligible, real)
    plac[CUTOFF_TEST] = real[eligible[-1]]           # test also gets a shifted vector
    return real, plac, eligible


# ------------------------------------------------------- downstream overlay
# Verbatim transcription of the occurrence overlay that produced the LB-useful
# X3 component (continue_best_bas_final6h.py: p_apply / fit_occ_params_on_past /
# walk_occ_candidate).  Identical for every arm; only p_new changes.
TABLE_WEIGHT = 0.55
BASE_COMPONENTS = ("S1-E03a", "S1-E02", "S1-DIST", "ETX-AVG3", "SEQ-AVG3")
BASE_WEIGHTS = (0.10, 0.20, 0.25, 0.225, 0.225)
CORE_TABLE = {"S1-E03a": 0.10 / TABLE_WEIGHT, "S1-E02": 0.20 / TABLE_WEIGHT,
              "S1-DIST": 0.25 / TABLE_WEIGHT}
REPO_ARTIFACTS = Path(r"C:/Users/Admin/Desktop/OZON-E-CUP/artifacts")
FIXED_OCC_PARAMS = (-.08, .75, .12, .025)


def clipz(z):
    return np.clip(np.nan_to_num(np.asarray(z, np.float64), nan=0.0, posinf=20.0, neginf=0.0),
                   0.0, 20.0)


def sigmoid(x):
    x = np.clip(np.asarray(x, np.float64), -35, 35)
    return 1 / (1 + np.exp(-x))


def logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def p_apply(base_table, p_base, mu, p_new, down=1.0, up=.15, shift=0.0, threshold=None):
    """Asymmetric occurrence correction: a drop in predicted occurrence is trusted
    `down` times, a rise only `up` times, and the move is scaled by the
    conditional magnitude mu."""
    pp = sigmoid(logit(p_new) + float(shift))
    delta = pp - np.asarray(p_base, np.float64)
    strength = np.where(delta < 0, float(down), float(up))
    if threshold is not None:
        strength = np.where(np.abs(delta) >= float(threshold), strength, 0.0)
    return clipz(np.asarray(base_table, np.float64) + strength * delta * np.asarray(mu, np.float64))


def fit_occ_params_on_past(bank, ids, arm):
    """Same small regularized grid, fitted only on folds strictly before the one
    being scored.  Walk-forward: no fold ever tunes its own overlay."""
    best = None
    for sh in (-.22, -.14, -.08, 0.0, .06):
        for dn in (.45, .65, .85, 1.0):
            for up in (.05, .12, .22):
                for th in (None, .025, .05):
                    num = den = 0.0
                    for j in ids:
                        r = bank[FOLDS[j]]
                        z = p_apply(r["table_core"], r["p"], r["mu"], r[f"p_{arm}"],
                                    dn, up, sh, th)
                        num += FOLD_WEIGHTS[j] * float(np.mean((z - r["true_z"]) ** 2))
                        den += FOLD_WEIGHTS[j]
                    if den == 0:
                        continue
                    obj = num / den + .0015 * sh * sh + .00025 * (dn - .75) ** 2                         + .00020 * (up - .12) ** 2
                    if best is None or obj < best[0]:
                        best = (obj, sh, dn, up, th)
    return best[1:]


def walk_occ_candidate(bank, arm):
    out, pars = {}, []
    for i, f in enumerate(FOLDS):
        r = bank[f]
        pa = fit_occ_params_on_past(bank, list(range(i)), arm) if i > 0 else FIXED_OCC_PARAMS
        sh, dn, up, th = pa
        out[f] = p_apply(r["table_core"], r["p"], r["mu"], r[f"p_{arm}"], dn, up, sh, th)
        pars.append(pa)
    return out, pars


def _fold_component(name: str, fold: dt.date) -> dict[str, np.ndarray]:
    d = np.load(REPO_ARTIFACTS / f"oof_{name}.npz", allow_pickle=False)
    m = np.asarray(d["cutoff"], dtype="U10") == fold.isoformat()
    o = np.argsort(d["user_id"][m], kind="stable")
    return {k: d[k][m][o] for k in ("user_id", "y", "z")}


def load_bank() -> dict:
    """Exactly reconstructible OOF base: the friend blend, its 55% tabular slot,
    and the hurdle p / mu the overlay needs."""
    bank = {}
    for f in FOLDS:
        parts = {n: _fold_component(n, f) for n in BASE_COMPONENTS}
        uid = parts[BASE_COMPONENTS[0]]["user_id"].astype(np.int64)
        y = parts[BASE_COMPONENTS[0]]["y"].astype(np.float64)
        for n, p in parts.items():
            if not np.array_equal(p["user_id"].astype(np.int64), uid):
                raise AssertionError(f"{f}: component {n} is not user-aligned")
            if not np.array_equal(p["y"].astype(np.float64), y):
                raise AssertionError(f"{f}: component {n} carries a different target")
        friend = np.average(np.vstack([parts[n]["z"].astype(np.float64) for n in BASE_COMPONENTS]),
                            axis=0, weights=BASE_WEIGHTS)
        table_core = sum(w * parts[n]["z"].astype(np.float64) for n, w in CORE_TABLE.items())
        h = np.load(OUT / f"hurdle_{f:%Y%m%d}.npz", allow_pickle=False)
        if not np.array_equal(h["user_id"].astype(np.int64), uid):
            raise AssertionError(f"{f}: hurdle is not user-aligned with the OOF bank")
        r = {"uid": uid, "y": y, "true_z": np.log1p(np.maximum(y, 0.0)),
             "friend": friend, "table_core": table_core,
             "p": np.clip(h["p"].astype(np.float64), 1e-7, 1 - 1e-7),
             "mu": np.maximum(h["mu"].astype(np.float64), 0.0)}
        for arm in ARMS:
            path = OUT / f"occ_{arm}_{f:%Y%m%d}.npz"
            if not path.exists():
                continue
            d = np.load(path, allow_pickle=False)
            if not np.array_equal(d["user_id"].astype(np.int64), uid):
                raise AssertionError(f"{f}: occ {arm} is not user-aligned")
            r[f"p_{arm}"] = np.clip(d["p"].astype(np.float64), 1e-7, 1 - 1e-7)
        bank[f] = r
    return bank


# ------------------------------------------------------------------------ stages
def stage_global_state() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    man = write_manifest(OUT / "feature_manifest.json")
    log(f"feature manifest: {man['n_global']} global + {man['n_user_relative']} percentile "
        f"+ {man['n_interactions']} interactions = {man['n_total']}")
    cuts = [c for c in cutoff_grid(min_history=90) if c <= CORRIDOR_END] + [CUTOFF_TEST]
    log(f"computing global state on {len(cuts)} cutoffs "
        f"({cuts[0]} .. {cuts[-2]} + TEST {cuts[-1]})")
    tab = global_state_table(cuts)
    tab.write_parquet(OUT / "global_state.parquet")
    tab.write_csv(OUT / "global_state.csv")
    log(f"saved {OUT / 'global_state.parquet'}  shape={tab.shape}")


def stage_support_audit() -> None:
    tab = pl.read_parquet(OUT / "global_state.parquet")
    names = global_feature_names()
    cut = tab["cutoff"].to_list()
    test_i = cut.index(CUTOFF_TEST.isoformat())
    clean = [i for i in range(len(cut)) if i != test_i]
    # the occurrence recipe trains on the last 10 eligible cutoffs before a fold,
    # so also report support against the tightest window actually used
    last10 = clean[-10:]
    rep = {"n_clean_cutoffs": len(clean), "test_cutoff": CUTOFF_TEST.isoformat(),
           "clean_range": [cut[clean[0]], cut[clean[-1]]], "features": {}}
    outside, outside10 = [], []
    for n in names:
        v = tab[n].to_numpy().astype(np.float64)
        tr, te = v[clean], v[test_i]
        t10 = v[last10]
        lo, hi = float(tr.min()), float(tr.max())
        sd = float(tr.std())
        zs = float((te - tr.mean()) / sd) if sd > 0 else 0.0
        inside = bool(lo <= te <= hi)
        inside10 = bool(float(t10.min()) <= te <= float(t10.max()))
        rep["features"][n] = {"train_min": lo, "train_max": hi, "train_mean": float(tr.mean()),
                              "train_std": sd, "test": float(te), "z": zs,
                              "in_support_all29": inside, "in_support_last10": inside10}
        if not inside:
            outside.append((n, zs, lo, hi, float(te)))
        if not inside10:
            outside10.append(n)
    outside.sort(key=lambda r: -abs(r[1]))
    rep["n_outside_support_all29"] = len(outside)
    rep["n_outside_support_last10"] = len(outside10)
    rep["share_outside_support_all29"] = len(outside) / len(names)
    rep["share_outside_support_last10"] = len(outside10) / len(names)
    rep["worst_outside"] = [{"feature": n, "z": z, "train_min": lo, "train_max": hi, "test": te}
                            for n, z, lo, hi, te in outside[:30]]
    (OUT / "global_support_audit.json").write_text(
        json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"global features: {len(names)}")
    log(f"outside support of all 29 clean cutoffs at TEST: {len(outside)} "
        f"({100 * len(outside) / len(names):.1f}%)")
    log(f"outside support of the last 10 (the occ training window): {len(outside10)} "
        f"({100 * len(outside10) / len(names):.1f}%)")
    for n, z, lo, hi, te in outside[:20]:
        log(f"  {n:38s} z={z:+8.2f}  train[{lo:,.4f} .. {hi:,.4f}]  test={te:,.4f}")


def stage_hurdle(threads: int = 10) -> None:
    """Reference occurrence probability p and conditional magnitude mu.

    Exact `recent_hurdle` expert of run_best_bas_fixedstack_14h_v2.py: the overlay
    needs p and mu, and they must come from the same recipe for every arm."""
    import lightgbm as lgb
    from src.features import to_np
    from src.train import select_features, xy
    from src.features import feature_names
    OUT.mkdir(parents=True, exist_ok=True)
    s = occ_setup(HURDLE_TAU)
    for V in FOLDS:
        path = OUT / f"hurdle_{V:%Y%m%d}.npz"
        if path.exists():
            log(f"hurdle {V}: reuse")
            continue
        t0 = _time()
        Xv, yv = xy(V, s)
        feats = select_features(feature_names(Xv), s.drop_groups, s.keep_only)
        cuts = list(s.train_cutoffs(V))
        X, y, w = assemble_augmented(cuts, s, feats, V, None, {}, None)
        log(f"hurdle {V}: {len(cuts)} cutoffs, {X.shape[0]:,} rows, {len(feats)} features")
        pc = dict(COMMON_PARAMS, objective="binary", metric="binary_logloss",
                  verbosity=-1, num_threads=threads, seed=42)
        pr = dict(COMMON_PARAMS, objective="regression", metric="rmse",
                  verbosity=-1, num_threads=threads, seed=42)
        ds = lgb.Dataset(X, (y > 0).astype(np.int8), weight=w, params=pc,
                         free_raw_data=True).construct()
        clf = lgb.train(pc, ds, num_boost_round=HURDLE_ROUNDS)
        del ds
        pos = y > 0
        Xp, yp, wp = np.asarray(X[pos], np.float32), np.log1p(y[pos].astype(np.float64)), w[pos]
        del X
        ds = lgb.Dataset(Xp, yp, weight=wp, params=pr, free_raw_data=True).construct()
        del Xp
        reg = lgb.train(pr, ds, num_boost_round=HURDLE_ROUNDS)
        del ds
        A = to_np(Xv, feats)
        p = np.clip(clf.predict(A), 1e-7, 1 - 1e-7)
        mu = np.maximum(reg.predict(A), 0.0)
        np.savez_compressed(path, user_id=Xv["user_id"].to_numpy().astype(np.int64),
                            y=np.asarray(yv, np.float32), p=p.astype(np.float32),
                            mu=mu.astype(np.float32))
        log(f"hurdle {V}: done in {_time() - t0:.0f}s  mean_p={p.mean():.4f} mean_mu={mu.mean():.3f}")


def stage_occ(arm: str, threads: int = 10) -> None:
    """One occurrence arm across the four folds."""
    import lightgbm as lgb
    from src.features import feature_names, to_np
    from src.train import select_features, xy
    if arm not in ARMS:
        raise KeyError(arm)
    OUT.mkdir(parents=True, exist_ok=True)
    real, plac, _ = load_global_maps()
    gmap = {"BASE": {}, "GLOBAL": real, "PLACEBO": plac}[arm]
    df = None if arm == "BASE" else load().lazy()
    s = occ_setup(OCC_TAU)
    params = occ_params(threads)
    for V in FOLDS:
        path = OUT / f"occ_{arm}_{V:%Y%m%d}.npz"
        if path.exists():
            log(f"occ {arm} {V}: reuse")
            continue
        t0 = _time()
        Xv, yv = xy(V, s)
        feats = select_features(feature_names(Xv), s.drop_groups, s.keep_only)
        cuts = list(s.train_cutoffs(V))[-OCC_MAXCUTS:]
        a = None if arm == "BASE" else arm
        X, y, w = assemble_augmented(cuts, s, feats, V, a, gmap, df)
        log(f"occ {arm} {V}: {len(cuts)} cutoffs, {X.shape[0]:,} rows, {X.shape[1]} columns")
        ds = lgb.Dataset(X, (y > 0).astype(np.int8), weight=w, params=params,
                         free_raw_data=True).construct()
        del X
        model = lgb.train(params, ds, num_boost_round=OCC_ROUNDS)
        del ds
        A = to_np(Xv, feats)
        if a is not None:
            A = np.column_stack([A, new_block(V, Xv, a, gmap, df)])
        p = np.clip(model.predict(A), 1e-7, 1 - 1e-7)
        np.savez_compressed(path, user_id=Xv["user_id"].to_numpy().astype(np.int64),
                            y=np.asarray(yv, np.float32), p=p.astype(np.float32))
        log(f"occ {arm} {V}: done in {_time() - t0:.0f}s  mean_p={p.mean():.4f}")



def _metrics(y, z):
    from src.validation import calibrate, rmsle_z
    off, cal = calibrate(y, z)
    return {"rmsle_raw": rmsle_z(y, z), "rmsle_cal": cal, "offset": float(off),
            "mean_z": float(np.mean(z))}


def _fold_frame(V: dt.date):
    """rec_buy / rec_any / tenure_frac for the fold's own scored cross-section."""
    from src.train import xy
    Xv, _ = xy(V, occ_setup(OCC_TAU))
    return (Xv["user_id"].to_numpy().astype(np.int64),
            Xv["rec_buy"].to_numpy().astype(np.float64),
            Xv["tenure_frac"].to_numpy().astype(np.float64),
            Xv["w180_days_buy"].to_numpy().astype(np.float64))


def stage_evaluate() -> None:
    import json
    from sklearn.metrics import roc_auc_score
    from src.validation import calibrate, rmsle_z, wcv
    bank = load_bank()
    arms = [a for a in ARMS if all(f"p_{a}" in bank[f] for f in FOLDS)]
    log(f"arms available: {arms}")
    cand, pars = {}, {}
    for a in arms:
        cand[a], pars[a] = walk_occ_candidate(bank, a)

    rep = {"arms": arms, "overlay_params": {a: [list(map(lambda x: x, q)) for q in pars[a]]
                                            for a in arms},
           "table_weight": TABLE_WEIGHT, "folds": {}, "wcv": {}, "contrasts": {}}
    fold_final = {a: [] for a in arms}
    fold_slot = {a: [] for a in arms}
    friend_scores = []

    for i, f in enumerate(FOLDS):
        r = bank[f]
        uid, rec_buy, tenure_frac, w180_buy = _fold_frame(f)
        if not np.array_equal(uid, r["uid"]):
            raise AssertionError(f"{f}: fold frame is not aligned with the OOF bank")
        y, tz = r["y"], r["true_z"]
        pos = y > 0
        fr = _metrics(y, r["friend"])
        friend_scores.append(fr["rmsle_cal"])
        entry = {"n": int(len(uid)), "pos_rate": float(pos.mean()), "friend": fr,
                 "table_core": _metrics(y, r["table_core"]), "arms": {}}
        for a in arms:
            c = cand[a][f]
            final = r["friend"] + TABLE_WEIGHT * (c - r["table_core"])
            m_final, m_slot = _metrics(y, final), _metrics(y, c)
            fold_final[a].append(m_final["rmsle_cal"])
            fold_slot[a].append(m_slot["rmsle_cal"])
            corr = c - r["table_core"]
            resid = tz - r["friend"]
            off_f, _ = calibrate(y, final)
            zc = np.maximum(final + off_f, 0.0)
            d = {"final": m_final, "slot": m_slot,
                 "auc_activity": float(roc_auc_score(pos.astype(int), r[f"p_{a}"])),
                 "mean_p": float(r[f"p_{a}"].mean()),
                 "correction_var": float(np.var(corr)),
                 "correction_mean": float(np.mean(corr)),
                 "correction_abs_p99": float(np.quantile(np.abs(corr), 0.99)),
                 "extreme_share_gt_0p10": float((np.abs(corr) > 0.10).mean()),
                 "corr_correction_residual": float(np.corrcoef(corr, resid)[0, 1]),
                 "rmsle_zero": float(np.sqrt(np.mean(zc[~pos] ** 2))),
                 "rmsle_pos": float(np.sqrt(np.mean((zc[pos] - tz[pos]) ** 2))),
                 "overlay_params": list(pars[a][i]),
                 "segments": {}}
            for tag, mask in (("rec_buy_0_7", rec_buy <= 7),
                              ("rec_buy_8_14", (rec_buy > 7) & (rec_buy <= 14)),
                              ("rec_buy_15_60", (rec_buy > 14) & (rec_buy <= 60)),
                              ("rec_buy_61_plus", rec_buy > 60),
                              ("never_bought", ~np.isfinite(rec_buy)),
                              ("tenure_low", tenure_frac <= np.quantile(tenure_frac, 0.25)),
                              ("tenure_high", tenure_frac >= np.quantile(tenure_frac, 0.75)),
                              ("hist_support_0", w180_buy <= 0),
                              ("hist_support_1_3", (w180_buy >= 1) & (w180_buy <= 3)),
                              ("hist_support_4_plus", w180_buy >= 4)):
                if mask.sum() < 500:
                    continue
                d["segments"][tag] = {"n": int(mask.sum()),
                                      "rmsle": float(np.sqrt(np.mean((zc[mask] - tz[mask]) ** 2)))}
            entry["arms"][a] = d
        rep["folds"][f.isoformat()] = entry

    rep["wcv"]["friend"] = wcv(friend_scores)
    for a in arms:
        rep["wcv"][f"final_{a}"] = wcv(fold_final[a])
        rep["wcv"][f"slot_{a}"] = wcv(fold_slot[a])
    for a in arms:
        rep["contrasts"][f"final_{a}_minus_friend"] = rep["wcv"][f"final_{a}"] - rep["wcv"]["friend"]
    def contrast(x, y_):
        if x not in arms or y_ not in arms:
            return None
        per = [fold_final[x][i] - fold_final[y_][i] for i in range(4)]
        return {"wcv_delta": rep["wcv"][f"final_{x}"] - rep["wcv"][f"final_{y_}"],
                "fold_deltas": per, "wins": int(sum(v < 0 for v in per)),
                "latest_delta": per[-1],
                "slot_wcv_delta": rep["wcv"][f"slot_{x}"] - rep["wcv"][f"slot_{y_}"]}
    rep["contrasts"]["GLOBAL_minus_BASE"] = contrast("GLOBAL", "BASE")
    rep["contrasts"]["GLOBAL_minus_PLACEBO"] = contrast("GLOBAL", "PLACEBO")
    rep["contrasts"]["PLACEBO_minus_BASE"] = contrast("PLACEBO", "BASE")
    (OUT / "fold_results.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False),
                                           encoding="utf-8")

    log("")
    log(f"friend (STRONGEST_CURRENT) wCV = {rep['wcv']['friend']:.10f}")
    for a in arms:
        log(f"  final_{a:8s} wCV = {rep['wcv'][f'final_{a}']:.10f}   "
            f"vs friend {rep['contrasts'][f'final_{a}_minus_friend']:+.6f}")
    log("")
    for k in ("GLOBAL_minus_BASE", "GLOBAL_minus_PLACEBO", "PLACEBO_minus_BASE"):
        c = rep["contrasts"][k]
        if c is None:
            continue
        log(f"{k:22s} dwCV={c['wcv_delta']:+.6f}  wins {c['wins']}/4  "
            f"10-16 {c['latest_delta']:+.6f}  folds "
            + " ".join(f"{v:+.6f}" for v in c["fold_deltas"]))
    log("")
    for a in arms:
        aucs = [rep["folds"][f.isoformat()]["arms"][a]["auc_activity"] for f in FOLDS]
        log(f"AUC({a:8s}) = " + " ".join(f"{v:.6f}" for v in aucs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["global-state", "support-audit", "hurdle", "occ", "evaluate"])
    ap.add_argument("--arm", choices=list(ARMS))
    ap.add_argument("--threads", type=int, default=10)
    a = ap.parse_args()
    if a.stage == "global-state":
        stage_global_state()
    elif a.stage == "support-audit":
        stage_support_audit()
    elif a.stage == "hurdle":
        stage_hurdle(a.threads)
    elif a.stage == "occ":
        if not a.arm:
            raise SystemExit("--arm is required for --stage occ")
        stage_occ(a.arm, a.threads)
    elif a.stage == "evaluate":
        stage_evaluate()


if __name__ == "__main__":
    main()
