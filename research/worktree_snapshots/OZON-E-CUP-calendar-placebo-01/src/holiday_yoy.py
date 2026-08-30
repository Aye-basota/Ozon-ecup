"""HOLIDAY-YOY — персональная сезонность 14.02–15.03.2025.

High-risk эксперимент поверх S1-DIST-MIX. Core-пайплайн не меняется: этот файл
добавляет cutoff-safe признаки отдельным join'ом и подменяет только слот S1-DIST
при неизменных весах 0.15/0.30/0.10/0.45. S1-CAP (E03a) сохраняется.

Одна точка входа:
    python src/holiday_yoy.py --stage diagnostic
    python src/holiday_yoy.py --stage cv
    python src/holiday_yoy.py --stage analyze
    python src/holiday_yoy.py --stage predict
    python src/holiday_yoy.py --stage submission
    python src/holiday_yoy.py                    # всё последовательно

Новые model features строятся только через build_features(cutoff_date). Для
cutoff до 2025-04-14 post-сосед праздника ещё не наблюдаем, поэтому все
двухсторонние *_nbr колонки равны NaN; используются только *_pre. Ни одна
колонка не читает event_date > cutoff_date.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (ARTIFACTS, CUTOFF_TEST, FOLD_WEIGHTS_S1, ROOT, SEED,
                        SUBMISSIONS, VAL_FOLDS_S1)
from src.data import load, sample_submit
from src.features import (feature_names, make_xy, panel_users, to_np)
from src.report import evaluate, format_report, save_report
from src.tracking import load_oof, save_oof
from src.validation import calibrate, rmsle_z

PREFIX = "HOLIDAY-YOY"
COMPONENT_ID = f"{PREFIX}-DIST"
RESULTS = ROOT / "research" / "strategies" / "results" / PREFIX
LEVEL = 2.3293
WEIGHTS = {"S1-E10": 0.15, "S1-E02": 0.30, "S1-E03a": 0.10, "S1-DIST": 0.45}
TEST_VARIANTS = {"S1-E10": "S1-NORM", "S1-E02": "S1-UNC",
                 "S1-E03a": "S1-CAP", "S1-DIST": "S1-DIST"}

HOLIDAY = (dt.date(2025, 2, 14), dt.date(2025, 3, 15))
PRE = (dt.date(2025, 1, 15), dt.date(2025, 2, 13))
POST = (dt.date(2025, 3, 16), dt.date(2025, 4, 14))

BASE_METRICS = ("gmv", "orders", "days_buy", "cart", "searches", "catalog")
CHANNELS = ("gmv", "orders", "cart")
HY_COLS = ([f"hy_{m}_{kind}" for m in BASE_METRICS
            for kind in ("pre", "rel_pre", "nbr", "rel")]
           + [f"hy_search_catalog_{m}_{kind}" for m in CHANNELS
              for kind in ("pre", "rel_pre", "nbr", "rel")]
           + ["hy_purchase_score", "hy_no_holiday_history",
              "hy_positive_history", "hy_negative_history"])

_RAW_CACHE: dict[str, pl.DataFrame] = {}


def _log(msg: str) -> None:
    print(msg, flush=True)


def _mask(a: dt.date, b: dt.date) -> pl.Expr:
    return (pl.col("event_date") >= a) & (pl.col("event_date") <= b)


def _period_exprs(tag: str, a: dt.date, b: dt.date) -> list[pl.Expr]:
    m = _mask(a, b)
    return [
        pl.when(m).then(pl.col("gmv")).otherwise(0.0).sum().alias(f"gmv_{tag}"),
        pl.when(m).then(pl.col("to_ord")).otherwise(0).sum().alias(f"orders_{tag}"),
        (m & (pl.col("gmv") > 0)).sum().alias(f"days_buy_{tag}"),
        pl.when(m).then(pl.col("to_cart")).otherwise(0).sum().alias(f"cart_{tag}"),
        pl.when(m).then(pl.col("searches")).otherwise(0).sum().alias(f"searches_{tag}"),
        pl.when(m).then(pl.col("cat")).otherwise(0).sum().alias(f"catalog_{tag}"),
        pl.when(m).then(pl.col("gmv_search")).otherwise(0.0).sum().alias(f"search_gmv_{tag}"),
        pl.when(m).then(pl.col("gmv_cat")).otherwise(0.0).sum().alias(f"catalog_gmv_{tag}"),
        pl.when(m).then(pl.col("search_to_ord")).otherwise(0).sum().alias(f"search_orders_{tag}"),
        pl.when(m).then(pl.col("cat_to_ord")).otherwise(0).sum().alias(f"catalog_orders_{tag}"),
        pl.when(m).then(pl.col("search_to_cart")).otherwise(0).sum().alias(f"search_cart_{tag}"),
        pl.when(m).then(pl.col("cat_to_cart")).otherwise(0).sum().alias(f"catalog_cart_{tag}"),
    ]


def _holiday_raw(cutoff_date: dt.date) -> pl.DataFrame:
    """Фиксированные holiday/pre/post агрегаты, физически обрезанные по cutoff."""
    if cutoff_date < HOLIDAY[1]:
        raise ValueError(f"HOLIDAY-YOY недоступен на cutoff {cutoff_date}: holiday ещё не завершён")
    full = cutoff_date >= POST[1]
    key = "full" if full else "pre"
    if key in _RAW_CACHE:
        return _RAW_CACHE[key]

    periods = [("pre", *PRE), ("holiday", *HOLIDAY)]
    if full:
        periods.append(("post", *POST))
    max_date = max(b for _, _, b in periods)
    assert max_date <= cutoff_date, f"lookahead: source {max_date} > cutoff {cutoff_date}"
    exprs = [e for tag, a, b in periods for e in _period_exprs(tag, a, b)]
    raw = (load().lazy()
           .filter((pl.col("event_date") >= PRE[0]) & (pl.col("event_date") <= max_date)
                   & (pl.col("event_date") <= cutoff_date))
           .group_by("user_id").agg(exprs).collect().sort("user_id"))
    _RAW_CACHE[key] = raw
    return raw


def _positive_median(x: np.ndarray) -> float:
    p = np.asarray(x, float)
    p = p[np.isfinite(p) & (p > 0)]
    return max(float(np.median(p)) if len(p) else 1.0, 1.0)


def _shrunk_delta(h: np.ndarray, neighbors: list[np.ndarray], support: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
    """log-response пользователя, усаженный к популяционной реакции.

    Возвращается центрированное персональное отклонение. Поэтому глобальный
    сезонный уровень не дублируется и L*=2.3293 остаётся отдельным слоем.
    """
    raw = np.log1p(h.astype(float))
    raw -= np.mean(np.vstack([np.log1p(x.astype(float)) for x in neighbors]), axis=0)
    ok = np.isfinite(raw) & (support > 0)
    prior = float(np.median(raw[ok])) if ok.any() else 0.0
    tau = _positive_median(support)
    rel = support / (support + tau)
    return (rel * (raw - prior)).astype(np.float32), rel.astype(np.float32)


def _mix_delta(sh: np.ndarray, ch: np.ndarray,
               neighbors: list[tuple[np.ndarray, np.ndarray]], support: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray]:
    alpha = 0.5
    raw = np.log((sh.astype(float) + alpha) / (ch.astype(float) + alpha))
    raw -= np.mean(np.vstack([
        np.log((s.astype(float) + alpha) / (c.astype(float) + alpha))
        for s, c in neighbors
    ]), axis=0)
    ok = np.isfinite(raw) & (support > 0)
    prior = float(np.median(raw[ok])) if ok.any() else 0.0
    tau = _positive_median(support)
    rel = support / (support + tau)
    return (rel * (raw - prior)).astype(np.float32), rel.astype(np.float32)


def _arr(raw: pl.DataFrame, col: str) -> np.ndarray:
    return raw[col].to_numpy().astype(np.float64, copy=False)


def holiday_features(cutoff_date: dt.date) -> pl.DataFrame:
    """Только новые HOLIDAY-YOY колонки для build_features(cutoff_date)."""
    raw = _holiday_raw(cutoff_date)
    full = cutoff_date >= POST[1]
    out: dict[str, np.ndarray] = {"user_id": raw["user_id"].to_numpy()}

    support_metric = {"gmv": "days_buy", "orders": "orders", "days_buy": "days_buy",
                      "cart": "cart", "searches": "searches", "catalog": "catalog"}
    for m in BASE_METRICS:
        h, p = _arr(raw, f"{m}_holiday"), _arr(raw, f"{m}_pre")
        sm = support_metric[m]
        sp = _arr(raw, f"{sm}_holiday") + _arr(raw, f"{sm}_pre")
        out[f"hy_{m}_pre"], out[f"hy_{m}_rel_pre"] = _shrunk_delta(h, [p], sp)
        if full:
            q = _arr(raw, f"{m}_post")
            sf = sp + _arr(raw, f"{sm}_post")
            out[f"hy_{m}_nbr"], out[f"hy_{m}_rel"] = _shrunk_delta(h, [p, q], sf)
        else:
            out[f"hy_{m}_nbr"] = np.full(raw.height, np.nan, np.float32)
            out[f"hy_{m}_rel"] = np.full(raw.height, np.nan, np.float32)

    for m in CHANNELS:
        sh, ch = _arr(raw, f"search_{m}_holiday"), _arr(raw, f"catalog_{m}_holiday")
        sp, cp = _arr(raw, f"search_{m}_pre"), _arr(raw, f"catalog_{m}_pre")
        sup = sh + ch + sp + cp
        out[f"hy_search_catalog_{m}_pre"], out[f"hy_search_catalog_{m}_rel_pre"] = \
            _mix_delta(sh, ch, [(sp, cp)], sup)
        if full:
            sq, cq = _arr(raw, f"search_{m}_post"), _arr(raw, f"catalog_{m}_post")
            supf = sup + sq + cq
            out[f"hy_search_catalog_{m}_nbr"], out[f"hy_search_catalog_{m}_rel"] = \
                _mix_delta(sh, ch, [(sp, cp), (sq, cq)], supf)
        else:
            out[f"hy_search_catalog_{m}_nbr"] = np.full(raw.height, np.nan, np.float32)
            out[f"hy_search_catalog_{m}_rel"] = np.full(raw.height, np.nan, np.float32)

    score_cols = [out[f"hy_{m}_{'nbr' if full else 'pre'}"]
                  for m in ("gmv", "orders", "days_buy")]
    score = np.nanmean(np.vstack(score_cols), axis=0).astype(np.float32)
    purchase_support = (_arr(raw, "days_buy_holiday") + _arr(raw, "days_buy_pre")
                        + (_arr(raw, "days_buy_post") if full else 0.0))
    no_hist = purchase_support <= 0
    out["hy_purchase_score"] = np.where(no_hist, 0.0, score).astype(np.float32)
    out["hy_no_holiday_history"] = no_hist.astype(np.float32)
    out["hy_positive_history"] = ((~no_hist) & (score > 0)).astype(np.float32)
    out["hy_negative_history"] = ((~no_hist) & (score <= 0)).astype(np.float32)
    f = pl.DataFrame(out)
    assert f.columns == ["user_id"] + HY_COLS
    return f.sort("user_id")


def build_features(cutoff_date: dt.date, L: int | None = None,
                   norm_long: bool = True) -> pl.DataFrame:
    """Единственная точка построения feature-frame эксперимента."""
    from src.features import features_cached
    base = features_cached(cutoff_date, L, norm_long)
    hy = holiday_features(cutoff_date)
    return base.join(hy, on="user_id", how="left")


def make_xy_holiday(T: dt.date, L: int | None, n_blocks: int, with_target: bool = True,
                    norm_long: bool = True):
    """Панель и target берутся из боевого make_xy; новые колонки — из build_features."""
    X0, y = make_xy(T, L, n_blocks, with_target=with_target, norm_long=norm_long)
    hy = holiday_features(T)
    X = X0.join(hy, on="user_id", how="left")
    flag_cols = ["hy_no_holiday_history", "hy_positive_history", "hy_negative_history",
                 "hy_purchase_score"]
    X = X.with_columns([pl.col(c).fill_null(1.0 if c == "hy_no_holiday_history" else 0.0)
                        for c in flag_cols])
    X = X.with_columns([pl.col(c).cast(pl.Float32) for c in HY_COLS])
    return X, y


def audit_leakage() -> dict:
    early = holiday_features(dt.date(2025, 4, 3))
    late = holiday_features(dt.date(2025, 4, 17))
    nbr = [c for c in HY_COLS if c.endswith("_nbr") or c.endswith("_rel")]
    early_all_missing = all(early[c].null_count() == 0 and early[c].is_nan().all() for c in nbr)
    late_finite = all(late[c].is_finite().sum() > 0 for c in nbr)
    assert early_all_missing, "двухсторонняя feature стала доступна до конца post-окна"
    assert late_finite, "двухсторонняя feature не появилась после конца post-окна"
    return {"early_cutoff": "2025-04-03", "early_nbr_all_nan": early_all_missing,
            "late_cutoff": "2025-04-17", "late_nbr_has_finite": late_finite,
            "max_source_pre": HOLIDAY[1].isoformat(), "max_source_full": POST[1].isoformat(),
            "test_cutoff": CUTOFF_TEST.isoformat(), "passed": True}


# ---------------------------------------------------------------- diagnostic 2025 -> 2026
DIAG_METRICS = BASE_METRICS


def _pair_aggregates(year: int, focal: tuple[int, int, int, int],
                     neighbor: tuple[int, int, int, int]) -> pl.DataFrame:
    """focal/neighbor = (month, day, month, day), оба окна включительны."""
    fa, fb = dt.date(year, focal[0], focal[1]), dt.date(year, focal[2], focal[3])
    na, nb = dt.date(year, neighbor[0], neighbor[1]), dt.date(year, neighbor[2], neighbor[3])
    assert fb <= dt.date(year, 2, 13) if year == 2026 else True
    exprs = _period_exprs("focal", fa, fb) + _period_exprs("neighbor", na, nb)
    return (load().lazy().filter((pl.col("event_date") >= min(fa, na))
                                & (pl.col("event_date") <= max(fb, nb)))
            .group_by("user_id").agg(exprs).collect())


def _average_ranks(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    s = x[order]
    ranks = np.empty(len(x), float)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and s[j] == s[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1)
        i = j
    return ranks


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if not n1 or not n0:
        return float("nan")
    r = _average_ranks(np.asarray(score, float))
    return float((r[y].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))


def _diag_metric(frame: pl.DataFrame, metric: str) -> dict:
    h25 = frame[f"{metric}_focal_25"].to_numpy().astype(float)
    n25 = frame[f"{metric}_neighbor_25"].to_numpy().astype(float)
    h26 = frame[f"{metric}_focal_26"].to_numpy().astype(float)
    n26 = frame[f"{metric}_neighbor_26"].to_numpy().astype(float)
    support25, support26 = h25 + n25, h26 + n26
    xraw = np.log1p(h25) - np.log1p(n25)
    yraw = np.log1p(h26) - np.log1p(n26)
    p25 = float(np.median(xraw[support25 > 0])) if (support25 > 0).any() else 0.0
    p26 = float(np.median(yraw[support26 > 0])) if (support26 > 0).any() else 0.0
    rel = support25 / (support25 + _positive_median(support25))
    x = rel * (xraw - p25)
    y = yraw - p26
    hist = (support25 > 0) & (support26 > 0)

    # Детерминированный 2-fold cross-fit простого slope; seed только config.SEED.
    uid = frame["user_id"].to_numpy().astype(np.uint64)
    split = ((uid * np.uint64(2654435761) + np.uint64(SEED)) % np.uint64(1000)) < 500
    pred = np.zeros(len(x), float)
    slopes = []
    for test in (split, ~split):
        tr = (~test) & hist
        den = float(np.dot(x[tr], x[tr]))
        slope = float(np.dot(x[tr], y[tr]) / den) if den > 0 else 0.0
        pred[test] = slope * x[test]
        slopes.append(slope)
    mse0 = float(np.mean(y[hist] ** 2)) if hist.any() else float("nan")
    mse1 = float(np.mean((y[hist] - pred[hist]) ** 2)) if hist.any() else float("nan")
    return {"metric": metric, "n": int(len(x)), "n_both_history": int(hist.sum()),
            "history_share": float(hist.mean()), "pearson_all": _corr(x, y),
            "pearson_history": _corr(x[hist], y[hist]),
            "spearman_history": _corr(_average_ranks(x[hist]), _average_ranks(y[hist])),
            "auc_positive_2026": _auc(y[hist] > 0, x[hist]),
            "slope_crossfit": float(np.mean(slopes)), "mse_baseline": mse0,
            "mse_yoy": mse1, "r2_oos": 1.0 - mse1 / mse0 if mse0 > 0 else float("nan")}


def run_diagnostic() -> dict:
    """New-Year YoY (holiday) и соседний обычный период тем же 14-дневным методом."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    panel = panel_users(CUTOFF_TEST, 3).select("user_id")
    probes = {
        "yoy_holiday": ((1, 1, 1, 14), (1, 15, 1, 28)),
        "placebo": ((1, 15, 1, 28), (1, 29, 2, 11)),
    }
    rows = []
    for probe, (focal, neighbor) in probes.items():
        pair = panel
        for year, suf in ((2025, "25"), (2026, "26")):
            a = _pair_aggregates(year, focal, neighbor)
            a = a.rename({c: f"{c}_{suf}" for c in a.columns if c != "user_id"})
            pair = pair.join(a, on="user_id", how="left")
        pair = pair.fill_null(0)
        for metric in DIAG_METRICS:
            rows.append({"probe": probe, **_diag_metric(pair, metric)})

    diag = pl.DataFrame(rows)
    diag.write_csv(RESULTS / "diagnostic_metrics.csv")
    purchase = ["gmv", "orders", "days_buy"]
    def summary_for(probe: str) -> dict:
        d = diag.filter((pl.col("probe") == probe) & pl.col("metric").is_in(purchase))
        return {"median_pearson_history": float(d["pearson_history"].median()),
                "median_spearman_history": float(d["spearman_history"].median()),
                "median_r2_oos": float(d["r2_oos"].median()),
                "median_auc_positive": float(d["auc_positive_2026"].median())}
    yoy, placebo = summary_for("yoy_holiday"), summary_for("placebo")
    signal_pass = (yoy["median_pearson_history"] > 0.02
                   and yoy["median_pearson_history"] > placebo["median_pearson_history"] + 0.01
                   and yoy["median_r2_oos"] > max(0.0, placebo["median_r2_oos"]))
    summary = {"method": "14-day focal response versus adjacent 14-day neighbor",
               "yoy_window": "Jan01-Jan14 vs Jan15-Jan28, 2025->2026",
               "placebo_window": "Jan15-Jan28 vs Jan29-Feb11, 2025->2026",
               "yoy": yoy, "placebo": placebo, "signal_pass": bool(signal_pass),
               "leakage_audit": audit_leakage()}
    (RESULTS / "diagnostic_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(diag.write_csv())
    _log(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


# ---------------------------------------------------------------- CV / prediction
def _patch_train_xy():
    import src.train as tr
    original = tr.xy

    def hy_xy(T, s, with_target=True, blocks=None):
        b = s.panel_blocks if blocks is None else blocks
        k = (PREFIX, T, s.L, b, with_target, s.norm_long)
        if len(tr._XY) > 6:
            tr._XY.clear()
        if k not in tr._XY:
            tr._XY[k] = make_xy_holiday(T, s.L, b, with_target=with_target,
                                        norm_long=s.norm_long)
        return tr._XY[k]
    tr.xy = hy_xy
    return tr, original


def run_cv(vals: list[dt.date] | None = None, part: str | None = None) -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    leakage = audit_leakage()
    exp_id = COMPONENT_ID if not part else f"{COMPONENT_ID}-{part}"
    tr, original_xy = _patch_train_xy()
    original_fit_free = tr.fit_free
    gains: list[np.ndarray] = []

    def capture_fit(s, box, ytr, wtr):
        model = original_fit_free(s, box, ytr, wtr)
        booster = model[0] if s.model == "dist" else model
        gains.append(booster.feature_importance("gain").astype(float))
        return model

    tr.fit_free = capture_fit
    try:
        s = tr.Setup(L=None, min_history=90, step=7, panel_blocks=3, train_blocks=1,
                     model="dist", rounds=250, norm_long=True,
                     params={"seed": SEED}, vals=vals)
        result = tr.run(exp_id,
                        "HOLIDAY-YOY: S1-DIST + cutoff-safe shrunk personal holiday response",
                        s, save_model_feats=True, verbose_imp=False, no_log=True, ref="S1-DIST")
    finally:
        tr.fit_free = original_fit_free
        tr.xy = original_xy
        tr._XY.clear()

    feats = result["feats"]
    G = np.vstack(gains)
    imp = pl.DataFrame({"feature": feats, "gain_mean": G.mean(axis=0),
                        "gain_std": G.std(axis=0), "gain_folds": (G > 0).sum(axis=0)})
    total = max(float(imp["gain_mean"].sum()), 1.0)
    imp = imp.with_columns((pl.col("gain_mean") / total).alias("gain_share")).sort(
        "gain_mean", descending=True).with_row_index("rank", offset=1)
    imp.write_csv(RESULTS / (f"importance_{part}.csv" if part else "importance.csv"))
    hy_imp = imp.filter(pl.col("feature").str.starts_with("hy_"))
    _log("\nHOLIDAY-YOY importance:\n" + hy_imp.head(20).write_csv())
    (RESULTS / "leakage_audit.json").write_text(json.dumps(leakage, indent=2), encoding="utf-8")
    return result


def _aligned_oof(exps: list[str]):
    ds = [load_oof(e) for e in exps]
    base_key = np.char.add(np.asarray(ds[0]["cutoff"], dtype="U10"),
                           np.asarray(ds[0]["user_id"]).astype("U20"))
    order = np.argsort(base_key)
    key = base_key[order]
    Z = []
    for d in ds:
        k = np.char.add(np.asarray(d["cutoff"], dtype="U10"),
                        np.asarray(d["user_id"]).astype("U20"))
        o = np.argsort(k)
        assert np.array_equal(k[o], key), "OOF rows differ"
        Z.append(np.asarray(d["z"])[o])
    d0 = ds[0]
    return (np.vstack(Z), np.asarray(d0["y"])[order], np.asarray(d0["cutoff"], dtype="U10")[order],
            np.asarray(d0["user_id"])[order])


def _fold_auc(y: np.ndarray, z: np.ndarray, cut: np.ndarray) -> tuple[list[float], float]:
    vals = []
    for V in VAL_FOLDS_S1:
        m = cut == V.isoformat()
        vals.append(_auc(y[m] > 0, z[m]))
    w = np.asarray(FOLD_WEIGHTS_S1, float)
    return vals, float(np.dot(w, vals) / w.sum())


def _segment_codes(uid: np.ndarray, cut: np.ndarray) -> np.ndarray:
    out = np.empty(len(uid), dtype="U20")
    for c in sorted(set(cut.tolist())):
        m = cut == c
        q = pl.DataFrame({"user_id": uid[m]}).join(
            holiday_features(dt.date.fromisoformat(c)).select(
                "user_id", "hy_purchase_score", "hy_no_holiday_history"),
            on="user_id", how="left")
        no = q["hy_no_holiday_history"].fill_null(1).to_numpy() > 0.5
        score = q["hy_purchase_score"].fill_null(0).to_numpy()
        out[m] = np.where(no, "no-history", np.where(score > 0, "positive", "negative"))
    return out


def _segment_table(y, base, new, cut, segment) -> pl.DataFrame:
    offsets = {"base": {}, "new": {}}
    for c in sorted(set(cut.tolist())):
        m = cut == c
        offsets["base"][c] = calibrate(y[m], base[m])[0]
        offsets["new"][c] = calibrate(y[m], new[m])[0]
    rows = []
    for s in ("positive", "negative", "no-history"):
        fb, fn, ab, an = [], [], [], []
        n = 0
        for c in sorted(set(cut.tolist())):
            m = (cut == c) & (segment == s)
            n += int(m.sum())
            fb.append(rmsle_z(y[m], base[m] + offsets["base"][c]))
            fn.append(rmsle_z(y[m], new[m] + offsets["new"][c]))
            ab.append(_auc(y[m] > 0, base[m])); an.append(_auc(y[m] > 0, new[m]))
        w = np.asarray(FOLD_WEIGHTS_S1, float)
        bw, nw = float(np.dot(w, fb) / w.sum()), float(np.dot(w, fn) / w.sum())
        ba, na = float(np.dot(w, ab) / w.sum()), float(np.dot(w, an) / w.sum())
        rows.append({"segment": s, "n_oof": n, "share": n / len(y),
                     "base_wcv": bw, "new_wcv": nw, "delta_wcv": nw - bw,
                     "base_auc": ba, "new_auc": na, "delta_auc": na - ba})
    return pl.DataFrame(rows)


def analyze_cv() -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    exps = list(WEIGHTS) + [COMPONENT_ID]
    Z, y, cut, uid = _aligned_oof(exps)
    base_parts = Z[:4]
    base = np.average(base_parts, axis=0, weights=list(WEIGHTS.values()))
    new_parts = base_parts.copy()
    new_parts[3] = Z[4]
    new = np.average(new_parts, axis=0, weights=list(WEIGHTS.values()))
    rb, rn = evaluate(y, base, cut), evaluate(y, new, cut)
    auc_b_f, auc_b = _fold_auc(y, base, cut)
    auc_n_f, auc_n = _fold_auc(y, new, cut)
    fold_rows = []
    for i, c in enumerate(rb["folds"]):
        fold_rows.append({"fold": c, "base_rmsle_cal": rb["fold_cal"][i],
                          "new_rmsle_cal": rn["fold_cal"][i],
                          "delta_rmsle": rn["fold_cal"][i] - rb["fold_cal"][i],
                          "base_auc": auc_b_f[i], "new_auc": auc_n_f[i],
                          "delta_auc": auc_n_f[i] - auc_b_f[i]})
    pl.DataFrame(fold_rows).write_csv(RESULTS / "cv_folds.csv")
    seg = _segment_codes(uid, cut)
    segtab = _segment_table(y, base, new, cut, seg)
    segtab.write_csv(RESULTS / "segments.csv")

    diag_path = RESULTS / "diagnostic_summary.json"
    diagnostic = json.loads(diag_path.read_text(encoding="utf-8")) if diag_path.exists() else {}
    delta = float(rn["wcv"] - rb["wcv"])
    folds_better = int(sum(n < b for n, b in zip(rn["fold_cal"], rb["fold_cal"])))
    last_better = rn["fold_cal"][-1] < rb["fold_cal"][-1]
    ordinary_verdict = ("PASS" if delta <= -0.002 and folds_better >= 3 and last_better
                        else "DEVELOP" if delta <= -0.0005 and folds_better >= 3 and last_better
                        else "NEUTRAL" if abs(delta) <= 0.0005 else "FAIL")
    signal_pass = bool(diagnostic.get("signal_pass", False))
    if ordinary_verdict == "PASS":
        decision = "SEND"
    elif ordinary_verdict == "NEUTRAL" and signal_pass:
        decision = "SEND_HIGH_RISK"
    else:
        decision = "DO_NOT_SEND"
    summary = {"prefix": PREFIX, "base": "S1-DIST-MIX",
               "change": "replace only 0.45 S1-DIST slot by HOLIDAY-YOY-DIST",
               "weights": WEIGHTS, "level": LEVEL, "base_wcv": rb["wcv"],
               "new_wcv": rn["wcv"], "delta_wcv": delta,
               "folds_better": folds_better, "last_fold_better": bool(last_better),
               "base_fold_cal": rb["fold_cal"], "new_fold_cal": rn["fold_cal"],
               "base_auc": auc_b, "new_auc": auc_n, "delta_auc": auc_n - auc_b,
               "var_z_new_minus_base": float(np.var(new - base)),
               "corr_predictions": float(np.corrcoef(base, new)[0, 1]),
               "ordinary_cv_verdict": ordinary_verdict,
               "diagnostic_signal_pass": signal_pass, "decision": decision}
    (RESULTS / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    save_oof(f"{PREFIX}-MIX", uid, cut, new, y)
    save_report(f"{PREFIX}-MIX", rn, extra=summary)
    _log(format_report(rn, rb))
    _log("\nfolds:\n" + pl.DataFrame(fold_rows).write_csv())
    _log("segments:\n" + segtab.write_csv())
    _log(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def run_predict() -> None:
    """Обучает только новый DIST-слот на всех 29 cutoff'ах; остальные z не трогает."""
    import src.train as tr
    RESULTS.mkdir(parents=True, exist_ok=True)
    tr, original_xy = _patch_train_xy()
    try:
        s = tr.Setup(L=None, min_history=90, step=7, panel_blocks=3, train_blocks=1,
                     model="dist", rounds=250, norm_long=True, params={"seed": SEED})
        # Сначала только имена колонок, сам frame освобождаем перед большой матрицей.
        Xt0, _ = make_xy_holiday(CUTOFF_TEST, s.L, s.panel_blocks,
                                 with_target=False, norm_long=s.norm_long)
        feats = feature_names(Xt0)
        del Xt0
        gc.collect()
        cuts = s.grid()
        _log(f"full train: {len(cuts)} cutoffs {min(cuts)}..{max(cuts)}, {len(feats)} features")
        Xtr, ytr, wtr = tr.assemble(cuts, s, feats)
        tr._XY.clear()
        box = [Xtr]
        del Xtr
        model = tr.fit_free(s, box, ytr, None)
        del box, ytr, wtr
        gc.collect()
        Xt, _ = make_xy_holiday(CUTOFF_TEST, s.L, s.panel_blocks,
                                with_target=False, norm_long=s.norm_long)
        At = to_np(Xt, feats)
        z = np.maximum(tr.infer(s, model, At), 0.0)
        np.save(ARTIFACTS / f"ztest_{COMPONENT_ID}.npy", z)
        np.save(ARTIFACTS / f"uid_{COMPONENT_ID}.npy", Xt["user_id"].to_numpy())
        booster = model[0]
        imp = pl.DataFrame({"feature": feats,
                            "gain_full": booster.feature_importance("gain").astype(float)}).sort(
            "gain_full", descending=True).with_row_index("rank_full", offset=1)
        imp.write_csv(RESULTS / "importance_full.csv")
        _log(f"saved artifacts/ztest_{COMPONENT_ID}.npy, mean z={z.mean():.6f}")
    finally:
        tr.xy = original_xy
        tr._XY.clear()


def make_submission(force: bool = False) -> Path | None:
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    if not force and not summary["decision"].startswith("SEND"):
        _log(f"submission skipped: {summary['decision']}")
        return None
    names = [TEST_VARIANTS[k] for k in WEIGHTS]
    names[-1] = COMPONENT_ID
    Z = [np.load(ARTIFACTS / f"ztest_{n}.npy") for n in names]
    uid = np.load(ARTIFACTS / f"uid_{names[0]}.npy")
    for n in names[1:]:
        u = np.load(ARTIFACTS / f"uid_{n}.npy")
        assert np.array_equal(u, uid), f"uid order differs: {n}"
    z = np.average(np.vstack(Z), axis=0, weights=list(WEIGHTS.values()))
    delta = LEVEL - float(z.mean())
    z = np.maximum(z + delta, 0.0)
    # Клип может едва сдвинуть mean; фиксированная точка сохраняет L* точно.
    for _ in range(5):
        z = np.maximum(z + (LEVEL - float(z.mean())), 0.0)
    pred = np.maximum(np.expm1(z), 0.0)
    sub = pl.DataFrame({"user_id": uid, "predict": pred.astype(np.float64)})
    order = sample_submit().select("user_id").with_row_index("o")
    sub = sub.join(order, on="user_id", how="inner").sort("o").drop("o")
    assert sub.height == 250_000 and sub["user_id"].n_unique() == 250_000
    assert sub["user_id"].to_list() == sample_submit()["user_id"].to_list()
    assert np.isfinite(pred).all() and (pred >= 0).all()
    assert abs(float(np.log1p(sub["predict"].to_numpy()).mean()) - LEVEL) < 1e-5
    path = SUBMISSIONS / f"submission_{PREFIX}.csv"
    sub.write_csv(path, float_precision=6)
    _log(f"submission saved: {path}; level={np.log1p(sub['predict']).mean():.6f}")
    return path


def _diagnostic_excess_betas() -> dict[str, float]:
    """YoY slope минус placebo slope: фон персональной персистентности вычтен."""
    d = pl.read_csv(RESULTS / "diagnostic_metrics.csv")
    out = {}
    for metric in DIAG_METRICS:
        y = float(d.filter((pl.col("probe") == "yoy_holiday")
                           & (pl.col("metric") == metric))["slope_crossfit"][0])
        p = float(d.filter((pl.col("probe") == "placebo")
                           & (pl.col("metric") == metric))["slope_crossfit"][0])
        out[metric] = max(y - p, 0.0)
    return out


def _fast_adjustment(uid: np.ndarray, cutoff: dt.date,
                     betas: dict[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Прямой shrinkage-YoY residual, с нулевым средним среди имеющих историю."""
    cols = [f"hy_{m}_nbr" for m in DIAG_METRICS]
    h = holiday_features(cutoff).select(["user_id", "hy_purchase_score",
                                         "hy_no_holiday_history"] + cols)
    q = (pl.DataFrame({"user_id": uid}).with_row_index("_o")
         .join(h, on="user_id", how="left").sort("_o"))
    no = q["hy_no_holiday_history"].fill_null(1.0).to_numpy() > 0.5
    terms = []
    for m, c in zip(DIAG_METRICS, cols):
        x = q[c].fill_null(0.0).fill_nan(0.0).to_numpy().astype(float)
        terms.append(betas[m] * x)
    raw = np.mean(np.vstack(terms), axis=0)
    hist = ~no
    adj = np.zeros(len(uid), float)
    if hist.any():
        adj[hist] = raw[hist] - raw[hist].mean()
    score = q["hy_purchase_score"].fill_null(0.0).to_numpy().astype(float)
    return adj, no, score


def fast_evaluate_and_submit() -> dict:
    """Deadline-safe вариант: готовая смесь + direct YoY correction, без retrain."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    betas = _diagnostic_excess_betas()
    Z, y, cut, uid = _aligned_oof(list(WEIGHTS))
    base = np.average(Z, axis=0, weights=list(WEIGHTS.values()))
    adj = np.zeros(len(y), float)
    no = np.zeros(len(y), bool)
    score = np.zeros(len(y), float)
    for c in sorted(set(cut.tolist())):
        m = cut == c
        adj[m], no[m], score[m] = _fast_adjustment(uid[m], dt.date.fromisoformat(c), betas)
    new = base + adj
    rb, rn = evaluate(y, base, cut), evaluate(y, new, cut)
    auc_b_f, auc_b = _fold_auc(y, base, cut)
    auc_n_f, auc_n = _fold_auc(y, new, cut)
    fold_rows = []
    for i, c in enumerate(rb["folds"]):
        fold_rows.append({"fold": c, "base_rmsle_cal": rb["fold_cal"][i],
                          "new_rmsle_cal": rn["fold_cal"][i],
                          "delta_rmsle": rn["fold_cal"][i] - rb["fold_cal"][i],
                          "base_auc": auc_b_f[i], "new_auc": auc_n_f[i],
                          "delta_auc": auc_n_f[i] - auc_b_f[i]})
    pl.DataFrame(fold_rows).write_csv(RESULTS / "fast_cv_folds.csv")
    segment = np.where(no, "no-history", np.where(score > 0, "positive", "negative"))
    seg = _segment_table(y, base, new, cut, segment)
    seg.write_csv(RESULTS / "fast_segments.csv")
    imp = pl.DataFrame({"feature": [f"hy_{m}_nbr" for m in DIAG_METRICS],
                        "coefficient_yoy_minus_placebo": [betas[m] / len(DIAG_METRICS)
                                                          for m in DIAG_METRICS]})
    imp.write_csv(RESULTS / "fast_importance.csv")

    delta = float(rn["wcv"] - rb["wcv"])
    diagnostic = json.loads((RESULTS / "diagnostic_summary.json").read_text(encoding="utf-8"))
    decision = ("SEND_HIGH_RISK" if abs(delta) <= 0.0005 and diagnostic["signal_pass"]
                else "SEND" if delta <= -0.002 else "DO_NOT_SEND")

    # Test: исходная смесь неизменна, E03a/S1-CAP остаётся 0.10.
    names = [TEST_VARIANTS[k] for k in WEIGHTS]
    Zt = [np.load(ARTIFACTS / f"ztest_{n}.npy") for n in names]
    uidt = np.load(ARTIFACTS / f"uid_{names[0]}.npy")
    for n in names[1:]:
        assert np.array_equal(np.load(ARTIFACTS / f"uid_{n}.npy"), uidt), n
    zt_base = np.average(np.vstack(Zt), axis=0, weights=list(WEIGHTS.values()))
    at, no_t, score_t = _fast_adjustment(uidt, CUTOFF_TEST, betas)
    zt = zt_base + at
    for _ in range(10):
        zt = np.maximum(zt + (LEVEL - float(zt.mean())), 0.0)
    np.save(ARTIFACTS / f"ztest_{PREFIX}-FAST.npy", zt)
    np.save(ARTIFACTS / f"uid_{PREFIX}-FAST.npy", uidt)
    pred = np.maximum(np.expm1(zt), 0.0)
    sub = pl.DataFrame({"user_id": uidt, "predict": pred.astype(np.float64)})
    order = sample_submit().select("user_id").with_row_index("o")
    sub = sub.join(order, on="user_id", how="inner").sort("o").drop("o")
    assert sub.height == 250_000 and sub["user_id"].n_unique() == 250_000
    assert sub["user_id"].to_list() == sample_submit()["user_id"].to_list()
    assert np.isfinite(pred).all() and (pred >= 0).all()
    level = float(np.log1p(sub["predict"].to_numpy()).mean())
    assert abs(level - LEVEL) < 1e-5
    path = SUBMISSIONS / f"submission_{PREFIX}.csv"
    assert not path.exists(), f"не перезаписываю существующий {path}"
    sub.write_csv(path, float_precision=6)

    summary = {"variant": f"{PREFIX}-FAST", "method": "direct zero-mean YoY correction",
               "betas_yoy_minus_placebo": betas, "weights_unchanged": WEIGHTS,
               "level": LEVEL, "submission_level": level,
               "base_wcv": rb["wcv"], "new_wcv": rn["wcv"], "delta_wcv": delta,
               "base_fold_cal": rb["fold_cal"], "new_fold_cal": rn["fold_cal"],
               "base_auc": auc_b, "new_auc": auc_n, "delta_auc": auc_n - auc_b,
               "var_z_new_minus_base": float(np.var(adj)),
               "test_var_adjustment": float(np.var(at)),
               "test_max_abs_adjustment": float(np.max(np.abs(at))),
               "test_no_history_share": float(no_t.mean()),
               "diagnostic_signal_pass": diagnostic["signal_pass"],
               "decision": decision, "submission": str(path)}
    (RESULTS / "fast_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    save_oof(f"{PREFIX}-FAST", uid, cut, new, y)
    save_report(f"{PREFIX}-FAST", rn, extra=summary)
    _log(format_report(rn, rb))
    _log("\n" + pl.DataFrame(fold_rows).write_csv())
    _log(seg.write_csv())
    _log(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["diagnostic", "cv", "analyze", "predict",
                                        "submission", "fast", "all"], default="all")
    ap.add_argument("--force-submission", action="store_true")
    ap.add_argument("--vals", nargs="*", default=None,
                    help="CV subset: validation dates YYYY-MM-DD")
    ap.add_argument("--part", default=None, help="unique suffix for a partial CV artifact")
    a = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    stages = ["diagnostic", "cv", "analyze", "predict", "submission"] \
        if a.stage == "all" else [a.stage]
    for stage in stages:
        _log(f"\n{'=' * 28} {stage.upper()} {'=' * 28}")
        if stage == "diagnostic":
            run_diagnostic()
        elif stage == "cv":
            vals = [dt.date.fromisoformat(v) for v in a.vals] if a.vals else None
            run_cv(vals=vals, part=a.part)
        elif stage == "analyze":
            analyze_cv()
        elif stage == "predict":
            run_predict()
        elif stage == "fast":
            fast_evaluate_and_submit()
        else:
            make_submission(a.force_submission)
    _log(f"HOLIDAY-YOY completed in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
