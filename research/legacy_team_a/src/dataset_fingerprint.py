"""EXP-058: dataset/user-identity fingerprints against a joint permutation.

One CPU command performs the independent raw-parquet integrity audit, builds
cutoff-safe fingerprint columns, trains the paired PERM/REAL UNC arms on the
2025-10-16 fold, and evaluates the fixed STRONGEST_CURRENT slot replacement::

    python src/dataset_fingerprint.py

The historical seed-42 BASE arm is the exact, already materialized replay from
EXP-046.  Reusing that bitwise replay saves one 600-round fit without changing
the registered A/B/C experiment.  No test prediction or leaderboard action is
implemented in this module.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import hashlib
import inspect
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import polars as pl

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import models
from src.config import (ARTIFACTS, CUTOFF_TEST, DATA_START, LGB_PARAMS, RAW_PARQUET,
                        ROOT, SAMPLE_SUBMIT, SEED)
from src.data import load, sample_submit
from src.features import (build_features, feature_names, features_cached, panel_users,
                          target, to_np)
from src.merge_oof import auc_positive
from src.report import evaluate
from src.tabular_backbone_refresh import component_setup, saved_features
from src.train import _XY, xy
from src.validation import calibrate, rmsle_z


EXP_NUM = 58
EXP_ID = "DATASET-FINGERPRINT"
PREFIX = "FINGERPRINT_EXP058"
RUN_DIR = ARTIFACTS / PREFIX
RESULTS = ROOT / "research" / "strategies" / "results" / PREFIX
PILOT_FOLD = dt.date(2025, 10, 16)
PILOT_LABEL = PILOT_FOLD.isoformat()
ROUNDS = 600
NEAR_PEARSON = 0.99999
NEAR_SPEARMAN = 0.9995
NOVELTY_SAMPLE = 50_000
ROW_GROUP_CACHE = RUN_DIR / "rowgroup_membership.parquet"
BASE_REPLAY_NPZ = ARTIFACTS / "TBR_EXP046" / "TBR_EXP046_UNC_S42_V1016.npz"
BASE_REPLAY_META = ARTIFACTS / "TBR_EXP046" / "TBR_EXP046_UNC_S42_V1016.json"

BASE_COMPONENTS = {
    "CAP": (ARTIFACTS / "oof_S1-E03a.npz", 0.10),
    "UNC": (ARTIFACTS / "oof_S1-E02.npz", 0.20),
    "DIST": (ARTIFACTS / "oof_S1-DIST.npz", 0.25),
    "ETX": (ARTIFACTS / "oof_ETX-AVG3.npz", 0.225),
    "SEQ": (ARTIFACTS / "oof_SEQ-AVG3.npz", 0.225),
}
EXPECTED_STRONG = 1.741278566
PANEL_SCHEDULE_START = dt.date(2025, 4, 3)

IDENTITY_FIELDS = [
    "fp_uid_rank_frac", "fp_uid_rank_bucket256", "fp_uid_rank_bucket4096",
    "fp_uid_bits_low16", "fp_uid_bits_high16", "fp_sample_rank_frac",
    "fp_sample_id_neighborhood",
]
AVAILABILITY_FIELDS = [
    "fp_first_observed_day", "fp_last_observed_day", "fp_observed_days",
    "fp_active_months", "fp_month_mask_lo7", "fp_month_mask_hi7",
    "fp_initial_missing_prefix", "fp_longest_internal_absence",
    "fp_rows_q0", "fp_rows_q1", "fp_rows_q2", "fp_rows_q3", "fp_rows_q4",
    "fp_raw_file_count", "fp_partition_count", "fp_rowgroup_count",
    "fp_rowgroup_first", "fp_rowgroup_last", "fp_rowgroup_span",
]
PANEL_FIELDS = [
    "fp_panel1_pass_count", "fp_panel3_pass_count",
    "fp_panel1_first_day", "fp_panel3_first_day",
]
FINGERPRINT_FIELDS = IDENTITY_FIELDS + AVAILABILITY_FIELDS + PANEL_FIELDS
FORBIDDEN_FEATURE_TOKENS = ("target", "future", "label", "mean_y", "encoding")


def log(*args: Any) -> None:
    print(*args, flush=True)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2,
                              sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        fields.extend(k for k in row if k not in fields)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: json.dumps(jsonable(v), ensure_ascii=False)
                           if isinstance(v, (dict, list, tuple)) else v
                           for k, v in row.items()} for row in rows])
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(a.dtype.str.encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def target_mask(dates: np.ndarray, cutoff: dt.date, horizon: int = 30) -> np.ndarray:
    """The registered target boundary: strictly after T and through T+30."""
    d = np.asarray(dates, dtype="datetime64[D]")
    t = np.datetime64(cutoff)
    return (d > t) & (d <= t + np.timedelta64(horizon, "D"))


def splitmix64(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.uint64).copy()
    with np.errstate(over="ignore"):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))


def _manual_targets(user_id: np.ndarray, dates: np.ndarray, gmv: np.ndarray,
                    universe: np.ndarray, cutoffs: Iterable[dt.date]) -> dict[str, np.ndarray]:
    idx = np.searchsorted(universe, user_id)
    if np.any(idx == len(universe)) or not np.array_equal(universe[idx], user_id):
        raise AssertionError("raw user outside declared universe")
    out: dict[str, np.ndarray] = {}
    for cutoff in cutoffs:
        mask = target_mask(dates, cutoff) & (gmv > 0)
        out[cutoff.isoformat()] = np.bincount(
            idx[mask], weights=gmv[mask], minlength=len(universe)).astype(np.float64)
    return out


def _sample_users() -> np.ndarray:
    chosen: list[np.ndarray] = []
    for cutoff in (dt.date(2025, 9, 4), PILOT_FOLD):
        users = panel_users(cutoff, 3)["user_id"].to_numpy()
        pos = np.linspace(0, len(users) - 1, 64, dtype=np.int64)
        chosen.append(users[pos])
    return np.unique(np.concatenate(chosen)).astype(np.int64)


def _sample_integrity_audit() -> dict[str, Any]:
    """Small PyArrow reconstruction, completed before the full raw scan."""
    import pyarrow.parquet as pq

    users = _sample_users()
    columns = ["user_id", "event_date", "gmv", "gmv_search", "gmv_cat",
               "searches", "cat", "to_cart", "to_ord"]
    table = pq.read_table(RAW_PARQUET, columns=columns,
                          filters=[("user_id", "in", users.tolist())])
    raw_u = table["user_id"].to_numpy().astype(np.int64)
    dates = table["event_date"].to_numpy().astype("datetime64[D]")
    gmv = table["gmv"].to_numpy().astype(np.float64)
    gs = table["gmv_search"].to_numpy().astype(np.float64)
    gc_ = table["gmv_cat"].to_numpy().astype(np.float64)
    order = np.lexsort((dates.astype(np.int64), raw_u))
    raw_u, dates, gmv, gs, gc_ = (a[order] for a in (raw_u, dates, gmv, gs, gc_))
    duplicate = int(np.sum((raw_u[1:] == raw_u[:-1]) & (dates[1:] == dates[:-1])))
    max_channel_error = float(np.max(np.abs(gs + gc_ - gmv)))
    cuts = (dt.date(2025, 9, 4), PILOT_FOLD)
    manual = _manual_targets(raw_u, dates, gmv, users, cuts)
    comparisons = []
    boundary = []
    missing = []
    for cutoff in cuts:
        prod = target(cutoff, pl.DataFrame({"user_id": users}))["y"].to_numpy()
        y = manual[cutoff.isoformat()]
        comparisons.append({
            "cutoff": cutoff.isoformat(), "n_users": len(users),
            "max_abs_target_diff": float(np.max(np.abs(y - prod))),
            "max_abs_log1p_diff": float(np.max(np.abs(np.log1p(y) - np.log1p(prod)))),
        })
        t = np.datetime64(cutoff)
        for delta in (0, 1, 30, 31):
            m = dates == t + np.timedelta64(delta, "D")
            boundary.append({"cutoff": cutoff.isoformat(), "day_delta": delta,
                             "rows": int(m.sum()), "positive_gmv": float(gmv[m & (gmv > 0)].sum()),
                             "included": bool(1 <= delta <= 30)})
    for user in users[:32]:
        d = np.unique(dates[raw_u == user].astype(np.int64))
        missing.append({"user_id": int(user), "observed_days": len(d),
                        "span_days": int(d[-1] - d[0] + 1),
                        "missing_inside_span": int(d[-1] - d[0] + 1 - len(d))})
    status = (duplicate == 0 and max_channel_error <= 1e-6
              and all(r["max_abs_target_diff"] <= 1e-9
                      and r["max_abs_log1p_diff"] <= 1e-12 for r in comparisons))
    audit = {"status": "PASS" if status else "CRITICAL_STOP", "users": users,
             "raw_rows": len(raw_u), "duplicate_user_day_count": duplicate,
             "max_abs_gmv_search_plus_cat_minus_gmv": max_channel_error,
             "target_comparisons": comparisons, "target_boundary_rows": boundary,
             "missing_date_examples": missing}
    write_json(RESULTS / "integrity_sample.json", audit)
    if not status:
        raise RuntimeError(f"CRITICAL_STOP sample integrity mismatch: {audit}")
    return audit


def _full_integrity_audit() -> dict[str, Any]:
    """Independent streaming PyArrow audit of every raw row."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(RAW_PARQUET)
    ss = pl.read_csv(SAMPLE_SUBMIT)
    sample_order = ss["user_id"].to_numpy().astype(np.int64)
    if len(sample_order) != len(np.unique(sample_order)):
        raise RuntimeError("CRITICAL_STOP duplicate sample users")
    universe = np.sort(sample_order)
    n = len(universe)
    cuts = (dt.date(2025, 9, 4), dt.date(2025, 9, 18),
            dt.date(2025, 10, 2), PILOT_FOLD)
    totals = {c.isoformat(): np.zeros(n, np.float64) for c in cuts}
    observed_at = {c.isoformat(): np.zeros(n, np.int32) for c in cuts}
    last_at = {c.isoformat(): np.full(n, -1, np.int32) for c in cuts}
    first = np.full(n, np.iinfo(np.int32).max, np.int32)
    last = np.full(n, -1, np.int32)
    observed = np.zeros(n, np.int32)
    last_seen_date = np.full(n, -1, np.int32)
    raw_seen = np.zeros(n, bool)
    rowgroup_rows: list[dict[str, Any]] = []
    row_count = duplicate = order_violations = negative_gmv = zero_state_rows = 0
    channel_bad = 0
    channel_max = 0.0
    prev_user: int | None = None
    prev_date: int | None = None
    columns = ["user_id", "event_date", "gmv", "gmv_search", "gmv_cat",
               "searches", "cat", "to_cart", "to_ord"]
    for rg in range(pf.num_row_groups):
        batch = pf.read_row_group(rg, columns=columns)
        u = batch["user_id"].to_numpy().astype(np.int64)
        dates_np = batch["event_date"].to_numpy().astype("datetime64[D]")
        day = (dates_np - np.datetime64(DATA_START)).astype(np.int32)
        gmv = batch["gmv"].to_numpy().astype(np.float64)
        gs = batch["gmv_search"].to_numpy().astype(np.float64)
        gc_ = batch["gmv_cat"].to_numpy().astype(np.float64)
        searches = batch["searches"].to_numpy()
        cat = batch["cat"].to_numpy()
        carts = batch["to_cart"].to_numpy()
        orders = batch["to_ord"].to_numpy()
        idx = np.searchsorted(universe, u)
        if np.any(idx == n) or not np.array_equal(universe[idx], u):
            raise RuntimeError("CRITICAL_STOP raw/sample user-set mismatch")
        raw_seen[idx] = True
        row_count += len(u)
        negative_gmv += int(np.sum(gmv < 0))
        zero_state_rows += int(np.sum((searches == 0) & (cat == 0) & (carts == 0)
                                      & (orders == 0)))
        err = np.abs(gs + gc_ - gmv)
        channel_bad += int(np.sum(err > 1e-6))
        channel_max = max(channel_max, float(err.max(initial=0.0)))
        if len(u) > 1:
            same = u[1:] == u[:-1]
            duplicate += int(np.sum(same & (day[1:] == day[:-1])))
            order_violations += int(np.sum((u[1:] < u[:-1]) | (same & (day[1:] < day[:-1]))))
        if prev_user is not None:
            duplicate += int(u[0] == prev_user and day[0] == prev_date)
            order_violations += int(u[0] < prev_user or (u[0] == prev_user and day[0] < prev_date))
        prev_user, prev_date = int(u[-1]), int(day[-1])
        for cutoff in cuts:
            history_mask = dates_np <= np.datetime64(cutoff)
            observed_at[cutoff.isoformat()] += np.bincount(
                idx[history_mask], minlength=n).astype(np.int32)
            np.maximum.at(last_at[cutoff.isoformat()], idx[history_mask], day[history_mask])
            mask = target_mask(dates_np, cutoff) & (gmv > 0)
            totals[cutoff.isoformat()] += np.bincount(
                idx[mask], weights=gmv[mask], minlength=n)
        starts = np.r_[0, np.flatnonzero(u[1:] != u[:-1]) + 1]
        ends = np.r_[starts[1:], len(u)]
        for lo, hi in zip(starts, ends):
            j = int(idx[lo])
            seg = day[lo:hi]
            unique_count = int(len(seg) - np.sum(seg[1:] == seg[:-1]))
            if last_seen_date[j] == int(seg[0]):
                unique_count -= 1
            first[j] = min(first[j], int(seg[0]))
            last[j] = max(last[j], int(seg[-1]))
            observed[j] += unique_count
            last_seen_date[j] = int(seg[-1])
            rowgroup_rows.append({"user_id": int(u[lo]), "rowgroup_id": rg,
                                  "min_day": int(seg[0]), "max_day": int(seg[-1]),
                                  "rows": int(hi - lo)})
        if (rg + 1) % 25 == 0 or rg + 1 == pf.num_row_groups:
            log(f"  integrity raw row groups {rg + 1}/{pf.num_row_groups}")

    if not raw_seen.all():
        raise RuntimeError("CRITICAL_STOP sample users absent from raw parquet")
    metadata_rows = int(pf.metadata.num_rows)
    missing_days = (last.astype(np.int64) - first.astype(np.int64) + 1
                    - observed.astype(np.int64))
    comparisons = []
    target_ok = True
    feature_ok = True
    oof = np.load(ARTIFACTS / "oof_S1-E02.npz")
    for cutoff in cuts:
        panel = panel_users(cutoff, 3)["user_id"].to_numpy().astype(np.int64)
        pidx = np.searchsorted(universe, panel)
        manual = totals[cutoff.isoformat()][pidx]
        prod = target(cutoff, pl.DataFrame({"user_id": panel}))["y"].to_numpy()
        om = np.asarray(oof["cutoff"], dtype="U10") == cutoff.isoformat()
        oo = np.argsort(np.asarray(oof["user_id"])[om])
        oof_u = np.asarray(oof["user_id"])[om][oo]
        oof_y = np.asarray(oof["y"])[om][oo]
        max_prod = float(np.max(np.abs(manual - prod)))
        max_oof = float(np.max(np.abs(manual - oof_y)))
        oof_float32_exact = np.array_equal(manual.astype(np.float32), oof_y.astype(np.float32))
        target_ok &= (np.array_equal(panel, oof_u) and max_prod <= 1e-9 and oof_float32_exact)
        base = build_features(cutoff, L=None, norm_long=False).select(
            "user_id", "all_days_present", "tenure", "rec_any")
        joined = (pl.DataFrame({"user_id": panel}).join(base, on="user_id", how="left")
                  .sort("user_id"))
        obs = observed_at[cutoff.isoformat()][pidx]
        cutoff_last = last_at[cutoff.isoformat()][pidx]
        max_days = float(np.max(np.abs(joined["all_days_present"].to_numpy() - obs)))
        max_tenure = float(np.max(np.abs(joined["tenure"].to_numpy() -
                                          (cutoff - DATA_START).days + first[pidx])))
        max_rec = float(np.max(np.abs(joined["rec_any"].to_numpy() -
                                      ((cutoff - DATA_START).days - cutoff_last))))
        feature_ok &= max_days == 0 and max_tenure == 0 and max_rec == 0
        comparisons.append({
            "cutoff": cutoff.isoformat(), "panel_rows": len(panel),
            "panel_sorted_unique": bool(np.all(np.diff(panel) > 0)),
            "manual_vs_production_target_max_abs": max_prod,
            "manual_vs_oof_target_max_abs": max_oof,
            "manual_vs_oof_target_float32_exact": oof_float32_exact,
            "manual_vs_production_log1p_max_abs": float(
                np.max(np.abs(np.log1p(manual) - np.log1p(prod)))),
            "all_days_present_max_abs": max_days,
            "tenure_max_abs": max_tenure, "rec_any_max_abs": max_rec,
        })
    status = (row_count == metadata_rows and duplicate == 0 and order_violations == 0
              and negative_gmv == 0 and channel_bad == 0 and target_ok and feature_ok
              and len(universe) == 250_000)
    audit = {
        "status": "PASS" if status else "CRITICAL_STOP", "raw_rows": row_count,
        "parquet_metadata_rows": metadata_rows, "raw_users": int(raw_seen.sum()),
        "sample_rows": len(sample_order), "sample_unique": len(np.unique(sample_order)),
        "raw_user_set_equals_sample": bool(raw_seen.all()),
        "duplicate_user_day_count": duplicate, "raw_sort_order_violations": order_violations,
        "negative_gmv_rows": negative_gmv, "channel_equality_bad_rows": channel_bad,
        "max_abs_gmv_search_plus_cat_minus_gmv": channel_max,
        "zero_activity_rows_preserved": zero_state_rows,
        "missing_date_behavior": {
            "definition": "no dense fill; observed raw user-days only",
            "users_with_internal_missing_dates": int(np.sum(missing_days > 0)),
            "total_missing_dates_inside_observed_spans": int(missing_days.sum()),
            "observed_days_min_max": [int(observed.min()), int(observed.max())],
        },
        "target_and_feature_comparisons": comparisons,
        "sample_order_sha256": sha256_array(sample_order),
        "sorted_universe_sha256": sha256_array(universe),
        "raw_parquet_sha256": sha256_file(RAW_PARQUET),
    }
    write_json(RESULTS / "integrity_full.json", audit)
    if not status:
        raise RuntimeError(f"CRITICAL_STOP full integrity mismatch: {audit}")
    rg = pl.DataFrame(rowgroup_rows).sort(["user_id", "rowgroup_id"])
    rg.write_parquet(ROW_GROUP_CACHE)
    return audit


def run_integrity_audit() -> dict[str, Any]:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    sample = _sample_integrity_audit()
    full_path = RESULTS / "integrity_full.json"
    if full_path.exists() and ROW_GROUP_CACHE.exists():
        full = json.loads(full_path.read_text(encoding="utf-8"))
        if full.get("status") != "PASS":
            raise RuntimeError("CRITICAL_STOP cached integrity audit is not PASS")
    else:
        full = _full_integrity_audit()
    out = {"sample": sample["status"], "full": full["status"],
           "critical_stop": False}
    write_json(RESULTS / "integrity_summary.json", out)
    return out


def _universe() -> np.ndarray:
    return np.sort(sample_submit()["user_id"].to_numpy().astype(np.int64))


def _identity_frame() -> pl.DataFrame:
    path = RUN_DIR / "identity.parquet"
    if path.exists():
        return pl.read_parquet(path)
    raw_rank_ids = _universe()
    ss = sample_submit()["user_id"].to_numpy().astype(np.int64)
    sample_pos = np.empty(len(raw_rank_ids), np.int64)
    sample_pos[np.searchsorted(raw_rank_ids, ss)] = np.arange(len(ss), dtype=np.int64)
    rank = np.arange(len(raw_rank_ids), dtype=np.int64)
    scale = max(len(raw_rank_ids) - 1, 1)
    frame = pl.DataFrame({
        "user_id": raw_rank_ids,
        "fp_uid_rank_frac": rank / scale,
        "fp_uid_rank_bucket256": np.minimum(rank * 256 // len(rank), 255),
        "fp_uid_rank_bucket4096": np.minimum(rank * 4096 // len(rank), 4095),
        "fp_uid_bits_low16": (raw_rank_ids.astype(np.uint64) & np.uint64(0xFFFF)).astype(np.uint32),
        "fp_uid_bits_high16": ((raw_rank_ids.astype(np.uint64) >> np.uint64(16))
                               & np.uint64(0xFFFF)).astype(np.uint32),
        "fp_sample_rank_frac": sample_pos / scale,
        "fp_sample_id_neighborhood": (np.abs(sample_pos - rank) <= 256).astype(np.uint8),
    })
    frame.write_parquet(path)
    return frame


def _history_frame(cutoff: dt.date) -> pl.DataFrame:
    """Raw-history fields; every event-valued expression is filtered <= cutoff."""
    hist = load().lazy().filter(pl.col("event_date") <= cutoff)
    month_idx = ((pl.col("event_date").dt.year() - 2025) * 12
                 + pl.col("event_date").dt.month() - 1).cast(pl.Int16)
    quarter_idx = ((pl.col("event_date").dt.year() - 2025) * 4
                   + (pl.col("event_date").dt.month() - 1) // 3).cast(pl.Int8)
    day = (pl.col("event_date") - pl.lit(DATA_START)).dt.total_days().cast(pl.Int32)
    work = hist.with_columns(__month=month_idx, __quarter=quarter_idx, __day=day)
    month_aggs = [(pl.col("__month") == m).any().cast(pl.UInt8).alias(f"__m{m}")
                   for m in range(14)]
    quarter_aggs = [(pl.col("__quarter") == q).sum().alias(f"fp_rows_q{q}")
                     for q in range(5)]
    f = work.group_by("user_id").agg([
        pl.col("__day").min().alias("fp_first_observed_day"),
        pl.col("__day").max().alias("fp_last_observed_day"),
        pl.col("event_date").n_unique().alias("fp_observed_days"),
        (pl.col("__day").sort().diff().max().fill_null(1) - 1)
        .clip(lower_bound=0).alias("fp_longest_internal_absence"),
        *month_aggs, *quarter_aggs,
    ]).collect()
    f = f.with_columns([
        pl.sum_horizontal([pl.col(f"__m{m}") for m in range(14)]).alias("fp_active_months"),
        pl.sum_horizontal([pl.col(f"__m{m}") * (1 << m) for m in range(7)])
        .alias("fp_month_mask_lo7"),
        pl.sum_horizontal([pl.col(f"__m{m}") * (1 << (m - 7)) for m in range(7, 14)])
        .alias("fp_month_mask_hi7"),
        pl.col("fp_first_observed_day").alias("fp_initial_missing_prefix"),
        pl.lit(1).alias("fp_raw_file_count"), pl.lit(1).alias("fp_partition_count"),
    ]).drop([f"__m{m}" for m in range(14)])
    return f


def _rowgroup_frame(cutoff: dt.date) -> pl.DataFrame:
    if not ROW_GROUP_CACHE.exists():
        raise FileNotFoundError("integrity audit must create rowgroup membership first")
    day = (cutoff - DATA_START).days
    return (pl.scan_parquet(ROW_GROUP_CACHE).filter(pl.col("min_day") <= day)
            .group_by("user_id").agg([
                pl.col("rowgroup_id").n_unique().alias("fp_rowgroup_count"),
                pl.col("rowgroup_id").min().alias("fp_rowgroup_first"),
                pl.col("rowgroup_id").max().alias("fp_rowgroup_last"),
            ]).with_columns((pl.col("fp_rowgroup_last") - pl.col("fp_rowgroup_first"))
                            .alias("fp_rowgroup_span")).collect())


def panel_schedule(cutoff: dt.date) -> list[dt.date]:
    out, current = [], PANEL_SCHEDULE_START
    while current <= cutoff:
        out.append(current)
        current += dt.timedelta(days=7)
    return out


def summarize_panel_memberships(universe: np.ndarray, cutoff: dt.date,
                                schedule: list[dt.date],
                                members1: list[np.ndarray],
                                members3: list[np.ndarray]) -> pl.DataFrame:
    """Cumulative eligibility using schedule dates no later than cutoff."""
    n = len(universe)
    c1 = np.zeros(n, np.int16); c3 = np.zeros(n, np.int16)
    f1 = np.full(n, -1, np.int16); f3 = np.full(n, -1, np.int16)
    for date, u1, u3 in zip(schedule, members1, members3):
        if date > cutoff:
            raise AssertionError("future panel membership supplied to fingerprint builder")
        day = (date - DATA_START).days
        for users, count, first in ((u1, c1, f1), (u3, c3, f3)):
            idx = np.searchsorted(universe, users)
            if np.any(idx == n) or not np.array_equal(universe[idx], users):
                raise AssertionError("panel user outside universe")
            unseen = first[idx] < 0
            first[idx[unseen]] = day
            count[idx] += 1
    return pl.DataFrame({
        "user_id": universe, "fp_panel1_pass_count": c1,
        "fp_panel3_pass_count": c3, "fp_panel1_first_day": f1,
        "fp_panel3_first_day": f3,
    })


def _panel_history_frame(cutoff: dt.date) -> pl.DataFrame:
    path = RUN_DIR / f"panel_history_{cutoff:%Y%m%d}.parquet"
    if path.exists():
        return pl.read_parquet(path)
    schedule = panel_schedule(cutoff)
    m1 = [panel_users(date, 1)["user_id"].to_numpy().astype(np.int64) for date in schedule]
    m3 = [panel_users(date, 3)["user_id"].to_numpy().astype(np.int64) for date in schedule]
    out = summarize_panel_memberships(_universe(), cutoff, schedule, m1, m3)
    out.write_parquet(path)
    return out


def fingerprint_path(cutoff: dt.date) -> Path:
    return RUN_DIR / f"fingerprints_{cutoff:%Y%m%d}.parquet"


def build_fingerprint_features(cutoff: dt.date) -> pl.DataFrame:
    """Join new columns to the canonical ``build_features(cutoff)`` universe.

    The canonical call is the sole base-feature construction path.  Fingerprint
    columns are an isolated additive join and never rewrite existing columns.
    """
    base_users = build_features(cutoff, L=None, norm_long=False).select("user_id")
    identity = _identity_frame()
    hist = _history_frame(cutoff)
    rowgroups = _rowgroup_frame(cutoff)
    panels = _panel_history_frame(cutoff)
    out = (identity.join(hist, on="user_id", how="left")
           .join(rowgroups, on="user_id", how="left")
           .join(panels, on="user_id", how="left"))
    numeric = [c for c in FINGERPRINT_FIELDS if c in out.columns]
    out = (out.with_columns([pl.col(c).fill_null(-1) for c in numeric])
           .select(["user_id"] + FINGERPRINT_FIELDS).sort("user_id"))
    if out.columns != ["user_id"] + FINGERPRINT_FIELDS:
        raise AssertionError(f"fingerprint schema mismatch: {out.columns}")
    if out.height != len(_universe()) or out["user_id"].n_unique() != out.height:
        raise AssertionError("fingerprint universe is not exact")
    if not set(base_users["user_id"].to_list()).issubset(set(out["user_id"].to_list())):
        raise AssertionError("canonical build_features universe not covered")
    return out


def fingerprint_cached(cutoff: dt.date) -> pl.DataFrame:
    path = fingerprint_path(cutoff)
    if path.exists():
        out = pl.read_parquet(path)
        if out.columns != ["user_id"] + FINGERPRINT_FIELDS:
            raise AssertionError(f"stale fingerprint schema at {path}")
        return out
    out = build_fingerprint_features(cutoff)
    out.write_parquet(path)
    return out


def fixed_permutation(universe: np.ndarray, signatures: np.ndarray,
                      seed: int = SEED) -> np.ndarray:
    """One deterministic bijection, deranged inside each non-singleton stratum."""
    universe = np.asarray(universe, np.int64)
    signatures = np.asarray(signatures)
    if len(universe) != len(signatures) or np.any(np.diff(universe) <= 0):
        raise ValueError("universe/signature contract failed")
    rng = np.random.default_rng(seed)
    mapped = universe.copy()
    for value in np.unique(signatures):
        idx = np.flatnonzero(signatures == value)
        if len(idx) <= 1:
            continue
        order = rng.permutation(idx)
        mapped[order] = universe[np.roll(order, 1)]
    if not np.array_equal(np.sort(mapped), universe):
        raise AssertionError("permutation is not bijective")
    return mapped


def build_permutation(train_cutoffs: list[dt.date]) -> pl.DataFrame:
    path = RUN_DIR / "permutation.parquet"
    if path.exists():
        return pl.read_parquet(path)
    universe = _universe()
    signature = np.zeros(len(universe), np.uint32)
    for bit, cutoff in enumerate(train_cutoffs):
        users = panel_users(cutoff, 1)["user_id"].to_numpy().astype(np.int64)
        signature[np.searchsorted(universe, users)] |= np.uint32(1 << bit)
    val_users = panel_users(PILOT_FOLD, 3)["user_id"].to_numpy().astype(np.int64)
    signature[np.searchsorted(universe, val_users)] |= np.uint32(1 << len(train_cutoffs))
    mapped = fixed_permutation(universe, signature, SEED)
    out = pl.DataFrame({"user_id": universe, "mapped_user_id": mapped,
                        "row_incidence_signature": signature})
    out.write_parquet(path)
    return out


def permuted_fingerprints(users: np.ndarray, fingerprint: pl.DataFrame,
                          mapping: pl.DataFrame, fields: list[str]) -> np.ndarray:
    left = pl.DataFrame({"user_id": np.asarray(users, np.int64),
                         "__order": np.arange(len(users), dtype=np.int64)})
    donor = fingerprint.select([pl.col("user_id").alias("mapped_user_id"), *fields])
    joined = (left.join(mapping.select("user_id", "mapped_user_id"), on="user_id", how="left")
              .join(donor, on="mapped_user_id", how="left").sort("__order"))
    if joined.select(fields).null_count().row(0) != tuple(0 for _ in fields):
        raise AssertionError("PERM fingerprint join produced nulls")
    return joined.select(fields).to_numpy().astype(np.float32)


def _fill_for_corr(a: np.ndarray) -> np.ndarray:
    x = np.asarray(a, np.float64).copy()
    for j in range(x.shape[1]):
        finite = np.isfinite(x[:, j])
        value = float(np.median(x[finite, j])) if finite.any() else 0.0
        x[~finite, j] = value
    return x


def novelty_audit(candidates: pl.DataFrame, existing: pl.DataFrame,
                  users: np.ndarray) -> tuple[list[str], list[dict[str, Any]]]:
    """Target-free exact/near-duplicate screen, fixed before model fitting."""
    from scipy.stats import rankdata

    ids = np.asarray(users, np.int64)
    hashes = splitmix64(ids)
    take = np.argsort(hashes)[:min(NOVELTY_SAMPLE, len(ids))]
    left = pl.DataFrame({"user_id": ids, "__order": np.arange(len(ids))})
    c = (left.join(candidates, on="user_id", how="left").sort("__order")
         .select(FINGERPRINT_FIELDS).to_numpy())[take]
    base_names = [name for name in existing.columns if name != "user_id"]
    b = (left.join(existing, on="user_id", how="left").sort("__order")
         .select(base_names).to_numpy())[take]
    c, b = _fill_for_corr(c), _fill_for_corr(b)
    kept: list[str] = []
    kept_arrays: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    rb = np.column_stack([rankdata(b[:, j], method="average") for j in range(b.shape[1])])
    for j, name in enumerate(FINGERPRINT_FIELDS):
        x = c[:, j]
        row: dict[str, Any] = {"feature": name, "n_audit": len(x)}
        if float(np.std(x)) == 0.0:
            row.update(status="REMOVED_CONSTANT", matched="", max_abs_pearson=1.0,
                       max_abs_spearman=1.0)
            rows.append(row); continue
        refs = b if not kept_arrays else np.column_stack([b, *kept_arrays])
        ref_names = base_names + kept
        xr = rankdata(x, method="average")
        pearson = np.array([np.corrcoef(x, refs[:, k])[0, 1] if np.std(refs[:, k]) else 0.0
                            for k in range(refs.shape[1])])
        rank_refs = rb if not kept_arrays else np.column_stack(
            [rb, *[rankdata(v, method="average") for v in kept_arrays]])
        spearman = np.array([np.corrcoef(xr, rank_refs[:, k])[0, 1]
                             if np.std(rank_refs[:, k]) else 0.0
                             for k in range(rank_refs.shape[1])])
        ip = int(np.nanargmax(np.abs(pearson)))
        is_ = int(np.nanargmax(np.abs(spearman)))
        exact = any(np.array_equal(x, refs[:, k]) for k in range(refs.shape[1]))
        near = (abs(pearson[ip]) >= NEAR_PEARSON or abs(spearman[is_]) >= NEAR_SPEARMAN)
        matched = ref_names[ip] if abs(pearson[ip]) >= abs(spearman[is_]) else ref_names[is_]
        status = "REMOVED_EXACT" if exact else "REMOVED_NEAR" if near else "KEPT"
        row.update(status=status, matched=matched,
                   max_abs_pearson=float(abs(pearson[ip])),
                   max_abs_spearman=float(abs(spearman[is_])))
        rows.append(row)
        if status == "KEPT":
            kept.append(name); kept_arrays.append(x)
    if not kept:
        raise RuntimeError("novelty audit removed every fingerprint field")
    return kept, rows


def _availability_audit(test_fp: pl.DataFrame, kept: list[str]) -> dict[str, Any]:
    matrix = test_fp.select(kept).to_numpy()
    forbidden = [name for name in kept if any(token in name.lower()
                                              for token in FORBIDDEN_FEATURE_TOKENS)]
    source = inspect.getsource(build_fingerprint_features)
    audit = {
        "test_cutoff": CUTOFF_TEST.isoformat(), "rows": test_fp.height,
        "unique_users": test_fp["user_id"].n_unique(), "schema_identical": True,
        "kept_fields": kept, "all_finite": bool(np.isfinite(matrix).all()),
        "forbidden_feature_names": forbidden,
        "builder_accepts_target": "target" in inspect.signature(build_fingerprint_features).parameters,
        "history_filter_present": "cutoff" in source,
        "raw_files": 1, "physical_partitions": 1, "rowgroups": 250,
        "metadata_definition_available_at_test": True,
    }
    audit["status"] = "PASS" if (audit["rows"] == 250_000 and audit["unique_users"] == 250_000
                                        and audit["all_finite"] and not forbidden
                                        and not audit["builder_accepts_target"]) else "FAIL"
    return audit


def build_fingerprints_and_controls() -> dict[str, Any]:
    setup = component_setup("UNC", SEED, vals=[PILOT_FOLD])
    cuts = setup.train_cutoffs(PILOT_FOLD)
    for i, cutoff in enumerate([*cuts, PILOT_FOLD, CUTOFF_TEST], 1):
        fp = fingerprint_cached(cutoff)
        log(f"  fingerprints {i}/{len(cuts) + 2}: {cutoff} {fp.height:,}x{fp.width - 1}")
    mapping = build_permutation(cuts)
    val_users = panel_users(PILOT_FOLD, 3)["user_id"].to_numpy().astype(np.int64)
    val_fp = fingerprint_cached(PILOT_FOLD)
    e10 = features_cached(PILOT_FOLD, L=None, norm_long=True)
    kept, novelty = novelty_audit(val_fp, e10, val_users)
    write_csv(RESULTS / "novelty_audit.csv", novelty)
    test_audit = _availability_audit(fingerprint_cached(CUTOFF_TEST), kept)
    write_json(RESULTS / "test_metadata_availability.json", test_audit)
    if test_audit["status"] != "PASS":
        raise RuntimeError(f"test metadata unavailable: {test_audit}")

    marginal_rows = []
    for cutoff, blocks in [(c, 1) for c in cuts] + [(PILOT_FOLD, 3)]:
        users = panel_users(cutoff, blocks)["user_id"].to_numpy().astype(np.int64)
        fp = fingerprint_cached(cutoff)
        real = (pl.DataFrame({"user_id": users, "__order": np.arange(len(users))})
                .join(fp.select("user_id", *kept), on="user_id", how="left")
                .sort("__order").select(kept).to_numpy().astype(np.float32))
        perm = permuted_fingerprints(users, fp, mapping, kept)
        mapped = (pl.DataFrame({"user_id": users}).join(
            mapping.select("user_id", "mapped_user_id"), on="user_id")["mapped_user_id"]
                  .to_numpy())
        invariant = np.array_equal(np.sort(mapped), np.sort(users))
        # Exact panel invariance implies exact joint marginal preservation.  Verify
        # it numerically on the validation panel and one early train panel.
        numeric = True
        if cutoff in (cuts[0], PILOT_FOLD):
            numeric = all(np.array_equal(np.sort(real[:, j]), np.sort(perm[:, j]))
                          for j in range(len(kept)))
        marginal_rows.append({"cutoff": cutoff.isoformat(), "blocks": blocks,
                              "rows": len(users), "mapped_panel_invariant": invariant,
                              "numeric_marginals_checked": cutoff in (cuts[0], PILOT_FOLD),
                              "numeric_marginals_identical": numeric})
        if not invariant or not numeric:
            raise AssertionError(f"REAL/PERM marginal contract failed at {cutoff}")
    write_csv(RESULTS / "permutation_marginals.csv", marginal_rows)
    mapped = mapping["mapped_user_id"].to_numpy()
    universe = mapping["user_id"].to_numpy()
    manifest = {
        "seed": SEED, "mapping": "one fixed bijection across all cutoffs",
        "stratification": "exact model-row incidence signature; control-only, never a feature",
        "users": len(universe), "changed_fraction": float(np.mean(mapped != universe)),
        "singleton_fraction": float(np.mean(mapped == universe)),
        "bijection": bool(np.array_equal(np.sort(mapped), np.sort(universe))),
        "mapping_sha256": sha256_array(mapped), "kept_fields": kept,
        "removed_fields": [r["feature"] for r in novelty if r["status"] != "KEPT"],
        "feature_count_before": len(FINGERPRINT_FIELDS), "feature_count_after": len(kept),
        "marginals": marginal_rows, "test_metadata": test_audit,
    }
    write_json(RESULTS / "fingerprint_manifest.json", manifest)
    return manifest


def _reference_base() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if not BASE_REPLAY_NPZ.exists() or not BASE_REPLAY_META.exists():
        raise FileNotFoundError("EXP-046 exact UNC replay is required")
    replay = np.load(BASE_REPLAY_NPZ)
    uid = replay["user_id"].astype(np.int64)
    y = replay["y"].astype(np.float64)
    z = replay["z_r600"].astype(np.float64)
    historical = np.load(ARTIFACTS / "oof_S1-E02.npz")
    mask = np.asarray(historical["cutoff"], dtype="U10") == PILOT_LABEL
    order = np.argsort(np.asarray(historical["user_id"])[mask])
    hu = np.asarray(historical["user_id"])[mask][order]
    hy = np.asarray(historical["y"])[mask][order]
    hz = np.asarray(historical["z"])[mask][order]
    exact = (np.array_equal(uid, hu) and np.array_equal(replay["y"], hy.astype(np.float32))
             and np.array_equal(replay["z_r600"], hz.astype(np.float32)))
    meta = json.loads(BASE_REPLAY_META.read_text(encoding="utf-8"))
    val_x, val_y = xy(PILOT_FOLD, component_setup("UNC", SEED, vals=[PILOT_FOLD]))
    audit = {
        "source": BASE_REPLAY_NPZ, "source_sha256": sha256_file(BASE_REPLAY_NPZ),
        "historical_exact_prediction_match": exact,
        "validation_users_match": bool(np.array_equal(uid, val_x["user_id"].to_numpy())),
        "validation_target_match": bool(np.array_equal(replay["y"], val_y.astype(np.float32))),
        "feature_order_match": feature_names(val_x) == saved_features("UNC"),
        "feature_count": len(saved_features("UNC")), "rounds": ROUNDS, "seed": SEED,
        "n_train": meta["n_train"], "train_target_sha256": meta["train_target_sha256"],
        "recipe_sha256": meta["recipe_sha256"],
    }
    audit["status"] = "PASS" if all(audit[k] for k in (
        "historical_exact_prediction_match", "validation_users_match",
        "validation_target_match", "feature_order_match")) else "FAIL"
    write_json(RESULTS / "base_exact_replay.json", audit)
    if audit["status"] != "PASS":
        raise RuntimeError(f"BASE exact replay failed: {audit}")
    return uid, y, z, meta


def _arm_paths(arm: str) -> dict[str, Path]:
    return {"npz": RUN_DIR / f"{PREFIX}_{arm}.npz",
            "model": RUN_DIR / f"{PREFIX}_{arm}.txt",
            "meta": RUN_DIR / f"{PREFIX}_{arm}.json"}


def _real_fingerprints(users: np.ndarray, fp: pl.DataFrame, fields: list[str]) -> np.ndarray:
    left = pl.DataFrame({"user_id": users, "__order": np.arange(len(users))})
    out = (left.join(fp.select("user_id", *fields), on="user_id", how="left")
           .sort("__order").select(fields))
    if any(out.null_count().row(0)):
        raise AssertionError("REAL fingerprint join produced nulls")
    return out.to_numpy().astype(np.float32)


def _assemble_arm(arm: str, fields: list[str], mapping: pl.DataFrame,
                  cuts: list[dt.date]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    setup = component_setup("UNC", SEED, vals=[PILOT_FOLD])
    base_fields = saved_features("UNC")
    sizes = [panel_users(c, 1).height for c in cuts]
    matrix = np.empty((sum(sizes), len(base_fields) + len(fields)), np.float32)
    targets: list[np.ndarray] = []
    row_hash = hashlib.sha256()
    pos = 0
    for i, (cutoff, n_rows) in enumerate(zip(cuts, sizes), 1):
        xb, yb = xy(cutoff, setup, blocks=1)
        users = xb["user_id"].to_numpy().astype(np.int64)
        base = to_np(xb, base_fields)
        fp = fingerprint_cached(cutoff)
        extra = (_real_fingerprints(users, fp, fields) if arm == "REAL-FP"
                 else permuted_fingerprints(users, fp, mapping, fields))
        if base.shape != (n_rows, len(base_fields)) or extra.shape != (n_rows, len(fields)):
            raise AssertionError("arm matrix block shape mismatch")
        matrix[pos:pos + n_rows, :len(base_fields)] = base
        matrix[pos:pos + n_rows, len(base_fields):] = extra
        targets.append(yb)
        row_hash.update(cutoff.isoformat().encode())
        row_hash.update(np.ascontiguousarray(users).tobytes())
        pos += n_rows
        del xb, yb, base, extra
        log(f"    {arm} matrix {i}/{len(cuts)} {cutoff} rows={pos:,}")
    y = np.concatenate(targets)
    audit = {"rows": len(y), "columns": matrix.shape[1],
             "base_feature_count": len(base_fields), "fingerprint_feature_count": len(fields),
             "feature_order": base_fields + fields,
             "feature_order_sha256": sha256_array(np.asarray(base_fields + fields, dtype="U")),
             "row_order_sha256": row_hash.hexdigest(), "target_sha256": sha256_array(y),
             "target_dtype": str(y.dtype), "cutoffs": [c.isoformat() for c in cuts]}
    return matrix, y, audit


def train_arm(arm: str, fields: list[str], mapping: pl.DataFrame,
              cuts: list[dt.date], reference_meta: dict[str, Any]) -> dict[str, Any]:
    paths = _arm_paths(arm)
    complete = all(path.exists() for path in paths.values())
    if complete:
        return json.loads(paths["meta"].read_text(encoding="utf-8"))
    if any(path.exists() for path in paths.values()):
        raise FileExistsError(f"partial arm artifacts; refusing overwrite: {paths}")
    t0 = time.time()
    matrix, y, assembly = _assemble_arm(arm, fields, mapping, cuts)
    if len(y) != reference_meta["n_train"] or assembly["target_sha256"] != reference_meta["train_target_sha256"]:
        raise AssertionError(f"{arm} rows/target differ from exact BASE replay")
    ds = models.make_datasets("direct", matrix, y, None, {"seed": SEED})[0]
    del matrix, y
    _XY.clear(); gc.collect()
    booster = models.train_direct_ds(ds, {"seed": SEED}, rounds=ROUNDS)
    setup = component_setup("UNC", SEED, vals=[PILOT_FOLD])
    xv, yv = xy(PILOT_FOLD, setup)
    users = xv["user_id"].to_numpy().astype(np.int64)
    base = to_np(xv, saved_features("UNC"))
    fp = fingerprint_cached(PILOT_FOLD)
    extra = (_real_fingerprints(users, fp, fields) if arm == "REAL-FP"
             else permuted_fingerprints(users, fp, mapping, fields))
    av = np.hstack([base, extra])
    z = np.maximum(booster.predict(av), 0.0).astype(np.float32)
    report = evaluate(yv, z, np.full(len(yv), PILOT_LABEL, dtype="U10"))
    gain = booster.feature_importance("gain").astype(np.float64)
    split = booster.feature_importance("split").astype(np.int64)
    booster.save_model(str(paths["model"]), num_iteration=ROUNDS)
    np.savez_compressed(paths["npz"], user_id=users,
                        cutoff=np.full(len(users), PILOT_LABEL, dtype="U10"),
                        y=yv.astype(np.float32), z=z, importance_gain=gain,
                        importance_split=split)
    meta = {
        "arm": arm, "prefix": PREFIX, "seed": SEED, "rounds": ROUNDS,
        "params": {**LGB_PARAMS, "seed": SEED}, "early_stopping": False,
        "assembly": assembly, "validation_rows": len(users),
        "validation_row_sha256": sha256_array(users),
        "validation_target_sha256": sha256_array(yv.astype(np.float32)),
        "prediction_sha256": sha256_array(z), "rmsle_cal": report["fold_cal"][0],
        "offset": report["per_fold"][0]["offset"], "auc": auc_positive(yv, z),
        "runtime_s": time.time() - t0,
        "artifacts": {k: str(v.resolve()) for k, v in paths.items() if k != "meta"},
    }
    write_json(paths["meta"], meta)
    del booster, ds, xv, yv, base, extra, av
    _XY.clear(); gc.collect()
    return meta


def run_models() -> dict[str, Any]:
    manifest = json.loads((RESULTS / "fingerprint_manifest.json").read_text(encoding="utf-8"))
    fields = manifest["kept_fields"]
    mapping = pl.read_parquet(RUN_DIR / "permutation.parquet")
    _, _, _, ref_meta = _reference_base()
    setup = component_setup("UNC", SEED, vals=[PILOT_FOLD])
    cuts = setup.train_cutoffs(PILOT_FOLD)
    perm = train_arm("PERM-FP", fields, mapping, cuts, ref_meta)
    real = train_arm("REAL-FP", fields, mapping, cuts, ref_meta)
    same = (perm["assembly"]["rows"] == real["assembly"]["rows"]
            and perm["assembly"]["columns"] == real["assembly"]["columns"]
            and perm["assembly"]["row_order_sha256"] == real["assembly"]["row_order_sha256"]
            and perm["assembly"]["target_sha256"] == real["assembly"]["target_sha256"]
            and perm["assembly"]["feature_order_sha256"] == real["assembly"]["feature_order_sha256"]
            and perm["params"] == real["params"] and perm["rounds"] == real["rounds"])
    out = {"base": "PASS exact EXP-046 replay", "perm": perm, "real": real,
           "same_rows_features_config": same}
    write_json(RESULTS / "model_manifest.json", out)
    if not same:
        raise AssertionError("PERM/REAL paired model contract failed")
    return out


def _load_fold_component(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    mask = np.asarray(data["cutoff"], dtype="U10") == PILOT_LABEL
    order = np.argsort(np.asarray(data["user_id"])[mask])
    return (np.asarray(data["user_id"])[mask][order].astype(np.int64),
            np.asarray(data["y"])[mask][order].astype(np.float64),
            np.asarray(data["z"])[mask][order].astype(np.float64))


def _strongest_fold() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    uid0 = y0 = None
    z = None
    components: dict[str, np.ndarray] = {}
    for name, (path, weight) in BASE_COMPONENTS.items():
        uid, y, pred = _load_fold_component(path)
        if uid0 is None:
            uid0, y0 = uid, y
            z = np.zeros(len(uid), np.float64)
        elif not np.array_equal(uid0, uid) or not np.array_equal(y0, y):
            raise AssertionError(f"STRONGEST component alignment failed: {name}")
        components[name] = pred
        z += weight * pred
    assert uid0 is not None and y0 is not None and z is not None
    score = evaluate(y0, z, np.full(len(y0), PILOT_LABEL, dtype="U10"))["fold_cal"][0]
    if abs(score - EXPECTED_STRONG) > 5e-7:
        raise AssertionError(f"STRONGEST exact reconstruction failed: {score}")
    return uid0, y0, z, components


def _metrics(y: np.ndarray, z: np.ndarray) -> dict[str, Any]:
    offset, score = calibrate(y, z)
    pred = np.maximum(z + offset, 0.0)
    residual = np.log1p(y) - pred
    zero = y == 0
    return {"rmsle_raw": rmsle_z(y, z), "offset": offset, "rmsle_cal": score,
            "auc": auc_positive(y, z), "mean_z": float(np.mean(z)),
            "zero_rmsle_fixed_offset": float(np.sqrt(np.mean(residual[zero] ** 2))),
            "positive_rmsle_fixed_offset": float(np.sqrt(np.mean(residual[~zero] ** 2))),
            "zero_mse_fixed_offset": float(np.mean(residual[zero] ** 2)),
            "positive_mse_fixed_offset": float(np.mean(residual[~zero] ** 2))}


def _fixed_group_score(y: np.ndarray, z: np.ndarray, offset: float, mask: np.ndarray) -> float:
    residual = np.log1p(y[mask]) - np.maximum(z[mask] + offset, 0.0)
    return float(np.sqrt(np.mean(residual ** 2)))


def _cohort_rows(y: np.ndarray, predictors: dict[str, np.ndarray], metrics: dict[str, dict[str, Any]],
                 cohorts: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for axis, labels in cohorts.items():
        for value in np.unique(labels):
            mask = labels == value
            if mask.sum() < 100:
                continue
            scores = {name: _fixed_group_score(y, z, metrics[name]["offset"], mask)
                      for name, z in predictors.items()}
            rows.append({"axis": axis, "cohort": int(value), "n": int(mask.sum()),
                         **{f"rmsle_{k}": v for k, v in scores.items()},
                         "real_minus_perm_slot": scores["REAL_SLOT"] - scores["PERM_SLOT"],
                         "real_minus_strongest": scores["REAL_SLOT"] - scores["STRONGEST"]})
    return rows


def _importance_analysis(fields: list[str], perm: np.lib.npyio.NpzFile,
                         real: np.lib.npyio.NpzFile) -> dict[str, Any]:
    from scipy.stats import spearmanr

    names = saved_features("UNC") + fields
    gp = np.asarray(perm["importance_gain"], np.float64)
    gr = np.asarray(real["importance_gain"], np.float64)
    fp_start = len(saved_features("UNC"))
    fp_gain = gr[fp_start:]
    total_fp = float(fp_gain.sum())
    order = np.argsort(fp_gain)[::-1]
    top = [{"feature": fields[i], "gain": float(fp_gain[i]),
            "share_of_fp_gain": float(fp_gain[i] / total_fp) if total_fp else 0.0}
           for i in order[:min(15, len(fields))]]
    return {
        "feature_names": names, "perm_real_gain_spearman_all": float(spearmanr(gp, gr).statistic),
        "perm_real_gain_spearman_fingerprint": float(
            spearmanr(gp[fp_start:], gr[fp_start:]).statistic) if len(fields) > 1 else 1.0,
        "real_fingerprint_gain_share_total": float(total_fp / gr.sum()) if gr.sum() else 0.0,
        "real_top_fingerprint_share": top[0]["share_of_fp_gain"] if top else 0.0,
        "real_top_fingerprint": top[0]["feature"] if top else None,
        "top_fingerprint_features": top,
    }


def analyze() -> dict[str, Any]:
    manifest = json.loads((RESULTS / "fingerprint_manifest.json").read_text(encoding="utf-8"))
    fields = manifest["kept_fields"]
    uid, y, strongest, components = _strongest_fold()
    base_uid, base_y, base_unc, _ = _reference_base()
    if not np.array_equal(uid, base_uid) or not np.array_equal(y.astype(np.float32), base_y.astype(np.float32)):
        raise AssertionError("BASE replay/STRONGEST alignment failed")
    perm = np.load(_arm_paths("PERM-FP")["npz"])
    real = np.load(_arm_paths("REAL-FP")["npz"])
    for arm, data in (("PERM", perm), ("REAL", real)):
        if not np.array_equal(uid, data["user_id"]) or not np.array_equal(y.astype(np.float32), data["y"]):
            raise AssertionError(f"{arm} validation alignment failed")
    z_perm = np.asarray(perm["z"], np.float64)
    z_real = np.asarray(real["z"], np.float64)
    perm_slot = strongest + 0.20 * (z_perm - components["UNC"])
    real_slot = strongest + 0.20 * (z_real - components["UNC"])
    predictors = {"BASE_UNC": base_unc, "PERM_FP": z_perm, "REAL_FP": z_real,
                  "STRONGEST": strongest, "PERM_SLOT": perm_slot, "REAL_SLOT": real_slot}
    metrics = {name: _metrics(y, z) for name, z in predictors.items()}
    delta_standalone = metrics["REAL_FP"]["rmsle_cal"] - metrics["PERM_FP"]["rmsle_cal"]
    delta_slot_perm = metrics["REAL_SLOT"]["rmsle_cal"] - metrics["PERM_SLOT"]["rmsle_cal"]
    delta_slot_strong = metrics["REAL_SLOT"]["rmsle_cal"] - metrics["STRONGEST"]["rmsle_cal"]
    dz = z_real - z_perm
    dz_slot = real_slot - perm_slot
    perm_cal = np.maximum(perm_slot + metrics["PERM_SLOT"]["offset"], 0.0)
    residual = np.log1p(y) - perm_cal
    residual_alignment = float(np.corrcoef(dz_slot - dz_slot.mean(), residual)[0, 1])
    fp = fingerprint_cached(PILOT_FOLD)
    meta = (pl.DataFrame({"user_id": uid, "__order": np.arange(len(uid))})
            .join(fp, on="user_id", how="left").sort("__order"))
    first_day = meta["fp_first_observed_day"].to_numpy().astype(np.int64)
    month_lookup = np.array([(DATA_START + dt.timedelta(days=int(d))).year * 12
                             + (DATA_START + dt.timedelta(days=int(d))).month
                             - (2025 * 12 + 1) for d in first_day])
    cohorts = {
        "user_hash_half": (splitmix64(uid) & np.uint64(1)).astype(np.int8),
        "uid_rank_bucket256": meta["fp_uid_rank_bucket256"].to_numpy().astype(np.int16),
        "sample_rank_bucket256": np.minimum(
            (meta["fp_sample_rank_frac"].to_numpy() * 256).astype(np.int16), 255),
        "first_observed_month": month_lookup.astype(np.int16),
        "raw_rowgroup_first": meta["fp_rowgroup_first"].to_numpy().astype(np.int16),
    }
    cohort_rows = _cohort_rows(y, predictors, metrics, cohorts)
    write_csv(RESULTS / "cohort_diagnostics.csv", cohort_rows)
    mapping = pl.read_parquet(RUN_DIR / "permutation.parquet")
    real_fp = _real_fingerprints(uid, fp, fields).astype(np.float64)
    perm_fp = permuted_fingerprints(uid, fp, mapping, fields).astype(np.float64)
    treatment_rows = []
    for j, field in enumerate(fields):
        changed = real_fp[:, j] != perm_fp[:, j]
        treatment_rows.append({
            "feature": field, "changed_fraction_validation": float(np.mean(changed)),
            "var_real_minus_perm": float(np.var(real_fp[:, j] - perm_fp[:, j])),
            "pearson_real_perm": float(np.corrcoef(real_fp[:, j], perm_fp[:, j])[0, 1])
            if np.std(real_fp[:, j]) and np.std(perm_fp[:, j]) else 1.0,
            "intervened": bool(changed.any()),
        })
    write_csv(RESULTS / "treatment_field_audit.csv", treatment_rows)
    invariant_treatment_fields = [r["feature"] for r in treatment_rows if not r["intervened"]]
    halves = [r for r in cohort_rows if r["axis"] == "user_hash_half"]
    both_halves = (len(halves) == 2 and all(r["real_minus_perm_slot"] < 0
                                            and r["real_minus_strongest"] < 0 for r in halves))
    importance = _importance_analysis(fields, perm, real)
    write_json(RESULTS / "feature_importance.json", importance)
    endpoint = max(delta_slot_perm, delta_slot_strong)
    mechanism = residual_alignment > 0 and delta_slot_perm < 0
    available = manifest["test_metadata"]["status"] == "PASS"
    suspicious_only = importance["real_top_fingerprint_share"] >= 0.80
    if not available or not both_halves or delta_slot_perm >= -0.0003 or delta_slot_strong >= -0.0003:
        verdict = "REJECT"
    elif endpoint <= -0.0007 and mechanism and not suspicious_only:
        verdict = "STRONG_PASS"
    elif endpoint <= -0.0005 and mechanism and not suspicious_only:
        verdict = "PASS_TO_FULL_FOLDS"
    elif endpoint <= -0.0003:
        verdict = "BORDERLINE_STOP"
    else:
        verdict = "REJECT"
    summary = {
        "experiment": EXP_ID, "prefix": PREFIX, "fold": PILOT_LABEL,
        "base_exact_replay": "PASS", "integrity_audit": "PASS",
        "fingerprints_before_novelty": len(FINGERPRINT_FIELDS),
        "fingerprints_after_novelty": len(fields), "kept_fields": fields,
        "metrics": metrics,
        "primary_deltas": {
            "REAL_minus_PERM_standalone": delta_standalone,
            "REAL_SLOT_minus_PERM_SLOT": delta_slot_perm,
            "REAL_SLOT_minus_STRONGEST": delta_slot_strong,
            "auc_REAL_minus_PERM": metrics["REAL_FP"]["auc"] - metrics["PERM_FP"]["auc"],
            "zero_rmsle_REAL_SLOT_minus_PERM_SLOT":
                metrics["REAL_SLOT"]["zero_rmsle_fixed_offset"]
                - metrics["PERM_SLOT"]["zero_rmsle_fixed_offset"],
            "positive_rmsle_REAL_SLOT_minus_PERM_SLOT":
                metrics["REAL_SLOT"]["positive_rmsle_fixed_offset"]
                - metrics["PERM_SLOT"]["positive_rmsle_fixed_offset"],
        },
        "residual_alignment_corr": residual_alignment,
        "var_delta_standalone": float(np.var(dz)), "var_delta_slot": float(np.var(dz_slot)),
        "mean_delta_standalone": float(np.mean(dz)),
        "both_user_hash_halves_improve": both_halves,
        "gain_not_only_level": bool(delta_slot_perm < 0),
        "mechanism_correct": mechanism, "single_suspicious_field_only": suspicious_only,
        "test_metadata_available": available, "feature_importance": importance,
        "treatment_field_audit": treatment_rows,
        "invariant_under_incidence_matched_permutation": invariant_treatment_fields,
        "verdict": verdict, "full_folds_run": False, "test_inference_run": False,
        "public_lb_used": False,
    }
    write_json(RESULTS / "summary.json", summary)
    write_json(RESULTS / "config.json", {
        "base_recipe": component_setup("UNC", SEED, vals=[PILOT_FOLD]).as_dict(),
        "rounds": ROUNDS, "seed": SEED, "base_features": saved_features("UNC"),
        "fingerprint_features": fields, "novelty_thresholds": {
            "pearson": NEAR_PEARSON, "spearman": NEAR_SPEARMAN},
        "ensemble_weights": {name: weight for name, (_, weight) in BASE_COMPONENTS.items()},
        "slot_replacement": "replace UNC 0.20 only", "weight_tuning": False,
    })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--integrity-only", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    args = parser.parse_args()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.analysis_only:
        summary = analyze()
        log(json.dumps(jsonable(summary["primary_deltas"]), indent=2))
        log("verdict", summary["verdict"])
        return
    run_integrity_audit()
    log("integrity audit complete")
    if args.integrity_only:
        return
    build_fingerprints_and_controls()
    log("fingerprints built")
    if args.build_only:
        return
    run_models()
    log("models complete")
    summary = analyze()
    log("analysis complete")
    log(json.dumps(jsonable(summary["primary_deltas"]), indent=2))
    log("verdict", summary["verdict"])


if __name__ == "__main__":
    main()
