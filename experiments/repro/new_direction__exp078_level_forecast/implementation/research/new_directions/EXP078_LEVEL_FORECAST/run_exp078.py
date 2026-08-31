"""EXP078: strict-forward forecast of the global 30-day log-GMV level.

This is deliberately a scalar/panel experiment.  It does not train or search a
user-level model and it never reads leaderboard history.  The only score values
below are the incumbent and target explicitly supplied in the EXP078 protocol,
used solely to reproduce the headroom identity.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OZON = Path(r"C:\Users\Admin\Desktop\OZON-E-CUP")
PROCESSED = OZON / "data" / "processed"
ALPHA_PATH = Path(r"C:\Users\Admin\Downloads\SUBMIT_ORTH_ALPHA.csv")
SUBMISSION_PATH = ROOT / "submissions" / "SUBMIT_EXP078_LEVEL_FORECAST.csv"

DATA_START = date(2025, 1, 1)
DATA_END = date(2026, 2, 13)
TEST_CUTOFF = date(2026, 2, 13)
SELECTION_START = date(2025, 11, 16)
CLEAN_TARGET_END = date(2025, 11, 15)
GRID_START = date(2025, 4, 3)  # first 14d-grid point with a complete recent-90d window
GRID_LAST_LABEL = date(2026, 1, 8)
CANONICAL_FOLDS = [date(2025, 9, 4), date(2025, 9, 18),
                   date(2025, 10, 2), date(2025, 10, 16)]
CANONICAL_WEIGHTS = np.asarray([1.0, 2.0, 4.0, 8.0])

ANCHOR_RMSLE = 1.6461597403364463
TARGET_RMSLE = 1.6446514942
LEVEL_RMSE_GATE = 0.070
RELATIVE_GATE = 0.20
BIAS_GATE = 0.025
DEPLOY_ABS_GATE = 0.05
MAX_LAMBDA = 0.5

# Pre-registered forecaster settings.  There is no model/alpha sweep.
RIDGE_ALPHA = 10.0
MIN_TRAIN_LABELS = 6
CORRELATION_CUTOFF = 0.995
MAX_L2_FEATURES = 8
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 7802026

# Exactly the real panel channels from the verified sequence artifact.  The
# four leading stored fields are derived indicators; the remaining 10 plus cat
# are the 11 raw competition channels.
PANEL_CHANNELS = [
    "present", "cat", "buy", "ponly", "searches", "search_to_cart",
    "search_to_ord", "cat_to_cart", "cat_to_ord", "to_cart", "to_ord",
    "gmv_search", "gmv_cat", "gmv",
]
RAW_CHANNELS = [
    "cat", "searches", "search_to_cart", "search_to_ord", "cat_to_cart",
    "cat_to_ord", "to_cart", "to_ord", "gmv_search", "gmv_cat", "gmv",
]

# Small target-free pool.  Priority is fixed, correlation pruning is performed
# only on the training history visible at each origin, and at most 8 survive.
L2_POOL = [
    "state_level_gmv30",
    "state_level_gmv90",
    "log1p_gmv30_q75",
    "log1p_gmv30_q90",
    "frac_purchase_30",
    "mean_orders_30",
    "mean_searches_30",
    "mean_event_days_30",
    "mean_recency_purchase",
    "step_level_gmv30",
]


@dataclass
class Forecast:
    value: float
    features: list[str]
    beta: float | None = None


def iso(d: date) -> str:
    return d.isoformat()


def parse_date(value: Any) -> date:
    return date.fromisoformat(str(value)[:10])


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def write_bytes_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite a different artifact: {path}")
        return
    path.write_bytes(payload)


def write_text_once(path: Path, text: str) -> None:
    write_bytes_once(path, text.encode("utf-8"))


def write_json_once(path: Path, value: Any) -> None:
    write_text_once(path, json.dumps(jsonable(value), ensure_ascii=False, indent=2) + "\n")


def write_csv_once(path: Path, frame: pd.DataFrame) -> None:
    write_text_once(path, frame.to_csv(index=False, lineterminator="\n"))


def write_parquet_once(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_parquet(path)
        pd.testing.assert_frame_equal(old, frame, check_exact=True, check_dtype=True)
        return
    frame.to_parquet(path, index=False)


def day_index(d: date) -> int:
    return (d - DATA_START).days


def date_grid(start: date, end: date, step: int) -> list[date]:
    out: list[date] = []
    cur = start
    while cur <= end:
        out.append(cur)
        cur += timedelta(days=step)
    return out


def recency_mean(binary: np.ndarray, cap: int) -> float:
    has = binary.any(axis=1)
    rec = np.full(len(binary), float(cap), dtype=np.float64)
    if np.any(has):
        rec[has] = np.argmax(binary[has, ::-1], axis=1)
    return float(np.mean(rec))


def summarize_values(prefix: str, values: np.ndarray) -> dict[str, float]:
    q25, q50, q75, q90 = np.quantile(values, [0.25, 0.50, 0.75, 0.90])
    return {
        f"gmv{prefix}_mean": float(np.mean(values)),
        f"gmv{prefix}_q25": float(q25),
        f"gmv{prefix}_median": float(q50),
        f"gmv{prefix}_q75": float(q75),
        f"gmv{prefix}_q90": float(q90),
        f"state_level_gmv{prefix}": float(np.mean(np.log1p(values))),
    }


def eligibility_mask(present: np.ndarray, d: int) -> np.ndarray:
    mask = np.ones(len(present), dtype=bool)
    for block in range(3):
        end = d - 30 * block
        start = end - 29
        if start < 0:
            return np.zeros(len(present), dtype=bool)
        mask &= np.any(present[:, start:end + 1] > 0, axis=1)
    return mask


def build_panel_rows(cutoffs: list[date], cohort: str,
                     gmv: np.ndarray, panel: np.ndarray) -> pd.DataFrame:
    if cohort not in {"fixed_final_250k", "historical_eligible"}:
        raise ValueError(cohort)
    rows: list[dict[str, Any]] = []
    chunk = 25_000
    for cutoff in cutoffs:
        d = day_index(cutoff)
        target_end = cutoff + timedelta(days=30)
        if d < 89:
            raise AssertionError("recent-90d window is incomplete")

        buckets: dict[str, list[np.ndarray]] = {
            "gmv30": [], "gmv60": [], "gmv90": [], "prev30": [], "prev60": [],
            "orders30": [], "searches30": [], "event30": [], "buydays30": [],
            "rec_any": [], "rec_buy": [], "target": [],
        }
        cohort_n = 0
        for lo in range(0, gmv.shape[0], chunk):
            hi = min(lo + chunk, gmv.shape[0])
            gb = np.asarray(gmv[lo:hi], dtype=np.float64)
            present = np.asarray(panel[lo:hi, :, PANEL_CHANNELS.index("present")], dtype=np.float32)
            buy = np.asarray(panel[lo:hi, :, PANEL_CHANNELS.index("buy")], dtype=np.float32)
            if cohort == "fixed_final_250k":
                mask = np.ones(hi - lo, dtype=bool)
            else:
                mask = eligibility_mask(present, d)
            cohort_n += int(np.sum(mask))
            if not np.any(mask):
                continue

            g = gb[mask]
            p = present[mask]
            b = buy[mask]
            buckets["gmv30"].append(np.sum(g[:, d - 29:d + 1], axis=1))
            buckets["gmv60"].append(np.sum(g[:, d - 59:d + 1], axis=1))
            buckets["gmv90"].append(np.sum(g[:, d - 89:d + 1], axis=1))
            buckets["prev30"].append(np.sum(g[:, d - 59:d - 29], axis=1))
            if d >= 119:
                buckets["prev60"].append(np.sum(g[:, d - 119:d - 59], axis=1))

            orders_log = np.asarray(
                panel[lo:hi, d - 29:d + 1, PANEL_CHANNELS.index("to_ord")],
                dtype=np.float32,
            )[mask]
            searches_log = np.asarray(
                panel[lo:hi, d - 29:d + 1, PANEL_CHANNELS.index("searches")],
                dtype=np.float32,
            )[mask]
            buckets["orders30"].append(np.sum(np.expm1(orders_log), axis=1))
            buckets["searches30"].append(np.sum(np.expm1(searches_log), axis=1))
            buckets["event30"].append(np.sum(p[:, d - 29:d + 1] > 0, axis=1))
            buckets["buydays30"].append(np.sum(b[:, d - 29:d + 1] > 0, axis=1))
            buckets["rec_any"].append(np.asarray([recency_mean(p[:, d - 89:d + 1] > 0, 91)]))
            buckets["rec_buy"].append(np.asarray([recency_mean(b[:, d - 89:d + 1] > 0, 91)]))
            if target_end <= DATA_END:
                buckets["target"].append(np.sum(g[:, d + 1:d + 31], axis=1))

        if cohort == "fixed_final_250k" and cohort_n != 250_000:
            raise AssertionError(f"fixed cohort changed at {cutoff}: {cohort_n}")
        vals = {name: np.concatenate(parts) if parts else np.asarray([], dtype=float)
                for name, parts in buckets.items()}
        for name in ("gmv30", "gmv60", "gmv90", "prev30", "orders30",
                     "searches30", "event30", "buydays30"):
            if len(vals[name]) != cohort_n:
                raise AssertionError(f"bad {name} length at {cutoff}")

        rec_any = float(np.average(vals["rec_any"], weights=[min(chunk, cohort_n - i * chunk)
                        for i in range(len(vals["rec_any"]))])) if len(vals["rec_any"]) else math.nan
        rec_buy = float(np.average(vals["rec_buy"], weights=[min(chunk, cohort_n - i * chunk)
                        for i in range(len(vals["rec_buy"]))])) if len(vals["rec_buy"]) else math.nan
        # Recompute recency means exactly when eligibility made chunk sizes unequal.
        if cohort == "historical_eligible":
            rec_any_num = rec_buy_num = 0.0
            rec_den = 0
            for lo in range(0, gmv.shape[0], chunk):
                hi = min(lo + chunk, gmv.shape[0])
                p = np.asarray(panel[lo:hi, :, 0], dtype=np.float32)
                b = np.asarray(panel[lo:hi, :, 2], dtype=np.float32)
                mask = eligibility_mask(p, d)
                n = int(np.sum(mask))
                if n:
                    rec_any_num += recency_mean(p[mask, d - 89:d + 1] > 0, 91) * n
                    rec_buy_num += recency_mean(b[mask, d - 89:d + 1] > 0, 91) * n
                    rec_den += n
            rec_any, rec_buy = rec_any_num / rec_den, rec_buy_num / rec_den

        row: dict[str, Any] = {
            "cutoff": iso(cutoff),
            "target_start": iso(cutoff + timedelta(days=1)),
            "target_end": iso(target_end),
            "cohort": cohort,
            "cohort_n": cohort_n,
            "is_clean": bool(target_end <= CLEAN_TARGET_END),
            "selection_overlap": bool(target_end >= SELECTION_START),
            "history_days": d + 1,
        }
        row.update(summarize_values("30", vals["gmv30"]))
        row.update(summarize_values("60", vals["gmv60"]))
        row.update(summarize_values("90", vals["gmv90"]))
        row.update({
            "log1p_gmv30_q75": float(np.log1p(row["gmv30_q75"])),
            "log1p_gmv30_q90": float(np.log1p(row["gmv30_q90"])),
            "frac_purchase_30": float(np.mean(vals["gmv30"] > 0)),
            "frac_purchase_60": float(np.mean(vals["gmv60"] > 0)),
            "frac_purchase_90": float(np.mean(vals["gmv90"] > 0)),
            "mean_purchase_days_30": float(np.mean(vals["buydays30"])),
            "mean_orders_30": float(np.mean(vals["orders30"])),
            "mean_searches_30": float(np.mean(vals["searches30"])),
            "mean_event_days_30": float(np.mean(vals["event30"])),
            "mean_recency_any": rec_any,
            "mean_recency_purchase": rec_buy,
            "previous_state_level_gmv30": float(np.mean(np.log1p(vals["prev30"]))),
            "step_level_gmv30": float(row["state_level_gmv30"] - np.mean(np.log1p(vals["prev30"]))),
            "previous_state_level_gmv60": math.nan,
            "step_level_gmv60": math.nan,
            "target_level": math.nan,
            "target_frac_zero": math.nan,
        })
        if len(vals["prev60"]) == cohort_n:
            prev60_level = float(np.mean(np.log1p(vals["prev60"])))
            row["previous_state_level_gmv60"] = prev60_level
            row["step_level_gmv60"] = float(row["state_level_gmv60"] - prev60_level)
        if target_end <= DATA_END:
            if len(vals["target"]) != cohort_n:
                raise AssertionError("target length mismatch")
            row["target_level"] = float(np.mean(np.log1p(vals["target"])))
            row["target_frac_zero"] = float(np.mean(vals["target"] <= 0))
        rows.append(row)
        print(f"panel {cohort:22s} {cutoff} n={cohort_n:6d} "
              f"state={row['state_level_gmv30']:.6f} target={row['target_level']:.6f}", flush=True)
    return pd.DataFrame(rows)


def assert_leakage_protocol(panel: pd.DataFrame) -> None:
    clean = panel[panel.is_clean]
    late = panel[~panel.is_clean]
    if clean.empty or late.empty:
        raise AssertionError("both clean and diagnostic rows are required")
    if max(map(parse_date, clean.target_end)) > CLEAN_TARGET_END:
        raise AssertionError("contaminated target in clean set")
    if min(map(parse_date, late.target_end)) <= CLEAN_TARGET_END:
        raise AssertionError("clean target mislabeled as diagnostic")
    if not late.selection_overlap.all():
        raise AssertionError("late diagnostic does not overlap selection interval")


def available_training(panel: pd.DataFrame, origin: date, clean_only: bool = True,
                       excluded: Iterable[int] = ()) -> list[int]:
    excluded_set = set(excluded)
    idx = []
    for i, row in panel.iterrows():
        if i in excluded_set:
            continue
        if clean_only and not bool(row.is_clean):
            continue
        if parse_date(row.target_end) <= origin and np.isfinite(row.target_level):
            idx.append(int(i))
    if any(parse_date(panel.loc[i, "target_end"]) > origin for i in idx):
        raise AssertionError("label embargo violation")
    if clean_only and any(not bool(panel.loc[i, "is_clean"]) for i in idx):
        raise AssertionError("selection-overlap row used for fitting")
    return idx


def latest_known(panel: pd.DataFrame, origin: date, train_idx: list[int]) -> int:
    if not train_idx:
        raise ValueError("no observed target level")
    return max(train_idx, key=lambda i: parse_date(panel.loc[i, "target_end"]))


def transition_pairs(panel: pd.DataFrame, train_idx: list[int]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    allowed = set(train_idx)
    for end in train_idx:
        origin = parse_date(panel.loc[end, "cutoff"])
        bases = [i for i in train_idx if i != end and i in allowed
                 and parse_date(panel.loc[i, "target_end"]) <= origin]
        if bases:
            base = max(bases, key=lambda i: parse_date(panel.loc[i, "target_end"]))
            pairs.append((base, end))
    return pairs


def prune_features(panel: pd.DataFrame, train_idx: list[int]) -> list[str]:
    keep: list[str] = []
    train = panel.loc[train_idx]
    for feature in L2_POOL:
        values = train[feature].to_numpy(np.float64)
        if not np.isfinite(values).all() or float(np.std(values)) <= 1e-12:
            continue
        correlated = False
        for previous in keep:
            rho = float(np.corrcoef(values, train[previous].to_numpy(np.float64))[0, 1])
            if np.isfinite(rho) and abs(rho) >= CORRELATION_CUTOFF:
                correlated = True
                break
        if not correlated:
            keep.append(feature)
        if len(keep) == MAX_L2_FEATURES:
            break
    if not keep:
        raise AssertionError("correlation pruning removed every L2 feature")
    return keep


def forecast_method(panel: pd.DataFrame, method: str, origin_row: pd.Series,
                    train_idx: list[int]) -> Forecast:
    if len(train_idx) < MIN_TRAIN_LABELS:
        raise ValueError("insufficient training labels")
    origin = parse_date(origin_row.cutoff)
    base = latest_known(panel, origin, train_idx)
    if method == "L0":
        return Forecast(float(panel.loc[base, "target_level"]), [])
    if method == "L1":
        pairs = transition_pairs(panel, train_idx)
        if len(pairs) < 3:
            raise ValueError("insufficient historical transitions")
        dx = np.asarray([panel.loc[e, "state_level_gmv30"] - panel.loc[b, "state_level_gmv30"]
                         for b, e in pairs], dtype=np.float64)
        dy = np.asarray([panel.loc[e, "target_level"] - panel.loc[b, "target_level"]
                         for b, e in pairs], dtype=np.float64)
        den = float(dx @ dx)
        if den <= 1e-15:
            raise ValueError("degenerate L1 proxy steps")
        beta = float(dx @ dy / den)
        step = float(origin_row.state_level_gmv30 - panel.loc[base, "state_level_gmv30"])
        return Forecast(float(panel.loc[base, "target_level"] + beta * step),
                        ["state_level_gmv30"], beta)
    if method == "L2":
        features = prune_features(panel, train_idx)
        X = panel.loc[train_idx, features].to_numpy(np.float64)
        y = panel.loc[train_idx, "target_level"].to_numpy(np.float64)
        x = origin_row[features].to_numpy(np.float64)
        mean = np.mean(X, axis=0)
        scale = np.std(X, axis=0)
        if np.any(scale <= 1e-12) or not np.isfinite(x).all():
            raise AssertionError("invalid L2 design")
        model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
        model.fit((X - mean) / scale, y)
        value = float(model.predict(((x - mean) / scale)[None, :])[0])
        return Forecast(value, features)
    raise ValueError(method)


def rolling_predictions(panel: pd.DataFrame, methods: Iterable[str] = ("L0", "L1", "L2"),
                        min_train: int = MIN_TRAIN_LABELS) -> pd.DataFrame:
    old_min = globals()["MIN_TRAIN_LABELS"]
    rows: list[dict[str, Any]] = []
    for k, row in panel.iterrows():
        if not bool(row.is_clean):
            continue
        origin = parse_date(row.cutoff)
        train_idx = available_training(panel, origin, clean_only=True)
        if len(train_idx) < min_train:
            continue
        for method in methods:
            try:
                # 28d sensitivity may explicitly accept four history labels;
                # L0/L1 need no global mutation, while Ridge is well-defined.
                if len(train_idx) < old_min:
                    if method == "L0":
                        pred = Forecast(float(panel.loc[latest_known(panel, origin, train_idx), "target_level"]), [])
                    elif method == "L1":
                        pairs = transition_pairs(panel, train_idx)
                        if len(pairs) < 2:
                            continue
                        dx = np.asarray([panel.loc[e, "state_level_gmv30"] - panel.loc[b, "state_level_gmv30"] for b, e in pairs])
                        dy = np.asarray([panel.loc[e, "target_level"] - panel.loc[b, "target_level"] for b, e in pairs])
                        beta = float(dx @ dy / (dx @ dx))
                        base = latest_known(panel, origin, train_idx)
                        pred = Forecast(float(panel.loc[base, "target_level"] + beta *
                                              (row.state_level_gmv30 - panel.loc[base, "state_level_gmv30"])),
                                        ["state_level_gmv30"], beta)
                    else:
                        features = prune_features(panel, train_idx)
                        X = panel.loc[train_idx, features].to_numpy(float)
                        y = panel.loc[train_idx, "target_level"].to_numpy(float)
                        mu, sd = X.mean(0), X.std(0)
                        model = Ridge(alpha=RIDGE_ALPHA).fit((X - mu) / sd, y)
                        pred = Forecast(float(model.predict(((row[features].to_numpy(float) - mu) / sd)[None])[0]), features)
                else:
                    pred = forecast_method(panel, method, row, train_idx)
            except ValueError:
                continue
            base = latest_known(panel, origin, train_idx)
            actual = float(row.target_level)
            actual_change = actual - float(panel.loc[base, "target_level"])
            predicted_change = pred.value - float(panel.loc[base, "target_level"])
            rows.append({
                "cutoff": row.cutoff,
                "method": method,
                "n_train_labels": len(train_idx),
                "latest_known_cutoff": panel.loc[base, "cutoff"],
                "latest_known_target_end": panel.loc[base, "target_end"],
                "target_level": actual,
                "level_hat": pred.value,
                "error": pred.value - actual,
                "abs_error": abs(pred.value - actual),
                "actual_change": actual_change,
                "predicted_change": predicted_change,
                "sign_correct": bool(np.sign(actual_change) == np.sign(predicted_change)),
                "beta": pred.beta,
                "features": "|".join(pred.features),
            })
    out = pd.DataFrame(rows)
    common = sorted(set.intersection(*(set(out.loc[out.method == m, "cutoff"]) for m in methods)))
    return out[out.cutoff.isin(common)].reset_index(drop=True)


def method_metrics(rolling: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, d in rolling.groupby("method", sort=True):
        e = d.error.to_numpy(np.float64)
        rho, p = spearmanr(np.arange(len(e)), e)
        last = d.tail(3)
        rows.append({
            "method": method,
            "n_validation": len(d),
            "RMSE_level": float(np.sqrt(np.mean(e * e))),
            "MAE_level": float(np.mean(np.abs(e))),
            "bias": float(np.mean(e)),
            "sign_accuracy": float(np.mean(d.sign_correct)),
            "last3_RMSE": float(np.sqrt(np.mean(np.square(last.error)))),
            "error_time_spearman": float(rho),
            "error_time_spearman_p": float(p),
            "obvious_monotonic_drift": bool(np.isfinite(rho) and abs(rho) >= 0.8 and p < 0.10),
        })
    return pd.DataFrame(rows)


def circular_block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, n, size=math.ceil(n / block))
    idx = np.concatenate([(np.arange(block) + s) % n for s in starts])[:n]
    return idx


def bootstrap_metrics(rolling: pd.DataFrame) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    piv = rolling.pivot(index="cutoff", columns="method", values="error").sort_index()
    errors = {m: piv[m].to_numpy(float) for m in piv.columns}
    draws = {m: {"rmse": [], "bias": []} for m in piv.columns}
    improvement: dict[str, list[float]] = {m: [] for m in piv.columns if m != "L0"}
    n = len(piv)
    for _ in range(BOOTSTRAP_REPLICATES):
        idx = circular_block_indices(n, min(2, n), rng)
        for method, e in errors.items():
            x = e[idx]
            draws[method]["rmse"].append(float(np.sqrt(np.mean(x * x))))
            draws[method]["bias"].append(float(np.mean(x)))
        for method in improvement:
            improvement[method].append(draws[method]["rmse"][-1] / draws["L0"]["rmse"][-1] - 1.0)
    result: dict[str, Any] = {"block_length": min(2, n), "replicates": BOOTSTRAP_REPLICATES, "methods": {}}
    for method in piv.columns:
        result["methods"][method] = {
            "RMSE_95CI": np.quantile(draws[method]["rmse"], [0.025, 0.975]),
            "bias_95CI": np.quantile(draws[method]["bias"], [0.025, 0.975]),
        }
        if method != "L0":
            result["methods"][method]["relative_RMSE_change_95CI"] = np.quantile(improvement[method], [0.025, 0.975])
    return result


def primary_gates(metrics: pd.DataFrame) -> tuple[dict[str, dict[str, bool]], list[str]]:
    l0 = metrics.set_index("method").loc["L0"]
    gates: dict[str, dict[str, bool]] = {}
    passing: list[str] = []
    for method in ("L1", "L2"):
        row = metrics.set_index("method").loc[method]
        g = {
            "RMSE_le_0.070": bool(row.RMSE_level <= LEVEL_RMSE_GATE),
            "at_least_20pct_better_L0": bool(row.RMSE_level <= (1.0 - RELATIVE_GATE) * l0.RMSE_level),
            "last3_not_worse_L0": bool(row.last3_RMSE <= l0.last3_RMSE),
            "abs_bias_le_0.025": bool(abs(row.bias) <= BIAS_GATE),
            "no_obvious_monotonic_drift": bool(not row.obvious_monotonic_drift),
        }
        gates[method] = g
        if all(g.values()):
            passing.append(method)
    return gates, passing


def load_exp077_module():
    path = ROOT / "research" / "new_directions" / "EXP077_FORWARD_STACK" / "run_exp077.py"
    spec = importlib.util.spec_from_file_location("exp077_reuse", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def composition_proxy() -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    exp077 = load_exp077_module()
    canon = pd.read_parquet(exp077.E75 / "clean_forward_predictions.parquet")
    canon["cutoff"] = canon.cutoff.astype(str)
    masks = {fold: canon.cutoff.to_numpy() == fold for fold in exp077.FOLDS}
    y = canon.target_log.to_numpy(np.float64)
    Z, _ = exp077.load_reference_bank(canon)
    sample = pd.read_csv(exp077.SAMPLE_PATH)
    uid = sample.user_id.to_numpy(np.int64)
    z_alpha = exp077.load_submission_log(exp077.ALPHA_PATH, uid)
    reconstruction, _, _ = exp077.reconstruct_alpha(uid, z_alpha)
    shares = {family: reconstruction["family_shares"][family]
              for family in exp077.PROXY_FAMILIES}
    z_proxy, _ = exp077.build_composition_proxy(Z, exp077.REFERENCE_BANK, y, masks, shares)
    return canon, z_proxy, reconstruction


def exact_delta(y: np.ndarray, z: np.ndarray, correction: float) -> tuple[float, float, int]:
    before = float(np.sqrt(np.mean(np.square(y - z))))
    z_new = np.maximum(z + correction, 0.0)
    after = float(np.sqrt(np.mean(np.square(y - z_new))))
    return after * after - before * before, after - before, int(np.sum(z + correction < 0.0))


def production_validation(method: str, eligible: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    canon, z_proxy, reconstruction = composition_proxy()
    rows: list[dict[str, Any]] = []
    for fold in CANONICAL_FOLDS:
        fold_s = iso(fold)
        state = eligible.loc[eligible.cutoff == fold_s]
        if len(state) != 1:
            raise AssertionError(f"missing eligible panel row {fold}")
        state_row = state.iloc[0]
        train_idx = available_training(eligible, fold, clean_only=True)
        pred = forecast_method(eligible, method, state_row, train_idx)
        m = canon.cutoff.to_numpy() == fold_s
        target = canon.loc[m, "target_log"].to_numpy(np.float64)
        z = z_proxy[m]
        if int(np.sum(m)) != int(state_row.cohort_n):
            raise AssertionError(f"eligible cohort size mismatch at {fold}")
        if abs(float(np.mean(target)) - float(state_row.target_level)) > 2e-12:
            raise AssertionError(f"target-level parity failed at {fold}")
        true_c = float(np.mean(target) - np.mean(z))
        raw_c = float(pred.value - np.mean(z))
        d_mse = raw_c * raw_c - 2.0 * raw_c * true_c
        exact_mse, exact_rmsle, n_clip = exact_delta(target, z, raw_c)
        rows.append({
            "Fold": fold_s,
            "target_level": float(np.mean(target)),
            "forecast_target_level": pred.value,
            "mean_z_production_proxy": float(np.mean(z)),
            "true_c": true_c,
            "forecast_c": raw_c,
            "Delta_MSE": d_mse,
            "Delta_RMSLE": exact_rmsle,
            "exact_Delta_MSE": exact_mse,
            "clipped_rows": n_clip,
            "n": int(np.sum(m)),
            "n_train_labels": len(train_idx),
            "features": "|".join(pred.features),
            "beta": pred.beta,
        })
    prod = pd.DataFrame(rows)
    if float(np.max(np.abs(prod.Delta_MSE - prod.exact_Delta_MSE))) > 5e-4:
        raise AssertionError("unexpectedly large clipping discrepancy")

    # Strictly forward calibration diagnostic: a fold sees only earlier pairs.
    lambdas = []
    forward_dmse = []
    forward_drmsle = []
    for k, row in prod.iterrows():
        if k == 0:
            lam = 0.0
        else:
            x = prod.loc[:k - 1, "forecast_c"].to_numpy(float)
            y = prod.loc[:k - 1, "true_c"].to_numpy(float)
            den = float(x @ x)
            lam = 0.0 if den <= 1e-15 else float(np.clip(x @ y / den, 0.0, MAX_LAMBDA))
        corr = lam * float(row.forecast_c)
        m = canon.cutoff.to_numpy() == row.Fold
        target = canon.loc[m, "target_log"].to_numpy(float)
        z = z_proxy[m]
        dm, dr, _ = exact_delta(target, z, corr)
        lambdas.append(lam)
        forward_dmse.append(dm)
        forward_drmsle.append(dr)
    prod["lambda_forward"] = lambdas
    prod["forecast_c_forward_shrunk"] = prod.forecast_c * prod.lambda_forward
    prod["Delta_MSE_forward_shrunk"] = forward_dmse
    prod["Delta_RMSLE_forward_shrunk"] = forward_drmsle

    x = prod.forecast_c.to_numpy(float)
    y = prod.true_c.to_numpy(float)
    lambda_raw = float(x @ y / (x @ x)) if float(x @ x) > 1e-15 else 0.0
    lambda_final = float(np.clip(lambda_raw, 0.0, MAX_LAMBDA))
    calibration = {
        "lambda_unclipped": lambda_raw,
        "lambda_final": lambda_final,
        "hard_bounds": [0.0, MAX_LAMBDA],
        "rolling_lambdas": lambdas,
        "reconstruction_R2": reconstruction["R2"],
        "weighted_raw_Delta_MSE": float(np.average(prod.Delta_MSE, weights=CANONICAL_WEIGHTS)),
        "weighted_raw_Delta_RMSLE": float(np.average(prod.Delta_RMSLE, weights=CANONICAL_WEIGHTS)),
        "weighted_forward_shrunk_Delta_MSE": float(np.average(prod.Delta_MSE_forward_shrunk, weights=CANONICAL_WEIGHTS)),
        "weighted_forward_shrunk_Delta_RMSLE": float(np.average(prod.Delta_RMSLE_forward_shrunk, weights=CANONICAL_WEIGHTS)),
    }
    return prod, calibration


def forecast_test(panel: pd.DataFrame, method: str, test_state: pd.Series,
                  excluded: Iterable[int] = ()) -> Forecast:
    train_idx = [int(i) for i, row in panel.iterrows()
                 if bool(row.is_clean) and np.isfinite(row.target_level) and i not in set(excluded)]
    if any(parse_date(panel.loc[i, "target_end"]) > CLEAN_TARGET_END for i in train_idx):
        raise AssertionError("late label entered TEST fit")
    return forecast_method(panel, method, test_state, train_idx)


def test_uncertainty(rolling: pd.DataFrame, method: str, target_hat: float,
                     raw_c: float, prod: pd.DataFrame) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    e = rolling.loc[rolling.method == method, "error"].to_numpy(float)
    x = prod.forecast_c.to_numpy(float)
    y = prod.true_c.to_numpy(float)
    target_draws = np.empty(BOOTSTRAP_REPLICATES)
    deploy_draws = np.empty(BOOTSTRAP_REPLICATES)
    lambda_draws = np.empty(BOOTSTRAP_REPLICATES)
    for b in range(BOOTSTRAP_REPLICATES):
        ei = circular_block_indices(len(e), min(2, len(e)), rng)
        calibration_error = float(np.mean(e[ei]))
        target_draws[b] = target_hat - calibration_error
        pi = circular_block_indices(len(x), 2, rng)
        xb, yb = x[pi], y[pi]
        den = float(xb @ xb)
        lam = 0.0 if den <= 1e-15 else float(np.clip(xb @ yb / den, 0.0, MAX_LAMBDA))
        lambda_draws[b] = lam
        deploy_draws[b] = lam * (raw_c - calibration_error)
    return {
        "method": method,
        "interpretation": "95% moving-block calibration interval; late selection-overlap labels excluded",
        "target_level_hat_95CI": np.quantile(target_draws, [0.025, 0.975]),
        "deploy_c_95CI": np.quantile(deploy_draws, [0.025, 0.975]),
        "lambda_bootstrap_95CI": np.quantile(lambda_draws, [0.025, 0.975]),
        "rolling_error_RMSE": float(np.sqrt(np.mean(e * e))),
    }


def sensitivity_analysis(panel: pd.DataFrame, eligible: pd.DataFrame,
                         metrics: pd.DataFrame, passing: list[str], selected: str,
                         anchor_mean: float, lam: float) -> pd.DataFrame:
    test_state = panel.loc[panel.cutoff == iso(TEST_CUTOFF)].iloc[0]
    clean_idx = [int(i) for i, r in panel.iterrows() if bool(r.is_clean)]
    rows: list[dict[str, Any]] = []
    for method in ("L1", "L2"):
        f = forecast_test(panel, method, test_state)
        raw = f.value - anchor_mean
        rows.append({"scenario": f"{method}_full_clean", "method": method,
                     "target_level_hat": f.value, "raw_c": raw, "deploy_c": lam * raw,
                     "sign": int(np.sign(raw)), "selected_method": method == selected,
                     "primary_pass": method in passing})
    for omitted in clean_idx:
        f = forecast_test(panel, selected, test_state, excluded=[omitted])
        raw = f.value - anchor_mean
        rows.append({"scenario": f"leave_one_out_{panel.loc[omitted, 'cutoff']}", "method": selected,
                     "target_level_hat": f.value, "raw_c": raw, "deploy_c": lam * raw,
                     "sign": int(np.sign(raw)), "selected_method": True,
                     "primary_pass": True})

    # Explicit first/last exclusions are named even though they are members of LOO.
    for tag, omitted in (("exclude_earliest_clean", clean_idx[0]), ("exclude_latest_clean", clean_idx[-1])):
        f = forecast_test(panel, selected, test_state, excluded=[omitted])
        raw = f.value - anchor_mean
        rows.append({"scenario": tag, "method": selected, "target_level_hat": f.value,
                     "raw_c": raw, "deploy_c": lam * raw, "sign": int(np.sign(raw)),
                     "selected_method": True, "primary_pass": True})

    # 14d -> 28d diagnostic; it never participates in method/parameter choice.
    clean = panel[panel.is_clean].iloc[::2].reset_index(drop=True)
    test28 = forecast_method(clean, selected, test_state, list(range(len(clean))))
    raw28 = test28.value - anchor_mean
    rows.append({"scenario": "spacing_28d_diagnostic", "method": selected,
                 "target_level_hat": test28.value, "raw_c": raw28, "deploy_c": lam * raw28,
                 "sign": int(np.sign(raw28)), "selected_method": True,
                 "primary_pass": True})
    return pd.DataFrame(rows)


def clipping_diagnostic(anchor: pd.DataFrame, correction: float) -> dict[str, Any]:
    z = np.log1p(anchor.predict.to_numpy(np.float64))
    z_new = np.maximum(z + correction, 0.0)
    applied = z_new - z
    return {
        "requested_c": correction,
        "clipped_rows": int(np.sum(z + correction < 0.0)),
        "zero_anchor_rows": int(np.sum(z == 0.0)),
        "mean_applied_log_shift": float(np.mean(applied)),
        "rms_applied_log_shift": float(np.sqrt(np.mean(applied * applied))),
        "max_abs_deviation_from_constant": float(np.max(np.abs(applied - correction))),
    }


def markdown_table(frame: pd.DataFrame, columns: list[str], formats: dict[str, str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---:" if c in formats else "---" for c in columns) + " |"
    lines = [header, sep]
    for _, row in frame.iterrows():
        vals = []
        for col in columns:
            value = row[col]
            vals.append(format(value, formats[col]) if col in formats else str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def create_submission(anchor: pd.DataFrame, correction: float) -> str:
    if SUBMISSION_PATH.exists():
        raise FileExistsError(f"submission already exists; refusing overwrite: {SUBMISSION_PATH}")
    if list(anchor.columns) != ["user_id", "predict"]:
        raise AssertionError("anchor format differs from user_id,predict")
    z = np.log1p(anchor.predict.to_numpy(np.float64))
    pred = np.expm1(np.maximum(z + correction, 0.0))
    out = pd.DataFrame({"user_id": anchor.user_id.to_numpy(np.int64), "predict": pred})
    if len(out) != 250_000 or out.user_id.duplicated().any():
        raise AssertionError("submission key audit failed")
    if not np.isfinite(out.predict).all() or np.any(out.predict < 0):
        raise AssertionError("submission prediction audit failed")
    payload = out.to_csv(index=False, float_format="%.10f", lineterminator="\n").encode("utf-8")
    write_bytes_once(SUBMISSION_PATH, payload)
    check = pd.read_csv(SUBMISSION_PATH)
    if not np.array_equal(check.user_id.to_numpy(np.int64), anchor.user_id.to_numpy(np.int64)):
        raise AssertionError("serialized order mismatch")
    if not np.isfinite(check.predict).all() or np.any(check.predict < 0):
        raise AssertionError("serialized prediction audit failed")
    return sha256_file(SUBMISSION_PATH)


def main() -> None:
    required = [PROCESSED / "seq_gmv_v1.npy", PROCESSED / "seq_panel_v1.npy",
                PROCESSED / "seq_uid_v1.npy", ALPHA_PATH]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    if PANEL_CHANNELS[1:2] + PANEL_CHANNELS[4:] != RAW_CHANNELS:
        raise AssertionError("11-channel mapping drift")

    headroom_delta_mse = TARGET_RMSLE ** 2 - ANCHOR_RMSLE ** 2
    equivalent_c = math.sqrt(-headroom_delta_mse)
    headroom = {
        "anchor_RMSLE": ANCHOR_RMSLE,
        "target_RMSLE": TARGET_RMSLE,
        "remaining_Delta_RMSLE": TARGET_RMSLE - ANCHOR_RMSLE,
        "remaining_Delta_MSE": headroom_delta_mse,
        "equivalent_abs_intercept": equivalent_c,
        "identity": "Delta MSE = c^2 - 2*c*mean(target_log-z); c*=mean(target_log)-mean(z)",
    }
    print(json.dumps(headroom, indent=2), flush=True)

    gmv = np.load(PROCESSED / "seq_gmv_v1.npy", mmap_mode="r")
    panel_array = np.load(PROCESSED / "seq_panel_v1.npy", mmap_mode="r")
    uid = np.load(PROCESSED / "seq_uid_v1.npy", mmap_mode="r")
    if gmv.shape != (250_000, 409) or panel_array.shape != (250_000, 409, 14):
        raise AssertionError(f"unexpected panel shapes: {gmv.shape}, {panel_array.shape}")
    if len(uid) != 250_000 or np.any(np.diff(uid) <= 0):
        raise AssertionError("fixed cohort id audit failed")

    labeled_grid = date_grid(GRID_START, GRID_LAST_LABEL, 14)
    # Canonical folds must be exact grid members, not nearest-date matches.
    if not set(CANONICAL_FOLDS).issubset(set(labeled_grid)):
        raise AssertionError("canonical folds missing from 14d grid")
    build_grid = labeled_grid + [TEST_CUTOFF]
    fixed = build_panel_rows(build_grid, "fixed_final_250k", gmv, panel_array)
    eligible = build_panel_rows(labeled_grid, "historical_eligible", gmv, panel_array)
    assert_leakage_protocol(fixed[fixed.cutoff != iso(TEST_CUTOFF)])
    assert_leakage_protocol(eligible)

    # The requested artifact contains the fixed cohort and labelled sequence;
    # TEST state is kept separately because target_level is unknowable.
    panel_dataset = fixed[fixed.cutoff != iso(TEST_CUTOFF)].reset_index(drop=True)
    write_parquet_once(HERE / "panel_level_dataset.parquet", panel_dataset)
    write_parquet_once(HERE / "panel_level_dataset_eligible.parquet", eligible.reset_index(drop=True))
    write_parquet_once(HERE / "test_panel_state.parquet", fixed[fixed.cutoff == iso(TEST_CUTOFF)].reset_index(drop=True))

    clean_fixed = panel_dataset[panel_dataset.is_clean].reset_index(drop=True)
    if len(clean_fixed) != 15 or parse_date(clean_fixed.iloc[-1].target_end) != CLEAN_TARGET_END:
        raise AssertionError("unexpected primary clean grid")
    rolling = rolling_predictions(clean_fixed)
    metrics = method_metrics(rolling)
    boot = bootstrap_metrics(rolling)
    gates, passing = primary_gates(metrics)
    write_csv_once(HERE / "rolling_validation.csv", rolling)
    write_csv_once(HERE / "rolling_metrics.csv", metrics)
    write_json_once(HERE / "bootstrap.json", boot)

    l0_rmse = float(metrics.set_index("method").loc["L0", "RMSE_level"])
    if passing:
        selected = min(passing, key=lambda m: float(metrics.set_index("method").loc[m, "RMSE_level"]))
    else:
        selected = ""

    anchor = pd.read_csv(ALPHA_PATH)
    if list(anchor.columns) != ["user_id", "predict"] or len(anchor) != 250_000:
        raise AssertionError("ORTH_ALPHA anchor audit failed")
    if anchor.user_id.duplicated().any() or not np.isfinite(anchor.predict).all() or np.any(anchor.predict < 0):
        raise AssertionError("invalid ORTH_ALPHA anchor")
    anchor_mean = float(np.mean(np.log1p(anchor.predict.to_numpy(np.float64))))

    prod = pd.DataFrame()
    calibration: dict[str, Any] = {}
    sensitivity = pd.DataFrame()
    uncertainty: dict[str, Any] = {}
    test_result: dict[str, Any] = {"status": "NOT_RUN_PRIMARY_GATE_FAILED"}
    clipping: dict[str, Any] = {}
    full_gates: dict[str, bool] = {}
    expected_robust: float | None = None
    submission_sha = "N/A"
    verdict = "NO_GO_LEVEL"

    if selected:
        clean_eligible = eligible[eligible.is_clean].reset_index(drop=True)
        prod, calibration = production_validation(selected, clean_eligible)
        write_csv_once(HERE / "production_like_validation.csv", prod)
        write_json_once(HERE / "shrinkage.json", calibration)

        test_state = fixed.loc[fixed.cutoff == iso(TEST_CUTOFF)].iloc[0]
        test_forecast = forecast_test(clean_fixed, selected, test_state)
        raw_c_test = float(test_forecast.value - anchor_mean)
        lam = float(calibration["lambda_final"])
        deploy_c = float(lam * raw_c_test)
        uncertainty = test_uncertainty(rolling, selected, test_forecast.value, raw_c_test, prod)
        sensitivity = sensitivity_analysis(clean_fixed, clean_eligible, metrics, passing, selected,
                                           anchor_mean, lam)
        write_csv_once(HERE / "sensitivity.csv", sensitivity)
        write_json_once(HERE / "test_uncertainty.json", uncertainty)
        clipping = clipping_diagnostic(anchor, deploy_c)
        write_json_once(HERE / "clipping_diagnostic.json", clipping)

        selected_sens = sensitivity[(sensitivity.selected_method) &
                                    sensitivity.scenario.str.startswith(("leave_one_out_", "exclude_", "spacing_"))]
        main_sign = int(np.sign(raw_c_test))
        sign_stable = bool(main_sign != 0 and (selected_sens.sign == main_sign).all())
        both_sign_agree = True
        if set(passing) == {"L1", "L2"}:
            model_signs = sensitivity[sensitivity.scenario.isin(["L1_full_clean", "L2_full_clean"])].sign
            both_sign_agree = bool(len(model_signs) == 2 and model_signs.nunique() == 1 and model_signs.iloc[0] != 0)
        selected_metric = metrics.set_index("method").loc[selected]
        l0_metric = metrics.set_index("method").loc["L0"]
        full_gates = {
            "1_primary_RMSE_le_0.070": bool(selected_metric.RMSE_level <= LEVEL_RMSE_GATE),
            "2_at_least_20pct_better_freeze": bool(selected_metric.RMSE_level <= 0.8 * l0_metric.RMSE_level),
            "3_last3_not_worse_freeze": bool(selected_metric.last3_RMSE <= l0_metric.last3_RMSE),
            "4_production_improves_at_least_3of4": bool(int(np.sum(prod.Delta_MSE < 0)) >= 3),
            "5_recency_weighted_production_Delta_MSE_negative": bool(calibration["weighted_raw_Delta_MSE"] < 0),
            "6_latest_canonical_Delta_MSE_nonpositive": bool(float(prod.iloc[-1].Delta_MSE) <= 0),
            "7_forward_calibrated_lambda_positive": bool(lam > 0),
            "8_lambda_le_0.5": bool(lam <= MAX_LAMBDA),
            "9_abs_deploy_c_le_0.05": bool(abs(deploy_c) <= DEPLOY_ABS_GATE),
            "10_passing_L1_L2_sign_agreement": both_sign_agree,
            "11_no_obvious_monotonic_error_drift": bool(not selected_metric.obvious_monotonic_drift),
            "12_selected_method_sensitivity_sign_stable": sign_stable,
        }
        expected_robust = float(calibration["weighted_forward_shrunk_Delta_RMSLE"])
        verdict = "GO" if all(full_gates.values()) else "NO_GO"
        test_result = {
            "status": "FORECASTED_NOT_DEPLOYED" if verdict != "GO" else "DEPLOYABLE",
            "method": selected,
            "target_level_hat": test_forecast.value,
            "anchor_mean_log1p": anchor_mean,
            "raw_c_test": raw_c_test,
            "lambda": lam,
            "deploy_c": deploy_c,
            "theoretical_gap_abs_c": equivalent_c,
            "deploy_vs_theoretical_gap_ratio": abs(deploy_c) / equivalent_c,
            "features": test_forecast.features,
            "beta": test_forecast.beta,
            "uncertainty": uncertainty,
        }
        if verdict == "GO":
            submission_sha = create_submission(anchor, deploy_c)

    # Required 28d rolling diagnostic (never used in any gate/choice).
    spacing28 = clean_fixed.iloc[::2].reset_index(drop=True)
    rolling28 = rolling_predictions(spacing28, min_train=4)
    metrics28 = method_metrics(rolling28) if not rolling28.empty else pd.DataFrame()
    write_csv_once(HERE / "spacing_28d_rolling.csv", rolling28)
    if not metrics28.empty:
        write_csv_once(HERE / "spacing_28d_metrics.csv", metrics28)

    results = {
        "verdict": verdict,
        "headroom": headroom,
        "leakage": {
            "selection_interval": [iso(SELECTION_START), iso(DATA_END)],
            "clean_rule": f"target_end <= {CLEAN_TARGET_END}",
            "clean_cutoffs": clean_fixed.cutoff.tolist(),
            "diagnostic_cutoffs_excluded_from_all_fits_and_gates": panel_dataset.loc[~panel_dataset.is_clean, "cutoff"].tolist(),
        },
        "panel": {
            "primary_rows": len(panel_dataset),
            "clean_rows": len(clean_fixed),
            "fixed_cohort_n": 250_000,
            "raw_channels": RAW_CHANNELS,
            "baseline_panel_mean": "not used: no cutoff-safe full-250k historical ORTH_ALPHA proxy exists",
        },
        "rolling_metrics": metrics.to_dict(orient="records"),
        "primary_gates": gates,
        "primary_passing": passing,
        "selected_method": selected or None,
        "spacing_28d_diagnostic": metrics28.to_dict(orient="records") if not metrics28.empty else [],
        "production": prod.to_dict(orient="records") if not prod.empty else [],
        "shrinkage": calibration,
        "test": test_result,
        "clipping": clipping,
        "full_gates": full_gates,
        "expected_robust_Delta_RMSLE": expected_robust,
        "submission": {
            "path": str(SUBMISSION_PATH) if verdict == "GO" else "N/A",
            "SHA256": submission_sha,
            "created": verdict == "GO",
        },
        "artifact_hashes": {
            "SUBMIT_ORTH_ALPHA": sha256_file(ALPHA_PATH),
            "panel_level_dataset": sha256_file(HERE / "panel_level_dataset.parquet"),
        },
    }
    write_json_once(HERE / "results.json", results)

    metric_view = metrics.copy()
    metric_view["improvement_vs_L0_pct"] = 100.0 * (1.0 - metric_view.RMSE_level / l0_rmse)
    metric_md = markdown_table(metric_view,
        ["method", "n_validation", "RMSE_level", "MAE_level", "bias", "last3_RMSE", "sign_accuracy", "improvement_vs_L0_pct"],
        {"n_validation": ".0f", "RMSE_level": ".6f", "MAE_level": ".6f", "bias": "+.6f",
         "last3_RMSE": ".6f", "sign_accuracy": ".3f", "improvement_vs_L0_pct": "+.1f"})
    if prod.empty:
        prod_md = "Production-like stage was not run because neither L1 nor L2 passed the primary level gate."
    else:
        prod_md = markdown_table(prod, ["Fold", "true_c", "forecast_c", "Delta_MSE", "Delta_RMSLE"],
                                 {"true_c": "+.6f", "forecast_c": "+.6f", "Delta_MSE": "+.6f", "Delta_RMSLE": "+.6f"})

    best_method = metrics.sort_values("RMSE_level").iloc[0]
    improvement = 100.0 * (1.0 - best_method.RMSE_level / l0_rmse)
    if test_result.get("target_level_hat") is None:
        test_section = "Primary gate failed; TEST target level and correction were not produced."
    else:
        tci = uncertainty["target_level_hat_95CI"]
        cci = uncertainty["deploy_c_95CI"]
        test_section = (
            f"- selected method: `{selected}`\n"
            f"- `target_level_hat = {test_result['target_level_hat']:.9f}`; 95% block interval "
            f"`[{tci[0]:.6f}, {tci[1]:.6f}]`\n"
            f"- anchor `mean(log1p(predict)) = {anchor_mean:.9f}`\n"
            f"- `raw_c = {test_result['raw_c_test']:+.9f}`\n"
            f"- `lambda = {test_result['lambda']:.9f}`\n"
            f"- `deploy_c = {test_result['deploy_c']:+.9f}`; 95% block interval "
            f"`[{cci[0]:+.6f}, {cci[1]:+.6f}]`\n"
            f"- `|deploy_c| / 0.070451 = {test_result['deploy_vs_theoretical_gap_ratio']:.3f}`; "
            "the forecast was not pulled toward the theoretical gap."
        )
    lambda_text = "N/A" if not calibration else f"{calibration['lambda_final']:.9f}"
    stability_text = "N/A" if sensitivity.empty else (
        f"selected-method raw-c sign stable in "
        f"{int((sensitivity[sensitivity.selected_method].sign == np.sign(test_result['raw_c_test'])).sum())}/"
        f"{len(sensitivity[sensitivity.selected_method])} reported selected-method scenarios"
    )
    output_path = str(SUBMISSION_PATH) if verdict == "GO" else "No submission created."
    bootstrap_text = "; ".join(
        f"{method} RMSE 95% CI [{values['RMSE_95CI'][0]:.6f}, {values['RMSE_95CI'][1]:.6f}]"
        for method, values in boot["methods"].items()
    )
    if metrics28.empty:
        sensitivity_text = "28-day spacing diagnostic was unavailable."
    else:
        sensitivity_text = (
            "The preregistered 28-day spacing diagnostic has only three validation origins: "
            + ", ".join(f"{r.method} RMSE={r.RMSE_level:.6f}, bias={r.bias:+.6f}"
                        for _, r in metrics28.iterrows())
            + ". It is diagnostic only and cannot rescue the failed 14-day primary gate. "
              "Leave-one-clean-cutoff-out TEST-sign checks were not run because the primary "
              "gate forbids producing a TEST correction."
        )
    expected_text = (
        "N/A: the primary level gate stopped the experiment before production-like calibration."
        if expected_robust is None else
        f"Recency-weighted exact production-like `Delta RMSLE` using strictly prefix-calibrated "
        f"shrinkage: `{expected_robust:+.9f}`."
    )
    report = f"""# EXP078 — Forward Global Level Forecast

## Verdict

**{verdict}**

All fixed gates were evaluated without a leaderboard probe or leaderboard-based sign/scale choice.

## Headroom math

- remaining `Delta RMSLE = {headroom['remaining_Delta_RMSLE']:+.12f}`
- remaining `Delta MSE = {headroom_delta_mse:+.12f}` (magnitude `{abs(headroom_delta_mse):.12f}`)
- equivalent optimally removable level error `|c| = sqrt(-Delta MSE) = {equivalent_c:.9f}`

This is scale arithmetic, not evidence that the current submission is wrong by that amount.

## Leakage / cohort audit

The raw table is the fixed final cohort of 250,000 users.  Primary selection and all fitted
coefficients use only rows with `target_end <= 2025-11-15`.  The assertion is executable in
`run_exp078.py`; later labelled rows overlap `2025-11-16..2026-02-13`, are marked diagnostics,
and are excluded from method choice, coefficient fitting, shrinkage, and GO/NO-GO.

The canonical production proxy exists only on the historical 3-block-eligible users.  For the
production-like test, the same scalar method is therefore refit strict-forward on matched
eligible panel states; target/proxy row counts and mean target levels are asserted identical.

## Panel-level dataset

`panel_level_dataset.parquet` contains the 14-day fixed-cohort sequence, scalar target-free
state, and `target_level`.  It contains mean/median/q25/q75/q90 recent GMV for 30/60/90 days,
purchase fractions, mean purchase-day/order/search/event-day counts, recencies, and 30-vs-30 /
60-vs-60 changes.  The 11 real raw channels are `{', '.join(RAW_CHANNELS)}`.  No full-cohort,
cutoff-safe historical ORTH_ALPHA prediction exists, so no synthetic baseline panel mean was
fabricated.  `panel_level_dataset_eligible.parquet` is the matched production diagnostic.

## Rolling validation

Each validation prediction uses labels only when their 30-day target has already ended at the
validation origin.  L0 is freeze, L1 is the fixed 30-day panel step with historical beta, and L2
is Ridge (`alpha={RIDGE_ALPHA}`) with at most {MAX_L2_FEATURES} training-history-pruned predictors.

{metric_md}

Primary gates: `{json.dumps(gates, ensure_ascii=False)}`.  Passing methods: `{passing}`.

Moving-block uncertainty (`block_length=2`, 20,000 draws): {bootstrap_text}.

## Production-like validation

{prod_md}

## Shrinkage

- final historical forward estimate `lambda = {lambda_text}`; hard-clipped to `[0, 0.5]`
- stability: {stability_text}
- forward-prefix lambdas: `{calibration.get('rolling_lambdas', 'N/A')}`

## Sensitivity / falsification

{sensitivity_text}

## TEST forecast

{test_section}

## Expected robust effect

{expected_text}  The strong-result reference is `<= -0.0005`; it is descriptive, not a
post-result gate change.  Clipping diagnostics: `{json.dumps(clipping, ensure_ascii=False)}`.

## Output

- {output_path}
- SHA256: `{submission_sha}`
- anchor SHA256: `{sha256_file(ALPHA_PATH)}`
- no file was uploaded

## Final conclusion

The best clean level RMSE is `{best_method.RMSE_level:.6f}` for `{best_method.method}`, an
improvement of `{improvement:.1f}%` versus freeze (`{l0_rmse:.6f}`).  The preregistered verdict
is **{verdict}**.  No residual/temporal search, EXP075 correction, user-level correction, public
calibration, probe, or automatic submission was used.
"""
    write_text_once(HERE / "REPORT.md", report)
    print(json.dumps(jsonable({
        "verdict": verdict,
        "best_method": str(best_method.method),
        "best_RMSE": float(best_method.RMSE_level),
        "improvement_vs_freeze_pct": improvement,
        "selected": selected or None,
        "test": test_result,
        "expected_robust_Delta_RMSLE": expected_robust,
        "submission_sha": submission_sha,
    }), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
