"""EXP-058 EXACT-ANNIVERSARY-WINDOW.

The executable is intentionally fail-closed.  It writes the preregistration and
novelty/baseline audits before it is allowed to read the primary target.  A
missing production checkpoint therefore stops the run unless an explicitly
supplied, target-blind recipe-parity checkpoint passes the registered replay
audit.

The pure feature/control/cross-fit functions live here as well so that all
registered seams can be unit-tested without reading competition data.

Run from the experiment worktree::

    python src/exact_anniversary.py
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import hashlib
import importlib
import json
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import polars as pl

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import SEED
from src.validation import calibrate, rmsle_z


EXP_ID = "EXACT_ANNIVERSARY_EXP058"
PRIMARY_CUTOFF = dt.date(2026, 1, 14)
PRODUCTION_CUTOFF = dt.date(2026, 2, 13)
TARGET_DAYS = 30
SHIFT_DAYS = 30

# No result-dependent tuning.  This is the same normalized ridge penalty that
# was selected on every LOFO fold in EXP-041, now fixed before target access.
RIDGE_LAMBDA = 1e-3
SHRINK_GRID = (0.25, 0.50, 0.75, 1.00)
CORRECTION_BOUND = 1.50
DEPTH_CLIP = 289
RATIO_PRIOR = 1.0
RATIO_BOUNDS = (0.10, 10.0)
LOG_DIFF_BOUNDS = (-4.0, 4.0)

ANNUAL_BASE = (
    "gmv",
    "orders",
    "buy_days",
    "carts",
    "searches",
    "catalog",
    "active_days",
    "raw_row_days",
    "explicit_zero_days",
    "gmv_search",
    "gmv_catalog",
    "buyer",
)
SCALE_METRICS = ("gmv", "orders", "buy_days", "active_days")
ANNUAL_SOURCE_COLUMNS = tuple(f"old_{name}" for name in ANNUAL_BASE) + tuple(
    f"old_pre{days}_{name}" for days in (30, 60) for name in SCALE_METRICS
)

BASE_COMPONENTS = {
    "S1-CAP": 0.10,
    "S1-UNC": 0.20,
    "S1-DIST": 0.25,
    "ETX-01-S42-DCW": 0.075,
    "ETX-01-S43-DCW": 0.075,
    "ETX-01-S44-DCW": 0.075,
    "SEQ-01": 0.075,
    "SEQ-C289-S43": 0.075,
    "SEQ-C289-S44": 0.075,
}


@dataclass(frozen=True)
class DateWindow:
    start: dt.date
    end: dt.date

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1


@dataclass(frozen=True)
class CrossFitArm:
    name: str
    z: np.ndarray
    correction: np.ndarray
    group: np.ndarray
    selected_shrink: Mapping[int, float]


def shift_calendar_year(day: dt.date, years: int = -1) -> dt.date:
    """Calendar-year shift; 29 February maps to 28 February when needed."""
    try:
        return day.replace(year=day.year + years)
    except ValueError:
        if day.month == 2 and day.day == 29:
            return day.replace(year=day.year + years, day=28)
        raise


def target_window(cutoff: dt.date) -> DateWindow:
    return DateWindow(cutoff + dt.timedelta(days=1),
                      cutoff + dt.timedelta(days=TARGET_DAYS))


def anniversary_window(cutoff: dt.date) -> DateWindow:
    target = target_window(cutoff)
    return DateWindow(shift_calendar_year(target.start),
                      shift_calendar_year(target.end))


def shifted_year_window(cutoff: dt.date) -> DateWindow:
    old = anniversary_window(cutoff)
    delta = dt.timedelta(days=SHIFT_DAYS)
    return DateWindow(old.start + delta, old.end + delta)


def recent_window(cutoff: dt.date, days: int) -> DateWindow:
    return DateWindow(cutoff - dt.timedelta(days=days - 1), cutoff)


def pre_window(window: DateWindow, days: int) -> DateWindow:
    return DateWindow(window.start - dt.timedelta(days=days),
                      window.start - dt.timedelta(days=1))


def splitmix64(user_ids: np.ndarray) -> np.ndarray:
    """The repository's deterministic user hash, returned before bit masking."""
    x = np.asarray(user_ids, dtype=np.uint64).copy()
    with np.errstate(over="ignore"):
        x += np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x ^= x >> np.uint64(31)
    return x


def user_group(user_ids: np.ndarray) -> np.ndarray:
    return (splitmix64(user_ids) & np.uint64(1)).astype(np.int8)


def inner_group(user_ids: np.ndarray) -> np.ndarray:
    return ((splitmix64(user_ids) >> np.uint64(1)) & np.uint64(1)).astype(np.int8)


def _required_raw_columns() -> tuple[str, ...]:
    return ("user_id", "event_date", "searches", "cat", "to_cart", "to_ord",
            "gmv", "gmv_search", "gmv_cat")


def _check_raw_schema(raw: pl.DataFrame) -> None:
    missing = sorted(set(_required_raw_columns()) - set(raw.columns))
    if missing:
        raise AssertionError(f"missing raw columns: {missing}")


def aggregate_window(raw: pl.DataFrame, users: pl.DataFrame, window: DateWindow,
                     cutoff: dt.date, prefix: str) -> pl.DataFrame:
    """Aggregate one inclusive window, fail-closed against post-cutoff access."""
    _check_raw_schema(raw)
    if window.end > cutoff:
        raise AssertionError(f"feature window {window} crosses cutoff {cutoff}")
    observed = raw.filter((pl.col("event_date") >= window.start)
                          & (pl.col("event_date") <= window.end))
    if observed.height and observed["event_date"].max() > cutoff:
        raise AssertionError("post-cutoff row reached feature aggregation")
    active = ((pl.col("searches") > 0) | (pl.col("cat") > 0)
              | (pl.col("to_cart") > 0) | (pl.col("to_ord") > 0)
              | (pl.col("gmv") > 0))
    zero = ((pl.col("searches") == 0) & (pl.col("cat") == 0)
            & (pl.col("to_cart") == 0) & (pl.col("to_ord") == 0)
            & (pl.col("gmv") == 0))
    agg = observed.group_by("user_id").agg([
        pl.col("gmv").sum().alias(f"{prefix}_gmv"),
        pl.col("to_ord").sum().alias(f"{prefix}_orders"),
        pl.col("event_date").filter(pl.col("gmv") > 0).n_unique()
        .alias(f"{prefix}_buy_days"),
        pl.col("to_cart").sum().alias(f"{prefix}_carts"),
        pl.col("searches").sum().alias(f"{prefix}_searches"),
        pl.col("cat").sum().alias(f"{prefix}_catalog"),
        pl.col("event_date").filter(active).n_unique().alias(f"{prefix}_active_days"),
        pl.col("event_date").n_unique().alias(f"{prefix}_raw_row_days"),
        pl.col("event_date").filter(zero).n_unique()
        .alias(f"{prefix}_explicit_zero_days"),
        pl.col("gmv_search").sum().alias(f"{prefix}_gmv_search"),
        pl.col("gmv_cat").sum().alias(f"{prefix}_gmv_catalog"),
        (pl.col("gmv") > 0).any().cast(pl.Int8).alias(f"{prefix}_buyer"),
    ])
    out = users.select("user_id").join(agg, on="user_id", how="left").sort("user_id")
    value_cols = [name for name in out.columns if name != "user_id"]
    return out.with_columns([pl.col(name).fill_null(0) for name in value_cols])


def purchase_recency(raw: pl.DataFrame, users: pl.DataFrame, cutoff: dt.date) -> pl.DataFrame:
    observed = raw.filter((pl.col("event_date") <= cutoff) & (pl.col("gmv") > 0))
    rec = observed.group_by("user_id").agg(
        (pl.lit(cutoff) - pl.col("event_date").max()).dt.total_days().alias("recent_rec_buy")
    )
    return users.select("user_id").join(rec, on="user_id", how="left").sort("user_id")


def build_features(cutoff: dt.date, raw: pl.DataFrame, users: pl.DataFrame,
                   shifted: bool = False) -> pl.DataFrame:
    """The experiment's sole feature entry point, keyed by cutoff date.

    It returns an opt-in sidecar of new columns and never rewrites the project's
    existing feature cache.
    """
    old = shifted_year_window(cutoff) if shifted else anniversary_window(cutoff)
    parts = [aggregate_window(raw, users, old, cutoff, "old")]
    for days in (30, 60):
        parts.append(aggregate_window(raw, users, recent_window(cutoff, days), cutoff,
                                      f"current{days}"))
        parts.append(aggregate_window(raw, users, pre_window(old, days), cutoff,
                                      f"old_pre{days}"))
    parts.append(purchase_recency(raw, users, cutoff))
    out = parts[0]
    for part in parts[1:]:
        out = out.join(part, on="user_id", how="left")
    if out.height != users.height or out["user_id"].n_unique() != users.height:
        raise AssertionError("source feature alignment failed")
    return out.sort("user_id")


def feature_matrix(sources: pl.DataFrame) -> pl.DataFrame:
    """Registered 29-column model matrix, identical for REAL/SHUFFLED/SHIFTED."""
    anniv_expr = []
    for name in ANNUAL_BASE:
        if name == "gmv":
            anniv_expr += [pl.col("old_gmv").cast(pl.Float64).alias("anniv_gmv"),
                           pl.col("old_gmv").cast(pl.Float64).log1p().alias("anniv_lgmv")]
        else:
            anniv_expr.append(pl.col(f"old_{name}").cast(pl.Float64).alias(f"anniv_{name}"))
    derived = []
    for days in (30, 60):
        for name in SCALE_METRICS:
            current = pl.col(f"current{days}_{name}").cast(pl.Float64)
            old = pl.col(f"old_pre{days}_{name}").cast(pl.Float64)
            ratio = ((current + RATIO_PRIOR) / (old + RATIO_PRIOR)).clip(*RATIO_BOUNDS)
            diff = (current.log1p() - old.log1p()).clip(*LOG_DIFF_BOUNDS)
            derived += [ratio.alias(f"scale_w{days}_{name}_ratio"),
                        diff.alias(f"scale_w{days}_{name}_diff")]
    out = sources.select([pl.col("user_id")] + anniv_expr + derived)
    expected = 1 + 13 + 16
    if out.width != expected:
        raise AssertionError(f"registered feature width changed: {out.width - 1}")
    return out


def _rank_decile(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.asarray(values), kind="stable")
    ranks = np.empty(len(order), dtype=np.int64)
    ranks[order] = np.arange(len(order), dtype=np.int64)
    return np.minimum(9, ranks * 10 // max(len(order), 1)).astype(np.int8)


def shuffle_strata(sources: pl.DataFrame) -> np.ndarray:
    gmv_decile = _rank_decile(sources["current30_gmv"].to_numpy())
    orders = sources["current30_orders"].to_numpy()
    order_bucket = np.select([orders == 0, orders == 1, orders <= 3, orders <= 7],
                             [0, 1, 2, 3], default=4).astype(np.int8)
    rec = sources["recent_rec_buy"].to_numpy()
    rec_bucket = np.select([np.isfinite(rec) & (rec <= 14),
                            np.isfinite(rec) & (rec <= 60)],
                           [0, 1], default=2).astype(np.int8)
    return (gmv_decile.astype(np.int16) * 15
            + order_bucket.astype(np.int16) * 3 + rec_bucket.astype(np.int16))


def shuffle_anniversary(sources: pl.DataFrame, cutoff: dt.date,
                        seed: int = SEED) -> tuple[pl.DataFrame, dict]:
    """Jointly permute all annual-side sources within registered recent strata."""
    strata = shuffle_strata(sources)
    perm = np.arange(sources.height)
    rng = np.random.default_rng(np.random.SeedSequence([seed, cutoff.toordinal()]))
    for value in np.unique(strata):
        idx = np.flatnonzero(strata == value)
        if len(idx) > 1:
            perm[idx] = rng.permutation(idx)
    out = sources.clone()
    for name in ANNUAL_SOURCE_COLUMNS:
        out = out.with_columns(pl.Series(name, sources[name].to_numpy()[perm]))
    audit = {
        "n": sources.height,
        "strata": int(len(np.unique(strata))),
        "moved_share": float(np.mean(perm != np.arange(len(perm)))),
        "strata_preserved": bool(np.array_equal(strata, shuffle_strata(out))),
        "joint_permutation_sha256": hashlib.sha256(perm.tobytes()).hexdigest(),
    }
    return out, audit


def matrix_np(features: pl.DataFrame) -> np.ndarray:
    cols = [name for name in features.columns if name != "user_id"]
    matrix = features.select(cols).to_numpy().astype(np.float64, copy=False)
    if not np.isfinite(matrix).all():
        raise AssertionError("non-finite registered feature")
    return matrix


def _fit_ridge(X: np.ndarray, y: np.ndarray):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X)
    model = Ridge(alpha=len(X) * RIDGE_LAMBDA, fit_intercept=True,
                  solver="cholesky").fit(scaler.transform(X), y)
    return scaler, model


def _centered_clipped_prediction(scaler, model, X_donor: np.ndarray,
                                 X_recipient: np.ndarray) -> tuple[np.ndarray, float]:
    donor = model.predict(scaler.transform(X_donor))
    center = float(donor.mean())
    recipient = model.predict(scaler.transform(X_recipient)) - center
    return np.clip(recipient, -CORRECTION_BOUND, CORRECTION_BOUND), center


def crossfit_arm(name: str, features: pl.DataFrame, user_ids: np.ndarray,
                 y: np.ndarray, z_base: np.ndarray) -> CrossFitArm:
    """Two-sided A->B/B->A fit; shrink is selected only on the donor inner split."""
    X = matrix_np(features)
    uid = np.asarray(user_ids)
    y = np.asarray(y, dtype=float)
    z_base = np.asarray(z_base, dtype=float)
    if not (len(X) == len(uid) == len(y) == len(z_base)):
        raise AssertionError("cross-fit arrays are not aligned")
    groups, inner = user_group(uid), inner_group(uid)
    residual = np.log1p(y) - z_base
    correction = np.empty(len(uid), dtype=np.float64)
    selected: dict[int, float] = {}
    for recipient in (0, 1):
        donor = 1 - recipient
        donor_train = (groups == donor) & (inner == 0)
        donor_valid = (groups == donor) & (inner == 1)
        recipient_mask = groups == recipient
        if np.any(donor_train & recipient_mask) or np.any(donor_valid & recipient_mask):
            raise AssertionError("cross-fit isolation failed")
        scaler_i, model_i = _fit_ridge(X[donor_train], residual[donor_train])
        corr_i, _ = _centered_clipped_prediction(
            scaler_i, model_i, X[donor_train], X[donor_valid])
        scores = []
        for shrink in SHRINK_GRID:
            scores.append(calibrate(y[donor_valid],
                                    z_base[donor_valid] + shrink * corr_i)[1])
        shrink = float(SHRINK_GRID[int(np.argmin(scores))])
        selected[recipient] = shrink

        donor_all = groups == donor
        scaler, model = _fit_ridge(X[donor_all], residual[donor_all])
        corr, _ = _centered_clipped_prediction(scaler, model, X[donor_all], X[recipient_mask])
        correction[recipient_mask] = corr
    return CrossFitArm(name=name, z=z_base + correction * np.where(groups == 0,
                        selected[0], selected[1]), correction=correction,
                       group=groups, selected_shrink=selected)


def arm_metrics(arm: CrossFitArm, y: np.ndarray, z_base: np.ndarray,
                sources_real: pl.DataFrame) -> list[dict]:
    """Primary metrics per recipient half; cohort masks are target-independent."""
    y = np.asarray(y, float)
    z_base = np.asarray(z_base, float)
    residual = np.log1p(y) - z_base
    annual = sources_real["old_buyer"].to_numpy() > 0
    recent = sources_real["current30_buy_days"].to_numpy() > 0
    rec = sources_real["recent_rec_buy"].to_numpy()
    q1, q2 = np.quantile(z_base, [1 / 3, 2 / 3])
    cohort_masks = {
        "all": np.ones(len(y), bool),
        "target_zero": y == 0,
        "target_positive": y > 0,
        "annual_buyer": annual,
        "annual_nonbuyer": ~annual,
        "recent_buyer": recent,
        "recent_nonbuyer": ~recent,
        "rec_buy_15_60": np.isfinite(rec) & (rec >= 15) & (rec <= 60),
        "base_low": z_base <= q1,
        "base_mid": (z_base > q1) & (z_base <= q2),
        "base_high": z_base > q2,
    }
    rows = []
    for group in (0, 1):
        half = arm.group == group
        base_offset, base_score = calibrate(y[half], z_base[half])
        arm_offset, arm_score = calibrate(y[half], arm.z[half])
        corr = arm.correction[half]
        rows.append({
            "arm": arm.name, "group": "A" if group == 0 else "B",
            "cohort": "all", "n": int(half.sum()), "base": base_score,
            "score": arm_score, "delta": arm_score - base_score,
            "base_offset": base_offset, "offset": arm_offset,
            "correction_mean": float(corr.mean()), "correction_var": float(corr.var()),
            "correction_residual_corr": float(np.corrcoef(corr, residual[half])[0, 1]),
            "shrink": arm.selected_shrink[group],
        })
        for cohort, mask in cohort_masks.items():
            if cohort == "all":
                continue
            selected = half & mask
            if not selected.any():
                continue
            b = rmsle_z(y[selected], z_base[selected] + base_offset)
            s = rmsle_z(y[selected], arm.z[selected] + arm_offset)
            rows.append({"arm": arm.name, "group": "A" if group == 0 else "B",
                         "cohort": cohort, "n": int(selected.sum()), "base": b,
                         "score": s, "delta": s - b})
    return rows


def align_submission(sample: pl.DataFrame, predictions: pl.DataFrame) -> pl.DataFrame:
    """Strict sample order/composition alignment used only after all gates pass."""
    required = {"user_id", "predict"}
    if set(predictions.columns) != required:
        raise AssertionError(f"submission columns must be {sorted(required)}")
    if predictions["user_id"].n_unique() != predictions.height:
        raise AssertionError("duplicate prediction user_id")
    order = sample.select("user_id").with_row_index("_order")
    out = order.join(predictions, on="user_id", how="left").sort("_order").drop("_order")
    if out.height != sample.height or out["predict"].null_count():
        raise AssertionError("submission user_id composition mismatch")
    values = out["predict"].to_numpy()
    if not np.isfinite(values).all() or np.any(values < 0):
        raise AssertionError("invalid submission predictions")
    return out


def _asset_exists(roots: Iterable[Path], filename: str) -> tuple[bool, str | None]:
    for root in roots:
        direct = root / "artifacts" / filename
        if direct.is_file():
            return True, str(direct)
        archives = root / "weights_archives"
        if archives.is_dir():
            for archive in sorted(archives.glob("*.zip")):
                with zipfile.ZipFile(archive) as handle:
                    if filename in handle.namelist():
                        return True, f"{archive}!{filename}"
    return False, None


def baseline_preflight(source_root: Path, checkpoint_roots: Iterable[Path] = ()) -> dict:
    """Prove that exact arbitrary-cutoff STRONGEST inference is reproducible."""
    required_checkpoints = {
        "ETX seed42": "model_ETX-01-S42-TEST.pt",
        "ETX seed43": "model_ETX-01-S43-TEST.pt",
        "ETX seed44": "model_ETX-01-S44-TEST.pt",
        "TCN seed42": "model_SEQ-01-TEST.pt",
        "TCN seed43": "model_SEQ-C289-S43-TEST.pt",
        "TCN seed44": "model_SEQ-C289-S44-TEST.pt",
    }
    roots = [Path(p) for p in checkpoint_roots] + [source_root]
    rows = []
    for component, filename in required_checkpoints.items():
        found, location = _asset_exists(roots, filename)
        rows.append({"component": component, "filename": filename,
                     "found": found, "location": location})
    support = {
        "raw": (source_root / "data/raw/train.parquet").is_file(),
        "sample": (source_root / "data/raw/sample_submit.csv").is_file(),
        "strongest_submission": (source_root / "submissions/submission_STRONGEST_CURRENT.csv").is_file(),
        "feature_code": (source_root / "src/features.py").is_file(),
        "sequence_code": (source_root / "src/seq.py").is_file(),
        "etx_code": (source_root / "src/etx.py").is_file(),
    }
    missing_checkpoints = [row["filename"] for row in rows if not row["found"]]
    missing_support = [name for name, found in support.items() if not found]
    status = "PASS" if not missing_checkpoints and not missing_support else "BLOCKED_BASELINE_PARITY"
    return {
        "status": status,
        "reason": (None if status == "PASS" else
                   "Exact arbitrary-cutoff STRONGEST_CURRENT cannot be replayed without every "
                   "production neural checkpoint."),
        "checkpoints": rows,
        "support": support,
        "missing_checkpoints": missing_checkpoints,
        "missing_support": missing_support,
        "forbidden_fallbacks": [
            "replace seed-42 by seed-43/44 average",
            "reuse 2026-02-13 prediction at 2026-01-14 (post-cutoff leakage)",
            "change STRONGEST_CURRENT weights or baseline family",
        ],
    }


def preregistration() -> dict:
    return {
        "experiment": EXP_ID,
        "primary_cutoff": str(PRIMARY_CUTOFF),
        "production_cutoff": str(PRODUCTION_CUTOFF),
        "ordinary_temporal_cv": "unavailable",
        "validation": "PSEUDO-PRODUCTION CROSS-FIT",
        "transfer_risk": "one-month calendar extrapolation",
        "year_shift": "calendar-year; Feb-29 clamps to Feb-28",
        "shifted_control_days": SHIFT_DAYS,
        "features": {"annual": list(ANNUAL_BASE), "scale_metrics": list(SCALE_METRICS),
                     "scale_horizons": [30, 60], "n_model_features": 29},
        "ratio_recipe": {"prior": RATIO_PRIOR, "clip": list(RATIO_BOUNDS),
                         "log_diff_clip": list(LOG_DIFF_BOUNDS)},
        "ridge": {"normalized_lambda": RIDGE_LAMBDA,
                  "sklearn_alpha": "n_donor * normalized_lambda",
                  "alpha_sweep": False},
        "correction": {"center": "donor prediction mean", "clip": [-CORRECTION_BOUND,
                         CORRECTION_BOUND], "shrink_grid_donor_inner_only": list(SHRINK_GRID)},
        "base_components": BASE_COMPONENTS,
        "strong_pass": {"real_base_max": -0.0010, "halves_correct_sign": True,
                        "real_shuffled_max": -0.0007, "real_shifted_max": -0.0005,
                        "annual_buyer_share_min": 0.10,
                        "annual_buyer_halves_correct_sign": True},
        "borderline": [-0.0010, -0.0005],
        "recipe_parity": {"canonical_cutoff": str(PRODUCTION_CUTOFF),
                          "corr_min": 0.996, "var_delta_max": 0.018,
                          "abs_mean_delta_max": 0.02,
                          "note": ("target-blind replay audit for an explicitly authorized retrain; "
                                   "limits are no looser than the observed production S42/S43/S44 "
                                   "pairwise seed envelope")},
        "production_support": {"max_abs_smd": 1.0,
                               "correction_variance_ratio": [0.5, 2.0],
                               "annual_buyer_share_abs_diff_max": 0.10,
                               "clipping_share_max": 0.05},
        "seed_source": "src.config.SEED",
    }


def novelty_audit() -> dict:
    return {
        "exact_user_target_window_anniversary_previously_tested": False,
        "matched_control_previously_tested": False,
        "closest_prior": "exp_023 HOLIDAY-YOY",
        "exp_023_simultaneous_changes": [
            "holiday response (within-year before/after contrast), not target-window GMV lookup",
            "2025->2026 response slope minus a separate placebo slope",
            "six behavior groups averaged with reliability shrinkage",
            "zero-mean post-processing over S1-DIST-MIX, not STRONGEST_CURRENT",
            "holiday-history eligibility and one-year New-Year-to-Feb/Mar transfer",
        ],
        "why_not_equivalent": {
            "180d_365d_all_history": "aggregate different dates and cannot identify exact calendar alignment",
            "holiday_correction": "models response around a holiday proxy, not the same user's exact target dates",
            "long_history_capacity": "adds old-history capacity without the SHIFTED architecture-matched control",
        },
        "decision": "NOVEL; continue to baseline-parity preflight",
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _checkpoint_path(parity: Mapping, filename: str) -> Path:
    row = next(r for r in parity["checkpoints"] if r["filename"] == filename)
    location = row.get("location")
    if not location or "!" in location:
        raise AssertionError(f"checkpoint must be a directly readable file: {filename}")
    return Path(location)


def _load_torch_state(path: Path, build_model):
    import torch

    saved = torch.load(path, map_location="cpu", weights_only=False)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(saved["cfg"]).to(dev)
    model.load_state_dict(saved["state"])
    model.eval()
    return model, saved["cfg"], dev


def _load_external_etx(source_root: Path):
    """Import the production ETX implementation without copying it into this worktree."""
    import src

    external = str(source_root / "src")
    if external not in src.__path__:
        src.__path__.append(external)
    return importlib.import_module("src.etx")


def _neural_components(cutoff: dt.date, user_ids: np.ndarray, source_root: Path,
                       parity: Mapping) -> dict[str, np.ndarray]:
    from src import seq

    rows = seq.user_rows(user_ids)
    out: dict[str, np.ndarray] = {}
    seq_specs = {
        "SEQ-01": "model_SEQ-01-TEST.pt",
        "SEQ-C289-S43": "model_SEQ-C289-S43-TEST.pt",
        "SEQ-C289-S44": "model_SEQ-C289-S44-TEST.pt",
    }
    for name, filename in seq_specs.items():
        model, cfg, dev = _load_torch_state(_checkpoint_path(parity, filename), seq.build_model)
        out[name] = np.maximum(seq.predict(model, cutoff, rows, cfg, dev,
                                           depth_clip=DEPTH_CLIP), 0.0)
        del model
        gc.collect()
        if dev.type == "cuda":
            import torch
            torch.cuda.empty_cache()

    etx = _load_external_etx(source_root)
    etx_specs = {
        "ETX-01-S42-DCW": "model_ETX-01-S42-TEST.pt",
        "ETX-01-S43-DCW": "model_ETX-01-S43-TEST.pt",
        "ETX-01-S44-DCW": "model_ETX-01-S44-TEST.pt",
    }
    # All production ETX checkpoints were trained only on Thursday cutoffs.
    dow_shift = float(3 - cutoff.weekday())
    for name, filename in etx_specs.items():
        model, cfg, dev = _load_torch_state(_checkpoint_path(parity, filename), etx.build_model)
        tokenizer = etx.Tokenizer(dev)
        tokenizer.depth_cap = DEPTH_CLIP
        tokenizer.cdow_shift = dow_shift
        out[name] = np.maximum(etx.predict(model, tokenizer, cutoff, rows, cfg, dev,
                                           depth_clip=DEPTH_CLIP), 0.0)
        del model, tokenizer
        gc.collect()
        if dev.type == "cuda":
            import torch
            torch.cuda.empty_cache()
    return out


def _tabular_component(cutoff: dt.date, users: pl.DataFrame, L: int | None,
                       norm_long: bool, model_name: str, rounds: int) -> np.ndarray:
    """Train the registered full-corridor tabular member and infer fixed users."""
    from src.features import build_features as project_build_features
    from src.features import feature_names, to_np
    from src.predict import train_full
    from src.train import Setup, infer, select_features

    setup = Setup(L=L, min_history=90, step=7, train_blocks=1, model=model_name,
                  rounds=rounds, norm_long=norm_long)
    frame_all = project_build_features(cutoff, setup.L, setup.norm_long)
    feats = select_features(feature_names(frame_all), setup.drop_groups, None)
    missing = users.join(frame_all.select("user_id"), on="user_id", how="anti")
    if missing.height:
        raise AssertionError(f"{missing.height} primary users lack tabular history")
    frame = users.join(frame_all, on="user_id", how="left").sort("user_id")
    matrix = to_np(frame, feats)
    setup.params = dict(setup.params, seed=SEED)
    trained = train_full(setup, feats, [model_name])
    setup_i, model = trained[model_name]
    z = np.maximum(infer(setup_i, model, matrix), 0.0)
    del model, matrix, frame, frame_all, trained
    gc.collect()
    return z


def strongest_at(cutoff: dt.date, users: pl.DataFrame, source_root: Path,
                 parity: Mapping, output_root: Path) -> tuple[np.ndarray, dict]:
    """Exact production recipe on an arbitrary cutoff and fixed user panel."""
    user_ids = users["user_id"].to_numpy()
    components: dict[str, np.ndarray] = {}

    def cached(name: str, make) -> np.ndarray:
        path = output_root / f"primary_z_{name}.npy"
        if path.is_file():
            values = np.load(path)
            if values.shape == (len(user_ids),):
                return values
        values = np.asarray(make(), dtype=np.float64)
        if values.shape != (len(user_ids),) or not np.isfinite(values).all():
            raise AssertionError(f"invalid {name} primary prediction")
        np.save(path, values)
        return values

    components["S1-CAP"] = cached(
        "S1-CAP", lambda: _tabular_component(cutoff, users, 180, False, "direct", 600))
    components["S1-UNC"] = cached(
        "S1-UNC", lambda: _tabular_component(cutoff, users, None, False, "direct", 600))
    components["S1-DIST"] = cached(
        "S1-DIST", lambda: _tabular_component(cutoff, users, None, True, "dist", 250))
    missing_neural = [name for name in BASE_COMPONENTS
                      if name not in components
                      and not (output_root / f"primary_z_{name}.npy").is_file()]
    generated_neural = (_neural_components(cutoff, user_ids, source_root, parity)
                        if missing_neural else {})
    for name in BASE_COMPONENTS:
        if name in components:
            continue
        components[name] = cached(name, lambda name=name: generated_neural[name])
    if set(components) != set(BASE_COMPONENTS):
        raise AssertionError("STRONGEST component identity changed")
    z = sum(BASE_COMPONENTS[name] * components[name] for name in BASE_COMPONENTS)
    np.save(output_root / "primary_user_id.npy", user_ids)
    np.save(output_root / "primary_z_strongest.npy", z)
    audit = {
        "cutoff": str(cutoff), "n": len(z), "weights_sum": sum(BASE_COMPONENTS.values()),
        "mean_z": float(z.mean()), "components": {
            name: {"mean": float(values.mean()), "var": float(values.var())}
            for name, values in components.items()
        },
    }
    return z, audit


def recipe_parity_audit(source_root: Path, worktree: Path, parity: Mapping) -> dict:
    """Target-blind replay check for the explicitly authorized seed-42 retrain."""
    new_path = worktree / "artifacts" / "ztest_SEQ-01.npy"
    ref_path = source_root / "artifacts" / "ztest_SEQ-01.npy"
    if not new_path.is_file() or not ref_path.is_file():
        return {"status": "BLOCKED_BASELINE_PARITY", "reason": "missing target-blind replay arrays"}
    new, ref = np.load(new_path), np.load(ref_path)
    if new.shape != ref.shape:
        return {"status": "BLOCKED_BASELINE_PARITY", "reason": "replay shape mismatch"}
    delta = new.astype(float) - ref.astype(float)
    metrics = {"corr": float(np.corrcoef(new, ref)[0, 1]),
               "var_delta": float(delta.var()), "mean_delta": float(delta.mean()),
               "max_abs_delta": float(np.max(np.abs(delta)))}
    limits = preregistration()["recipe_parity"]
    passed = (metrics["corr"] >= limits["corr_min"]
              and metrics["var_delta"] <= limits["var_delta_max"]
              and abs(metrics["mean_delta"]) <= limits["abs_mean_delta_max"])
    return {"status": "PASS_RETRAINED_RECIPE_PARITY" if passed else "BLOCKED_BASELINE_PARITY",
            "canonical_reference": str(ref_path), "retrained_prediction": str(new_path),
            "metrics": metrics, "limits": limits,
            "bitwise_equal": bool(np.array_equal(new, ref))}


def _target_labels(raw: pl.DataFrame, users: pl.DataFrame, cutoff: dt.date) -> np.ndarray:
    window = target_window(cutoff)
    labels = (raw.filter((pl.col("event_date") >= window.start)
                         & (pl.col("event_date") <= window.end))
              .group_by("user_id").agg(pl.col("gmv").sum().alias("y")))
    aligned = users.join(labels, on="user_id", how="left").with_columns(
        pl.col("y").fill_null(0.0)).sort("user_id")
    return aligned["y"].to_numpy()


def feature_window_support(raw: pl.DataFrame, cutoff: dt.date) -> dict:
    first, last = raw["event_date"].min(), raw["event_date"].max()

    def row(name: str, window: DateWindow) -> dict:
        observed_start, observed_end = max(window.start, first), min(window.end, last, cutoff)
        observed = max(0, (observed_end - observed_start).days + 1)
        return {"name": name, "start": str(window.start), "end": str(window.end),
                "expected_days": window.days, "observable_calendar_days": observed,
                "complete": observed == window.days}

    rows = []
    for shifted, tag in ((False, "REAL"), (True, "SHIFTED")):
        old = shifted_year_window(cutoff) if shifted else anniversary_window(cutoff)
        rows.append(row(f"{tag}_anniversary", old))
        for days in (30, 60):
            rows.append(row(f"{tag}_pre{days}", pre_window(old, days)))
    return {"raw_start": str(first), "raw_end": str(last), "cutoff": str(cutoff),
            "windows": rows,
            "note": ("Exact anniversary windows are fully observed. Pre-window scale "
                     "denominators are intentionally raw observable sums; the primary REAL "
                     "pre-windows are left-truncated by the 2025-01-01 data boundary, so "
                     "scale features are interpreted conservatively and are not normalized.")}


def _all_metrics(arms: Iterable[CrossFitArm], y: np.ndarray, z_base: np.ndarray,
                 sources: pl.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for arm in arms:
        rows.extend(arm_metrics(arm, y, z_base, sources))
    return rows


def verdict(real: CrossFitArm, shuffled: CrossFitArm, shifted: CrossFitArm,
            y: np.ndarray, sources: pl.DataFrame) -> dict:
    rows = {}
    for arm in (real, shuffled, shifted):
        for group in (0, 1):
            mask = arm.group == group
            rows[(arm.name, group)] = calibrate(y[mask], arm.z[mask])[1]
    base_scores = {group: calibrate(y[real.group == group],
                                   real.z[real.group == group]
                                   - real.correction[real.group == group]
                                   * real.selected_shrink[group])[1]
                   for group in (0, 1)}
    real_base = [rows[(real.name, g)] - base_scores[g] for g in (0, 1)]
    real_shuf = [rows[(real.name, g)] - rows[(shuffled.name, g)] for g in (0, 1)]
    real_shift = [rows[(real.name, g)] - rows[(shifted.name, g)] for g in (0, 1)]
    alignment = []
    annual_delta = []
    annual = sources["old_buyer"].to_numpy() > 0
    for group in (0, 1):
        mask = real.group == group
        residual = np.log1p(y[mask]) - (real.z[mask]
                   - real.correction[mask] * real.selected_shrink[group])
        alignment.append(float(np.corrcoef(real.correction[mask], residual)[0, 1]))
        cohort = mask & annual
        base_offset = calibrate(y[mask], real.z[mask]
                      - real.correction[mask] * real.selected_shrink[group])[0]
        real_offset = calibrate(y[mask], real.z[mask])[0]
        base_cohort = (real.z[cohort]
                       - real.correction[cohort] * real.selected_shrink[group])
        annual_delta.append(rmsle_z(y[cohort], real.z[cohort] + real_offset)
                            - rmsle_z(y[cohort], base_cohort + base_offset))
    mean_gain = float(np.mean(real_base))
    annual_share = float(annual.mean())
    strong = (mean_gain <= -0.0010 and all(v < 0 for v in real_base)
              and float(np.mean(real_shuf)) <= -0.0007
              and float(np.mean(real_shift)) <= -0.0005
              and all(v > 0 for v in alignment)
              and annual_share >= 0.10 and all(v < 0 for v in annual_delta))
    borderline = (-0.0010 < mean_gain <= -0.0005
                  and all(v < 0 for v in real_base)
                  and float(np.mean(real_shuf)) < 0
                  and float(np.mean(real_shift)) < 0
                  and all(v > 0 for v in alignment))
    label = "STRONG_PASS" if strong else ("BORDERLINE" if borderline else "REJECT")
    return {"verdict": label, "real_minus_base_halves": real_base,
            "real_minus_base_mean": mean_gain, "real_minus_shuffled_halves": real_shuf,
            "real_minus_shuffled_mean": float(np.mean(real_shuf)),
            "real_minus_shifted_halves": real_shift,
            "real_minus_shifted_mean": float(np.mean(real_shift)),
            "correction_residual_alignment": alignment,
            "annual_buyer_share": annual_share,
            "annual_buyer_delta_halves": annual_delta}


def _fit_full_correction(features: pl.DataFrame, y: np.ndarray,
                         z_base: np.ndarray, shrink: float):
    X = matrix_np(features)
    residual = np.log1p(y) - z_base
    scaler, model = _fit_ridge(X, residual)
    donor_prediction = model.predict(scaler.transform(X))
    center = float(donor_prediction.mean())
    return scaler, model, center, shrink


def _support_audit(primary_sources: pl.DataFrame, production_sources: pl.DataFrame,
                   primary_features: pl.DataFrame, production_features: pl.DataFrame,
                   correction_primary: np.ndarray, correction_production: np.ndarray) -> dict:
    X0, X1 = matrix_np(primary_features), matrix_np(production_features)
    pooled = np.sqrt((X0.var(0) + X1.var(0)) / 2.0)
    smd = np.divide(X1.mean(0) - X0.mean(0), pooled,
                    out=np.zeros_like(pooled), where=pooled > 1e-12)
    var_ratio = float(correction_production.var() / max(correction_primary.var(), 1e-12))
    buyer0 = float((primary_sources["old_buyer"].to_numpy() > 0).mean())
    buyer1 = float((production_sources["old_buyer"].to_numpy() > 0).mean())
    clip_share = float(np.mean(np.abs(correction_production) >= CORRECTION_BOUND - 1e-12))
    limits = preregistration()["production_support"]
    passed = (float(np.max(np.abs(smd))) <= limits["max_abs_smd"]
              and limits["correction_variance_ratio"][0] <= var_ratio
              <= limits["correction_variance_ratio"][1]
              and abs(buyer1 - buyer0) <= limits["annual_buyer_share_abs_diff_max"]
              and clip_share <= limits["clipping_share_max"])
    return {"status": "PASS" if passed else "OUT_OF_SUPPORT",
            "max_abs_smd": float(np.max(np.abs(smd))),
            "correction_variance_ratio": var_ratio,
            "annual_buyer_share_primary": buyer0,
            "annual_buyer_share_production": buyer1,
            "annual_buyer_share_abs_diff": abs(buyer1 - buyer0),
            "clipping_share": clip_share,
            "anniv_gmv_quantiles_primary": np.quantile(primary_sources["old_gmv"].to_numpy(),
                                                       [0.5, 0.9, 0.99]).tolist(),
            "anniv_gmv_quantiles_production": np.quantile(production_sources["old_gmv"].to_numpy(),
                                                          [0.5, 0.9, 0.99]).tolist(),
            "limits": limits}


def _production_candidate(source_root: Path, worktree: Path, raw: pl.DataFrame,
                          primary_users: pl.DataFrame, primary_sources: pl.DataFrame,
                          primary_features: pl.DataFrame, real: CrossFitArm,
                          y: np.ndarray, z_base: np.ndarray, output_root: Path) -> dict:
    sample = pl.read_csv(source_root / "data/raw/sample_submit.csv")
    production_users = sample.select("user_id").sort("user_id")
    production_sources = build_features(PRODUCTION_CUTOFF, raw, production_users)
    production_features = feature_matrix(production_sources)
    shrink = float(min(real.selected_shrink.values()))
    scaler, model, center, _ = _fit_full_correction(primary_features, y, z_base, shrink)
    cp = np.clip(model.predict(scaler.transform(matrix_np(primary_features))) - center,
                 -CORRECTION_BOUND, CORRECTION_BOUND)
    ct = np.clip(model.predict(scaler.transform(matrix_np(production_features))) - center,
                 -CORRECTION_BOUND, CORRECTION_BOUND)
    support = _support_audit(primary_sources, production_sources, primary_features,
                             production_features, cp, ct)
    _write_json(output_root / "production_support.json", support)
    if support["status"] != "PASS":
        return {"created": False, "reason": support["status"], "support": support}

    strongest = pl.read_csv(source_root / "submissions/submission_STRONGEST_CURRENT.csv")
    if strongest.height != sample.height or strongest["user_id"].n_unique() != sample.height:
        raise AssertionError("STRONGEST_CURRENT composition invalid")
    ordered = production_users.join(strongest, on="user_id", how="left").sort("user_id")
    if ordered["predict"].null_count():
        raise AssertionError("STRONGEST_CURRENT row alignment failed")
    z_test = np.log1p(ordered["predict"].to_numpy())
    z_candidate = z_test + shrink * ct
    z_candidate += z_test.mean() - z_candidate.mean()
    pred = np.maximum(np.expm1(z_candidate), 0.0)
    candidate = pl.DataFrame({"user_id": production_users["user_id"], "predict": pred})
    candidate = align_submission(sample, candidate)
    path = worktree / "submissions" / "submission_EXACT_ANNIVERSARY.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_csv(path, float_precision=6)
    np.save(output_root / "production_correction.npy", ct)
    return {"created": True, "path": str(path), "shrink": shrink,
            "base_mean_z": float(z_test.mean()), "candidate_mean_z": float(z_candidate.mean()),
            "support": support}


def run(source_root: Path, output_root: Path, allow_retrained_tcn: bool = False) -> dict:
    started = time.time()
    worktree = Path(__file__).resolve().parent.parent
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = preregistration()
    novelty = novelty_audit()
    _write_json(output_root / "preregistered_manifest.json", manifest)
    _write_json(output_root / "novelty_audit.json", novelty)
    parity = baseline_preflight(source_root, [worktree] if allow_retrained_tcn else [])
    replay = None
    seed42 = next(r for r in parity["checkpoints"] if r["component"] == "TCN seed42")
    if (parity["status"] == "PASS" and allow_retrained_tcn
            and seed42["location"] == str(worktree / "artifacts" / "model_SEQ-01-TEST.pt")):
        replay = recipe_parity_audit(source_root, worktree, parity)
        parity["retrained_seed42_audit"] = replay
        if replay["status"] != "PASS_RETRAINED_RECIPE_PARITY":
            parity["status"] = "BLOCKED_BASELINE_PARITY"
            parity["reason"] = replay.get("reason", "target-blind recipe replay outside limits")
    _write_json(output_root / "baseline_parity.json", parity)
    summary: dict = {
        "experiment": EXP_ID,
        "status": parity["status"],
        "novelty": novelty["decision"],
        "target_labels_read": False,
        "features_materialized": False,
        "cross_fit_performed": False,
        "production_candidate_created": False,
        "baseline_parity": parity,
        "required_statements": {
            "ORDINARY TEMPORAL CV": "unavailable",
            "PSEUDO-PRODUCTION CROSS-FIT": "not performed: baseline parity blocked",
            "TRANSFER RISK": "one-month calendar extrapolation",
        },
    }
    if parity["status"] != "PASS":
        _write_json(output_root / "summary.json", summary)
        print(parity["status"])
        for missing in parity["missing_checkpoints"]:
            print(f"missing: {missing}")
        return summary

    # Target access starts only after novelty, manifest, asset and target-blind
    # replay audits are persisted.
    sample = pl.read_csv(source_root / "data/raw/sample_submit.csv")
    users = sample.select("user_id").sort("user_id")
    z_base, inference_audit = strongest_at(PRIMARY_CUTOFF, users, source_root,
                                           parity, output_root)
    _write_json(output_root / "primary_baseline_inference.json", inference_audit)
    raw = pl.read_parquet(source_root / "data/raw/train.parquet")
    _write_json(output_root / "feature_window_support.json",
                feature_window_support(raw, PRIMARY_CUTOFF))
    sources_real = build_features(PRIMARY_CUTOFF, raw, users)
    sources_shuffled, shuffle_audit = shuffle_anniversary(sources_real, PRIMARY_CUTOFF)
    sources_shifted = build_features(PRIMARY_CUTOFF, raw, users, shifted=True)
    features_real = feature_matrix(sources_real)
    features_shuffled = feature_matrix(sources_shuffled)
    features_shifted = feature_matrix(sources_shifted)
    y = _target_labels(raw, users, PRIMARY_CUTOFF)
    np.save(output_root / "primary_y.npy", y)
    _write_json(output_root / "shuffle_audit.json", shuffle_audit)
    real = crossfit_arm("ANNIV_REAL", features_real, users["user_id"].to_numpy(), y, z_base)
    shuffled = crossfit_arm("ANNIV_SHUFFLED", features_shuffled,
                            users["user_id"].to_numpy(), y, z_base)
    shifted = crossfit_arm("ANNIV_SHIFTED", features_shifted,
                           users["user_id"].to_numpy(), y, z_base)
    metrics = _all_metrics((real, shuffled, shifted), y, z_base, sources_real)
    pl.DataFrame(metrics).write_csv(output_root / "crossfit_metrics.csv")
    for arm in (real, shuffled, shifted):
        np.save(output_root / f"correction_{arm.name}.npy", arm.correction)
        np.save(output_root / f"z_{arm.name}.npy", arm.z)
    decision = verdict(real, shuffled, shifted, y, sources_real)
    _write_json(output_root / "verdict.json", decision)
    production = {"created": False, "reason": decision["verdict"]}
    if decision["verdict"] == "STRONG_PASS":
        production = _production_candidate(source_root, worktree, raw, users, sources_real,
                                           features_real, real, y, z_base, output_root)
    summary.update({
        "status": decision["verdict"], "target_labels_read": True,
        "features_materialized": True, "cross_fit_performed": True,
        "production_candidate_created": production["created"],
        "verdict": decision, "production": production,
        "elapsed_seconds": time.time() - started,
        "required_statements": {
            "ORDINARY TEMPORAL CV": "unavailable",
            "PSEUDO-PRODUCTION CROSS-FIT": "performed",
            "TRANSFER RISK": "one-month calendar extrapolation",
        },
    })
    _write_json(output_root / "summary.json", summary)
    print(decision["verdict"])
    return summary


def main() -> None:
    worktree = Path(__file__).resolve().parent.parent
    default_source = worktree.parent / "OZON-E-CUP"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=default_source,
                        help="read-only repository containing data and production artifacts")
    parser.add_argument("--output-root", type=Path,
                        default=worktree / "artifacts" / EXP_ID)
    parser.add_argument("--allow-retrained-tcn", action="store_true",
                        help="use the explicitly authorized seed-42 recipe replay after its "
                             "target-blind parity audit passes")
    args = parser.parse_args()
    run(args.source_root.resolve(), args.output_root.resolve(), args.allow_retrained_tcn)


if __name__ == "__main__":
    main()
