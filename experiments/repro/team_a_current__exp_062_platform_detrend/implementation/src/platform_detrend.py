"""EXP-062 PLATFORM-DETREND matched-placebo residual preflight.

Run with one command:
    python src/platform_detrend.py
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import polars as pl

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CUTOFF_TEST, FOLD_WEIGHTS_S1, ROOT, SEED, VAL_FOLDS_S1
from src.features import (PLATFORM_DETREND_COLUMNS, _platform_daily_factors,
                          build_features, panel_users)
from src.open_funnel import (ALIGNED, BASE_WCV, CONTROL_COLUMNS, _cross_user_prediction,
                             _finite_matrix, _score, _sha256, _splitmix64,
                             _two_sided_candidate, _write_csv, _write_json)


ARTIFACTS = ROOT / "artifacts" / "PLATFORM_DETREND_EXP062"
RESULTS = ROOT / "research" / "strategies" / "results" / "PLATFORM_DETREND_EXP062"
FACTOR_COLUMNS = [
    "_pd_present_factor", "_pd_searches_factor", "_pd_carts_factor",
    "_pd_orders_factor", "_pd_gmv_factor",
]


def _factor_stats(cutoff: dt.date) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    real = _platform_daily_factors(cutoff, "real")
    placebo = _platform_daily_factors(cutoff, "shuffled")
    rows: list[dict[str, Any]] = []
    for name in FACTOR_COLUMNS:
        x = real[name].to_numpy().astype(float)
        rows.append({
            "cutoff": cutoff.isoformat(), "factor": name,
            "min": float(np.min(x)), "q01": float(np.quantile(x, 0.01)),
            "median": float(np.median(x)), "q99": float(np.quantile(x, 0.99)),
            "max": float(np.max(x)), "mean": float(np.mean(x)), "std": float(np.std(x)),
        })
    a = real.select(FACTOR_COLUMNS).to_numpy()
    b = placebo.select(FACTOR_COLUMNS).to_numpy()
    audit = {
        "cutoff": cutoff.isoformat(), "days": real.height,
        "panel_rows": panel_users(cutoff).height,
        "factor_marginal_max_abs_diff": float(np.max(np.abs(
            np.sort(a, axis=0) - np.sort(b, axis=0)))),
        "date_alignment_changed_share": float(np.mean(np.any(np.abs(a - b) > 1e-12, axis=1))),
        "source_max_event_date": cutoff.isoformat(),
    }
    return rows, audit


def _feature_pair(cutoff: dt.date, aligned: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, Any]]:
    paths = {
        source: ARTIFACTS / f"features_{source}_{cutoff:%Y%m%d}.parquet"
        for source in ("real", "shuffled")
    }
    extras: dict[str, pl.DataFrame] = {}
    base_path = ROOT / "data" / "processed" / f"feat_{cutoff:%Y%m%d}_LnormNone.parquet"
    base = None
    for source, path in paths.items():
        if path.exists():
            extra = pl.read_parquet(path)
        else:
            if base is None:
                base = pl.read_parquet(base_path)
            enriched = build_features(cutoff, L=None, norm_long=True,
                                      platform_detrend_source=source, base_features=base)
            extra = enriched.select(["user_id"] + PLATFORM_DETREND_COLUMNS)
            extra.write_parquet(path)
        if extra["user_id"].n_unique() != extra.height:
            raise AssertionError(f"{source} platform feature keys are not unique")
        extras[source] = extra
    frame = aligned
    for source, extra in extras.items():
        frame = frame.join(extra.rename({c: f"{c}_{source}" for c in PLATFORM_DETREND_COLUMNS}),
                           on="user_id", how="left")
    fill = [f"{c}_{source}" for source in ("real", "shuffled")
            for c in PLATFORM_DETREND_COLUMNS]
    frame = frame.with_columns([pl.col(c).fill_null(0) for c in fill])
    if frame.height != aligned.height or frame["user_id"].to_list() != aligned["user_id"].to_list():
        raise AssertionError("platform feature alignment changed exact OOF order")
    audit = {
        "cutoff": cutoff.isoformat(), "rows": frame.height,
        "unique_users": frame["user_id"].n_unique(),
        "panel_matches_aligned": panel_users(cutoff).height == frame.height,
        "real_cache_sha256": _sha256(paths["real"]),
        "shuffled_cache_sha256": _sha256(paths["shuffled"]),
        "finite_or_null": bool(all(frame[c].is_finite().fill_null(True).all() for c in fill)),
    }
    return frame, audit


def main() -> None:
    started = time.time()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    needed = ["cutoff", "user_id", "y_true", "z_strong_raw", "r_strong"]
    needed += [c for c in CONTROL_COLUMNS if c not in needed]
    core = pl.read_parquet(ALIGNED, columns=needed)

    fold_rows: list[dict[str, Any]] = []
    scale_rows: list[dict[str, Any]] = []
    corr_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    factor_rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    predictions: list[pl.DataFrame] = []

    for fold_index, cutoff in enumerate(VAL_FOLDS_S1):
        fold = core.filter(pl.col("cutoff") == cutoff.isoformat()).sort("user_id")
        frame, audit = _feature_pair(cutoff, fold)
        stats, factor_audit = _factor_stats(cutoff)
        factor_rows.extend(stats)
        audit.update(factor_audit)
        audits.append(audit)

        uid = frame["user_id"].to_numpy()
        y = frame["y_true"].to_numpy().astype(float)
        z = frame["z_strong_raw"].to_numpy().astype(float)
        residual = frame["r_strong"].to_numpy().astype(float)
        X_control = _finite_matrix(frame, CONTROL_COLUMNS)
        real_names = [f"{c}_real" for c in PLATFORM_DETREND_COLUMNS]
        placebo_names = [f"{c}_shuffled" for c in PLATFORM_DETREND_COLUMNS]
        X_real = _finite_matrix(frame, real_names)
        X_placebo = _finite_matrix(frame, placebo_names)
        for j, name in enumerate(PLATFORM_DETREND_COLUMNS):
            for source, X in (("REAL", X_real), ("PLACEBO", X_placebo)):
                finite = np.isfinite(X[:, j])
                value = X[finite, j]
                target = residual[finite]
                corr = 0.0 if np.std(value) == 0 else float(np.corrcoef(value, target)[0, 1])
                corr_rows.append({"cutoff": cutoff.isoformat(), "source": source,
                                  "feature": name, "pearson_residual": corr,
                                  "mean": float(np.mean(value)), "std": float(np.std(value))})

        arms = {
            "REAL": np.concatenate([X_control, X_real], axis=1),
            "PLACEBO": np.concatenate([X_control, X_placebo], axis=1),
            "CONTROL_ONLY": X_control,
        }
        base_score = _score(y, z)
        applied_by_arm: dict[str, np.ndarray] = {}
        for arm, X in arms.items():
            correction = _cross_user_prediction(X, residual, uid)
            applied, curves = _two_sided_candidate(y, z, correction, uid)
            candidate_score = _score(y, z + applied)
            applied_by_arm[arm] = applied
            fold_rows.append({
                "cutoff": cutoff.isoformat(), "fold_index": fold_index, "arm": arm,
                "n": len(y), "base_score": base_score, "candidate_score": candidate_score,
                "delta": candidate_score - base_score, "correction_std": float(np.std(applied)),
                "correction_residual_corr": float(np.corrcoef(applied, residual)[0, 1])
                if np.std(applied) > 0 else 0.0,
            })
            scale_rows.extend({"cutoff": cutoff.isoformat(), "arm": arm, **row} for row in curves)

        side = (_splitmix64(uid, salt=0x51) & np.uint64(1)).astype(np.int8)
        masks = {
            "ALL": np.ones(len(y), bool),
            "RECENT_SEARCH": frame["w90_days_search"].fill_null(0).to_numpy() >= 10,
            "LOW_BUY90": frame["w90_days_buy"].fill_null(0).to_numpy() <= 1,
        }
        for arm, applied in applied_by_arm.items():
            for side_value in (0, 1):
                for segment, base_mask in masks.items():
                    mask = (side == side_value) & base_mask
                    sb = _score(y[mask], z[mask])
                    sc = _score(y[mask], z[mask] + applied[mask])
                    segment_rows.append({"cutoff": cutoff.isoformat(), "arm": arm,
                                         "recipient_side": side_value, "segment": segment,
                                         "n": int(mask.sum()), "base_score": sb,
                                         "candidate_score": sc, "delta": sc - sb})
        predictions.append(pl.DataFrame({
            "cutoff": [cutoff.isoformat()] * len(y), "user_id": uid,
            "real_correction": applied_by_arm["REAL"].astype(np.float32),
            "placebo_correction": applied_by_arm["PLACEBO"].astype(np.float32),
            "control_correction": applied_by_arm["CONTROL_ONLY"].astype(np.float32),
        }))

    test_stats, test_audit = _factor_stats(CUTOFF_TEST)
    factor_rows.extend(test_stats)
    audits.append({"scope": "test_input_only", **test_audit})
    _write_csv(RESULTS / "fold_metrics.csv", fold_rows)
    _write_csv(RESULTS / "scale_curves.csv", scale_rows)
    _write_csv(RESULTS / "feature_correlations.csv", corr_rows)
    _write_csv(RESULTS / "segment_metrics.csv", segment_rows)
    _write_csv(RESULTS / "factor_support.csv", factor_rows)
    _write_json(RESULTS / "leakage_selection_regime_audit.json", audits)
    pl.concat(predictions).write_parquet(ARTIFACTS / "cross_user_corrections.parquet")

    weights = np.asarray(FOLD_WEIGHTS_S1, float)
    summary: dict[str, Any] = {
        "experiment_id": 62, "prefix": "PLATFORM_DETREND_EXP062",
        "development_reference": "STRONGEST-CURRENT / exp_037",
        "baseline_wcv_expected": BASE_WCV, "seed": int(SEED),
        "feature_columns": PLATFORM_DETREND_COLUMNS, "control_columns": CONTROL_COLUMNS,
        "source_aligned_sha256": _sha256(ALIGNED), "arms": {},
    }
    for arm in ("REAL", "PLACEBO", "CONTROL_ONLY"):
        rows = [r for r in fold_rows if r["arm"] == arm]
        scores = np.asarray([r["candidate_score"] for r in rows])
        deltas = np.asarray([r["delta"] for r in rows])
        summary["arms"][arm] = {
            "fold_scores": scores.tolist(), "fold_deltas": deltas.tolist(),
            "wcv": float(np.average(scores, weights=weights)),
            "delta_wcv": float(np.average(deltas, weights=weights)),
            "improved_folds": int(np.sum(deltas < 0)), "late_delta": float(deltas[-1]),
        }
    real = summary["arms"]["REAL"]
    placebo = summary["arms"]["PLACEBO"]
    selected = [r for r in scale_rows if r["arm"] == "REAL" and r["selected"]]
    nonzero_scales = all(float(r["scale"]) > 0 for r in selected)
    real_minus_placebo = float(real["delta_wcv"] - placebo["delta_wcv"])
    audits_pass = all(a.get("finite_or_null", True) and a.get("panel_matches_aligned", True)
                      and a.get("factor_marginal_max_abs_diff", 0.0) <= 1e-12 for a in audits)
    passed = (audits_pass and real["delta_wcv"] <= -0.0005
              and real["improved_folds"] >= 3 and real["late_delta"] < 0
              and real_minus_placebo <= -0.0003 and nonzero_scales)
    summary["runtime_s"] = time.time() - started
    summary["decision"] = {
        "real_minus_placebo_delta_wcv": real_minus_placebo,
        "all_real_selected_scales_nonzero": nonzero_scales,
        "audits_pass": audits_pass, "success_gate_passed": bool(passed),
        "verdict": "CONTINUE" if passed else "REJECT",
        "next": "canonical model pilot" if passed else "vision reset",
    }
    _write_json(RESULTS / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
