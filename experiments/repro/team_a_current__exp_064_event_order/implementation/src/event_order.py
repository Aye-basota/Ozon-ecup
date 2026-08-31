"""EXP-064: explicit daily funnel-transition preflight.

Run: python src/event_order.py
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
import sys
import time

import numpy as np
import polars as pl

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import FOLD_WEIGHTS_S1, ROOT, SEED, VAL_FOLDS_S1
from src.features import EVENT_ORDER_COLUMNS, build_features
from src.open_funnel import (
    CONTROL_COLUMNS, _cross_user_prediction, _finite_matrix, _score, _sha256,
    _splitmix64, _two_sided_candidate, _write_csv, _write_json,
)


ARTIFACTS = ROOT / "artifacts" / "EVENT_ORDER_EXP064"
RESULTS = ROOT / "research" / "strategies" / "results" / "EVENT_ORDER_EXP064"
ALIGNED = ROOT / "artifacts" / "RESDISC_053" / "aligned_oof.parquet"
BASE_WCV = 1.7475098625201952


def _build_extra(cutoff: dt.date, source: str) -> tuple[pl.DataFrame, dict]:
    cache = ARTIFACTS / f"features_{source}_{cutoff:%Y%m%d}.parquet"
    if cache.exists():
        extra = pl.read_parquet(cache)
    else:
        base_path = ROOT / "data" / "processed" / f"feat_{cutoff:%Y%m%d}_LnormNone.parquet"
        base = pl.read_parquet(base_path)
        enriched = build_features(
            cutoff, L=None, norm_long=True, event_order_source=source, base_features=base)
        extra = enriched.select(["user_id"] + EVENT_ORDER_COLUMNS)
        extra.write_parquet(cache)
    assert extra["user_id"].n_unique() == extra.height
    return extra, {
        "source": source, "cache": str(cache.relative_to(ROOT)),
        "cache_sha256": _sha256(cache), "rows": extra.height,
        "unique_users": extra["user_id"].n_unique(),
        "finite_or_null": bool(all(extra[c].is_finite().fill_null(True).all()
                                   for c in EVENT_ORDER_COLUMNS)),
        "source_max_event_date": cutoff.isoformat(),
    }


def _feature_frames(cutoff: dt.date, aligned: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    real, real_audit = _build_extra(cutoff, "real")
    shuf, shuf_audit = _build_extra(cutoff, "shuffled")
    real = aligned.join(real, on="user_id", how="left").with_columns(
        [pl.col(c).fill_null(0) for c in EVENT_ORDER_COLUMNS])
    shuf = aligned.join(shuf, on="user_id", how="left").with_columns(
        [pl.col(c).fill_null(0) for c in EVENT_ORDER_COLUMNS])
    assert real.height == aligned.height == shuf.height
    assert real["user_id"].to_list() == aligned["user_id"].to_list() == shuf["user_id"].to_list()
    xr = _finite_matrix(real, EVENT_ORDER_COLUMNS)
    xs = _finite_matrix(shuf, EVENT_ORDER_COLUMNS)
    transition_parity = bool(np.array_equal(
        real["eo90_transition_count"].to_numpy(), shuf["eo90_transition_count"].to_numpy()))
    changed = np.any(np.nan_to_num(xr) != np.nan_to_num(xs), axis=1)
    audit = {
        "cutoff": cutoff.isoformat(), "rows": real.height,
        "real": real_audit, "shuffled": shuf_audit,
        "transition_count_exact_parity": transition_parity,
        "changed_feature_row_share": float(changed.mean()),
        "movement_pass": bool(changed.mean() > 0.20),
    }
    return real, shuf, audit


def main() -> None:
    started = time.time()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    needed = ["cutoff", "user_id", "y_true", "z_strong_raw", "r_strong"]
    needed += [c for c in CONTROL_COLUMNS if c not in needed]
    core = pl.read_parquet(ALIGNED, columns=needed)
    fold_rows: list[dict] = []
    scale_rows: list[dict] = []
    corr_rows: list[dict] = []
    segment_rows: list[dict] = []
    audits: list[dict] = []
    predictions: list[pl.DataFrame] = []

    for fold_index, cutoff in enumerate(VAL_FOLDS_S1):
        fold = core.filter(pl.col("cutoff") == cutoff.isoformat()).sort("user_id")
        real, shuf, audit = _feature_frames(cutoff, fold)
        audits.append(audit)
        uid = real["user_id"].to_numpy()
        y = real["y_true"].to_numpy().astype(float)
        z = real["z_strong_raw"].to_numpy().astype(float)
        residual = real["r_strong"].to_numpy().astype(float)
        x_real = _finite_matrix(real, EVENT_ORDER_COLUMNS)
        x_shuf = _finite_matrix(shuf, EVENT_ORDER_COLUMNS)
        x_control = _finite_matrix(real, CONTROL_COLUMNS)
        for j, name in enumerate(EVENT_ORDER_COLUMNS):
            finite = np.isfinite(x_real[:, j])
            value, target = x_real[finite, j], residual[finite]
            corr = 0.0 if np.std(value) == 0 else float(np.corrcoef(value, target)[0, 1])
            corr_rows.append({
                "cutoff": cutoff.isoformat(), "feature": name,
                "pearson_residual": corr, "nonzero_share": float(np.mean(value != 0)),
                "mean": float(np.mean(value)), "std": float(np.std(value)),
            })
        arms = {
            "REAL": np.concatenate([x_control, x_real], axis=1),
            "SHUFFLED": np.concatenate([x_control, x_shuf], axis=1),
            "CONTROL_ONLY": x_control,
        }
        base_score = _score(y, z)
        applied_by_arm: dict[str, np.ndarray] = {}
        for arm, matrix in arms.items():
            correction = _cross_user_prediction(matrix, residual, uid)
            applied, curves = _two_sided_candidate(y, z, correction, uid)
            applied_by_arm[arm] = applied
            score = _score(y, z + applied)
            fold_rows.append({
                "cutoff": cutoff.isoformat(), "fold_index": fold_index, "arm": arm,
                "n": len(y), "base_score": base_score, "candidate_score": score,
                "delta": score - base_score, "correction_std": float(np.std(applied)),
                "correction_residual_corr": float(np.corrcoef(applied, residual)[0, 1])
                if np.std(applied) > 0 else 0.0,
            })
            scale_rows.extend({"cutoff": cutoff.isoformat(), "arm": arm, **row} for row in curves)
        side = (_splitmix64(uid, salt=0x51) & np.uint64(1)).astype(np.int8)
        support = real["eo90_transition_count"].to_numpy() >= 5
        for arm, applied in applied_by_arm.items():
            for side_value in (0, 1):
                for segment, mask0 in {"ALL": np.ones(len(y), bool), "TRANSITIONS_GE5": support}.items():
                    mask = (side == side_value) & mask0
                    bs, cs = _score(y[mask], z[mask]), _score(y[mask], z[mask] + applied[mask])
                    segment_rows.append({
                        "cutoff": cutoff.isoformat(), "arm": arm,
                        "recipient_side": side_value, "segment": segment, "n": int(mask.sum()),
                        "base_score": bs, "candidate_score": cs, "delta": cs - bs,
                    })
        predictions.append(pl.DataFrame({
            "cutoff": [cutoff.isoformat()] * len(y), "user_id": uid,
            "real_correction": applied_by_arm["REAL"].astype(np.float32),
            "shuffled_correction": applied_by_arm["SHUFFLED"].astype(np.float32),
            "control_correction": applied_by_arm["CONTROL_ONLY"].astype(np.float32),
        }))
        print(f"completed {cutoff}: movement={audit['changed_feature_row_share']:.3f}", flush=True)

    _write_csv(RESULTS / "fold_metrics.csv", fold_rows)
    _write_csv(RESULTS / "scale_curves.csv", scale_rows)
    _write_csv(RESULTS / "feature_correlations.csv", corr_rows)
    _write_csv(RESULTS / "segment_metrics.csv", segment_rows)
    _write_json(RESULTS / "leakage_alignment_shuffle_audit.json", audits)
    pl.concat(predictions).write_parquet(ARTIFACTS / "cross_user_corrections.parquet")
    weights = np.asarray(FOLD_WEIGHTS_S1, float)
    summary = {
        "experiment_id": 64, "prefix": "EVENT_ORDER_EXP064",
        "development_reference": "STRONGEST-CURRENT / exp_037",
        "baseline_wcv_expected": BASE_WCV, "seed": int(SEED),
        "feature_columns": EVENT_ORDER_COLUMNS, "control_columns": CONTROL_COLUMNS,
        "source_aligned_sha256": _sha256(ALIGNED), "arms": {},
    }
    for arm in ("REAL", "SHUFFLED", "CONTROL_ONLY"):
        rows = [r for r in fold_rows if r["arm"] == arm]
        scores = np.asarray([r["candidate_score"] for r in rows])
        deltas = np.asarray([r["delta"] for r in rows])
        summary["arms"][arm] = {
            "fold_scores": scores.tolist(), "fold_deltas": deltas.tolist(),
            "wcv": float(np.average(scores, weights=weights)),
            "delta_wcv": float(np.average(deltas, weights=weights)),
            "improved_folds": int((deltas < 0).sum()), "late_delta": float(deltas[-1]),
        }
    real_s, shuf_s = summary["arms"]["REAL"], summary["arms"]["SHUFFLED"]
    selected = [r for r in scale_rows if r["arm"] == "REAL" and r["selected"]]
    nonzero = all(float(r["scale"]) > 0 for r in selected)
    audit_pass = all(a["transition_count_exact_parity"] and a["movement_pass"]
                     and a["real"]["finite_or_null"] and a["shuffled"]["finite_or_null"]
                     for a in audits)
    real_minus_shuf = float(real_s["delta_wcv"] - shuf_s["delta_wcv"])
    passed = bool(
        audit_pass and real_s["delta_wcv"] <= -0.0005
        and real_s["improved_folds"] >= 3 and real_s["late_delta"] < 0
        and real_minus_shuf <= -0.0003 and nonzero)
    summary["runtime_s"] = time.time() - started
    summary["decision"] = {
        "audits_pass": audit_pass, "real_minus_shuffled_delta_wcv": real_minus_shuf,
        "all_real_selected_scales_nonzero": nonzero, "success_gate_passed": passed,
        "verdict": "CONTINUE" if passed else "REJECT",
        "next": "canonical low-capacity model pilot" if passed else "final integration audit",
    }
    _write_json(RESULTS / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
