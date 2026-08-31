"""EXP-061: cutoff-safe unresolved funnel representation preflight.

Run with one command:
    python src/open_funnel.py

This is an artifact-first falsification, not a production model.  It joins new
opt-in columns built by ``build_features(cutoff)`` to the exact exp_037 OOF
baseline, then compares a cross-user residual probe with a joint within-state
shuffle of only the new columns.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ROOT, SEED, VAL_FOLDS_S1, FOLD_WEIGHTS_S1
from src.features import OPEN_FUNNEL_COLUMNS, build_features
from src.validation import calibrate


ARTIFACTS = ROOT / "artifacts" / "OPEN_FUNNEL_EXP061"
RESULTS = ROOT / "research" / "strategies" / "results" / "OPEN_FUNNEL_EXP061"
ALIGNED = ROOT / "artifacts" / "RESDISC_053" / "aligned_oof.parquet"
BASE_WCV = 1.7475098625201952
SCALES = np.asarray([0.0, 0.25, 0.50, 0.75, 1.0], dtype=float)

CONTROL_COLUMNS = [
    "z_strong_raw", "p_act_dist", "etx_minus_seq", "rec_buy",
    "w30_days_buy", "w90_days_buy", "w90_days_search", "w90_days_cart",
    "w90_days_present", "w90_buyday_rate", "w90_cart2ord",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
                    encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    names: list[str] = []
    for row in rows:
        for name in row:
            if name not in names:
                names.append(name)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def _splitmix64(values: np.ndarray, salt: int = 0) -> np.ndarray:
    x = np.asarray(values, dtype=np.uint64) + np.uint64(salt)
    x = x + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def _finite_matrix(frame: pl.DataFrame, names: list[str]) -> np.ndarray:
    X = frame.select(names).to_numpy().astype(np.float32)
    X[~np.isfinite(X)] = np.nan
    return X


def _state_strata(frame: pl.DataFrame) -> np.ndarray:
    rec = frame["rec_buy"].fill_null(999).to_numpy()
    buys = frame["w90_days_buy"].fill_null(0).to_numpy()
    present = frame["w90_days_present"].fill_null(0).to_numpy()
    pred = frame["z_strong_raw"].to_numpy()
    rec_bin = np.digitize(rec, [7, 14, 30, 60, 90, 180])
    buy_bin = np.digitize(buys, [0, 1, 2, 5])
    present_bin = np.minimum((present // 15).astype(int), 5)
    edges = np.quantile(pred, [0.2, 0.4, 0.6, 0.8])
    pred_bin = np.digitize(pred, edges)
    return (((rec_bin * 5 + buy_bin) * 6 + present_bin) * 5 + pred_bin).astype(np.int32)


def _joint_shuffle(X: np.ndarray, strata: np.ndarray, salt: int) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(int(SEED) + salt)
    perm = np.arange(len(strata))
    order = np.argsort(strata, kind="stable")
    sorted_s = strata[order]
    bounds = np.r_[0, np.flatnonzero(sorted_s[1:] != sorted_s[:-1]) + 1, len(order)]
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        idx = order[lo:hi]
        if len(idx) > 1:
            perm[idx] = rng.permutation(idx)
    shuffled = X[perm]
    return shuffled, {
        "n_strata": int(len(bounds) - 1),
        "singleton_share": float(np.mean(np.bincount(strata)[strata] == 1)),
        "moved_share": float(np.mean(perm != np.arange(len(perm)))),
        "max_abs_marginal_mean_diff": float(np.nanmax(np.abs(
            np.nanmean(X, axis=0) - np.nanmean(shuffled, axis=0)))),
    }


def _params() -> dict[str, Any]:
    return {
        "objective": "regression_l1", "metric": "l1", "learning_rate": 0.035,
        "num_leaves": 15, "min_data_in_leaf": 1000, "lambda_l2": 50.0,
        "feature_fraction": 1.0, "bagging_fraction": 1.0, "bagging_freq": 0,
        "max_bin": 63, "seed": int(SEED), "feature_fraction_seed": int(SEED),
        "bagging_seed": int(SEED), "data_random_seed": int(SEED),
        "deterministic": True, "force_row_wise": True, "num_threads": 4,
        "verbosity": -1,
    }


def _cross_user_prediction(X: np.ndarray, target: np.ndarray, uid: np.ndarray) -> np.ndarray:
    group = (_splitmix64(uid) % np.uint64(4)).astype(np.int8)
    pred = np.full(len(target), np.nan, dtype=float)
    for held in range(4):
        train = group != held
        valid = ~train
        ds = lgb.Dataset(X[train], label=target[train].astype(np.float32), free_raw_data=True)
        model = lgb.train(_params(), ds, num_boost_round=120)
        pred[valid] = model.predict(X[valid])
    if not np.isfinite(pred).all():
        raise AssertionError("cross-user prediction is incomplete")
    return np.clip(pred, -0.25, 0.25)


def _two_sided_candidate(y: np.ndarray, z: np.ndarray, correction: np.ndarray,
                         uid: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    side = (_splitmix64(uid, salt=0x51) & np.uint64(1)).astype(np.int8)
    applied = np.zeros(len(y), dtype=float)
    rows: list[dict[str, Any]] = []
    for recipient in (0, 1):
        donor = side != recipient
        receive = side == recipient
        curve = []
        for scale in SCALES:
            _, score = calibrate(y[donor], z[donor] + float(scale) * correction[donor])
            curve.append((float(scale), score))
        best_score = min(score for _, score in curve)
        selected = min(scale for scale, score in curve if score <= best_score + 1e-5)
        applied[receive] = selected * correction[receive]
        rows.extend({
            "recipient_side": int(recipient), "scale": scale, "donor_score": score,
            "selected": scale == selected, "n_donor": int(donor.sum()),
            "n_recipient": int(receive.sum()),
        } for scale, score in curve)
    return applied, rows


def _score(y: np.ndarray, z: np.ndarray) -> float:
    return calibrate(y, z)[1]


def _feature_frame(cutoff: dt.date, aligned: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, Any]]:
    cache = ARTIFACTS / f"features_{cutoff:%Y%m%d}.parquet"
    if cache.exists():
        extra = pl.read_parquet(cache)
    else:
        base_path = ROOT / "data" / "processed" / f"feat_{cutoff:%Y%m%d}_LnormNone.parquet"
        base = pl.read_parquet(base_path)
        enriched = build_features(cutoff, L=None, norm_long=True,
                                  open_funnel_features=True, base_features=base)
        extra = enriched.select(["user_id"] + OPEN_FUNNEL_COLUMNS)
        extra.write_parquet(cache)
    if extra["user_id"].n_unique() != extra.height:
        raise AssertionError("open-funnel feature keys are not unique")
    joined = aligned.join(extra, on="user_id", how="left")
    joined = joined.with_columns([pl.col(c).fill_null(0) for c in OPEN_FUNNEL_COLUMNS])
    if joined.height != aligned.height or joined["user_id"].to_list() != aligned["user_id"].to_list():
        raise AssertionError("feature alignment changed exact OOF row order")
    audit = {
        "cutoff": cutoff.isoformat(), "rows": joined.height,
        "unique_users": joined["user_id"].n_unique(), "source_max_event_date": cutoff.isoformat(),
        "cache": str(cache), "cache_sha256": _sha256(cache),
        "finite_or_null": bool(all(joined[c].is_finite().fill_null(True).all()
                                    for c in OPEN_FUNNEL_COLUMNS)),
    }
    return joined, audit


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    needed = ["cutoff", "user_id", "y_true", "z_strong_raw", "r_strong"]
    needed += [c for c in CONTROL_COLUMNS if c not in needed]
    core = pl.read_parquet(ALIGNED, columns=needed)

    fold_rows: list[dict[str, Any]] = []
    scale_rows: list[dict[str, Any]] = []
    corr_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    predictions: list[pl.DataFrame] = []

    for fold_index, cutoff in enumerate(VAL_FOLDS_S1):
        fold = core.filter(pl.col("cutoff") == cutoff.isoformat()).sort("user_id")
        frame, audit = _feature_frame(cutoff, fold)
        audits.append(audit)
        uid = frame["user_id"].to_numpy()
        y = frame["y_true"].to_numpy().astype(float)
        z = frame["z_strong_raw"].to_numpy().astype(float)
        residual = frame["r_strong"].to_numpy().astype(float)
        X_open = _finite_matrix(frame, OPEN_FUNNEL_COLUMNS)
        X_control = _finite_matrix(frame, CONTROL_COLUMNS)
        strata = _state_strata(frame)
        X_shuffled_open, shuffle_audit = _joint_shuffle(X_open, strata, salt=100 + fold_index)
        audit.update({f"shuffle_{k}": v for k, v in shuffle_audit.items()})

        for j, name in enumerate(OPEN_FUNNEL_COLUMNS):
            finite = np.isfinite(X_open[:, j])
            value = X_open[finite, j]
            target = residual[finite]
            corr = 0.0 if np.std(value) == 0 else float(np.corrcoef(value, target)[0, 1])
            corr_rows.append({"cutoff": cutoff.isoformat(), "feature": name,
                              "pearson_residual": corr, "nonzero_share": float(np.mean(value != 0)),
                              "mean": float(np.mean(value)), "std": float(np.std(value))})

        arms = {
            "REAL": np.concatenate([X_control, X_open], axis=1),
            "SHUFFLED": np.concatenate([X_control, X_shuffled_open], axis=1),
            "CONTROL_ONLY": X_control,
        }
        base_score = _score(y, z)
        arm_applied: dict[str, np.ndarray] = {}
        for arm, X in arms.items():
            correction = _cross_user_prediction(X, residual, uid)
            applied, curves = _two_sided_candidate(y, z, correction, uid)
            candidate_score = _score(y, z + applied)
            arm_applied[arm] = applied
            fold_rows.append({
                "cutoff": cutoff.isoformat(), "fold_index": fold_index, "arm": arm,
                "n": len(y), "base_score": base_score, "candidate_score": candidate_score,
                "delta": candidate_score - base_score,
                "correction_std": float(np.std(applied)),
                "correction_residual_corr": float(np.corrcoef(applied, residual)[0, 1])
                if np.std(applied) > 0 else 0.0,
            })
            for row in curves:
                scale_rows.append({"cutoff": cutoff.isoformat(), "arm": arm, **row})

        side = (_splitmix64(uid, salt=0x51) & np.uint64(1)).astype(np.int8)
        for arm, applied in arm_applied.items():
            for side_value in (0, 1):
                for segment, mask0 in {
                    "ALL": np.ones(len(y), bool),
                    "LOW_BUY90": frame["w90_days_buy"].fill_null(0).to_numpy() <= 1,
                    "REC_BUY_GT30": frame["rec_buy"].fill_null(999).to_numpy() > 30,
                }.items():
                    mask = (side == side_value) & mask0
                    segment_rows.append({
                        "cutoff": cutoff.isoformat(), "arm": arm, "recipient_side": side_value,
                        "segment": segment, "n": int(mask.sum()),
                        "base_score": _score(y[mask], z[mask]),
                        "candidate_score": _score(y[mask], z[mask] + applied[mask]),
                    })

        predictions.append(pl.DataFrame({
            "cutoff": [cutoff.isoformat()] * len(y), "user_id": uid,
            "real_correction": arm_applied["REAL"].astype(np.float32),
            "shuffled_correction": arm_applied["SHUFFLED"].astype(np.float32),
            "control_correction": arm_applied["CONTROL_ONLY"].astype(np.float32),
        }))

    _write_csv(RESULTS / "fold_metrics.csv", fold_rows)
    _write_csv(RESULTS / "scale_curves.csv", scale_rows)
    _write_csv(RESULTS / "feature_correlations.csv", corr_rows)
    for row in segment_rows:
        row["delta"] = row["candidate_score"] - row["base_score"]
    _write_csv(RESULTS / "segment_metrics.csv", segment_rows)
    _write_json(RESULTS / "leakage_alignment_audit.json", audits)
    pl.concat(predictions).write_parquet(ARTIFACTS / "cross_user_corrections.parquet")

    weights = np.asarray(FOLD_WEIGHTS_S1, float)
    summary: dict[str, Any] = {
        "experiment_id": 61, "prefix": "OPEN_FUNNEL_EXP061",
        "development_reference": "STRONGEST-CURRENT / exp_037",
        "baseline_wcv_expected": BASE_WCV,
        "folds": [d.isoformat() for d in VAL_FOLDS_S1],
        "seed": int(SEED), "feature_columns": OPEN_FUNNEL_COLUMNS,
        "control_columns": CONTROL_COLUMNS, "source_aligned_sha256": _sha256(ALIGNED),
        "arms": {},
    }
    for arm in ("REAL", "SHUFFLED", "CONTROL_ONLY"):
        rows = [r for r in fold_rows if r["arm"] == arm]
        scores = np.asarray([r["candidate_score"] for r in rows])
        deltas = np.asarray([r["delta"] for r in rows])
        summary["arms"][arm] = {
            "fold_scores": scores.tolist(), "fold_deltas": deltas.tolist(),
            "wcv": float(np.average(scores, weights=weights)),
            "delta_wcv": float(np.average(deltas, weights=weights)),
            "improved_folds": int(np.sum(deltas < 0)),
            "late_delta": float(deltas[-1]),
        }
    real = summary["arms"]["REAL"]
    shuf = summary["arms"]["SHUFFLED"]
    selected = [r for r in scale_rows if r["arm"] == "REAL" and r["selected"]]
    nonzero_scales = all(float(r["scale"]) > 0 for r in selected)
    real_minus_shuf = float(real["delta_wcv"] - shuf["delta_wcv"])
    passed = (real["delta_wcv"] <= -0.0005 and real["improved_folds"] >= 3
              and real["late_delta"] < 0 and real_minus_shuf <= -0.0003
              and nonzero_scales and all(a["finite_or_null"] for a in audits))
    summary["decision"] = {
        "real_minus_shuffled_delta_wcv": real_minus_shuf,
        "all_real_selected_scales_nonzero": nonzero_scales,
        "success_gate_passed": bool(passed),
        "verdict": "CONTINUE" if passed else "REJECT",
        "next": "canonical feature/model pilot" if passed else "pivot to MONETARY-TAIL",
    }
    _write_json(RESULTS / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
