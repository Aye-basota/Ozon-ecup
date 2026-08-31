"""Dense clean-cutoff ensemble for team-b Strategy 1.

This script keeps the existing team-b feature sets but changes the training
scheme toward the stronger Strategy 1 practice: many weekly train cutoffs and
clean validation cutoffs that avoid the late poisoned window.

Examples:
    python src/dense_ensemble.py cv --folds 1
    python src/dense_ensemble.py cv --folds 1 --recent-train-cutoffs 8
    python src/dense_ensemble.py submit --output exp_011_dense_clean_logens.csv
"""
import argparse
import gc
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import SEED, SUBMISSIONS
from src.features import build_features
from src.train import TEST_CUTOFF, build_target, rmsle

CLEAN_START = date(2025, 4, 3)
CLEAN_END = date(2025, 10, 16)
CLEAN_VALS = [date(2025, 9, 4), date(2025, 9, 18), date(2025, 10, 2), date(2025, 10, 16)]

ENSEMBLE_CONFIG = [
    {"feature_set": "recency", "scale": 0.64, "weight": 0.5},
    {"feature_set": "long_buy", "scale": 0.62, "weight": 0.5},
]


def clean_grid(start: date = CLEAN_START, end: date = CLEAN_END, step_days: int = 7) -> list[str]:
    out = []
    cur = start
    while cur <= end:
        out.append(cur.isoformat())
        cur += timedelta(days=step_days)
    return out


def train_cutoffs_for_val(val_cutoff: str) -> list[str]:
    val = date.fromisoformat(val_cutoff)
    return [c for c in clean_grid() if date.fromisoformat(c) + timedelta(days=30) <= val]


def make_model():
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        objective="regression",
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=0.05,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )


def make_xy(cutoff: str, feature_set: str):
    x = build_features(cutoff, feature_set=feature_set)
    y = build_target(cutoff).reindex(x.index).fillna(0.0)
    x = x.replace([np.inf, -np.inf], 0).fillna(0).astype("float32")
    return x, y


def fit_predict_z(feature_set: str, train_cutoffs: list[str], pred_cutoff: str, scale: float):
    x_parts = []
    y_parts = []
    for cutoff in train_cutoffs:
        x_cutoff, y_cutoff = make_xy(cutoff, feature_set)
        x_parts.append(x_cutoff)
        y_parts.append(y_cutoff)
        print(f"{feature_set} train cutoff {cutoff}: X={x_cutoff.shape}, target_mean={y_cutoff.mean():.4f}")

    x_train = pd.concat(x_parts, axis=0)
    y_train = pd.concat(y_parts, axis=0)
    del x_parts, y_parts
    gc.collect()

    model = make_model()
    model.fit(x_train, np.log1p(y_train))
    del x_train, y_train
    gc.collect()

    x_pred = build_features(pred_cutoff, feature_set=feature_set)
    x_pred = x_pred.replace([np.inf, -np.inf], 0).fillna(0).astype("float32")
    pred = np.clip(np.expm1(model.predict(x_pred)) * scale, 0, None)
    return pd.Series(np.log1p(pred), index=x_pred.index)


def choose_train_cutoffs(cutoffs: list[str], recent_train_cutoffs: int | None) -> list[str]:
    if recent_train_cutoffs and recent_train_cutoffs > 0:
        return cutoffs[-recent_train_cutoffs:]
    return cutoffs


def ensemble_config(components: str, recency_weight: float) -> list[dict]:
    if components == "recency":
        return [{**ENSEMBLE_CONFIG[0], "weight": 1.0}]
    if components == "long_buy":
        return [{**ENSEMBLE_CONFIG[1], "weight": 1.0}]
    return [
        {**ENSEMBLE_CONFIG[0], "weight": recency_weight},
        {**ENSEMBLE_CONFIG[1], "weight": 1.0 - recency_weight},
    ]


def ensemble_z(train_cutoffs: list[str], pred_cutoff: str, recency_weight: float, global_scale: float, components: str):
    config = ensemble_config(components, recency_weight)
    z_parts = []
    total_weight = sum(item["weight"] for item in config)
    for item in config:
        z = fit_predict_z(item["feature_set"], train_cutoffs, pred_cutoff, item["scale"])
        z_parts.append(z * item["weight"])
    z_mix = sum(z_parts) / total_weight
    if global_scale != 1.0:
        pred = np.clip(np.expm1(z_mix) * global_scale, 0, None)
        z_mix = pd.Series(np.log1p(pred), index=z_mix.index)
    return z_mix


def parse_scale_grid(raw: str, fallback: float) -> list[float]:
    if not raw:
        return [fallback]
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def run_cv(args):
    vals = [d.isoformat() for d in CLEAN_VALS[-args.folds:]]
    rows = []
    for val_cutoff in vals:
        train_cutoffs = choose_train_cutoffs(train_cutoffs_for_val(val_cutoff), args.recent_train_cutoffs)
        print(
            f"\nval {val_cutoff}: components={args.components} "
            f"train_cutoffs={len(train_cutoffs)} {train_cutoffs[0]}..{train_cutoffs[-1]}",
            flush=True,
        )
        z_val_raw = ensemble_z(train_cutoffs, val_cutoff, args.recency_weight, 1.0, args.components)
        y_val = build_target(val_cutoff).reindex(z_val_raw.index).fillna(0.0)
        for scale in parse_scale_grid(args.scale_grid, args.global_scale):
            pred_val = np.clip(np.expm1(z_val_raw) * scale, 0, None)
            z_val = pd.Series(np.log1p(pred_val), index=z_val_raw.index)
            row = {
                "val_cutoff": val_cutoff,
                "n_train_cutoffs": len(train_cutoffs),
                "global_scale": scale,
                "rmse": root_mean_squared_error(y_val, pred_val),
                "mae": mean_absolute_error(y_val, pred_val),
                "rmsle": rmsle(y_val, pred_val),
                "bias": float(np.log1p(y_val).mean() - z_val.mean()),
            }
            rows.append(row)
            print(
                f"val {val_cutoff} scale={scale:.4f}: RMSE={row['rmse']:.6f} MAE={row['mae']:.6f} "
                f"RMSLE={row['rmsle']:.6f} bias={row['bias']:+.4f}",
                flush=True,
            )

    print("\nsummary")
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    best = summary.sort_values("rmsle").iloc[0]
    print(f"best scale={best['global_scale']:.4f} RMSLE={best['rmsle']:.6f}")


def run_submit(args):
    train_cutoffs = choose_train_cutoffs(clean_grid(), args.recent_train_cutoffs)
    print(
        f"submit components={args.components} "
        f"train_cutoffs={len(train_cutoffs)} {train_cutoffs[0]}..{train_cutoffs[-1]}",
        flush=True,
    )
    z_test_raw = ensemble_z(train_cutoffs, TEST_CUTOFF, args.recency_weight, 1.0, args.components)
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    scales = parse_scale_grid(args.scale_grid, args.global_scale)
    for scale in scales:
        pred = np.clip(np.expm1(z_test_raw) * scale, 0, None)
        submission = pd.DataFrame({"user_id": z_test_raw.index, "predict": pred})
        out_name = args.output
        if len(scales) > 1:
            suffix = f"scale{int(round(scale * 100)):03d}"
            out_name = f"{Path(args.output).stem}_{suffix}{Path(args.output).suffix}"
        out_path = SUBMISSIONS / out_name
        submission.to_csv(out_path, index=False)
        print(f"saved {out_path} rows={len(submission)} scale={scale:.4f}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ["cv", "submit"]:
        p = sub.add_parser(name)
        p.add_argument("--recency-weight", type=float, default=0.5)
        p.add_argument("--global-scale", type=float, default=1.0)
        p.add_argument("--folds", type=int, default=1)
        p.add_argument("--recent-train-cutoffs", type=int, default=0)
        p.add_argument("--components", choices=["both", "recency", "long_buy"], default="both")
        p.add_argument("--scale-grid", default="")
        p.add_argument("--output", default="exp_011_dense_clean_logens.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.cmd == "cv":
        run_cv(args)
    elif args.cmd == "submit":
        run_submit(args)


if __name__ == "__main__":
    main()
