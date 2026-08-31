"""Classifier gate for exp_014.

The regressor predicts expected GMV. This script adds a separate classifier for
P(target > 0) and optionally zeroes low-probability users before submission.

Examples:
    python src/classifier_gate.py cv --threshold-grid 0.05,0.10,0.15
    python src/classifier_gate.py submit --threshold 0.10 --output exp_014_gate_thr010.csv
"""
import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import SEED, SUBMISSIONS
from src.dense_ensemble import clean_grid, ensemble_z
from src.features import build_features
from src.train import TEST_CUTOFF, build_target, rmsle

MAIN_FOLDS = [
    {"fold": 1, "train_cutoff": "2025-12-15", "val_cutoff": "2026-01-14"},
    {"fold": 2, "train_cutoff": "2025-11-15", "val_cutoff": "2025-12-15"},
]


def make_classifier():
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        objective="binary",
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=0.05,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )


def make_classifier_xy(cutoff: str, feature_set: str):
    x = build_features(cutoff, feature_set=feature_set)
    y = (build_target(cutoff).reindex(x.index).fillna(0.0) > 0).astype("int8")
    x = x.replace([np.inf, -np.inf], 0).fillna(0).astype("float32")
    return x, y


def fit_predict_proba(train_cutoffs: list[str], pred_cutoff: str, feature_set: str) -> pd.Series:
    x_parts = []
    y_parts = []
    for cutoff in train_cutoffs:
        x_cutoff, y_cutoff = make_classifier_xy(cutoff, feature_set)
        x_parts.append(x_cutoff)
        y_parts.append(y_cutoff)
        print(
            f"classifier cutoff {cutoff}: X={x_cutoff.shape}, "
            f"positive_rate={y_cutoff.mean():.4f}",
            flush=True,
        )

    x_train = pd.concat(x_parts, axis=0)
    y_train = pd.concat(y_parts, axis=0)
    del x_parts, y_parts
    gc.collect()

    model = make_classifier()
    model.fit(x_train, y_train)
    del x_train, y_train
    gc.collect()

    x_pred = build_features(pred_cutoff, feature_set=feature_set)
    x_pred = x_pred.replace([np.inf, -np.inf], 0).fillna(0).astype("float32")
    proba = model.predict_proba(x_pred)[:, 1]
    return pd.Series(proba, index=x_pred.index)


def parse_grid(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def apply_gate(base_pred: pd.Series, proba: pd.Series, threshold: float) -> pd.Series:
    gated = base_pred.copy()
    gated.loc[proba.reindex(gated.index) < threshold] = 0.0
    return gated


def run_cv(args):
    rows = []
    thresholds = parse_grid(args.threshold_grid)
    for fold in MAIN_FOLDS:
        train_cutoffs = [fold["train_cutoff"]]
        val_cutoff = fold["val_cutoff"]
        print(f"\nfold={fold['fold']} train={train_cutoffs[0]} val={val_cutoff}", flush=True)
        z_base = ensemble_z(train_cutoffs, val_cutoff, args.recency_weight, args.global_scale, args.components)
        base_pred = pd.Series(np.expm1(z_base), index=z_base.index)
        proba = fit_predict_proba(train_cutoffs, val_cutoff, args.classifier_feature_set)
        y_val = build_target(val_cutoff).reindex(base_pred.index).fillna(0.0)

        rows.append(
            {
                "fold": fold["fold"],
                "threshold": -1.0,
                "rmsle": rmsle(y_val, base_pred),
                "zero_share": float((base_pred == 0).mean()),
                "false_zero_rate": 0.0,
            }
        )
        print(f"base RMSLE={rows[-1]['rmsle']:.6f}", flush=True)

        positives = y_val > 0
        for threshold in thresholds:
            pred = apply_gate(base_pred, proba, threshold)
            zeroed = pred == 0
            row = {
                "fold": fold["fold"],
                "threshold": threshold,
                "rmsle": rmsle(y_val, pred),
                "zero_share": float(zeroed.mean()),
                "false_zero_rate": float((zeroed & positives).sum() / positives.sum()),
            }
            rows.append(row)
            print(
                f"thr={threshold:.3f} RMSLE={row['rmsle']:.6f} "
                f"zero_share={row['zero_share']:.3f} false_zero_rate={row['false_zero_rate']:.3f}",
                flush=True,
            )

    summary = pd.DataFrame(rows)
    print("\nby threshold")
    print(
        summary.groupby("threshold", as_index=False)
        .agg(
            rmsle_mean=("rmsle", "mean"),
            rmsle_std=("rmsle", "std"),
            zero_share_mean=("zero_share", "mean"),
            false_zero_rate_mean=("false_zero_rate", "mean"),
        )
        .sort_values("rmsle_mean")
        .to_string(index=False)
    )


def run_submit(args):
    train_cutoffs = clean_grid()[-args.recent_train_cutoffs :]
    print(f"submit train_cutoffs={len(train_cutoffs)} {train_cutoffs[0]}..{train_cutoffs[-1]}", flush=True)
    z_base = ensemble_z(train_cutoffs, TEST_CUTOFF, args.recency_weight, args.global_scale, args.components)
    base_pred = pd.Series(np.expm1(z_base), index=z_base.index)
    proba = fit_predict_proba(train_cutoffs, TEST_CUTOFF, args.classifier_feature_set)
    pred = apply_gate(base_pred, proba, args.threshold)

    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    out_path = SUBMISSIONS / args.output
    pd.DataFrame({"user_id": pred.index, "predict": pred}).to_csv(out_path, index=False)
    print(
        f"saved {out_path} rows={len(pred)} threshold={args.threshold:.4f} "
        f"zero_share={(pred == 0).mean():.4f}",
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ["cv", "submit"]:
        p = sub.add_parser(name)
        p.add_argument("--components", choices=["both", "recency", "long_buy"], default="both")
        p.add_argument("--recency-weight", type=float, default=0.5)
        p.add_argument("--global-scale", type=float, default=1.2)
        p.add_argument("--classifier-feature-set", choices=["recency", "long_buy"], default="long_buy")
        p.add_argument("--threshold-grid", default="0.02,0.05,0.08,0.10,0.12,0.15,0.20")
        p.add_argument("--threshold", type=float, default=0.10)
        p.add_argument("--recent-train-cutoffs", type=int, default=8)
        p.add_argument("--output", default="exp_014_classifier_gate.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.cmd == "cv":
        run_cv(args)
    elif args.cmd == "submit":
        run_submit(args)


if __name__ == "__main__":
    main()
