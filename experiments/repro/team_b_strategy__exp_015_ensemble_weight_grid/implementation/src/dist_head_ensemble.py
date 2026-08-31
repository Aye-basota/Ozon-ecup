"""Distribution-head ensemble for exp_017.

This transfers the strongest tabular idea from team-a: predict a distribution
over z=log1p(y) bins and use its expectation instead of direct L2 regression.

Examples:
    python src/dist_head_ensemble.py cv
    python src/dist_head_ensemble.py submit --output exp_017_dist_post_order.csv
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
from src.dense_ensemble import clean_grid, fit_predict_z
from src.features import build_features
from src.train import TEST_CUTOFF, build_target, rmsle

MAIN_FOLDS = [
    {"fold": 1, "train_cutoff": "2025-12-15", "val_cutoff": "2026-01-14"},
    {"fold": 2, "train_cutoff": "2025-11-15", "val_cutoff": "2025-12-15"},
]

ZERO_EDGE = 1e-9


def parse_grid(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def z_bin_edges(z: np.ndarray, n_bins: int) -> np.ndarray:
    positive = z[z > 0]
    quantiles = np.quantile(positive, np.linspace(0, 1, n_bins)[1:-1])
    return np.concatenate([[ZERO_EDGE], quantiles])


def bin_labels(z: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges, z, side="right").astype("int32")


def bin_centroids(z: np.ndarray, labels: np.ndarray, n_bins: int) -> np.ndarray:
    counts = np.bincount(labels, minlength=n_bins)
    totals = np.bincount(labels, weights=z, minlength=n_bins)
    centroids = np.zeros(n_bins, dtype="float64")
    last = 0.0
    for idx in range(n_bins):
        if counts[idx] > 0:
            last = totals[idx] / counts[idx]
        centroids[idx] = last
    return centroids


def make_xy(cutoff: str, feature_set: str):
    x = build_features(cutoff, feature_set=feature_set)
    y = build_target(cutoff).reindex(x.index).fillna(0.0)
    x = x.replace([np.inf, -np.inf], 0).fillna(0).astype("float32")
    return x, y


def make_dist_model(args, n_classes: int):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        objective="multiclass",
        num_class=n_classes,
        n_estimators=args.rounds,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=0.05,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )


def fit_predict_dist_z(
    feature_set: str,
    train_cutoffs: list[str],
    pred_cutoff: str,
    scale: float,
    args,
) -> pd.Series:
    x_parts = []
    y_parts = []
    for cutoff in train_cutoffs:
        x_cutoff, y_cutoff = make_xy(cutoff, feature_set)
        x_parts.append(x_cutoff)
        y_parts.append(y_cutoff)
        print(f"dist {feature_set} cutoff {cutoff}: X={x_cutoff.shape}, target_mean={y_cutoff.mean():.4f}", flush=True)

    x_train = pd.concat(x_parts, axis=0)
    y_train = pd.concat(y_parts, axis=0)
    del x_parts, y_parts
    gc.collect()

    z_train = np.log1p(y_train.to_numpy())
    edges = z_bin_edges(z_train, args.bins)
    labels = bin_labels(z_train, edges)
    centroids = bin_centroids(z_train, labels, args.bins)
    print(
        f"dist bins={args.bins}, positive_rate={(z_train > 0).mean():.4f}, "
        f"centroids[0]={centroids[0]:.4f}, centroids[-1]={centroids[-1]:.4f}",
        flush=True,
    )

    model = make_dist_model(args, args.bins)
    model.fit(x_train, labels)
    del x_train, y_train, z_train, labels
    gc.collect()

    x_pred = build_features(pred_cutoff, feature_set=feature_set)
    x_pred = x_pred.replace([np.inf, -np.inf], 0).fillna(0).astype("float32")
    proba = model.predict_proba(x_pred)
    z_pred = proba @ centroids
    pred = np.clip(np.expm1(z_pred) * scale, 0, None)
    return pd.Series(np.log1p(pred), index=x_pred.index)


def mix_predictions(z_recency: pd.Series, z_second: pd.Series, recency_weight: float, global_scale: float) -> pd.Series:
    z_mix = z_recency * recency_weight + z_second * (1.0 - recency_weight)
    pred = np.clip(np.expm1(z_mix) * global_scale, 0, None)
    return pd.Series(pred, index=z_mix.index)


def run_cv(args):
    rows = []
    weights = parse_grid(args.weight_grid)
    for fold in MAIN_FOLDS:
        train_cutoffs = [fold["train_cutoff"]]
        val_cutoff = fold["val_cutoff"]
        print(f"\nfold={fold['fold']} train={train_cutoffs[0]} val={val_cutoff}", flush=True)
        z_recency = fit_predict_z("recency", train_cutoffs, val_cutoff, scale=0.64)
        z_dist = fit_predict_dist_z("long_buy_post_order", train_cutoffs, val_cutoff, scale=0.62, args=args)
        y_val = build_target(val_cutoff).reindex(z_recency.index).fillna(0.0)
        for weight in weights:
            pred = mix_predictions(z_recency, z_dist, weight, args.global_scale)
            row = {
                "fold": fold["fold"],
                "recency_weight": weight,
                "rmsle": rmsle(y_val, pred),
                "pred_mean": float(pred.mean()),
                "bias": float(np.log1p(y_val).mean() - np.log1p(pred).mean()),
            }
            rows.append(row)
            print(
                f"dist_post_order w_rec={weight:.3f} RMSLE={row['rmsle']:.6f} "
                f"bias={row['bias']:+.4f}",
                flush=True,
            )

    summary = pd.DataFrame(rows)
    grouped = (
        summary.groupby("recency_weight", as_index=False)
        .agg(
            rmsle_mean=("rmsle", "mean"),
            rmsle_std=("rmsle", "std"),
            bias_mean=("bias", "mean"),
            pred_mean=("pred_mean", "mean"),
        )
        .sort_values("rmsle_mean")
    )
    print("\nby recency_weight")
    print(grouped.to_string(index=False))
    print("\nbest", grouped.iloc[0].to_dict())


def run_submit(args):
    train_cutoffs = clean_grid()[-args.recent_train_cutoffs :]
    print(f"submit train_cutoffs={len(train_cutoffs)} {train_cutoffs[0]}..{train_cutoffs[-1]}", flush=True)
    z_recency = fit_predict_z("recency", train_cutoffs, TEST_CUTOFF, scale=0.64)
    z_dist = fit_predict_dist_z("long_buy_post_order", train_cutoffs, TEST_CUTOFF, scale=0.62, args=args)
    pred = mix_predictions(z_recency, z_dist, args.recency_weight, args.global_scale)

    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    out_path = SUBMISSIONS / args.output
    pd.DataFrame({"user_id": pred.index, "predict": pred}).to_csv(out_path, index=False)
    print(
        f"saved {out_path} rows={len(pred)} recency_weight={args.recency_weight:.3f} "
        f"scale={args.global_scale:.3f}",
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ["cv", "submit"]:
        p = sub.add_parser(name)
        p.add_argument("--weight-grid", default="0.4,0.5,0.6")
        p.add_argument("--recency-weight", type=float, default=0.5)
        p.add_argument("--global-scale", type=float, default=1.2)
        p.add_argument("--recent-train-cutoffs", type=int, default=8)
        p.add_argument("--bins", type=int, default=16)
        p.add_argument("--rounds", type=int, default=250)
        p.add_argument("--learning-rate", type=float, default=0.05)
        p.add_argument("--num-leaves", type=int, default=31)
        p.add_argument("--output", default="exp_017_dist_post_order.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.cmd == "cv":
        run_cv(args)
    elif args.cmd == "submit":
        run_submit(args)


if __name__ == "__main__":
    main()
