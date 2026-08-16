"""Post-order feature experiment for exp_016.

Compare the current recency + long_buy ensemble against recency +
long_buy_post_order, keeping the same validation folds and scale.

Examples:
    python src/post_order_ensemble.py cv
    python src/post_order_ensemble.py submit --output exp_016_post_order.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import SUBMISSIONS
from src.dense_ensemble import clean_grid, fit_predict_z
from src.train import TEST_CUTOFF, build_target, rmsle

MAIN_FOLDS = [
    {"fold": 1, "train_cutoff": "2025-12-15", "val_cutoff": "2026-01-14"},
    {"fold": 2, "train_cutoff": "2025-11-15", "val_cutoff": "2025-12-15"},
]


def parse_grid(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def mix_predictions(z_recency: pd.Series, z_second: pd.Series, recency_weight: float, global_scale: float) -> pd.Series:
    z_mix = z_recency * recency_weight + z_second * (1.0 - recency_weight)
    pred = np.clip(np.expm1(z_mix) * global_scale, 0, None)
    return pd.Series(pred, index=z_mix.index)


def run_cv(args):
    rows = []
    weights = parse_grid(args.weight_grid)
    variants = {
        "baseline_long_buy": "long_buy",
        "post_order": "long_buy_post_order",
    }
    for fold in MAIN_FOLDS:
        train_cutoffs = [fold["train_cutoff"]]
        val_cutoff = fold["val_cutoff"]
        print(f"\nfold={fold['fold']} train={train_cutoffs[0]} val={val_cutoff}", flush=True)
        z_recency = fit_predict_z("recency", train_cutoffs, val_cutoff, scale=0.64)
        z_by_variant = {
            name: fit_predict_z(feature_set, train_cutoffs, val_cutoff, scale=0.62)
            for name, feature_set in variants.items()
        }
        y_val = build_target(val_cutoff).reindex(z_recency.index).fillna(0.0)
        for variant, z_second in z_by_variant.items():
            for weight in weights:
                pred = mix_predictions(z_recency, z_second, weight, args.global_scale)
                row = {
                    "fold": fold["fold"],
                    "variant": variant,
                    "recency_weight": weight,
                    "rmsle": rmsle(y_val, pred),
                    "pred_mean": float(pred.mean()),
                    "bias": float(np.log1p(y_val).mean() - np.log1p(pred).mean()),
                }
                rows.append(row)
                print(
                    f"{variant} w_rec={weight:.3f} RMSLE={row['rmsle']:.6f} "
                    f"bias={row['bias']:+.4f}",
                    flush=True,
                )

    summary = pd.DataFrame(rows)
    print("\nby variant/weight")
    grouped = (
        summary.groupby(["variant", "recency_weight"], as_index=False)
        .agg(
            rmsle_mean=("rmsle", "mean"),
            rmsle_std=("rmsle", "std"),
            bias_mean=("bias", "mean"),
            pred_mean=("pred_mean", "mean"),
        )
        .sort_values("rmsle_mean")
    )
    print(grouped.to_string(index=False))
    print("\nbest", grouped.iloc[0].to_dict())


def run_submit(args):
    train_cutoffs = clean_grid()[-args.recent_train_cutoffs :]
    print(f"submit train_cutoffs={len(train_cutoffs)} {train_cutoffs[0]}..{train_cutoffs[-1]}", flush=True)
    z_recency = fit_predict_z("recency", train_cutoffs, TEST_CUTOFF, scale=0.64)
    z_post = fit_predict_z("long_buy_post_order", train_cutoffs, TEST_CUTOFF, scale=0.62)
    pred = mix_predictions(z_recency, z_post, args.recency_weight, args.global_scale)

    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    out_path = SUBMISSIONS / args.output
    pd.DataFrame({"user_id": pred.index, "predict": pred}).to_csv(out_path, index=False)
    print(
        f"saved {out_path} rows={len(pred)} "
        f"recency_weight={args.recency_weight:.3f} scale={args.global_scale:.3f}",
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
        p.add_argument("--output", default="exp_016_post_order.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.cmd == "cv":
        run_cv(args)
    elif args.cmd == "submit":
        run_submit(args)


if __name__ == "__main__":
    main()
