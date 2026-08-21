"""CatBoost blend diagnostic for exp_018.

Adds a CPU CatBoost regressor as a third, low-weight component on top of the
current exp_017 recency + post-order dist-head ensemble.

Examples:
    python src/catboost_blend.py cv
    python src/catboost_blend.py submit --output exp_018_catboost_blend.csv
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
from src.dist_head_ensemble import MAIN_FOLDS, fit_predict_dist_z
from src.features import build_features
from src.train import TEST_CUTOFF, build_target, rmsle


def parse_grid(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def make_xy(cutoff: str, feature_set: str):
    x = build_features(cutoff, feature_set=feature_set)
    y = build_target(cutoff).reindex(x.index).fillna(0.0)
    x = x.replace([np.inf, -np.inf], 0).fillna(0).astype("float32")
    return x, y


def make_catboost(args):
    from catboost import CatBoostRegressor

    return CatBoostRegressor(
        loss_function="RMSE",
        iterations=args.cat_iterations,
        learning_rate=args.cat_learning_rate,
        depth=args.cat_depth,
        l2_leaf_reg=args.cat_l2,
        random_seed=SEED,
        task_type="CPU",
        thread_count=-1,
        allow_writing_files=False,
        verbose=100,
    )


def fit_predict_cat_z(feature_set: str, train_cutoffs: list[str], pred_cutoff: str, scale: float, args) -> pd.Series:
    x_parts = []
    y_parts = []
    for cutoff in train_cutoffs:
        x_cutoff, y_cutoff = make_xy(cutoff, feature_set)
        x_parts.append(x_cutoff)
        y_parts.append(y_cutoff)
        print(f"cat {feature_set} cutoff {cutoff}: X={x_cutoff.shape}, target_mean={y_cutoff.mean():.4f}", flush=True)

    x_train = pd.concat(x_parts, axis=0)
    y_train = pd.concat(y_parts, axis=0)
    del x_parts, y_parts
    gc.collect()

    model = make_catboost(args)
    model.fit(x_train, np.log1p(y_train))
    del x_train, y_train
    gc.collect()

    x_pred = build_features(pred_cutoff, feature_set=feature_set)
    x_pred = x_pred.replace([np.inf, -np.inf], 0).fillna(0).astype("float32")
    pred = np.clip(np.expm1(model.predict(x_pred)) * scale, 0, None)
    return pd.Series(np.log1p(pred), index=x_pred.index)


def mix_three(
    z_recency: pd.Series,
    z_dist: pd.Series,
    z_cat: pd.Series,
    recency_weight: float,
    cat_weight: float,
    global_scale: float,
) -> pd.Series:
    dist_weight = 1.0 - recency_weight - cat_weight
    if dist_weight < 0:
        raise ValueError("recency_weight + cat_weight must be <= 1")
    z_mix = z_recency * recency_weight + z_dist * dist_weight + z_cat * cat_weight
    pred = np.clip(np.expm1(z_mix) * global_scale, 0, None)
    return pd.Series(pred, index=z_mix.index)


def run_cv(args):
    rows = []
    cat_weights = parse_grid(args.cat_weight_grid)
    for fold in MAIN_FOLDS:
        train_cutoffs = [fold["train_cutoff"]]
        val_cutoff = fold["val_cutoff"]
        print(f"\nfold={fold['fold']} train={train_cutoffs[0]} val={val_cutoff}", flush=True)

        z_recency = fit_predict_z("recency", train_cutoffs, val_cutoff, scale=args.recency_scale)
        z_dist = fit_predict_dist_z("long_buy_post_order", train_cutoffs, val_cutoff, scale=args.dist_scale, args=args)
        z_cat = fit_predict_cat_z("long_buy_post_order", train_cutoffs, val_cutoff, scale=args.cat_scale, args=args)
        y_val = build_target(val_cutoff).reindex(z_recency.index).fillna(0.0)

        for cat_weight in cat_weights:
            pred = mix_three(z_recency, z_dist, z_cat, args.recency_weight, cat_weight, args.global_scale)
            row = {
                "fold": fold["fold"],
                "recency_weight": args.recency_weight,
                "dist_weight": 1.0 - args.recency_weight - cat_weight,
                "cat_weight": cat_weight,
                "rmsle": rmsle(y_val, pred),
                "pred_mean": float(pred.mean()),
                "bias": float(np.log1p(y_val).mean() - np.log1p(pred).mean()),
            }
            rows.append(row)
            print(
                f"cat_weight={cat_weight:.3f} dist_weight={row['dist_weight']:.3f} "
                f"RMSLE={row['rmsle']:.6f} bias={row['bias']:+.4f}",
                flush=True,
            )

    summary = pd.DataFrame(rows)
    grouped = (
        summary.groupby(["recency_weight", "dist_weight", "cat_weight"], as_index=False)
        .agg(
            rmsle_mean=("rmsle", "mean"),
            rmsle_std=("rmsle", "std"),
            bias_mean=("bias", "mean"),
            pred_mean=("pred_mean", "mean"),
        )
        .sort_values("rmsle_mean")
    )
    print("\nby weights")
    print(grouped.to_string(index=False))
    print("\nbest", grouped.iloc[0].to_dict())


def run_submit(args):
    train_cutoffs = clean_grid()[-args.recent_train_cutoffs :]
    print(f"submit train_cutoffs={len(train_cutoffs)} {train_cutoffs[0]}..{train_cutoffs[-1]}", flush=True)
    z_recency = fit_predict_z("recency", train_cutoffs, TEST_CUTOFF, scale=args.recency_scale)
    z_dist = fit_predict_dist_z("long_buy_post_order", train_cutoffs, TEST_CUTOFF, scale=args.dist_scale, args=args)
    z_cat = fit_predict_cat_z("long_buy_post_order", train_cutoffs, TEST_CUTOFF, scale=args.cat_scale, args=args)
    pred = mix_three(z_recency, z_dist, z_cat, args.recency_weight, args.cat_weight, args.global_scale)

    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    out_path = SUBMISSIONS / args.output
    pd.DataFrame({"user_id": pred.index, "predict": pred}).to_csv(out_path, index=False)
    print(
        f"saved {out_path} rows={len(pred)} recency_weight={args.recency_weight:.3f} "
        f"dist_weight={1.0 - args.recency_weight - args.cat_weight:.3f} "
        f"cat_weight={args.cat_weight:.3f} scale={args.global_scale:.3f}",
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ["cv", "submit"]:
        p = sub.add_parser(name)
        p.add_argument("--recency-weight", type=float, default=0.5)
        p.add_argument("--cat-weight", type=float, default=0.1)
        p.add_argument("--cat-weight-grid", default="0.00,0.05,0.10,0.15,0.20")
        p.add_argument("--global-scale", type=float, default=1.2)
        p.add_argument("--recent-train-cutoffs", type=int, default=8)
        p.add_argument("--recency-scale", type=float, default=0.64)
        p.add_argument("--dist-scale", type=float, default=0.62)
        p.add_argument("--cat-scale", type=float, default=0.62)
        p.add_argument("--bins", type=int, default=16)
        p.add_argument("--rounds", type=int, default=250)
        p.add_argument("--learning-rate", type=float, default=0.05)
        p.add_argument("--num-leaves", type=int, default=31)
        p.add_argument("--cat-iterations", type=int, default=800)
        p.add_argument("--cat-learning-rate", type=float, default=0.05)
        p.add_argument("--cat-depth", type=int, default=6)
        p.add_argument("--cat-l2", type=float, default=3.0)
        p.add_argument("--output", default="exp_018_catboost_blend.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.cmd == "cv":
        run_cv(args)
    elif args.cmd == "submit":
        run_submit(args)


if __name__ == "__main__":
    main()
