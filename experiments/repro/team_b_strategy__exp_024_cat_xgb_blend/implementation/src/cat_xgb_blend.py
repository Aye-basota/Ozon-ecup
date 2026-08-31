"""Five-component blend with XGBoost and CatBoost behavior models.

This extends exp023 by adding CatBoost as another model-family component on
`behavior_v1` features. All components are blended in log1p space.

Examples:
    python src/cat_xgb_blend.py cv --weight-step 0.05
    python src/cat_xgb_blend.py submit --weights 0.25 0.10 0.25 0.25 0.15 \
        --output exp_024_cat_xgb_blend.csv
"""
from __future__ import annotations

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
from src.xgboost_blend import fit_predict_xgb_z


COMPONENTS = ["recency", "post_order_dist", "behavior_dist", "xgb_behavior", "cat_behavior"]


def weight_grid(n: int, step: float) -> list[np.ndarray]:
    units = int(round(1.0 / step))
    out = []

    def rec(prefix: list[int], left: int, k: int) -> None:
        if k == 1:
            out.append(np.asarray(prefix + [left], dtype=float) / units)
            return
        for v in range(left + 1):
            rec(prefix + [v], left - v, k - 1)

    rec([], units, n)
    return out


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


def component_predictions(train_cutoffs: list[str], pred_cutoff: str, args) -> dict[str, pd.Series]:
    return {
        "recency": fit_predict_z("recency", train_cutoffs, pred_cutoff, scale=args.recency_scale),
        "post_order_dist": fit_predict_dist_z(
            "long_buy_post_order", train_cutoffs, pred_cutoff, scale=args.dist_scale, args=args
        ),
        "behavior_dist": fit_predict_dist_z(
            "behavior_v1", train_cutoffs, pred_cutoff, scale=args.dist_scale, args=args
        ),
        "xgb_behavior": fit_predict_xgb_z(args.xgb_feature_set, train_cutoffs, pred_cutoff, args.xgb_scale, args),
        "cat_behavior": fit_predict_cat_z(args.cat_feature_set, train_cutoffs, pred_cutoff, args.cat_scale, args),
    }


def mix_z(preds: dict[str, pd.Series], weights: np.ndarray, global_scale: float) -> pd.Series:
    z = sum(preds[name] * w for name, w in zip(COMPONENTS, weights))
    if global_scale != 1.0:
        pred = np.clip(np.expm1(z) * global_scale, 0, None)
        z = pd.Series(np.log1p(pred), index=z.index)
    return z


def run_cv(args) -> None:
    rows = []
    ws = weight_grid(len(COMPONENTS), args.weight_step)
    ws = [
        w for w in ws
        if w[2] >= args.min_behavior_weight
        and w[3] <= args.max_xgb_weight
        and w[4] <= args.max_cat_weight
    ]
    print(f"weight candidates={len(ws)} components={COMPONENTS}", flush=True)

    for fold in MAIN_FOLDS:
        train_cutoffs = [fold["train_cutoff"]]
        val_cutoff = fold["val_cutoff"]
        print(f"\nfold={fold['fold']} train={train_cutoffs[0]} val={val_cutoff}", flush=True)
        preds = component_predictions(train_cutoffs, val_cutoff, args)
        y_val = build_target(val_cutoff).reindex(preds["recency"].index).fillna(0.0)

        for w in ws:
            z = mix_z(preds, w, args.global_scale)
            pred = np.clip(np.expm1(z), 0, None)
            rows.append({
                "fold": fold["fold"],
                "recency_weight": w[0],
                "post_order_dist_weight": w[1],
                "behavior_dist_weight": w[2],
                "xgb_behavior_weight": w[3],
                "cat_behavior_weight": w[4],
                "rmsle": rmsle(y_val, pred),
                "bias": float(np.log1p(y_val).mean() - z.mean()),
                "pred_log_mean": float(z.mean()),
                "pred_mean": float(pred.mean()),
            })

        best_fold = pd.DataFrame([r for r in rows if r["fold"] == fold["fold"]]).sort_values("rmsle").iloc[0]
        print(
            "best fold "
            f"w=({best_fold.recency_weight:.2f}, {best_fold.post_order_dist_weight:.2f}, "
            f"{best_fold.behavior_dist_weight:.2f}, {best_fold.xgb_behavior_weight:.2f}, "
            f"{best_fold.cat_behavior_weight:.2f}) RMSLE={best_fold.rmsle:.6f} "
            f"bias={best_fold.bias:+.4f}",
            flush=True,
        )

    summary = pd.DataFrame(rows)
    grouped = (
        summary.groupby(
            [
                "recency_weight",
                "post_order_dist_weight",
                "behavior_dist_weight",
                "xgb_behavior_weight",
                "cat_behavior_weight",
            ],
            as_index=False,
        )
        .agg(
            rmsle_mean=("rmsle", "mean"),
            rmsle_std=("rmsle", "std"),
            bias_mean=("bias", "mean"),
            pred_log_mean=("pred_log_mean", "mean"),
            pred_mean=("pred_mean", "mean"),
        )
        .sort_values("rmsle_mean")
    )
    print("\nby weights")
    print(grouped.head(args.top).to_string(index=False))
    out = Path("artifacts") / "exp024_cat_xgb_blend_grid.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(out, index=False)
    print(f"\nsaved {out}", flush=True)


def run_submit(args) -> None:
    if len(args.weights) != len(COMPONENTS):
        raise ValueError(f"--weights must have {len(COMPONENTS)} values: {COMPONENTS}")
    w = np.asarray(args.weights, dtype=float)
    if (w < 0).any():
        raise ValueError("--weights must be non-negative")
    w = w / w.sum()
    train_cutoffs = clean_grid()[-args.recent_train_cutoffs :]
    print(
        f"submit train_cutoffs={len(train_cutoffs)} {train_cutoffs[0]}..{train_cutoffs[-1]} "
        f"weights={dict(zip(COMPONENTS, np.round(w, 3)))}",
        flush=True,
    )
    preds = component_predictions(train_cutoffs, TEST_CUTOFF, args)
    z = mix_z(preds, w, args.global_scale)
    pred = np.clip(np.expm1(z), 0, None)
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    out_path = SUBMISSIONS / args.output
    pd.DataFrame({"user_id": z.index, "predict": pred}).to_csv(out_path, index=False)
    print(
        f"saved {out_path} rows={len(pred)} mean_log1p={z.mean():.6f} mean_pred={pred.mean():.6f}",
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ["cv", "submit"]:
        p = sub.add_parser(name)
        p.add_argument("--global-scale", type=float, default=1.2)
        p.add_argument("--recent-train-cutoffs", type=int, default=8)
        p.add_argument("--recency-scale", type=float, default=0.64)
        p.add_argument("--dist-scale", type=float, default=0.62)
        p.add_argument("--bins", type=int, default=16)
        p.add_argument("--rounds", type=int, default=250)
        p.add_argument("--learning-rate", type=float, default=0.05)
        p.add_argument("--num-leaves", type=int, default=31)
        p.add_argument("--xgb-feature-set", default="behavior_v1")
        p.add_argument("--xgb-scale", type=float, default=0.62)
        p.add_argument("--xgb-rounds", type=int, default=450)
        p.add_argument("--xgb-learning-rate", type=float, default=0.03)
        p.add_argument("--xgb-max-depth", type=int, default=6)
        p.add_argument("--xgb-min-child-weight", type=float, default=20.0)
        p.add_argument("--xgb-l2", type=float, default=5.0)
        p.add_argument("--cat-feature-set", default="behavior_v1")
        p.add_argument("--cat-scale", type=float, default=0.62)
        p.add_argument("--cat-iterations", type=int, default=500)
        p.add_argument("--cat-learning-rate", type=float, default=0.05)
        p.add_argument("--cat-depth", type=int, default=6)
        p.add_argument("--cat-l2", type=float, default=3.0)
    sub.choices["cv"].add_argument("--weight-step", type=float, default=0.05)
    sub.choices["cv"].add_argument("--min-behavior-weight", type=float, default=0.20)
    sub.choices["cv"].add_argument("--max-xgb-weight", type=float, default=0.30)
    sub.choices["cv"].add_argument("--max-cat-weight", type=float, default=0.20)
    sub.choices["cv"].add_argument("--top", type=int, default=20)
    sub.choices["submit"].add_argument("--weights", nargs="+", type=float, required=True)
    sub.choices["submit"].add_argument("--output", default="exp_024_cat_xgb_blend.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "cv":
        run_cv(args)
    elif args.cmd == "submit":
        run_submit(args)


if __name__ == "__main__":
    main()
