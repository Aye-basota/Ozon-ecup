"""Evaluate behavior_v1 feature batch for exp_019.

The goal is feature selection, not final ensembling: compare the current
post-order feature set with behavior_v1 and save gain/permutation importance.

Examples:
    python src/behavior_feature_eval.py cv
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

from src.config import SEED
from src.dense_ensemble import fit_predict_z, make_model
from src.features import build_features
from src.train import build_target, rmsle

ARTIFACTS = ROOT / "artifacts"
MAIN_FOLDS = [
    {"fold": 1, "train_cutoff": "2025-12-15", "val_cutoff": "2026-01-14"},
    {"fold": 2, "train_cutoff": "2025-11-15", "val_cutoff": "2025-12-15"},
]


def parse_grid(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def make_xy(cutoff: str, feature_set: str):
    x = build_features(cutoff, feature_set=feature_set)
    y = build_target(cutoff).reindex(x.index).fillna(0.0)
    x = x.replace([np.inf, -np.inf], 0).fillna(0).astype("float32")
    return x, y


def fit_component(feature_set: str, train_cutoff: str, pred_cutoff: str, scale: float):
    x_train, y_train = make_xy(train_cutoff, feature_set)
    x_pred = build_features(pred_cutoff, feature_set=feature_set)
    x_pred = x_pred.replace([np.inf, -np.inf], 0).fillna(0).astype("float32")
    print(
        f"{feature_set} train cutoff {train_cutoff}: X={x_train.shape}, "
        f"target_mean={y_train.mean():.4f}",
        flush=True,
    )

    model = make_model()
    model.fit(x_train, np.log1p(y_train))
    del x_train, y_train
    gc.collect()

    pred = np.clip(np.expm1(model.predict(x_pred)) * scale, 0, None)
    z_pred = pd.Series(np.log1p(pred), index=x_pred.index)
    return z_pred, model, x_pred


def mix_predictions(z_recency: pd.Series, z_second: pd.Series, recency_weight: float, global_scale: float) -> pd.Series:
    z_mix = z_recency * recency_weight + z_second * (1.0 - recency_weight)
    pred = np.clip(np.expm1(z_mix) * global_scale, 0, None)
    return pd.Series(pred, index=z_mix.index)


def gain_importance(model, feature_names: list[str], fold: int) -> pd.DataFrame:
    booster = model.booster_
    return pd.DataFrame(
        {
            "fold": fold,
            "feature": feature_names,
            "gain": booster.feature_importance(importance_type="gain"),
            "split": booster.feature_importance(importance_type="split"),
        }
    ).sort_values(["gain", "split"], ascending=False)


def permutation_importance(
    model,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    z_recency: pd.Series,
    recency_weight: float,
    global_scale: float,
    component_scale: float,
    base_rmsle: float,
    features: list[str],
    fold: int,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(SEED + fold)
    for feature in features:
        x_perm = x_val.copy()
        x_perm[feature] = rng.permutation(x_perm[feature].to_numpy())
        pred_component = np.clip(np.expm1(model.predict(x_perm)) * component_scale, 0, None)
        z_component = pd.Series(np.log1p(pred_component), index=x_val.index)
        pred = mix_predictions(z_recency, z_component, recency_weight, global_scale)
        score = rmsle(y_val, pred)
        rows.append(
            {
                "fold": fold,
                "feature": feature,
                "base_rmsle": base_rmsle,
                "permuted_rmsle": score,
                "delta": score - base_rmsle,
            }
        )
        print(f"perm fold={fold} feature={feature} delta={score - base_rmsle:+.6f}", flush=True)
    return pd.DataFrame(rows).sort_values("delta", ascending=False)


def run_cv(args):
    ARTIFACTS.mkdir(exist_ok=True)
    rows = []
    importance_parts = []
    permutation_parts = []
    weights = parse_grid(args.weight_grid)

    for fold in MAIN_FOLDS:
        fold_id = int(fold["fold"])
        train_cutoff = fold["train_cutoff"]
        val_cutoff = fold["val_cutoff"]
        print(f"\nfold={fold_id} train={train_cutoff} val={val_cutoff}", flush=True)

        z_recency = fit_predict_z("recency", [train_cutoff], val_cutoff, scale=args.recency_scale)
        y_val = build_target(val_cutoff).reindex(z_recency.index).fillna(0.0)

        fold_models = {}
        for variant, feature_set in [("post_order", "long_buy_post_order"), ("behavior_v1", "behavior_v1")]:
            z_second, model, x_val = fit_component(feature_set, train_cutoff, val_cutoff, scale=args.second_scale)
            fold_models[variant] = (z_second, model, x_val)
            for weight in weights:
                pred = mix_predictions(z_recency, z_second, weight, args.global_scale)
                score = rmsle(y_val, pred)
                row = {
                    "fold": fold_id,
                    "variant": variant,
                    "recency_weight": weight,
                    "rmsle": score,
                    "pred_mean": float(pred.mean()),
                    "bias": float(np.log1p(y_val).mean() - np.log1p(pred).mean()),
                }
                rows.append(row)
                print(
                    f"{variant} w_rec={weight:.3f} RMSLE={row['rmsle']:.6f} "
                    f"bias={row['bias']:+.4f}",
                    flush=True,
                )

        z_behavior, behavior_model, x_behavior = fold_models["behavior_v1"]
        best_behavior_row = min(
            [row for row in rows if row["fold"] == fold_id and row["variant"] == "behavior_v1"],
            key=lambda item: item["rmsle"],
        )
        fold_gain = gain_importance(behavior_model, list(x_behavior.columns), fold_id)
        importance_parts.append(fold_gain)
        top_features = fold_gain.head(args.permutation_top)["feature"].tolist()
        permutation_parts.append(
            permutation_importance(
                behavior_model,
                x_behavior,
                y_val,
                z_recency,
                best_behavior_row["recency_weight"],
                args.global_scale,
                args.second_scale,
                best_behavior_row["rmsle"],
                top_features,
                fold_id,
            )
        )
        del fold_models, x_behavior, behavior_model, z_behavior
        gc.collect()

    summary = pd.DataFrame(rows)
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
    importance = pd.concat(importance_parts, axis=0)
    importance_summary = (
        importance.groupby("feature", as_index=False)
        .agg(gain_mean=("gain", "mean"), split_mean=("split", "mean"))
        .sort_values(["gain_mean", "split_mean"], ascending=False)
    )
    permutation = pd.concat(permutation_parts, axis=0)
    permutation_summary = (
        permutation.groupby("feature", as_index=False)
        .agg(delta_mean=("delta", "mean"), delta_min=("delta", "min"), delta_max=("delta", "max"))
        .sort_values("delta_mean", ascending=False)
    )

    summary.to_csv(ARTIFACTS / "exp019_behavior_v1_cv.csv", index=False)
    importance.to_csv(ARTIFACTS / "exp019_behavior_v1_gain_by_fold.csv", index=False)
    importance_summary.to_csv(ARTIFACTS / "exp019_behavior_v1_gain_summary.csv", index=False)
    permutation.to_csv(ARTIFACTS / "exp019_behavior_v1_permutation_by_fold.csv", index=False)
    permutation_summary.to_csv(ARTIFACTS / "exp019_behavior_v1_permutation_summary.csv", index=False)

    print("\nby variant/weight")
    print(grouped.to_string(index=False))
    print("\ntop gain")
    print(importance_summary.head(30).to_string(index=False))
    print("\ntop permutation")
    print(permutation_summary.head(30).to_string(index=False))
    print("\nbest", grouped.iloc[0].to_dict())


def parse_args():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("cv")
    p.add_argument("--weight-grid", default="0.5")
    p.add_argument("--global-scale", type=float, default=1.2)
    p.add_argument("--recency-scale", type=float, default=0.64)
    p.add_argument("--second-scale", type=float, default=0.62)
    p.add_argument("--permutation-top", type=int, default=30)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.cmd == "cv":
        run_cv(args)


if __name__ == "__main__":
    main()
