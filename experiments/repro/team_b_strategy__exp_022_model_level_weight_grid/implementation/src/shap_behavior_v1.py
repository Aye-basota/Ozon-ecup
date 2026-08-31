"""Native LightGBM SHAP-style contributions for full behavior_v1.

This is an explanation utility for exp_019. It trains the direct LightGBM
component on the same main validation folds and uses LightGBM's native
`pred_contrib=True` output to summarize feature contributions on validation.

Run:
    python src/shap_behavior_v1.py
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
from src.features import build_features
from src.train import build_target

ARTIFACTS = ROOT / "artifacts"
MAIN_FOLDS = [
    {"fold": 1, "train_cutoff": "2025-12-15", "val_cutoff": "2026-01-14"},
    {"fold": 2, "train_cutoff": "2025-11-15", "val_cutoff": "2025-12-15"},
]


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


def sample_rows(x: pd.DataFrame, y: pd.Series, sample_size: int, fold: int) -> tuple[pd.DataFrame, pd.Series]:
    if sample_size <= 0 or len(x) <= sample_size:
        return x, y
    rng = np.random.default_rng(SEED + fold)
    idx = rng.choice(len(x), size=sample_size, replace=False)
    return x.iloc[idx].copy(), y.iloc[idx].copy()


def summarize_contrib(contrib: np.ndarray, x: pd.DataFrame, fold: int) -> pd.DataFrame:
    # Last column is the expected-value/base contribution.
    values = contrib[:, :-1]
    feature_values = x.to_numpy(dtype="float64", copy=False)
    rows = pd.DataFrame(
        {
            "fold": fold,
            "feature": x.columns,
            "mean_abs_shap": np.mean(np.abs(values), axis=0),
            "mean_shap": np.mean(values, axis=0),
            "mean_value": np.mean(feature_values, axis=0),
            "std_value": np.std(feature_values, axis=0),
        }
    )
    rows["is_b1"] = rows["feature"].str.startswith("b1_")
    return rows.sort_values("mean_abs_shap", ascending=False)


def run(args):
    ARTIFACTS.mkdir(exist_ok=True)
    parts = []
    for fold_cfg in MAIN_FOLDS:
        fold = int(fold_cfg["fold"])
        train_cutoff = fold_cfg["train_cutoff"]
        val_cutoff = fold_cfg["val_cutoff"]
        print(f"\nfold={fold} train={train_cutoff} val={val_cutoff}", flush=True)

        x_train, y_train = make_xy(train_cutoff, args.feature_set)
        x_val, y_val = make_xy(val_cutoff, args.feature_set)
        x_sample, y_sample = sample_rows(x_val, y_val, args.sample_size, fold)
        print(
            f"feature_set={args.feature_set} train={x_train.shape} "
            f"val={x_val.shape} sample={x_sample.shape}",
            flush=True,
        )

        model = make_model()
        model.fit(x_train, np.log1p(y_train))
        del x_train, y_train, x_val, y_val
        gc.collect()

        contrib = model.booster_.predict(x_sample, pred_contrib=True)
        if contrib.ndim != 2:
            raise RuntimeError(f"Expected 2D contrib array, got shape={contrib.shape}")
        fold_summary = summarize_contrib(contrib, x_sample, fold)
        parts.append(fold_summary)
        fold_path = ARTIFACTS / f"exp019_behavior_v1_shap_fold{fold}.csv"
        fold_summary.to_csv(fold_path, index=False)
        print(f"saved {fold_path}", flush=True)
        print(fold_summary.head(args.print_top).to_string(index=False), flush=True)

        del model, x_sample, y_sample, contrib
        gc.collect()

    by_fold = pd.concat(parts, axis=0)
    summary = (
        by_fold.groupby(["feature", "is_b1"], as_index=False)
        .agg(
            mean_abs_shap=("mean_abs_shap", "mean"),
            mean_shap=("mean_shap", "mean"),
            mean_value=("mean_value", "mean"),
            std_value=("std_value", "mean"),
        )
        .sort_values("mean_abs_shap", ascending=False)
    )
    summary_path = ARTIFACTS / "exp019_behavior_v1_shap_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nsaved {summary_path}", flush=True)
    print("\ntop overall")
    print(summary.head(args.print_top).to_string(index=False), flush=True)
    print("\ntop b1")
    print(summary.loc[summary["is_b1"]].head(args.print_top).to_string(index=False), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-set", default="behavior_v1")
    parser.add_argument("--sample-size", type=int, default=50000)
    parser.add_argument("--print-top", type=int, default=30)
    return parser.parse_args()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
