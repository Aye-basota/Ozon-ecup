"""Train the LightGBM baseline and run the 2-fold local validation.

Run: python src/train.py
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import ROOT, SEED
from src.features import (
    FEATURES,
    get_all_users,
    load_sample_submit,
    load_train_data,
    make_dataset,
    validate_data_report,
)
from src.validation import get_folds, metric, run_validation


ARTIFACTS = ROOT / "artifacts"
NAIVE30_FOLD1_RANGE = (2.0, 2.4)
MIN_LGBM_IMPROVEMENT = 0.05
REFERENCE_FOLD = {"fold": 3, "train_cutoff": "2025-10-16", "val_cutoff": "2025-11-15"}
CALIBRATION_DELTA = -0.17
LGBM_PARAMS = dict(
    objective="regression",
    learning_rate=0.05,
    num_leaves=63,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    min_data_in_leaf=100,
    lambda_l2=1.0,
    verbose=-1,
    seed=SEED,
)


def make_lgbm(seed: int = SEED, n_estimators: int = 3000) -> lgb.LGBMRegressor:
    params = dict(LGBM_PARAMS)
    params["seed"] = seed
    return lgb.LGBMRegressor(n_estimators=n_estimators, **params)


def fit_lgbm(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame | None = None,
    y_va: pd.Series | None = None,
    seed: int = SEED,
    n_estimators: int = 3000,
) -> lgb.LGBMRegressor:
    model = make_lgbm(seed=seed, n_estimators=n_estimators)
    y_tr_log = np.log1p(y_tr)

    if X_va is None or y_va is None:
        model.fit(X_tr, y_tr_log)
        return model

    model.fit(
        X_tr,
        y_tr_log,
        eval_set=[(X_va, np.log1p(y_va))],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(100)],
    )
    return model


def predict_gmv(model: lgb.LGBMRegressor, X: pd.DataFrame, num_iteration: int | None = None) -> np.ndarray:
    pred_log = model.predict(X, num_iteration=num_iteration)
    return np.expm1(pred_log + CALIBRATION_DELTA).clip(min=0)


def _save_oof(
    fold: int,
    all_users: pd.Index,
    y_true: pd.Series,
    pred_lgbm: np.ndarray,
    pred_naive30: pd.Series,
    pred_naive90: pd.Series,
) -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    oof = pd.DataFrame(
        {
            "user_id": all_users.to_numpy(),
            "y_true": y_true.to_numpy(),
            "pred_lgbm": pred_lgbm,
            "pred_naive30": pred_naive30.to_numpy(),
            "pred_naive90": pred_naive90.to_numpy(),
        }
    )
    oof.to_csv(ARTIFACTS / f"oof_fold{fold}.csv", index=False)


def append_experiment_log(results: pd.DataFrame, experiment_id: str, experiment_name: str) -> None:
    path = ROOT / "experiments.md"
    if not path.exists():
        path.write_text(
            "id | дата | гипотеза | fold1 | fold2 | fold3 | mean | Δ к лучшему | вердикт | LB | заметки\n"
            "--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---\n",
            encoding="utf-8",
        )

    fold1 = results.loc[results["fold"] == 1].iloc[0]
    fold2 = results.loc[results["fold"] == 2].iloc[0]
    fold3_rows = results.loc[results["fold"] == 3]
    fold3_score = "NA" if fold3_rows.empty else f"{fold3_rows.iloc[0]['lgbm']:.6f}"
    mean_lgbm = results.loc[results["fold"].isin([1, 2]), "lgbm"].mean()
    row = (
        f"{experiment_id} | {date.today().isoformat()} | {experiment_name} | "
        f"{fold1['lgbm']:.6f} | {fold2['lgbm']:.6f} | {fold3_score} | "
        f"{mean_lgbm:.6f} | ? | pending | ? | err_{experiment_id}.csv\n"
    )
    if row in path.read_text(encoding="utf-8"):
        return
    with path.open("a", encoding="utf-8") as fh:
        fh.write(row)


def _handle_baseline_sanity(message: str, allow_sanity_fail: bool) -> None:
    if allow_sanity_fail:
        print(f"WARNING: {message}; continuing because --allow-sanity-fail is set")
        return
    raise RuntimeError(message)


def _validate_baseline_scores(
    fold: int,
    score_naive30: float,
    score_lgbm: float,
    allow_sanity_fail: bool,
) -> float:
    improvement = score_naive30 - score_lgbm

    lo, hi = NAIVE30_FOLD1_RANGE
    if fold == 1 and not (lo <= score_naive30 <= hi):
        _handle_baseline_sanity(
            f"Sanity check failed: fold 1 naive-30={score_naive30:.6f}, expected {lo:.2f}..{hi:.2f}",
            allow_sanity_fail,
        )

    if improvement < MIN_LGBM_IMPROVEMENT:
        _handle_baseline_sanity(
            f"Sanity check failed: fold {fold} LGBM improves naive-30 only by {improvement:.6f}",
            allow_sanity_fail,
        )

    return improvement


def _add_error_row(rows: list[dict[str, float | int | str]], group: str, name: str, values: dict[str, object]) -> None:
    mask = np.asarray(values["mask"], dtype=bool)
    y_log = np.asarray(values["y_log"])
    pred_log = np.asarray(values["pred_log"])
    sq_log_error = np.asarray(values["sq_log_error"])
    total_error = float(values["total_error"])

    if mask.sum() == 0:
        rows.append({"group": group, "segment": name, "count": 0, "sle_share": 0.0, "bias": 0.0, "rmsle": 0.0})
        return

    segment_error = sq_log_error[mask]
    rows.append(
        {
            "group": group,
            "segment": name,
            "count": int(mask.sum()),
            "sle_share": float(segment_error.sum() / total_error),
            "bias": float((y_log[mask] - pred_log[mask]).mean()),
            "rmsle": float(np.sqrt(segment_error.mean())),
        }
    )


def save_error_analysis(experiment_id: str, fold_artifacts: dict[int, dict[str, object]]) -> None:
    if 1 not in fold_artifacts:
        return

    artifact = fold_artifacts[1]
    X_va = artifact["X_va"]
    y_va = artifact["y_va"]
    pred_lgbm = np.asarray(artifact["pred_lgbm"]).clip(min=0)
    y_values = y_va.to_numpy()
    y_log = np.log1p(y_values)
    pred_log = np.log1p(pred_lgbm)
    sq_log_error = (y_log - pred_log) ** 2
    values = {"y_log": y_log, "pred_log": pred_log, "sq_log_error": sq_log_error, "total_error": sq_log_error.sum()}
    rows: list[dict[str, float | int | str]] = []

    values["mask"] = y_values == 0
    _add_error_row(rows, "target", "y_eq_0", values)
    values["mask"] = y_values > 0
    _add_error_row(rows, "target", "y_gt_0", values)

    # Rank before qcut keeps ten equal-sized buckets even with many zero historical GMV values.
    deciles = pd.qcut(X_va["w365_gmv"].rank(method="first"), 10, labels=False)
    for decile in range(10):
        values["mask"] = deciles.to_numpy() == decile
        _add_error_row(rows, "w365_gmv_decile", f"d{decile}", values)

    values["mask"] = X_va["tenure"].to_numpy() <= 30
    _add_error_row(rows, "tenure", "le_30", values)
    values["mask"] = X_va["tenure"].to_numpy() > 30
    _add_error_row(rows, "tenure", "gt_30", values)

    rec_buy = X_va["rec_buy"].to_numpy()
    for name, mask in [
        ("le_14", rec_buy <= 14),
        ("15_60", (rec_buy > 14) & (rec_buy <= 60)),
        ("gt_60", (rec_buy > 60) & (rec_buy < 999)),
        ("no_buy", rec_buy >= 999),
    ]:
        values["mask"] = mask
        _add_error_row(rows, "rec_buy", name, values)

    whale_threshold = np.quantile(y_values, 0.999)
    values["mask"] = (y_values >= whale_threshold) & (y_values > 0)
    _add_error_row(rows, "whales", "top_0_1_pct_y", values)

    err = pd.DataFrame(rows)
    err.to_csv(ARTIFACTS / f"err_{experiment_id}.csv", index=False)
    print(f"\nerror_analysis={ARTIFACTS / f'err_{experiment_id}.csv'}")
    print(err.sort_values("sle_share", ascending=False).head(8).to_string(index=False))


def run_cv(
    allow_sanity_fail: bool = False,
    experiment_id: str = "exp_run",
    experiment_name: str = "manual run",
    include_fold3: bool = True,
) -> float:
    df = load_train_data()
    sample_submit = load_sample_submit()
    all_users = validate_data_report(df, sample_submit)
    fold_artifacts: dict[int, dict[str, object]] = {}

    ARTIFACTS.mkdir(exist_ok=True)

    def fit_predict_lgbm(
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_va: pd.DataFrame,
        y_va: pd.Series,
        fold_cfg: dict[str, int | str],
    ) -> np.ndarray:
        fold = int(fold_cfg["fold"])
        model = fit_lgbm(X_tr, y_tr, X_va, y_va)
        best_iteration = int(getattr(model, "best_iteration_", 0) or model.n_estimators)
        print(f"calibration_delta={CALIBRATION_DELTA:.2f}")
        pred_lgbm = predict_gmv(model, X_va, num_iteration=best_iteration)
        pred_naive30 = X_va["w30_gmv"].clip(lower=0)
        pred_naive90 = (X_va["w90_gmv"] / 3.0).clip(lower=0)
        fold_artifacts[fold] = {
            "model": model,
            "X_va": X_va,
            "y_va": y_va,
            "pred_lgbm": pred_lgbm,
            "pred_naive30": pred_naive30,
            "pred_naive90": pred_naive90,
            "best_iteration": best_iteration,
        }
        return pred_lgbm

    main_folds = get_folds()
    cv_score = run_validation(
        df=df,
        all_users=all_users,
        make_dataset_fn=make_dataset,
        features=FEATURES,
        fit_predict_fn=fit_predict_lgbm,
        folds=main_folds,
    )
    if include_fold3:
        run_validation(
            df=df,
            all_users=all_users,
            make_dataset_fn=make_dataset,
            features=FEATURES,
            fit_predict_fn=fit_predict_lgbm,
            folds=[REFERENCE_FOLD],
        )

    results: list[dict[str, float | int]] = []
    for fold in sorted(fold_artifacts):
        artifact = fold_artifacts[fold]
        y_va = artifact["y_va"]
        pred_lgbm = artifact["pred_lgbm"]
        pred_naive30 = artifact["pred_naive30"]
        pred_naive90 = artifact["pred_naive90"]
        best_iteration = int(artifact["best_iteration"])

        score_naive30 = metric(y_va, pred_naive30)
        score_naive90 = metric(y_va, pred_naive90)
        score_lgbm = metric(y_va, pred_lgbm)
        improvement = _validate_baseline_scores(fold, score_naive30, score_lgbm, allow_sanity_fail)

        print(
            f"fold={fold} naive30={score_naive30:.6f} naive90={score_naive90:.6f} "
            f"lgbm={score_lgbm:.6f} improvement={improvement:.6f} best_iter={best_iteration}"
        )

        model = artifact["model"]
        joblib.dump(model, ARTIFACTS / f"lgbm_fold{fold}.joblib")
        model.booster_.save_model(str(ARTIFACTS / f"lgbm_fold{fold}.txt"))
        _save_oof(fold, all_users, y_va, pred_lgbm, pred_naive30, pred_naive90)

        results.append(
            {
                "fold": fold,
                "naive30": score_naive30,
                "naive90": score_naive90,
                "lgbm": score_lgbm,
                "improvement": improvement,
                "best_iteration": best_iteration,
            }
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(ARTIFACTS / "cv_results.csv", index=False)
    save_error_analysis(experiment_id, fold_artifacts)
    append_experiment_log(results_df, experiment_id, experiment_name)

    print(f"\nCV mean RMSLE folds1-2={cv_score:.6f}")
    return cv_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-sanity-fail",
        action="store_true",
        help="Continue after sanity-check failures that were reviewed manually.",
    )
    parser.add_argument("--experiment-id", default="exp_run")
    parser.add_argument("--experiment-name", default="manual run")
    parser.add_argument("--skip-fold3", action="store_true")
    args = parser.parse_args()
    run_cv(
        allow_sanity_fail=args.allow_sanity_fail,
        experiment_id=args.experiment_id,
        experiment_name=args.experiment_name,
        include_fold3=not args.skip_fold3,
    )


if __name__ == "__main__":
    main()
