"""Run A1 temporal ensemble validation.

Run: python src/temporal_ensemble.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.features import FEATURES, get_all_users, load_sample_submit, load_train_data, make_dataset, validate_data_report
from src.train import ARTIFACTS, CALIBRATION_DELTA, REFERENCE_FOLD, fit_lgbm, save_error_analysis
from src.validation import assert_cutoff_gap, get_folds, metric


EXPERIMENT_ID = "exp_034"
EXPERIMENT_NAME = "A1 temporal ensemble 3 neighboring cutoffs"


def temporal_train_cutoffs(val_cutoff: str | pd.Timestamp) -> list[str]:
    """Return the 3 neighboring single-model cutoffs from PLAN.md A1."""
    val = pd.Timestamp(val_cutoff)
    return [
        (val - pd.Timedelta(days=58)).date().isoformat(),
        (val - pd.Timedelta(days=44)).date().isoformat(),
        (val - pd.Timedelta(days=30)).date().isoformat(),
    ]


def predict_log(model, X: pd.DataFrame, best_iteration: int) -> np.ndarray:
    return np.asarray(model.predict(X, num_iteration=best_iteration), dtype="float64")


def amount_from_log(pred_log: np.ndarray) -> np.ndarray:
    return np.expm1(pred_log + CALIBRATION_DELTA).clip(min=0)


def save_oof(
    fold: int,
    all_users: pd.Index,
    y_true: pd.Series,
    pred_temporal: np.ndarray,
    pred_central: np.ndarray,
) -> None:
    path = ARTIFACTS / f"oof_{EXPERIMENT_ID}_fold{fold}.csv"
    oof = pd.DataFrame(
        {
            "user_id": all_users.to_numpy(),
            "y_true": y_true.to_numpy(),
            "pred_temporal": pred_temporal,
            "pred_central": pred_central,
        }
    )
    oof.to_csv(path, index=False)


def run() -> float:
    ARTIFACTS.mkdir(exist_ok=True)
    df = load_train_data()
    sample_submit = load_sample_submit()
    all_users = validate_data_report(df, sample_submit)

    folds = get_folds() + [REFERENCE_FOLD]
    result_rows: list[dict[str, float | int | str]] = []
    member_rows: list[dict[str, float | int | str]] = []
    fold_artifacts: dict[int, dict[str, object]] = {}

    for fold_cfg in folds:
        fold = int(fold_cfg["fold"])
        val_cutoff = str(fold_cfg["val_cutoff"])
        central_cutoff = str(fold_cfg["train_cutoff"])
        train_cutoffs = temporal_train_cutoffs(val_cutoff)
        assert train_cutoffs[-1] == central_cutoff

        print(f"\nfold={fold} val_cutoff={val_cutoff} temporal_cutoffs={train_cutoffs}")
        X_va, y_va = make_dataset(df, val_cutoff, all_users)
        assert list(X_va.columns) == FEATURES

        member_logs: list[np.ndarray] = []
        central_pred: np.ndarray | None = None

        for train_cutoff in train_cutoffs:
            assert_cutoff_gap(train_cutoff, val_cutoff)
            X_tr, y_tr = make_dataset(df, train_cutoff, all_users)
            assert list(X_tr.columns) == FEATURES

            model = fit_lgbm(X_tr, y_tr, X_va, y_va)
            best_iteration = int(getattr(model, "best_iteration_", 0) or model.n_estimators)
            pred_log = predict_log(model, X_va, best_iteration)
            pred_amount = amount_from_log(pred_log)
            score = metric(y_va, pred_amount)
            print(
                f"fold={fold} member_train_cutoff={train_cutoff} "
                f"score={score:.6f} best_iter={best_iteration}"
            )

            member_logs.append(pred_log)
            if train_cutoff == central_cutoff:
                central_pred = pred_amount

            member_rows.append(
                {
                    "fold": fold,
                    "val_cutoff": val_cutoff,
                    "train_cutoff": train_cutoff,
                    "score": score,
                    "best_iteration": best_iteration,
                }
            )

        if central_pred is None:
            raise RuntimeError(f"central cutoff {central_cutoff} was not trained for fold {fold}")

        pred_temporal = amount_from_log(np.mean(np.vstack(member_logs), axis=0))
        central_score = metric(y_va, central_pred)
        temporal_score = metric(y_va, pred_temporal)
        improvement = central_score - temporal_score
        print(
            f"fold={fold} central={central_score:.6f} temporal={temporal_score:.6f} "
            f"improvement={improvement:.6f}"
        )

        result_rows.append(
            {
                "fold": fold,
                "val_cutoff": val_cutoff,
                "central_cutoff": central_cutoff,
                "temporal_cutoffs": ",".join(train_cutoffs),
                "central": central_score,
                "temporal": temporal_score,
                "improvement": improvement,
            }
        )
        save_oof(fold, all_users, y_va, pred_temporal, central_pred)
        fold_artifacts[fold] = {"X_va": X_va, "y_va": y_va, "pred_lgbm": pred_temporal}

    results = pd.DataFrame(result_rows)
    members = pd.DataFrame(member_rows)
    results.to_csv(ARTIFACTS / f"{EXPERIMENT_ID}_temporal_ensemble_results.csv", index=False)
    members.to_csv(ARTIFACTS / f"{EXPERIMENT_ID}_temporal_ensemble_members.csv", index=False)
    save_error_analysis(EXPERIMENT_ID, fold_artifacts)

    main_mean = float(results.loc[results["fold"].isin([1, 2]), "temporal"].mean())
    all_mean = float(results["temporal"].mean())
    central_main_mean = float(results.loc[results["fold"].isin([1, 2]), "central"].mean())
    central_all_mean = float(results["central"].mean())
    print(f"\n{EXPERIMENT_ID} {EXPERIMENT_NAME}")
    print(f"mean_temporal_folds1_2={main_mean:.6f}")
    print(f"mean_temporal_folds1_2_3={all_mean:.6f}")
    print(f"mean_central_folds1_2={central_main_mean:.6f}")
    print(f"mean_central_folds1_2_3={central_all_mean:.6f}")
    print(f"mean_improvement_folds1_2={central_main_mean - main_mean:.6f}")
    print(f"mean_improvement_folds1_2_3={central_all_mean - all_mean:.6f}")
    return main_mean


def main() -> None:
    run()


if __name__ == "__main__":
    main()
