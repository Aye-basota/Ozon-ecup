"""Generate baseline submissions.

Run after successful validation:
python src/predict.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import ROOT, SEED, SUBMISSIONS
from src.features import FEATURES, get_all_users, load_sample_submit, load_train_data, make_dataset
from src.train import ARTIFACTS, fit_lgbm, predict_gmv


TEST_CUTOFF = "2026-02-14"
SUBMIT_TRAIN_CUTOFF = "2026-01-14"


def _best_iteration_from_fold1() -> int:
    path = ARTIFACTS / "cv_results.csv"
    if not path.exists():
        raise FileNotFoundError("Run python src/train.py first: artifacts/cv_results.csv is missing")

    cv = pd.read_csv(path)
    fold1 = cv.loc[cv["fold"] == 1]
    if fold1.empty:
        raise ValueError("artifacts/cv_results.csv has no fold 1")
    return int(fold1.iloc[0]["best_iteration"])


def make_submit(pred: pd.Series | np.ndarray, path: Path, all_users: pd.Index, sample_submit: pd.DataFrame) -> None:
    SUBMISSIONS.mkdir(exist_ok=True)
    pred_series = pd.Series(np.asarray(pred), index=all_users, name="predict").clip(lower=0)
    sub = pd.DataFrame({"user_id": all_users.to_numpy(), "predict": pred_series.to_numpy()})
    sub = sample_submit[["user_id"]].merge(sub, on="user_id", how="left")

    assert list(sub.columns) == ["user_id", "predict"]
    assert sub["predict"].notna().all()
    assert (sub["predict"] >= 0).all()
    assert len(sub) == len(sample_submit)
    sub.to_csv(path, index=False)
    print(f"saved {path}")


def main() -> None:
    df = load_train_data()
    all_users = get_all_users(df)
    sample_submit = load_sample_submit()
    sample_users = pd.Index(sample_submit["user_id"].unique()).sort_values()
    assert set(sample_users) == set(all_users)

    X_test = make_dataset(df, TEST_CUTOFF, all_users, need_target=False)
    assert list(X_test.columns) == FEATURES

    make_submit(X_test["w30_gmv"], SUBMISSIONS / "A_naive30.csv", all_users, sample_submit)
    make_submit(X_test["w90_gmv"] / 3.0, SUBMISSIONS / "B_naive90.csv", all_users, sample_submit)

    best_iteration = _best_iteration_from_fold1()
    print(f"training submit LGBM with n_estimators={best_iteration}")
    X_tr, y_tr = make_dataset(df, SUBMIT_TRAIN_CUTOFF, all_users)
    assert list(X_tr.columns) == list(X_test.columns)

    model = fit_lgbm(X_tr, y_tr, seed=SEED, n_estimators=best_iteration)
    pred_lgbm = predict_gmv(model, X_test)
    make_submit(pred_lgbm, SUBMISSIONS / "C_lgbm.csv", all_users, sample_submit)

    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(model, ARTIFACTS / "lgbm_submit.joblib")
    model.booster_.save_model(str(ARTIFACTS / "lgbm_submit.txt"))


if __name__ == "__main__":
    main()
