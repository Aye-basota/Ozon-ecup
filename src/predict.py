import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import SUBMISSIONS
from src.features import build_features
from src.train import FINAL_TEST_CUTOFF, train_submit_models


CURRENT_WEIGHT = 0.55
TEAM_WEIGHT = 0.45
CURRENT_LOG_SCALE = 1.12
DEFAULT_OUTPUT = "final_classic_ml.csv"


def predict_log(models, df, features):
    pred_classifier = models["lgbm_classifier"].predict_proba(df[features])[:, 1]
    pred_regressor_log = models["lgbm_regressor"].predict(df[features])

    prediction_log = (
        0.8 * models["catboost_regressor"].predict(df[features])
        + 0.15 * (pred_classifier * pred_regressor_log)
        + 0.05 * models["lgbm_second_regressor"].predict(df[features])
    )

    return prediction_log


def predict_gmv(models, df, features):
    prediction_log = predict_log(models, df, features)
    prediction_gmv = np.expm1(prediction_log).clip(0)

    return prediction_gmv


def predict_current_log(models, df_raw, cutoff_date, features):
    df_features = build_features(df_raw, cutoff_date)
    prediction_log = predict_log(models, df_features, features) * CURRENT_LOG_SCALE
    return pd.Series(prediction_log, index=df_features["user_id"])


def predict_final_log(models, df_raw, cutoff_date=FINAL_TEST_CUTOFF):
    from src.team_model import predict_log as predict_team_log

    z_current = predict_current_log(
        models["current"],
        df_raw,
        cutoff_date,
        models["current_features"],
    )
    z_team = predict_team_log(
        models["team"],
        models["team_meta"],
        df_raw,
        cutoff_date=cutoff_date,
        level=models["team_meta"]["level"],
    )
    z_team = z_team.reindex(z_current.index).fillna(z_current)

    return CURRENT_WEIGHT * z_current + TEAM_WEIGHT * z_team


def predict_final_gmv(models, df_raw, cutoff_date=FINAL_TEST_CUTOFF):
    prediction_log = predict_final_log(models, df_raw, cutoff_date)
    prediction_gmv = np.expm1(prediction_log.to_numpy()).clip(0)
    return pd.Series(prediction_gmv, index=prediction_log.index)


def main():
    print("read train data", flush=True)
    df_raw = pd.read_parquet(ROOT / "data" / "train.parquet")
    models = train_submit_models(df_raw)
    print("predict final ensemble", flush=True)
    prediction = predict_final_gmv(models, df_raw)

    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    out_path = SUBMISSIONS / DEFAULT_OUTPUT
    submission = pd.DataFrame({
        "user_id": prediction.index,
        "predict": prediction.to_numpy(),
    })
    submission.to_csv(out_path, index=False)

    print(f"saved {out_path}", flush=True)
    print(f"rows={len(submission)}", flush=True)
    print(f"current_weight={CURRENT_WEIGHT} team_weight={TEAM_WEIGHT}", flush=True)
    print(f"current_log_scale={CURRENT_LOG_SCALE}", flush=True)
    print(f"mean_log1p={np.log1p(submission['predict']).mean():.6f}", flush=True)


if __name__ == "__main__":
    main()
