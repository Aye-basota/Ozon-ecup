import sys
from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SEED, TARGET_DAYS
from src.features import build_df


FINAL_TRAIN_CUTOFF = pd.Timestamp("2026-01-15")
FINAL_TEST_CUTOFF = pd.Timestamp("2026-02-14")

TARGET_COLUMNS = [
    "user_id",
    "target_gmv",
    "target_gmv_above_zero",
    "target_log_gmv",
]


def get_feature_columns(df_train):
    return [
        col for col in df_train.columns
        if col not in TARGET_COLUMNS
    ]


def make_models():
    seed = SEED

    lgbm_classifier = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=63,
        min_data_in_leaf=100,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )

    lgbm_regressor = LGBMRegressor(
        objective="regression",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=63,
        min_data_in_leaf=100,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=1.0,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )

    lgbm_second_regressor = LGBMRegressor(
        objective="regression",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=63,
        min_data_in_leaf=100,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=1.0,
        random_state=seed * 2,
        n_jobs=-1,
        verbose=-1,
    )

    catboost_regressor = CatBoostRegressor(
        iterations=450,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=5,
        random_strength=1.0,
        bagging_temperature=1.0,
        loss_function="RMSE",
        random_seed=seed,
        verbose=False,
    )

    return {
        "lgbm_classifier": lgbm_classifier,
        "lgbm_regressor": lgbm_regressor,
        "lgbm_second_regressor": lgbm_second_regressor,
        "catboost_regressor": catboost_regressor,
    }


def fit_models(models, df_train, features):
    models["lgbm_classifier"].fit(
        df_train[features],
        df_train["target_gmv_above_zero"],
    )

    mask = df_train["target_gmv_above_zero"] == 1
    models["lgbm_regressor"].fit(
        df_train.loc[mask, features],
        df_train.loc[mask, "target_log_gmv"],
    )

    models["lgbm_second_regressor"].fit(
        df_train[features],
        df_train["target_log_gmv"],
    )

    models["catboost_regressor"].fit(
        df_train[features],
        df_train["target_log_gmv"],
    )

    return models


def train_models(df_raw, cutoff_date, target_days=TARGET_DAYS):
    df_train = build_df(df_raw, cutoff_date, target_days)
    features = get_feature_columns(df_train)
    models = make_models()

    fit_models(models, df_train, features)

    return models, features


def _format_cutoff(cutoff_date):
    return pd.to_datetime(cutoff_date).date().isoformat()


def _pack_final_models(current_models, current_features, team_models, team_meta):
    return {
        "current": current_models,
        "current_features": current_features,
        "team": team_models,
        "team_meta": team_meta,
    }


def train_final_models(df_raw, train_cutoff):
    from src.team_model import train_models as train_team_models

    train_cutoff = pd.to_datetime(train_cutoff)
    print(f"train current models: cutoff={_format_cutoff(train_cutoff)}", flush=True)
    current_models, current_features = train_models(
        df_raw,
        train_cutoff,
        TARGET_DAYS,
    )
    print("train team models", flush=True)
    team_models, team_meta = train_team_models(
        df_raw,
        train_cutoff=train_cutoff,
    )
    return _pack_final_models(current_models, current_features, team_models, team_meta)


def train_submit_models(df_raw):
    from src.team_model import train_models as train_team_models

    print(f"train current models: cutoff={_format_cutoff(FINAL_TRAIN_CUTOFF)}", flush=True)
    current_models, current_features = train_models(
        df_raw,
        FINAL_TRAIN_CUTOFF,
        TARGET_DAYS,
    )
    print("train team models", flush=True)
    team_models, team_meta = train_team_models(df_raw)
    return _pack_final_models(current_models, current_features, team_models, team_meta)
