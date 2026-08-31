from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostRegressor

from src.config import SEED, TARGET_DAYS
from src.features import build_df


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
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        min_data_in_leaf=100,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )

    lgbm_regressor = LGBMRegressor(
        objective="regression",
        n_estimators=700,
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
        n_estimators=700,
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
