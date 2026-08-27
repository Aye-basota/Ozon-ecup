import numpy as np


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
