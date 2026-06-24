"""
Score_Predictor: XGBoost regression model for NPL final innings score prediction.
"""

import os
import joblib
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

SCORE_MODEL_PATH = "models/score_model.pkl"
METRICS_LOG_PATH = "models/metrics.log"


def train_score_model(X_train, y_train) -> XGBRegressor:
    """Train an XGBRegressor on the provided training data.

    Serializes the fitted model to models/score_model.pkl and returns it.
    """
    os.makedirs("models", exist_ok=True)
    model = XGBRegressor(random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, SCORE_MODEL_PATH)
    return model


def evaluate_score_model(model: XGBRegressor, X_test, y_test) -> dict:
    """Evaluate the score model on the test set.

    Computes MAE and RMSE, appends them to models/metrics.log in key=value
    format, and returns a dict with keys 'MAE' and 'RMSE'.
    """
    y_pred = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    os.makedirs("models", exist_ok=True)
    with open(METRICS_LOG_PATH, "a") as f:
        f.write(f"score_MAE={mae:.4f}\n")
        f.write(f"score_RMSE={rmse:.4f}\n")

    return {"MAE": mae, "RMSE": rmse}
