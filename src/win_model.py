"""
Win_Predictor: Logistic Regression and Random Forest classification models
for NPL match win probability prediction.
"""

import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

LR_MODEL_PATH = "models/lr_model.pkl"
RF_MODEL_PATH = "models/rf_model.pkl"
METRICS_LOG_PATH = "models/metrics.log"


def train_win_models(X_train, y_train):
    """Train LogisticRegression and RandomForestClassifier on the provided training data.

    Serializes both fitted models to models/lr_model.pkl and models/rf_model.pkl,
    then returns a tuple (lr_model, rf_model).
    """
    os.makedirs("models", exist_ok=True)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    joblib.dump(lr, LR_MODEL_PATH)

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    joblib.dump(rf, RF_MODEL_PATH)

    return (lr, rf)


def evaluate_win_models(lr, rf, X_test, y_test) -> dict:
    """Evaluate both win models on the test set.

    Computes accuracy, precision, recall, and F1 for each model, appends
    results to models/metrics.log in key=value format, and returns a dict
    with keys 'lr' and 'rf', each containing accuracy/precision/recall/f1.
    """
    results = {}

    for name, model in [("lr", lr), ("rf", rf)]:
        y_pred = model.predict(X_test)
        accuracy = float(accuracy_score(y_test, y_pred))
        precision = float(precision_score(y_test, y_pred, zero_division=0))
        recall = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        results[name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    os.makedirs("models", exist_ok=True)
    with open(METRICS_LOG_PATH, "a") as f:
        for name, metrics in results.items():
            f.write(f"{name}_accuracy={metrics['accuracy']:.4f}\n")
            f.write(f"{name}_precision={metrics['precision']:.4f}\n")
            f.write(f"{name}_recall={metrics['recall']:.4f}\n")
            f.write(f"{name}_f1={metrics['f1']:.4f}\n")

    return results
