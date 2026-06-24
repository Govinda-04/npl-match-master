"""
Training entry point for the NPL Match Win Prediction system.

Orchestrates the full training pipeline for both the score model and win models.
"""

import argparse
import os

from src.preprocess import (
    load_score_data,
    load_win_data,
    clean_data,
    fit_encoders,
    apply_encoders,
    save_encoders,
    split_data,
    SCORE_REQUIRED_COLS,
    WIN_REQUIRED_COLS,
)
from src.score_model import train_score_model, evaluate_score_model
from src.win_model import train_win_models, evaluate_win_models

SCORE_ENCODERS_PATH = "models/score_encoders.pkl"
WIN_ENCODERS_PATH = "models/win_encoders.pkl"

SCORE_CAT_COLS = ["batting_team", "bowling_team", "venue", "city"]
WIN_CAT_COLS = ["Batting_team", "Bowling_team", "city"]

SCORE_DROP_COLS = ["match_id", "season", "start_date", "total_score"]
SCORE_TARGET = "total_score"
WIN_TARGET = "result"


def main():
    parser = argparse.ArgumentParser(description="Train NPL match prediction models")
    parser.add_argument(
        "--score-csv",
        default="data/score_data.csv",
        help="Path to score dataset CSV (default: data/score_data.csv)",
    )
    parser.add_argument(
        "--win-csv",
        default="data/win_data.csv",
        help="Path to win dataset CSV (default: data/win_data.csv)",
    )
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)

    # ── Score model pipeline ──────────────────────────────────────────────────
    print(f"Loading score data from {args.score_csv} ...")
    score_df = load_score_data(args.score_csv)

    print("Cleaning score data ...")
    score_df = clean_data(score_df, SCORE_REQUIRED_COLS)

    print("Fitting and saving score encoders ...")
    score_encoders = fit_encoders(score_df, SCORE_CAT_COLS)
    save_encoders(score_encoders, SCORE_ENCODERS_PATH)

    print("Applying score encoders ...")
    score_df = apply_encoders(score_df, score_encoders)

    # Drop non-feature columns, keeping target separate
    score_feature_df = score_df.drop(
        columns=[c for c in SCORE_DROP_COLS if c != SCORE_TARGET and c in score_df.columns]
    )

    print("Splitting score data (80/20) ...")
    X_train_s, X_test_s, y_train_s, y_test_s = split_data(score_feature_df, SCORE_TARGET)

    print("Training score model ...")
    score_model = train_score_model(X_train_s, y_train_s)

    print("Evaluating score model ...")
    score_metrics = evaluate_score_model(score_model, X_test_s, y_test_s)

    # ── Win model pipeline ────────────────────────────────────────────────────
    print(f"\nLoading win data from {args.win_csv} ...")
    win_df = load_win_data(args.win_csv)

    print("Cleaning win data ...")
    win_df = clean_data(win_df, WIN_REQUIRED_COLS)

    print("Fitting and saving win encoders ...")
    win_encoders = fit_encoders(win_df, WIN_CAT_COLS)
    save_encoders(win_encoders, WIN_ENCODERS_PATH)

    print("Applying win encoders ...")
    win_df = apply_encoders(win_df, win_encoders)

    print("Splitting win data (80/20) ...")
    X_train_w, X_test_w, y_train_w, y_test_w = split_data(win_df, WIN_TARGET)

    print("Training win models ...")
    lr_model, rf_model = train_win_models(X_train_w, y_train_w)

    print("Evaluating win models ...")
    win_metrics = evaluate_win_models(lr_model, rf_model, X_test_w, y_test_w)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n=== Training Summary ===")
    print(f"Score Model  — MAE: {score_metrics['MAE']:.4f}, RMSE: {score_metrics['RMSE']:.4f}")
    for model_name, m in win_metrics.items():
        print(
            f"Win Model ({model_name.upper()}) — "
            f"Accuracy: {m['accuracy']:.4f}, "
            f"Precision: {m['precision']:.4f}, "
            f"Recall: {m['recall']:.4f}, "
            f"F1: {m['f1']:.4f}"
        )
    print(f"\nArtifacts saved to models/")
    print(f"  score_model.pkl, score_encoders.pkl")
    print(f"  lr_model.pkl, rf_model.pkl, win_encoders.pkl")
    print(f"  metrics.log")


if __name__ == "__main__":
    main()
