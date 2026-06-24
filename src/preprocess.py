"""
Data_Pipeline: CSV loading, cleaning, encoding, and splitting for NPL match data.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# Required columns for each dataset
SCORE_REQUIRED_COLS = [
    "match_id", "season", "venue", "start_date", "innings", "balls",
    "batting_team", "bowling_team", "cur_score", "crr", "balls_left",
    "wicket_left", "last_5_ov", "city", "total_score",
]

WIN_REQUIRED_COLS = [
    "Batting_team", "Bowling_team", "city", "Runs_left", "Balls_left",
    "Wickets_left", "Total_score", "crr", "rrr", "result",
]


def load_score_data(path: str) -> pd.DataFrame:
    """Load score CSV and validate required columns are present."""
    df = pd.read_csv(path)
    for col in SCORE_REQUIRED_COLS:
        if col not in df.columns:
            raise ValueError(col)
    return df


def load_win_data(path: str) -> pd.DataFrame:
    """Load win CSV and validate required columns are present."""
    df = pd.read_csv(path)
    for col in WIN_REQUIRED_COLS:
        if col not in df.columns:
            raise ValueError(col)
    return df


def clean_data(df: pd.DataFrame, required_cols: list) -> pd.DataFrame:
    """Drop rows with NaN in required columns and drop duplicate rows."""
    df = df.dropna(subset=required_cols)
    df = df.drop_duplicates()
    return df.reset_index(drop=True)


def fit_encoders(df: pd.DataFrame, cat_cols: list) -> dict:
    """Fit a LabelEncoder per categorical column and return a dict of encoders."""
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        encoders[col] = le
    return encoders


def apply_encoders(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """Transform categorical columns to non-negative integers using fitted encoders."""
    df = df.copy()
    for col, le in encoders.items():
        df[col] = le.transform(df[col].astype(str))
    return df


def save_encoders(encoders: dict, path: str) -> None:
    """Serialize encoders dict to disk via joblib."""
    joblib.dump(encoders, path)


def load_encoders(path: str) -> dict:
    """Load encoders dict from disk via joblib."""
    return joblib.load(path)


def split_data(df: pd.DataFrame, target_col: str, test_size: float = 0.2):
    """Split DataFrame into 80/20 train/test sets.

    Returns (X_train, X_test, y_train, y_test).
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=42)
