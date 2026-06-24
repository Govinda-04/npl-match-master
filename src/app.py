"""
Flask API for NPL Match Win Prediction.

Loads serialized model artifacts at startup and exposes two prediction endpoints:
  POST /predict/score  — predicted final innings score
  POST /predict/win    — win probabilities from LR and RF models
"""

import sys
import logging
import numpy as np
import joblib
from flask import Flask, jsonify, request, render_template

app = Flask(__name__, template_folder="../templates", static_folder="../static")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Artifact paths ────────────────────────────────────────────────────────────
_SCORE_MODEL_PATH = "models/score_model.pkl"
_LR_MODEL_PATH = "models/lr_model.pkl"
_RF_MODEL_PATH = "models/rf_model.pkl"
_SCORE_ENCODERS_PATH = "models/score_encoders.pkl"
_WIN_ENCODERS_PATH = "models/win_encoders.pkl"

# ── Startup: load all artifacts ───────────────────────────────────────────────
def _load(path):
    try:
        return joblib.load(path)
    except Exception:
        print(f"ERROR: Failed to load artifact: {path}", file=sys.stderr)
        sys.exit(1)


score_model = _load(_SCORE_MODEL_PATH)
lr_model = _load(_LR_MODEL_PATH)
rf_model = _load(_RF_MODEL_PATH)
score_encoders = _load(_SCORE_ENCODERS_PATH)
win_encoders = _load(_WIN_ENCODERS_PATH)

# ── Feature column order (must match training order from train.py) ─────────────
# Score: innings, balls, batting_team, bowling_team, cur_score, crr,
#        balls_left, wicket_left, last_5_ov, venue, city
_SCORE_REQUIRED_FIELDS = [
    "batting_team", "bowling_team", "venue", "cur_score",
    "crr", "balls_left", "wicket_left", "last_5_ov", "city",
]
_SCORE_NUMERIC_FIELDS = ["cur_score", "crr", "balls_left", "wicket_left", "last_5_ov"]
_SCORE_CAT_FIELDS = ["batting_team", "bowling_team", "venue", "city"]

# Win: Batting_team, Bowling_team, city, Runs_left, Balls_left,
#      Wickets_left, Total_score, crr, rrr
_WIN_REQUIRED_FIELDS = [
    "Batting_team", "Bowling_team", "city", "Runs_left",
    "Balls_left", "Wickets_left", "Total_score", "crr", "rrr",
]
_WIN_NUMERIC_FIELDS = ["Runs_left", "Balls_left", "Wickets_left", "Total_score", "crr", "rrr"]
_WIN_CAT_FIELDS = ["Batting_team", "Bowling_team", "city"]


# ── Helpers ───────────────────────────────────────────────────────────────────
def _validate_fields(data, required_fields, numeric_fields, cat_fields, encoders):
    """Validate presence, types, and categorical values. Returns (error_response, None) or (None, parsed_data)."""
    # Check presence
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400

    # Check numeric types
    parsed = {}
    for field in required_fields:
        parsed[field] = data[field]

    for field in numeric_fields:
        try:
            parsed[field] = float(data[field])
        except (TypeError, ValueError):
            return jsonify({"error": f"Invalid value for field: {field}"}), 400

    # Check categorical values against encoder classes
    for field in cat_fields:
        val = str(data[field]).strip()
        if not val:
            return jsonify({"error": f"Missing required field: {field}"}), 400
        encoder = encoders.get(field)
        if encoder is not None and val not in encoder.classes_:
            return jsonify({"error": f"Unrecognized value for {field}: {val}"}), 400
        parsed[field] = val

    return None, parsed


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("home.html")


@app.route("/predict")
def predict():
    return render_template("predict.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/predict/score", methods=["POST"])
def predict_score():
    data = request.get_json(force=True, silent=True) or {}

    err, parsed = _validate_fields(
        data,
        _SCORE_REQUIRED_FIELDS,
        _SCORE_NUMERIC_FIELDS,
        _SCORE_CAT_FIELDS,
        score_encoders,
    )
    if err is not None:
        return err

    # Encode categoricals
    batting_team_enc = score_encoders["batting_team"].transform([parsed["batting_team"]])[0]
    bowling_team_enc = score_encoders["bowling_team"].transform([parsed["bowling_team"]])[0]
    venue_enc = score_encoders["venue"].transform([parsed["venue"]])[0]
    city_enc = score_encoders["city"].transform([parsed["city"]])[0]

    # Build feature array in exact training column order:
    # venue, innings, balls, batting_team, bowling_team, cur_score, crr,
    # balls_left, wicket_left, last_5_ov, city
    # innings = 1 (score predictor is for 1st innings)
    # balls = 120 - balls_left (derived)
    balls_bowled = max(0, 120 - int(parsed["balls_left"]))
    features = np.array([[
        venue_enc,
        1,                        # innings (score predictor = 1st innings)
        balls_bowled,             # balls bowled so far
        batting_team_enc,
        bowling_team_enc,
        parsed["cur_score"],
        parsed["crr"],
        parsed["balls_left"],
        parsed["wicket_left"],
        parsed["last_5_ov"],
        city_enc,
    ]], dtype=float)

    prediction = score_model.predict(features)
    predicted_score = int(round(float(prediction[0])))
    return jsonify({"predicted_score": predicted_score})


@app.route("/predict/win", methods=["POST"])
def predict_win():
    data = request.get_json(force=True, silent=True) or {}

    err, parsed = _validate_fields(
        data,
        _WIN_REQUIRED_FIELDS,
        _WIN_NUMERIC_FIELDS,
        _WIN_CAT_FIELDS,
        win_encoders,
    )
    if err is not None:
        return err

    # Encode categoricals
    batting_team_enc = win_encoders["Batting_team"].transform([parsed["Batting_team"]])[0]
    bowling_team_enc = win_encoders["Bowling_team"].transform([parsed["Bowling_team"]])[0]
    city_enc = win_encoders["city"].transform([parsed["city"]])[0]

    # Build feature array in training column order:
    # Batting_team, Bowling_team, city, Runs_left, Balls_left, Wickets_left, Total_score, crr, rrr
    features = np.array([[
        batting_team_enc,
        bowling_team_enc,
        city_enc,
        parsed["Runs_left"],
        parsed["Balls_left"],
        parsed["Wickets_left"],
        parsed["Total_score"],
        parsed["crr"],
        parsed["rrr"],
    ]], dtype=float)

    def _proba(model):
        proba = model.predict_proba(features)[0]
        batting_prob = round(float(proba[1]) * 100, 1)
        bowling_prob = round(100 - batting_prob, 1)
        return {"batting_team_prob": batting_prob, "bowling_team_prob": bowling_prob}

    return jsonify({"lr": _proba(lr_model), "rf": _proba(rf_model)})


# ── Global error handler ──────────────────────────────────────────────────────
@app.errorhandler(Exception)
def handle_unexpected_error(e):
    logger.exception("Unexpected error")
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(debug=False, port=5000)
