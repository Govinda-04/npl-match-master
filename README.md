# 🏏 NPL Match Master

AI-powered cricket match predictions for the **Nepal Premier League (NPL)**.

Built with real ball-by-ball data from the 2024 and 2025 NPL seasons, this Flask web app provides two live predictions:

- **Victory Predictor** — win probability for both teams during a 2nd innings chase
- **Score Predictor** — projected final total during the 1st innings

---

## Screenshots

> Run the app locally and visit `http://127.0.0.1:5000`

---

## Features

- 🏆 Win probability using Logistic Regression & Random Forest
- 📊 Final score prediction using XGBoost Regression
- 📈 Visual charts (Chart.js) for prediction results
- 🇳🇵 Covers all 8 NPL franchises across 2024 & 2025 seasons
- ⚡ Real-time predictions via Flask REST API

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.12, Flask |
| ML Models | scikit-learn, XGBoost |
| Data | pandas, numpy |
| Frontend | HTML, CSS, JavaScript, Chart.js |
| Serialization | joblib |

---

## Project Structure

```
NPL/
├── data/               # Raw NPL match data (CSV)
├── models/             # Trained model artifacts (.pkl)
├── src/
│   ├── app.py          # Flask application & API routes
│   ├── train.py        # Model training script
│   ├── prepare_data.py # Data preparation
│   ├── preprocess.py   # Feature engineering
│   ├── score_model.py  # Score prediction model
│   └── win_model.py    # Win probability model
├── static/             # CSS and JavaScript
├── templates/          # Jinja2 HTML templates
├── tests/              # Unit and property tests
└── requirements.txt    # Python dependencies
```

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/Govinda-04/npl-match-master.git
cd npl-match-master
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
python src/app.py
```

### 4. Open in browser

```
http://127.0.0.1:5000
```

---

## ML Models

| Model | Task | Performance |
|-------|------|-------------|
| XGBoost Regressor | Score Prediction | MAE < 5 runs |
| Logistic Regression | Win Probability | ~80% accuracy |
| Random Forest | Win Probability | High precision |

---

## Data

Ball-by-ball data from **64 NPL matches** (2024 & 2025 seasons) covering:
- All 8 franchises
- Deliveries, runs, wickets, match outcomes
- Venue: Tribhuvan University International Cricket Ground, Kirtipur

---

## License

MIT License — free to use and modify.
