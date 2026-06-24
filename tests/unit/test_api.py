"""
Unit tests for Flask API error responses.

Covers:
- Startup exits with code 1 when a model file is missing (Requirement 5.3)
- POST /predict/score returns 400 when a required field is missing (Requirement 2.4)
- POST /predict/win returns 400 when a required field is missing (Requirement 3.4)
"""

import importlib
import sys
import types
import unittest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers to build a minimal fake app without hitting disk
# ---------------------------------------------------------------------------

def _make_fake_encoder(classes):
    """Return a mock LabelEncoder with the given classes_ list."""
    enc = MagicMock()
    enc.classes_ = classes
    enc.transform = lambda vals: [classes.index(v) for v in vals]
    return enc


def _make_fake_score_encoders():
    teams = ["Kathmandu Kings", "Pokhara Avengers", "Biratnagar Warriors",
             "Chitwan Tigers", "Janakpur Bolts", "Lumbini Lions"]
    venues = ["TU Cricket Ground", "Pokhara Stadium"]
    cities = ["Kathmandu", "Pokhara", "Biratnagar", "Chitwan", "Janakpur", "Lumbini"]
    return {
        "batting_team": _make_fake_encoder(teams),
        "bowling_team": _make_fake_encoder(teams),
        "venue": _make_fake_encoder(venues),
        "city": _make_fake_encoder(cities),
    }


def _make_fake_win_encoders():
    teams = ["Kathmandu Kings", "Pokhara Avengers", "Biratnagar Warriors",
             "Chitwan Tigers", "Janakpur Bolts", "Lumbini Lions"]
    cities = ["Kathmandu", "Pokhara", "Biratnagar", "Chitwan", "Janakpur", "Lumbini"]
    return {
        "Batting_team": _make_fake_encoder(teams),
        "Bowling_team": _make_fake_encoder(teams),
        "city": _make_fake_encoder(cities),
    }


def _make_fake_score_model():
    m = MagicMock()
    m.predict.return_value = [145.0]
    return m


def _make_fake_win_model():
    m = MagicMock()
    m.predict_proba.return_value = [[0.38, 0.62]]
    return m


def _load_app_with_mocked_artifacts():
    """
    Import (or re-import) src.app with joblib.load patched to return fake
    artifacts so no real .pkl files are needed.
    """
    load_sequence = [
        _make_fake_score_model(),
        _make_fake_win_model(),
        _make_fake_win_model(),
        _make_fake_score_encoders(),
        _make_fake_win_encoders(),
    ]
    call_count = {"n": 0}

    def fake_load(path):
        idx = call_count["n"]
        call_count["n"] += 1
        return load_sequence[idx]

    # Remove cached module so the module-level _load() calls run again
    for mod_name in list(sys.modules.keys()):
        if "src.app" in mod_name or mod_name == "src.app":
            del sys.modules[mod_name]

    with patch("joblib.load", side_effect=fake_load):
        import src.app as app_module

    return app_module


# ---------------------------------------------------------------------------
# Test: startup exits with code 1 when a model file is missing
# ---------------------------------------------------------------------------

class TestStartupFailure(unittest.TestCase):

    def test_exits_with_code_1_when_model_file_missing(self):
        """Requirement 5.3 — missing artifact must cause sys.exit(1)."""
        # Remove cached module so module-level code re-runs
        for mod_name in list(sys.modules.keys()):
            if "src.app" in mod_name or mod_name == "src.app":
                del sys.modules[mod_name]

        def raise_file_not_found(path):
            raise FileNotFoundError(f"No such file: {path}")

        with patch("joblib.load", side_effect=raise_file_not_found):
            with self.assertRaises(SystemExit) as ctx:
                import src.app  # noqa: F401

        self.assertEqual(ctx.exception.code, 1)


# ---------------------------------------------------------------------------
# Test: /predict/score returns 400 for missing required fields
# ---------------------------------------------------------------------------

class TestScoreEndpointMissingFields(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app_module = _load_app_with_mocked_artifacts()
        cls.client = app_module.app.test_client()

    def _valid_payload(self):
        return {
            "batting_team": "Kathmandu Kings",
            "bowling_team": "Pokhara Avengers",
            "venue": "TU Cricket Ground",
            "cur_score": 85,
            "crr": 8.5,
            "balls_left": 60,
            "wicket_left": 6,
            "last_5_ov": 42,
            "city": "Kathmandu",
        }

    def _post(self, payload):
        return self.client.post(
            "/predict/score",
            json=payload,
            content_type="application/json",
        )

    def test_missing_batting_team_returns_400(self):
        payload = self._valid_payload()
        del payload["batting_team"]
        resp = self._post(payload)
        self.assertEqual(resp.status_code, 400)
        body = resp.get_json()
        self.assertIn("error", body)

    def test_missing_bowling_team_returns_400(self):
        payload = self._valid_payload()
        del payload["bowling_team"]
        resp = self._post(payload)
        self.assertEqual(resp.status_code, 400)
        body = resp.get_json()
        self.assertIn("error", body)

    def test_missing_venue_returns_400(self):
        payload = self._valid_payload()
        del payload["venue"]
        resp = self._post(payload)
        self.assertEqual(resp.status_code, 400)
        body = resp.get_json()
        self.assertIn("error", body)

    def test_missing_cur_score_returns_400(self):
        payload = self._valid_payload()
        del payload["cur_score"]
        resp = self._post(payload)
        self.assertEqual(resp.status_code, 400)
        body = resp.get_json()
        self.assertIn("error", body)

    def test_missing_balls_left_returns_400(self):
        payload = self._valid_payload()
        del payload["balls_left"]
        resp = self._post(payload)
        self.assertEqual(resp.status_code, 400)
        body = resp.get_json()
        self.assertIn("error", body)

    def test_error_message_identifies_missing_field(self):
        payload = self._valid_payload()
        del payload["city"]
        resp = self._post(payload)
        self.assertEqual(resp.status_code, 400)
        body = resp.get_json()
        self.assertIn("city", body["error"])


# ---------------------------------------------------------------------------
# Test: /predict/win returns 400 for missing required fields
# ---------------------------------------------------------------------------

class TestWinEndpointMissingFields(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app_module = _load_app_with_mocked_artifacts()
        cls.client = app_module.app.test_client()

    def _valid_payload(self):
        return {
            "Batting_team": "Kathmandu Kings",
            "Bowling_team": "Pokhara Avengers",
            "city": "Kathmandu",
            "Runs_left": 63,
            "Balls_left": 60,
            "Wickets_left": 6,
            "Total_score": 148,
            "crr": 8.5,
            "rrr": 6.3,
        }

    def _post(self, payload):
        return self.client.post(
            "/predict/win",
            json=payload,
            content_type="application/json",
        )

    def test_missing_batting_team_returns_400(self):
        payload = self._valid_payload()
        del payload["Batting_team"]
        resp = self._post(payload)
        self.assertEqual(resp.status_code, 400)
        body = resp.get_json()
        self.assertIn("error", body)

    def test_missing_bowling_team_returns_400(self):
        payload = self._valid_payload()
        del payload["Bowling_team"]
        resp = self._post(payload)
        self.assertEqual(resp.status_code, 400)
        body = resp.get_json()
        self.assertIn("error", body)

    def test_missing_city_returns_400(self):
        payload = self._valid_payload()
        del payload["city"]
        resp = self._post(payload)
        self.assertEqual(resp.status_code, 400)
        body = resp.get_json()
        self.assertIn("error", body)

    def test_missing_runs_left_returns_400(self):
        payload = self._valid_payload()
        del payload["Runs_left"]
        resp = self._post(payload)
        self.assertEqual(resp.status_code, 400)
        body = resp.get_json()
        self.assertIn("error", body)

    def test_missing_rrr_returns_400(self):
        payload = self._valid_payload()
        del payload["rrr"]
        resp = self._post(payload)
        self.assertEqual(resp.status_code, 400)
        body = resp.get_json()
        self.assertIn("error", body)

    def test_error_message_identifies_missing_field(self):
        payload = self._valid_payload()
        del payload["Total_score"]
        resp = self._post(payload)
        self.assertEqual(resp.status_code, 400)
        body = resp.get_json()
        self.assertIn("Total_score", body["error"])


if __name__ == "__main__":
    unittest.main()
