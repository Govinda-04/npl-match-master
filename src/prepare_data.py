"""
prepare_data.py — Convert raw NPL CSV files into training-ready score and win datasets.

Usage:
    python -m src.prepare_data \
        --ball-by-ball data/NPL-2024.csv \
        --matches      data/NPL_matches.csv \
        --out-score    data/score_data.csv \
        --out-win      data/win_data.csv
"""

import argparse
import os
import pandas as pd


# ── helpers ───────────────────────────────────────────────────────────────────

def _is_legal(row) -> bool:
    """Return True if the delivery counts as a legal ball (not a wide/no-ball extra)."""
    return row["wide_runs"] == 0 and row["noball_runs"] == 0


def _over_to_balls(over_str) -> int:
    """Convert '14.3' → 87 legal balls bowled."""
    try:
        parts = str(over_str).split(".")
        overs = int(parts[0])
        balls = int(parts[1]) if len(parts) > 1 else 0
        return overs * 6 + balls
    except Exception:
        return 0


# ── score dataset ─────────────────────────────────────────────────────────────

def build_score_dataset(bbb: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """
    Build a ball-by-ball score dataset from the ball-by-ball CSV.

    For every legal delivery in the 1st innings we record the match state
    at that point and the final innings total as the target.

    Columns produced:
        match_id, season, venue, start_date, innings, balls,
        batting_team, bowling_team, cur_score, crr, balls_left,
        wicket_left, last_5_ov, city, total_score
    """
    # ── pre-compute final totals per (match_id, inning) from matches CSV ──────
    # matches CSV has team_1_runs / team_2_runs
    totals = {}
    for _, row in matches.iterrows():
        mid = int(row["id"])
        totals[(mid, 1)] = int(row["team_1_runs"])
        totals[(mid, 2)] = int(row["team_2_runs"])

    # ── build a lookup: match_id → (venue, city, season, date) ───────────────
    meta = {}
    for _, row in matches.iterrows():
        mid = int(row["id"])
        meta[mid] = {
            "venue": str(row["venue"]),
            "city":  str(row["city"]),
            "season": int(row["season"]),
            "date":  str(row["date"]),
        }

    records = []

    for (match_id, inning), grp in bbb.groupby(["match_id", "inning"]):
        match_id = int(match_id)
        inning   = int(inning)

        if (match_id, inning) not in totals:
            continue
        total_score = totals[(match_id, inning)]
        if match_id not in meta:
            continue
        m = meta[match_id]

        batting_team  = grp["batting_team"].iloc[0]
        bowling_team  = grp["bowling_team"].iloc[0]

        # sort by ball_over
        grp = grp.copy()
        grp["_ball_num"] = grp["ball_over"].apply(_over_to_balls)
        grp = grp.sort_values("_ball_num")

        cum_score   = 0
        cum_wickets = 0
        cum_legal   = 0          # legal balls bowled so far
        last5_runs  = 0          # runs in last 5 overs (30 legal balls)
        window      = []         # (legal_ball_index, runs_on_that_ball)

        for _, ball in grp.iterrows():
            runs_this = int(ball["total_runs"])
            is_legal  = _is_legal(ball)
            is_wicket = str(ball["player_dismissed"]).strip() not in ("", "nan")

            cum_score += runs_this
            if is_wicket:
                cum_wickets += 1
            if is_legal:
                cum_legal += 1
                window.append((cum_legal, runs_this))

            # keep only last 30 legal balls in window
            window = [(b, r) for b, r in window if cum_legal - b < 30]
            last5_runs = sum(r for _, r in window)

            # only record from over 5 onwards (30 legal balls) to have
            # meaningful features, and only for 1st innings
            if inning == 1 and cum_legal >= 6:
                balls_left = max(0, 120 - cum_legal)
                crr = round(cum_score / (cum_legal / 6), 2) if cum_legal > 0 else 0.0

                records.append({
                    "match_id":     match_id,
                    "season":       m["season"],
                    "venue":        m["venue"],
                    "start_date":   m["date"],
                    "innings":      inning,
                    "balls":        cum_legal,
                    "batting_team": batting_team,
                    "bowling_team": bowling_team,
                    "cur_score":    cum_score,
                    "crr":          crr,
                    "balls_left":   balls_left,
                    "wicket_left":  max(0, 10 - cum_wickets),
                    "last_5_ov":    last5_runs,
                    "city":         m["city"],
                    "total_score":  total_score,
                })

    df = pd.DataFrame(records)
    return df


# ── win dataset ───────────────────────────────────────────────────────────────

def build_win_dataset(bbb: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """
    Build a ball-by-ball win-probability dataset from the 2nd innings.

    For every legal delivery in the 2nd innings we record the match state
    and whether the batting team ultimately won (result=1) or lost (result=0).

    Columns produced:
        Batting_team, Bowling_team, city, Runs_left, Balls_left,
        Wickets_left, Total_score, crr, rrr, result
    """
    # ── pre-compute targets and winners ──────────────────────────────────────
    match_info = {}
    for _, row in matches.iterrows():
        mid     = int(row["id"])
        target  = int(row["team_1_runs"]) + 1   # 2nd innings needs this many
        winner  = str(row["winner"])
        team2   = str(row["team_2"])
        city    = str(row["city"])
        venue   = str(row["venue"])
        match_info[mid] = {
            "target": target,
            "winner": winner,
            "team2":  team2,
            "city":   city,
            "venue":  venue,
        }

    records = []

    for (match_id, inning), grp in bbb.groupby(["match_id", "inning"]):
        match_id = int(match_id)
        inning   = int(inning)

        if inning != 2:
            continue
        if match_id not in match_info:
            continue

        info         = match_info[match_id]
        target       = info["target"]
        batting_team = grp["batting_team"].iloc[0]
        bowling_team = grp["bowling_team"].iloc[0]
        result       = 1 if info["winner"] == batting_team else 0

        grp = grp.copy()
        grp["_ball_num"] = grp["ball_over"].apply(_over_to_balls)
        grp = grp.sort_values("_ball_num")

        cum_score   = 0
        cum_wickets = 0
        cum_legal   = 0

        for _, ball in grp.iterrows():
            runs_this = int(ball["total_runs"])
            is_legal  = _is_legal(ball)
            is_wicket = str(ball["player_dismissed"]).strip() not in ("", "nan")

            cum_score += runs_this
            if is_wicket:
                cum_wickets += 1
            if is_legal:
                cum_legal += 1

            # record from over 1 onwards
            if cum_legal >= 6:
                balls_left   = max(0, 120 - cum_legal)
                runs_left    = max(0, target - cum_score)
                crr          = round(cum_score / (cum_legal / 6), 2) if cum_legal > 0 else 0.0
                rrr          = round(runs_left / (balls_left / 6), 2) if balls_left > 0 else 0.0

                records.append({
                    "Batting_team":  batting_team,
                    "Bowling_team":  bowling_team,
                    "city":          info["city"],
                    "Runs_left":     runs_left,
                    "Balls_left":    balls_left,
                    "Wickets_left":  max(0, 10 - cum_wickets),
                    "Total_score":   target - 1,   # 1st innings total
                    "crr":           crr,
                    "rrr":           rrr,
                    "result":        result,
                })

    df = pd.DataFrame(records)
    return df


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare NPL training datasets")
    parser.add_argument("--ball-by-ball", default="data/NPL-2024.csv",
                        help="Path to ball-by-ball CSV")
    parser.add_argument("--matches",      default="data/NPL_matches.csv",
                        help="Path to matches CSV")
    parser.add_argument("--out-score",    default="data/score_data.csv",
                        help="Output path for score dataset")
    parser.add_argument("--out-win",      default="data/win_data.csv",
                        help="Output path for win dataset")
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)

    print(f"Reading ball-by-ball data from {args.ball_by_ball} ...")
    bbb = pd.read_csv(args.ball_by_ball)
    # normalise column names
    bbb.columns = [c.strip() for c in bbb.columns]

    print(f"Reading matches data from {args.matches} ...")
    matches = pd.read_csv(args.matches)
    matches.columns = [c.strip() for c in matches.columns]

    print("Building score dataset ...")
    score_df = build_score_dataset(bbb, matches)
    score_df.to_csv(args.out_score, index=False)
    print(f"  → {len(score_df)} rows saved to {args.out_score}")

    print("Building win dataset ...")
    win_df = build_win_dataset(bbb, matches)
    win_df.to_csv(args.out_win, index=False)
    print(f"  → {len(win_df)} rows saved to {args.out_win}")

    print("\nDone. Now run:")
    print("  python -m src.train --score-csv data/score_data.csv --win-csv data/win_data.csv")


if __name__ == "__main__":
    main()
