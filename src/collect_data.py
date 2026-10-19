
from pathlib import Path
from time import sleep

import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats

SEASON = "2025-26"
OUTPUT = Path("data/raw")


def fetch_stats(measure_type):
    response = leaguedashplayerstats.LeagueDashPlayerStats(
        season=SEASON,
        season_type_all_star="Regular Season",
        measure_type_detailed_defense=measure_type,
        per_mode_detailed="Totals",
        timeout=60,
    )

    return response.get_data_frames()[0]


# Collect traditional box-score statistics
base = fetch_stats("Base")

# Give the API a short break
sleep(2)

# Collect advanced statistics
advanced = fetch_stats("Advanced")

advanced_columns = [
    "PLAYER_ID",
    "AST_PCT",
    "AST_TO",
    "AST_RATIO",
    "TM_TOV_PCT",
    "EFG_PCT",
    "TS_PCT",
]

# Check available columns before selecting them
available = [
    col for col in advanced_columns
    if col in advanced.columns
]

print("Available advanced columns:", available)
print("Missing columns:",
      set(advanced_columns) - set(available))

# Combine using the unique player identifier
players = base.merge(
    advanced[available],
    on="PLAYER_ID",
    how="left",
    validate="one_to_one",
)

OUTPUT.mkdir(parents=True, exist_ok=True)

players.to_csv(
    OUTPUT / "player_stats.csv",
    index=False,
)

print("Dataset shape:", players.shape)
print(players.head())
