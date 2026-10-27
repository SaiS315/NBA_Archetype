
from pathlib import Path
from time import sleep

import pandas as pd
from nba_api.stats.endpoints import commonplayerinfo

ROOT = Path(__file__).resolve().parent.parent

INPUT = ROOT / "data/processed/player_two_way_profiles_tracking.csv"
OUTPUT = ROOT / "data/processed/player_positions.csv"

def main():
    players = pd.read_csv(INPUT)

    if not players["PLAYER_ID"].is_unique:
        raise ValueError("Duplicate PLAYER_ID values")

    records = []

    for i, player in players.iterrows():
        player_id = int(player["PLAYER_ID"])
        name = player["PLAYER_NAME"]

        try:
            response = commonplayerinfo.CommonPlayerInfo(
                player_id=player_id,
                timeout=60
            )

            info = response.get_data_frames()[0]

            position = (
                info.iloc[0]["POSITION"]
                if not info.empty
                else None
            )

            records.append({
                "PLAYER_ID": player_id,
                "PLAYER_NAME": name,
                "POSITION": position
            })

        except Exception as error:
            print(f"Error retrieving {name}: {error}")

            records.append({
                "PLAYER_ID": player_id,
                "PLAYER_NAME": name,
                "POSITION": None
            })

        if (i + 1) % 25 == 0:
            print(f"Processed {i + 1}/{len(players)} players")

        sleep(0.6)

    positions = pd.DataFrame(records)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    positions.to_csv(OUTPUT, index=False)

    print("\nPlayers:", len(positions))
    print("Missing positions:", positions["POSITION"].isna().sum())

    print("\nPosition distribution:")
    print(positions["POSITION"].value_counts(dropna=False))

    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()
