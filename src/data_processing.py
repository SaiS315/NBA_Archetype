
from pathlib import Path

import numpy as np
import pandas as pd

INPUT = Path("data/raw/player_stats.csv")
OUTPUT = Path("data/processed/offensive_features.csv")

# Initial eligibility rule; revisit during analysis
MIN_MINUTES = 500


def safe_divide(numerator, denominator):
    """Divide safely, returning NaN for zero denominators."""
    return numerator.div(denominator.replace(0, np.nan))


def load_data():
    df = pd.read_csv(INPUT)

    required = [
        "PLAYER_ID", "PLAYER_NAME", "MIN",
        "PTS", "AST", "TOV", "FGM", "FGA",
        "FG3M", "FG3A", "FTA", "OREB",
        "AST_PCT"
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df["PLAYER_ID"].duplicated().any():
        raise ValueError(
            "Duplicate players found. Verify season-level aggregation."
        )

    return df


def clean_data(df):
    df = df.copy()

    numeric_columns = [
        "MIN", "PTS", "AST", "TOV", "FGM",
        "FGA", "FG3M", "FG3A", "FTA",
        "OREB", "AST_PCT"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Exclude players with insufficient playing time
    df = df[df["MIN"] >= MIN_MINUTES].copy()

    # Essential statistics must be available
    df = df.dropna(subset=numeric_columns)

    # Remove logically inconsistent records
    df = df[
        (df["FGA"] >= df["FG3A"]) &
        (df["AST"] >= 0) &
        (df["TOV"] >= 0)
    ].copy()

    return df


def engineer_features(df):
    df = df.copy()

    # Scoring and shooting volume
    df["PTS_PER36"] = df["PTS"] / df["MIN"] * 36
    df["FG3A_PER36"] = df["FG3A"] / df["MIN"] * 36

    df["FG2A"] = df["FGA"] - df["FG3A"]
    df["FG2A_PER36"] = df["FG2A"] / df["MIN"] * 36

    # Playmaking and rebounding
    df["AST_PER36"] = df["AST"] / df["MIN"] * 36
    df["OREB_PER36"] = df["OREB"] / df["MIN"] * 36
    df["FTA_PER36"] = df["FTA"] / df["MIN"] * 36

    # Shooting efficiency
    df["EFG_PCT"] = safe_divide(
        df["FGM"] + 0.5 * df["FG3M"],
        df["FGA"]
    )

    df["TS_PCT"] = safe_divide(
        df["PTS"],
        2 * (df["FGA"] + 0.44 * df["FTA"])
    )

    # Ball security
    df["AST_TO"] = safe_divide(
        df["AST"],
        df["TOV"]
    )

    possessions_used = (
        df["FGA"] + 0.44 * df["FTA"]
        + df["AST"] + df["TOV"]
    )

    df["TO_RATIO"] = (
        safe_divide(df["TOV"] * 100, possessions_used)
    )

    return df


def save_features(df):
    features = [
        "PTS_PER36",
        "FG3A_PER36",
        "FG2A_PER36",
        "AST_PER36",
        "AST_PCT",
        "AST_TO",
        "TO_RATIO",
        "TS_PCT",
        "EFG_PCT",
        "FTA_PER36",
        "OREB_PER36"
    ]

    columns = ["PLAYER_ID", "PLAYER_NAME"] + features

    processed = df[columns].replace(
        [np.inf, -np.inf], np.nan
    )

    # Temporarily exclude incomplete player profiles.
    # We will examine excluded records before modeling.
    complete = processed.dropna(subset=features)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    complete.to_csv(OUTPUT, index=False)

    print("Processed players:", len(complete))
    print("Excluded incomplete profiles:",
          len(processed) - len(complete))
    print(complete.head())

    return complete


def main():
    df = load_data()
    print("Original players:", len(df))

    df = clean_data(df)
    print("Eligible players:", len(df))

    df = engineer_features(df)
    save_features(df)


if __name__ == "__main__":
    main()
