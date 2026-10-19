"""Cluster NBA offensive player profiles using standardized season features.

Run from the repository root: python src/clustering.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

INPUT = Path("data/processed/offensive_features.csv")
OUTPUT = Path("results")
FIGURES = OUTPUT / "figures"
RANDOM_STATE = 42
K_VALUES = range(2, 11)

FEATURES = [
    "PTS_PER36", "FG3A_PER36", "FG2A_PER36", "AST_PCT",
    "AST_TO", "TO_RATIO", "TS_PCT", "FTA_PER36", "OREB_PER36",
]


def load_features():
    df = pd.read_csv(INPUT)
    required = ["PLAYER_ID", "PLAYER_NAME", *FEATURES]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if df["PLAYER_ID"].duplicated().any():
        raise ValueError("Expected one season row per player (duplicate PLAYER_ID).")
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    valid = np.isfinite(df[FEATURES].to_numpy()).all(axis=1)
    if not valid.all():
        print(f"Dropping {(~valid).sum()} rows with missing or infinite feature values")
    df = df.loc[valid].copy().reset_index(drop=True)
    if len(df) < 11:
        raise ValueError("Need at least 11 valid players to evaluate k=2..10.")
    return df


def plot_line(x, y, title, ylabel, filename):
    plt.figure(figsize=(7, 4))
    plt.plot(x, y, marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(list(x))
    plt.tight_layout()
    plt.savefig(FIGURES / filename, dpi=180)
    plt.close()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    players = load_features()
    scaler = StandardScaler()
    X = scaler.fit_transform(players[FEATURES])

    metrics = []
    for k in K_VALUES:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels = model.fit_predict(X)
        sizes = np.bincount(labels, minlength=k)
        metrics.append({
            "k": k,
            "inertia": model.inertia_,
            "silhouette_score": silhouette_score(X, labels),
            "smallest_cluster": int(sizes.min()),
            "largest_cluster": int(sizes.max()),
        })

    evaluation = pd.DataFrame(metrics)
    evaluation.to_csv(OUTPUT / "k_selection_metrics.csv", index=False)
    print("\nK-means model comparison:\n")
    print(evaluation.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # Highest silhouette is a starting point, not necessarily the final basketball choice.
    best_k = int(evaluation.loc[evaluation["silhouette_score"].idxmax(), "k"])
    print(f"\nProvisional silhouette-based k: {best_k}")
    final_model = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=20)
    players["CLUSTER"] = final_model.fit_predict(X)

    player_output = players[["PLAYER_ID", "PLAYER_NAME", "CLUSTER", *FEATURES]].copy()
    player_output.to_csv(OUTPUT / "player_clusters.csv", index=False)

    profiles = players.groupby("CLUSTER")[FEATURES].mean()
    profiles.insert(0, "PLAYER_COUNT", players.groupby("CLUSTER").size())
    profiles.to_csv(OUTPUT / "cluster_profiles.csv")

    standardized_profiles = pd.DataFrame(X, columns=FEATURES).assign(
        CLUSTER=players["CLUSTER"].to_numpy()
    ).groupby("CLUSTER")[FEATURES].mean()
    standardized_profiles.to_csv(OUTPUT / "cluster_profiles_zscore.csv")

    # Closest players to each model centroid in standardized feature space.
    distance_to_centroid = np.linalg.norm(
        X - final_model.cluster_centers_[players["CLUSTER"].to_numpy()], axis=1
    )
    representatives = players[["PLAYER_ID", "PLAYER_NAME", "CLUSTER"]].copy()
    representatives["DISTANCE_TO_CENTROID"] = distance_to_centroid
    representatives = (
        representatives.sort_values("DISTANCE_TO_CENTROID")
        .groupby("CLUSTER", group_keys=False)
        .head(5)
        .sort_values(["CLUSTER", "DISTANCE_TO_CENTROID"])
    )
    representatives.to_csv(OUTPUT / "representative_players.csv", index=False)

    plot_line(evaluation["k"], evaluation["silhouette_score"],
              "Silhouette score by k", "Silhouette score", "silhouette_by_k.png")
    plot_line(evaluation["k"], evaluation["inertia"],
              "K-means inertia by k", "Inertia", "inertia_by_k.png")

    # PCA is for visualization only; clustering uses all nine standardized features.
    reduced = PCA(n_components=2).fit_transform(X)
    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1],
                          c=players["CLUSTER"], cmap="tab10", alpha=0.8, s=25)
    plt.xlabel("Principal component 1")
    plt.ylabel("Principal component 2")
    plt.title(f"Offensive player clusters (k={best_k}; PCA projection)")
    plt.colorbar(scatter, label="Cluster ID", ticks=range(best_k))
    plt.tight_layout()
    plt.savefig(FIGURES / "pca_clusters.png", dpi=180)
    plt.close()

    print("\nCluster profiles:\n", profiles.round(3).to_string())
    print("\nRepresentative players:\n", representatives.to_string(index=False))
    print(f"\nSaved outputs to {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
