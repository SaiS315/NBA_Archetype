"""Compare k=3..6 NBA offensive archetype clustering solutions.

Run from repository root: python src/refine_clusters.py
Requires: data/processed/offensive_features.csv
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

SOURCE = Path('data/processed/offensive_features.csv')
OUT = Path('results/refinement')
FIG = OUT / 'figures'
KS = (3, 4, 5, 6)
SEED = 42
FEATURES = [
    'PTS_PER36', 'FG3A_PER36', 'FG2A_PER36', 'AST_PCT',
    'AST_TO', 'TO_RATIO', 'TS_PCT', 'FTA_PER36', 'OREB_PER36',
]


def load_data():
    if not SOURCE.exists():
        raise FileNotFoundError(f'Missing {SOURCE}. Run data_processing.py first.')
    df = pd.read_csv(SOURCE)
    required = ['PLAYER_ID', 'PLAYER_NAME', *FEATURES]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    if df['PLAYER_ID'].isna().any() or df['PLAYER_ID'].duplicated().any():
        raise ValueError('Expected one valid unique PLAYER_ID per row.')
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    keep = np.isfinite(df[FEATURES].to_numpy(dtype=float)).all(axis=1)
    print(f'Invalid/incomplete rows excluded: {(~keep).sum()}')
    df = df.loc[keep].copy().reset_index(drop=True)
    if len(df) <= max(KS):
        raise ValueError('Insufficient complete player rows for clustering.')
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df = load_data()
    X = StandardScaler().fit_transform(df[FEATURES])
    coords = PCA(n_components=2).fit_transform(X)  # visualization only

    comparisons = []
    assignments = df[['PLAYER_ID', 'PLAYER_NAME']].copy()
    models = {}
    for k in KS:
        model = KMeans(n_clusters=k, n_init=20, random_state=SEED)
        labels = model.fit_predict(X)
        models[k] = model
        assignments[f'cluster_k{k}'] = labels
        counts = np.bincount(labels, minlength=k)
        silhouette = silhouette_score(X, labels)

        # Robustness to alternate random initializations; labels matched via ARI.
        alternative_labels = [
            KMeans(n_clusters=k, n_init=20, random_state=s).fit_predict(X)
            for s in (7, 21, 99, 123, 2026)
        ]
        stability = float(np.mean([
            adjusted_rand_score(labels, other) for other in alternative_labels
        ]))
        comparisons.append({
            'k': k, 'silhouette_score': silhouette,
            'inertia': model.inertia_,
            'smallest_cluster': int(counts.min()),
            'largest_cluster': int(counts.max()),
            'largest_cluster_pct': float(counts.max() / len(df) * 100),
            'mean_ARI_across_seeds': stability,
        })

        data = df.copy()
        data['CLUSTER'] = labels
        profile = data.groupby('CLUSTER')[FEATURES].mean()
        profile.insert(0, 'PLAYER_COUNT', data.groupby('CLUSTER').size())
        profile.to_csv(OUT / f'cluster_profiles_k{k}.csv')

        zprofile = pd.DataFrame(X, columns=FEATURES).assign(CLUSTER=labels)
        zprofile.groupby('CLUSTER')[FEATURES].mean().to_csv(
            OUT / f'cluster_profiles_zscore_k{k}.csv'
        )

        distances = np.linalg.norm(X - model.cluster_centers_[labels], axis=1)
        reps = data[['PLAYER_ID', 'PLAYER_NAME', 'CLUSTER']].copy()
        reps['DISTANCE_TO_CENTROID'] = distances
        reps.sort_values(['CLUSTER', 'DISTANCE_TO_CENTROID']).groupby(
            'CLUSTER', group_keys=False
        ).head(5).to_csv(OUT / f'representative_players_k{k}.csv', index=False)

        plt.figure(figsize=(9, 6))
        scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels,
                              cmap='tab10', s=24, alpha=.75)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title(f'K-means: k={k} (PCA projection, not model space)')
        plt.colorbar(scatter, ticks=range(k), label='Cluster ID')
        plt.tight_layout()
        plt.savefig(FIG / f'pca_k{k}.png', dpi=170)
        plt.close()

    pd.DataFrame(comparisons).to_csv(OUT / 'model_comparison.csv', index=False)
    assignments.to_csv(OUT / 'player_assignments_k3_to_k6.csv', index=False)
    for k in (4, 5, 6):
        table = pd.crosstab(
            assignments['cluster_k3'], assignments[f'cluster_k{k}'],
            rownames=['k3_cluster'], colnames=[f'k{k}_cluster']
        )
        table.to_csv(OUT / f'k3_vs_k{k}_crosswalk.csv')

    print(f'Players analyzed: {len(df)}')
    print('\nModel comparison:')
    print(pd.DataFrame(comparisons).round(3).to_string(index=False))
    for k in KS:
        print(f'\nk={k} cluster sizes: ' +
              str(np.bincount(models[k].labels_, minlength=k).tolist()))
    print(f'\nOutputs saved in: {OUT.resolve()}')
    print('Note: Cluster IDs are arbitrary and cannot be compared by number across k.')


if __name__ == '__main__':
    main()
