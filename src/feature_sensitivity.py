"""Step 10: test robustness of the five NBA offensive archetypes.
Run from project root: python src/feature_sensitivity.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

DATA = Path('data/processed/offensive_features.csv')
PRIOR = Path('results/refinement/player_assignments_k3_to_k6.csv')
OUT = Path('results/sensitivity')
FEATURES = [
    'PTS_PER36', 'FG3A_PER36', 'FG2A_PER36', 'AST_PCT',
    'AST_TO', 'TO_RATIO', 'TS_PCT', 'FTA_PER36', 'OREB_PER36'
]
EXPERIMENTS = {
    'baseline': FEATURES,
    'without_AST_TO': [f for f in FEATURES if f != 'AST_TO'],
    'without_PTS_PER36': [f for f in FEATURES if f != 'PTS_PER36'],
    'EFG_instead_of_TS': ['EFG_PCT' if f == 'TS_PCT' else f for f in FEATURES],
}

def main():
    if not DATA.exists():
        raise FileNotFoundError(f'{DATA} missing. Run data_processing.py first.')
    df = pd.read_csv(DATA)
    required = ['PLAYER_ID', 'PLAYER_NAME', *set().union(*EXPERIMENTS.values())]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Missing columns: {missing}')
    if df['PLAYER_ID'].duplicated().any():
        raise ValueError('Duplicate PLAYER_ID values: one row per player required.')
    all_features = sorted(set().union(*EXPERIMENTS.values()))
    df[all_features] = df[all_features].apply(pd.to_numeric, errors='coerce')
    good = np.isfinite(df[all_features].to_numpy(dtype=float)).all(axis=1)
    df = df.loc[good].copy().reset_index(drop=True)
    if len(df) < 10:
        raise ValueError('Not enough complete player rows.')
    OUT.mkdir(parents=True, exist_ok=True)
    assignments = df[['PLAYER_ID', 'PLAYER_NAME']].copy()
    rows = []
    base_labels = None
    for name, cols in EXPERIMENTS.items():
        X = StandardScaler().fit_transform(df[cols])
        labels = KMeans(n_clusters=5, n_init=20, random_state=42).fit_predict(X)
        assignments[name] = labels
        if name == 'baseline':
            base_labels = labels
        counts = np.bincount(labels, minlength=5)
        rows.append({
            'experiment': name,
            'features': len(cols),
            'players': len(df),
            'silhouette_score': round(float(silhouette_score(X, labels)), 4),
            'ARI_vs_baseline': round(float(adjusted_rand_score(base_labels, labels)), 4),
            'smallest_cluster': int(counts.min()),
            'largest_cluster': int(counts.max()),
        })
    if PRIOR.exists():
        prior = pd.read_csv(PRIOR)[['PLAYER_ID', 'cluster_k5']]
        paired = assignments[['PLAYER_ID', 'baseline']].merge(
            prior, on='PLAYER_ID', how='inner', validate='one_to_one'
        )
        if len(paired) == len(assignments) == len(prior):
            ari = adjusted_rand_score(paired['cluster_k5'], paired['baseline'])
            print(f'Baseline vs previous k=5 assignments ARI: {ari:.4f}')
        else:
            print('Warning: Player sets differ from previous k=5 run; comparisons may not be exact.')
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUT / 'sensitivity_comparison.csv', index=False)
    assignments.to_csv(OUT / 'sensitivity_assignments.csv', index=False)
    print('\n' + comparison.to_string(index=False))
    print(f'\nSaved outputs to {OUT.resolve()}')
    print('ARI compares groupings regardless of arbitrary cluster numbers.')

if __name__ == '__main__':
    main()
