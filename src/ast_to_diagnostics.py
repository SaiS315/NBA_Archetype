"""Step 11: Investigate sensitivity to AST/TO without modifying raw data.
Run from repository root: python src/ast_to_diagnostics.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

DATA = Path('data/processed/offensive_features.csv')
OUT = Path('results/ast_to_diagnostics')
FEATURES = ['PTS_PER36','FG3A_PER36','FG2A_PER36','AST_PCT','AST_TO',
            'TO_RATIO','TS_PCT','FTA_PER36','OREB_PER36']


def fit(df, feature_cols):
    X = StandardScaler().fit_transform(df[feature_cols])
    model = KMeans(n_clusters=5, n_init=20, random_state=42)
    labels = model.fit_predict(X)
    return labels, silhouette_score(X, labels)


def main():
    df = pd.read_csv(DATA)
    expected = ['PLAYER_ID','PLAYER_NAME'] + FEATURES
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    df = df.copy()
    df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors='coerce')
    valid = np.isfinite(df[FEATURES].to_numpy()).all(axis=1)
    if not valid.all():
        print(f'Excluding {sum(~valid)} rows with incomplete inputs')
        df = df.loc[valid].copy()
    if df['PLAYER_ID'].duplicated().any():
        raise ValueError('Expected one row per PLAYER_ID')
    OUT.mkdir(parents=True, exist_ok=True)

    # Describe extremes and inspect which players they represent.
    print('AST/TO distribution:')
    print(df['AST_TO'].describe(percentiles=[.01,.05,.25,.50,.75,.95,.99]).to_string())
    df.nlargest(20, 'AST_TO')[['PLAYER_NAME','AST_TO','AST_PCT','TO_RATIO']].to_csv(
        OUT/'highest_ast_to_players.csv', index=False)

    baseline, bs = fit(df, FEATURES)
    results = [{'experiment':'baseline','ARI_vs_baseline':1.0,'silhouette':bs}]
    assignments = df[['PLAYER_ID','PLAYER_NAME']].copy()
    assignments['baseline'] = baseline

    # Limit AST/TO extremes, preserving the same sample and all nine features.
    for percentile in [0.95, 0.99]:
        alt = df.copy()
        cap = float(alt['AST_TO'].quantile(percentile))
        alt['AST_TO'] = alt['AST_TO'].clip(upper=cap)
        labels, score = fit(alt, FEATURES)
        key = f'cap_ast_to_p{int(percentile*100)}'
        results.append({'experiment':key,
                        'ARI_vs_baseline':adjusted_rand_score(baseline, labels),
                        'silhouette':score})
        assignments[key] = labels
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT/'ast_to_comparison.csv', index=False)
    assignments.to_csv(OUT/'ast_to_assignments.csv', index=False)
    print('\n',results_df.to_string(index=False))
    print('Saved results to', OUT)

if __name__ == '__main__':
    main()
