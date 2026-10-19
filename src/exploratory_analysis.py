from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_PATH = Path('data/processed/offensive_features.csv')
FIGURE_DIR = Path('results/figures')
REPORT_DIR = Path('results')
FEATURES = [
    'PTS_PER36', 'FG3A_PER36', 'FG2A_PER36', 'AST_PER36',
    'AST_PCT', 'AST_TO', 'TO_RATIO', 'TS_PCT', 'EFG_PCT',
    'FTA_PER36', 'OREB_PER36',
]

def main():
    df = pd.read_csv(DATA_PATH)
    required = ['PLAYER_ID', 'PLAYER_NAME', *FEATURES]
    missing_columns = sorted(set(required) - set(df.columns))
    if missing_columns:
        raise ValueError(f'Missing columns: {missing_columns}')
    if df['PLAYER_ID'].duplicated().any():
        raise ValueError('Duplicate PLAYER_ID values found')

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    x = df[FEATURES].apply(pd.to_numeric, errors='coerce')
    print('Number of players:', len(df))
    print('\nMissing values per feature:\n', x.isna().sum().to_string())
    print('\nDescriptive statistics:\n', x.describe().T.round(3).to_string())
    x.describe().T.to_csv(REPORT_DIR / 'feature_summary.csv')

    corr = x.corr()
    corr.to_csv(REPORT_DIR / 'feature_correlations.csv')
    fig, ax = plt.subplots(figsize=(11, 9))
    img = ax.imshow(corr.values, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURES)), FEATURES, rotation=60, ha='right')
    ax.set_yticks(range(len(FEATURES)), FEATURES)
    for i in range(len(FEATURES)):
        for j in range(len(FEATURES)):
            value = corr.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f'{value:.2f}', ha='center', va='center', fontsize=8)
    fig.colorbar(img, ax=ax, label='Pearson correlation')
    ax.set_title('Offensive feature correlations')
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / 'correlation_heatmap.png', dpi=180)
    plt.close(fig)

    strong = []
    for i, a in enumerate(FEATURES):
        for b in FEATURES[i+1:]:
            value = corr.loc[a, b]
            if pd.notna(value) and abs(value) >= 0.8:
                strong.append((a, b, round(value, 3)))
    print('\nStrongly correlated pairs (|r| >= 0.80):')
    print(*(strong or ['None']), sep='\n')

    axes_per_row = 3
    rows = (len(FEATURES) + axes_per_row - 1) // axes_per_row
    fig, axes = plt.subplots(rows, axes_per_row, figsize=(15, 3.4 * rows))
    for ax, feature in zip(axes.flat, FEATURES):
        ax.hist(x[feature].dropna(), bins=25, edgecolor='white')
        ax.set_title(feature)
        ax.set_ylabel('Players')
    for ax in axes.flat[len(FEATURES):]:
        ax.set_visible(False)
    fig.suptitle('Distributions of candidate offensive features')
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / 'feature_distributions.png', dpi=180)
    plt.close(fig)

    z = (x - x.mean()) / x.std(ddof=0).replace(0, np.nan)
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.boxplot([z[col].dropna().values for col in FEATURES], tick_labels=FEATURES)
    ax.tick_params(axis='x', rotation=55)
    ax.set_ylabel('Z-score')
    ax.set_title('Feature outliers (standardized for comparison)')
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / 'standardized_boxplots.png', dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x['AST_PER36'], x['PTS_PER36'], alpha=0.55, s=20)
    ax.set_xlabel('Assists per 36 minutes')
    ax.set_ylabel('Points per 36 minutes')
    ax.set_title('Scoring vs. playmaking')
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / 'scoring_vs_playmaking.png', dpi=180)
    plt.close(fig)

    print('\nSaved charts to:', FIGURE_DIR)
    print('Saved statistical summaries to:', REPORT_DIR)

if __name__ == '__main__':
    main()
