import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.data.data_loader import SportsDataLoader
from src.features.feature_engineering import FeatureEngineer

plt.style.use('seaborn')
sns.set_palette('husl')

# 1. Load and Combine Data from Multiple Seasons
seasons = ['21_22', '22_23', '23_24']
dfs = []
for season in seasons:
    try:
        df = pd.read_csv(f'data/raw/{season}.csv')
        df['season'] = season
        dfs.append(df)
        print(f"Loaded {season} data: {len(df)} rows")
    except FileNotFoundError:
        print(f"No data file found for season {season}")
df = pd.concat(dfs, ignore_index=True)
print(f'\nTotal combined data: {len(df)} rows')

# 2. Data Cleaning and Preparation
df['outcome_numeric'] = df['outcome'].map({'win': 1, 'loss': 0})
df = df.dropna(subset=['outcome_numeric'])
categorical_columns = ['team', 'match_location']
print('Missing values per column:')
print(df.isnull().sum())
print(df.head())

# 3. Feature Engineering and Selection
feature_engineer = FeatureEngineer()
non_feature_cols = ['matchid', 'team', 'outcome', 'season']
feature_cols = [col for col in df.columns if col not in non_feature_cols]
df_features = df[feature_cols + ['team', 'match_location']]
df_processed = feature_engineer.preprocess_features(
    df_features,
    target_column='outcome_numeric',
    categorical_columns=categorical_columns
)
selected_features = feature_engineer.mrmr_feature_selection(
    df_processed,
    target_column='outcome_numeric',
    n_features=10
)
print('Selected features:')
print(selected_features)

# 4. Feature Importance Analysis
importance_scores = feature_engineer.get_feature_importance(
    df_processed,
    target_column='outcome_numeric'
)
plt.figure(figsize=(12, 6))
importance_scores.plot(kind='bar')
plt.title('Feature Importance Scores')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 5. Correlation Analysis
correlation_matrix = df_processed[selected_features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()

# 6. Season-wise Analysis
season_stats = df.groupby('season').agg({
    'outcome_numeric': ['count', 'mean'],
})
print('Season-wise Statistics:')
print(season_stats)
plt.figure(figsize=(10, 6))
season_stats['outcome_numeric']['mean'].plot(kind='bar')
plt.title('Win Rate by Season')
plt.ylabel('Win Rate')
plt.tight_layout()
plt.show()