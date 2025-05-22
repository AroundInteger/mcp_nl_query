import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.features.feature_engineering import FeatureEngineer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
import re

def filter_outcome_correlated_features(df):
    """
    Remove features that are obvious correlates of match outcome.
    These are features that are direct results of the match outcome.
    """
    outcome_correlated_features = [
        'points_scored', 'points_conceded',  # Direct result of match
        'tries_scored', 'tries_conceded',    # Direct result of match
        'conversions_scored', 'conversions_conceded',  # Direct result of match
        'penalties_scored', 'penalties_conceded',      # Direct result of match
        'drop_goals_scored', 'drop_goals_conceded',    # Direct result of match
        'outcome', 'outcome_numeric'         # Target variable
    ]
    
    return [col for col in df.columns if col not in outcome_correlated_features]

def categorize_features(features):
    """
    Categorize features into absolute (_i) and relative (_r) metrics.
    """
    absolute_features = [f for f in features if f.endswith('_i')]
    relative_features = [f for f in features if f.endswith('_r')]
    other_features = [f for f in features if not (f.endswith('_i') or f.endswith('_r'))]
    
    return {
        'absolute': absolute_features,
        'relative': relative_features,
        'other': other_features
    }

def remove_away_duplicates(df):
    """
    Remove away versions of matches to avoid duplication.
    Keep only home team records.
    """
    return df[df['match_location'] == 'home']

def strip_suffix(feature_list):
    # Remove _i or _r suffix for display
    return [re.sub(r'(_i|_r)$', '', f) for f in feature_list]

st.title("Rugby Union MCP Dashboard")

# Data Processing Notes
st.sidebar.header("Data Processing Notes")
st.sidebar.markdown("""
### Data Structure Decisions
1. **Home Team Records Only**
   - Only home team records are used to avoid duplicate matches
   - Each match appears once in the dataset
   - This ensures statistical independence of observations

2. **Feature Types**
   - **Absolute Metrics (_i)**: Raw values provided by data provider
   - **Relative Metrics (_r)**: Home team value minus away team value
   - These provide different perspectives on team performance

3. **Feature Selection**
   - Obvious outcome correlates (e.g., points scored) are excluded
   - These features are direct results of the match outcome
   - Including them would make the analysis circular

### Analysis Approach
- SVD projection is computed at league/season level
- Team-level analysis focuses on performance patterns
- Predictions use both absolute and relative metrics

**Note:** For clarity, feature names are displayed without their `_i` or `_r` suffixes, but the analysis uses the full feature names internally.
""")

# File selector
season = st.selectbox("Select season", ['21_22', '22_23', '23_24'])
if season == '21_22':
    df = pd.read_csv("data/raw/21_22_anonymized.csv")
else:
    df = pd.read_csv(f"data/raw/{season}.csv")

# Remove away duplicates
df = remove_away_duplicates(df)
st.caption(f"Note: Only home team records are shown to avoid duplication. Total matches: {len(df)}")

# Metric type selector
metric_type = st.radio(
    "Choose metric type for analysis:",
    ("Absolute metrics only", "Relative metrics only", "Both")
)

def filter_by_metric_type(features, metric_type):
    if metric_type == "Absolute metrics only":
        return [f for f in features if f.endswith('_i')]
    elif metric_type == "Relative metrics only":
        return [f for f in features if f.endswith('_r')]
    else:
        return features  # Both

# --- SVD Projection (2D) at League/Season Level ---
st.subheader("SVD Projection (2D) - League/Season Level")
st.caption("""
This projection is computed using all teams in the selected season.
- Each point represents a match
- Colors indicate match outcome (win/loss)
- The projection helps identify patterns in team performance across the league
""")

# Prepare data for SVD (exclude the target column)
df['outcome_numeric'] = df['outcome'].map({'win': 1, 'loss': 0})
df = df.dropna(subset=['outcome_numeric'])
categorical_columns = ['team', 'match_location']
feature_engineer = FeatureEngineer()

# Filter out obvious outcome-correlated features
non_feature_cols = ['matchid', 'team', 'outcome', 'season']
feature_cols = [col for col in df.columns if col not in non_feature_cols]
df_features = df[feature_cols + ['team', 'match_location']]
filtered_features = filter_outcome_correlated_features(df_features)
filtered_features = filter_by_metric_type(filtered_features, metric_type)

# Categorize features
feature_categories = categorize_features(filtered_features)

# Display feature categories (with suffixes stripped)
st.subheader("Features Used in Analysis")
st.markdown("""
### Feature Categories

#### Absolute Metrics (_i)
These are raw values provided by the data provider, representing actual measurements or counts.
""")
st.write(strip_suffix(feature_categories['absolute']))

st.markdown("""
#### Relative Metrics (_r)
These represent the difference between home and away team values (home - away).
They show the relative advantage/disadvantage of the home team.
""")
st.write(strip_suffix(feature_categories['relative']))

st.markdown("""
#### Other Features
Additional features that don't fall into the above categories.
""")
st.write(strip_suffix(feature_categories['other']))

# For league/season-level analysis
cols_to_use = filtered_features + [col for col in categorical_columns if col not in filtered_features]
df_processed = feature_engineer.preprocess_features(
    df_features[cols_to_use],
    target_column='outcome_numeric',
    categorical_columns=categorical_columns
)

X_svd = df_processed.drop(columns=['outcome_numeric'], errors='ignore')
y_svd = df['outcome_numeric']
teams_svd = df['team']

svd = TruncatedSVD(n_components=2, random_state=42)
X_svd_proj = svd.fit_transform(X_svd)

fig4, ax4 = plt.subplots(figsize=(8, 6))
scatter = ax4.scatter(X_svd_proj[:, 0], X_svd_proj[:, 1], c=y_svd, cmap='coolwarm', alpha=0.7)
legend1 = ax4.legend(*scatter.legend_elements(), title='Outcome')
ax4.add_artist(legend1)
ax4.set_xlabel('SVD Component 1')
ax4.set_ylabel('SVD Component 2')
ax4.set_title(f'SVD Projection (2D) for {season} (All Teams)')
st.pyplot(fig4)

# --- Team selector for summary, win rate, and prediction ---
teams = sorted(df['team'].unique())
selected_team = st.selectbox("Select team for summary and prediction", teams)

# Filter data for selected team
df_team = df[df['team'] == selected_team]

# Data preview
st.subheader(f"Data Preview for {selected_team}")
st.caption("""
Showing home matches only. Each row represents a single match.
- _i columns show absolute values
- _r columns show relative advantage over away team
""")
st.dataframe(df_team.head())

# Download button for filtered team data
csv = df_team.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download team data as CSV",
    data=csv,
    file_name=f'{selected_team}_matches.csv',
    mime='text/csv',
)

# Feature engineering for team-level analysis
df_team['outcome_numeric'] = df_team['outcome'].map({'win': 1, 'loss': 0})
df_team = df_team.dropna(subset=['outcome_numeric'])
categorical_columns = ['team', 'match_location']
feature_engineer = FeatureEngineer()

# Filter out obvious outcome-correlated features for team analysis
non_feature_cols = ['matchid', 'team', 'outcome', 'season']
feature_cols = [col for col in df_team.columns if col not in non_feature_cols]
df_features = df_team[feature_cols + ['team', 'match_location']]
filtered_features = filter_outcome_correlated_features(df_features)
filtered_features = filter_by_metric_type(filtered_features, metric_type)

# For team-level analysis
cols_to_use = filtered_features + [col for col in categorical_columns if col not in filtered_features]
df_processed = feature_engineer.preprocess_features(
    df_features[cols_to_use],
    target_column='outcome_numeric',
    categorical_columns=categorical_columns
)
if 'outcome_numeric' not in df_processed.columns:
    df_processed['outcome_numeric'] = df_team['outcome_numeric'].values

# Feature importance
importance_scores = feature_engineer.get_feature_importance(
    df_processed,
    target_column='outcome_numeric'
)
st.subheader("Feature Importance (Team Level)")
st.caption("""
Note: Obvious outcome-correlated features (e.g., points scored) have been excluded.
This analysis shows which pre-match and in-game factors most strongly influence match outcomes.
- _i features show importance of absolute values
- _r features show importance of relative advantages
Feature names are displayed without their _i/_r suffixes for clarity.
""")
fig, ax = plt.subplots(figsize=(10, 4))
# Plot with stripped feature names
display_names = [re.sub(r'(_i|_r)$', '', f) for f in importance_scores.head(10).index]
importance_scores.head(10).plot(kind='bar', ax=ax)
ax.set_xticklabels(display_names, rotation=45, ha='right')
st.pyplot(fig)

# Correlation heatmap
st.subheader("Correlation Heatmap (Team Level)")
st.caption("""
Shows relationships between the most important features.
Helps identify which factors tend to occur together.
- Positive correlations (red) indicate features that increase together
- Negative correlations (blue) indicate features that move in opposite directions
Feature names are displayed without their _i/_r suffixes for clarity.
""")
selected_features = importance_scores.head(10).index.tolist()
display_corr_names = [re.sub(r'(_i|_r)$', '', f) for f in selected_features]
corr = df_processed[selected_features].corr()
fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax2,
            xticklabels=display_corr_names, yticklabels=display_corr_names)
st.pyplot(fig2)

# Win rate
st.subheader("Win Rate")
st.caption("Based on home matches only")
win_rate = df_team['outcome_numeric'].mean()
st.metric("Win Rate", f"{win_rate:.2%}")

# --- Prediction Section ---
st.subheader("Predict Match Outcome")
st.caption("""
Predictions are based on pre-match and in-game indicators only.
Select a match row to see the predicted probability of winning.
The model considers both absolute (_i) and relative (_r) metrics.
Feature names are displayed without their _i/_r suffixes for clarity.
""")

# Prepare data for prediction
if len(df_processed) > 0:
    X = df_processed.drop(columns=['outcome_numeric'])
    y = df_processed['outcome_numeric']
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    # Let user select a row to predict
    row_idx = st.number_input("Select match row for prediction (0 = first row)", min_value=0, max_value=len(X)-1, value=0)
    input_row = X.iloc[[row_idx]]

    pred_proba = model.predict_proba(input_row)[0][1]
    st.write(f"**Predicted probability of win for {selected_team} in selected match:** {pred_proba:.2%}")
else:
    st.write("Not enough data for prediction.")