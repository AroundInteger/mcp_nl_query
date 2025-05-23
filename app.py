import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.features.feature_engineering import FeatureEngineer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
import re
import io

# --- Utility Functions ---
def filter_outcome_correlated_features(df):
    """
    Remove features that are obvious correlates of match outcome.
    These are features that are direct results of the match outcome.
    """
    # Only explicitly list columns not covered by the dynamic search
    outcome_correlated_features = [
        'outcome', 'outcome_numeric'
    ]
    # Dynamically remove any columns containing these terms
    terms_to_remove = [
        'final_points', 'yellow_cards', 'red_cards',
        'points_scored', 'points_conceded',
        'tries_scored', 'tries_conceded',
        'conversions_scored', 'conversions_conceded',
        'penalties_scored', 'penalties_conceded',
        'drop_goals_scored', 'drop_goals_conceded'
    ]
    for term in terms_to_remove:
        outcome_correlated_features.extend([col for col in df.columns if term in col.lower()])
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

# --- Streamlit App ---
st.title("Rugby Union MCP Dashboard")

# Section: Data Selection
st.header("1. Data Selection")
season = st.selectbox(
    "Select season",
    ['21_22', '22_23', '23_24'],
    help="Choose which rugby season's data to analyze."
)
feature_type = st.radio(
    'Select feature type to include in analysis:',
    ['Absolute (_i)', 'Relative (_r)', 'Both'],
    index=2,
    help="Choose which type of features to include in the analysis."
)
st.markdown("---")

# Load and preprocess data
@st.cache_data(show_spinner=False)
def load_data(season):
    df = pd.read_csv(f"data/raw/{season}_anonymized.csv")
    return df

df = load_data(season)
df = remove_away_duplicates(df)
df['outcome_numeric'] = df['outcome'].map({'win': 1, 'loss': 0})
st.caption(f"Note: Only home team records are shown to avoid duplication. Total matches: {len(df)}")

# Prepare features
non_feature_cols = ['matchid', 'team', 'outcome', 'season']
feature_cols = [col for col in df.columns if col not in non_feature_cols]
df_features = df[feature_cols + ['team', 'match_location']]
filtered_features = filter_outcome_correlated_features(df_features)
feature_categories = categorize_features(filtered_features)

# Section: Feature Selection
st.header("2. Feature Selection")
# --- Feature selection based on toggle ---
if feature_type == 'Absolute (_i)':
    selected_features_list = feature_categories['absolute'] + feature_categories['other']
elif feature_type == 'Relative (_r)':
    selected_features_list = feature_categories['relative'] + feature_categories['other']
else:  # Both
    selected_features_list = feature_categories['absolute'] + feature_categories['relative'] + feature_categories['other']

# Show features being used
with st.expander("Show features used in analysis", expanded=False):
    st.markdown("**Absolute Metrics (_i):**")
    st.write(strip_suffix([f for f in selected_features_list if f.endswith('_i')]))
    st.markdown("**Relative Metrics (_r):**")
    st.write(strip_suffix([f for f in selected_features_list if f.endswith('_r')]))
    st.markdown("**Other Features:**")
    st.write(strip_suffix([f for f in selected_features_list if not (f.endswith('_i') or f.endswith('_r'))]))

st.markdown("---")

# Data Processing Notes (sidebar)
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

# --- Feature Engineering ---
categorical_columns = ['team', 'match_location']
feature_engineer = FeatureEngineer()
df_processed = feature_engineer.preprocess_features(
    df_features[selected_features_list],
    target_column='outcome_numeric',
    categorical_columns=categorical_columns
)

# Section: SVD Projection
st.header("3. SVD Projection (2D)")
st.caption("""
This projection is computed using all teams in the selected season.
- Each point represents a match
- Colors indicate match outcome (win/loss)
- The projection helps identify patterns in team performance across the league
""")

# Prepare data for SVD (exclude the target column)
X_svd = df_processed.drop(columns=['outcome_numeric'], errors='ignore')
if 'outcome_numeric' in df_processed.columns:
    y_svd = df_processed['outcome_numeric']
else:
    y_svd = df.loc[df_processed.index, 'outcome_numeric']
teams_svd = df['team']

if len(X_svd) > 1:
    svd = TruncatedSVD(n_components=2, random_state=42)
    X_svd_proj = svd.fit_transform(X_svd)
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    
    # Create scatter plot
    scatter = ax4.scatter(X_svd_proj[:, 0], X_svd_proj[:, 1], c=y_svd, cmap='coolwarm', alpha=0.7)
    
    # Create custom legend
    unique_outcomes = pd.Series(y_svd).dropna().unique()
    if len(unique_outcomes) > 0:
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor='red' if outcome == 1 else 'blue',
                      markersize=10, label='Win' if outcome == 1 else 'Loss')
            for outcome in unique_outcomes
        ]
        ax4.legend(handles=legend_elements, title='Outcome')
    
    ax4.set_xlabel('SVD Component 1')
    ax4.set_ylabel('SVD Component 2')
    ax4.set_title(f'SVD Projection (2D) for {season} (All Teams)')
    
    # Display the plot
    st.pyplot(fig4)
    
    # Save the plot to a bytes buffer for download
    buf = io.BytesIO()
    fig4.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    
    # Add download button
    st.download_button(
        "Download SVD Plot as PNG",
        data=buf,
        file_name="svd_projection.png",
        mime="image/png"
    )
    
    # Clean up
    plt.close(fig4)
else:
    st.warning("Not enough data for SVD projection.")

st.markdown("---")

# Section: Team Summary, Win Rate, and Prediction
st.header("4. Team Summary, Win Rate, and Prediction")
teams = sorted(df['team'].unique())
selected_team = st.selectbox("Select team for summary and prediction", teams, help="Choose a team to view detailed stats and predictions.")
df_team = df[df['team'] == selected_team]

# Data preview in expander
with st.expander(f"Show data preview for {selected_team}", expanded=False):
    st.caption("Showing home matches only. Each row represents a single match. _i columns show absolute values, _r columns show relative advantage over away team.")
    st.dataframe(df_team.head())
    st.download_button(
        label="Download team data as CSV",
        data=df_team.to_csv(index=False).encode('utf-8'),
        file_name=f'{selected_team}_matches.csv',
        mime='text/csv',
    )

# --- Team-level Feature Engineering ---
df_team['outcome_numeric'] = df_team['outcome'].map({'win': 1, 'loss': 0})
df_team = df_team.dropna(subset=['outcome_numeric'])
feature_engineer = FeatureEngineer()
df_features = df_team[[col for col in df_team.columns if col in selected_features_list] + ['team', 'match_location']]
df_processed = feature_engineer.preprocess_features(
    df_features,
    target_column='outcome_numeric',
    categorical_columns=categorical_columns
)

# --- Feature Importance ---
if 'outcome_numeric' not in df_processed.columns:
    df_for_importance = df_processed.copy()
    df_for_importance['outcome_numeric'] = df_team['outcome_numeric'].values
else:
    df_for_importance = df_processed

if len(df_for_importance) > 1:
    importance_scores = feature_engineer.get_feature_importance(
        df_for_importance,
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
    display_names = [re.sub(r'(_i|_r)$', '', f) for f in importance_scores.head(10).index]
    importance_scores.head(10).plot(kind='bar', ax=ax)
    ax.set_xticklabels(display_names, rotation=45, ha='right')
    
    # Display the plot
    st.pyplot(fig)
    
    # Save the plot to a bytes buffer for download
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    
    # Add download button
    st.download_button(
        "Download Feature Importance Plot as PNG",
        data=buf,
        file_name="feature_importance.png",
        mime="image/png"
    )
    
    # Clean up
    plt.close(fig)
else:
    st.warning("Not enough data for feature importance analysis.")

# --- Correlation Heatmap ---
if len(df_for_importance) > 1:
    st.subheader("Correlation Heatmap (Team Level)")
    st.caption("""
    Shows relationships between the most important features.
    Helps identify which factors tend to occur together.
    - Positive correlations (red) indicate features that increase together
    - Negative correlations (blue) indicate features that move in opposite directions
    Feature names are displayed without their _i/_r suffixes for clarity.
    """)
    selected_features = importance_scores.head(10).index.tolist()
    
    # Filter out non-numeric columns
    numeric_features = df_for_importance[selected_features].select_dtypes(include=['float64', 'int64']).columns
    
    if len(numeric_features) > 0:
        # Create a copy of the data for correlation analysis
        corr_data = df_for_importance[numeric_features].copy()
        
        # Fill missing values with median
        corr_data = corr_data.fillna(corr_data.median())
        
        # Calculate correlation matrix
        corr = corr_data.corr()
        
        # Display the plot
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        display_corr_names = [re.sub(r'(_i|_r)$', '', f) for f in numeric_features]
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax2,
                    xticklabels=display_corr_names, yticklabels=display_corr_names)
        
        st.pyplot(fig2)
        
        # Save the plot to a bytes buffer for download
        buf2 = io.BytesIO()
        fig2.savefig(buf2, format='png', bbox_inches='tight')
        buf2.seek(0)
        
        # Add download button
        st.download_button(
            "Download Correlation Heatmap as PNG",
            data=buf2,
            file_name="correlation_heatmap.png",
            mime="image/png"
        )
        
        # Clean up
        plt.close(fig2)
        
        # Show data quality information
        with st.expander("Show data quality information"):
            st.write("Number of features in correlation analysis:", len(numeric_features))
            st.write("Features included:", list(numeric_features))
            missing_info = df_for_importance[numeric_features].isna().sum()
            st.write("Missing values per feature (before filling):")
            st.write(missing_info[missing_info > 0])
    else:
        st.warning("No numeric features available for correlation analysis.")
else:
    st.warning("Not enough data for correlation analysis.")

# --- Win Rate ---
st.subheader("Win Rate")
st.caption("Based on home matches only")
if len(df_team) > 0:
    win_rate = df_team['outcome_numeric'].mean()
    st.metric("Win Rate", f"{win_rate:.2%}")
else:
    st.warning("Not enough data to calculate win rate.")

# --- Prediction Section ---
st.subheader("Predict Match Outcome")
st.caption("""
Predictions are based on pre-match and in-game indicators only.
Select a match row to see the predicted probability of winning.
The model considers both absolute (_i) and relative (_r) metrics.
Feature names are displayed without their _i/_r suffixes for clarity.
""")
if len(df_processed) > 0:
    if 'outcome_numeric' not in df_processed.columns:
        df_processed['outcome_numeric'] = df_team['outcome_numeric'].values
    X = df_processed.drop(columns=['outcome_numeric'], errors='ignore')
    y = df_processed['outcome_numeric']
    if len(X) > 0 and len(y.unique()) > 1:
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        row_idx = st.number_input("Select match row for prediction (0 = first row)", min_value=0, max_value=len(X)-1, value=0)
        input_row = X.iloc[[row_idx]]
        pred_proba = model.predict_proba(input_row)[0][1]
        st.write(f"**Predicted probability of win for {selected_team} in selected match:** {pred_proba:.2%}")
    else:
        st.warning("Not enough data or only one class present for prediction.")
else:
    st.write("Not enough data for prediction.")