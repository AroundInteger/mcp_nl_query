import pandas as pd
import numpy as np

# Load your data
input_path = 'data/raw/21_22.csv'
output_path = 'data/raw/21_22_anonymized.csv'

df = pd.read_csv(input_path)

# 1. Replace team names with codes
unique_teams = df['team'].unique()
team_map = {name: f'T{idx+1}' for idx, name in enumerate(unique_teams)}
df['team'] = df['team'].map(team_map)

# If you have an 'opponent' or similar column, anonymize it too
if 'opponent' in df.columns:
    unique_opponents = df['opponent'].unique()
    opp_map = {name: f'T{idx+1}' for idx, name in enumerate(unique_opponents)}
    df['opponent'] = df['opponent'].map(opp_map)

# 2. Randomize match order
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 3. (Optional) Remove or randomize matchid
if 'matchid' in df.columns:
    df['matchid'] = np.arange(1, len(df)+1)

# Save anonymized data
df.to_csv(output_path, index=False)
print(f'Anonymized file saved as {output_path}') 