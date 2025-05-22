# Rugby Union MCP Dashboard

This app provides interactive analysis and predictive modeling for rugby union match data. It is designed for sports practitioners and researchers, with no programming experience required.

## Features
- **Season and Team Selection:** Explore data by season and team.
- **SVD Projection:** Visualize league-wide match patterns.
- **Feature Importance:** See which metrics most influence match outcomes.
- **Correlation Heatmap:** Discover relationships between key metrics.
- **Win Rate:** View team win rates for home matches.
- **Predict Match Outcome:** Get win probability predictions for each match.
- **Download Data:** Download filtered team data as a CSV file.

## How to Use (No Programming Needed)
1. **Open the App:**
   - If shared as a web link (e.g., via Streamlit Cloud), just click the link and use your browser.
   - If running locally, see below.
2. **Select a Season and Team:** Use the dropdown menus.
3. **Explore the Visualizations:** Review the SVD plot, feature importance, heatmap, and win rate.
4. **Predict Outcomes:** Select a match row to see the predicted win probability.
5. **Download Data:** Click the download button to get a CSV of the selected team's matches.

## How to Deploy (for Admins)
### Option 1: Streamlit Community Cloud (Recommended)
1. Push this repository to GitHub.
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud) and sign in with GitHub.
3. Click "New app", select your repo and branch, and set the main file to `app.py`.
4. Click "Deploy". Share the resulting link with your team.

### Option 2: Run Locally
1. Install Python 3.8+ and pip.
2. Clone this repository and navigate to the folder.
3. (Optional) Create and activate a virtual environment.
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
5. Run the app:
   ```
   streamlit run app.py
   ```
6. Open the local URL in your browser (usually http://localhost:8501).

## Data
- The `data/` folder contains sample data for demonstration.
- For privacy, do not upload sensitive data to public repositories.

## Support
For help, contact the project maintainer or open an issue on GitHub.

## How to Update the App (for Contributors)

If you make changes to `app.py` (or any other file), follow these steps to push your updates to GitHub and trigger a redeploy on Streamlit Cloud:

1. Open your terminal and navigate to the project folder:
   ```sh
   cd /Users/rowanbrown/Documents/GitHub/mcp_nl_query
   ```
2. Check which files have changed:
   ```sh
   git status
   ```
3. Add your changes:
   ```sh
   git add .
   ```
4. Commit your changes (replace the message with a short description of what you changed, e.g. "Update app.py to fix SVD bug"):
   ```sh
   git commit -m "Update app.py to fix SVD bug"
   ```
5. Push to GitHub:
   ```sh
   git push
   ```

**Note:** After pushing, Streamlit Cloud will automatically redeploy the app with your latest changes.

