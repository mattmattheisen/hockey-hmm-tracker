# main.py

import streamlit as st
import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from io import BytesIO

# Page Config
st.set_page_config(page_title="Hockey HMM Tracker", layout="wide")

# Title & Introduction
st.title("🏒 High School Hockey HMM Tracker")
st.markdown("""
This tool uses a Hidden Markov Model to reveal your team's underlying performance states from game stats.

Below are the five states (numeric + name + definition):

| State # | Name           | Definition                                                  |
|:-------:|----------------|-------------------------------------------------------------|
| 1       | Locked‑In      | High offense & possession, low penalties.                   |
| 2       | Improving      | Upward trend in shots and faceoff wins.                     |
| 3       | Fatigued       | Late‑game drop‑offs, higher penalty minutes.                |
| 4       | Demoralized    | Poor results and discipline issues.                         |
| 5       | Overconfident  | Good scoreline but sloppy fundamentals.                     |
""")

# Sidebar
st.sidebar.header("Hockey HMM Settings")
uploaded_file = st.sidebar.file_uploader("Upload game stats CSV", type=["csv"])
n_states = st.sidebar.slider("Number of States", min_value=2, max_value=5, value=3)
if uploaded_file:
    st.sidebar.markdown("""
**CSV Requirements:**
- GameDate (YYYY-MM-DD)
- Opponent, Venue
- GoalsFor, GoalsAgainst
- ShotsFor, ShotsAgainst
- PenaltyMinutes
- FaceoffWinPct
""")

# Stop if No File
if not uploaded_file:
    st.write("👉 Upload your game‑by‑game CSV to begin.")
    st.stop()

# Data Load & Scaling
df = pd.read_csv(uploaded_file, parse_dates=['GameDate']).sort_values('GameDate')
features = ['GoalsFor','GoalsAgainst','ShotsFor','ShotsAgainst','PenaltyMinutes','FaceoffWinPct']
X = StandardScaler().fit_transform(df[features])

# HMM Fit & Predict
model = hmm.GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100, random_state=42)
model.fit(X)
df['StateNum'] = model.predict(X)

# Label Mapping
means = model.means_
gd = means[:, features.index('GoalsFor')] - means[:, features.index('GoalsAgainst')]
order = np.argsort(gd)[::-1]
names = ["Locked‑In","Improving","Fatigued","Demoralized","Overconfident"]
label_map = {old: names[i] for i, old in enumerate(order[:n_states])}
df['StateLabel'] = df['StateNum'].map(label_map)

# Coaching Notes
notes = {
    "Locked‑In":    "Stay the course—maintain systems and line rotations.",
    "Improving":    "Leverage momentum: add competitive drills in practice.",
    "Fatigued":     "Emphasize recovery: short shifts, light skill work.",
    "Demoralized":  "Rebuild confidence: puck‑handling and team bonding.",
    "Overconfident":"Reinforce fundamentals: focus on detail and discipline."
}
df['CoachNote'] = df['StateLabel'].map(notes)

# Display Table with State #
st.subheader("Game Results & Coach Notes")
emoji_map = {
    "Locked‑In":"🟢",
    "Improving":"🔵",
    "Fatigued":"🟠",
    "Demoralized":"🔴",
    "Overconfident":"🟣"
}
df['StateEmoji'] = df['StateLabel'].map(emoji_map)
cols = ['GameDate','Opponent','Venue'] + features + ['StateNum','StateLabel','StateEmoji','CoachNote']
st.dataframe(df[cols], height=500)

# State Distribution Chart
st.subheader("📊 Season State Distribution")
dist = df['StateLabel'].value_counts().reindex(names[:n_states], fill_value=0)
fig = px.bar(
    x=dist.index,
    y=dist.values,
    labels={'x':'State','y':'Games Played'},
    title="Games Spent in Each State"
)
st.plotly_chart(fig, use_container_width=True)

# Export Multi‑Sheet Excel
buffer = BytesIO()
with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
    # Data sheet
    df[cols].to_excel(writer, sheet_name="Data", index=False)
    # Legend sheet
    legend = pd.DataFrame([{
        "State #": i+1,
        "Name": names[i],
        "Definition": [
            "High offense & possession, low penalties.",
            "Upward trend in shots and faceoff wins.",
            "Late‑game drop‑offs, higher penalty minutes.",
            "Poor results and discipline issues.",
            "Good scoreline but sloppy fundamentals."
        ][i]
    } for i in range(n_states)])
    legend.to_excel(writer, sheet_name="Legend", index=False)
    # Summary sheet
    summary = pd.DataFrame({"State": dist.index, "Count": dist.values})
    summary.to_excel(writer, sheet_name="Summary", index=False)
buffer.seek(0)

st.download_button(
    "📥 Download Coach Report",
    data=buffer,
    file_name="hockey_hmm_coach_report.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
