import streamlit as st
import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import plotly.express as px

st.set_page_config(page_title="Hockey HMM Tracker", layout="wide")

# Sidebar controls
st.sidebar.title("Hockey HMM Settings")
uploaded_file = st.sidebar.file_uploader("Upload game stats CSV", type=["csv"])
n_states = st.sidebar.slider("Number of Hidden States", 2, 5, 3)
st.sidebar.markdown("""
**CSV columns required:**
- GameDate (YYYY-MM-DD)
- Opponent
- Venue
- GoalsFor, GoalsAgainst
- ShotsFor, ShotsAgainst
- PenaltyMinutes
- FaceoffWinPct
""")

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=['GameDate'])
    df = df.sort_values('GameDate')

    # Features for HMM
    features = ['GoalsFor', 'GoalsAgainst', 'ShotsFor',
                'ShotsAgainst', 'PenaltyMinutes', 'FaceoffWinPct']
    X = df[features].values

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit HMM
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=100,
        random_state=42
    )
    model.fit(X_scaled)

    # Decode and label
    states = model.predict(X_scaled)
    df['State'] = states
    df['StateLabel'] = df['State'].map(lambda s: f"State {s+1}")

    # Show table
    st.subheader("Game Stats & Inferred States")
    st.dataframe(df[['GameDate','Opponent','Venue'] + features + ['StateLabel']])

    # State timeline
    st.subheader("Hidden State Over Time")
    fig1 = px.line(df, x='GameDate', y='State', markers=True,
                   title="Hidden State by Game")
    st.plotly_chart(fig1, use_container_width=True)

    # State distribution
    st.subheader("State Distribution")
    counts = df['StateLabel'].value_counts().sort_index()
    fig2 = px.bar(x=counts.index, y=counts.values,
                  labels={'x':'Hidden State','y':'Games Count'},
                  title="Games per Hidden State")
    st.plotly_chart(fig2, use_container_width=True)

else:
    st.write("👉 Upload a CSV file with game data to begin.")

