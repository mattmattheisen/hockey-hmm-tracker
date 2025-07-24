import streamlit as st
import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# ——— Page Config —————————————————————————————————————————————————————
st.set_page_config(page_title="Hockey HMM Tracker", layout="wide")

# ——— Sidebar ————————————————————————————————————————————————————————
st.sidebar.title("Hockey HMM Settings")
uploaded_file = st.sidebar.file_uploader("Upload game stats CSV", type=["csv"])
n_states = st.sidebar.slider("Number of Hidden States", 2, 5, 3)
if uploaded_file:
    st.sidebar.markdown("""
    **CSV must include these columns:**
    - GameDate (YYYY-MM-DD)
    - Opponent, Venue
    - GoalsFor, GoalsAgainst
    - ShotsFor, ShotsAgainst
    - PenaltyMinutes
    - FaceoffWinPct
    """)

# ——— Main App ————————————————————————————————————————————————————————
if not uploaded_file:
    st.write("👉 Upload a CSV file with game data to begin.")
    st.stop()

# 1) Load & preprocess
df = pd.read_csv(uploaded_file, parse_dates=['GameDate']).sort_values('GameDate')
features = ['GoalsFor','GoalsAgainst','ShotsFor','ShotsAgainst','PenaltyMinutes','FaceoffWinPct']
X = StandardScaler().fit_transform(df[features].values)

# 2) Fit HMM
model = hmm.GaussianHMM(n_components=n_states, covariance_type="full",
                        n_iter=100, random_state=42)
model.fit(X)
states = model.predict(X)
df['State'] = states

# 3) Order states by goal differential & assign labels
means = model.means_  # shape (n_states, len(features))
goal_diff = means[:, features.index('GoalsFor')] - means[:, features.index('GoalsAgainst')]
order = np.argsort(goal_diff)[::-1]  # best offense first

# Define your coach‑friendly names (ensure list ≥ max n_states)
friendly_names = ["Locked‑In","Improving","Fatigued","Demoralized","Overconfident"]
state_label_map = {old: friendly_names[i] for i, old in enumerate(order[:n_states])}
df['StateLabel'] = df['State'].map(lambda s: state_label_map[s])

# 4) Define descriptions and coaching tips
descriptions = {
    "Locked‑In":    "High offense/possession, low penalties. Maintain the plan.",
    "Improving":    "Uptrend in performance. Increase tactical intensity.",
    "Fatigued":     "Late‑game drop‑offs. Lighten practice load this week.",
    "Demoralized":  "High goals against & penalties. Reinforce fundamentals & morale.",
    "Overconfident":"Good results but sloppy. Reinforce discipline."
}
tips = {k: v for k, v in descriptions.items()}  # same for simplicity

# 5) Add coach notes
df['CoachNote'] = df['StateLabel'].map(lambda L: tips[L])

# ——— Legend & Summary —————————————————————————————————————————————————
st.subheader("🔑 State Legend & Coaching Tips")
for label in state_label_map.values():
    st.markdown(f"**{label}**: {descriptions[label]}")

# Trend summary
last_states = df['StateLabel'].tail(3).tolist()
trend = " → ".join(last_states)
st.markdown(f"**Current Trend (last 3 games):** {trend}")

# ——— Data Table with Color Coding —————————————————————————————————————
def color_map(label):
    palette = {
        "Locked‑In":    "background-color: #c8e6c9",
        "Improving":    "background-color: #bbdefb",
        "Fatigued":     "background-color: #ffe0b2",
        "Demoralized":  "background-color: #ffcdd2",
        "Overconfident":"background-color: #d1c4e9"
    }
    return [palette.get(label, "")]*len(df.columns)

st.subheader("Game Stats & Coach Notes")
styled = df[['GameDate','Opponent','Venue'] + features + ['StateLabel','CoachNote']] \
    .style.apply(lambda row: color_map(row['StateLabel']), axis=1)
st.write(styled, unsafe_allow_html=True)

# ——— Charts ————————————————————————————————————————————————————————
fig1 = px.line(df, x='GameDate', y='State', markers=True,
               title="Hidden State Over Time")
st.plotly_chart(fig1, use_container_width=True)

counts = df['StateLabel'].value_counts().reindex(friendly_names[:n_states], fill_value=0)
fig2 = px.bar(x=counts.index, y=counts.values,
              labels={'x':'Hidden State','y':'Game Count'},
              title="Games per Hidden State")
st.plotly_chart(fig2, use_container_width=True)

# ——— Download Multi‑Sheet Excel ————————————————————————————————————————
from io import BytesIO
buffer = BytesIO()
with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
    # Data sheet
    df[['GameDate','Opponent','Venue'] + features + ['StateLabel','CoachNote']] \
      .to_excel(writer, sheet_name="Data", index=False)
    # Legend sheet
    pd.DataFrame([
        {"State": label, "Description": descriptions[label], "Tip": tips[label]}
        for label in state_label_map.values()
    ]).to_excel(writer, sheet_name="Legend", index=False)
    # Summary sheet
    pd.DataFrame({
        "State": counts.index,
        "Count": counts.values
    }).to_excel(writer, sheet_name="Summary", index=False)
    writer.save()
buffer.seek(0)

st.download_button(
    "📥 Download Coach Report (Excel)",
    data=buffer,
    file_name="hockey_hmm_report.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
