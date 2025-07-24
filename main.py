import streamlit as st
import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from io import BytesIO

# ——— Page Config —————————————————————————————————————————————————————
st.set_page_config(page_title="Hockey HMM Tracker", layout="wide")

# ——— Title & Introduction —————————————————————————————————————————————————
st.title("🏒 High School Hockey HMM Tracker")
st.markdown("""
**What are Hidden States?**  
This app uses a Hidden Markov Model to infer your team’s underlying performance patterns from game stats.  
Each *hidden state* represents a distinct mode of play:
- **Locked‑In**: High offense & possession, low penalties.
- **Improving**: Upward trend in shots and faceoff wins.
- **Fatigued**: Late‑game drop‑offs, higher penalty minutes.
- **Demoralized**: Poor results and discipline issues.
- **Overconfident**: Good scoreline but sloppy fundamentals.

Use these insights to tailor practices and tactics each week.
""")

# ——— Sidebar ————————————————————————————————————————————————————————
st.sidebar.title("Hockey HMM Settings")
uploaded_file = st.sidebar.file_uploader("Upload game stats CSV", type=["csv"])
n_states = st.sidebar.slider("Number of Hidden States", 2, 5, 3)
if uploaded_file:
    st.sidebar.markdown("""
**CSV must include these columns:**
- GameDate (YYYY-MM-DD)
- Opponent
- Venue
- GoalsFor, GoalsAgainst
- ShotsFor, ShotsAgainst
- PenaltyMinutes
- FaceoffWinPct
""")

# ——— Stop if No File —————————————————————————————————————————————————————
if not uploaded_file:
    st.write("👉 Upload a CSV file with game data to begin.")
    st.stop()

# ——— Load & Preprocess ————————————————————————————————————————————————————
df = pd.read_csv(uploaded_file, parse_dates=['GameDate']).sort_values('GameDate')
features = ['GoalsFor','GoalsAgainst','ShotsFor','ShotsAgainst','PenaltyMinutes','FaceoffWinPct']
X = StandardScaler().fit_transform(df[features].values)

# ——— Fit HMM ——————————————————————————————————————————————————————————————
model = hmm.GaussianHMM(n_components=n_states,
                        covariance_type="full",
                        n_iter=100,
                        random_state=42)
model.fit(X)
df['State'] = model.predict(X)

# ——— Label States ————————————————————————————————————————————————————————
means = model.means_
idx_for = features.index('GoalsFor')
idx_against = features.index('GoalsAgainst')
goal_diff = means[:, idx_for] - means[:, idx_against]
order = np.argsort(goal_diff)[::-1]

friendly_names = ["Locked‑In","Improving","Fatigued","Demoralized","Overconfident"]
state_label_map = {old: friendly_names[i] for i, old in enumerate(order[:n_states])}
df['StateLabel'] = df['State'].map(lambda s: state_label_map[s])

# ——— Descriptions & Tips ——————————————————————————————————————————————————
descriptions = {
    "Locked‑In":     "High offense/possession, low penalties. Maintain the plan.",
    "Improving":     "Upward trend in performance. Increase tactical intensity.",
    "Fatigued":      "Late‑game drop‑offs. Lighten practice load this week.",
    "Demoralized":   "High goals against & penalties. Reinforce fundamentals & morale.",
    "Overconfident": "Good scoreline but sloppy fundamentals. Reinforce discipline."
}
df['CoachNote'] = df['StateLabel'].map(descriptions)

# ——— Legend & Trend ——————————————————————————————————————————————————————
st.subheader("🔑 State Legend & Coaching Tips")
for label in state_label_map.values():
    st.markdown(f"**{label}**: {descriptions[label]}")

last_states = df['StateLabel'].tail(3).tolist()
trend = " → ".join(last_states)
st.markdown(f"**Current Trend (last 3 games):** {trend}")

# ——— Hidden State Over Time Chart ————————————————————————————————————————
st.subheader("📈 Hidden State Over Time")
fig1 = px.line(df, x='GameDate', y='State', markers=True,
               title="Hidden State by Game (y-axis = numeric state)")
st.plotly_chart(fig1, use_container_width=True)

# ——— Chart Legend ——————————————————————————————————————————————————————
st.subheader("📋 Hidden State Chart Legend")
for old in sorted(state_label_map):
    st.markdown(f"- **State {old}**: {state_label_map[old]}")

# ——— Emoji‑Coded Table ————————————————————————————————————————————————————
emojis = {
    "Locked‑In":     "🟢",
    "Improving":     "🔵",
    "Fatigued":      "🟠",
    "Demoralized":   "🔴",
    "Overconfident": "🟣"
}
df['StateEmoji'] = df['StateLabel'].map(emojis)

st.subheader("Game Stats & Coach Notes")
display_cols = ['GameDate','Opponent','Venue'] + features + ['StateEmoji','StateLabel','CoachNote']
st.dataframe(df[display_cols])

# ——— State Distribution Chart —————————————————————————————————————————————
st.subheader("📊 State Distribution")
counts = df['StateLabel'].value_counts().reindex(friendly_names[:n_states], fill_value=0)
fig2 = px.bar(x=counts.index, y=counts.values,
              labels={'x':'Hidden State','y':'Game Count'},
              title="Games per Hidden State")
st.plotly_chart(fig2, use_container_width=True)

# ——— Download Multi‑Sheet Excel ——————————————————————————————————————————
buffer = BytesIO()
with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
    df[display_cols].to_excel(writer, sheet_name="Data", index=False)
    pd.DataFrame([
        {"State": label, "Description": descriptions[label]}
        for label in state_label_map.values()
    ]).to_excel(writer, sheet_name="Legend", index=False)
    pd.DataFrame({"State": counts.index, "Count": counts.values}) \
      .to_excel(writer, sheet_name="Summary", index=False)
buffer.seek(0)

st.download_button(
    "📥 Download Coach Report (Excel)",
    data=buffer,
    file_name="hockey_hmm_report.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
