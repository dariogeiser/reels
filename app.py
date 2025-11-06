import pandas as pd
import streamlit as st
import plotly.express as px
from redcap import Project

# -----------------------
# Streamlit configuration
# -----------------------
st.set_page_config(page_title="📊 Reels Study Dashboard", layout="wide")

# -----------------------
# Constants
# -----------------------
API_URL = "https://redcap.zhaw.ch/api/"
API_TOKEN = "1138E5B5387C496FDE8315A37C29FDB4"

COMPLETE_FIELDS = [
    "consent_and_mri_screening_complete",
    "mri_safety_screening_short_form_complete",
    "brain_fog_mood_complete",
    "demographics_complete",
    "daily_dairy_complete",
    "imaging_intake_complete",
]

BASE_FIELDS = [
    "record_id", "dose_group", "gender", "age_calc",
    "reel_baseline_duration", "reel_minutes", "sleep_min",
    "fatigue_today", "mood_level", "cognitive_score1"
] + COMPLETE_FIELDS

# -----------------------
# REDCap data fetch (no event filter)
# -----------------------
@st.cache_data(ttl=600, show_spinner=False)
def get_redcap(api_url, api_token, fields):
    """Fetch all records (all events) from REDCap"""
    try:
        proj = Project(api_url, api_token)
        df = pd.DataFrame(
            proj.export_records(
                fields=fields,
                raw_or_label="label",
                export_data_access_groups=True,
                format_type="json"
            )
        )
        if df.empty:
            st.warning("No data found in REDCap project.")

        return df
    except Exception as e:
        st.error(f"API error: {e}")
        return pd.DataFrame()

# -----------------------
# Utility functions
# -----------------------
def compute_progress(df):
    """Calculate form completion progress per participant"""
    df = df.copy()
    for f in COMPLETE_FIELDS:
        df[f] = pd.to_numeric(df[f], errors="coerce")
    df["forms_total"] = len(COMPLETE_FIELDS)
    df["forms_done"] = df[COMPLETE_FIELDS].apply(lambda s: (s == 2).sum(), axis=1)
    df["progress_pct"] = (100 * df["forms_done"] / df["forms_total"]).round(0)
    return df

# -----------------------
# Sidebar
# -----------------------
with st.sidebar:
    st.header("🔗 REDCap Connection")
    api_url = st.text_input("API URL", API_URL)
    api_token = st.text_input("API Token", API_TOKEN, type="password")
    min_progress = st.slider("Min Progress (%)", 0, 100, 0)
    refresh = st.button("🔄 Refresh Data")

# -----------------------
# Data loading
# -----------------------
df = get_redcap(api_url, api_token, BASE_FIELDS)
if df.empty:
    st.stop()

# Normalize column names
df.columns = [c.strip().lower() for c in df.columns]

# Detect and rename record_id dynamically
rid_col = next((c for c in df.columns if "record" in c), None)
if rid_col:
    df.rename(columns={rid_col: "record_id"}, inplace=True)
else:
    st.error("record_id column not found")
    st.write(df.head())
    st.stop()

# Compute progress
df = compute_progress(df)

# Normalize dose_group
if "dose_group" in df.columns:
    df["dose_group_num"] = df["dose_group"].astype(str).str.extract(r"(\d+)").astype(float)
    df["dose_group"] = df["dose_group_num"].fillna(pd.to_numeric(df["dose_group"], errors="coerce"))
    df["dose_group"] = df.groupby("record_id")["dose_group"].ffill().bfill()
    df["dose_group"] = pd.to_numeric(df["dose_group"], errors="coerce")

# Filter by progress only
df = df[df["progress_pct"].fillna(0) >= min_progress]

if len(df) == 0:
    st.warning("No participants match the current filters.")
    st.stop()

# -----------------------
# Dashboard Tabs
# -----------------------
tabs = st.tabs([
    "1️⃣ Enrollment & Group Assignment",
    "2️⃣ Demographics Overview",
    "3️⃣ Baseline Media Use",
    "4️⃣ Daily Diary Metrics",
    "5️⃣ Cognitive & Mood Outcomes",
    "6️⃣ Study Progress Summary",
    "7️⃣ Aggregated Insights",
    "8️⃣ Completion Summary"
])

# --- 1. Enrollment ---
with tabs[0]:
    st.header("Participant Enrollment & Group Assignment")
    c1, c2 = st.columns(2)
    c1.metric("Participants", df["record_id"].nunique())
    c2.metric("Groups Represented", df["dose_group"].nunique())

    if "dose_group" in df.columns:
        fig1 = px.pie(df, names="dose_group", title="Dose Group Distribution",
                      color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig1, use_container_width=True)

    if "gender" in df.columns:
        fig2 = px.histogram(df, x="gender", color="dose_group", barmode="group",
                            title="Gender by Dose Group")
        st.plotly_chart(fig2, use_container_width=True)

# --- 2. Demographics ---
with tabs[1]:
    st.header("Demographics Overview")
    st.metric("Average Age", f"{pd.to_numeric(df['age_calc'], errors='coerce').mean():.1f}")
    fig3 = px.histogram(df, x="age_calc", nbins=15, title="Age Distribution",
                        color_discrete_sequence=["#3a7ca5"])
    st.plotly_chart(fig3, use_container_width=True)

    fig4 = px.pie(df, names="gender", title="Gender Composition",
                  color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig4, use_container_width=True)

# --- 3. Baseline Media Use ---
with tabs[2]:
    st.header("Baseline Media Use")
    if "reel_baseline_duration" in df.columns:
        fig5 = px.pie(df, names="reel_baseline_duration",
                      title="Daily Instagram Reel Use Before Study")
        st.plotly_chart(fig5, use_container_width=True)

# --- 4. Daily Diary ---
with tabs[3]:
    st.header("Daily Diary Metrics")
    st.metric("Avg Sleep (min)", f"{pd.to_numeric(df['sleep_min'], errors='coerce').mean():.1f}")
    st.metric("Avg Fatigue", f"{pd.to_numeric(df['fatigue_today'], errors='coerce').mean():.1f}")

    if "sleep_min" in df.columns:
        fig6 = px.box(df, x="dose_group", y="sleep_min", points="all",
                      title="Sleep Duration by Group")
        st.plotly_chart(fig6, use_container_width=True)

    if "fatigue_today" in df.columns:
        fig7 = px.box(df, x="dose_group", y="fatigue_today", points="all",
                      title="Fatigue by Group")
        st.plotly_chart(fig7, use_container_width=True)

# --- 5. Cognitive & Mood ---
with tabs[4]:
    st.header("Cognitive and Mood Outcomes")
    if "cognitive_score1" in df.columns:
        fig8 = px.box(df, x="dose_group", y="cognitive_score1", points="all",
                      title="Cognitive Scores by Group")
        st.plotly_chart(fig8, use_container_width=True)

    if "mood_level" in df.columns:
        fig9 = px.box(df, x="dose_group", y="mood_level", points="all",
                      title="Mood Levels by Group")
        st.plotly_chart(fig9, use_container_width=True)

# --- 6. Study Progress ---
with tabs[5]:
    st.header("Study Progress Summary")
    fig10 = px.bar(df.sort_values("progress_pct"), x="progress_pct", y="record_id",
                   color="dose_group", orientation="h", text="progress_pct",
                   title="Per-Participant Progress")
    fig10.update_layout(xaxis_title="Progress (%)", yaxis_title="Record ID", height=500)
    st.plotly_chart(fig10, use_container_width=True)

# --- 7. Aggregated Insights ---
with tabs[6]:
    st.header("Aggregated Insights")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Participants", df["record_id"].nunique())
    c2.metric("Mean Age", f"{pd.to_numeric(df['age_calc'], errors='coerce').mean():.1f}")

    c3.metric("Mean Sleep", f"{pd.to_numeric(df['sleep_min'], errors='coerce').mean():.1f}")
    c4.metric("Mean Fatigue", f"{pd.to_numeric(df['fatigue_today'], errors='coerce').mean():.1f}")
    c5.metric("Mean Mood", f"{pd.to_numeric(df['mood_level'], errors='coerce').mean():.1f}")

# --- 8. Completion Summary ---
# --- 9. Individual Progress ---
with tabs[7]:
    st.header("Current Progress per Participant")

    # Aggregiere, falls mehrere Events pro Teilnehmer existieren
    progress_df = (
        df.groupby("record_id", as_index=False)["progress_pct"]
        .mean()
        .sort_values("progress_pct", ascending=False)
    )

    # Balkendiagramm mit Plotly
    fig = px.bar(
        progress_df,
        x="progress_pct",
        y="record_id",
        orientation="h",
        text="progress_pct",
        color="progress_pct",
        color_continuous_scale="Blues",
        title="Participant Progress (%)"
    )
    fig.update_layout(
        xaxis_title="Progress (%)",
        yaxis_title="Participant ID",
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)

    # Optional: Tabelle
    st.dataframe(progress_df.style.format({"progress_pct": "{:.0f}%"}), use_container_width=True)

