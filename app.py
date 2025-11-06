import pandas as pd
import streamlit as st
import plotly.express as px
from redcap import Project

# -------------------------------------------------------
# Streamlit base configuration
# -------------------------------------------------------
st.set_page_config(
    page_title="Reels Study Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------
# Global Style Sheet
# -------------------------------------------------------
st.markdown("""
<style>
/* Base layout */
.main {
    background-color: #f5f6f8;
    padding: 1.5rem 2rem;
    font-family: 'Inter', sans-serif;
}

/* Typography */
h1, h2, h3 {
    color: #1a1a1a;
    font-weight: 600;
}
h2 { margin-top: 1rem; }
p, label, .stMarkdown {
    color: #333333;
}

/* Metric cards */
[data-testid="stMetricValue"] {
    color: #0a0a0a;
    font-weight: 600;
}
div[data-testid="stMetric"] {
    background-color: #ffffff;
    border: 1px solid #e5e5e5;
    border-radius: 10px;
    padding: 10px 14px;
    margin: 6px 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #f0f1f3;
}
section[data-testid="stSidebar"] .stMarkdown {
    color: #222 !important;
}

/* Tabs */
div[data-baseweb="tab-list"] {
    background-color: #ffffff;
    border-bottom: 1px solid #ddd;
}
button[data-baseweb="tab"] {
    font-weight: 500 !important;
    color: #444 !important;
    background-color: transparent !important;
}
button[data-baseweb="tab"]:hover {
    color: #0a84ff !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    border-bottom: 3px solid #0a84ff !important;
    color: #0a84ff !important;
}

/* Charts */
.js-plotly-plot {
    background: white !important;
    border-radius: 10px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    padding: 5px;
}

/* Tables */
.dataframe {
    background: #fff;
    border: 1px solid #ddd;
    border-radius: 6px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# Constants
# -------------------------------------------------------
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

# -------------------------------------------------------
# REDCap Data Fetch
# -------------------------------------------------------
@st.cache_data(ttl=600)
def get_redcap(api_url, api_token, fields):
    try:
        proj = Project(api_url, api_token)
        data = proj.export_records(
            fields=fields,
            raw_or_label="label",
            export_data_access_groups=True,
            format_type="json"
        )
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"API error: {e}")
        return pd.DataFrame()

# -------------------------------------------------------
# Helper
# -------------------------------------------------------
def compute_progress(df):
    df = df.copy()
    for f in COMPLETE_FIELDS:
        df[f] = pd.to_numeric(df[f], errors="coerce")
    df["forms_total"] = len(COMPLETE_FIELDS)
    df["forms_done"] = df[COMPLETE_FIELDS].apply(lambda s: (s == 2).sum(), axis=1)
    df["progress_pct"] = (100 * df["forms_done"] / df["forms_total"]).round(0)
    return df

# -------------------------------------------------------
# Sidebar Controls
# -------------------------------------------------------
with st.sidebar:
    st.title("Dashboard Settings")
    api_url = st.text_input("REDCap API URL", API_URL)
    api_token = st.text_input("REDCap API Token", API_TOKEN, type="password")
    st.markdown("---")
    min_progress = st.slider("Minimum Progress (%)", 0, 100, 0)
    refresh = st.button("Reload Data")

# -------------------------------------------------------
# Data Load
# -------------------------------------------------------
df = get_redcap(api_url, api_token, BASE_FIELDS)
if df.empty:
    st.stop()

df.columns = [c.strip().lower() for c in df.columns]
rid_col = next((c for c in df.columns if "record" in c), None)
if not rid_col:
    st.error("record_id not found")
    st.stop()
df.rename(columns={rid_col: "record_id"}, inplace=True)

df = compute_progress(df)
df["dose_group"] = pd.to_numeric(df["dose_group"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
df = df[df["progress_pct"].fillna(0) >= min_progress]
if df.empty:
    st.warning("No participants match filters.")
    st.stop()

# -------------------------------------------------------
# Tabs
# -------------------------------------------------------
tabs = st.tabs([
    "Enrollment",
    "Demographics",
    "Media Use",
    "Daily Diary",
    "Cognition & Mood",
    "Progress Overview",
    "Aggregated Insights",
    "Participant Detail"
])

# -------------------------------------------------------
# Tab 1: Enrollment
# -------------------------------------------------------
with tabs[0]:
    st.subheader("Participant Enrollment and Group Assignment")
    c1, c2 = st.columns(2)
    c1.metric("Total Participants", df["record_id"].nunique())
    c2.metric("Groups Represented", df["dose_group"].nunique())

    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.pie(df, names="dose_group", title="Dose Group Distribution",
                      color_discrete_sequence=px.colors.qualitative.Safe)
        fig1.update_layout(showlegend=True, height=400)
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = px.histogram(df, x="gender", color="dose_group",
                            barmode="group", title="Gender by Group")
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)

# -------------------------------------------------------
# Tab 2: Demographics
# -------------------------------------------------------
with tabs[1]:
    st.subheader("Demographic Overview")
    avg_age = pd.to_numeric(df["age_calc"], errors="coerce").mean()
    c1, c2 = st.columns(2)
    c1.metric("Average Age", f"{avg_age:.1f}")
    c2.metric("Gender Balance", f"{df['gender'].nunique()} categories")

    col1, col2 = st.columns(2)
    with col1:
        fig3 = px.histogram(df, x="age_calc", nbins=15,
                            title="Age Distribution", color_discrete_sequence=["#4b8bbe"])
        st.plotly_chart(fig3, use_container_width=True)
    with col2:
        fig4 = px.pie(df, names="gender", title="Gender Composition",
                      color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig4, use_container_width=True)

# -------------------------------------------------------
# Tab 3: Media Use
# -------------------------------------------------------
with tabs[2]:
    st.subheader("Baseline Media Use")
    if "reel_baseline_duration" in df.columns:
        fig5 = px.pie(df, names="reel_baseline_duration",
                      title="Daily Instagram Reel Use Before Study",
                      color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig5, use_container_width=True)

# -------------------------------------------------------
# Tab 4: Daily Diary
# -------------------------------------------------------
with tabs[3]:
    st.subheader("Daily Diary Metrics")
    c1, c2 = st.columns(2)
    c1.metric("Mean Sleep (min)", f"{pd.to_numeric(df['sleep_min'], errors='coerce').mean():.1f}")
    c2.metric("Mean Fatigue", f"{pd.to_numeric(df['fatigue_today'], errors='coerce').mean():.1f}")

    col1, col2 = st.columns(2)
    with col1:
        fig6 = px.box(df, x="dose_group", y="sleep_min", points="all",
                      title="Sleep Duration by Group")
        st.plotly_chart(fig6, use_container_width=True)
    with col2:
        fig7 = px.box(df, x="dose_group", y="fatigue_today", points="all",
                      title="Fatigue by Group")
        st.plotly_chart(fig7, use_container_width=True)

# -------------------------------------------------------
# Tab 5: Cognition & Mood
# -------------------------------------------------------
with tabs[4]:
    st.subheader("Cognitive and Mood Outcomes")
    col1, col2 = st.columns(2)
    with col1:
        if "cognitive_score1" in df.columns:
            fig8 = px.box(df, x="dose_group", y="cognitive_score1", points="all",
                          title="Cognitive Scores by Group")
            st.plotly_chart(fig8, use_container_width=True)
    with col2:
        if "mood_level" in df.columns:
            fig9 = px.box(df, x="dose_group", y="mood_level", points="all",
                          title="Mood Levels by Group")
            st.plotly_chart(fig9, use_container_width=True)

# -------------------------------------------------------
# Tab 6: Progress
# -------------------------------------------------------
with tabs[5]:
    st.subheader("Study Progress Summary")
    fig10 = px.bar(df.sort_values("progress_pct"),
                   x="progress_pct", y="record_id",
                   color="dose_group", orientation="h",
                   text="progress_pct", title="Per-Participant Progress",
                   color_continuous_scale="Blues")
    fig10.update_layout(xaxis_title="Progress (%)", yaxis_title="Participant ID", height=500)
    st.plotly_chart(fig10, use_container_width=True)

# -------------------------------------------------------
# Tab 7: Aggregated Insights
# -------------------------------------------------------
with tabs[6]:
    st.subheader("Aggregated Summary Metrics")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Participants", df["record_id"].nunique())
    c2.metric("Mean Age", f"{pd.to_numeric(df['age_calc'], errors='coerce').mean():.1f}")
    c3.metric("Mean Sleep", f"{pd.to_numeric(df['sleep_min'], errors='coerce').mean():.1f}")
    c4.metric("Mean Fatigue", f"{pd.to_numeric(df['fatigue_today'], errors='coerce').mean():.1f}")
    c5.metric("Mean Mood", f"{pd.to_numeric(df['mood_level'], errors='coerce').mean():.1f}")

# -------------------------------------------------------
# Tab 8: Participant Detail
# -------------------------------------------------------
with tabs[7]:
    st.subheader("Individual Participant Progress")
    progress_df = (
        df.groupby("record_id", as_index=False)["progress_pct"]
        .mean()
        .sort_values("progress_pct", ascending=False)
    )
    fig = px.bar(progress_df, x="progress_pct", y="record_id",
                 orientation="h", text="progress_pct",
                 color="progress_pct", color_continuous_scale="Blues",
                 title="Progress by Participant")
    fig.update_layout(xaxis_title="Progress (%)", yaxis_title="Participant ID", height=600)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(progress_df.style.format({"progress_pct": "{:.0f}%"}), use_container_width=True)
