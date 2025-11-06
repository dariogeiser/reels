import pandas as pd
import streamlit as st
import plotly.express as px
from redcap import Project

# -------------------------------------------------------
# Streamlit Config
# -------------------------------------------------------
st.set_page_config(page_title="📊 Reels Study Dashboard", layout="wide")

# -------------------------------------------------------
# Constants
# -------------------------------------------------------
API_URL = "https://redcap.zhaw.ch/api/"
API_TOKEN = "1138E5B5387C496FDE8315A37C29FDB4"

FORMS = [
    "consent_and_mri_screening_complete",
    "mri_safety_screening_short_form_complete",
    "brain_fog_mood_complete",
    "demographics_complete",
    "daily_dairy_complete",
    "imaging_intake_complete",
]

FIELDS = [
    "record_id",
    "gender", "age_calc", "dose_group", "reel_baseline_duration",
    "reel_minutes", "sleep_min", "fatigue_today",
    "mood_level", "cognitive_score1"
] + FORMS


# -------------------------------------------------------
# REDCap Fetch
# -------------------------------------------------------
@st.cache_data(ttl=600)
def get_data(api_url, api_token, fields):
    try:
        proj = Project(api_url, api_token)
        data = proj.export_records(
            fields=fields,
            events=None,  # include all events
            raw_or_label="label",
            export_data_access_groups=True,
            format_type="json",
        )
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"API error: {e}")
        return pd.DataFrame()

# -------------------------------------------------------
# Data cleaning helpers
# -------------------------------------------------------
def clean_reels_data(df):
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()

    # Normalize dose group (text or numeric)
    df["dose_group"] = (
        df["dose_group"].astype(str).str.strip()
        .replace({
            "0": "0 min/day",
            "60": "60 min/day",
            "180": "180 min/day",
            ">180": ">180 min/day",
            "nan": None,
        })
    )
    # Ensure only the 4 correct labels remain
    df.loc[
        ~df["dose_group"].isin(["0 min/day", "60 min/day", "180 min/day", ">180 min/day"]),
        "dose_group",
    ] = None

    print(df["dose_group"] )

    df["gender"] = (
        df["gender"]
        .astype(str)
        .str.strip()
        .replace({
            "-": "prefer not to say",
            "": "prefer not to say",
            "nan": "prefer not to say",
            "None": "prefer not to say"
        })
    )

    # Convert numeric columns safely
    for c in ["age_calc", "sleep_min", "fatigue_today", "mood_level", "cognitive_score1", "reel_minutes"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Compute progress
    for f in FORMS:
        df[f] = pd.to_numeric(df[f].replace("-", None), errors="coerce")
    df["forms_total"] = len(FORMS)
    df["forms_done"] = df[FORMS].apply(lambda s: (s == 2).sum(), axis=1)
    df["progress_pct"] = (100 * df["forms_done"] / df["forms_total"]).round(0)

    return df

# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("REDCap API URL", API_URL)
    api_token = st.text_input("API Token", API_TOKEN, type="password")
    min_progress = st.slider("Minimum Progress (%)", 0, 100, 0)
    refresh = st.button("Reload Data")

# -------------------------------------------------------
# Load
# -------------------------------------------------------
df = get_data(api_url, api_token, FIELDS)
if df.empty:
    st.stop()

df = clean_reels_data(df)
df = df[df["progress_pct"] >= min_progress]
if df.empty:
    st.warning("No records match current filters.")
    st.stop()

# -------------------------------------------------------
# Separate events
# -------------------------------------------------------
df_t0 = df[df["redcap_event_name"].str.contains("T0", case=False, na=False)]

df_t1 = df[df["redcap_event_name"].str.contains("T1", case=False, na=False)]
df_days = df[df["redcap_event_name"].str.contains("Day", case=False, na=False)]

# -------------------------------------------------------
# Tabs
# -------------------------------------------------------
tabs = st.tabs([
    "Enrollment",
    "Demographics (T0)",
    "Daily Diary (Days 1–7)",
    "Cognition & Mood (T0 vs T1)",
    "Progress",
    "Aggregated Metrics",
    "Participants",
])

# -------------------------------------------------------
# Enrollment
# -------------------------------------------------------
with tabs[0]:
    st.subheader("Participant Enrollment and Group Assignment")
    total = df["record_id"].nunique()
    c1, c2 = st.columns(2)
    c1.metric("Total Participants", total)
    c2.metric("Groups Represented", df["dose_group"].nunique())

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.pie(df_t0, names="dose_group", title="Dose Group Distribution"), use_container_width=True)
    with c2:
        st.plotly_chart(px.histogram(df_t0, x="gender", color="dose_group", barmode="group",
                                     title="Gender by Dose Group"), use_container_width=True)

# -------------------------------------------------------
# Demographics
# -------------------------------------------------------
with tabs[1]:
    st.subheader("Demographics Overview (Baseline T0)")
    if not df_t0.empty:
        avg_age = df_t0["age_calc"].mean()
        c1, c2 = st.columns(2)
        c1.metric("Average Age", f"{avg_age:.1f}")
        c2.metric("Gender Categories", f"{df_t0['gender'].nunique()}")

        st.plotly_chart(px.histogram(df_t0, x="age_calc", nbins=10, title="Age Distribution"), use_container_width=True)
        st.plotly_chart(px.pie(df_t0, names="gender", title="Gender Composition"), use_container_width=True)
    else:
        st.info("No baseline (T0) data found.")

# -------------------------------------------------------
# Daily Diary
# -------------------------------------------------------
with tabs[2]:
    st.subheader("Daily Diary Metrics (Days 1–7)")
    if not df_days.empty:
        c1, c2 = st.columns(2)
        c1.metric("Avg Sleep (min)", f"{df_days['sleep_min'].mean():.1f}")
        c2.metric("Avg Fatigue", f"{df_days['fatigue_today'].mean():.1f}")

        st.plotly_chart(px.box(df_days, x="dose_group", y="sleep_min", points="all",
                               title="Sleep Duration by Group"), use_container_width=True)
        st.plotly_chart(px.box(df_days, x="dose_group", y="fatigue_today", points="all",
                               title="Fatigue by Group"), use_container_width=True)
    else:
        st.info("No daily diary data available.")

# -------------------------------------------------------
# Cognition & Mood
# -------------------------------------------------------
with tabs[3]:
    st.subheader("Cognition & Mood Comparison (T0 vs T1)")
    merged = pd.merge(
        df_t0[["record_id", "dose_group", "cognitive_score1", "mood_level"]],
        df_t1[["record_id", "cognitive_score1", "mood_level"]],
        on="record_id",
        suffixes=("_t0", "_t1")
    )
    if not merged.empty:
        merged["Δ Cognitive"] = merged["cognitive_score1_t1"] - merged["cognitive_score1_t0"]
        merged["Δ Mood"] = merged["mood_level_t1"] - merged["mood_level_t0"]

        st.plotly_chart(px.box(merged, x="dose_group", y="Δ Cognitive", points="all",
                               title="Change in Cognitive Score (T1 − T0)"), use_container_width=True)
        st.plotly_chart(px.box(merged, x="dose_group", y="Δ Mood", points="all",
                               title="Change in Mood (T1 − T0)"), use_container_width=True)
    else:
        st.info("No T0/T1 comparison data available yet.")

# -------------------------------------------------------
# Progress
# -------------------------------------------------------
with tabs[4]:
    st.subheader("Form Completion Progress (All Events)")
    fig = px.bar(df.sort_values("progress_pct"), x="progress_pct", y="record_id",
                 color="dose_group", orientation="h",
                 text="progress_pct", title="Completion by Participant")
    fig.update_layout(xaxis_title="Progress (%)", yaxis_title="Record ID")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------
# Aggregated
# -------------------------------------------------------
with tabs[5]:
    st.subheader("Aggregated Study Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean Cognition", f"{df['cognitive_score1'].mean():.1f}")
    c2.metric("Mean Mood", f"{df['mood_level'].mean():.1f}")
    c3.metric("Mean Sleep", f"{df['sleep_min'].mean():.1f}")
    c4.metric("Mean Fatigue", f"{df['fatigue_today'].mean():.1f}")

# -------------------------------------------------------
# Participants
# -------------------------------------------------------
with tabs[6]:
    st.subheader("Participant Overview Table")
    cols = ["record_id", "redcap_event_name", "dose_group", "age_calc", "progress_pct"] + FORMS
    st.dataframe(df[cols].sort_values(["record_id", "redcap_event_name"]), use_container_width=True)
