import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
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
    "record_id", "redcap_event_name",
    "gender", "age_calc", "dose_group", "reel_baseline_duration",
    "reel_minutes", "sleep_min", "fatigue_today",
    "mood_level", "cognitive_score1"
] + FORMS

# -------------------------------------------------------
# Fetch Data
# -------------------------------------------------------
@st.cache_data(ttl=600)
def fetch_data(url, token, fields):
    try:
        proj = Project(url, token)
        data = proj.export_records(
            fields=[f for f in fields if f != "redcap_event_name"],
            raw_or_label="label",
            export_data_access_groups=True,
            format_type="json",
        )

        df = pd.DataFrame(data)

        # Ensure _complete fields exist even if REDCap doesn't return them automatically
        for f in FORMS:
            if f not in df.columns:
                df[f] = None

        return df
    except Exception as e:
        st.error(f"REDCap API error: {e}")
        return pd.DataFrame()


# -------------------------------------------------------
# Clean + Enrich Data
# -------------------------------------------------------
def clean_data(df):
    if df.empty:
        return df
    df.columns = df.columns.str.lower().str.strip()

    # Gender normalization
    df["gender"] = (
        df["gender"].astype(str).str.strip().replace(
            {"-": "prefer not to say", "": "prefer not to say",
             "nan": "prefer not to say", "None": "prefer not to say"}
        )
    )

    # Convert numeric fields
    num_cols = ["age_calc", "sleep_min", "fatigue_today", "mood_level",
                "cognitive_score1", "reel_minutes"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Dose group normalization
    df["dose_group"] = (
        df["dose_group"].astype(str).str.strip().replace({
            "0": "0 min/day",
            "60": "60 min/day",
            "180": "180 min/day",
            ">180": ">180 min/day",
            "nan": "Not completed", "": "Not completed", "-": "Not completed"
        })
    )

    # Progress calculation
    for f in FORMS:
        df[f] = pd.to_numeric(df[f], errors="coerce").fillna(0)

    df["forms_total"] = len(FORMS)
    df["forms_done"] = df[FORMS].apply(lambda s: (s == 2).sum(), axis=1)
    df["progress_pct"] = (100 * df["forms_done"] / df["forms_total"]).round(1)

    return df

# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("REDCap API URL", API_URL)
    refresh = st.button("Reload Data")
# -------------------------------------------------------
# Load Data
# -------------------------------------------------------
df = fetch_data(api_url, API_TOKEN, FIELDS)
if df.empty:
    st.stop()

df = clean_data(df)

# Event subsets
df_t0 = df[df["redcap_event_name"].str.contains("T0", case=False, na=False)]
df_t1 = df[df["redcap_event_name"].str.contains("T1", case=False, na=False)]
df_days = df[df["redcap_event_name"].str.contains("Day", case=False, na=False)]

# -------------------------------------------------------
# Tabs
# -------------------------------------------------------
tabs = st.tabs([
    "Enrollment & Groups",
    "Demographics",
    "Daily Diary",
    "Cognition & Mood",
    "Progress Matrix",
    "Aggregated Metrics",
])

# -------------------------------------------------------
# Tab 1 – Enrollment
# -------------------------------------------------------
with tabs[0]:
    st.subheader("Enrollment & Dose Group Overview")

    total = df["record_id"].nunique()
    c1, c2 = st.columns(2)
    c1.metric("Total Participants", total)
    c2.metric("Dose Groups", df_t0["dose_group"].nunique())

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            px.pie(df_t0, names="dose_group", title="Dose Group Distribution"),
            use_container_width=True
        )
    with c2:
        st.plotly_chart(
            px.histogram(df_t0, x="gender", color="dose_group", barmode="group",
                         title="Gender by Dose Group"),
            use_container_width=True
        )

# -------------------------------------------------------
# Tab 2 – Demographics
# -------------------------------------------------------
with tabs[1]:
    st.subheader("Demographics Overview (T0)")
    if not df_t0.empty:
        avg_age = df_t0["age_calc"].mean(skipna=True)
        c1, c2 = st.columns(2)
        c1.metric("Average Age", f"{avg_age:.1f}" if pd.notna(avg_age) else "N/A")
        c2.metric("Gender Categories", f"{df_t0['gender'].nunique()}")

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.histogram(df_t0, x="age_calc", nbins=10,
                                         title="Age Distribution"), use_container_width=True)
        with col2:
            st.plotly_chart(px.pie(df_t0, names="gender",
                                   title="Gender Composition"), use_container_width=True)
    else:
        st.info("No baseline data available.")

# -------------------------------------------------------
# Tab 3 – Daily Diary
# -------------------------------------------------------

with tabs[2]:
    
    st.subheader("Daily Diary")

    # ########### COPY df_days ###########
    df_days = df_days.copy()

    # ---------------------------------------
    # Merge dose_group from baseline
    # ---------------------------------------
    df_baseline = df[df["redcap_event_name"].str.contains("T0", case=False, na=False)][
        ["record_id", "dose_group"]
    ]

    df_days = df_days.merge(df_baseline, on="record_id", how="left")

    # Normalize to a single clean dose_group column
    if "dose_group_x" in df_days.columns or "dose_group_y" in df_days.columns:
        df_days["dose_group"] = df_days.get("dose_group_y").combine_first(df_days.get("dose_group_x"))
        df_days.drop(columns=[c for c in ["dose_group_x", "dose_group_y"] if c in df_days.columns], inplace=True)

    # ---------------------------------------
    # Extract Day ("Day 1", "Day 2", ...)
    # ---------------------------------------
    df_days["day"] = df_days["redcap_event_name"].str.extract(r"[Dd]ay\s*(\d+)")
    df_days = df_days[df_days["day"].notna()].copy()
    df_days["day"] = df_days["day"].astype(int)

    # ---------------------------------------
    # Clean implausible values
    # ---------------------------------------
    df_days = df_days[
        (df_days["sleep_min"].between(0, 200)) &
        (df_days["reel_minutes"].between(0, 300))
    ].copy()

    # Convert to numeric
    df_days["sleep_min"] = pd.to_numeric(df_days["sleep_min"], errors="coerce")
    df_days["reel_minutes"] = pd.to_numeric(df_days["reel_minutes"], errors="coerce")

    if df_days.empty:
        st.warning("No valid daily diary entries found.")
        st.stop()

    # -------------------------------------------------------
    # Mean Sleep/Reels per Day & per Dose Group
    # -------------------------------------------------------

    st.subheader("📈 Group-Level Daily Trends (Means per Day)")

    # Group by Day + Dose Group
    df_mean = (
        df_days.groupby(["day", "dose_group"])[["sleep_min", "reel_minutes"]]
        .mean()
        .reset_index()
    )

    cA, cB = st.columns(2)

    # Mean Sleep plot
    with cA:
        fig_mean_sleep = px.line(
            df_mean,
            x="day",
            y="sleep_min",
            color="dose_group",
            markers=True,
            title="Mean Sleep per Day per Dose Group",
            labels={"sleep_min": "Mean Sleep (min)", "dose_group": "Dose Group"},
        )
        fig_mean_sleep.update_layout(template="simple_white")
        st.plotly_chart(fig_mean_sleep, use_container_width=True)

    # Mean Reels plot
    with cB:
        fig_mean_reels = px.line(
            df_mean,
            x="day",
            y="reel_minutes",
            color="dose_group",
            markers=True,
            title="Mean Reels per Day per Dose Group",
            labels={"reel_minutes": "Mean Reels (min)", "dose_group": "Dose Group"},
        )
        fig_mean_reels.update_layout(template="simple_white")
        st.plotly_chart(fig_mean_reels, use_container_width=True)
         
                

    # -------------------------------------------------------
    # Confounder Scatter: Mean Sleep vs Mean Reels per Participant
    # With Dose Group Coloring + Trendline
    # -------------------------------------------------------

    col11, col22 = st.columns(2)

    with col11:
        if ("sleep_min" in df_days.columns) and ("reel_minutes" in df_days.columns):

            st.subheader("📊 Mean Sleep vs Mean Reels Consumption per Participant")

            # compute participant-level means
            agg = (
                df_days.groupby(["record_id", "dose_group"])[["sleep_min", "reel_minutes"]]
                .mean()
                .reset_index()
            )

            fig_mean_scatter = px.scatter(
                agg,
                x="sleep_min",
                y="reel_minutes",
                color="dose_group",
                text="record_id",
                trendline="ols",
                labels={
                    "sleep_min": "Mean Sleep (min)",
                    "reel_minutes": "Mean Reels (min)",
                     "dose_group": "Dose Group"
                })

            # aesthetics
            fig_mean_scatter.update_traces(textposition='top center')
            fig_mean_scatter.update_layout(template="simple_white")

            # set axis ranges for better interpretation
            fig_mean_scatter.update_xaxes(range=[0, 200])
            fig_mean_scatter.update_yaxes(range=[0, 200])

            st.plotly_chart(fig_mean_scatter, width="stretch")

        else:
            st.info("Missing columns sleep_min or reel_minutes.")

        # -------------------------------------------------------
        # Metrics per Dose Group
        # -------------------------------------------------------
        with col22:
            st.subheader("🎬 Daily Diary Metrics per Dose Group")

            desired_order = ["0 min/day", "60 min/day", "180 min/day", ">180 min/day"]
            groups = [g for g in desired_order if g in df_days["dose_group"].dropna().unique()]
            # Dose groups present in df_days

            if len(groups) == 0:
                st.info("No dose group information available in daily diary entries.")
            else:
                cols = st.columns(len(groups))

                for col, group in zip(cols, groups):
                    df_g = df_days[df_days["dose_group"] == group]

                    # Mean values
                    mean_sleep = df_g["sleep_min"].mean()
                    mean_reels = df_g["reel_minutes"].mean()

                    # Completion rate
                    n_participants = df_g["record_id"].nunique()
                    n_days = df_g["day"].nunique()  # usually 7 days
                    expected_entries = n_participants * n_days

                    completed_entries = (
                        df_g[["record_id", "day"]]
                        .drop_duplicates()
                        .shape[0]
                    )

                    completion_rate = (
                        completed_entries / expected_entries * 100
                        if expected_entries > 0 else 0
                    )

                                    # Output (improved card-style layout)
                    col.markdown(
                        f"""
                        <div style="
                            background-color:#F8F9FC;
                            padding:14px;
                            border-radius:12px;
                            box-shadow:0 2px 8px rgba(0,0,0,0.08);
                            border-left:6px solid #4A90E2;
                            margin-bottom:16px;
                        ">
                            <h4 style="margin-top:0;">Dose Group {group}</h4>
                            <p style="margin:4px 0;"><b>Mean Sleep:</b> {mean_sleep:.1f} min</p>
                            <p style="margin:4px 0;"><b>Mean Reels:</b> {mean_reels:.1f} min</p>
                            <p style="margin:4px 0;"><b>Completion Rate:</b> {completion_rate:.1f}%</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
# -------------------------------------------------------
# Tab 4 – Cognition & Mood
# -------------------------------------------------------
# -------------------------------------------------------
# Tab – Reels Impact Insights
# -------------------------------------------------------
# -------------------------------------------------------
# Tab 3 – Reels Impact Insights
# -------------------------------------------------------
import statsmodels.api as sm
import plotly.io as pio
pio.templates.default = "plotly_white"

# -------------------------------------------------------
# Tab 3 – Reels Impact Insights
# -------------------------------------------------------
import statsmodels.api as sm
import plotly.io as pio
pio.templates.default = "plotly_white"

# -------------------------------------------------------
# Tab 3 – Reels Impact Insights
# -------------------------------------------------------
import statsmodels.api as sm
import plotly.io as pio
pio.templates.default = "plotly_white"

with tabs[3]:
    st.subheader("Reels Impact Insights")

    # -------------------------------------------------------
    # Copy DF + extract numeric values where needed
    # -------------------------------------------------------
    df_insight = df.copy()

    print(df_insight["cognitive_score1"].to_string())

    num_cols = ["sleep_min", "fatigue_today", "mood_level",
                "cognitive_score1", "reel_minutes"]

    for col in num_cols:
        df_insight[col] = (
            df_insight[col]
            .astype(str)
            .str.extract(r"(\d+\.?\d*)")     # extracts numbers only
            .astype(float)
        )

    # -------------------------------------------------------
    # Clean event names
    # -------------------------------------------------------
    df_insight["event_clean"] = (
        df_insight["redcap_event_name"]
        .astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # -------------------------------------------------------
    # Dose group mapping from T0 Baseline
    # -------------------------------------------------------
    df_ref = df_insight[df_insight["event_clean"].str.contains("t0 baseline")]

    dose_map = (
        df_ref[["record_id", "dose_group"]]
        .drop_duplicates(subset=["record_id"])
        .set_index("record_id")["dose_group"]
    )

    df_insight["dose_group"] = df_insight.apply(
        lambda r: dose_map.get(r["record_id"], r["dose_group"]),
        axis=1
    )

    # -------------------------------------------------------
    # Normalize dose groups
    # -------------------------------------------------------
    df_insight["dose_group"] = (
        df_insight["dose_group"]
        .astype(str)
        .str.lower()
        .str.strip()
        .replace({
            "0": "0 min/day",
            "1": "60 min/day",
            "2": "180 min/day",
            "3": ">180 min/day",
            "0 min/day": "0 min/day",
            "60 min/day": "60 min/day",
            "180 min/day": "180 min/day",
            ">180 min/day": ">180 min/day",
            "": "Not completed",
            "nan": "Not completed",
            "none": "Not completed",
            "-": "Not completed"
        })
    )

    dose_order = ["0 min/day", "60 min/day", "180 min/day", ">180 min/day"]

    # -------------------------------------------------------
    # Build T1 subset
    # -------------------------------------------------------
    df_t1 = df_insight[df_insight["event_clean"].str.contains("t1 post")].copy()
    df_t1["dose_group"] = pd.Categorical(
        df_t1["dose_group"], categories=dose_order, ordered=True
    )

    # -------------------------------------------------------
    # Build Days 1–7 subset
    # -------------------------------------------------------
    df_days = df_insight[df_insight["event_clean"].str.contains("day")].copy()

    # ======================================================================
    # 1. Mood by Dose Group (T1)  — Cognitive score removed (missing in T1)
    # ======================================================================
    st.subheader("Mood by Dose Group (T1)")

    fig_mood = px.box(
        df_t1, x="dose_group", y="mood_level", points="all",
        category_orders={"dose_group": dose_order},
        title="Mood Level by Dose Group (T1)"
    )
    fig_mood.update_xaxes(categoryorder="array", categoryarray=dose_order)
    st.plotly_chart(fig_mood, use_container_width=True)

    # ======================================================================
    # 2. Regression: Reel Minutes vs Mood Level (T1)
    # ======================================================================
    st.subheader("Reel Minutes vs Mood Level (T1)")

    df_model = df_t1.dropna(subset=["reel_minutes", "mood_level"])

    if not df_model.empty:
        X = sm.add_constant(df_model["reel_minutes"])
        model = sm.OLS(df_model["mood_level"], X).fit()

        slope = model.params.get("reel_minutes", 0)
        intercept = model.params.get("const", 0)

        df_model["predicted"] = intercept + slope * df_model["reel_minutes"]

        fig_reg = px.scatter(
            df_model, x="reel_minutes", y="mood_level",
            color="dose_group", category_orders={"dose_group": dose_order},
            title="Reel Minutes vs Mood Level (OLS Fit)"
        )

        fig_reg.add_traces(
            px.line(df_model, x="reel_minutes", y="predicted").data
        )

        st.plotly_chart(fig_reg, use_container_width=True)

        st.caption(
            f"OLS model: mood = {intercept:.2f} + {slope:.3f} × reels "
            f"(p = {model.pvalues.get('reel_minutes', float('nan')):.3f})"
        )
    else:
        st.info("No valid T1 data for regression.")

    # ======================================================================
    # 3. Sleep vs Fatigue (Days 1–7)
    # ======================================================================
    st.subheader("Sleep Duration vs Fatigue (Days 1–7)")

    if not df_days.empty:
        fig_sf = px.scatter(
            df_days, x="sleep_min", y="fatigue_today",
            color="dose_group", trendline="ols",
            category_orders={"dose_group": dose_order},
            title="Sleep vs Fatigue by Dose Group (Days 1–7)"
        )
        st.plotly_chart(fig_sf, use_container_width=True)
    else:
        st.info("No daily diary data available.")

    # ======================================================================
    # 4. Correlation Matrix (T1)
    # ======================================================================
    st.subheader("Correlations (T1 Data)")

    corr_cols = ["reel_minutes", "sleep_min", "fatigue_today",
                 "mood_level", "cognitive_score1"]

    df_corr = df_t1[corr_cols].corr().round(2)

    st.dataframe(
        df_corr.style.background_gradient(cmap="RdYlBu_r", axis=None)
    )









# -------------------------------------------------------
# Tab 5 – Progress Matrix
# -------------------------------------------------------
with tabs[4]:
    st.subheader("Form Completion Overview (REDCap Style)")

    if not df.empty:
        # Alle Events definieren, die theoretisch existieren
        expected_events = [
            "T0 Baseline (Pre)",
            "Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7",
            "T1 Post"
        ]
        n_expected = len(expected_events)

        # Sicherstellen, dass Spalte existiert
        if "redcap_event_name" not in df.columns:
            st.error("Missing column 'redcap_event_name'.")
        else:
            # Berechne pro record_id, wie viele Events existieren
            event_counts = (
                df.groupby("record_id")["redcap_event_name"]
                .nunique()
                .reset_index(name="Completed Events")
            )

            # Fortschritt in %
            event_counts["Progress %"] = (event_counts["Completed Events"] / n_expected * 100).round(1)

            # Plot
            fig = px.bar(
                event_counts,
                x="record_id",
                y="Progress %",
                text="Progress %",
                color="Progress %",
                color_continuous_scale="Greens",
                title="Completion per Participant"
            )
            fig.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
            fig.update_layout(
                yaxis_range=[0, 100],
                xaxis_title="Participant ID",
                yaxis_title="Progress %",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Durchschnittsfortschritt anzeigen
            mean_progress = event_counts["Progress %"].mean()
            st.metric("Average completion", f"{mean_progress:.1f}%")

            # Optional: Tabelle anzeigen
            st.dataframe(event_counts)
    else:
        st.info("No progress data available.")



# -------------------------------------------------------
# Tab 6 – Aggregated Metrics
# -------------------------------------------------------
with tabs[5]:
    st.subheader("Aggregated Study Metrics")
    for c in ["cognitive_score1", "mood_level", "sleep_min", "fatigue_today"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean Cognitive Score", f"{df['cognitive_score1'].mean(skipna=True):.1f}")
    c2.metric("Mean Mood", f"{df['mood_level'].mean(skipna=True):.1f}")
    c3.metric("Mean Sleep (min)", f"{df['sleep_min'].mean(skipna=True):.1f}")
    c4.metric("Mean Fatigue", f"{df['fatigue_today'].mean(skipna=True):.1f}")

    st.plotly_chart(px.scatter(df_days, x="sleep_min", y="fatigue_today",
                               color="dose_group", title="Sleep vs Fatigue by Group"),
                    use_container_width=True)
