"""
Customer Segmentation Dashboard
=================================
Interactive Streamlit dashboard for exploring the customer segments produced
by scripts/run_analysis.py (or notebooks/customer_segmentation.ipynb).

Run locally:
    streamlit run dashboard/app.py

Expects outputs/customers_with_segments.csv relative to the project root.
If it isn't there yet, run the analysis pipeline first:
    python scripts/run_analysis.py
"""

import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Customer Segmentation Dashboard", page_icon="🧩",
                    layout="wide", initial_sidebar_state="expanded")

PRIMARY = "#2563EB"
PALETTE = ["#2563EB", "#F59E0B", "#10B981", "#EF4444", "#8B5CF6", "#06B6D4"]

st.markdown("""
<style>
    .stMetric { background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 14px; }
    div[data-testid="stMetricValue"] { font-size: 1.5rem; color: #1E3A8A; }
    .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "..", "outputs", "customers_with_segments.csv"),
        "outputs/customers_with_segments.csv",
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        return None, None
    df = pd.read_csv(path)
    summary_path = os.path.join(os.path.dirname(path), "segment_summary.json")
    summary = None
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
    return df, summary


df, summary = load_data()

if df is None:
    st.error(
        "Could not find `outputs/customers_with_segments.csv`.\n\n"
        "Run the analysis pipeline first from the project root:\n\n"
        "```bash\npython scripts/run_analysis.py\n```\n"
        "or execute `notebooks/customer_segmentation.ipynb`, then relaunch this dashboard."
    )
    st.stop()

segment_names = sorted(df["SegmentName"].unique().tolist())
seg_color_map = {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(segment_names)}

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
st.sidebar.title("🧩 Filters")
sel_segments = st.sidebar.multiselect("Segment", segment_names, default=segment_names)
sel_tiers = st.sidebar.multiselect("Membership tier", sorted(df["MembershipTier"].unique().tolist()),
                                     default=sorted(df["MembershipTier"].unique().tolist()))
sel_channels = st.sidebar.multiselect("Preferred channel", sorted(df["PreferredChannel"].unique().tolist()),
                                        default=sorted(df["PreferredChannel"].unique().tolist()))
age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
age_range = st.sidebar.slider("Age range", age_min, age_max, (age_min, age_max))

st.sidebar.divider()
if summary:
    st.sidebar.caption(f"Model: K-Means, K={summary['k_selected']}, "
                        f"silhouette score={summary['silhouette_score']}")
st.sidebar.caption(
    "Data source: synthetic customer dataset generated for this project "
    "(see `scripts/generate_data.py`). Replace `data/customer_data_raw.csv` "
    "with your own data (same columns) to segment real customers."
)

mask = (
    df["SegmentName"].isin(sel_segments)
    & df["MembershipTier"].isin(sel_tiers)
    & df["PreferredChannel"].isin(sel_channels)
    & df["Age"].between(age_range[0], age_range[1])
)
fdf = df.loc[mask].copy()

if fdf.empty:
    st.warning("No customers match the selected filters.")
    st.stop()

# ---------------------------------------------------------------------------
# Header + KPIs
# ---------------------------------------------------------------------------
st.title("🧩 Customer Segmentation Dashboard")
st.caption(f"Showing **{len(fdf):,}** customers across **{fdf['SegmentName'].nunique()}** segments")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Customers", f"{len(fdf):,}")
k2.metric("Avg Total Spend", f"${fdf['MonetaryTotal'].mean():,.0f}")
k3.metric("Avg Frequency", f"{fdf['Frequency'].mean():.1f}")
k4.metric("Avg Recency (days)", f"{fdf['RecencyDays'].mean():.0f}")
k5.metric("Avg Annual Income", f"${fdf['AnnualIncome'].mean():,.0f}")

st.divider()

tab_overview, tab_profiles, tab_explore, tab_data = st.tabs(
    ["📊 Segment Overview", "🎯 Segment Profiles", "🔍 Explore", "🗂️ Raw Data"]
)

# --- Overview ------------------------------------------------------------
with tab_overview:
    col1, col2 = st.columns(2)

    counts = fdf["SegmentName"].value_counts().reset_index()
    counts.columns = ["Segment", "Customers"]
    fig1 = px.bar(counts, x="Segment", y="Customers", color="Segment",
                   color_discrete_map=seg_color_map, title="Customer Count by Segment")
    fig1.update_layout(showlegend=False)
    col1.plotly_chart(fig1, use_container_width=True)

    fig2 = px.pie(counts, names="Segment", values="Customers", title="Segment Share",
                   color="Segment", color_discrete_map=seg_color_map, hole=0.4)
    col2.plotly_chart(fig2, use_container_width=True)

    if "PC1" in fdf.columns and "PC2" in fdf.columns:
        fig3 = px.scatter(fdf, x="PC1", y="PC2", color="SegmentName", color_discrete_map=seg_color_map,
                           title="Segments in PCA Space", opacity=0.7,
                           hover_data=["CustomerID", "MonetaryTotal", "Frequency", "RecencyDays"])
        st.plotly_chart(fig3, use_container_width=True)

# --- Profiles -----------------------------------------------------------
with tab_profiles:
    metric_cols = ["Age", "AnnualIncome", "TenureMonths", "RecencyDays", "Frequency",
                    "MonetaryTotal", "AvgBasketValue"]
    profile = fdf.groupby("SegmentName")[metric_cols].mean().round(1)
    profile["CustomerCount"] = fdf.groupby("SegmentName").size()
    st.subheader("Average metrics per segment")
    st.dataframe(profile.style.background_gradient(cmap="Blues", subset=metric_cols),
                 use_container_width=True)

    metric_choice = st.selectbox("Compare a metric across segments", metric_cols, index=5)
    fig4 = px.bar(profile.reset_index(), x="SegmentName", y=metric_choice, color="SegmentName",
                   color_discrete_map=seg_color_map, title=f"Average {metric_choice} by Segment")
    fig4.update_layout(showlegend=False)
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Segment vs. membership tier")
    tier_ct = pd.crosstab(fdf["SegmentName"], fdf["MembershipTier"])
    fig5 = px.imshow(tier_ct, text_auto=True, color_continuous_scale="Blues",
                       title="Segment x Membership Tier (customer counts)")
    st.plotly_chart(fig5, use_container_width=True)

# --- Explore --------------------------------------------------------------
with tab_explore:
    col1, col2 = st.columns(2)
    x_axis = col1.selectbox("X axis", metric_cols, index=4)
    y_axis = col2.selectbox("Y axis", metric_cols, index=5)
    fig6 = px.scatter(fdf, x=x_axis, y=y_axis, color="SegmentName", color_discrete_map=seg_color_map,
                        opacity=0.7, hover_data=["CustomerID"],
                        title=f"{y_axis} vs. {x_axis} by Segment")
    st.plotly_chart(fig6, use_container_width=True)

    fig7 = px.box(fdf, x="SegmentName", y=y_axis, color="SegmentName", color_discrete_map=seg_color_map,
                   title=f"Distribution of {y_axis} by Segment")
    fig7.update_layout(showlegend=False)
    st.plotly_chart(fig7, use_container_width=True)

# --- Raw data ---------------------------------------------------------
with tab_data:
    st.subheader("Filtered customer data")
    st.dataframe(fdf, use_container_width=True, height=420)
    st.download_button("⬇️ Download filtered data as CSV",
                        data=fdf.to_csv(index=False).encode("utf-8"),
                        file_name="filtered_customer_segments.csv", mime="text/csv")

st.divider()
st.caption("Built with Streamlit + Plotly · Customer Segmentation Analysis project · "
           "Data is synthetic and generated for demonstration purposes.")
