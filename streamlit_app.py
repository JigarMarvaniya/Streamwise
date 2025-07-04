import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide", page_title="StreamWise OTT Dashboard", page_icon="📊")

@st.cache_data
def load_data():
    try:
        return pd.read_csv("streamwise_survey_synthetic.csv")
    except Exception as e:
        st.error("CSV file not found. Please upload it in the sidebar.")
        return pd.DataFrame()

# Sidebar
with st.sidebar:
    st.markdown("## 📥 Upload your StreamWise Survey CSV")
    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
    else:
        df = load_data()

    if not df.empty:
        st.markdown("---")
        st.markdown("### Filter Data")

        # Sidebar filters
        gender_filter = st.multiselect("Gender", df["Gender"].unique(), default=list(df["Gender"].unique()))
        income_filter = st.multiselect("Income", df["Income"].unique(), default=list(df["Income"].unique()))
        location_filter = st.multiselect("Location", df["Location"].unique(), default=list(df["Location"].unique()))
        billing_filter = st.multiselect("BillingCycle", df["BillingCycle"].unique(), default=list(df["BillingCycle"].unique()))
        plan_filter = st.multiselect("PlanType", df["PlanType"].unique(), default=list(df["PlanType"].unique()))

        df = df[
            (df["Gender"].isin(gender_filter)) &
            (df["Income"].isin(income_filter)) &
            (df["Location"].isin(location_filter)) &
            (df["BillingCycle"].isin(billing_filter)) &
            (df["PlanType"].isin(plan_filter))
        ]

# Tabs
tabs = st.tabs([
    "📌 About StreamWise",
    "📊 Data Visualization",
    "🧠 Classification",
    "🧬 Clustering & Persona",
    "📈 Association Rules",
    "🧮 Regression"
])

# Tab 1: About StreamWise
with tabs[0]:
    st.image("streamwise_logo.png", width=200)
    st.markdown("## About StreamWise")
    st.markdown("""
    **Problem Statement:**  
    Many regional OTT platforms in emerging markets struggle to optimize subscriber retention, engagement, and pricing, lacking deep analytics and data science capabilities in-house.  
    This leads to high churn, suboptimal pricing, and poor personalization.

    **Business Objectives:**
    - Empower OTT operators with smart, low-code analytics.
    - Reduce churn with predictive modeling and engagement segmentation.
    - Personalize offers and pricing based on behavioral analytics.
    - Identify actionable user personas and business levers.
    - Enable data-driven, MBA-grade strategic decisions through powerful, interactive dashboards.
    """)

# Tab 2: Data Visualization
with tabs[1]:
    if df.empty:
        st.warning("Please upload a valid CSV file with data.")
    else:
        st.markdown("## 📊 Dashboard Visualizations")

        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Total Users", len(df))
        kpi2.metric("Churned Users", df[df['Churn'] == 1].shape[0])
        kpi3.metric("Avg Tenure", round(df['Tenure'].mean(), 1))

        st.markdown("---")

        # Churn by BillingCycle
        fig1 = px.histogram(df, x="BillingCycle", color="Churn", barmode="group", title="Churn by Billing Cycle")
        st.plotly_chart(fig1, use_container_width=True)

        # Price willingness by Age
        fig2 = px.box(df, x="PriceWillingness", y="Age", color="Churn", title="Price Willingness vs Age")
        st.plotly_chart(fig2, use_container_width=True)

        # Engagement vs Churn
        fig3 = px.scatter(df, x="EngagementScore", y="Churn", color="PlanType", title="Engagement vs Churn")
        st.plotly_chart(fig3, use_container_width=True)

        # App Rating Distribution
        fig4 = px.histogram(df, x="AppRating", nbins=10, title="App Rating Distribution")
        st.plotly_chart(fig4, use_container_width=True)

        # Tenure by PlanType
        fig5 = px.box(df, x="PlanType", y="Tenure", title="Tenure by Plan Type")
        st.plotly_chart(fig5, use_container_width=True)

        # Churn by Location
        fig6 = px.bar(df.groupby("Location")["Churn"].mean().reset_index(), x="Location", y="Churn", title="Churn Rate by Location")
        st.plotly_chart(fig6, use_container_width=True)

        # Satisfaction vs Churn (real chart)
        fig7 = px.box(df, x="Churn", y="Satisfaction", color="Churn", title="Satisfaction vs Churn")
        st.plotly_chart(fig7, use_container_width=True)

        # Top Features (only if exists)
        if "TopFeatures" in df.columns:
            top_feat = df['TopFeatures'].value_counts().reset_index()
            top_feat.columns = ["Feature", "Count"]
            fig8 = px.bar(top_feat, x="Feature", y="Count", title="Top Valued Features")
            st.plotly_chart(fig8, use_container_width=True)
        else:
            st.info("🛑 'TopFeatures' column not found in your dataset. Skipping feature preference chart.")

        # Device preference
        if "Device" in df.columns:
            fig9 = px.pie(df, names="Device", title="Preferred Devices")
            st.plotly_chart(fig9, use_container_width=True)

        # Age distribution
        fig10 = px.histogram(df, x="Age", nbins=20, title="Age Distribution")
        st.plotly_chart(fig10, use_container_width=True)

# Placeholder tabs
for i in range(2, len(tabs)):
    with tabs[i]:
        st.markdown("### 🚧 Coming soon: This module will be live shortly.")
