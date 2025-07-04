import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Set page config
st.set_page_config(page_title="StreamWise Dashboard", layout="wide")

# Load logo and audio
def load_logo_audio():
    try:
        st.image("streamwise_logo.png", width=150)
    except Exception:
        st.warning("Logo not found.")
    try:
        st.audio("streamwise_audio.mp3", format="audio/mp3", start_time=0)
    except Exception:
        st.warning("Intro audio file not found. Skipping audio...")

# Load dataset
@st.cache_data
def load_data():
    try:
        return pd.read_csv("streamwise_survey_synthetic.csv")
    except FileNotFoundError:
        st.error("Dataset not found! Please upload `streamwise_survey_synthetic.csv` using the sidebar.")
        return pd.DataFrame()

# Sidebar controls
def sidebar_filters(df):
    st.sidebar.title("📁 Upload your StreamWise Survey CSV")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

    st.sidebar.header("🎛️ Filter Data")
    gender = st.sidebar.multiselect("Gender", options=df["Gender"].dropna().unique(), default=df["Gender"].dropna().unique())
    income = st.sidebar.multiselect("Income", options=df["Income"].dropna().unique(), default=df["Income"].dropna().unique())
    location = st.sidebar.multiselect("Location", options=df["Location"].dropna().unique(), default=df["Location"].dropna().unique())
    billing = st.sidebar.multiselect("BillingCycle", options=df["BillingCycle"].dropna().unique(), default=df["BillingCycle"].dropna().unique())
    plan = st.sidebar.multiselect("PlanType", options=df["PlanType"].dropna().unique(), default=df["PlanType"].dropna().unique())

    df_filtered = df[
        df["Gender"].isin(gender) &
        df["Income"].isin(income) &
        df["Location"].isin(location) &
        df["BillingCycle"].isin(billing) &
        df["PlanType"].isin(plan)
    ]
    return df_filtered

# About Page
def about_tab():
    st.title("📌 About StreamWise")
    st.markdown("### Problem Statement:")
    st.markdown(
        "Many regional OTT platforms in emerging markets struggle to optimize subscriber retention, engagement, and pricing, "
        "lacking deep analytics and data science capabilities in-house. This leads to high churn, suboptimal pricing, and poor personalization."
    )

    st.markdown("### Business Objectives:")
    st.markdown("""
    - 🎯 Empower OTT operators with smart, low-code analytics.  
    - 📉 Reduce churn with predictive modeling and engagement segmentation.  
    - 🎁 Personalize offers and pricing based on behavioral analytics.  
    - 🧠 Identify actionable user personas and business levers.  
    - 📊 Enable data-driven, MBA-grade strategic decisions through interactive dashboards.  
    """)

# Data Visualization Page
def data_visualization_tab(df):
    st.title("📊 Data Visualization")
    if df.empty:
        st.info("Upload data to view dashboard insights.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Age Distribution")
        fig1 = px.histogram(df, x="Age", nbins=20, color_discrete_sequence=["#FF4B4B"])
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("App Rating Distribution")
        fig2 = px.histogram(df, x="AppRating", nbins=10, color_discrete_sequence=["#36A2EB"])
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Tenure by Plan Type")
        fig3 = px.box(df, x="PlanType", y="Tenure", color="PlanType")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.subheader("Churn Rate by Billing Cycle")
        churn_by_billing = df.groupby("BillingCycle")["Churn"].mean().reset_index()
        fig4 = px.bar(churn_by_billing, x="BillingCycle", y="Churn", color="BillingCycle")
        st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Price Willingness vs Age")
    fig5 = px.scatter(df, x="Age", y="WillingnessToPay", color="Gender", trendline="ols")
    st.plotly_chart(fig5, use_container_width=True)

    st.subheader("Engagement vs Churn")
    fig6 = px.box(df, x="Churn", y="EngagementScore", color="Churn")
    st.plotly_chart(fig6, use_container_width=True)

    if "TopFeature" in df.columns:
        st.subheader("Top Valued Features")
        top_feat = df["TopFeature"].value_counts().reset_index()
        top_feat.columns = ["Feature", "Count"]
        fig7 = px.pie(top_feat, values="Count", names="Feature", hole=0.4)
        st.plotly_chart(fig7, use_container_width=True)

    if "DeviceUsed" in df.columns:
        st.subheader("Preferred Devices")
        device_data = df["DeviceUsed"].value_counts().reset_index()
        device_data.columns = ["Device", "Count"]
        fig8 = px.bar(device_data, x="Device", y="Count", color="Device")
        st.plotly_chart(fig8, use_container_width=True)

    st.subheader("Satisfaction vs Churn")
    fig9 = px.violin(df, y="Satisfaction", x="Churn", box=True, color="Churn")
    st.plotly_chart(fig9, use_container_width=True)

    st.subheader("Churn by Location")
    churn_by_loc = df.groupby("Location")["Churn"].mean().reset_index()
    fig10 = px.bar(churn_by_loc, x="Location", y="Churn", color="Location")
    st.plotly_chart(fig10, use_container_width=True)

# Main App
def main():
    load_logo_audio()
    df = load_data()
    df_filtered = sidebar_filters(df)

    tabs = st.tabs([
        "📌 About StreamWise",
        "📊 Data Visualization",
        "🧠 Classification",
        "🧬 Clustering & Persona",
        "📈 Association Rules",
        "🧮 Regression"
    ])

    with tabs[0]:
        about_tab()
    with tabs[1]:
        data_visualization_tab(df_filtered)
    with tabs[2]:
        st.info("Coming soon: Classification module.")
    with tabs[3]:
        st.info("Coming soon: Clustering & Persona module.")
    with tabs[4]:
        st.info("Coming soon: Association Rules module.")
    with tabs[5]:
        st.info("Coming soon: Regression module.")

if __name__ == "__main__":
    main()
