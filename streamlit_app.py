import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Page setup
st.set_page_config(page_title="StreamWise Dashboard", layout="wide")

# Load branding
def load_logo_audio():
    if os.path.exists("streamwise_logo.png"):
        st.image("streamwise_logo.png", width=150)
    else:
        st.warning("⚠️ Logo not found.")
    
    if os.path.exists("streamwise_audio.mp3"):
        st.audio("streamwise_audio.mp3", format="audio/mp3")
    else:
        st.warning("🎧 Intro audio not found.")

# Load dataset
@st.cache_data
def load_data():
    try:
        return pd.read_csv("streamwise_survey_synthetic.csv")
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return pd.DataFrame()

# Filters
def sidebar_filters(df):
    st.sidebar.title("📁 Upload Your Dataset")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

    st.sidebar.header("🎛️ Filters")
    gender = st.sidebar.multiselect("Gender", df["Gender"].unique(), default=df["Gender"].unique())
    income = st.sidebar.multiselect("Income", df["Income"].unique(), default=df["Income"].unique())
    location = st.sidebar.multiselect("Location", df["Location"].unique(), default=df["Location"].unique())
    billing = st.sidebar.multiselect("BillingCycle", df["BillingCycle"].unique(), default=df["BillingCycle"].unique())
    plan = st.sidebar.multiselect("PlanType", df["PlanType"].unique(), default=df["PlanType"].unique())

    df_filtered = df[
        (df["Gender"].isin(gender)) &
        (df["Income"].isin(income)) &
        (df["Location"].isin(location)) &
        (df["BillingCycle"].isin(billing)) &
        (df["PlanType"].isin(plan))
    ]
    return df_filtered

# About Tab
def about_tab():
    st.title("📌 About StreamWise")
    st.markdown("""
    **Problem Statement:**  
    Many regional OTT platforms in emerging markets struggle to optimize subscriber retention, engagement, and pricing, lacking deep analytics and data science capabilities in-house.

    **Business Objectives:**  
    - Empower OTT operators with smart, low-code analytics  
    - Reduce churn via predictive modeling  
    - Personalize offers using behavioral segmentation  
    - Drive strategic decisions with intuitive dashboards
    """)

# Data Visualization Tab
def data_visualization_tab(df):
    st.title("📊 Data Visualization")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Age Distribution")
        st.plotly_chart(px.histogram(df, x="Age", nbins=20, color_discrete_sequence=["#FF4B4B"]), use_container_width=True)

    with col2:
        st.subheader("App Rating Distribution")
        st.plotly_chart(px.histogram(df, x="AppRating", nbins=10, color_discrete_sequence=["#36A2EB"]), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Tenure by Plan Type")
        st.plotly_chart(px.box(df, x="PlanType", y="SubscriptionTenureMonths", color="PlanType"), use_container_width=True)

    with col4:
        st.subheader("Churn Rate by Billing Cycle")
        churn_data = df.groupby("BillingCycle")["ConsideringCancellation"].apply(lambda x: (x == "Yes").mean()).reset_index(name="ChurnRate")
        st.plotly_chart(px.bar(churn_data, x="BillingCycle", y="ChurnRate", color="BillingCycle"), use_container_width=True)

    st.subheader("Price Willingness vs Age")
    st.plotly_chart(px.scatter(df, x="Age", y="PriceWillingness", color="Gender", trendline="ols"), use_container_width=True)

    st.subheader("Engagement (WatchTime) vs Churn")
    st.plotly_chart(px.box(df, x="ConsideringCancellation", y="AvgWeeklyWatchTime", color="ConsideringCancellation"), use_container_width=True)

    if "ValuedFeatures" in df.columns:
        st.subheader("Top Valued Features")
        top_vals = df["ValuedFeatures"].value_counts().head(10).reset_index()
        top_vals.columns = ["Feature", "Count"]
        st.plotly_chart(px.bar(top_vals, x="Feature", y="Count", color="Feature"), use_container_width=True)

    if "ViewingDevices" in df.columns:
        st.subheader("Preferred Devices")
        dev = df["ViewingDevices"].value_counts().reset_index()
        dev.columns = ["Device", "Count"]
        st.plotly_chart(px.pie(dev, names="Device", values="Count", hole=0.4), use_container_width=True)

    st.subheader("Satisfaction (App Rating) vs Churn")
    st.plotly_chart(px.box(df, x="ConsideringCancellation", y="AppRating", color="ConsideringCancellation"), use_container_width=True)

    st.subheader("Churn by Location")
    churn_loc = df.groupby("Location")["ConsideringCancellation"].apply(lambda x: (x == "Yes").mean()).reset_index(name="ChurnRate")
    st.plotly_chart(px.bar(churn_loc, x="Location", y="ChurnRate", color="Location"), use_container_width=True)

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
