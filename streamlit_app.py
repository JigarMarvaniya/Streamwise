import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="StreamWise", layout="wide")

# Load Data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("streamwise_survey_synthetic.csv")
        return df
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return pd.DataFrame()

df = load_data()

# Sidebar Filters
def show_sidebar_filters(df):
    st.sidebar.title("📂 Filter Data")

    gender = st.sidebar.multiselect("Gender", df["Gender"].unique(), default=df["Gender"].unique())
    income = st.sidebar.multiselect("Income", df["Income"].unique(), default=df["Income"].unique())
    location = st.sidebar.multiselect("Location", df["Location"].unique(), default=df["Location"].unique())
    billing = st.sidebar.multiselect("BillingCycle", df["BillingCycle"].unique(), default=df["BillingCycle"].unique())
    plan = st.sidebar.multiselect("PlanType", df["PlanType"].unique(), default=df["PlanType"].unique())

    filtered_df = df[
        (df["Gender"].isin(gender)) &
        (df["Income"].isin(income)) &
        (df["Location"].isin(location)) &
        (df["BillingCycle"].isin(billing)) &
        (df["PlanType"].isin(plan))
    ]
    return filtered_df

# Header
def show_header():
    st.markdown("## 📌 About StreamWise")
    if os.path.exists("streamwise_logo.png"):
        st.image("streamwise_logo.png", width=180)
    else:
        st.warning("Logo file not found. Skipping image...")

# Intro Content
def show_about():
    st.markdown("### About StreamWise")
    st.markdown("""
    **Problem Statement:**  
    Many regional OTT platforms in emerging markets struggle to optimize subscriber retention, engagement, and pricing, lacking deep analytics and data science capabilities in-house. This leads to high churn, suboptimal pricing, and poor personalization.

    **Business Objectives:**
    - Empower OTT operators with smart, low-code analytics.
    - Reduce churn with predictive modeling and engagement segmentation.
    - Personalize offers and pricing based on behavioral analytics.
    - Identify actionable user personas and business levers.
    - Enable data-driven, MBA-grade strategic decisions through powerful, interactive dashboards.
    """)

# Data Visualization Tab
def data_visualization_tab(df):
    st.markdown("## 📊 Data Visualization")
    try:
        fig, ax = plt.subplots(figsize=(8,5))
        sns.boxplot(data=df, x="ConsideringCancellation", y="AppRating", palette="coolwarm", ax=ax)
        ax.set_title("Satisfaction vs Churn")
        ax.set_xlabel("Considering Cancellation?")
        ax.set_ylabel("App Rating")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error rendering chart: {e}")

# App Runner
def main():
    st.markdown("# 🎬 Welcome to StreamWise")
    st.markdown("""
    Your all-in-one, MBA-level OTT analytics and churn prediction dashboard.  
    _Analyze. Segment. Act._ – in true Netflix style.
    """)
    
    tabs = st.tabs([
        "📌 About StreamWise", 
        "📊 Data Visualization", 
        "🧠 Classification", 
        "🧬 Clustering & Persona", 
        "📈 Association Rules", 
        "🧮 Regression"
    ])

    with tabs[0]:
        show_header()
        show_about()

    with tabs[1]:
        filtered_df = show_sidebar_filters(df)
        data_visualization_tab(filtered_df)

    with tabs[2]:
        st.info("Coming soon: Classification module...")

    with tabs[3]:
        st.info("Coming soon: Clustering & Persona module...")

    with tabs[4]:
        st.info("Coming soon: Association Rules module...")

    with tabs[5]:
        st.info("Coming soon: Regression module...")

if __name__ == "__main__":
    main()
