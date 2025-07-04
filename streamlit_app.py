import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="StreamWise", layout="wide")

# -------------------- SIDEBAR --------------------
st.sidebar.title("📂 Upload your StreamWise Survey CSV")

# Upload CSV
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    default_path = "streamwise_survey_synthetic.csv"
    if os.path.exists(default_path):
        df = pd.read_csv(default_path)
    else:
        st.error("❌ No data file found. Please upload a CSV.")
        st.stop()

# -------------------- FILTERS --------------------
st.sidebar.markdown("### 🎛️ Filter Data")

def multi_filter(col):
    options = df[col].dropna().unique().tolist()
    selected = st.sidebar.multiselect(f"{col}", options, default=options)
    return df[df[col].isin(selected)]

filter_cols = ["Gender", "Income", "Location", "BillingCycle", "PlanType"]
for col in filter_cols:
    df = multi_filter(col)

# -------------------- TABS --------------------
tabs = st.tabs([
    "📌 About StreamWise",
    "📊 Data Visualization",
    "🧠 Classification",
    "🧬 Clustering & Persona",
    "📈 Association Rules",
    "🧮 Regression"
])

# -------------------- TAB 1: INTRO --------------------
with tabs[0]:
    st.title("About StreamWise")

    if os.path.exists("streamwise_logo.png"):
        st.image("streamwise_logo.png", width=200)

    st.markdown("### **Problem Statement:**")
    st.write(
        "Many regional OTT platforms in emerging markets struggle to optimize subscriber retention, "
        "engagement, and pricing, lacking deep analytics and data science capabilities in-house. "
        "This leads to high churn, suboptimal pricing, and poor personalization."
    )

    st.markdown("### **Business Objectives:**")
    st.markdown("""
    - Empower OTT operators with smart, low-code analytics.
    - Reduce churn with predictive modeling and engagement segmentation.
    - Personalize offers and pricing based on behavioral analytics.
    - Identify actionable user personas and business levers.
    - Enable data-driven, MBA-grade strategic decisions through powerful, interactive dashboards.
    """)

# -------------------- TAB 2: DATA VISUALIZATION --------------------
with tabs[1]:
    st.subheader("📊 Satisfaction vs Churn")
    if "AppRating" in df.columns and "ConsideringCancellation" in df.columns:
        fig, ax = plt.subplots()
        df.boxplot(column="AppRating", by="ConsideringCancellation", ax=ax)
        plt.title("App Rating by Cancellation Intent")
        plt.suptitle("")
        plt.xlabel("Considering Cancellation")
        plt.ylabel("App Rating")
        st.pyplot(fig)
    else:
        st.warning("Columns `AppRating` or `ConsideringCancellation` missing from dataset.")

# -------------------- PLACEHOLDER TABS --------------------
with tabs[2]:
    st.info("Coming soon: Classification module.")

with tabs[3]:
    st.info("Coming soon: Clustering & Persona module.")

with tabs[4]:
    st.info("Coming soon: Association Rules module.")

with tabs[5]:
    st.info("Coming soon: Regression module.")
