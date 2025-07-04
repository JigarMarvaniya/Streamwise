import streamlit as st
import pandas as pd
import plotly.express as px

# -------------------------------
# Load Data from CSV
# -------------------------------
@st.cache_data
def load_data():
    try:
        # Load uploaded CSV or default GitHub CSV
        if "uploaded_csv" in st.session_state:
            df = pd.read_csv(st.session_state["uploaded_csv"])
        else:
            df = pd.read_csv("streamwise_survey_synthetic.csv")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


# -------------------------------
# Sidebar Upload + Filters
# -------------------------------
def sidebar_controls():
    st.sidebar.markdown("### 📥 Upload Your StreamWise CSV")
    uploaded = st.sidebar.file_uploader("Upload streamwise_survey_synthetic.csv", type=["csv"])
    if uploaded:
        st.session_state["uploaded_csv"] = uploaded

    with st.sidebar.expander("🎯 Filter Data", expanded=True):
        st.multiselect("Gender", ["Male", "Female", "Other", "Prefer not to say"], key="gender_filter")
        st.multiselect("Income", ["<20K", "20K–40K", "40K–60K", "60K–100K", ">100K"], key="income_filter")
        st.multiselect("Location", ["Rural", "Urban", "Semi-Urban"], key="location_filter")
        st.multiselect("Billing Cycle", ["Monthly", "Quarterly", "Yearly"], key="billing_filter")
        st.multiselect("Plan Type", ["Basic", "Standard", "Premium", "Family"], key="plan_filter")


# -------------------------------
# About Tab
# -------------------------------
def about_page():
    st.image("https://raw.githubusercontent.com/JigarMarvaniya/Streamwise/new-functionality/assets/streamwise_logo.png", width=200)
    st.markdown("---")
    st.markdown("## 🧾 About StreamWise")

    st.markdown("### 🎯 **Problem Statement:**")
    st.markdown("""
    Many regional OTT platforms in emerging markets struggle to optimize subscriber retention, engagement, and pricing, 
    lacking deep analytics and data science capabilities in-house. This leads to high churn, suboptimal pricing, and poor personalization.
    """)

    st.markdown("### 📌 **Business Objectives:**")
    st.markdown("""
    - Empower OTT operators with smart, low-code analytics.  
    - Reduce churn with predictive modeling and engagement segmentation.  
    - Personalize offers and pricing based on behavioral analytics.  
    - Identify actionable user personas and business levers.  
    - Enable data-driven, MBA-grade strategic decisions through powerful, interactive dashboards.  
    """)


# -------------------------------
# Data Visualization Tab
# -------------------------------
def data_visualization_tab():
    st.markdown("## 📊 Data-Driven Business Insights")
    df = load_data()
    if df.empty:
        st.warning("No data loaded. Please upload streamwise_survey_synthetic.csv.")
        return

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Users", f"{len(df):,}")
    col2.metric("Churn Rate (%)", f"{(df['ConsideringCancellation'] == 'Yes').mean() * 100:.1f}%")
    col3.metric("Avg Weekly Watch", f"{df['AvgWeeklyWatchTime'].mean():.1f} hrs")
    col4.metric("Avg App Rating", f"{df['AppRating'].mean():.2f}")

    # Visuals
    st.subheader("Churn Rate by Billing Cycle")
    churn_billing = df.groupby("BillingCycle")["ConsideringCancellation"].apply(lambda x: (x == "Yes").mean()).reset_index()
    fig1 = px.bar(churn_billing, x="BillingCycle", y="ConsideringCancellation", color="BillingCycle")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Price Willingness vs Age")
    fig2 = px.scatter(df, x="Age", y="PriceWillingness", color="ConsideringCancellation", opacity=0.5)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Engagement vs Churn")
    fig3 = px.bar(df, x="ConsideringCancellation", y="AvgWeeklyWatchTime", color="ConsideringCancellation")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("App Rating Distribution")
    fig4 = px.histogram(df, x="AppRating", color_discrete_sequence=["red"])
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Tenure by Plan Type")
    fig5 = px.box(df, x="PlanType", y="SubscriptionTenureMonths", color="PlanType")
    st.plotly_chart(fig5, use_container_width=True)

    st.subheader("Age Distribution")
    fig6 = px.histogram(df, x="Age", color_discrete_sequence=["red"])
    st.plotly_chart(fig6, use_container_width=True)

    st.subheader("Churn by Location")
    churn_loc = df.groupby("Location")["ConsideringCancellation"].apply(lambda x: (x == "Yes").mean()).reset_index()
    fig7 = px.bar(churn_loc, x="Location", y="ConsideringCancellation", color="Location")
    st.plotly_chart(fig7, use_container_width=True)

    st.subheader("Top Valued Features")
    top_feat = df['TopFeatures'].value_counts().reset_index()
    top_feat.columns = ['Feature', 'Count']
    fig8 = px.bar(top_feat, x="Feature", y="Count", color="Count", color_continuous_scale="Reds")
    st.plotly_chart(fig8, use_container_width=True)

    st.subheader("Preferred Viewing Devices")
    dev = df['PreferredDevice'].value_counts().reset_index()
    dev.columns = ['Device', 'Count']
    fig9 = px.bar(dev, x="Device", y="Count", color="Count", color_continuous_scale="Reds")
    st.plotly_chart(fig9, use_container_width=True)

    st.subheader("Satisfaction vs Churn")
    fig10 = px.box(df, x="ConsideringCancellation", y="SatisfactionScore", color="ConsideringCancellation")
    st.plotly_chart(fig10, use_container_width=True)


# -------------------------------
# Main App
# -------------------------------
def main():
    st.set_page_config(page_title="StreamWise", layout="wide")

    # Theme
    st.markdown("""
        <style>
        .stApp { background-color: #111; color: white; }
        .css-10trblm { color: #ff4b4b; }
        .block-container { padding-top: 2rem; }
        .stTabs [data-baseweb="tab-list"] button {
            color: white;
            background-color: #1c1c1c;
            border: none;
        }
        .stTabs [aria-selected="true"] {
            border-bottom: 3px solid red;
            color: red;
        }
        </style>
    """, unsafe_allow_html=True)

    sidebar_controls()

    tabs = st.tabs([
        "📌 About StreamWise",
        "📊 Data Visualization",
        "🧠 Classification",
        "🧬 Clustering & Persona",
        "📈 Association Rules",
        "🧮 Regression"
    ])

    with tabs[0]: about_page()
    with tabs[1]: data_visualization_tab()
    with tabs[2]: st.info("Coming soon: Classification models.")
    with tabs[3]: st.info("Coming soon: Clustering and persona segments.")
    with tabs[4]: st.info("Coming soon: Association rule mining.")
    with tabs[5]: st.info("Coming soon: Regression-based predictions.")


if __name__ == "__main__":
    main()
