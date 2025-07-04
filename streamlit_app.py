import streamlit as st

# -------------------------------
# Sidebar Upload + Filters
# -------------------------------
def sidebar_controls():
    st.sidebar.markdown("### 📥 Upload your StreamWise Survey CSV")
    st.sidebar.file_uploader("Drag and drop file here", type=["csv"], help="Limit 200MB per file • CSV only")

    with st.sidebar.expander("🎯 Filter Data", expanded=True):
        # Gender
        st.markdown("**Gender**")
        st.multiselect(" ", ["Male", "Female", "Other", "Prefer not to say"],
                       default=["Male", "Female", "Other", "Prefer not to say"], key="gender_filter")

        # Income
        st.markdown("**Income**")
        st.multiselect("  ", ["<20K", "20K–40K", "40K–60K", "60K–100K", ">100K"],
                       default=["<20K", "20K–40K", "40K–60K", "60K–100K", ">100K"], key="income_filter")

        # Location
        st.markdown("**Location**")
        st.multiselect("   ", ["Dubai", "Abu Dhabi", "Sharjah", "Others"],
                       default=["Dubai", "Abu Dhabi", "Sharjah", "Others"], key="location_filter")

        # Billing Cycle
        st.markdown("**Billing Cycle**")
        st.multiselect("    ", ["Monthly", "Quarterly", "Annually"],
                       default=["Monthly", "Quarterly", "Annually"], key="billing_filter")

        # Plan Type
        st.markdown("**Plan Type**")
        st.multiselect("     ", ["Basic", "Standard", "Premium"],
                       default=["Basic", "Standard", "Premium"], key="plan_filter")


# -------------------------------
# About Tab Content
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
# Main App Layout
# -------------------------------
def main():
    st.set_page_config(page_title="StreamWise", layout="wide")

    # Custom CSS for styling
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

    # Navigation tabs
    tab = st.tabs([
        "📌 About StreamWise",
        "📊 Data Visualization",
        "🧠 Classification",
        "🧬 Clustering & Persona",
        "📈 Association Rules",
        "🧮 Regression"
    ])

    with tab[0]:
        about_page()

    with tab[1]:
        st.info("Coming soon: Dashboard visualizations.")

    with tab[2]:
        st.info("Coming soon: Classification models.")

    with tab[3]:
        st.info("Coming soon: Clustering and persona segments.")

    with tab[4]:
        st.info("Coming soon: Market basket association rules.")

    with tab[5]:
        st.info("Coming soon: Regression-based predictions.")


if __name__ == "__main__":
    main()
