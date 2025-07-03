import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.express as px
import plotly.graph_objects as go
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc,
    mean_squared_error, mean_absolute_error, silhouette_score
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge, Lasso

try:
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
    MLXTEND_INSTALLED = True
except ImportError:
    MLXTEND_INSTALLED = False

try:
    import networkx as nx
    NETWORKX_INSTALLED = True
except ImportError:
    NETWORKX_INSTALLED = False

st.set_page_config(page_title="StreamWise | OTT Analytics Platform", layout="wide")

# ----- STYLING -----
st.markdown("""
    <style>
        body, .stApp {background-color: #141414 !important;}
        h1, h2, h3, h4, h5, h6, label, .stTextInput, .stSelectbox, .stMultiSelect, .stSlider, .stRadio {
            color: #f7f7f7 !important;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #141414 !important;
            color: #f7f7f7 !important;
            font-weight: bold;
            border-bottom: 4px solid #E50914 !important;
            font-size: 1.1rem;
        }
        .stTabs [aria-selected="false"] {
            border-bottom: 4px solid #222 !important;
            color: #888 !important;
        }
        .stButton>button {
            background-color: #E50914 !important;
            color: #fff !important;
            font-weight: bold !important;
            border-radius: 7px !important;
            border: none;
            font-size: 1.09rem !important;
        }
        .stButton>button:hover {
            background-color: #b0060c !important;
            box-shadow: 0 2px 16px #e5091480;
        }
        .metric-label, .stMetric {color: #fff !important;}
    </style>
    """, unsafe_allow_html=True)

# ---- LOGO & INTRO ----
def show_logo():
    if os.path.exists("streamwise_logo.png"):
        st.image("streamwise_logo.png", width=170)
    else:
        st.markdown("<div style='font-size:2.1rem;font-weight:800;color:#E50914;text-align:center;'>StreamWise</div>", unsafe_allow_html=True)

def show_intro():
    show_logo()
    st.markdown("<h1 style='color:#f7f7f7; margin-top:0.1em;'>Welcome to <span style='color:#E50914;'>StreamWise</span></h1>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:1.1rem; max-width:700px; color:#bbb;'>"
        "Your all-in-one, MBA-level OTT analytics and churn prediction dashboard.<br>"
        "<b>Analyze. Segment. Act.</b> – in true Netflix style.<br><br>"
        "Click the <span style='color:#E50914;font-weight:bold;'>StreamWise</span> button to get started!"
        "</div>", unsafe_allow_html=True
    )
    audio_placeholder = st.empty()
    if st.button("▶ StreamWise", key="intro_btn"):
        if os.path.exists("streamwise_audio.mp3"):
            audio_placeholder.audio("streamwise_audio.mp3", format='audio/mp3', start_time=0)
            time.sleep(3)
        st.session_state.show_intro = False
        st.experimental_rerun()
    st.markdown("<div style='margin-top:60px;color:#aaa;font-size:0.97rem;text-align:center;'>Inspired by Netflix • Powered by Streamlit</div>", unsafe_allow_html=True)
    st.stop()

if "show_intro" not in st.session_state:
    st.session_state.show_intro = True

if st.session_state.show_intro:
    show_intro()

with st.sidebar:
    if st.button("🏠 Home / Introduction"):
        st.session_state.show_intro = True
        st.experimental_rerun()

# ---- DATA UPLOAD & FILTERS ----
show_logo()
st.markdown("<hr style='border-top: 2.5px solid #E50914;'>", unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("Upload your StreamWise Survey CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
elif os.path.exists("streamwise_survey_synthetic.csv"):
    df = pd.read_csv("streamwise_survey_synthetic.csv")
else:
    st.error("No data file uploaded and no local 'streamwise_survey_synthetic.csv' found!")
    st.stop()

with st.sidebar:
    st.header("Filter Data")
    filter_cols = ['Gender', 'Income', 'Location', 'BillingCycle', 'PlanType']
    for col in filter_cols:
        if col in df.columns:
            opts = df[col].dropna().unique().tolist()
            chosen = st.multiselect(col, opts, default=opts, key=col)
            df = df[df[col].isin(chosen)]

tab_names = [
    "📝 About StreamWise",
    "📊 Data Visualization",
    "🧠 Classification",
    "👥 Clustering & Persona",
    "🔗 Association Rules",
    "📈 Regression"
]
tabs = st.tabs(tab_names)

# ---- ABOUT TAB ----
with tabs[0]:
    st.header("About StreamWise")
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

# ---- DATA VISUALIZATION TAB ----
with tabs[1]:
    st.subheader("Data-Driven Business Insights")
    try:
        # 1. KPIs
        kpis = {
            "Total Users": len(df),
            "Churn Rate (%)": f"{100*(df['ConsideringCancellation']=='Yes').mean():.1f}%",
            "Avg Weekly Watch Time": f"{df['AvgWeeklyWatchTime'].mean():.1f} hrs",
            "Avg App Rating": f"{df['AppRating'].mean():.2f}"
        }
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Users", kpis["Total Users"])
        k2.metric("Churn Rate (%)", kpis["Churn Rate (%)"])
        k3.metric("Avg Weekly Watch", kpis["Avg Weekly Watch Time"])
        k4.metric("Avg App Rating", kpis["Avg App Rating"])

        # 2. Churn Rate by Billing Cycle
        st.write("#### Churn Rate by Billing Cycle")
        d1 = df.groupby("BillingCycle")["ConsideringCancellation"].apply(lambda x: (x=="Yes").mean()).reset_index()
        fig1 = px.bar(d1, x="BillingCycle", y="ConsideringCancellation", color="BillingCycle", color_discrete_sequence=px.colors.sequential.Reds)
        st.plotly_chart(fig1, use_container_width=True)

        # 3. Price Willingness vs Age
        st.write("#### Price Willingness vs Age")
        fig2 = px.scatter(df, x="Age", y="PriceWillingness", color="ConsideringCancellation", color_discrete_map={"Yes":"#E50914","No":"#444"})
        st.plotly_chart(fig2, use_container_width=True)

        # 4. Engagement vs Churn
        st.write("#### Engagement vs Churn")
        d2 = df.groupby("ConsideringCancellation")["AvgWeeklyWatchTime"].mean().reset_index()
        fig3 = px.bar(d2, x="ConsideringCancellation", y="AvgWeeklyWatchTime", color="ConsideringCancellation", color_discrete_sequence=px.colors.sequential.Reds)
        st.plotly_chart(fig3, use_container_width=True)

        # 5. App Rating Distribution
        st.write("#### App Rating Distribution")
        fig4 = px.histogram(df, x="AppRating", nbins=15, color_discrete_sequence=["#E50914"])
        st.plotly_chart(fig4, use_container_width=True)

        # 6. Subscription Tenure by Plan Type
        st.write("#### Tenure by Plan Type")
        fig5 = px.box(df, x="PlanType", y="SubscriptionTenureMonths", color="PlanType", color_discrete_sequence=px.colors.sequential.Reds)
        st.plotly_chart(fig5, use_container_width=True)

        # 7. User Age Distribution
        st.write("#### Age Distribution")
        fig6 = px.histogram(df, x="Age", nbins=18, color_discrete_sequence=["#E50914"])
        st.plotly_chart(fig6, use_container_width=True)

        # 8. Churn by Location
        st.write("#### Churn by Location")
        d3 = df.groupby("Location")["ConsideringCancellation"].apply(lambda x: (x=="Yes").mean()).reset_index()
        fig7 = px.bar(d3, x="Location", y="ConsideringCancellation", color="Location", color_discrete_sequence=px.colors.sequential.Reds)
        st.plotly_chart(fig7, use_container_width=True)

        # 9. Features Valued Most
        st.write("#### Top Valued Features")
        if "ValuedFeatures" in df.columns:
            from collections import Counter
            features = Counter(", ".join(df["ValuedFeatures"].dropna()).split(", "))
            d_feat = pd.DataFrame(features.items(), columns=["Feature","Count"]).sort_values("Count", ascending=False).head(10)
            fig8 = px.bar(d_feat, x="Feature", y="Count", color="Count", color_continuous_scale=px.colors.sequential.Reds)
            st.plotly_chart(fig8, use_container_width=True)

        # 10. Preferred Devices
        st.write("#### Preferred Viewing Devices")
        if "ViewingDevices" in df.columns:
            devices = Counter(", ".join(df["ViewingDevices"].dropna()).split(", "))
            d_dev = pd.DataFrame(devices.items(), columns=["Device","Count"]).sort_values("Count", ascending=False).head(10)
            fig9 = px.bar(d_dev, x="Device", y="Count", color="Count", color_continuous_scale=px.colors.sequential.Reds)
            st.plotly_chart(fig9, use_container_width=True)

        # 11. Satisfaction vs. Churn
        st.write("#### Satisfaction vs Churn")
        if "Satisfaction" in df.columns:
            d5 = df.groupby("ConsideringCancellation")["Satisfaction"].mean().reset_index()
            fig10 = px.bar(d5, x="ConsideringCancellation", y="Satisfaction", color="ConsideringCancellation", color_discrete_sequence=px.colors.sequential.Reds)
            st.plotly_chart(fig10, use_container_width=True)
    except Exception as e:
        st.warning(f"Some charts could not be loaded: {e}")

# ---- CLASSIFICATION TAB ----
with tabs[2]:
    # ... same as earlier (already given above) ...

    # Paste your classification tab code here for brevity!

# ---- CLUSTERING TAB ----
with tabs[3]:
    # ... same as earlier (already given above) ...

    # Paste your clustering tab code here for brevity!

# ---- ASSOCIATION RULES TAB ----
with tabs[4]:
    st.header("Association Rule Mining: What Feature Combinations Drive Satisfaction or Churn?")
    try:
        if not MLXTEND_INSTALLED:
            st.warning("mlxtend is not installed. Please add 'mlxtend' to requirements.txt.")
        else:
            ar_cols = [c for c in df.columns if df[c].apply(lambda x: isinstance(x,str) and "," in str(x)).any()]
            if len(ar_cols) < 2:
                st.warning("Need at least 2 multi-select columns (e.g. 'ValuedFeatures', 'PreferredGenres', 'MainFrustrations') for association rule mining.")
            else:
                col1 = st.selectbox("First column (e.g. ValuedFeatures)", ar_cols, index=0)
                col2 = st.selectbox("Second column (e.g. MainFrustrations)", ar_cols, index=1)
                min_sup = st.slider("Minimum Support", 0.01, 0.3, 0.05, step=0.01)
                min_conf = st.slider("Minimum Confidence", 0.1, 1.0, 0.5, step=0.05)
                max_rules = st.slider("Show Top N Rules", 5, 30, 10, step=1)
                trans = []
                for i, row in df[[col1, col2]].dropna().iterrows():
                    items = set()
                    for v in str(row[col1]).split(","):
                        v = v.strip()
                        if v: items.add(v)
                    for v in str(row[col2]).split(","):
                        v = v.strip()
                        if v: items.add(v)
                    trans.append(list(items))
                te = TransactionEncoder()
                te_ary = te.fit(trans).transform(trans)
                df_enc = pd.DataFrame(te_ary, columns=te.columns_)
                freq = apriori(df_enc, min_support=min_sup, use_colnames=True)
                rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
                rules = rules.sort_values("confidence", ascending=False).head(max_rules)
                # 1. Frequent Items Bar Chart (frozenset to string)
                st.subheader("Top Frequent Features")
                freq_items = freq.sort_values("support", ascending=False).head(15)
                freq_items_display = freq_items.copy()
                freq_items_display["itemsets"] = freq_items_display["itemsets"].apply(lambda x: ', '.join(list(x)))
                fig = px.bar(freq_items_display, x="itemsets", y="support", color="support", color_continuous_scale=px.colors.sequential.Reds)
                fig.update_layout(
                    xaxis_tickangle=45, 
                    plot_bgcolor='#181818', 
                    paper_bgcolor='#181818', 
                    font=dict(color='#f7f7f7'),
                    title=dict(text="Top 15 Frequent Features", font=dict(color='#f7f7f7'))
                )
                st.plotly_chart(fig, use_container_width=True)
                st.info("These are the most common combinations of features or frustrations. Consider promoting or fixing the top items.")

                # 2. Association Rules Table
                st.subheader("Top Association Rules")
                if len(rules):
                    rules_show = rules[["antecedents","consequents","support","confidence","lift"]].copy()
                    rules_show["antecedents"] = rules_show["antecedents"].apply(lambda x: ', '.join(list(x)))
                    rules_show["consequents"] = rules_show["consequents"].apply(lambda x: ', '.join(list(x)))
                    st.dataframe(rules_show.style.background_gradient(axis=0, cmap="Reds"))
                    st.info("Rules with high confidence and lift indicate strong actionable connections. For example: users who value 'Family Sharing' often also want 'Ad-free'.")

                    # 3. Network Graph (only if networkx is installed)
                    if NETWORKX_INSTALLED:
                        G = nx.DiGraph()
                        for _, row in rules_show.iterrows():
                            G.add_edge(row["antecedents"], row["consequents"], weight=row["confidence"])
                        pos = nx.spring_layout(G, seed=42)
                        edge_x = []
                        edge_y = []
                        for edge in G.edges():
                            x0, y0 = pos[edge[0]]
                            x1, y1 = pos[edge[1]]
                            edge_x += [x0, x1, None]
                            edge_y += [y0, y1, None]
                        edge_trace = go.Scatter(
                            x=edge_x, y=edge_y, line=dict(width=1, color='#E50914'), hoverinfo='none', mode='lines')
                        node_x = []
                        node_y = []
                        text_labels = []
                        for node in G.nodes():
                            x, y = pos[node]
                            node_x.append(x)
                            node_y.append(y)
                            text_labels.append(node)
                        node_trace = go.Scatter(
                            x=node_x, y=node_y, mode='markers+text', text=text_labels, textposition="bottom center",
                            marker=dict(size=20, color="#f7f7f7", line=dict(width=2, color='#E50914')),
                            hoverinfo='text'
                        )
                        fig2 = go.Figure(data=[edge_trace, node_trace],
                                        layout=go.Layout(
                                            title=dict(text="Network of Top Feature Associations", font=dict(color="#f7f7f7")),
                                            showlegend=False,
                                            hovermode='closest',
                                            margin=dict(b=20,l=5,r=5,t=30),
                                            paper_bgcolor="#181818",
                                            plot_bgcolor="#181818"
                                        ))
                        st.plotly_chart(fig2, use_container_width=True)
                        st.info("Strongly connected features are excellent targets for bundles or targeted marketing. For instance, users who select 'Offline Download' also tend to value 'Multi-Device Support'.")
                    else:
                        st.warning("NetworkX is not installed. Please add 'networkx' to requirements.txt for network graphs.")
                else:
                    st.warning("No strong associations found. Adjust thresholds or try other columns.")

    except Exception as e:
        st.error(f"Association Rule Mining failed: {e}")

# ---- REGRESSION TAB ----
with tabs[5]:
    # ... (regression code from above, as already provided) ...
    pass  # Paste the regression tab code as you already have it

