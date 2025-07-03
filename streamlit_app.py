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

# --- THEME ---
st.set_page_config(page_title="StreamWise | OTT Analytics Platform", layout="wide")
st.markdown("""
    <style>
        body, .stApp {background-color: #141414 !important;}
        .main {background-color: #141414 !important;}
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
        .block-container {padding-top: 2rem;}
        .metric-label, .stMetric {color: #fff !important;}
    </style>
    """, unsafe_allow_html=True)

# --- INTRO PAGE ---
if "show_intro" not in st.session_state:
    st.session_state.show_intro = True

def run_intro():
    st.markdown("""
    <style>
    .centered-intro {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 85vh;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="centered-intro">', unsafe_allow_html=True)
    if os.path.exists("streamwise_logo.png"):
        st.image("streamwise_logo.png", width=220, output_format='auto', use_column_width=False)
    st.markdown(
        "<div style='font-size:2.5rem;font-weight:900;letter-spacing:2px;margin-top:20px;color:#f7f7f7;'>Welcome to <span style='color:#E50914;'>StreamWise</span></div>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div style='font-size:1.2rem;max-width:680px;margin:20px auto;color:#bbb;text-align:center;'>"
        "Your all-in-one, MBA-level OTT analytics and churn prediction dashboard.<br>"
        "<b>Analyze. Segment. Act.</b> – in true Netflix style.<br>"
        "Click the <span style='color:#E50914;font-weight:bold;'>StreamWise</span> button to get started!"
        "</div>", unsafe_allow_html=True
    )
    audio_placeholder = st.empty()
    start_btn = st.button("▶ StreamWise", key="start_streamwise", help="Start exploring the dashboard!", use_container_width=False)
    if start_btn:
        if os.path.exists("streamwise_audio.mp3"):
            audio_placeholder.audio("streamwise_audio.mp3", format='audio/mp3', start_time=0)
            time.sleep(3)
        st.session_state.show_intro = False
        st.experimental_rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='margin-top:60px;color:#aaa;font-size:0.97rem;text-align:center;'>"
        "Inspired by Netflix • Powered by Streamlit • Crafted for OTT MBA decision-makers"
        "</div>",
        unsafe_allow_html=True
    )
    st.stop()

if st.session_state.show_intro:
    run_intro()

with st.sidebar:
    if st.button("🏠 Home / Introduction"):
        st.session_state.show_intro = True
        st.experimental_rerun()

# --- LOGO HEADER ---
if os.path.exists("streamwise_logo.png"):
    st.markdown(
        "<div style='display:flex;align-items:center;justify-content:center;'>"
        "<img src='streamwise_logo.png' style='height:52px;margin-right:20px;'>"
        "<span style='font-size:2.1rem;font-weight:800;color:#E50914;'>StreamWise</span>"
        "</div>", unsafe_allow_html=True
    )
else:
    st.markdown(
        "<div style='font-size:2.1rem;font-weight:800;color:#E50914;text-align:center;'>StreamWise</div>",
        unsafe_allow_html=True
    )
st.markdown("<hr style='border-top: 2.5px solid #E50914;'>", unsafe_allow_html=True)

# --- DATA LOADING & FILTERS ---
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

# --- ABOUT TAB ---
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

# --- DATA VISUALIZATION TAB (as above, 10+ charts) ---
with tabs[1]:
    # ... (unchanged, use code from my previous long code block for this tab) ...
    # You can copy/paste the "with tabs[1]" code from above or from your current code

    # Omitted here for brevity since you have this already.

# --- CLASSIFICATION TAB (as above) ---
with tabs[2]:
    # ... (unchanged, use code from my previous long code block for this tab) ...

# --- CLUSTERING TAB (as above) ---
with tabs[3]:
    # ... (unchanged, use code from my previous long code block for this tab) ...

# --- ASSOCIATION RULES TAB (with frozenset fix) ---
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
                # Transaction encoding
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
                # 1. Frequent Items Bar Chart (fix: frozenset to string)
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
                    title="Top 15 Frequent Features"
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
                                            title="Network of Top Feature Associations",
                                            titlefont_size=18,
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

# --- REGRESSION TAB (with feature importance) ---
with tabs[5]:
    st.header("Regression Insights (Spend, Engagement, Tenure)")
    try:
        possible_targets = ["PriceWillingness","SubscriptionTenureMonths","AvgWeeklyWatchTime"]
        target_col = st.selectbox("Regression Target", possible_targets)
        reg_feats = [c for c in df.columns if c not in [target_col,"ConsideringCancellation","CancelReason"] and df[c].dtype in [np.int64,np.float64]]
        if len(reg_feats) < 2:
            st.warning("Not enough numeric features for regression.")
        else:
            Xr = df[reg_feats].fillna(df[reg_feats].mean())
            yr = df[target_col].fillna(df[target_col].mean())
            regs = {
                "Linear": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "Decision Tree": DecisionTreeRegressor(max_depth=4)
            }
            reg_results, metrics_table, fi_df_list = [], [], []
            preds = {}
            st.subheader("Model Performance")
            for name, model in regs.items():
                model.fit(Xr, yr)
                pred = model.predict(Xr)
                preds[name] = pred
                r2 = model.score(Xr, yr)
                rmse = np.sqrt(mean_squared_error(yr, pred))
                mae = mean_absolute_error(yr, pred)
                reg_results.append([name, r2, rmse, mae])

                # Feature importance for models
                if name in ["Ridge", "Lasso", "Linear"]:
                    fi = pd.Series(np.abs(model.coef_), index=reg_feats).sort_values(ascending=False)
                    fi_df = pd.DataFrame({'Feature': fi.index, 'Importance': fi.values, 'Model': name})
                    fi_df_list.append(fi_df)
                elif name == "Decision Tree":
                    fi = pd.Series(model.feature_importances_, index=reg_feats).sort_values(ascending=False)
                    fi_df = pd.DataFrame({'Feature': fi.index, 'Importance': fi.values, 'Model': name})
                    fi_df_list.append(fi_df)

            metrics_df = pd.DataFrame(reg_results, columns=["Model","R^2","RMSE","MAE"]).set_index("Model")
            st.dataframe(metrics_df.style.background_gradient(axis=0, cmap="Reds"))
            st.info("Feature selection and importance can guide pricing, tenure and engagement levers for your OTT business.")

            # Visualize Actual vs Predicted for all models
            st.subheader("Actual vs Predicted (All Models)")
            fig = go.Figure()
            for name, pred in preds.items():
                fig.add_trace(go.Scatter(y=yr, mode="lines", name="Actual", line=dict(color="#f7f7f7")))
                fig.add_trace(go.Scatter(y=pred, mode="lines", name=name+" Predicted", line=dict(width=2)))
            fig.update_layout(title="Actual vs Predicted", legend=dict(font=dict(color="#f7f7f7")), plot_bgcolor='#181818', paper_bgcolor='#181818', font=dict(color='#f7f7f7'))
            st.plotly_chart(fig, use_container_width=True)
            # Residual Plots
            st.subheader("Residuals (All Models)")
            for name, pred in preds.items():
                fig2, ax2 = plt.subplots()
                ax2.scatter(pred, yr-pred, alpha=0.5, c="#E50914")
                ax2.axhline(0, linestyle="--", color="#888")
                ax2.set_title(f"{name} Residual Plot", color="#f7f7f7")
                ax2.set_xlabel("Predicted", color="#f7f7f7")
                ax2.set_ylabel("Residuals", color="#f7f7f7")
                fig2.patch.set_facecolor('#181818')
                ax2.set_facecolor('#181818')
                ax2.tick_params(colors='#f7f7f7')
                st.pyplot(fig2)

            # Most Useful Feature Chart
            st.subheader("Most Useful Features for Regression Target")
            fi_all = pd.concat(fi_df_list)
            fi_agg = fi_all.groupby('Feature')['Importance'].mean().sort_values(ascending=False).head(8)
            fig3, ax3 = plt.subplots()
            fi_agg.plot(kind="barh", color="#E50914", ax=ax3)
            ax3.invert_yaxis()
            ax3.set_title("Top Features Driving " + target_col, color="#f7f7f7")
            fig3.patch.set_facecolor('#181818')
            ax3.set_facecolor('#181818')
            ax3.tick_params(colors='#f7f7f7')
            st.pyplot(fig3)
            st.info(f"These features have the greatest impact on predicting {target_col}. Focus on managing them for better business outcomes.")

    except Exception as e:
        st.error(f"Regression failed: {e}")
