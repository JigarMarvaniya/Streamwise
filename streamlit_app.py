import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.express as px
import plotly.graph_objects as go
import time
from collections import Counter

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

    # --- AUDIO FEATURE HERE ---
    audio_placeholder = st.empty()
    if os.path.exists("streamwise_audio.mp3"):
        with audio_placeholder:
            st.audio("streamwise_audio.mp3", format='audio/mp3', start_time=0)
    else:
        st.info("🎧 Intro audio file not found. Skipping audio...")

    if st.button("▶ StreamWise", key="intro_btn"):
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
    st.header("Churn Prediction (Classification Models)")
    label_col = "ConsideringCancellation"
    try:
        df_model = df.dropna(subset=[label_col]).copy()
        for col in df_model.columns:
            if df_model[col].dtype == 'object' and col != label_col:
                df_model[col] = pd.factorize(df_model[col])[0]
        y = df_model[label_col]
        if y.dtype == 'object':
            y = (y == "Yes").astype(int)
        else:
            y = y.astype(int)
        features = [c for c in df_model.columns if c != label_col and df_model[c].dtype in [np.int64, np.float64]]
        X = df_model[features]
        mask = ~X.isnull().any(axis=1) & ~pd.isnull(y)
        X = X[mask]
        y = y[mask]
        if X.shape[0] < 10:
            st.warning("Too few data points after filtering. Please broaden your filters or upload more data.")
        else:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, stratify=y, random_state=42)
            models = {
                "KNN": KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(random_state=0),
                "Random Forest": RandomForestClassifier(n_estimators=50, random_state=0),
                "GBRT": GradientBoostingClassifier(n_estimators=50, random_state=0)
            }
            metrics, model_objs = [], {}
            confusion_matrices = {}
            roc_curves = []
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                model_objs[name] = model
                metrics.append([
                    name,
                    accuracy_score(y_test, y_pred),
                    precision_score(y_test, y_pred, zero_division=0),
                    recall_score(y_test, y_pred, zero_division=0),
                    f1_score(y_test, y_pred, zero_division=0)
                ])
                confusion_matrices[name] = confusion_matrix(y_test, y_pred)
                # ROC
                if hasattr(model, "predict_proba"):
                    y_score = model.predict_proba(X_test)[:,1]
                else:
                    y_score = model.decision_function(X_test)
                fpr, tpr, _ = roc_curve(y_test, y_score)
                roc_auc = auc(fpr, tpr)
                roc_curves.append((name, fpr, tpr, roc_auc))

            metrics_df = pd.DataFrame(metrics, columns=["Model","Accuracy","Precision","Recall","F1"])
            st.dataframe(metrics_df.set_index("Model").style.background_gradient(axis=0, cmap="Reds"))
            st.info("Random Forest and GBRT deliver the highest accuracy and recall, making them reliable for churn prediction. Focus on features with high model importance for churn reduction.")

            # Confusion Matrices for All Models
            st.subheader("Confusion Matrices")
            cm_cols = st.columns(len(models))
            for i, (name, cm) in enumerate(confusion_matrices.items()):
                with cm_cols[i]:
                    st.write(f"**{name}**")
                    st.write(pd.DataFrame(cm, columns=["Pred No","Pred Yes"], index=["Actual No","Actual Yes"]))

            # ROC Curve for All Models
            st.subheader("ROC Curves: Model Comparison")
            fig, ax = plt.subplots()
            for name, fpr, tpr, roc_auc in roc_curves:
                ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")
            ax.plot([0,1],[0,1],'k--',alpha=0.4)
            ax.set_xlabel("False Positive Rate", color="#f7f7f7")
            ax.set_ylabel("True Positive Rate", color="#f7f7f7")
            ax.legend()
            fig.patch.set_facecolor('#181818')
            ax.set_facecolor('#181818')
            ax.tick_params(colors='#f7f7f7')
            st.pyplot(fig)
            st.info("All models perform above random; Random Forest/GBRT lead in separability.")

            # Feature Importance for Each Model (if available)
            st.subheader("Feature Importances")
            for name, model in model_objs.items():
                if hasattr(model, "feature_importances_"):
                    fi = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
                    fig2, ax2 = plt.subplots()
                    fi.head(7).plot(kind="barh", color="#E50914", ax=ax2)
                    ax2.invert_yaxis()
                    ax2.set_title(f"{name} Feature Importances", color="#f7f7f7")
                    fig2.patch.set_facecolor('#181818')
                    ax2.set_facecolor('#181818')
                    ax2.tick_params(colors='#f7f7f7')
                    st.pyplot(fig2)

            # Upload for prediction
            st.subheader("Upload new user data for churn prediction")
            pred_file = st.file_uploader("Upload CSV (no churn column)", key="pred_csv")
            if pred_file:
                new_df = pd.read_csv(pred_file)
                for c in new_df.columns:
                    if new_df[c].dtype == "object":
                        new_df[c] = pd.factorize(new_df[c])[0]
                new_X = new_df[features]
                for col in features:
                    if col not in new_X.columns:
                        new_X[col] = 0
                new_X = new_X[features].fillna(0)
                new_X_scaled = scaler.transform(new_X)
                new_preds = model_objs["Random Forest"].predict(new_X_scaled)
                new_df["PredictedChurn"] = ["Yes" if x else "No" for x in new_preds]
                st.write(new_df)
                st.download_button("Download Predictions", new_df.to_csv(index=False), file_name="churn_predictions.csv")

    except Exception as e:
        st.error(f"Classification failed: {e}")

# ---- CLUSTERING TAB ----
with tabs[3]:
    st.header("Customer Segmentation & Persona Analysis")
    try:
        cluster_feats = ['Age', 'SubscriptionTenureMonths', 'NumOTTSu', 'AvgWeeklyWatchTime', 'AppRating', 'PriceWillingness']
        cluster_feats = [c for c in cluster_feats if c in df.columns]
        if len(cluster_feats) < 3:
            st.warning("Not enough numeric features for clustering. Check your data columns!")
        else:
            Xc = df[cluster_feats].fillna(df[cluster_feats].mean())
            inertia, silhouettes = [], []
            range_clusters = list(range(2, 11))
            for k in range_clusters:
                km = KMeans(n_clusters=k, random_state=1).fit(Xc)
                inertia.append(km.inertia_)
                sil = silhouette_score(Xc, km.labels_)
                silhouettes.append(sil)
            st.subheader("Elbow Chart (Inertia/WCSS)")
            fig = px.line(x=range_clusters, y=inertia, markers=True, labels={"x":"No. of Clusters","y":"WCSS/Inertia"})
            fig.update_traces(line_color="#E50914")
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("Silhouette Score Chart")
            fig2 = px.line(x=range_clusters, y=silhouettes, markers=True, labels={"x":"No. of Clusters","y":"Silhouette Score"})
            fig2.update_traces(line_color="#E50914")
            st.plotly_chart(fig2, use_container_width=True)
            best_k = range_clusters[np.argmax(silhouettes)]
            st.success(f"Optimal number of clusters (by silhouette): {best_k}")

            n_clusters = st.slider("Number of Personas (clusters)", 2, 10, best_k)
            km = KMeans(n_clusters=n_clusters, random_state=1).fit(Xc)
            dfp = df.copy()
            dfp["Cluster"] = km.labels_
            persona_profiles = []
            for p in range(n_clusters):
                seg = dfp[dfp["Cluster"]==p]
                profile = {
                    "Persona": f"Persona {p+1}",
                    "Avg Age": seg["Age"].mean(),
                    "Avg WatchTime": seg["AvgWeeklyWatchTime"].mean(),
                    "Avg AppRating": seg["AppRating"].mean(),
                    "Avg Spend": seg["PriceWillingness"].mean(),
                    "Churn Rate": (seg["ConsideringCancellation"]=="Yes").mean(),
                    "Persona Size": len(seg)
                }
                persona_profiles.append(profile)
            st.write(pd.DataFrame(persona_profiles))
            st.download_button("Download Clustered Data", dfp.to_csv(index=False), file_name="personas_streamwise.csv")

            # 3D Visualization
            if all(c in cluster_feats for c in ["Age","AvgWeeklyWatchTime","PriceWillingness"]):
                st.subheader("3D Cluster Visualization")
                fig3d = px.scatter_3d(dfp, x="Age", y="AvgWeeklyWatchTime", z="PriceWillingness",
                        color="Cluster", color_continuous_scale=px.colors.sequential.Reds,
                        symbol="Cluster", labels={"Cluster": "Persona Cluster"}, opacity=0.8)
                fig3d.update_traces(marker=dict(size=5))
                st.plotly_chart(fig3d, use_container_width=True)
            else:
                st.info("3D plot needs Age, AvgWeeklyWatchTime, and PriceWillingness in your data.")
    except Exception as e:
        st.error(f"Clustering failed: {e}")

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
