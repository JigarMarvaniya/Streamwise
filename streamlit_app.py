import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor

try:
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
except ImportError:
    st.warning("mlxtend not found! Please add 'mlxtend' to your requirements.txt for association rules.")
import plotly.express as px

# -------------- NETFLIX DARK THEME --------------
st.set_page_config(page_title="StreamWise | OTT Analytics Platform", layout="wide")
st.markdown("""
    <style>
        body, .stApp {background-color: #141414 !important;}
        .main {background-color: #141414 !important;}
        .st-bb, .st-bb div {background: #141414 !important;}
        h1, h2, h3, h4, h5, h6 {color: #f7f7f7 !important;}
        .st-eb {color: #E50914 !important;}
        .st-c7 {background-color: #141414 !important;}
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
        .block-container {padding-top: 2rem;}
    </style>
    """, unsafe_allow_html=True)

# ---------------- LOGO HEADER ----------------
logo_path = "streamwise_logo.png"
if os.path.exists(logo_path):
    st.markdown(
        f"<div style='display:flex;align-items:center;'><img src='data:image/png;base64,{open(logo_path, 'rb').read().hex()}' style='height:56px;margin-right:20px;'/><span style='font-size:2.1rem;font-weight:800;color:#E50914;'>StreamWise</span></div>",
        unsafe_allow_html=True)
else:
    st.markdown(
        "<div style='font-size:2.1rem;font-weight:800;color:#E50914;'>StreamWise</div>",
        unsafe_allow_html=True)
st.markdown("<hr style='border-top: 2.5px solid #E50914;'>", unsafe_allow_html=True)

# ---------------- DATA LOADING & FILTERS ----------------
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

tabs = st.tabs([
    "📊 Data Visualization", 
    "🧠 Classification", 
    "👥 Clustering & Persona", 
    "🔗 Association Rules", 
    "📈 Regression"
])

# ---------------- DATA VISUALIZATION ----------------
with tabs[0]:
    st.subheader("Data-Driven Business Insights")
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

    try:
        churn = df.groupby("BillingCycle")["ConsideringCancellation"].apply(lambda x: (x=="Yes").mean())
        fig = px.bar(churn, title="Churn Rate by Billing Cycle", labels={"value":"Churn Rate","BillingCycle":"Billing Cycle"}, color_discrete_sequence=["#E50914"])
        st.plotly_chart(fig, use_container_width=True)
        ar = df.groupby("PlanType")["AppRating"].mean()
        fig = px.bar(ar, title="Average App Rating by Plan", color_discrete_sequence=["#E50914"])
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Some columns missing for these charts: {e}")

    st.markdown("**Download filtered dataset:**")
    st.download_button("Download Data", df.to_csv(index=False), file_name="filtered_streamwise_data.csv")

# ---------------- CLASSIFICATION ----------------
with tabs[1]:
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
            metrics_df = pd.DataFrame(metrics, columns=["Model","Accuracy","Precision","Recall","F1"])
            st.dataframe(metrics_df.set_index("Model").style.background_gradient(axis=0, cmap="Reds"))
            cm_model = st.selectbox("Show Confusion Matrix for", list(models.keys()))
            cm = confusion_matrix(y_test, model_objs[cm_model].predict(X_test))
            st.write(pd.DataFrame(cm, columns=["Predicted No","Predicted Yes"], index=["Actual No","Actual Yes"]))
    except Exception as e:
        st.error(f"Classification failed: {e}")

# ---------------- CLUSTERING & PERSONA ----------------
with tabs[2]:
    st.header("Customer Segmentation & Persona Analysis")
    try:
        cluster_feats = ['Age', 'SubscriptionTenureMonths', 'NumOTTSu', 'AvgWeeklyWatchTime', 'AppRating', 'PriceWillingness']
        cluster_feats = [c for c in cluster_feats if c in df.columns]
        if len(cluster_feats) < 3:
            st.warning("Not enough numeric features for clustering. Check your data columns!")
        else:
            Xc = df[cluster_feats].fillna(df[cluster_feats].mean())
            n_clusters = st.slider("Number of Personas (clusters)", 2, 10, 4)
            inertia = []
            for k in range(2, 11):
                km = KMeans(n_clusters=k, random_state=1).fit(Xc)
                inertia.append(km.inertia_)
            fig = px.line(x=list(range(2,11)), y=inertia, markers=True, labels={"x":"Clusters","y":"Inertia"})
            fig.update_traces(line_color="#E50914")
            st.plotly_chart(fig, use_container_width=True)
            km = KMeans(n_clusters=n_clusters, random_state=1).fit(Xc)
            dfp = df.copy()
            dfp["Persona"] = km.labels_
            persona_profiles = []
            for p in range(n_clusters):
                seg = dfp[dfp["Persona"]==p]
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
            st.download_button("Download Personas", dfp.to_csv(index=False), file_name="personas_streamwise.csv")
    except Exception as e:
        st.error(f"Clustering failed: {e}")

# ---------------- ASSOCIATION RULES ----------------
with tabs[3]:
    st.header("Association Rule Mining")
    try:
        ar_cols = [c for c in df.columns if df[c].apply(lambda x: isinstance(x,str) and "," in str(x)).any()]
        if len(ar_cols) < 2:
            st.warning("Need at least 2 multi-select columns for association rule mining.")
        else:
            col1 = st.selectbox("First column", ar_cols, index=0)
            col2 = st.selectbox("Second column", ar_cols, index=1)
            min_sup = st.slider("Minimum Support", 0.01, 0.3, 0.05, step=0.01)
            min_conf = st.slider("Minimum Confidence", 0.1, 1.0, 0.5, step=0.05)
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
            rules = rules.sort_values("confidence", ascending=False).head(10)
            if len(rules):
                st.dataframe(rules[["antecedents","consequents","support","confidence","lift"]])
            else:
                st.warning("No strong associations found. Adjust thresholds or try other columns.")
    except Exception as e:
        st.error(f"Association Rule Mining failed: {e}")

# ---------------- REGRESSION ----------------
with tabs[4]:
    st.header("Regression Insights (Spend, Engagement, Tenure)")
    try:
        target_col = st.selectbox("Regression Target", ["PriceWillingness","SubscriptionTenureMonths","AvgWeeklyWatchTime"])
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
            reg_results = []
            for name, model in regs.items():
                model.fit(Xr, yr)
                score = model.score(Xr, yr)
                reg_results.append([name, score])
            st.dataframe(pd.DataFrame(reg_results, columns=["Model","R^2"]).set_index("Model").style.background_gradient(axis=0, cmap="Reds"))
    except Exception as e:
        st.error(f"Regression failed: {e}")
