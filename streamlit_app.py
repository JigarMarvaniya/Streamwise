import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import plotly.express as px
import plotly.graph_objects as go

# ---- CUSTOM STYLING: Netflix-inspired on white base ----
st.set_page_config(page_title="StreamWise | OTT Analytics Platform", layout="wide")
st.markdown("""
    <style>
        body, .stApp {background-color: #fff !important;}
        .main {background-color: #fff !important;}
        .st-bb, .st-bb div {background: #fff !important;}
        h1, h2, h3, h4, h5, h6 {color: #121212 !important;}
        .st-eb {color: #E50914 !important;}
        .st-c7 {background-color: #fff !important;}
        .stTabs [data-baseweb="tab"] {
            background-color: #fff !important;
            color: #121212 !important;
            font-weight: bold;
            border-bottom: 4px solid #E50914 !important;
            font-size: 1.1rem;
        }
        .stTabs [aria-selected="false"] {
            border-bottom: 4px solid #eee !important;
            color: #666 !important;
        }
        .css-1aumxhk, .css-18e3th9, .stButton>button {
            background-color: #E50914 !important;
            color: white !important;
            font-weight: bold !important;
            border-radius: 7px !important;
            border: none;
            font-size: 1.15rem !important;
            transition: box-shadow .2s;
        }
        .stButton>button:hover {
            background-color: #b0060c !important;
            box-shadow: 0 2px 16px #e5091480;
        }
        .metric-label {color: #121212 !important;}
    </style>
    """, unsafe_allow_html=True)

# ---- INTRO PAGE LOGIC ----
if "show_intro" not in st.session_state:
    st.session_state.show_intro = True

def run_intro():
    st.markdown("<div style='height:7vh'></div>", unsafe_allow_html=True)
    st.image("streamwise_logo.png", width=220)
    st.markdown(
        "<div style='font-size:2.5rem;font-weight:900;letter-spacing:2px;margin-top:20px;color:#121212;'>Welcome to <span style='color:#E50914;'>StreamWise</span></div>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div style='font-size:1.2rem;max-width:680px;margin:20px auto;color:#444;text-align:center;'>"
        "Your all-in-one, MBA-level OTT analytics and churn prediction dashboard.<br>"
        "<b>Analyze. Segment. Act.</b> – in true Netflix style.<br>"
        "Click the <span style='color:#E50914;font-weight:bold;'>StreamWise</span> button to get started!"
        "</div>", unsafe_allow_html=True
    )
    st.markdown("<div style='height:1vh'></div>", unsafe_allow_html=True)
    c = st.columns([3,2,3])[1]
    with c:
        start_btn = st.button(
            "▶ StreamWise", 
            key="start_streamwise",
            help="Start exploring the dashboard!",
            use_container_width=True
        )
        if start_btn:
            st.session_state.show_intro = False

    # Footer
    st.markdown(
        "<div style='margin-top:80px;color:#aaa;font-size:0.97rem;text-align:center;'>"
        "Inspired by Netflix • Powered by Streamlit • Crafted for OTT MBA decision-makers"
        "</div>",
        unsafe_allow_html=True
    )
    st.stop()

# ---- SHOW INTRO PAGE OR DASHBOARD ----
if st.session_state.show_intro:
    run_intro()

# (Optional: Add a sidebar button to return to intro page)
with st.sidebar:
    if st.button("🏠 Home / Introduction"):
        st.session_state.show_intro = True
        st.experimental_rerun()

# ---- HEADER WITH LOGO ----
st.markdown(
    "<div style='display:flex;align-items:center;'>"
    "<img src='streamwise_logo.png' style='height:52px;margin-right:20px;'>"
    "<span style='font-size:2.1rem;font-weight:800;color:#E50914;'>StreamWise</span>"
    "</div>",
    unsafe_allow_html=True
)
st.markdown("<hr style='border-top: 2.5px solid #E50914;'>", unsafe_allow_html=True)

# ---- DATA LOADING & FILTERS ----
uploaded_file = st.sidebar.file_uploader("Upload your StreamWise Survey CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("streamwise_survey_synthetic.csv")

with st.sidebar:
    st.header("Filter Data")
    filter_cols = ['Gender', 'Income', 'Location', 'BillingCycle', 'PlanType']
    for col in filter_cols:
        if col in df.columns:
            opts = df[col].dropna().unique().tolist()
            chosen = st.multiselect(col, opts, default=opts, key=col)
            df = df[df[col].isin(chosen)]

# ---- NAVIGATION ----
tabs = st.tabs([
    "📊 Data Visualization", 
    "🧠 Classification", 
    "👥 Clustering & Persona", 
    "🔗 Association Rules", 
    "📈 Regression"
])

# ---- DATA VISUALIZATION TAB ----
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

    st.write("**Churn Rate by Billing Cycle**")
    churn = df.groupby("BillingCycle")["ConsideringCancellation"].apply(lambda x: (x=="Yes").mean())
    fig = px.bar(churn, title="Churn Rate by Billing Cycle", labels={"value":"Churn Rate","BillingCycle":"Billing Cycle"}, color_discrete_sequence=["#E50914"])
    st.plotly_chart(fig, use_container_width=True)
    st.info("Monthly billing cycles have higher churn. Push annual/quarterly plans with perks to improve retention.")

    st.write("**Average App Rating by Plan Type**")
    ar = df.groupby("PlanType")["AppRating"].mean()
    fig = px.bar(ar, title="Average App Rating by Plan", color_discrete_sequence=["#E50914"])
    st.plotly_chart(fig, use_container_width=True)
    st.info("Premium/Family plan users rate the app highest, showing perceived value.")

    st.write("**Weekly Watch Time by Top Genres**")
    for_genre = df["PreferredGenres"].dropna().str.split(",", expand=True).stack().value_counts().head(5).index
    genre_watch = []
    for g in for_genre:
        genre_watch.append((g, df[df["PreferredGenres"].fillna("").str.contains(g)]["AvgWeeklyWatchTime"].mean()))
    genre_df = pd.DataFrame(genre_watch, columns=["Genre","AvgWeeklyWatchTime"])
    fig = px.bar(genre_df, x="Genre", y="AvgWeeklyWatchTime", title="Avg Weekly Watch Time by Genre", color="Genre", color_discrete_sequence=px.colors.sequential.Reds)
    st.plotly_chart(fig, use_container_width=True)
    st.info("Action, Drama, Comedy genres keep users engaged. Curate more in these genres.")

    st.write("**Price Sensitivity vs Churn**")
    churn_price = df.groupby("PriceSensitivity")["ConsideringCancellation"].apply(lambda x: (x=="Yes").mean())
    fig = px.line(churn_price, markers=True, title="Price Sensitivity and Churn", labels={"value":"Churn Rate","PriceSensitivity":"Price Sensitivity"}, color_discrete_sequence=["#E50914"])
    st.plotly_chart(fig, use_container_width=True)
    st.info("Highly price-sensitive users churn more. Use tailored offers to keep them loyal.")

    st.write("**Most Valued Features**")
    feat = df["ValuedFeatures"].dropna().str.split(",", expand=True).stack().value_counts().head(7)
    fig = px.pie(values=feat.values, names=feat.index, color_discrete_sequence=px.colors.sequential.Reds)
    st.plotly_chart(fig, use_container_width=True)
    st.info("Offline download, recommendations, and family sharing are most valued features.")

    st.write("**NPS Distribution**")
    fig, ax = plt.subplots()
    sns.histplot(df["RecommendNPS"].dropna(), bins=10, color="#E50914")
    ax.set_title("NPS Distribution"); ax.set_xlabel("NPS Score")
    st.pyplot(fig)
    st.info("NPS is moderate-to-high; focus on moving passives to promoters.")

    st.write("**Watch Time Distribution**")
    fig, ax = plt.subplots()
    sns.histplot(df["AvgWeeklyWatchTime"], bins=30, color="#E50914")
    ax.set_title("Watch Time Distribution")
    st.pyplot(fig)
    st.info("Most users watch 7–18 hours/week; outliers may be binge-watchers or account sharers.")

    st.write("**App Rating by Age Group**")
    bins = pd.cut(df["Age"], bins=[16,25,35,50,65], labels=["16-25","26-35","36-50","51-65"])
    age_app = df.groupby(bins)["AppRating"].mean()
    fig = px.bar(age_app, labels={"value":"Avg App Rating","Age":"Age Group"}, title="App Rating by Age Group", color_discrete_sequence=["#E50914"])
    st.plotly_chart(fig, use_container_width=True)
    st.info("Older users rate the app higher. Improve onboarding for younger cohorts.")

    st.write("**Top User Frustrations**")
    fr = df["MainFrustrations"].dropna().str.split(",", expand=True).stack().value_counts().head(7)
    fig = px.bar(fr, title="Top User Frustrations", color_discrete_sequence=["#E50914"])
    st.plotly_chart(fig, use_container_width=True)
    st.info("Buffering and price increases are most common complaints.")

    st.write("**Impact of Discounts on Retention**")
    disc = df[df["ReceivedDiscount"]=="Yes"]["DiscountImpact"].value_counts(normalize=True)
    st.write(disc)
    st.info("Majority retained due to discounts. Use strategic offers to retain price-sensitive segments.")

    st.markdown("**Download filtered dataset:**")
    st.download_button("Download Data", df.to_csv(index=False), file_name="filtered_streamwise_data.csv")

# ---- CLASSIFICATION TAB (uses corrected code from earlier message) ----
with tabs[1]:
    st.header("Churn Prediction (Classification Models)")
    st.write("This section predicts churn using multiple models and highlights key predictors for action.")

    label_col = "ConsideringCancellation"
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
        st.error("Too few data points after filtering. Please broaden your filters or upload more data.")
        st.stop()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, stratify=y, random_state=42
    )
    if X_train.shape[0] == 0 or y_train.shape[0] == 0:
        st.error("No training data available after filtering! Please adjust your sidebar filters.")
        st.stop()
    if np.any(pd.isnull(X_train)) or np.any(pd.isnull(y_train)):
        st.error("Missing values detected in training data. Please clean or impute your data.")
        st.stop()
    models = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=0),
        "Random Forest": RandomForestClassifier(n_estimators=50, random_state=0),
        "GBRT": GradientBoostingClassifier(n_estimators=50, random_state=0)
    }
    metrics = []
    model_objs = {}
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
    st.info(
        "Random Forest and GBRT deliver the highest accuracy and recall, making them reliable for churn prediction. "
        "Focus on features with high model importance for churn reduction."
    )
    cm_model = st.selectbox("Show Confusion Matrix for", list(models.keys()))
    cm = confusion_matrix(y_test, model_objs[cm_model].predict(X_test))
    st.write(pd.DataFrame(cm, columns=["Predicted No","Predicted Yes"], index=["Actual No","Actual Yes"]))
    st.subheader("ROC Curve: Model Comparison")
    fig, ax = plt.subplots()
    for name, model in model_objs.items():
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:,1]
        else:
            y_score = model.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")
    ax.plot([0,1],[0,1],'k--',alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)
    st.info("All models perform significantly above random; ROC curves show Random Forest/GBRT lead in separability.")
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

# ---- CLUSTERING, ASSOCIATION RULES, REGRESSION: use as previously provided ----
# (For brevity, keep your previous code blocks for these tabs here)
# Let me know if you want these rewritten as well for new theme/UI or improved explanations!

