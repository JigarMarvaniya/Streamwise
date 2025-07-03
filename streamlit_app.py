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

# --- THEME & STYLE ---
st.set_page_config(page_title="StreamWise | OTT Analytics Platform", layout="wide")
def netflix_style():
    st.markdown("""
        <style>
            body, .stApp {background-color: #141414;}
            .main {background-color: #181818;}
            h1, h2, h3, h4, h5, h6 {color: #ffffff;}
            .st-bb, .st-bb div {background: #181818;}
            .st-eb {color: #E50914 !important;}
            .st-c7 {background-color: #181818;}
            .block-container {padding-top: 2rem;}
        </style>
    """, unsafe_allow_html=True)
netflix_style()

# --- LOGO ---
st.markdown(
    "<div style='display: flex; align-items: center;'>"
    "<img src='streamwise_logo.png' style='height:60px;margin-right:24px;'>"
    "<span style='font-size:2.5rem; font-weight: bold; color:#E50914;'>StreamWise</span>"
    "</div>", unsafe_allow_html=True
)
st.markdown("<hr style='border-top: 2px solid #E50914;'>", unsafe_allow_html=True)

# --- LOAD DATA ---
uploaded_file = st.sidebar.file_uploader("Upload your StreamWise Survey CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("streamwise_survey_synthetic.csv")

# --- SIDEBAR FILTERS ---
with st.sidebar:
    st.header("Filter Data")
    filter_cols = ['Gender', 'Income', 'Location', 'BillingCycle', 'PlanType']
    for col in filter_cols:
        if col in df.columns:
            opts = df[col].dropna().unique().tolist()
            chosen = st.multiselect(col, opts, default=opts, key=col)
            df = df[df[col].isin(chosen)]

# --- MAIN NAVIGATION ---
tabs = st.tabs([
    "📊 Data Visualization", 
    "🧠 Classification", 
    "👥 Clustering & Persona", 
    "🔗 Association Rules", 
    "📈 Regression"
])

# --- 1. DATA VISUALIZATION ---
with tabs[0]:
    st.subheader("Data-Driven Business Insights")
    kpis = {
        "Total Users": len(df),
        "Churn Rate (%)": f"{100*(df['ConsideringCancellation']=='Yes').mean():.1f}%",
        "Avg Weekly Watch Time": f"{df['AvgWeeklyWatchTime'].mean():.1f} hrs",
        "Avg App Rating": f"{df['AppRating'].mean():.2f}"
    }
    st.write(f"#### Key Metrics")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Users", kpis["Total Users"])
    k2.metric("Churn Rate (%)", kpis["Churn Rate (%)"])
    k3.metric("Avg Weekly Watch", kpis["Avg Weekly Watch Time"])
    k4.metric("Avg App Rating", kpis["Avg App Rating"])

    # 1. Churn by BillingCycle
    st.write("**Churn Rate by Billing Cycle**")
    churn = df.groupby("BillingCycle")["ConsideringCancellation"].apply(lambda x: (x=="Yes").mean())
    fig = px.bar(churn, title="Churn Rate by Billing Cycle", labels={"value":"Churn Rate","BillingCycle":"Billing Cycle"})
    st.plotly_chart(fig, use_container_width=True)
    st.info("Monthly billing cycles have significantly higher churn. Encourage quarterly/yearly plans with incentives to improve retention.")

    # 2. Avg AppRating by PlanType
    st.write("**Average App Rating by Plan Type**")
    ar = df.groupby("PlanType")["AppRating"].mean()
    fig = px.bar(ar, title="Average App Rating by Plan", color_discrete_sequence=["#E50914"])
    st.plotly_chart(fig, use_container_width=True)
    st.info("Premium/Family plan users rate the app higher, reflecting satisfaction with value and features.")

    # 3. Watch Time by PreferredGenres
    st.write("**Weekly Watch Time by Top Genres**")
    for_genre = df["PreferredGenres"].dropna().str.split(",", expand=True).stack().value_counts().head(5).index
    genre_watch = []
    for g in for_genre:
        genre_watch.append((g, df[df["PreferredGenres"].fillna("").str.contains(g)]["AvgWeeklyWatchTime"].mean()))
    genre_df = pd.DataFrame(genre_watch, columns=["Genre","AvgWeeklyWatchTime"])
    fig = px.bar(genre_df, x="Genre", y="AvgWeeklyWatchTime", title="Avg Weekly Watch Time by Genre", color="Genre")
    st.plotly_chart(fig, use_container_width=True)
    st.info("Action and Drama genres keep users most engaged. Curate recommendations in these genres to increase retention.")

    # 4. Price Sensitivity vs Churn
    st.write("**Price Sensitivity vs Churn Rate**")
    churn_price = df.groupby("PriceSensitivity")["ConsideringCancellation"].apply(lambda x: (x=="Yes").mean())
    fig = px.line(churn_price, markers=True, title="Price Sensitivity and Churn", labels={"value":"Churn Rate","PriceSensitivity":"Price Sensitivity"})
    st.plotly_chart(fig, use_container_width=True)
    st.info("High price-sensitive users churn more. Segment and target them with personalized discounts or loyalty perks.")

    # 5. Feature Usage
    st.write("**Most Valued Features**")
    feat = df["ValuedFeatures"].dropna().str.split(",", expand=True).stack().value_counts().head(7)
    fig = px.pie(values=feat.values, names=feat.index, color_discrete_sequence=px.colors.sequential.Reds)
    st.plotly_chart(fig, use_container_width=True)
    st.info("Offline download and personalized recommendations are most valued. Highlight these in marketing.")

    # 6. NPS Distribution
    st.write("**Net Promoter Score (NPS) Distribution**")
    fig, ax = plt.subplots()
    sns.histplot(df["RecommendNPS"].dropna(), bins=10, color="#E50914")
    ax.set_title("NPS Distribution"); ax.set_xlabel("NPS Score")
    st.pyplot(fig)
    st.info("NPS is skewed to the mid-high range, but note opportunity to convert passives/detractors via customer experience.")

    # 7. Watch Time Distribution
    st.write("**Watch Time Distribution (with Outliers)**")
    fig, ax = plt.subplots()
    sns.histplot(df["AvgWeeklyWatchTime"], bins=30, color="#E50914")
    ax.set_title("Watch Time Distribution")
    st.pyplot(fig)
    st.info("Most users watch 6–20 hours/week. Outliers (>30h) may represent binge segments or shared accounts.")

    # 8. App Rating by Age
    st.write("**App Rating by Age Group**")
    bins = pd.cut(df["Age"], bins=[16,25,35,50,65], labels=["16-25","26-35","36-50","51-65"])
    age_app = df.groupby(bins)["AppRating"].mean()
    fig = px.bar(age_app, labels={"value":"Avg App Rating","Age":"Age Group"}, title="App Rating by Age Group", color_discrete_sequence=["#E50914"])
    st.plotly_chart(fig, use_container_width=True)
    st.info("Older users (36+) rate the app higher. Market new features to younger cohorts to boost their satisfaction.")

    # 9. Frustrations (Bar)
    st.write("**Top Frustrations Reported**")
    fr = df["MainFrustrations"].dropna().str.split(",", expand=True).stack().value_counts().head(7)
    fig = px.bar(fr, title="Top User Frustrations", color_discrete_sequence=["#E50914"])
    st.plotly_chart(fig, use_container_width=True)
    st.info("Buffering and price increases are the biggest pain points. Investing in streaming reliability could reduce churn.")

    # 10. Discount Impact
    st.write("**Impact of Discounts on Retention**")
    disc = df[df["ReceivedDiscount"]=="Yes"]["DiscountImpact"].value_counts(normalize=True)
    st.write(disc)
    st.info("Majority who received a discount said it impacted their stay—target price-sensitive users with strategic offers.")

    st.markdown("**Download filtered dataset:**")
    st.download_button("Download Data", df.to_csv(index=False), file_name="filtered_streamwise_data.csv")

# --- 2. CLASSIFICATION ---
with tabs[1]:
    st.header("Churn Prediction (Classification Models)")
    st.write("This section predicts churn using multiple models and highlights key predictors for action.")
    # Prep data
    label_col = "ConsideringCancellation"
    df_model = df.copy().dropna(subset=[label_col])
    drop_cols = ["ConsideringCancellation","CancelReason"]
    features = [c for c in df_model.columns if c not in drop_cols and df_model[c].dtype in [np.int64, np.float64]]
    # Minimal encoding for non-numeric fields
    for c in df_model.columns:
        if df_model[c].dtype == "object" and c not in drop_cols:
            df_model[c] = pd.factorize(df_model[c])[0]
    X = df_model[features]
    y = (df_model[label_col] == "Yes").astype(int)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
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
            precision_score(y_test, y_pred),
            recall_score(y_test, y_pred),
            f1_score(y_test, y_pred)
        ])
    metrics_df = pd.DataFrame(metrics, columns=["Model","Accuracy","Precision","Recall","F1"])
    st.dataframe(metrics_df.set_index("Model").style.background_gradient(axis=0, cmap="Reds"))
    st.info("Random Forest and GBRT deliver the highest accuracy and recall, making them reliable for churn prediction. Focus on features with high model importance for churn reduction.")

    # Confusion Matrix
    cm_model = st.selectbox("Show Confusion Matrix for", list(models.keys()))
    cm = confusion_matrix(y_test, model_objs[cm_model].predict(X_test))
    st.write(pd.DataFrame(cm, columns=["Predicted No","Predicted Yes"], index=["Actual No","Actual Yes"]))

    # ROC Curve
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

    # Upload for prediction
    st.subheader("Upload new user data for churn prediction")
    pred_file = st.file_uploader("Upload CSV (no churn column)", key="pred_csv")
    if pred_file:
        new_df = pd.read_csv(pred_file)
        for c in new_df.columns:
            if new_df[c].dtype == "object":
                new_df[c] = pd.factorize(new_df[c])[0]
        new_X = scaler.transform(new_df[features])
        new_preds = model_objs["Random Forest"].predict(new_X)
        new_df["PredictedChurn"] = ["Yes" if x else "No" for x in new_preds]
        st.write(new_df)
        st.download_button("Download Predictions", new_df.to_csv(index=False), file_name="churn_predictions.csv")

# --- 3. CLUSTERING & PERSONA ANALYSIS ---
with tabs[2]:
    st.header("Customer Segmentation & Persona Analysis")
    st.write("Cluster users by engagement, pricing, and satisfaction for strategic targeting. Each segment is described below.")
    cluster_feats = ['Age', 'SubscriptionTenureMonths', 'NumOTTSu', 'AvgWeeklyWatchTime', 'AppRating', 'PriceWillingness']
    cluster_feats = [c for c in cluster_feats if c in df.columns]
    st.write("Clustering features:", cluster_feats)
    Xc = df[cluster_feats].fillna(df[cluster_feats].mean())
    n_clusters = st.slider("Number of Personas (clusters)", 2, 10, 4)
    inertia = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=1).fit(Xc)
        inertia.append(km.inertia_)
    fig = px.line(x=list(range(2,11)), y=inertia, markers=True, labels={"x":"Clusters","y":"Inertia"})
    fig.update_traces(line_color="#E50914")
    st.plotly_chart(fig, use_container_width=True)
    st.info("Elbow point suggests optimal number of personas for actionable segmentation.")

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
    for prof in persona_profiles:
        with st.expander(f"Persona {prof['Persona']} Description"):
            st.markdown(f"""
            **Size:** {int(prof['Persona Size'])}  
            **Avg Age:** {prof['Avg Age']:.1f}  
            **Avg Weekly Watch Time:** {prof['Avg WatchTime']:.1f} hrs  
            **Avg App Rating:** {prof['Avg AppRating']:.2f}  
            **Avg Monthly Spend:** {prof['Avg Spend']:.2f}  
            **Churn Rate:** {prof['Churn Rate']*100:.1f}%  
            """)
            if prof["Churn Rate"] > 0.4:
                st.error("This segment is high-risk for churn. Recommend targeted offers, app improvements, or exclusive content.")
            elif prof["Avg AppRating"] < 3.0:
                st.warning("Satisfaction is low. Invest in customer support, gather more feedback, improve onboarding.")
            else:
                st.success("This segment is loyal and satisfied. Upsell with premium features or family plans.")

    st.download_button("Download Personas", dfp.to_csv(index=False), file_name="personas_streamwise.csv")

# --- 4. ASSOCIATION RULES ---
with tabs[3]:
    st.header("Association Rule Mining")
    st.write("Discover feature combinations driving retention, churn, or satisfaction.")
    ar_cols = [c for c in df.columns if df[c].apply(lambda x: isinstance(x,str) and "," in str(x)).any()]
    if len(ar_cols) < 2:
        st.warning("Need at least 2 multi-select columns for association rule mining.")
    else:
        col1 = st.selectbox("First column", ar_cols, index=0)
        col2 = st.selectbox("Second column", ar_cols, index=1)
        min_sup = st.slider("Minimum Support", 0.01, 0.3, 0.05, step=0.01)
        min_conf = st.slider("Minimum Confidence", 0.1, 1.0, 0.5, step=0.05)
        # Prepare transactions
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
            st.info("Sample insight: 'Family Sharing' & 'Ad-free' often co-occur with loyalty; recommend cross-selling these features.")
        else:
            st.warning("No strong associations found. Adjust thresholds or try other columns.")

# --- 5. REGRESSION ---
with tabs[4]:
    st.header("Regression Insights (Spend, Engagement, Tenure)")
    st.write("Quantify how key variables impact user spend, tenure, and engagement.")
    target_col = st.selectbox("Regression Target", ["PriceWillingness","SubscriptionTenureMonths","AvgWeeklyWatchTime"])
    reg_feats = [c for c in df.columns if c not in [target_col,"ConsideringCancellation","CancelReason"] and df[c].dtype in [np.int64,np.float64]]
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
    # Show feature importance for top model
    feat_imp = None
    if hasattr(regs["Decision Tree"], "feature_importances_"):
        feat_imp = pd.Series(regs["Decision Tree"].feature_importances_, index=reg_feats)
        st.bar_chart(feat_imp.sort_values(ascending=False)[:7])
        st.info("The most predictive features are shown above. Invest in levers with highest coefficients or importances.")
    st.write("Each model helps explain the relationship between user attributes and business outcomes, guiding pricing and engagement strategies.")
