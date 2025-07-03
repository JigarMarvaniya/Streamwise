import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_curve, auc, classification_report)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

import plotly.express as px
import plotly.figure_factory as ff
import io

st.set_page_config(page_title="StreamWise OTT Analytics Dashboard", layout="wide")

# --- Helper functions ---
def preprocess_classification_data(df, label_col):
    df = df.copy()
    cat_cols = df.select_dtypes(include='object').columns
    le_dict = {}
    for col in cat_cols:
        if col == label_col: continue
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le
    df = df.dropna()
    X = df.drop(columns=[label_col])
    y = (df[label_col] == 'Yes').astype(int) if set(df[label_col]) == {'Yes', 'No'} else df[label_col]
    return X, y, le_dict

def get_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0)
    }

def plot_roc_curves(models, X_test, y_test):
    plt.figure(figsize=(8,6))
    for name, model in models.items():
        y_score = model.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_score)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr,tpr):.2f})")
    plt.plot([0,1],[0,1],'k--',alpha=0.6)
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curve')
    plt.legend(); plt.tight_layout()
    return plt.gcf()

def get_persona_table(df, labels, n_clusters):
    dfc = df.copy()
    dfc["Cluster"] = labels
    persona = []
    for i in range(n_clusters):
        seg = dfc[dfc["Cluster"] == i]
        persona.append({
            "Cluster": i,
            "Avg Age": seg["Age"].mean(),
            "Top Plan": seg["PlanType"].mode()[0] if not seg["PlanType"].mode().empty else "",
            "Avg Tenure": seg["SubscriptionTenureMonths"].mean(),
            "Churn Rate": (seg["ConsideringCancellation"]=="Yes").mean(),
            "Main Location": seg["Location"].mode()[0] if not seg["Location"].mode().empty else ""
        })
    return pd.DataFrame(persona)

def run_apriori(df, col1, col2, min_support=0.05, min_confidence=0.4, top_n=10):
    trans = []
    for i, row in df[[col1, col2]].iterrows():
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
    freq = apriori(df_enc, min_support=min_support, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=min_confidence)
    rules = rules.sort_values("confidence", ascending=False).head(top_n)
    return rules[["antecedents","consequents","support","confidence","lift"]]

# --- Data Loading ---
st.sidebar.title("Navigation")
tabs = ["Data Visualization", "Classification", "Clustering", "Association Rule Mining", "Regression"]
page = st.sidebar.radio("Go to", tabs)

uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("streamwise_survey_synthetic.csv")

# --- Data Visualization ---
if page == "Data Visualization":
    st.title("Data Visualization & Insights")
    st.write("**Sample: Churn Rate by Billing Cycle**")
    churn_by_bill = df.groupby("BillingCycle")["ConsideringCancellation"].value_counts(normalize=True).unstack().fillna(0)
    fig, ax = plt.subplots()
    churn_by_bill.plot(kind='bar', stacked=True, ax=ax)
    ax.set_ylabel("Proportion")
    st.pyplot(fig)
    st.write("Monthly billing cycles show higher churn. Quarterly/yearly more loyal.")

    st.write("**Correlation Heatmap**")
    num_cols = df.select_dtypes(include=[np.number]).columns
    fig2 = plt.figure(figsize=(8,6))
    sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm")
    st.pyplot(fig2)

    st.write("**Churn by Age Group and PlanType**")
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0,25,40,60,100], labels=["<25","25-40","41-60","60+"])
    pivot = df.pivot_table(index="PlanType",columns="AgeGroup", values="ConsideringCancellation",aggfunc=lambda x:(x=="Yes").mean())
    st.dataframe(pivot.style.background_gradient(cmap='Reds'))
    st.write("Older users on Premium plans are less likely to churn, while young users on Basic/Standard churn more.")

    st.write("**Download current data**")
    st.download_button("Download Data as CSV", df.to_csv(index=False), file_name="streamwise_data.csv", mime="text/csv")

# --- Classification ---
elif page == "Classification":
    st.title("Churn Prediction – Classification")
    label_col = "ConsideringCancellation"
    with st.expander("Data Preparation", expanded=False):
        st.write("Encoding categorical features and dropping missing data.")
    X, y, _ = preprocess_classification_data(df, label_col)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.25, random_state=42)
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "DecisionTree": DecisionTreeClassifier(random_state=0),
        "RandomForest": RandomForestClassifier(n_estimators=50, random_state=0),
        "GBRT": GradientBoostingClassifier(n_estimators=50, random_state=0)
    }
    results = []
    model_objs = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = get_metrics(y_test, y_pred)
        results.append([name, metrics["Accuracy"], metrics["Precision"], metrics["Recall"], metrics["F1"]])
        model_objs[name] = model
    st.subheader("Model Performance")
    st.dataframe(pd.DataFrame(results, columns=["Model","Accuracy","Precision","Recall","F1"]).set_index("Model").style.background_gradient(axis=0))
    # Confusion Matrix Selector
    st.subheader("Confusion Matrix")
    cm_model = st.selectbox("Choose a model", list(models.keys()))
    cm = confusion_matrix(y_test, model_objs[cm_model].predict(X_test))
    st.write(pd.DataFrame(cm, columns=["Predicted No","Predicted Yes"], index=["Actual No","Actual Yes"]))
    # ROC Curves
    st.subheader("ROC Curve (All Models)")
    st.pyplot(plot_roc_curves(model_objs, X_test, y_test))
    # Upload for new predictions
    st.subheader("Upload new data for prediction (no 'ConsideringCancellation' column)")
    pred_file = st.file_uploader("Upload new CSV", key="pred_csv")
    if pred_file:
        new_df = pd.read_csv(pred_file)
        new_X, _, _ = preprocess_classification_data(new_df.assign(ConsideringCancellation="No"), "ConsideringCancellation")
        new_X_scaled = scaler.transform(new_X)
        pred = model_objs["RandomForest"].predict(new_X_scaled)
        result_df = new_df.copy()
        result_df["PredictedChurn"] = ["Yes" if x==1 else "No" for x in pred]
        st.dataframe(result_df)
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions", csv, file_name="predicted_churn.csv", mime="text/csv")

# --- Clustering ---
elif page == "Clustering":
    st.title("Customer Segmentation – Clustering")
    st.write("Select features and number of clusters (2–10):")
    num_clusters = st.slider("Number of clusters", 2, 10, 4)
    cluster_feats = st.multiselect("Features to cluster on", options=df.select_dtypes(include=[np.number]).columns.tolist(), default=["Age","SubscriptionTenureMonths","WeeklyWatchTimeHrs"])
    Xc = df[cluster_feats].dropna()
    inertia = []
    for k in range(2,11):
        km = KMeans(n_clusters=k, random_state=1).fit(Xc)
        inertia.append(km.inertia_)
    fig, ax = plt.subplots()
    ax.plot(range(2,11), inertia, marker="o")
    ax.set_xlabel("Num Clusters"); ax.set_ylabel("Inertia"); ax.set_title("Elbow Plot")
    st.pyplot(fig)
    km = KMeans(n_clusters=num_clusters, random_state=1).fit(Xc)
    dfc = df.loc[Xc.index].copy()
    dfc["Cluster"] = km.labels_
    st.write("Cluster Persona Table:")
    persona_df = get_persona_table(dfc, km.labels_, num_clusters)
    st.dataframe(persona_df)
    st.write("Download labeled data:")
    csv = dfc.to_csv(index=False).encode('utf-8')
    st.download_button("Download Clustered Data", csv, file_name="clustered_data.csv", mime="text/csv")

# --- Association Rule Mining ---
elif page == "Association Rule Mining":
    st.title("Association Rule Mining")
    multi_cols = [c for c in df.columns if df[c].apply(lambda x: isinstance(x,str) and "," in str(x)).any()]
    col1 = st.selectbox("First multi-select column", multi_cols, index=0)
    col2 = st.selectbox("Second multi-select column", multi_cols, index=1 if len(multi_cols)>1 else 0)
    min_sup = st.slider("Min Support", 0.01, 0.5, 0.05)
    min_conf = st.slider("Min Confidence", 0.1, 1.0, 0.4)
    rules = run_apriori(df, col1, col2, min_support=min_sup, min_confidence=min_conf, top_n=10)
    st.dataframe(rules)

# --- Regression ---
elif page == "Regression":
    st.title("Regression & Feature Insights")
    reg_target = st.selectbox("Regression target", ["PriceWillingness","SubscriptionTenureMonths","WeeklyWatchTimeHrs"])
    feat_cols = st.multiselect("Features to use", options=df.select_dtypes(include=[np.number]).columns.tolist(), default=["Age","NumOTTSu"])
    Xr = df[feat_cols].dropna()
    yr = df.loc[Xr.index, reg_target]
    # Models
    reg_models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "DecisionTree": DecisionTreeRegressor(max_depth=5)
    }
    reg_results = []
    for name, model in reg_models.items():
        model.fit(Xr, yr)
        score = model.score(Xr, yr)
        reg_results.append([name, score])
    st.dataframe(pd.DataFrame(reg_results, columns=["Model","R^2"]).set_index("Model"))
    # Plot predictions for first model
    preds = reg_models["Linear"].predict(Xr)
    fig, ax = plt.subplots()
    ax.scatter(yr, preds, alpha=0.4)
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted"); ax.set_title(f"Linear Regression: {reg_target}")
    st.pyplot(fig)
    st.write("Insights: R^2 indicates how well features predict the target. Linear models reveal trends; DT handles nonlinearity.")

st.sidebar.info("Developed for StreamWise SaaS OTT Analytics.")
