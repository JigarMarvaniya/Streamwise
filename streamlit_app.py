import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.set_page_config(page_title="OTT Platform Behavior Dashboard", layout="wide")

# --- Data Upload ---
st.sidebar.title("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your OTT CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.warning("Please upload your Synthetic_OTT_Dataset.csv to get started.")
    st.stop()

# --- KPIs ---
st.title("OTT Subscriber Insights Dashboard")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Users", len(df))
if 'CancelledSubscription' in df.columns:
    col2.metric("Churn %", f"{100 * (df['CancelledSubscription']=='Yes').mean():.1f}%")
if 'AvgWatchTimePerWeek' in df.columns:
    col3.metric("Avg Weekly Watch Time", f"{df['AvgWatchTimePerWeek'].mean():.1f} hrs")
if 'AppRating' in df.columns:
    col4.metric("Avg App Rating", f"{df['AppRating'].mean():.2f}")

# --- Filters ---
st.sidebar.header("Filter Data")
filter_cols = ['AgeGroup', 'FamilySubscription', 'BillingCycle', 'Preferred Device', 'Referral Source']
for col in filter_cols:
    if col in df.columns:
        opts = df[col].dropna().unique().tolist()
        chosen = st.sidebar.multiselect(col, opts, default=opts)
        df = df[df[col].isin(chosen)]

# --- Tabbed Navigation ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Churn Analysis", "Engagement & Usage", "Satisfaction & Support", "Monetization", "Churn Prediction"
])

# 1. Churn Analysis
with tab1:
    st.subheader("Churn by Segment")
    if 'CancelledSubscription' in df.columns:
        col = st.selectbox("Segment By", ['AgeGroup', 'FamilySubscription', 'BillingCycle', 'Referral Source'])
        if col in df.columns:
            churn_rate = df.groupby(col)['CancelledSubscription'].apply(lambda x: (x=="Yes").mean())
            st.bar_chart(churn_rate)
            st.write("Higher churn in:", churn_rate.idxmax())
        # Boxplots
        for feat in ['AvgWatchTimePerWeek', 'MonthlyPlanPrice']:
            if feat in df.columns:
                st.write(f"**{feat} vs. Churn**")
                fig, ax = plt.subplots()
                sns.boxplot(x='CancelledSubscription', y=feat, data=df, ax=ax)
                st.pyplot(fig)

# 2. Engagement & Usage
with tab2:
    st.subheader("Engagement by Features")
    if 'PreferredGenre' in df.columns and 'AvgWatchTimePerWeek' in df.columns:
        genre_watch = df.groupby('PreferredGenre')['AvgWatchTimePerWeek'].mean()
        st.bar_chart(genre_watch)
    if 'Primary Viewing Time Slot' in df.columns and 'AvgWatchTimePerWeek' in df.columns:
        slot_watch = df.groupby('Primary Viewing Time Slot')['AvgWatchTimePerWeek'].mean()
        st.bar_chart(slot_watch)
    if 'Content Skipped Percentage' in df.columns:
        st.write("Distribution of Content Skipped")
        fig, ax = plt.subplots()
        sns.histplot(df['Content Skipped Percentage'].dropna(), bins=30, kde=True, ax=ax)
        st.pyplot(fig)

# 3. Satisfaction & Support
with tab3:
    st.subheader("App & Support Ratings")
    if 'AppRating' in df.columns:
        st.write("App Rating Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df['AppRating'].dropna(), bins=20, ax=ax)
        st.pyplot(fig)
    if 'CustomerSupportRating' in df.columns:
        st.write("Support Rating Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df['CustomerSupportRating'].dropna(), bins=10, ax=ax)
        st.pyplot(fig)
    if 'App Opens per Week' in df.columns:
        st.write("App Opens vs. App Rating")
        fig, ax = plt.subplots()
        sns.scatterplot(x='App Opens per Week', y='AppRating', data=df, ax=ax)
        st.pyplot(fig)

# 4. Monetization
with tab4:
    st.subheader("Plan Pricing and Churn")
    if 'MonthlyPlanPrice' in df.columns:
        st.write("Plan Price by Segment")
        seg = st.selectbox("Segment By", ['AgeGroup', 'FamilySubscription', 'BillingCycle'])
        price_seg = df.groupby(seg)['MonthlyPlanPrice'].mean()
        st.bar_chart(price_seg)
        if 'CancelledSubscription' in df.columns:
            st.write("Avg Price: Churned vs. Retained")
            st.dataframe(df.groupby('CancelledSubscription')['MonthlyPlanPrice'].mean())

# 5. Churn Prediction (Classification Model)
with tab5:
    st.subheader("Predict Churn (Demo Model)")
    features = ['MonthsSubscribed', 'MonthlyPlanPrice', 'AvgWatchTimePerWeek', 'AppRating']
    avail = [f for f in features if f in df.columns]
    if len(avail) >= 2 and 'CancelledSubscription' in df.columns:
        model_df = df.dropna(subset=avail+['CancelledSubscription'])
        X = model_df[avail]
        y = (model_df['CancelledSubscription']=='Yes').astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        clf = RandomForestClassifier(n_estimators=50, random_state=0)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        st.write(f"Model Test Accuracy: {score:.2f}")
        importances = pd.Series(clf.feature_importances_, index=avail)
        st.bar_chart(importances)
        st.write("Top churn predictors:", importances.sort_values(ascending=False).index[0])

# --- Data Download ---
st.sidebar.title("Download Data")
st.sidebar.download_button("Download Filtered Data", df.to_csv(index=False), file_name="filtered_ott.csv", mime="text/csv")
