
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.model_utils import preprocess_classification_data
from utils.viz_utils import plot_correlation_heatmap

st.set_page_config(page_title="StreamWise OTT Analytics Dashboard", layout="wide")

st.sidebar.title("Navigation")
tabs = ["Data Visualization", "Classification", "Clustering", "Association Rule Mining", "Regression"]
page = st.sidebar.radio("Go to", tabs)

uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("streamwise_survey_synthetic.csv")

if page == "Data Visualization":
    st.title("Data Visualization & Insights")
    st.subheader("Churn Rate by Billing Cycle")
    churn_by_bill = df.groupby("BillingCycle")["ConsideringCancellation"].value_counts(normalize=True).unstack().fillna(0)
    fig, ax = plt.subplots()
    churn_by_bill.plot(kind='bar', stacked=True, ax=ax)
    ax.set_ylabel("Proportion")
    st.pyplot(fig)
    st.write("Monthly billing cycles show higher churn. Analyze further below.")

    st.subheader("Correlation Heatmap")
    num_cols = df.select_dtypes(include=['int', 'float']).columns
    fig2 = plot_correlation_heatmap(df[num_cols])
    st.pyplot(fig2)

    st.write("Download current data:")
    st.download_button("Download Data as CSV", df.to_csv(index=False), file_name="streamwise_survey_synthetic.csv", mime="text/csv")

# --- More content for other tabs would go here (as in the detailed plan) ---
st.sidebar.markdown("---")
st.sidebar.info("Developed for StreamWise SaaS OTT Analytics.")
