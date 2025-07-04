import streamlit as st

# -------------------------------
# Main Intro Function
# -------------------------------
def show_intro():
    st.markdown("## Welcome to **StreamWise**")
    st.markdown("""
        Your all-in-one, MBA-level OTT analytics and churn prediction dashboard.  
        _Analyze. Segment. Act._ – in true Netflix style.  
        
        Click the **StreamWise** button to get started!
    """)

    # Try loading logo (local file first, then fallback URL)
    try:
        st.image("logo.png", width=150)
    except Exception:
        st.image("https://raw.githubusercontent.com/JigarMarvaniya/Streamwise/new-functionality/assets/logo.png", width=150)

    # Embed audio (intro.mp3 must be in same directory or skipped)
    try:
        audio_file = open("intro.mp3", 'rb')
        st.audio(audio_file.read(), format='audio/mp3')
    except FileNotFoundError:
        st.warning("Intro audio file not found. Skipping audio...")

    # Navigation button to main dashboard
    if st.button("▶️ StreamWise"):
        st.session_state["show_dashboard"] = True
        st.experimental_rerun()


# -------------------------------
# Main Dashboard Function
# -------------------------------
def show_dashboard():
    st.title("📊 StreamWise Dashboard")

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Data Visualisation", "🔍 Classification", "📦 Clustering", "📁 Upload/Download"])

    with tab1:
        st.header("Data Visualisation")
        st.markdown("Explore descriptive analytics here.")
        st.info("10+ charts to be added from your EDA script")

    with tab2:
        st.header("Classification Models")
        st.markdown("KNN, DT, RF, GBRT with confusion matrix and ROC.")
        st.warning("Implement model training here or call from model.py")

    with tab3:
        st.header("Clustering")
        st.markdown("Apply K-Means clustering and visualize customer segments.")

    with tab4:
        st.header("Upload / Predict New Data")
        uploaded_file = st.file_uploader("Upload data to predict churn", type=["csv"])
        if uploaded_file:
            st.success("File uploaded. Pass to your model for inference.")
            # Placeholder: Add your prediction logic here


# -------------------------------
# Main App Flow
# -------------------------------
def main():
    st.set_page_config(page_title="StreamWise", layout="wide")

    # Optional styling
    st.markdown("""
        <style>
        .stApp { background-color: #111; color: white; }
        .css-10trblm { color: #ff4b4b; }
        </style>
    """, unsafe_allow_html=True)

    # Show intro or dashboard
    if "show_dashboard" not in st.session_state:
        show_intro()
    else:
        show_dashboard()


# -------------------------------
# Run the App
# -------------------------------
if __name__ == "__main__":
    main()
