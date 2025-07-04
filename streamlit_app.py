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

    # Try loading logo
    try:
        st.image("logo.png", width=150)
    except Exception:
        st.image("https://raw.githubusercontent.com/JigarMarvaniya/Streamwise/new-functionality/assets/logo.png", width=150)

    # Embed audio (optional)
    try:
        audio_file = open("intro.mp3", 'rb')
        st.audio(audio_file.read(), format='audio/mp3')
    except FileNotFoundError:
        st.warning("Intro audio file not found. Skipping audio...")

    # Button to go to dashboard
    if st.button("▶️ StreamWise"):
        st.session_state["show_dashboard"] = True  # No explicit rerun


# -------------------------------
# Main Dashboard Function
# -------------------------------
def show_dashboard():
    st.title("📊 StreamWise Dashboard")

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Data Visualisation", "🔍 Classification", "📦 Clustering", "📁 Upload/Download"])

    with tab1:
        st.header("Data Visualisation")
        st.markdown("Explore descriptive analytics here.")
        st.info("Charts and insights coming soon...")

    with tab2:
        st.header("Classification Models")
        st.markdown("Apply ML models like KNN, DT, RF, GBRT.")
        st.warning("Model section under development.")

    with tab3:
        st.header("Clustering")
        st.markdown("Explore user clusters using K-Means.")

    with tab4:
        st.header("Upload / Predict New Data")
        uploaded_file = st.file_uploader("Upload data to predict churn", type=["csv"])
        if uploaded_file:
            st.success("File uploaded. Add model prediction logic here.")


# -------------------------------
# Main App Entry
# -------------------------------
def main():
    st.set_page_config(page_title="StreamWise", layout="wide")

    st.markdown("""
        <style>
        .stApp { background-color: #111; color: white; }
        .css-10trblm { color: #ff4b4b; }
        </style>
    """, unsafe_allow_html=True)

    # Session-based routing
    if "show_dashboard" not in st.session_state:
        show_intro()
    else:
        show_dashboard()


if __name__ == "__main__":
    main()
