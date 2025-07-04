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

    # ✅ Load logo image from GitHub
    try:
        st.image("https://raw.githubusercontent.com/JigarMarvaniya/Streamwise/new-functionality/assets/streamwise_logo.png", width=150)
    except Exception:
        st.warning("Logo could not be loaded.")

    # 🔇 No audio here — we'll inject it after login in the dashboard

    # Dashboard access button
    if st.button("▶️ StreamWise"):
        st.session_state["show_dashboard"] = True


# -------------------------------
# Main Dashboard Function
# -------------------------------
def show_dashboard():
    st.title("📊 StreamWise Dashboard")

    # ✅ Inject hidden autoplay audio once
    if "audio_played" not in st.session_state:
        st.session_state["audio_played"] = True
        st.markdown(
            """
            <audio autoplay hidden>
                <source src="https://raw.githubusercontent.com/JigarMarvaniya/Streamwise/new-functionality/assets/streamwise_audio.mp3" type="audio/mpeg">
            </audio>
            """,
            unsafe_allow_html=True
        )

    # Dashboard Tabs
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

    # Optional: dark theme styling
    st.markdown("""
        <style>
        .stApp { background-color: #111; color: white; }
        .css-10trblm { color: #ff4b4b; }
        </style>
    """, unsafe_allow_html=True)

    if "show_dashboard" not in st.session_state:
        show_intro()
    else:
        show_dashboard()


if __name__ == "__main__":
    main()
