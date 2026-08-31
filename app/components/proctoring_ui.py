import streamlit as st
import pandas as pd
import numpy as np

def render_live_proctoring_dashboard():
    """
    Renders the real-time telemetry dashboard during active exam windows.
    """
    st.title("Live Proctoring Dashboard")
    st.info("Listening for real-time browser extension telemetry...")
    
    # Mock data to simulate incoming telemetry
    col1, col2, col3 = st.columns(3)
    col1.metric("Active Sessions", "142")
    col2.metric("Rapid Paste Events", "7", delta="2", delta_color="inverse")
    col3.metric("Tab Switches (Suspicious)", "15", delta="5", delta_color="inverse")

    st.subheader("Recent Flagged Activity")
    
    # Mock dataframe
    data = {
        "Student ID": ["user_882", "user_119", "user_043"],
        "Event Type": ["Paste", "Tab Switch", "Paste"],
        "Timestamp": ["10:04:12 AM", "10:04:45 AM", "10:05:01 AM"],
        "Severity": ["High", "Medium", "High"]
    }
    
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)

