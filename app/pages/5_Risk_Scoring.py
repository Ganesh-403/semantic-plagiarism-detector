"""
app/pages/5_Risk_Scoring.py
---------------------------
Streamlit multi-page app: Plagiarism Risk Scoring Engine.

Multi-dimensional risk assessment with pattern detection,
trend analysis, and mitigation recommendations.
"""

import streamlit as st

from app.components.plagiarism_risk_scoring_engine import main

# Page configuration
st.set_page_config(
    page_title="Risk Scoring - Plagiarism Detector",
    page_icon="🛡️",
    layout="wide",
)

main()
