"""
app/pages/4_Citation_Integrity.py
---------------------------------
Streamlit multi-page app: Citation Integrity Dashboard.

Monitors citation patterns, detects manipulation, verifies source
authenticity, and tracks citation network health.
"""

import streamlit as st

from app.components.citation_integrity_dashboard import main

# Page configuration
st.set_page_config(
    page_title="Citation Integrity - Plagiarism Detector",
    page_icon="🔍",
    layout="wide",
)

main()
