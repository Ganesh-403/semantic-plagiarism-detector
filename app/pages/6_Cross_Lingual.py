"""
app/pages/6_Cross_Lingual.py
----------------------------
Streamlit multi-page app: Cross-Lingual Comparison Hub.

Multi-language document comparison with translation quality analysis,
cultural context detection, and cross-lingual plagiarism identification.
"""

import streamlit as st

from app.components.cross_lingual_comparison_hub import main

# Page configuration
st.set_page_config(
    page_title="Cross-Lingual Comparison - Plagiarism Detector",
    page_icon="🌐",
    layout="wide",
)

main()
