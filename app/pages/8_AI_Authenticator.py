"""
app/pages/8_AI_Authenticator.py
--------------------------------
Streamlit multi-page app: AI Content Authenticator.

Detects AI-generated content, deepfake text, and synthetic media
with confidence scoring and provenance tracking.
"""

import streamlit as st

from app.components.ai_content_authenticator import main

# Page configuration
st.set_page_config(
    page_title="AI Authenticator - Plagiarism Detector",
    page_icon="🤖",
    layout="wide",
)

main()
