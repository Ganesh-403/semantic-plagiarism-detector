"""
app/pages/7_Fingerprint_Vault.py
--------------------------------
Streamlit multi-page app: Document Fingerprint Vault.

Multi-algorithm fingerprint generation, comparison, tamper detection,
and similarity search for document integrity verification.
"""

import streamlit as st

from app.components.document_fingerprint_vault import main

# Page configuration
st.set_page_config(
    page_title="Fingerprint Vault - Plagiarism Detector",
    page_icon="🔐",
    layout="wide",
)

main()
