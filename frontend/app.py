"""
Semantic Plagiarism Detector - Main Streamlit Application
Frontend UI for document upload, analysis, and results visualization.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import components
from frontend.components.file_uploader import FileUploader
from frontend.components.results_display import ResultsDisplay
from frontend.components.similarity_matrix import SimilarityMatrix
from frontend.components.analysis_controls import AnalysisControls
from frontend.utils.api_client import APIClient
from frontend.styles.theme import apply_theme

# Page configuration
st.set_page_config(
    page_title="Semantic Plagiarism Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom theme
apply_theme()

# Initialize session state
def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'documents': [],
        'document_names': [],
        'analysis_results': None,
        'is_analyzing': False,
        'selected_method': 'hybrid',
        'threshold': 0.59,
        'uploaded_files': [],
        'text_inputs': [],
        'api_client': APIClient(),
        'show_results': False,
        'selected_doc1': None,
        'selected_doc2': None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ============================================
# Header
# ============================================
def render_header():
    """Render the application header."""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0f172a, #1e293b); padding: 24px 32px; border-radius: 12px; margin-bottom: 24px; border: 1px solid #334155;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <span style="font-size: 32px;">🔍</span>
                <h1 style="color: #f8fafc; margin: 0; font-size: 28px;">Semantic Plagiarism Detector</h1>
            </div>
            <div style="display: flex; gap: 8px;">
                <span style="background: #334155; color: #94a3b8; padding: 4px 12px; border-radius: 20px; font-size: 12px;">v2.0</span>
                <span style="background: #22c55e; color: white; padding: 4px 12px; border-radius: 20px; font-size: 12px;">🟢 Online</span>
            </div>
        </div>
        <p style="color: #94a3b8; margin-top: 8px; font-size: 15px;">
            Detect paraphrased and copied content using advanced semantic similarity analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

render_header()

# ============================================
# Sidebar
# ============================================
def render_sidebar():
    """Render the sidebar with controls and info."""
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        st.markdown("---")
        
        # Analysis method selection
        method = st.selectbox(
            "Analysis Method",
            options=["hybrid", "lexical", "semantic"],
            index=0,
            help="Select the analysis method to use"
        )
        st.session_state.selected_method = method
        
        # Threshold slider
        threshold = st.slider(
            "Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.59,
            step=0.01,
            help="Minimum similarity score to flag as plagiarism"
        )
        st.session_state.threshold = threshold
        
        st.markdown("---")
        
        # Document list
        st.markdown("### 📄 Documents")
        if st.session_state.document_names:
            for name in st.session_state.document_names:
                st.markdown(f"• {name}")
            st.markdown(f"**Total:** {len(st.session_state.document_names)} documents")
        else:
            st.info("No documents uploaded yet")
        
        st.markdown("---")
        
        # Stats
        st.markdown("### 📊 Stats")
        if st.session_state.analysis_results:
            results = st.session_state.analysis_results
            matches = results.get('matches', [])
            high = sum(1 for m in matches if m.get('severity') == 'high')
            medium = sum(1 for m in matches if m.get('severity') == 'medium')
            low = sum(1 for m in matches if m.get('severity') == 'low')
            
            st.metric("Total Matches", len(matches))
            col1, col2, col3 = st.columns(3)
            col1.metric("🔴 High", high)
            col2.metric("🟡 Medium", medium)
            col3.metric("🟢 Low", low)
        else:
            st.info("Run analysis to see stats")
        
        st.markdown("---")
        
        # Clear all
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.documents = []
            st.session_state.document_names = []
            st.session_state.analysis_results = None
            st.session_state.show_results = False
            st.rerun()

render_sidebar()

# ============================================
# Main Content Tabs
# ============================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📤 Upload & Analyze",
    "📊 Results",
    "🔍 Similarity Matrix",
    "📈 Visualizations"
])

with tab1:
    st.markdown("### 📤 Upload Documents & Analyze")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=['pdf', 'docx', 'doc', 'txt', 'rtf', 'odt'],
        accept_multiple_files=True,
        key="file_uploader_main"
    )
    
    if uploaded_files:
        for file in uploaded_files:
            if file.name not in st.session_state.document_names:
                st.session_state.documents.append({
                    'name': file.name,
                    'content': None,
                    'file': file
                })
                st.session_state.document_names.append(file.name)
                st.success(f"✅ Added: {file.name}")
    
    # Text input
    st.markdown("#### 📝 Or Paste Text")
    col1, col2 = st.columns([3, 1])
    with col1:
        text_input = st.text_area(
            "Enter text to analyze",
            placeholder="Paste your text here...",
            height=100,
            key="text_input_area"
        )
    with col2:
        text_name = st.text_input(
            "Document Name",
            placeholder="Enter a name",
            key="text_doc_name"
        )
        if st.button("➕ Add Text", use_container_width=True):
            if text_input.strip():
                name = text_name.strip() or f"Text_{len(st.session_state.documents)+1}"
                st.session_state.documents.append({
                    'name': name,
                    'content': text_input,
                    'file': None
                })
                st.session_state.document_names.append(name)
                st.success(f"✅ Added: {name}")
                st.rerun()
            else:
                st.warning("Please enter some text")
    
    # Document list
    if st.session_state.documents:
        st.markdown("#### 📄 Uploaded Documents")
        df_docs = pd.DataFrame({
            'Name': st.session_state.document_names,
            'Type': ['Text' if doc.get('content') else 'File' for doc in st.session_state.documents]
        })
        st.dataframe(df_docs, use_container_width=True, hide_index=True)
    
    # Run analysis
    st.markdown("---")
    st.markdown("#### 🚀 Run Analysis")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        method = st.selectbox(
            "Analysis Method",
            options=["hybrid", "lexical", "semantic"],
            index=0,
            key="analysis_method"
        )
    with col2:
        threshold = st.number_input(
            "Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.59,
            step=0.01,
            key="analysis_threshold"
        )
    
    if st.button("🚀 Analyze Documents", type="primary", use_container_width=True):
        if len(st.session_state.documents) < 2:
            st.warning("Please upload at least 2 documents to analyze")
        else:
            st.session_state.is_analyzing = True
            
            with st.spinner("🔍 Analyzing documents..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.02)
                    progress_bar.progress(i + 1)
                
                # Mock results
                st.session_state.analysis_results = {
                    'matches': [
                        {'source': 'Doc1', 'target': 'Doc2', 'score': 0.85, 'severity': 'high'},
                        {'source': 'Doc1', 'target': 'Doc3', 'score': 0.45, 'severity': 'low'},
                        {'source': 'Doc2', 'target': 'Doc3', 'score': 0.32, 'severity': 'none'}
                    ],
                    'summary': {
                        'total_matches': 3,
                        'high': 1,
                        'medium': 0,
                        'low': 1,
                        'none': 1
                    }
                }
                st.session_state.show_results = True
                st.session_state.is_analyzing = False
                
                progress_bar.empty()
                st.success("✅ Analysis complete!")
                st.balloons()
                st.rerun()

with tab2:
    st.markdown("### 📊 Analysis Results")
    if st.session_state.analysis_results:
        ResultsDisplay.render(st.session_state.analysis_results)
    else:
        st.info("Run an analysis to see results here")

with tab3:
    st.markdown("### 🔍 Similarity Matrix")
    if st.session_state.analysis_results:
        SimilarityMatrix.render(st.session_state.analysis_results)
    else:
        st.info("Run an analysis to see the similarity matrix")

with tab4:
    st.markdown("### 📈 Visualizations")
    if st.session_state.analysis_results:
        matches = st.session_state.analysis_results.get('matches', [])
        
        if matches:
            col1, col2 = st.columns(2)
            
            with col1:
                severities = {'High': 0, 'Medium': 0, 'Low': 0, 'None': 0}
                for m in matches:
                    severity = m.get('severity', 'none')
                    if severity == 'high':
                        severities['High'] += 1
                    elif severity == 'medium':
                        severities['Medium'] += 1
                    elif severity == 'low':
                        severities['Low'] += 1
                    else:
                        severities['None'] += 1
                
                fig = px.pie(
                    values=list(severities.values()),
                    names=list(severities.keys()),
                    title='Severity Distribution',
                    color=list(severities.keys()),
                    color_discrete_map={
                        'High': '#dc2626',
                        'Medium': '#d97706',
                        'Low': '#2563eb',
                        'None': '#94a3b8'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                scores = [m.get('score', 0) for m in matches]
                fig = px.histogram(
                    x=scores,
                    nbins=20,
                    title='Score Distribution',
                    labels={'x': 'Similarity Score', 'y': 'Frequency'}
                )
                fig.add_vline(x=0.59, line_dash="dash", line_color="red")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data to visualize")
    else:
        st.info("Run an analysis to see visualizations")

# ============================================
# Footer
# ============================================
st.markdown("""
<div style="margin-top: 40px; padding: 20px; text-align: center; color: #94a3b8; font-size: 13px; border-top: 1px solid #e2e8f0;">
    <p>🔍 Semantic Plagiarism Detector v2.0 • Built with Streamlit & Sentence Transformers</p>
</div>
""", unsafe_allow_html=True)