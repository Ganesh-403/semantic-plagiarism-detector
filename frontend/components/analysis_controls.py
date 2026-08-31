"""
Analysis controls component for the Semantic Plagiarism Detector
"""

import streamlit as st
from typing import Dict, Any, Optional


class AnalysisControls:
    """
    Component for analysis method selection and configuration.
    """
    
    METHODS = {
        'hybrid': {
            'label': 'Hybrid',
            'description': 'Combines lexical and semantic analysis for best results',
            'icon': '🔀'
        },
        'lexical': {
            'label': 'Lexical',
            'description': 'Word-level matching using TF-IDF and string metrics',
            'icon': '📝'
        },
        'semantic': {
            'label': 'Semantic',
            'description': 'Meaning-level matching using sentence embeddings',
            'icon': '🧠'
        }
    }
    
    @classmethod
    def render(cls) -> Dict[str, Any]:
        """
        Render the analysis controls component.
        
        Returns:
            Dictionary with selected configuration
        """
        st.markdown("""
        <style>
            .controls-container {
                background: white;
                border-radius: 12px;
                padding: 20px;
                border: 1px solid #e2e8f0;
                margin-bottom: 16px;
            }
            .method-card {
                padding: 12px 16px;
                border-radius: 8px;
                border: 2px solid #e2e8f0;
                cursor: pointer;
                transition: all 0.3s ease;
                margin-bottom: 8px;
            }
            .method-card:hover {
                border-color: #3b82f6;
                background: #eff6ff;
            }
            .method-card.selected {
                border-color: #3b82f6;
                background: #dbeafe;
            }
            .method-icon {
                font-size: 20px;
                margin-right: 8px;
            }
            .method-label {
                font-weight: 600;
                color: #1e293b;
            }
            .method-desc {
                font-size: 13px;
                color: #64748b;
                margin-top: 2px;
            }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 🎯 Analysis Method")
        
        # Method selection
        selected_method = st.radio(
            "Select analysis method",
            options=list(cls.METHODS.keys()),
            format_func=lambda x: f"{cls.METHODS[x]['icon']} {cls.METHODS[x]['label']}",
            key="method_radio",
            horizontal=True
        )
        
        # Description
        st.info(cls.METHODS[selected_method]['description'])
        
        # Advanced settings
        with st.expander("⚙️ Advanced Settings", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                threshold = st.slider(
                    "Similarity Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.59,
                    step=0.01,
                    help="Minimum score to flag as potential plagiarism"
                )
            
            with col2:
                chunk_size = st.number_input(
                    "Chunk Size (words)",
                    min_value=50,
                    max_value=500,
                    value=200,
                    step=50,
                    help="Number of words per chunk for analysis"
                )
        
        return {
            'method': selected_method,
            'threshold': threshold,
            'chunk_size': chunk_size
        }