"""
Custom theme for the Semantic Plagiarism Detector
"""

import streamlit as st


def apply_theme():
    """Apply custom CSS theme to the application."""
    
    st.markdown("""
    <style>
        /* Main theme */
        .stApp {
            background-color: #f8fafc;
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #1e293b !important;
        }
        
        /* Cards */
        .stCard {
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
            border: 1px solid #e2e8f0;
        }
        
        /* Buttons */
        .stButton button {
            background: linear-gradient(135deg, #3b82f6, #2563eb);
            color: white;
            border: none;
            padding: 8px 24px;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        }
        
        /* Sidebar */
        .css-1d391kg {
            background: white;
        }
        
        /* Metrics */
        .stMetric {
            background: white;
            padding: 12px;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 8px 16px;
            border-radius: 8px;
            font-weight: 500;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: #3b82f6;
            color: white;
        }
        
        /* Dataframes */
        .stDataFrame {
            border-radius: 8px;
            overflow: hidden;
        }
        
        /* Progress bar */
        .stProgress > div > div {
            background: linear-gradient(90deg, #3b82f6, #22c55e);
        }
        
        /* Success/Info/Warning/Error */
        .stAlert {
            border-radius: 8px;
            border: 1px solid #e2e8f0;
        }
    </style>
    """, unsafe_allow_html=True)