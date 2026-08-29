import streamlit as st
from app.session_keys import SessionKeys
from app.theme import get_theme_config

def render_sidebar():
    """Render the sidebar settings including theme selection options."""
    st.sidebar.title("Settings")
    
    # Theme selection option in Streamlit sidebar settings
    selected_theme = st.sidebar.selectbox(
        "Theme Mode",
        options=["Light", "Dark", "Accessible High Contrast"],
        key=SessionKeys.THEME if hasattr(SessionKeys, "THEME") else "theme"
    )
    
    theme_config = get_theme_config(selected_theme)
    
    # Apply custom CSS variables or Streamlit configuration based on theme
    st.sidebar.markdown(
        f"""
        <style>
            .stApp {{
                background-color: {theme_config['background']};
                color: {theme_config['text']};
            }}
        </style>
        """,
        unsafe_allow_html=True
    )
    
    return selected_theme
