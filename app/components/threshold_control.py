"""Threshold presets that remain synchronized with manual slider changes."""
import streamlit as st

PRESETS = {"Strict (0.80)": .80, "Balanced (0.59)": .59, "Lenient (0.45)": .45, "Custom": None}


def render_threshold_control(default=.59, on_change=None):
    def preset_changed():
        value = PRESETS[st.session_state["threshold_preset_radio"]]
        if value is not None:
            st.session_state["threshold_slider"] = value
            if on_change:
                on_change()

    def slider_changed():
        st.session_state["threshold_preset_radio"] = "Custom"
        if on_change:
            on_change()

    current = st.session_state.get("threshold_slider", default)
    if "threshold_preset_radio" not in st.session_state:
        st.session_state["threshold_preset_radio"] = next((label for label, value in PRESETS.items() if value is not None and abs(current-value) < .001), "Custom")
    st.markdown("### 🎯 Threshold Presets")
    st.radio("Select Evaluation Standard:", options=list(PRESETS), key="threshold_preset_radio", horizontal=True, on_change=preset_changed)
    return st.slider("Plagiarism Threshold (Hybrid)", .10, .99, value=current, step=.01, key="threshold_slider", on_change=slider_changed, help="Combined lexical and semantic score needed to flag a document pair. Default: 0.59.")
