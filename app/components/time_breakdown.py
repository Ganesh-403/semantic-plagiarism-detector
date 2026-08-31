from typing import List

import plotly.graph_objects as go
import streamlit as st

from src.utils.processing_time import StageTiming


def render_time_breakdown(stage_timings: list[StageTiming]) -> None:
    """Render a Plotly bar chart of stage timings.
    If no timings are provided, displays an informational message.
    """
    if not stage_timings:
        st.info("No timing data available – run the pipeline first.")
        return
    names = [t.stage_name for t in stage_timings]
    values = [t.duration_seconds for t in stage_timings]
    fig = go.Figure(
        data=[
            go.Bar(
                x=names,
                y=values,
                text=[f"{v:.2f}s" for v in values],
                textposition="auto",
            )
        ]
    )
    fig.update_layout(
        title="Pipeline Stage Timing (seconds)",
        xaxis_title="Stage",
        yaxis_title="Duration (s)",
    )
    st.plotly_chart(fig, use_container_width=True)
