# MIT License
#
# Copyright (c) 2026 Ganesh Kambli
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
