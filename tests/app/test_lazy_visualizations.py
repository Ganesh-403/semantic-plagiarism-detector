from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest


def test_heatmap_and_network_only_build_when_requested():
    with (
        patch("app.views.heatmap_view.plot_similarity_heatmap", return_value=None) as heatmap,
        patch("app.views.heatmap_view.plot_similarity_network", return_value=None) as network,
    ):
        at = AppTest.from_string("""
import pandas as pd
from app.views.heatmap_view import render_heatmap_view
matrix = pd.DataFrame([[1, .3], [.3, 1]], index=["a", "b"], columns=["a", "b"])
render_heatmap_view(matrix, .7, ["a", "b"])
""").run()
        assert not at.exception
        heatmap.assert_not_called()
        network.assert_not_called()
        at.checkbox(key="load_similarity_heatmap").check().run()
        heatmap.assert_called_once()
        network.assert_not_called()
        at.checkbox(key="load_similarity_heatmap").uncheck()
        at.checkbox(key="load_plagiarism_network").check().run()
        assert not at.exception
        assert heatmap.call_count == 1
        network.assert_called_once()


def test_analytics_charts_only_build_when_requested():
    with (
        patch("app.components.enhanced_dashboard.render_similarity_trend_chart") as trends,
        patch("app.components.enhanced_dashboard.render_document_activity_heatmap") as activity,
        patch("app.components.enhanced_dashboard.render_collusion_ring_dashboard") as patterns,
    ):
        at = AppTest.from_string("""
import pandas as pd
from app.components.enhanced_dashboard import render_enhanced_analytics_tab
render_enhanced_analytics_tab(pd.DataFrame(), [], {})
""").run()
        assert not at.exception
        for chart in (trends, activity, patterns):
            chart.assert_not_called()
        for key in ("trends", "activity", "patterns"):
            at.checkbox(key=f"load_analytics_{key}").check()
        at.run()
        assert not at.exception
        for chart in (trends, activity, patterns):
            chart.assert_called_once()
