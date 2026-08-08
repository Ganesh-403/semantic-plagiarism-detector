"""
tests/visualization/test_font_scale.py
--------------------------------------
Unit tests for global chart font scale configuration (Issue #1794).

Validates that the font_scale parameter correctly multiplies base font sizes
in Plotly figure layouts across different visualization modules.
"""

import pandas as pd
import networkx as nx
import pytest
from unittest.mock import MagicMock, patch

from src.visualization.network_graph import (
    render_network_plotly,
    plot_similarity_network,
)


class TestNetworkFontScale:
    """Test suite for font scaling in network graph visualizations."""

    @pytest.fixture
    def mock_network_data(self):
        """Provide a minimal mock network data dictionary."""
        return {
            "shapes": [],
            "edge_hover_trace": MagicMock(),
            "node_trace": MagicMock(textfont=MagicMock(size=10)),
            "graph": nx.Graph(),
            "pos": {},
            "tag_color_map": {},
            "document_tags": {},
        }

    def test_default_font_scale_is_one(self, mock_network_data):
        """Verify default font_scale (1.0) produces base font sizes."""
        fig = render_network_plotly(mock_network_data)

        # Base title size is 16
        assert fig.layout.title.font.size == 16
        # Base hover font size is 12
        assert fig.layout.font.size == 12

    def test_font_scale_doubles_sizes(self, mock_network_data):
        """Verify font_scale=2.0 doubles all font sizes."""
        fig = render_network_plotly(mock_network_data, font_scale=2.0)

        assert fig.layout.title.font.size == 32  # 16 * 2
        assert fig.layout.font.size == 24  # 12 * 2

    def test_font_scale_minimum_enforcement(self, mock_network_data):
        """Verify font_scale < 0.5 is clamped to 0.5 to prevent unreadable text."""
        fig = render_network_plotly(mock_network_data, font_scale=0.1)

        # Should be clamped to 0.5 scale
        assert fig.layout.title.font.size == 8  # 16 * 0.5
        assert fig.layout.font.size == 6  # 12 * 0.5

    def test_font_scale_fractional_values(self, mock_network_data):
        """Verify fractional font_scale values are handled and rounded to int."""
        fig = render_network_plotly(mock_network_data, font_scale=1.5)

        assert fig.layout.title.font.size == 24  # 16 * 1.5 = 24
        assert fig.layout.font.size == 18  # 12 * 1.5 = 18

    def test_plot_similarity_network_passes_font_scale(self):
        """Verify plot_similarity_network correctly passes font_scale to renderer."""
        df = pd.DataFrame(
            [[1.0, 0.8], [0.8, 1.0]], index=["doc1", "doc2"], columns=["doc1", "doc2"]
        )

        with patch(
            "src.visualization.network_graph.render_network_plotly"
        ) as mock_render:
            mock_render.return_value = MagicMock()

            plot_similarity_network(df, font_scale=1.75)

            mock_render.assert_called_once()
            call_kwargs = mock_render.call_args[1]
            assert call_kwargs["font_scale"] == 1.75

    def test_invalid_font_scale_type_coercion(self, mock_network_data):
        """Verify string representations of numbers are coerced to float."""
        fig = render_network_plotly(mock_network_data, font_scale="2.0")
        assert fig.layout.title.font.size == 32

    def test_node_trace_textfont_scaled(self, mock_network_data):
        """Verify the node trace textfont size is updated when scaled."""
        # Mock node_trace with updatable textfont
        mock_node_trace = MagicMock()
        mock_node_trace.textfont = MagicMock()
        mock_node_trace.textfont.size = 10

        mock_network_data["node_trace"] = mock_node_trace

        render_network_plotly(mock_network_data, font_scale=2.0)

        # Base node text size is 10. 10 * 2.0 = 20
        assert mock_node_trace.textfont.size == 20
