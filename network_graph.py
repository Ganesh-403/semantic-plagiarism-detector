"""
network_graph.py - Network cluster graph visualization with HTML export functionality
"""

import plotly.graph_objects as go
from typing import Optional, Dict, Any, List, Tuple
import json
import os
import tempfile
import webbrowser


class NetworkGraph:
    """
    Class for creating and managing network cluster visualizations with export capabilities.
    """
    
    def __init__(self):
        """Initialize the NetworkGraph with empty figure."""
        self.figure = None
        self.node_data = None
        self.edge_data = None
    
    def create_network_graph(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        title: str = "Network Cluster Graph",
        node_size: int = 10,
        edge_width: int = 1
    ) -> go.Figure:
        """
        Create a network graph visualization from nodes and edges.
        
        Args:
            nodes: List of node dictionaries with 'id', 'x', 'y', 'label', 'cluster' keys
            edges: List of edge dictionaries with 'source', 'target', 'weight' keys
            title: Graph title
            node_size: Size of node markers
            edge_width: Width of edge lines
        
        Returns:
            go.Figure: Plotly figure object
        """
        self.node_data = nodes
        self.edge_data = edges
        
        # Separate node coordinates and labels
        node_x = [node.get('x', 0) for node in nodes]
        node_y = [node.get('y', 0) for node in nodes]
        node_labels = [node.get('label', node.get('id', '')) for node in nodes]
        
        # Get cluster information for coloring
        clusters = [node.get('cluster', 0) for node in nodes]
        
        # Create edge traces (lines between nodes)
        edge_traces = []
        for edge in edges:
            source_id = edge['source']
            target_id = edge['target']
            
            # Find source and target node coordinates
            source_node = next((n for n in nodes if n['id'] == source_id), None)
            target_node = next((n for n in nodes if n['id'] == target_id), None)
            
            if source_node and target_node:
                edge_trace = go.Scatter(
                    x=[source_node['x'], target_node['x']],
                    y=[source_node['y'], target_node['y']],
                    line=dict(
                        width=edge.get('weight', edge_width),
                        color='#888'
                    ),
                    hoverinfo='none',
                    mode='lines',
                    showlegend=False
                )
                edge_traces.append(edge_trace)
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=node_labels,
            textposition="top center",
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                size=node_size,
                color=clusters,
                colorbar=dict(
                    thickness=15,
                    title='Cluster',
                    xanchor='left',
                    titleside='right'
                ),
                line=dict(width=2, color='white')
            ),
            hovertext=[f"Node: {label}<br>Cluster: {cluster}" 
                      for label, cluster in zip(node_labels, clusters)]
        )
        
        # Create figure
        self.figure = go.Figure(data=edge_traces + [node_trace])
        
        # Update layout
        self.figure.update_layout(
            title=dict(
                text=title,
                font=dict(size=16)
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=40, l=40, r=40, t=60),
            annotations=[
                dict(
                    text="Interactive Network Graph - Click and drag to explore",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=-0.05,
                    font=dict(size=12)
                )
            ],
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                title=''
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                title=''
            ),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return self.figure
    
    def export_network_html(
        self,
        figure: Optional[go.Figure] = None,
        include_plotlyjs: str = 'cdn',
        full_html: bool = True,
        config: Optional[dict[str, Any]] = None,
        filename: Optional[str] = None,
        auto_open: bool = False,
        **kwargs
    ) -> str:
        """
        Export the network graph as an interactive HTML string.
        
        Args:
            figure: Plotly figure object. If None, uses self.figure
            include_plotlyjs: How to include Plotly.js
                - 'cdn': Include from CDN (default)
                - 'directory': Include from local directory
                - 'require': Use require.js
                - False/None: Don't include
            full_html: Whether to generate a complete HTML document
            config: Plotly configuration options for the viewer
            filename: If provided, saves HTML to this file path
            auto_open: If True and filename provided, opens in browser
            **kwargs: Additional arguments passed to fig.to_html()
        
        Returns:
            str: HTML string containing the interactive network visualization
        
        Raises:
            ValueError: If no figure is available
        
        Examples:
            >>> graph = NetworkGraph()
            >>> graph.create_network_graph(nodes, edges)
            >>> html_string = graph.export_network_html()
            >>> # Save to file
            >>> graph.export_network_html(filename='network.html')
        """
        # Use provided figure or fallback to self.figure
        fig = figure or self.figure
        
        if fig is None:
            raise ValueError(
                "No figure available. Create a network graph first using "
                "create_network_graph() or provide a figure parameter."
            )
        
        # Default config for interactive features
        default_config = {
            'scrollZoom': True,
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['toImage'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'network_graph',
                'scale': 2
            }
        }
        
        if config:
            default_config.update(config)
        
        # Generate HTML
        html_string = fig.to_html(
            include_plotlyjs=include_plotlyjs,
            full_html=full_html,
            config=default_config,
            **kwargs
        )
        
        # Save to file if filename provided
        if filename:
            self._save_html_to_file(html_string, filename, auto_open)
        
        return html_string
    
    def _save_html_to_file(self, html_string: str, filename: str, auto_open: bool = False):
        """
        Save HTML string to a file.
        
        Args:
            html_string: HTML content to save
            filename: Path to save the HTML file
            auto_open: Whether to open in browser after saving
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        
        # Write HTML to file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_string)
        
        print(f"✅ Network graph exported to: {filename}")
        
        # Open in browser if requested
        if auto_open:
            webbrowser.open(f'file://{os.path.abspath(filename)}')
    
    def get_figure(self) -> Optional[go.Figure]:
        """Get the current figure."""
        return self.figure
    
    def set_figure(self, figure: go.Figure):
        """Set the current figure."""
        self.figure = figure


# ===================== UNIT TESTS =====================

import unittest
from unittest.mock import patch, MagicMock
import tempfile


class TestNetworkGraph(unittest.TestCase):
    """Unit tests for NetworkGraph class."""
    
    def setUp(self):
        """Set up test data."""
        self.sample_nodes = [
            {'id': 'A', 'x': 0, 'y': 0, 'label': 'Node A', 'cluster': 0},
            {'id': 'B', 'x': 1, 'y': 1, 'label': 'Node B', 'cluster': 1},
            {'id': 'C', 'x': 2, 'y': 0, 'label': 'Node C', 'cluster': 0},
            {'id': 'D', 'x': 1, 'y': -1, 'label': 'Node D', 'cluster': 1},
        ]
        self.sample_edges = [
            {'source': 'A', 'target': 'B', 'weight': 1},
            {'source': 'B', 'target': 'C', 'weight': 2},
            {'source': 'C', 'target': 'D', 'weight': 1},
            {'source': 'D', 'target': 'A', 'weight': 1},
        ]
        self.graph = NetworkGraph()
        self.graph.create_network_graph(self.sample_nodes, self.sample_edges)
    
    def test_create_network_graph(self):
        """Test network graph creation."""
        self.assertIsNotNone(self.graph.figure)
        self.assertIsInstance(self.graph.figure, go.Figure)
        self.assertEqual(len(self.graph.figure.data), 5)  # 4 edges + 1 node trace
    
    def test_export_network_html_returns_string(self):
        """Test that export_network_html returns an HTML string."""
        html_string = self.graph.export_network_html()
        
        # Should return a string
        self.assertIsInstance(html_string, str)
        
        # Should contain HTML elements
        self.assertIn('<!DOCTYPE html>', html_string)
        self.assertIn('<html', html_string)
        self.assertIn('Plotly', html_string)
        self.assertIn('</html>', html_string)
    
    def test_export_network_html_without_full_html(self):
        """Test export with full_html=False."""
        html_string = self.graph.export_network_html(full_html=False)
        
        self.assertIsInstance(html_string, str)
        # Should not have full HTML wrapper
        self.assertNotIn('<!DOCTYPE html>', html_string)
    
    def test_export_network_html_with_figure_parameter(self):
        """Test export with explicit figure parameter."""
        new_figure = go.Figure()
        new_figure.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 2, 3]))
        
        html_string = self.graph.export_network_html(figure=new_figure)
        
        self.assertIsInstance(html_string, str)
        self.assertIn('Plotly', html_string)
    
    def test_export_network_html_no_figure_error(self):
        """Test that error is raised when no figure is available."""
        empty_graph = NetworkGraph()
        
        with self.assertRaises(ValueError) as context:
            empty_graph.export_network_html()
        
        self.assertIn("No figure available", str(context.exception))
    
    def test_export_network_html_saves_to_file(self):
        """Test saving HTML to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, 'test_network.html')
            
            html_string = self.graph.export_network_html(filename=filename)
            
            # Check that file was created
            self.assertTrue(os.path.exists(filename))
            
            # Check file content
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.assertEqual(content, html_string)
            self.assertIn('Plotly', content)
    
    @patch('webbrowser.open')
    def test_export_network_html_auto_open(self, mock_webbrowser):
        """Test auto-open functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, 'test_network.html')
            
            self.graph.export_network_html(filename=filename, auto_open=True)
            
            # Verify webbrowser.open was called
            mock_webbrowser.assert_called_once()
            call_args = mock_webbrowser.call_args[0][0]
            self.assertIn(filename, call_args)
    
    def test_export_network_html_with_config(self):
        """Test export with custom config."""
        custom_config = {
            'scrollZoom': False,
            'displayModeBar': False
        }
        
        html_string = self.graph.export_network_html(config=custom_config)
        
        self.assertIsInstance(html_string, str)
        # Config should be included in the HTML
        self.assertIn('"scrollZoom":false', html_string.lower())
    
    def test_export_network_html_include_plotlyjs_options(self):
        """Test different include_plotlyjs options."""
        # Test with 'cdn' (default)
        html_cdn = self.graph.export_network_html(include_plotlyjs='cdn')
        self.assertIn('plotly.js', html_cdn)
        self.assertIn('https://cdn.plot.ly', html_cdn)
        
        # Test with False (don't include)
        html_no_plotly = self.graph.export_network_html(include_plotlyjs=False)
        self.assertNotIn('plotly.js', html_no_plotly)
    
    def test_export_network_html_with_kwargs(self):
        """Test passing additional kwargs to to_html."""
        html_string = self.graph.export_network_html(
            include_plotlyjs='cdn',
            full_html=True,
            default_width='800px',
            default_height='600px'
        )
        
        self.assertIsInstance(html_string, str)
        self.assertIn('Plotly', html_string)
    
    def test_multiple_exports(self):
        """Test exporting multiple times."""
        html1 = self.graph.export_network_html()
        html2 = self.graph.export_network_html()
        
        self.assertIsInstance(html1, str)
        self.assertIsInstance(html2, str)
        self.assertEqual(html1, html2)  # Should be identical


class TestNetworkGraphEdgeCases(unittest.TestCase):
    """Test edge cases for NetworkGraph."""
    
    def test_empty_nodes_and_edges(self):
        """Test graph with empty nodes and edges."""
        graph = NetworkGraph()
        graph.create_network_graph([], [])
        
        html_string = graph.export_network_html()
        
        self.assertIsInstance(html_string, str)
        self.assertIn('Plotly', html_string)
    
    def test_single_node(self):
        """Test graph with single node."""
        graph = NetworkGraph()
        nodes = [{'id': 'A', 'x': 0, 'y': 0, 'label': 'Node A', 'cluster': 0}]
        graph.create_network_graph(nodes, [])
        
        html_string = graph.export_network_html()
        
        self.assertIsInstance(html_string, str)
        self.assertIn('Node A', html_string)
    
    def test_missing_node_attributes(self):
        """Test nodes with missing attributes."""
        graph = NetworkGraph()
        nodes = [{'id': 'A'}]  # Missing x, y, label, cluster
        graph.create_network_graph(nodes, [])
        
        html_string = graph.export_network_html()
        
        self.assertIsInstance(html_string, str)
        # Should default x and y to 0
        self.assertIn('Plotly', html_string)


# ===================== INTEGRATION TEST =====================

class TestNetworkGraphIntegration(unittest.TestCase):
    """Integration tests for full workflow."""
    
    def test_full_workflow(self):
        """Test complete workflow from creation to export."""
        # 1. Create graph
        graph = NetworkGraph()
        
        nodes = [
            {'id': '1', 'x': 0, 'y': 0, 'label': 'Server', 'cluster': 0},
            {'id': '2', 'x': 1, 'y': 0, 'label': 'Database', 'cluster': 0},
            {'id': '3', 'x': 0.5, 'y': 1, 'label': 'API', 'cluster': 1},
            {'id': '4', 'x': 1.5, 'y': 1, 'label': 'Client', 'cluster': 1},
        ]
        edges = [
            {'source': '1', 'target': '2', 'weight': 3},
            {'source': '1', 'target': '3', 'weight': 2},
            {'source': '2', 'target': '4', 'weight': 1},
            {'source': '3', 'target': '4', 'weight': 2},
        ]
        
        figure = graph.create_network_graph(
            nodes, edges,
            title="System Architecture",
            node_size=15,
            edge_width=2
        )
        
        self.assertIsInstance(figure, go.Figure)
        
        # 2. Export to HTML
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, 'system_architecture.html')
            
            html_string = graph.export_network_html(
                filename=filename,
                auto_open=False
            )
            
            # 3. Verify output
            self.assertIsInstance(html_string, str)
            self.assertIn('System Architecture', html_string)
            self.assertIn('Server', html_string)
            self.assertIn('Database', html_string)
            self.assertIn('API', html_string)
            self.assertIn('Client', html_string)
            
            # 4. Verify file content
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.assertEqual(content, html_string)
            self.assertIn('Plotly', content)
            self.assertIn('modeBarButtonsToRemove', content)


# ===================== RUN TESTS =====================

if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)
