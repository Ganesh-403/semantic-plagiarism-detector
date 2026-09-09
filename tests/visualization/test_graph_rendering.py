"""Real Plotly figures and Streamlit graph rendering stay compatible with Plotly 6."""
import pytest
from streamlit.testing.v1 import AppTest
from src.visualization.citation_graph import plot_citation_network


@pytest.mark.parametrize('theme', [None, {'background': '#101010', 'ink': '#eeeeee'}])
def test_citation_network_contains_both_documents_and_shared_sources(theme):
    figure = plot_citation_network('essay-a', 'essay-b', [{'title': 'Research paper'}, {}], theme)
    assert len(figure.data) == 2
    assert list(figure.data[1].text) == ['essay-a', 'essay-b', 'Research paper', 'Unknown']
    assert len(figure.data[0].x) == 12
    assert figure.layout.title.text == 'Shared Bibliography Network (2 citations)'
    assert figure.layout.title.font.size == 16
    assert 'Research paper' in figure.to_json()


def test_empty_citation_network_explains_missing_data():
    figure = plot_citation_network('a', 'b', [])
    assert not figure.data
    assert 'No shared citations' in figure.layout.annotations[0].text


@pytest.mark.parametrize('populated', [False, True])
def test_interactive_graph_renders_students_documents_and_empty_state(populated):
    source = '''
import networkx as nx
from app.components.graph_ui import render_interactive_graph
network = nx.Graph()
if POPULATED:
    network.add_node('student', labels={'Student'}, name='Alex')
    network.add_node('essay', labels={'Document'}, title='Biology essay')
    network.add_edge('student', 'essay')
render_interactive_graph(network)
'''.replace('POPULATED', repr(populated))
    at = AppTest.from_string(source).run()
    assert not at.exception
    assert len(at.get('plotly_chart')) == int(populated)
    if not populated:
        assert any('No data' in item.value for item in at.info)
