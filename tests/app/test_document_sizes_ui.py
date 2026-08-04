import plotly.graph_objects as go

from src.visualization.analytics import plot_document_sizes


def test_plot_document_sizes_empty():
    """Verify plot_document_sizes handles empty word counts dictionary gracefully."""
    fig = plot_document_sizes({})
    assert isinstance(fig, go.Figure)
    # Check that the empty state annotation is present
    assert any(
        "No documents currently in the database" in (anno.text or "")
        for anno in fig.layout.annotations
    )


def test_plot_document_sizes_with_data():
    """Verify plot_document_sizes creates a bar chart when word counts are provided."""
    word_counts = {
        "doc1.txt": 100,
        "doc2.txt": 50,
        "doc_very_long_name_that_needs_to_be_truncated.txt": 500,
    }
    fig = plot_document_sizes(word_counts)
    assert isinstance(fig, go.Figure)

    # Check chart details
    assert fig.layout.title.text == "Document Word Counts"

    # Check data count and truncated labels
    # We should have 1 trace which is the bar chart
    assert len(fig.data) == 1
    trace = fig.data[0]
    assert trace.type == "bar"

    # Verify x values has truncated version of the long name
    x_values = list(trace.x)
    assert "doc1.txt" in x_values
    assert "doc2.txt" in x_values
    assert any("..." in label for label in x_values)
