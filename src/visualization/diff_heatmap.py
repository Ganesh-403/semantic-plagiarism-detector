"""
src/visualization/diff_heatmap.py
---------------------------------
Plotly visualizations for document version diffs.

Generates heatmaps and retention charts showing text evolution,
additions, deletions, and potential AI-injection spikes between
draft versions.
"""

import plotly.graph_objects as go
from typing import List, Dict, Any
from src.core.document_versioning import DiffBlock, DiffOp


def generate_diff_heatmap(
    blocks: list[DiffBlock],
    title: str = "Document Evolution Heatmap"
) -> go.Figure:
    """Generate a Plotly heatmap visualizing text changes between versions.
    
    The X-axis represents the character position in the text, and the
    color indicates the type of change (Equal, Insert, Delete, Replace).
    
    Args:
        blocks: List of DiffBlock objects from the versioning engine.
        title: Title for the Plotly figure.
        
    Returns:
        A configured Plotly Figure object.
    """
    # Map operations to colors
    color_map = {
        DiffOp.EQUAL: "#e2e8f0",   # Light gray
        DiffOp.INSERT: "#bbf7d0",  # Light green
        DiffOp.DELETE: "#fecaca",  # Light red
        DiffOp.REPLACE: "#fef08a"  # Light yellow
    }
    
    # Build data for the heatmap
    # We'll create a 1D heatmap (single row) where each cell represents a block
    z_data = []
    x_labels = []
    hover_texts = []
    
    for i, block in enumerate(blocks):
        # Assign a numeric value for the colorscale
        if block.op == DiffOp.EQUAL: val = 0
        elif block.op == DiffOp.INSERT: val = 1
        elif block.op == DiffOp.DELETE: val = 2
        else: val = 3
            
        z_data.append([val])
        x_labels.append(f"Block {i+1}")
        
        hover_text = (
            f"<b>{block.op.value.upper()}</b><br>"
            f"v1: [{block.start_v1}:{block.end_v1}]<br>"
            f"v2: [{block.start_v2}:{block.end_v2}]<br>"
            f"<b>v1 Text:</b> {block.text_v1[:50]}...<br>"
            f"<b>v2 Text:</b> {block.text_v2[:50]}..."
        )
        hover_texts.append(hover_text)
        
    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=x_labels,
        y=["Changes"],
        colorscale=[
            [0.0, color_map[DiffOp.EQUAL]],
            [0.33, color_map[DiffOp.INSERT]],
            [0.66, color_map[DiffOp.DELETE]],
            [1.0, color_map[DiffOp.REPLACE]]
        ],
        showscale=False,
        hovertext=hover_texts,
        hoverinfo="text"
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Text Blocks",
        yaxis_title="",
        height=200,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig


def generate_retention_chart(
    lineage: list[dict[str, Any]],
    retention_scores: list[float]
) -> go.Figure:
    """Generate a line chart showing text retention across multiple drafts.
    
    Args:
        lineage: List of version records from the DB.
        retention_scores: List of retention scores corresponding to each version.
        
    Returns:
        A configured Plotly Figure object.
    """
    versions = [f"v{v['version_number']}" for v in lineage]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=versions,
        y=retention_scores,
        mode="lines+markers",
        name="Text Retention",
        line=dict(color="#2563eb", width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title="Text Retention Across Drafts",
        xaxis_title="Draft Version",
        yaxis_title="Retention Score (0.0 - 1.0)",
        yaxis=dict(range=[0, 1.05]),
        height=400
    )
    
    return fig

import plotly.graph_objects as go

def generate_evolution_heatmap(diff_tokens: list[dict], block_size: int = 50) -> go.Figure:
    """
    Aggregates granular tokens into chunk blocks to graph an evolution map layout.
    Values represent modification density (Deletions/Additions = 1, Unchanged = 0).
    """
    chunks = [diff_tokens[i:i + block_size] for i in range(0, len(diff_tokens), block_size)]
    
    density_scores = []
    chunk_labels = []
    
    for idx, chunk in enumerate(chunks):
        changes = sum(1 for t in chunk if t["action"] in ["added", "deleted"])
        score = (changes / len(chunk)) * 100 if len(chunk) > 0 else 0
        density_scores.append(score)
        chunk_labels.append(f"Block {idx + 1}")

    # Build Plotly visual layout matrix maps
    fig = go.Figure(data=go.Heatmap(
        z=[density_scores],
        x=chunk_labels,
        y=["Modification Density"],
        colorscale="YlOrRd",
        zmin=0,
        zmax=100,
        colorbar=dict(title="Change %", titleside="top")
    ))

    fig.update_layout(
        title="Document Evolution & Edit Density Map",
        xaxis_title="Sequential Document Text Blocks",
        yaxis_title="",
        height=250,
        margin=dict(l=40, r=40, t=60, b=40),
        template="plotly_white"
    )
    
    return fig
