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

from .analytics import (
    plot_hierarchical_dendrogram,
    plot_high_severity_trends,
    plot_most_plagiarized_documents,
    plot_similarity_percentiles,
)
from .heatmap import (
    filter_heatmap_by_class_tag,
    plot_chunk_similarity_comparison,
    plot_differential_heatmap,
    plot_differential_heatmap_matplotlib,
    plot_document_similarity_heatmap,
    plot_similarity_heatmap,
    plot_similarity_heatmap_plotly,
)
from .network_graph import (
    build_network_data,
    calculate_force_directed_layout,
    export_graph_to_csv,
    export_graph_to_gexf,
    export_network_to_csv_bytes,
    export_network_to_gexf_bytes,
    plot_plagiarism_network_graph,
    plot_similarity_network,
    render_network_plotly,
)

__all__ = [
    "filter_heatmap_by_class_tag",
    "plot_similarity_heatmap",
    "plot_similarity_heatmap_plotly",
    "plot_document_similarity_heatmap",
    "plot_differential_heatmap",
    "plot_differential_heatmap_matplotlib",
    "plot_chunk_similarity_comparison",
    "build_network_data",
    "calculate_force_directed_layout",
    "export_graph_to_csv",
    "export_graph_to_gexf",
    "export_network_to_csv_bytes",
    "export_network_to_gexf_bytes",
    "render_network_plotly",
    "plot_similarity_network",
    "plot_plagiarism_network_graph",
    "plot_high_severity_trends",
    "plot_most_plagiarized_documents",
    "plot_hierarchical_dendrogram",
    "plot_similarity_percentiles",
]
