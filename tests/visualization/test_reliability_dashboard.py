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

"""
tests/visualization/test_reliability_dashboard.py
-------------------------------------------------
Tests for the inter-rater reliability visualizations (issue #3785).

`generate_reviewer_agreement_heatmap` was defined twice in the module. A second
copy had been appended below `generate_calibration_weight_chart`, carrying its
own duplicated imports and a stray path comment, and it shadowed the real
implementation above it.

The copy that won was broken in two ways: it passed `color_continuousScale` to
`px.imshow` (the keyword is `color_continuous_scale`), so every call raised
TypeError, and it dropped the `title` parameter the documented version accepts.
The heatmap could not be rendered at all.

These tests pin the surviving definition's identity and signature so a paste
like that cannot quietly win again.
"""

import inspect

import plotly.graph_objects as go
import pytest

from src.visualization import reliability_dashboard
from src.visualization.reliability_dashboard import (
    generate_calibration_weight_chart,
    generate_reviewer_agreement_heatmap,
)

KAPPA_MATRIX = [
    [1.00, 0.72, 0.31],
    [0.72, 1.00, 0.55],
    [0.31, 0.55, 1.00],
]
REVIEWERS = ["dr_ada", "dr_grace", "dr_alan"]


# ── the regression itself ──────────────────────────────────────────────────────


def test_heatmap_is_defined_exactly_once():
    """Two defs in one module means the later one silently wins."""
    source = inspect.getsource(reliability_dashboard)

    assert source.count("def generate_reviewer_agreement_heatmap(") == 1


def test_heatmap_keeps_its_documented_signature():
    """The shadowing copy dropped `title` and renamed both positionals."""
    parameters = inspect.signature(generate_reviewer_agreement_heatmap).parameters

    assert list(parameters) == ["reviewer_matrix", "reviewer_names", "title"]
    assert parameters["title"].default == "Reviewer Agreement Matrix"


def test_heatmap_does_not_raise():
    """The shadowing copy raised TypeError on the `color_continuousScale` typo."""
    figure = generate_reviewer_agreement_heatmap(KAPPA_MATRIX, REVIEWERS)

    assert isinstance(figure, go.Figure)


def test_module_imports_plotly_express_only_if_it_uses_it():
    """The duplicate block dragged in `plotly.express` and a second `go`."""
    source = inspect.getsource(reliability_dashboard)

    assert source.count("import plotly.graph_objects as go") == 1
    if "px." not in source:
        assert "import plotly.express as px" not in source


# ── heatmap contents ───────────────────────────────────────────────────────────


def test_heatmap_carries_the_kappa_matrix():
    figure = generate_reviewer_agreement_heatmap(KAPPA_MATRIX, REVIEWERS)
    trace = figure.data[0]

    assert [list(row) for row in trace.z] == KAPPA_MATRIX


def test_heatmap_labels_both_axes_with_the_reviewers():
    figure = generate_reviewer_agreement_heatmap(KAPPA_MATRIX, REVIEWERS)
    trace = figure.data[0]

    assert list(trace.x) == REVIEWERS
    assert list(trace.y) == REVIEWERS


def test_heatmap_uses_the_default_title():
    figure = generate_reviewer_agreement_heatmap(KAPPA_MATRIX, REVIEWERS)

    assert figure.layout.title.text == "Reviewer Agreement Matrix"


def test_heatmap_honours_a_custom_title():
    """The shadowing copy rejected this keyword outright."""
    figure = generate_reviewer_agreement_heatmap(
        KAPPA_MATRIX, REVIEWERS, title="Committee B — Semester 2"
    )

    assert figure.layout.title.text == "Committee B — Semester 2"


def test_heatmap_scale_spans_the_full_kappa_range():
    """Kappa runs from -1 to 1; clamping at 0 would hide disagreement."""
    trace = generate_reviewer_agreement_heatmap(KAPPA_MATRIX, REVIEWERS).data[0]

    assert trace.zmin == -1.0
    assert trace.zmax == 1.0


def test_heatmap_annotates_each_cell():
    trace = generate_reviewer_agreement_heatmap(KAPPA_MATRIX, REVIEWERS).data[0]

    assert [list(row) for row in trace.text] == [
        ["1.00", "0.72", "0.31"],
        ["0.72", "1.00", "0.55"],
        ["0.31", "0.55", "1.00"],
    ]


def test_heatmap_handles_negative_kappa():
    """Systematic disagreement is a real, renderable result."""
    matrix = [[1.0, -0.4], [-0.4, 1.0]]

    trace = generate_reviewer_agreement_heatmap(matrix, ["a", "b"]).data[0]

    assert [list(row) for row in trace.z] == matrix


def test_heatmap_handles_a_single_reviewer():
    figure = generate_reviewer_agreement_heatmap([[1.0]], ["solo"])

    assert list(figure.data[0].x) == ["solo"]


@pytest.mark.parametrize("size", [2, 5, 10])
def test_heatmap_scales_to_committee_size(size):
    matrix = [[1.0 if i == j else 0.5 for j in range(size)] for i in range(size)]
    names = [f"reviewer_{i}" for i in range(size)]

    trace = generate_reviewer_agreement_heatmap(matrix, names).data[0]

    assert len(trace.z) == size
    assert len(trace.x) == size


# ── calibration weight chart ───────────────────────────────────────────────────


def test_calibration_chart_returns_a_figure():
    figure = generate_calibration_weight_chart({"dr_ada": 0.9, "dr_alan": 0.4})

    assert isinstance(figure, go.Figure)


def test_calibration_chart_preserves_reviewer_order():
    weights = {"dr_ada": 0.91, "dr_grace": 0.62, "dr_alan": 0.33}

    trace = generate_calibration_weight_chart(weights).data[0]

    assert list(trace.x) == list(weights)
    assert list(trace.y) == list(weights.values())


def test_calibration_chart_colours_by_trust_band():
    """Red below 0.5, amber below 0.8, green at or above."""
    trace = generate_calibration_weight_chart(
        {"low": 0.2, "mid": 0.65, "high": 0.95}
    ).data[0]

    assert list(trace.marker.color) == ["#ef4444", "#f59e0b", "#10b981"]


def test_calibration_chart_band_boundaries_are_inclusive_upward():
    trace = generate_calibration_weight_chart({"a": 0.5, "b": 0.8}).data[0]

    assert list(trace.marker.color) == ["#f59e0b", "#10b981"]


def test_calibration_chart_honours_a_custom_title():
    figure = generate_calibration_weight_chart({"dr_ada": 0.9}, title="Weights Q3")

    assert figure.layout.title.text == "Weights Q3"


def test_calibration_chart_handles_no_reviewers():
    figure = generate_calibration_weight_chart({})

    assert len(figure.data[0].x) == 0
