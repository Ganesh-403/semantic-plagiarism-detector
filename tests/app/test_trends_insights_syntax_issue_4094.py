"""
tests/app/test_trends_insights_syntax_issue_4094.py
----------------------------------------------------
Regression tests for the two defects that kept ``app/pages/9_Trends_Insights.py``
from parsing (issue #4094).

Defect 1 — the confidence-band trace in ``plot_forecast()`` opened
``pd.concat([`` and closed it with ``)``:

    x=pd.concat([forecast["date"], forecast["date"][::-1]),

which is ``SyntaxError: closing parenthesis ')' does not match opening
parenthesis '['``. The ``y=`` line immediately below shows the correct shape.

Defect 2 — ``plot_top_flagged_pairs()`` passed ``yaxis`` to ``update_layout()``
twice: once as ``yaxis=""`` (a typo for ``yaxis_title=""``, which is what pairs
with the ``xaxis_title`` beside it) and once as ``yaxis=dict(autorange=
"reversed")``, the axis config a horizontal bar chart needs so the top-ranked
pair renders at the top. Python keeps only the second, so the first was
silently dropped — and once Defect 1 was fixed it became a hard
``SyntaxError: keyword argument repeated`` on the same call. Both had to move
together, which is why they share one test module.

Because the page runs ``st.set_page_config()`` at import and calls
``render_trends_insights()`` at the bottom, importing it normally would render
the page and hit the database. The ``page_module`` fixture below loads the
module's imports and function definitions only, so the plotting helpers can be
called directly.
"""

import ast
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

PAGE_PATH = (
    Path(__file__).resolve().parents[2] / "app" / "pages" / "9_Trends_Insights.py"
)

# Modules the page imports that we neither need nor want to execute here:
# streamlit would try to render, and the src.* chain drags in the embedding
# model and its ML dependencies.
STUBBED_MODULES = (
    "streamlit",
    "src",
    "src.core",
    "src.core.app_config",
    "src.db",
    "src.db.incidents",
)


@pytest.fixture(scope="module")
def page_source():
    return PAGE_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def page_tree(page_source):
    """Parsing this at all is the primary regression assert for Defect 1."""
    return ast.parse(page_source, filename=PAGE_PATH.name)


@pytest.fixture(scope="module")
def page_module(page_tree):
    """Execute the page's imports and defs, but not its top-level page calls."""
    saved = {name: sys.modules.get(name) for name in STUBBED_MODULES}
    for name in STUBBED_MODULES:
        sys.modules[name] = MagicMock()

    try:
        body = [
            node
            for node in page_tree.body
            if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call))
        ]
        namespace = {"__name__": "trends_insights_isolated"}
        exec(  # noqa: S102 - deliberately loading a page module without running it
            compile(ast.Module(body=body, type_ignores=[]), PAGE_PATH.name, "exec"),
            namespace,
        )
        yield namespace
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


@pytest.fixture
def incidents_df():
    """A small but realistic incident frame covering several days and pairs."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-04",
                    "2026-01-05",
                ]
            ),
            "severity": ["High", "Low", "High", "Critical", "Medium", "High"],
            "similarity_score": [0.91, 0.62, 0.83, 0.97, 0.71, 0.88],
            "document_a": ["a.txt", "c.txt", "a.txt", "a.txt", "e.txt", "a.txt"],
            "document_b": ["b.txt", "d.txt", "b.txt", "b.txt", "f.txt", "b.txt"],
        }
    )


# ── the page parses ────────────────────────────────────────────────────────────


def test_page_compiles(page_source):
    """The module must compile. This is the whole of Defect 1.

    Before the fix: SyntaxError at line 484.
    """
    compile(page_source, PAGE_PATH.name, "exec")


def test_page_has_no_duplicate_keyword_arguments(page_tree):
    """No call anywhere in the file may repeat a keyword argument.

    Defect 2 was one instance. A repeated keyword is always either a typo or a
    silently discarded setting, so this walks the whole tree rather than
    pinning the single call that happened to be broken.
    """
    duplicates = []
    for node in ast.walk(page_tree):
        if not isinstance(node, ast.Call):
            continue
        seen = set()
        for keyword in node.keywords:
            if keyword.arg is None:  # **kwargs unpacking
                continue
            if keyword.arg in seen:
                duplicates.append((node.lineno, keyword.arg))
            seen.add(keyword.arg)

    assert not duplicates, f"duplicate keyword arguments at {duplicates}"


def test_helper_functions_are_all_defined(page_module):
    """The plotting helpers survived the fix and are callable."""
    for name in (
        "generate_forecast",
        "plot_forecast",
        "plot_top_flagged_pairs",
        "plot_daily_trend",
        "generate_insights",
    ):
        assert callable(page_module[name]), f"{name} is missing or not callable"


# ── Defect 1: the confidence band actually builds ──────────────────────────────


def test_forecast_frame_has_the_columns_the_band_needs(page_module, incidents_df):
    """``generate_forecast`` must supply both bounds the band closes over."""
    forecast = page_module["generate_forecast"](incidents_df, days_ahead=5)
    assert list(forecast.columns) == ["date", "forecast", "upper_bound", "lower_bound"]
    assert len(forecast) == 5


def test_plot_forecast_emits_the_confidence_band(page_module, incidents_df):
    """Three traces: history, forecast line, and the band that was broken."""
    forecast = page_module["generate_forecast"](incidents_df, days_ahead=5)
    fig = page_module["plot_forecast"](incidents_df, forecast)
    assert len(fig.data) == 3


def test_confidence_band_x_is_a_closed_loop(page_module, incidents_df):
    """The band's x-run must be the dates forward then the same dates back.

    This is the exact expression the missing ``]`` broke. A closed polygon
    needs 2N points; anything else means the concat lost a side.
    """
    forecast = page_module["generate_forecast"](incidents_df, days_ahead=5)
    fig = page_module["plot_forecast"](incidents_df, forecast)
    band = fig.data[2]

    assert len(band.x) == 2 * len(forecast)
    forward = list(band.x[: len(forecast)])
    backward = list(band.x[len(forecast) :])
    assert forward == list(reversed(backward))


def test_confidence_band_y_pairs_upper_then_lower(page_module, incidents_df):
    """The y-run must be upper bounds forward, lower bounds reversed."""
    forecast = page_module["generate_forecast"](incidents_df, days_ahead=5)
    fig = page_module["plot_forecast"](incidents_df, forecast)
    band = fig.data[2]

    assert len(band.y) == 2 * len(forecast)
    assert list(band.y[: len(forecast)]) == list(forecast["upper_bound"])
    assert list(band.y[len(forecast) :]) == list(forecast["lower_bound"][::-1])


def test_confidence_band_is_filled(page_module, incidents_df):
    """``fill="toself"`` is what makes the closed loop render as a band."""
    forecast = page_module["generate_forecast"](incidents_df, days_ahead=5)
    fig = page_module["plot_forecast"](incidents_df, forecast)
    assert fig.data[2].fill == "toself"


def test_plot_forecast_without_a_forecast_skips_the_band(page_module, incidents_df):
    """An empty forecast frame leaves only the historical trace."""
    empty = pd.DataFrame(columns=["date", "forecast", "upper_bound", "lower_bound"])
    fig = page_module["plot_forecast"](incidents_df, empty)
    assert len(fig.data) == 1


# ── Defect 2: both axis settings survived ──────────────────────────────────────


def test_top_pairs_keeps_the_reversed_axis(page_module, incidents_df):
    """The horizontal bar chart must still rank downward from the top.

    This is the setting that *won* the duplicate, so it is the one a naive fix
    (deleting the second ``yaxis``) would have thrown away.
    """
    fig = page_module["plot_top_flagged_pairs"](incidents_df)
    assert fig.layout.yaxis.autorange == "reversed"


def test_top_pairs_keeps_the_blank_axis_title(page_module, incidents_df):
    """The ``yaxis=""`` typo was meant to be ``yaxis_title=""``.

    That is the setting that *lost* the duplicate. Document pair labels are
    self-describing, so the axis is deliberately left untitled next to the
    ``xaxis_title="Times Flagged"`` beside it.
    """
    fig = page_module["plot_top_flagged_pairs"](incidents_df)
    assert fig.layout.yaxis.title.text == ""
    assert fig.layout.xaxis.title.text == "Times Flagged"


def test_top_pairs_ranks_by_flag_count(page_module, incidents_df):
    """The most-flagged pair leads. a.txt/b.txt appears four times here."""
    fig = page_module["plot_top_flagged_pairs"](incidents_df)
    bar = fig.data[0]
    assert bar.orientation == "h"
    assert bar.x[0] == 4


def test_top_pairs_on_empty_input_returns_a_bare_figure(page_module):
    """No pairs means an empty figure rather than a crash."""
    empty = pd.DataFrame(
        columns=["date", "severity", "similarity_score", "document_a", "document_b"]
    )
    fig = page_module["plot_top_flagged_pairs"](empty)
    assert len(fig.data) == 0
