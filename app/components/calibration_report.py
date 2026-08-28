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
Calibration Report UI Component (Issue #2267).

Renders a "calibration report" section that shows where the currently
configured plagiarism threshold sits on the precision / recall curve
produced by the automated threshold calibration & backtest harness
(``scripts/calibrate_thresholds.py``).

The report reads the ``calibration`` block embedded in a recommended
threshold config (default ``config/thresholds.recommended.json``). When no
recommended config exists the section renders a lightweight empty state so
default behavior is unchanged.
"""

from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

from src.core.calibration import load_calibration_report
from src.core.config import DEFAULT_THRESHOLDS, load_threshold_config

logger = logging.getLogger(__name__)

DEFAULT_RECOMMENDED_CONFIG = (
    Path(__file__).resolve().parents[2] / "config" / "thresholds.recommended.json"
)


def _metric_or(calibration: dict, key: str, fallback: object) -> object:
    """Return a numeric metric from the calibration block or a fallback."""
    value = calibration.get(key, fallback)
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def render_calibration_report(config_path: str | Path | None = None) -> None:
    """Render the threshold calibration report section.

    Args:
        config_path: Optional path to a recommended threshold config. Defaults
            to ``config/thresholds.recommended.json`` relative to the repo root.
    """
    st.markdown("### 🎯 Threshold Calibration Report")

    if config_path is None:
        config_path = DEFAULT_RECOMMENDED_CONFIG
    config_path = Path(config_path)

    if not config_path.exists():
        st.caption(
            "No calibration report available. Run "
            "`python scripts/calibrate_thresholds.py --csv labeled_pairs.csv` "
            "to backtest thresholds and generate a recommended config."
        )
        return

    calibration = load_calibration_report(str(config_path))
    if not calibration:
        st.caption("Calibration config found, but it contains no calibration data.")
        return

    # The currently active threshold the app is using (defaults unchanged when
    # no calibrated config is loaded).
    active_threshold = load_threshold_config().plagiarism

    recommended = _metric_or(calibration, "recommended_threshold", active_threshold)
    precision = _metric_or(calibration, "precision", 0.0)
    recall = _metric_or(calibration, "recall", 0.0)
    f1 = _metric_or(calibration, "f1", 0.0)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Active Threshold",
        f"{active_threshold:.3f}",
        delta=f"{(recommended - active_threshold) * 100:+.1f} pp",
        delta_color="inverse",
        help="Threshold currently used by the detection pipeline.",
    )
    col2.metric("Recommended Threshold", f"{recommended:.3f}")
    col3.metric("Precision", f"{precision:.3f}")
    col4.metric("Recall", f"{recall:.3f}")

    dataset = calibration.get("dataset")
    score_column = calibration.get("score_column")
    samples = calibration.get("samples")
    details = " · ".join(
        part
        for part in [
            dataset,
            f"score column: {score_column}" if score_column else None,
            f"{samples} labeled pairs" if samples else None,
        ]
        if part
    )
    if details:
        st.caption(details)

    sweep = calibration.get("sweep")
    if sweep:
        try:
            from app.theme import get_chart_colors
            from src.visualization.analytics import plot_precision_recall_curve

            fig = plot_precision_recall_curve(
                list(sweep),
                current_threshold=active_threshold,
                theme_colors=get_chart_colors() if callable(get_chart_colors) else None,
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:  # noqa: BLE001 - report without breaking the app
            logger.warning("Failed to render calibration report chart: %s", exc)
            st.caption("The precision/recall curve could not be rendered.")

    st.caption(
        "Recommended thresholds can be adopted by setting the "
        "`THRESHOLD_CONFIG_PATH` environment variable or copying the config "
        "to `config/thresholds.json`. Defaults are: "
        f"plagiarism {DEFAULT_THRESHOLDS.plagiarism}, "
        f"medium {DEFAULT_THRESHOLDS.medium}, high {DEFAULT_THRESHOLDS.high}."
    )
