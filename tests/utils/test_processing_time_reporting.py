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
tests/utils/test_processing_time_reporting.py
---------------------------------------------
Regression tests for how processing time is recorded and reported.

Three defects are covered here:

* ``format_processing_duration()`` built its hours branch by hand and never
  appended the seconds, so a 3665-second estimate rendered as "1 hour 1 minute"
  while 125 seconds correctly rendered as "2 minutes 5 seconds". The ETA
  therefore changed shape at the 60-minute boundary and lost information.
* ``ProcessingTimer.time_block()`` only appended a span to ``durations`` when
  it had no parent, so a stage timed inside another stage was dropped from the
  list entirely.
* ``TimingUIRenderer`` emitted a ``</tr8>`` closing tag and interpolated stage
  names into a table rendered with ``unsafe_allow_html=True`` without escaping
  them.
"""

import re
import time
from html import escape

import pytest

from src.utils.processing_time import (
    ProcessingTimer,
    TimingUIRenderer,
    format_processing_duration,
    processing_eta_text,
)


class FakeExpander:
    """Minimal stand-in for the object ``st.expander()`` returns."""

    def __init__(self, label, expanded=False):
        self.label = label
        self.expanded = expanded

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeStreamlit:
    """Captures the markup ``render_debug_expander()`` would render."""

    def __init__(self):
        self.markdown_calls = []
        self.info_calls = []
        self.expanders = []

    def expander(self, label, expanded=False):
        expander = FakeExpander(label, expanded)
        self.expanders.append(expander)
        return expander

    def markdown(self, body, unsafe_allow_html=False):
        self.markdown_calls.append((body, unsafe_allow_html))

    def info(self, body):
        self.info_calls.append(body)

    @property
    def rendered_html(self):
        assert self.markdown_calls, "nothing was rendered"
        return self.markdown_calls[-1][0]


@pytest.fixture
def frozen_clock(monkeypatch):
    """Drive ``time.perf_counter()`` from a scripted list of readings."""

    def install(readings):
        remaining = list(readings)

        def fake_perf_counter():
            return remaining.pop(0)

        monkeypatch.setattr(time, "perf_counter", fake_perf_counter)

    return install


class TestFormatProcessingDurationKeepsSeconds:
    """The seconds component must survive the 60-minute boundary."""

    @pytest.mark.parametrize(
        "seconds,expected",
        [
            (3600, "1 hour"),
            (3601, "1 hour 1 second"),
            (3605, "1 hour 5 seconds"),
            (3660, "1 hour 1 minute"),
            (3665, "1 hour 1 minute 5 seconds"),
            (7325, "2 hours 2 minutes 5 seconds"),
            (7200, "2 hours"),
            (7260, "2 hours 1 minute"),
            (86399, "23 hours 59 minutes 59 seconds"),
        ],
    )
    def test_hours_include_minutes_and_seconds(self, seconds, expected):
        assert format_processing_duration(seconds) == expected

    @pytest.mark.parametrize(
        "seconds,expected",
        [
            (0, "less than a second"),
            (1, "1 second"),
            (45, "45 seconds"),
            (60, "1 minute"),
            (61, "1 minute 1 second"),
            (125, "2 minutes 5 seconds"),
            (3599, "59 minutes 59 seconds"),
        ],
    )
    def test_sub_hour_formatting(self, seconds, expected):
        """Durations below an hour keep their wording, now pluralized properly.

        The only change here is "1 minute 1 second", which the sub-hour branch
        used to render as "1 minute 1 seconds" because it hardcoded the plural.
        """
        assert format_processing_duration(seconds) == expected

    def test_no_component_is_silently_dropped(self):
        """Every non-zero unit in the duration appears in the output."""
        rendered = format_processing_duration(3665)

        assert "1 hour" in rendered
        assert "1 minute" in rendered
        assert "5 seconds" in rendered

    def test_zero_components_are_omitted(self):
        """A round number of hours does not render "0 minutes 0 seconds"."""
        rendered = format_processing_duration(7200)

        assert rendered == "2 hours"
        assert "0" not in rendered

    def test_eta_sentence_carries_the_full_duration(self):
        """The user-facing ETA string inherits the corrected formatting."""
        # 3665 seconds at the default 2.0 seconds per MB.
        total_bytes = int(3665 / 2.0 * 1024 * 1024)
        sentence = processing_eta_text(total_bytes)

        assert sentence.startswith("Estimated processing time: about ")
        assert "hour" in sentence
        assert re.search(r"\d+ seconds?", sentence)


class TestNestedSpansAreRecorded:
    """Every closed span belongs in ``durations``, nested or not."""

    def test_nested_blocks_record_both_spans(self, frozen_clock):
        frozen_clock([0.0, 1.0, 2.0, 3.5])
        timer = ProcessingTimer()

        with timer.time_block("outer"):
            with timer.time_block("inner"):
                pass

        # Innermost closes first, so it lands first in the list.
        assert timer.durations == [1.0, 3.5]

    def test_three_levels_of_nesting(self, frozen_clock):
        frozen_clock([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        timer = ProcessingTimer()

        with timer.time_block("a"):
            with timer.time_block("b"):
                with timer.time_block("c"):
                    pass

        assert timer.durations == [1.0, 3.0, 5.0]

    def test_nested_blocks_record_on_exception(self, frozen_clock):
        frozen_clock([0.0, 1.0, 2.0, 3.0])
        timer = ProcessingTimer()

        with pytest.raises(RuntimeError):
            with timer.time_block("outer"):
                with timer.time_block("inner"):
                    raise RuntimeError("Failed")

        assert timer.durations == [1.0, 3.0]
        assert timer._active_timers == 0

    def test_sibling_blocks_are_all_recorded(self, frozen_clock):
        frozen_clock([0.0, 1.0, 2.0, 4.0])
        timer = ProcessingTimer()

        with timer.time_block("first"):
            pass
        with timer.time_block("second"):
            pass

        assert timer.durations == [1.0, 2.0]

    def test_summary_still_aggregates_by_name(self, frozen_clock):
        frozen_clock([0.0, 1.0, 2.0, 3.5])
        timer = ProcessingTimer()

        with timer.time_block("outer"):
            with timer.time_block("inner"):
                pass

        summary = timer.get_summary()
        assert summary == {"outer": 3.5, "inner": 1.0}

    def test_span_tree_still_reflects_nesting(self, frozen_clock):
        """Flattening ``durations`` must not flatten the span tree."""
        frozen_clock([0.0, 1.0, 2.0, 3.5])
        timer = ProcessingTimer()

        with timer.time_block("outer"):
            with timer.time_block("inner"):
                pass

        assert len(timer.spans) == 1
        assert timer.spans[0].name == "outer"
        assert [child.name for child in timer.spans[0].children] == ["inner"]


class TestTimingTableMarkup:
    """The debug table must be well-formed and must escape its inputs."""

    def _render(self, stage_names, frozen_clock):
        readings = []
        for index in range(len(stage_names)):
            readings.extend([float(index), float(index) + 0.5])
        frozen_clock(readings)

        timer = ProcessingTimer()
        for name in stage_names:
            with timer.time_block(name):
                pass

        fake_st = FakeStreamlit()
        TimingUIRenderer.render_debug_expander(timer, st_module=fake_st)
        return fake_st

    def test_closing_row_tag_is_well_formed(self, frozen_clock):
        """The totals row used to close with </tr8>."""
        html = self._render(["Extract"], frozen_clock).rendered_html

        assert "</tr8>" not in html
        assert html.count("<tr") == html.count("</tr>")

    def test_stage_names_are_escaped(self, frozen_clock):
        """A stage name carrying markup must not reach the DOM as markup."""
        hostile = "<img src=x onerror=alert(1)>"
        html = self._render([hostile], frozen_clock).rendered_html

        assert hostile not in html
        assert escape(hostile) in html

    @pytest.mark.parametrize(
        "stage_name",
        [
            "Extract <b>bold</b>",
            'Parse "quoted.pdf"',
            "Chunk & Embed",
            "Compare <script>x</script>",
        ],
    )
    def test_special_characters_survive_as_text(self, stage_name, frozen_clock):
        html = self._render([stage_name], frozen_clock).rendered_html
        assert escape(stage_name) in html

    def test_table_is_rendered_with_unsafe_html_enabled(self, frozen_clock):
        """Escaping matters precisely because this flag is set."""
        fake_st = self._render(["Extract"], frozen_clock)
        _body, unsafe_allow_html = fake_st.markdown_calls[-1]
        assert unsafe_allow_html is True

    def test_every_stage_gets_a_row(self, frozen_clock):
        html = self._render(["Extract", "Chunk", "Embed"], frozen_clock).rendered_html

        for name in ("Extract", "Chunk", "Embed"):
            assert f"<td>{name}</td>" in html
        # One row per stage, plus the header row and the totals row.
        assert html.count("<tr") == 5

    def test_empty_timer_reports_no_data(self):
        fake_st = FakeStreamlit()
        TimingUIRenderer.render_debug_expander(ProcessingTimer(), st_module=fake_st)

        assert fake_st.info_calls == ["No timing data available."]
        assert fake_st.markdown_calls == []
