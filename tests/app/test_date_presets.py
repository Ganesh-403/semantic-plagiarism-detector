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

from datetime import date, timedelta

from app.streamlit_app import get_date_range_preset


def test_get_date_range_preset_last_14_days():
    """Verify Last 14 Days date range preset calculation (#1727)."""
    today = date.today()
    start_date, end_date = get_date_range_preset("Last 14 Days")
    assert end_date == today
    assert start_date == today - timedelta(days=14)


def test_get_date_range_preset_all_options():
    """Verify all date range presets return correct date boundaries."""
    today = date.today()

    s_today, e_today = get_date_range_preset("Today")
    assert (s_today, e_today) == (today, today)

    s_7, e_7 = get_date_range_preset("Last 7 Days")
    assert (s_7, e_7) == (today - timedelta(days=6), today)

    s_14, e_14 = get_date_range_preset("Last 14 Days")
    assert (s_14, e_14) == (today - timedelta(days=14), today)

    s_30, e_30 = get_date_range_preset("Last 30 Days")
    assert (s_30, e_30) == (today - timedelta(days=29), today)

    s_all, e_all = get_date_range_preset("All Time")
    assert (s_all, e_all) == (date(2020, 1, 1), today)
