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

from pathlib import Path

WARNING_LIST_PATH = Path("src/utils/warning_list.py")


def test_warning_list_uses_shared_pagination_helper():
    source = WARNING_LIST_PATH.read_text(encoding="utf-8")

    assert (
        "from src.utils.pagination import " "PaginationPage, paginate_items"
    ) in source
    assert "return paginate_items(" in source


def test_warning_list_no_longer_computes_page_bounds():
    source = WARNING_LIST_PATH.read_text(encoding="utf-8")

    assert "math.ceil(total_items / safe_page_size)" not in source
    assert "safe_page = min(max(1, int(page))" not in source


def test_warning_page_is_shared_pagination_type():
    source = WARNING_LIST_PATH.read_text(encoding="utf-8")

    assert ("WarningPage = " "PaginationPage[dict[str, Any]]") in source
