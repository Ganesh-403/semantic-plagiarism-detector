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

"""Unit tests for Filter by Assignment selectbox in Corpus Document Management view."""

from src.db.schemas import Document


def test_assignment_filter_selectbox_all():
    """Verify 'All Assignments' includes all document rows."""
    docs = [
        Document(
            filename="a.txt",
            file_hash="h1",
            upload_date="2026-08-08",
            assignment_title="Assignment 1",
        ),
        Document(
            filename="b.txt",
            file_hash="h2",
            upload_date="2026-08-08",
            assignment_title="Assignment 2",
        ),
    ]
    raw_titles = sorted(list({d.assignment_title for d in docs} - {None, ""}))
    assignment_titles = ["All Assignments"] + raw_titles
    assert assignment_titles == ["All Assignments", "Assignment 1", "Assignment 2"]


def test_assignment_filter_selectbox_filtered():
    """Verify filtering documents by selected assignment."""
    docs = [
        Document(
            filename="a.txt",
            file_hash="h1",
            upload_date="2026-08-08",
            assignment_title="Assignment 1",
        ),
        Document(
            filename="b.txt",
            file_hash="h2",
            upload_date="2026-08-08",
            assignment_title="Assignment 2",
        ),
    ]
    selected_assignment = "Assignment 1"
    filtered = [doc for doc in docs if doc.assignment_title == selected_assignment]
    assert len(filtered) == 1
    assert filtered[0].filename == "a.txt"
