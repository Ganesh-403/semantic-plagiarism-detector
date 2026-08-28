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

import pytest

from src import database  # Adjust based on your module structure


@pytest.fixture(autouse=True)
def setup_teardown(monkeypatch, tmp_path):
    """
    Refactored fixture to ensure isolated execution using either an in-memory SQLite database
    or a temporary database file that is automatically destroyed upon test completion.
    """
    # Option A: Using an in-memory SQLite database (or Option B using tmp_path)
    test_db = ":memory:"

    # Patch the database path or connection reference used by your app
    monkeypatch.setattr(database, "DEFAULT_DB_PATH", test_db)

    # Initialize the database schema for the test session
    database.init_db()

    yield

    # Teardown logic if required (in-memory DBs destroy themselves automatically)
