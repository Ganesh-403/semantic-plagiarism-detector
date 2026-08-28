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

from src.db.schemas import MatchResult


def test_match_result_integer_incident_id():
    # Test that integer incident_id round-trips correctly
    result = MatchResult(
        incident_id=123, document_a="a.txt", document_b="b.txt", similarity_score=0.9
    )
    assert result.incident_id == 123
    assert isinstance(result.incident_id, int)


def test_match_result_string_incident_id_conversion():
    # Test that string integer incident_id converts appropriately
    result = MatchResult(
        incident_id="123", document_a="a.txt", document_b="b.txt", similarity_score=0.9
    )
    assert result.incident_id == 123
    assert isinstance(result.incident_id, int)
