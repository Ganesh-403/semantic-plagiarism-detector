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

from app.theme import pipeline_progress_html


def test_pipeline_progress_html_without_eta():
    result = pipeline_progress_html(["Extract", "Chunk"])

    assert "Extract" in result
    assert "Chunk" in result
    assert "Estimated processing time" not in result


def test_pipeline_progress_html_displays_eta_under_indicator():
    result = pipeline_progress_html(
        ["Extract", "Chunk", "Embed"],
        active_index=1,
        estimated_seconds=75,
    )

    assert 'class="pipeline-steps"' in result
    assert 'class="pipeline-eta"' in result
    assert ("Estimated processing time: about " "1 minute 15 seconds") in result
    assert result.index("pipeline-steps") < result.index("pipeline-eta")
