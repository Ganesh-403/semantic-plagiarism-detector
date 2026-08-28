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
from typing import Union


def build_similarity_cache_key(session_id: str | Path, *, use_hybrid: bool) -> str:
    """Build a mode-specific analysis cache key.

    Lexical and Hybrid scoring produce different result spaces. Keeping the
    mode in the key prevents a result generated in one mode from being reused
    after the UI switches to the other mode.
    Converts pathing logic using pathlib.Path(p).as_posix() explicitly for cross-platform support (Issue #2939, #3028).
    """
    if isinstance(session_id, Path):
        sid_str = session_id.as_posix()
    elif session_id is not None:
        sid_str = Path(str(session_id)).as_posix()
    else:
        sid_str = ""

    suffix = "hybrid_v1" if use_hybrid else "lexical"
    return f"{sid_str}:analysis_results_{suffix}"
