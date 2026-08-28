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

from astroid import nodes
from pylint.checkers import BaseChecker


class StreamlitAntiPatternChecker(BaseChecker):
    name = "streamlit-anti-patterns"
    priority = -1
    msgs = {
        "W9001": (
            "Global variable mutation outside st.session_state is a Streamlit anti-pattern",
            "streamlit-global-mutation",
            "Avoid mutating global variables directly in Streamlit apps as it breaks the reactive execution model.",
        ),
    }

    def visit_global(self, node: nodes.Global) -> None:
        self.add_message("streamlit-global-mutation", node=node)


def register(linter):
    linter.register_checker(StreamlitAntiPatternChecker(linter))
