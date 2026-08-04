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
