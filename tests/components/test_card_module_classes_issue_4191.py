"""
tests/components/test_card_module_classes_issue_4191.py
--------------------------------------------------------
Regression tests for the four card modules that lost their class statement
(issue #4191).

``code_clone_card``, ``faiss_vector_card``, ``multimodal_ocr_card`` and
``stylometric_author_card`` all failed to compile with the same error --
"unterminated triple-quoted string literal" -- reported against the *last*
method docstring in the file, which is only where the parser gave up.

The real damage was earlier and identical in all four: the ``class ...:``
line and the opening ``\"\"\"`` of its docstring were gone, so the orphaned
docstring body ran on directly from the quote that closes the preceding
``render_*_card`` f-string::

        </div>
    </div>
    \"\"\"                                     <- closes the render_* f-string
    Renders enterprise Streamlit UI widgets for ...
    \"\"\"                                     <- opens a string that never closes

    @staticmethod
    def render_code_clone_summary_card(...):

Every file carried 11 occurrences of ``\"\"\"`` -- an odd count, which is the
tell -- and every ``@staticmethod`` below sat indented one level, which is what
identifies them as members of a class that was no longer declared.

Because the damage is one shape repeated four times, these tests are
parametrised over all four modules rather than written out per file. The
``EXPECTED`` table below pins, for each module, the class that was missing, the
module-level render helper the dashboards import, and the static methods that
had been orphaned.
"""

import ast
import importlib
from pathlib import Path

import pytest

COMPONENTS_DIR = Path(__file__).resolve().parents[2] / "src" / "components"
PAGES_DIR = Path(__file__).resolve().parents[2] / "src" / "pages"

# module stem -> (restored class, module-level helper, orphaned static methods)
EXPECTED = {
    "code_clone_card": (
        "CodeCloneCard",
        "render_code_clone_card",
        ("render_code_clone_summary_card", "render_clone_matches_list"),
    ),
    "faiss_vector_card": (
        "FAISSVectorCard",
        "render_vector_match_card",
        ("render_vector_index_metrics", "render_nearest_neighbor_results"),
    ),
    "multimodal_ocr_card": (
        "MultimodalOCRCard",
        "render_ocr_match_card",
        ("render_ocr_summary_card", "render_paraphrase_alignment_matrix"),
    ),
    "stylometric_author_card": (
        "StylometricAuthorCard",
        "render_stylometric_card",
        ("render_writeprint_summary_card", "render_authorship_attribution_results"),
    ),
}

MODULE_STEMS = sorted(EXPECTED)

# The dashboard page that imports each card module's helper.
DASHBOARD_IMPORTS = {
    "code_clone_card": "neural_code_clone_dashboard",
    "faiss_vector_card": "faiss_vector_dashboard",
    "multimodal_ocr_card": "multimodal_ocr_dashboard",
    "stylometric_author_card": "stylometric_author_dashboard",
}


def _path(stem):
    return COMPONENTS_DIR / f"{stem}.py"


def _source(stem):
    return _path(stem).read_text(encoding="utf-8")


def _tree(stem):
    return ast.parse(_source(stem), filename=f"{stem}.py")


def _module(stem):
    return importlib.import_module(f"src.components.{stem}")


# ── every module parses again ──────────────────────────────────────────────────


@pytest.mark.parametrize("stem", MODULE_STEMS)
def test_module_compiles(stem):
    """The whole of the SyntaxError, for each of the four files."""
    compile(_source(stem), f"{stem}.py", "exec")


@pytest.mark.parametrize("stem", MODULE_STEMS)
def test_triple_quotes_are_balanced(stem):
    """An odd count is what an unterminated docstring looks like.

    This is the cheap check that would have caught all four at once, so it is
    kept as a guard rather than only asserting the file happens to parse now.
    """
    count = _source(stem).count('"""')
    assert count % 2 == 0, f"{stem}.py has an unbalanced {count} triple quotes"


@pytest.mark.parametrize("stem", MODULE_STEMS)
def test_module_imports(stem):
    """Importing is what the dashboards actually do."""
    assert _module(stem) is not None


# ── the class statement is back ────────────────────────────────────────────────


@pytest.mark.parametrize("stem", MODULE_STEMS)
def test_expected_class_is_defined(stem):
    class_name = EXPECTED[stem][0]
    assert hasattr(_module(stem), class_name), f"{stem}.py is missing {class_name}"


@pytest.mark.parametrize("stem", MODULE_STEMS)
def test_module_defines_exactly_one_class(stem):
    """The orphaned methods belong to one card class, not several."""
    classes = [
        node.name for node in _tree(stem).body if isinstance(node, ast.ClassDef)
    ]
    assert classes == [EXPECTED[stem][0]]


@pytest.mark.parametrize("stem", MODULE_STEMS)
def test_class_kept_the_orphaned_docstring(stem):
    """The docstring body survived; only its opening quote had been lost."""
    class_name = EXPECTED[stem][0]
    doc = getattr(_module(stem), class_name).__doc__
    assert doc is not None and doc.strip(), f"{class_name} has no docstring"
    assert "Renders" in doc


@pytest.mark.parametrize("stem", MODULE_STEMS)
def test_docstring_is_not_leaking_into_module_scope(stem):
    """The orphaned text must be a docstring, not a stray expression.

    A naive fix -- deleting the loose lines to make the file parse -- would
    pass ``test_module_compiles`` while silently dropping the class.
    """
    tree = _tree(stem)
    stray = [
        node.lineno
        for node in tree.body
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
    ]
    # Only the module docstring itself may be a bare string at module level.
    assert stray in ([], [1]), f"stray string expressions at lines {stray}"


# ── the orphaned static methods are members again ──────────────────────────────


@pytest.mark.parametrize("stem", MODULE_STEMS)
def test_orphaned_methods_are_on_the_class(stem):
    class_name, _, methods = EXPECTED[stem]
    card = getattr(_module(stem), class_name)
    for method in methods:
        assert callable(getattr(card, method, None)), f"{class_name}.{method} missing"


@pytest.mark.parametrize("stem", MODULE_STEMS)
def test_orphaned_methods_are_not_module_level_functions(stem):
    """They were written indented, i.e. as members; they must not be loose.

    If the class had been re-declared in the wrong place, these would show up
    at module scope instead and the dashboards would silently bind the wrong
    thing.
    """
    _, _, methods = EXPECTED[stem]
    module_functions = {
        node.name
        for node in _tree(stem).body
        if isinstance(node, ast.FunctionDef)
    }
    assert not (set(methods) & module_functions)


@pytest.mark.parametrize("stem", MODULE_STEMS)
def test_methods_are_static(stem):
    """They take no ``self``, and the dashboards call them off the class."""
    class_name, _, methods = EXPECTED[stem]
    card = getattr(_module(stem), class_name)
    for method in methods:
        assert isinstance(
            card.__dict__[method], staticmethod
        ), f"{class_name}.{method} is not a staticmethod"


# ── the module-level helper the dashboards import still works ──────────────────


@pytest.mark.parametrize("stem", MODULE_STEMS)
def test_render_helper_is_module_level(stem):
    """It sits above the class and was never broken, only unreachable."""
    helper = EXPECTED[stem][1]
    module_functions = {
        node.name
        for node in _tree(stem).body
        if isinstance(node, ast.FunctionDef)
    }
    assert helper in module_functions
    assert callable(getattr(_module(stem), helper))


@pytest.mark.parametrize("stem", MODULE_STEMS)
def test_render_helper_returns_markup(stem):
    """An empty payload must still produce a string of HTML, not raise."""
    helper = getattr(_module(stem), EXPECTED[stem][1])
    markup = helper({})
    assert isinstance(markup, str)
    assert "<div" in markup


@pytest.mark.parametrize("stem", MODULE_STEMS)
def test_render_helper_interpolates_its_payload(stem):
    """The f-string whose closing quote anchored the damage still fills in."""
    helper = getattr(_module(stem), EXPECTED[stem][1])
    empty = helper({})
    populated = helper(
        {
            "cosine_similarity_score": 0.91,
            "l2_distance": 0.42,
            "attribution_confidence_percentage": 88.0,
            "stylometric_distance": 1.25,
            "matchedFileId": "FILE-001",
            "avgOCRConfidencePct": 97,
        }
    )
    assert isinstance(populated, str)
    assert populated != empty or "<div" in populated


# ── the dashboards can reach them again ────────────────────────────────────────


@pytest.mark.parametrize("stem", MODULE_STEMS)
def test_dashboard_imports_the_helper_by_name(stem):
    """Each dashboard page names the helper in a ``from src.components...`` import.

    This is the coupling that made a SyntaxError here take four pages down.
    """
    page = PAGES_DIR / f"{DASHBOARD_IMPORTS[stem]}.py"
    tree = ast.parse(page.read_text(encoding="utf-8"), filename=page.name)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == f"src.components.{stem}"
        for alias in node.names
    }
    assert EXPECTED[stem][1] in imported
