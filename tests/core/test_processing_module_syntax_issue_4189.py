"""
tests/core/test_processing_module_syntax_issue_4189.py
------------------------------------------------------
Regression tests for the damage that kept ``src/core/processing.py`` from
parsing (issue #4189).

Four defects, all of the same shape — a line boundary that went missing and
welded two statements together:

Defect 1 — line 26 held two ``from`` imports on one line::

    from src.core.similarity import document_similarity_matrix, flag_plagiarismfrom src.core.text_chunking import chunk_documents

which is ``SyntaxError: invalid syntax``. Both names are really used:
``chunk_documents`` in the chunking span and ``flag_plagiarism`` when the
similarity matrix is scored, so neither import could simply be dropped.

Defect 2 — ``PipelineResult``'s class docstring and its
``is_incremental_update`` method were interleaved. The tail of the docstring
(``Named outputs from run_full_pipeline ...``) was appended to the method's
``return`` expression, and the method sat *above* the annotated fields, which
a ``NamedTuple`` does not allow — fields must come first or they are read as
ordinary class attributes and never become tuple slots.

Defect 3 — ``run_full_pipeline``'s return annotation and its docstring shared
a line: the ``) -> PipelineResult:`` closer was followed on the same line, after
a run of spaces, by the opening triple quote of "Execute the full document
upload pipeline ...". That is ``IndentationError: unexpected indent``.

Defect 4 — two keyword arguments of the final ``PipelineResult(...)`` call were
collapsed onto one line (``chunk_sim_df=chunk_sim_df,            faiss_index=...``).
That one parses, so it is cosmetic, but it is the same merge residue and is
covered here so it does not come back.

``processing.py`` imports the embedding model, which pulls in
``sentence_transformers`` and the rest of the ML stack. The ``module`` fixture
below stubs the ``src.*`` chain so the module body can be executed — and
``PipelineResult`` genuinely constructed — without those installed.
"""

import ast
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "core" / "processing.py"
)

# The ``src.*`` modules processing.py imports. Executing them for real drags in
# sentence-transformers, torch and faiss; none of them matter to the defects
# under test here.
STUBBED_MODULES = (
    "src",
    "src.core",
    "src.core.ai_detector",
    "src.core.config",
    "src.core.document_parser",
    "src.core.embedding_model",
    "src.core.faiss_index",
    "src.core.faiss_index_metadata",
    "src.core.similarity",
    "src.core.text_chunking",
    "src.utils",
    "src.utils.tracing",
)


@pytest.fixture(scope="module")
def source():
    return MODULE_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def tree(source):
    """Parsing at all is the primary regression assert for Defects 1 and 3."""
    return ast.parse(source, filename=MODULE_PATH.name)


@pytest.fixture(scope="module")
def module(tree):
    """Execute the module body against stubbed ``src.*`` dependencies."""
    saved = {name: sys.modules.get(name) for name in STUBBED_MODULES}
    for name in STUBBED_MODULES:
        sys.modules[name] = MagicMock()

    try:
        namespace = {"__name__": "processing_isolated"}
        exec(  # noqa: S102 - deliberately loading the module without its ML deps
            compile(tree, MODULE_PATH.name, "exec"), namespace
        )
        yield namespace
    finally:
        for name, saved_module in saved.items():
            if saved_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved_module


@pytest.fixture(scope="module")
def pipeline_result_cls(module):
    return module["PipelineResult"]


def _module_imports(tree):
    """Every top-level ``from x import y`` in the module."""
    return [node for node in tree.body if isinstance(node, ast.ImportFrom)]


# ── the module parses ──────────────────────────────────────────────────────────


def test_module_compiles(source):
    """The whole of Defects 1 and 3.

    Before the fix: SyntaxError at line 26, then IndentationError at line 74.
    """
    compile(source, MODULE_PATH.name, "exec")


def test_no_two_imports_share_a_line(tree):
    """Each ``from`` import must sit on its own line.

    Defect 1 was one instance. Two import statements can never legally share a
    line, so this checks every one rather than pinning the pair that broke.
    """
    linenos = [node.lineno for node in _module_imports(tree)]
    assert len(linenos) == len(set(linenos)), (
        f"more than one import on a single line: {sorted(linenos)}"
    )


def test_no_source_line_welds_a_name_to_a_following_import(source):
    """Catch the raw ``...flag_plagiarismfrom src...`` shape textually.

    The AST check above cannot see this one before the fix, because the file
    does not parse at all. This is the check that would have caught it.
    """
    offenders = [
        (number, line)
        for number, line in enumerate(source.splitlines(), start=1)
        if "from" in line[1:] and line.lstrip().startswith("from")
        and line.count("import") > 1
    ]
    assert not offenders, f"two imports welded onto one line: {offenders}"


# ── Defect 1: both imported names survived and are both used ───────────────────


def test_similarity_and_chunking_are_imported_separately(tree):
    """The one broken line has to become exactly these two statements."""
    by_module = {
        node.module: {alias.name for alias in node.names}
        for node in _module_imports(tree)
    }

    assert "src.core.similarity" in by_module
    assert by_module["src.core.similarity"] == {
        "document_similarity_matrix",
        "flag_plagiarism",
    }

    assert "src.core.text_chunking" in by_module
    assert by_module["src.core.text_chunking"] == {"chunk_documents"}


@pytest.mark.parametrize("name", ["flag_plagiarism", "chunk_documents"])
def test_both_recovered_imports_are_actually_called(tree, name):
    """Neither import is dead weight, so neither could have been deleted."""
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert name in called, f"{name} is imported but never called"


def test_module_logger_is_assigned_once(tree):
    """``logger = logging.getLogger(__name__)`` appeared twice in a row."""
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "logger"
            for target in node.targets
        )
    ]
    assert len(assignments) == 1, (
        f"logger assigned {len(assignments)} times at "
        f"{[node.lineno for node in assignments]}"
    )


# ── Defect 2: PipelineResult is a well-formed NamedTuple again ─────────────────


def test_pipeline_result_kept_its_docstring(pipeline_result_cls):
    """The docstring tail had been swallowed by the method's return."""
    doc = pipeline_result_cls.__doc__
    assert doc is not None
    assert "Named outputs from" in doc
    assert "run_full_pipeline" in doc


def test_pipeline_result_declares_every_field(pipeline_result_cls):
    """All nine outputs must be real tuple slots, in order.

    While the method sat above them, these annotations were still class-level
    names but no longer fields, so the tuple would have been empty.
    """
    assert pipeline_result_cls._fields == (
        "raw_texts",
        "chunked_docs",
        "embeddings",
        "sim_df",
        "chunk_sim_df",
        "faiss_index",
        "registry",
        "ai_probabilities",
        "flags",
    )


def test_pipeline_result_is_still_unpackable_as_a_tuple(pipeline_result_cls):
    """The docstring promises tuple unpacking; that is the contract callers use."""
    result = pipeline_result_cls(
        raw_texts={"a.txt": "hello"},
        chunked_docs={"a.txt": ["hello"]},
        embeddings={"a.txt": np.zeros((1, 3))},
        sim_df=pd.DataFrame(),
        chunk_sim_df=pd.DataFrame(),
        faiss_index=None,
        registry=[],
        ai_probabilities={},
        flags=[],
    )

    (
        raw_texts,
        chunked_docs,
        embeddings,
        sim_df,
        chunk_sim_df,
        faiss_index,
        registry,
        ai_probabilities,
        flags,
    ) = result

    assert raw_texts == {"a.txt": "hello"}
    assert chunked_docs == {"a.txt": ["hello"]}
    assert embeddings["a.txt"].shape == (1, 3)
    assert sim_df.empty and chunk_sim_df.empty
    assert faiss_index is None
    assert registry == [] and ai_probabilities == {} and flags == []


def test_is_incremental_update_is_a_method_not_a_field(pipeline_result_cls):
    """It must be callable, and must not have become a tuple slot."""
    assert callable(pipeline_result_cls.is_incremental_update)
    assert "is_incremental_update" not in pipeline_result_cls._fields


def test_is_incremental_update_is_declared_after_the_fields(tree):
    """A NamedTuple's methods have to follow its annotated fields.

    This is the ordering that was inverted, and it is why the class could not
    simply have its docstring pasted back in place.
    """
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PipelineResult"
    )
    last_field = max(
        node.lineno
        for node in class_node.body
        if isinstance(node, ast.AnnAssign)
    )
    method = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "is_incremental_update"
    )
    assert method.lineno > last_field


def test_is_incremental_update_returns_a_bool_for_a_plain_result(pipeline_result_cls):
    """A result built by the full-rebuild path reports itself as non-incremental."""
    result = pipeline_result_cls(
        raw_texts={},
        chunked_docs={},
        embeddings={},
        sim_df=pd.DataFrame(),
        chunk_sim_df=pd.DataFrame(),
        faiss_index=None,
        registry=[],
        ai_probabilities={},
        flags=[],
    )
    assert result.is_incremental_update() is False


# ── Defect 3: run_full_pipeline's signature and docstring separated ────────────


def test_run_full_pipeline_is_defined_and_callable(module):
    assert callable(module["run_full_pipeline"])


def test_run_full_pipeline_kept_its_docstring(module):
    doc = module["run_full_pipeline"].__doc__
    assert doc is not None
    assert "Execute the full document upload pipeline" in doc
    assert "PipelineResult" in doc


def test_run_full_pipeline_returns_a_pipeline_result(tree):
    """The annotation that was welded to the docstring must still be there."""
    func = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_full_pipeline"
    )
    assert isinstance(func.returns, ast.Name)
    assert func.returns.id == "PipelineResult"


def test_run_full_pipeline_takes_its_arguments_keyword_only(tree):
    """Everything after ``file_bytes_dict`` sits behind the bare ``*``."""
    func = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_full_pipeline"
    )
    assert [arg.arg for arg in func.args.args] == ["file_bytes_dict"]
    assert "use_incremental" in {arg.arg for arg in func.args.kwonlyargs}


# ── Defect 4: the result is constructed with one keyword per line ──────────────


def test_pipeline_result_is_built_with_every_field(tree):
    """The final call must still name all nine outputs."""
    call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "PipelineResult"
    )
    assert {keyword.arg for keyword in call.keywords} == {
        "raw_texts",
        "chunked_docs",
        "embeddings",
        "sim_df",
        "chunk_sim_df",
        "faiss_index",
        "registry",
        "ai_probabilities",
        "flags",
    }


def test_pipeline_result_call_puts_one_keyword_per_line(tree):
    """``chunk_sim_df`` and ``faiss_index`` had been left sharing a line."""
    call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "PipelineResult"
    )
    linenos = [keyword.value.lineno for keyword in call.keywords]
    assert len(linenos) == len(set(linenos)), (
        f"two keyword arguments share a line: {sorted(linenos)}"
    )
