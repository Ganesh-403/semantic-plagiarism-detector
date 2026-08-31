"""Regression tests for #4079.

``src/api/app.py`` stopped parsing because a merge conflict was resolved by
deleting the ``<<<<<<<`` / ``=======`` / ``>>>>>>>`` markers but leaving the
branch names that sat beside them:

    @app.on_event("startup")
    def startup_event():
     feature/model-warmup-startup-4009
        from src.core.embedding_model import warmup_embedding_model
        warmup_embedding_model()

        print_startup_config_summary()

     main

Python reported ``IndentationError: unexpected indent`` at line 61 and the
whole REST API — every router, ``src/asgi_app.py``, ``tests/api/`` — became
unimportable.

Two things are covered here:

1. A repo-wide guard that no file under ``src/`` or ``app/`` carries conflict
   markers or the bare-branch-name residue that a stripped-marker resolution
   leaves behind. That is the shape of the bug, and it has now bitten this
   repo more than once, so it is worth catching by pattern rather than by
   file.
2. The behaviour ``startup_event`` was supposed to have all along: warm the
   embedding model *and* print the config summary, with the warm-up unable to
   take the process down.
"""

from __future__ import annotations

import ast
import importlib
import pathlib
import re

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

#: Directories whose every ``.py`` file must parse.
SOURCE_ROOTS = ("src", "app")

#: Real conflict markers, anchored at column zero the way git writes them.
CONFLICT_MARKER_RE = re.compile(r"^(?:<{7}|={7}|>{7})(?:\s|$)", re.MULTILINE)

#: The residue this issue is about: a line holding nothing but a branch-like
#: token, indented oddly, left over once the markers themselves are deleted.
#: ``main``, ``feature/model-warmup-startup-4009``, ``fix/some-thing`` — all
#: are valid Python expressions or near-expressions, so the parser's
#: complaint lands somewhere unhelpful.
BRANCH_RESIDUE_RE = re.compile(
    r"^[ \t]+(?:main|master|develop|"
    r"(?:feat|feature|fix|hotfix|chore|refactor|docs|test)[-/][\w./-]+)[ \t]*$",
    re.MULTILINE,
)


def _python_files(*roots: str) -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    for root in roots:
        base = REPO_ROOT / root
        if base.is_file() and base.suffix == ".py":
            files.append(base)
            continue
        if not base.is_dir():
            continue
        files.extend(
            p
            for p in base.rglob("*.py")
            if "__pycache__" not in p.parts and "node_modules" not in p.parts
        )
    return sorted(set(files))


#: Everything the residue guards sweep.
ALL_PYTHON_FILES = _python_files(*SOURCE_ROOTS)

#: What the parse guard covers. Deliberately narrower than ALL_PYTHON_FILES:
#: sweeping the whole tree today also trips over three unrelated files that
#: predate this branch, and folding those fixes in here would bury the one
#: change this PR is about. They are filed separately —
#: src/utils/warning_list.py is #4080, and app/pages/5_Writing_Style_Analyzer.py
#: (line 101) and app/pages/9_Trends_Insights.py (line 484) are two more
#: unparseable files this guard surfaced. Widen API_SOURCE_ROOTS to
#: SOURCE_ROOTS once those land.
API_SOURCE_ROOTS = ("src/api", "src/asgi_app.py")

API_PYTHON_FILES = _python_files(*API_SOURCE_ROOTS)


def test_source_tree_is_not_empty() -> None:
    """Guard the guard: an empty file list would make everything below vacuous."""
    assert len(ALL_PYTHON_FILES) > 100, (
        f"expected to find the project sources under {SOURCE_ROOTS}, "
        f"found only {len(ALL_PYTHON_FILES)} files"
    )
    assert len(API_PYTHON_FILES) > 10, (
        f"expected to find the API sources under {API_SOURCE_ROOTS}, "
        f"found only {len(API_PYTHON_FILES)} files"
    )
    assert (REPO_ROOT / "src" / "api" / "app.py") in API_PYTHON_FILES, (
        "the file this issue is about must be inside the parse guard"
    )


@pytest.mark.parametrize(
    "path", API_PYTHON_FILES, ids=lambda p: str(p.relative_to(REPO_ROOT))
)
def test_every_api_source_file_parses(path: pathlib.Path) -> None:
    """No file under src/ or app/ may fail to compile.

    This is what actually broke: ``compileall`` on ``main`` reported
    ``src/api/app.py`` and ``src/utils/warning_list.py``. A parse error is not
    a runtime edge case — it takes the importing module down with it, and the
    resulting collection error reads like an environment problem rather than a
    code problem.
    """
    source = path.read_text(encoding="utf-8", errors="replace")
    try:
        ast.parse(source, filename=str(path))
    except SyntaxError as exc:  # pragma: no cover - only on regression
        rel = path.relative_to(REPO_ROOT)
        pytest.fail(f"{rel}:{exc.lineno}: {exc.msg}\n    {(exc.text or '').rstrip()}")


@pytest.mark.parametrize(
    "path", ALL_PYTHON_FILES, ids=lambda p: str(p.relative_to(REPO_ROOT))
)
def test_no_conflict_markers(path: pathlib.Path) -> None:
    """No file may carry a literal git conflict marker."""
    source = path.read_text(encoding="utf-8", errors="replace")
    found = CONFLICT_MARKER_RE.findall(source)
    assert not found, (
        f"{path.relative_to(REPO_ROOT)} contains unresolved conflict markers: {found}"
    )


@pytest.mark.parametrize(
    "path", ALL_PYTHON_FILES, ids=lambda p: str(p.relative_to(REPO_ROOT))
)
def test_no_stray_branch_name_lines(path: pathlib.Path) -> None:
    """No file may carry a bare branch name where a marker used to be.

    This is the exact residue from #4079 — the markers were removed, the
    labels beside them were not.
    """
    source = path.read_text(encoding="utf-8", errors="replace")
    stray = [m.strip() for m in BRANCH_RESIDUE_RE.findall(source)]
    assert not stray, (
        f"{path.relative_to(REPO_ROOT)} has lines that look like merge residue: "
        f"{stray}"
    )


def test_branch_residue_pattern_matches_the_original_defect() -> None:
    """The detector must actually catch the snippet that caused this issue."""
    original = (
        '@app.on_event("startup")\n'
        "def startup_event():\n"
        " feature/model-warmup-startup-4009\n"
        "    from src.core.embedding_model import warmup_embedding_model\n"
        "    warmup_embedding_model()\n"
        "\n"
        "    print_startup_config_summary()\n"
        "\n"
        " main\n"
    )
    assert [m.strip() for m in BRANCH_RESIDUE_RE.findall(original)] == [
        "feature/model-warmup-startup-4009",
        "main",
    ]
    with pytest.raises(SyntaxError):
        ast.parse(original)


def test_branch_residue_pattern_does_not_fire_on_real_code() -> None:
    """Ordinary indented code must not be mistaken for residue.

    ``main`` as a bare indented statement is residue; ``main()``,
    ``main = ...``, ``return main`` and a dict key are not.
    """
    benign = "\n".join(
        [
            "def run():",
            "    main()",
            "    main = 1",
            "    return main",
            "    x = {'main': 1}",
            "    from a import main",
            "    # main",
            "    feature = 2",
            "    print(fix)",
        ]
    )
    assert BRANCH_RESIDUE_RE.findall(benign) == []


# ── startup_event behaviour ──────────────────────────────────────────────────


@pytest.fixture()
def app_module():
    """The ``src.api.app`` *module*.

    ``src/api/__init__.py`` re-exports the FastAPI instance under the name
    ``app``, which shadows the submodule of the same name, so
    ``import src.api.app as m`` hands back the FastAPI object rather than the
    module. Go through ``importlib`` to get the module itself.
    """
    return importlib.import_module("src.api.app")


def test_module_exposes_the_startup_hook(app_module) -> None:
    assert callable(app_module.startup_event)
    assert callable(app_module._warmup_embedding_model)


def test_startup_event_runs_both_steps(app_module, monkeypatch) -> None:
    """Both halves of the botched conflict must survive, not one of them.

    The conflict was between a branch adding the model warm-up and ``main``
    printing the config summary. They are independent, so the resolution keeps
    both.
    """
    calls: list[str] = []
    monkeypatch.setattr(
        app_module, "_warmup_embedding_model", lambda: calls.append("warmup") or True
    )
    monkeypatch.setattr(
        app_module, "print_startup_config_summary", lambda: calls.append("summary")
    )

    app_module.startup_event()

    assert calls == ["warmup", "summary"]


def test_startup_survives_a_failing_warmup(app_module, monkeypatch) -> None:
    """A warm-up that raises must not stop the API from coming up.

    Most of the surface — auth, corpus listing, admin — needs no embeddings.
    Refusing to boot because the model could not be preloaded trades a latency
    problem for an availability one.
    """
    printed: list[str] = []

    def boom() -> bool:
        raise RuntimeError("no weights cached and no network")

    monkeypatch.setattr("src.core.embedding_model.warmup_embedding_model", boom)
    monkeypatch.setattr(
        app_module, "print_startup_config_summary", lambda: printed.append("summary")
    )

    app_module.startup_event()

    assert printed == ["summary"], "the config summary must still be printed"


def test_warmup_reports_failure_rather_than_raising(app_module, monkeypatch) -> None:
    def boom() -> bool:
        raise RuntimeError("device disappeared")

    monkeypatch.setattr("src.core.embedding_model.warmup_embedding_model", boom)
    assert app_module._warmup_embedding_model() is False


def test_warmup_reports_success(app_module, monkeypatch) -> None:
    monkeypatch.setattr(
        "src.core.embedding_model.warmup_embedding_model", lambda: True
    )
    assert app_module._warmup_embedding_model() is True


def test_warmup_propagates_a_falsy_result(app_module, monkeypatch) -> None:
    """``warmup_embedding_model`` returns False on a handled failure."""
    monkeypatch.setattr(
        "src.core.embedding_model.warmup_embedding_model", lambda: False
    )
    assert app_module._warmup_embedding_model() is False


def test_warmup_survives_an_unimportable_ml_stack(app_module, monkeypatch) -> None:
    """The ML extras are optional; a missing torch must not break startup.

    An API deployment that only serves the non-embedding routes is a supported
    configuration, so the deferred import has to be allowed to fail.
    """
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "src.core.embedding_model":
            raise ModuleNotFoundError("No module named 'sentence_transformers'")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(
        importlib.sys.modules, "src.core.embedding_model", raising=False
    )
    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert app_module._warmup_embedding_model() is False


def test_embedding_model_is_not_imported_at_module_scope() -> None:
    """The heavy import must stay inside the function.

    ``src/api/app.py`` is imported by tooling that has no reason to pay for the
    ML stack — test collection, OpenAPI schema dumps, ``--help``. Keeping the
    import deferred is the point, so assert it rather than trusting it.
    """
    source = (REPO_ROOT / "src" / "api" / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            name = getattr(node, "module", None) or ""
            assert "embedding_model" not in name, (
                "src.core.embedding_model must be imported lazily, not at module scope"
            )


def test_startup_hook_is_registered_on_the_app(app_module) -> None:
    """Fixing the parse error is not enough if the hook is never wired up."""
    handlers = app_module.app.router.on_startup
    assert any(h.__name__ == "startup_event" for h in handlers), (
        f"startup_event not registered; found {[h.__name__ for h in handlers]}"
    )
