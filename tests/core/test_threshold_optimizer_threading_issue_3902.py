"""Regression coverage for issue #3902.

``src/core/threshold_optimizer.py`` created its process-wide singleton lock at
module scope::

    _optimizer_lock = threading.Lock()  # noqa: F821

without importing ``threading``.  Because the statement runs while the module
body executes, the ``NameError`` fired on import -- ``get_threshold_optimizer``
was never reachable at all.

The ``# noqa: F821`` is the interesting part of this bug's history: flake8 *did*
flag the undefined name, and the warning was suppressed rather than acted on.
The fix adds the import and removes the suppression.

Coverage here is deliberately in three layers:

``TestModuleImports``
    The literal symptom -- the module imports, and ``threading`` is bound to the
    real standard-library module.

``TestNoSuppressedUndefinedNames``
    An AST check that ``threading`` is genuinely imported rather than the
    ``NameError`` being papered over again, and that no ``# noqa: F821``
    remains in the file.  This is what stops the bug recurring in the same
    shape.

``TestSingletonUnderConcurrency``
    The lock is not decoration.  ``get_threshold_optimizer`` is a lazy
    double-checked singleton, so these tests assert the invariant the lock
    exists to protect: many threads racing on a cold module still observe
    exactly one ``ThresholdOptimizer``.
"""

from __future__ import annotations

import ast
import importlib
import threading as _threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

MODULE_NAME = "src.core.threshold_optimizer"
MODULE_PATH = Path(__file__).resolve().parents[2] / "src" / "core" / "threshold_optimizer.py"


@pytest.fixture()
def optimizer_module():
    """Import the module fresh and leave the global singleton reset.

    ``_optimizer`` is process-wide state.  Tests in this file deliberately
    exercise the cold-start path, so each one starts from ``None`` and restores
    whatever was there afterwards.
    """
    module = importlib.import_module(MODULE_NAME)
    previous = module._optimizer
    module._optimizer = None
    try:
        yield module
    finally:
        module._optimizer = previous


def _module_ast() -> ast.Module:
    return ast.parse(MODULE_PATH.read_text(encoding="utf-8-sig"), filename=str(MODULE_PATH))


class TestModuleImports:
    """The headline symptom of #3902."""

    def test_module_imports_without_nameerror(self) -> None:
        """Before the fix this raised NameError at line 506."""
        module = importlib.import_module(MODULE_NAME)
        assert module is not None

    def test_threading_is_the_stdlib_module(self) -> None:
        module = importlib.import_module(MODULE_NAME)
        assert hasattr(module, "threading"), (
            "src.core.threshold_optimizer does not bind 'threading'"
        )
        assert module.threading is _threading

    def test_module_level_lock_exists_and_is_a_real_lock(self, optimizer_module) -> None:
        lock = optimizer_module._optimizer_lock
        assert hasattr(lock, "acquire") and hasattr(lock, "release")

        # A threading.Lock is usable as a context manager and is not reentrant.
        assert lock.acquire(blocking=False) is True
        try:
            assert lock.acquire(blocking=False) is False, (
                "_optimizer_lock should be a non-reentrant threading.Lock"
            )
        finally:
            lock.release()

    def test_public_entry_point_is_callable(self, optimizer_module) -> None:
        assert callable(optimizer_module.get_threshold_optimizer)


class TestNoSuppressedUndefinedNames:
    """Guard the *shape* of the bug, not just its symptom."""

    def test_threading_is_imported_at_module_scope(self) -> None:
        tree = _module_ast()
        imported = {
            alias.asname or alias.name
            for node in tree.body
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        assert "threading" in imported, (
            "threading must be imported at module scope; the module creates "
            "threading.Lock() while its body executes"
        )

    def test_no_f821_suppression_remains(self) -> None:
        """`# noqa: F821` is how #3902 survived review -- it must not come back."""
        offenders = [
            f"line {lineno}: {line.strip()}"
            for lineno, line in enumerate(
                MODULE_PATH.read_text(encoding="utf-8-sig").splitlines(), start=1
            )
            if "F821" in line
        ]
        assert not offenders, (
            "F821 (undefined name) suppressions found in threshold_optimizer.py; "
            "fix the undefined name instead of silencing the linter:\n"
            + "\n".join(offenders)
        )

    def test_no_undefined_module_scope_names(self) -> None:
        """Every name read at module scope resolves to something defined there.

        This is a narrow re-implementation of the F821 check that flake8 was
        told to ignore, scoped to module level where a NameError is fatal at
        import time.
        """
        import builtins

        tree = _module_ast()
        defined: set[str] = set(dir(builtins))
        for node in tree.body:
            if isinstance(node, ast.Import):
                defined.update((a.asname or a.name).split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                defined.update(a.asname or a.name for a in node.names)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defined.add(node.name)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    for sub in ast.walk(target):
                        if isinstance(sub, ast.Name):
                            defined.add(sub.id)

        undefined: list[str] = []
        for node in tree.body:
            # Only statements executed directly at module scope; function and
            # class bodies are not evaluated at import time.
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            for sub in ast.walk(node):
                if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                    if sub.id not in defined:
                        undefined.append(f"line {sub.lineno}: {sub.id}")

        assert not undefined, (
            "names read at module scope with no module-scope binding "
            "(these become NameError at import):\n" + "\n".join(undefined)
        )


class TestSingletonUnderConcurrency:
    """The behaviour ``_optimizer_lock`` exists to guarantee."""

    def test_returns_same_instance_on_repeat_calls(self, optimizer_module) -> None:
        first = optimizer_module.get_threshold_optimizer()
        second = optimizer_module.get_threshold_optimizer()
        assert first is second

    def test_populates_the_module_global(self, optimizer_module) -> None:
        assert optimizer_module._optimizer is None
        instance = optimizer_module.get_threshold_optimizer()
        assert optimizer_module._optimizer is instance

    def test_returns_a_threshold_optimizer(self, optimizer_module) -> None:
        instance = optimizer_module.get_threshold_optimizer()
        assert isinstance(instance, optimizer_module.ThresholdOptimizer)

    @pytest.mark.parametrize("workers", [2, 8, 32])
    def test_concurrent_cold_start_yields_one_instance(
        self, optimizer_module, workers: int
    ) -> None:
        """Race many threads onto the cold singleton path at once.

        A barrier makes the threads arrive together, which is what actually
        exercises the lock -- without it they tend to serialise naturally and
        the test passes even when the locking is wrong.
        """
        barrier = _threading.Barrier(workers)

        def call() -> object:
            barrier.wait(timeout=10)
            return optimizer_module.get_threshold_optimizer()

        with ThreadPoolExecutor(max_workers=workers) as pool:
            instances = list(pool.map(lambda _: call(), range(workers)))

        assert len(instances) == workers
        assert len({id(obj) for obj in instances}) == 1, (
            f"{workers} threads observed "
            f"{len({id(o) for o in instances})} different optimizer instances; "
            "the singleton is not thread-safe"
        )

    def test_lock_is_actually_taken(self, optimizer_module, monkeypatch) -> None:
        """``get_threshold_optimizer`` enters ``_optimizer_lock``, not some other lock."""
        real_lock = optimizer_module._optimizer_lock
        entered: list[str] = []

        class RecordingLock:
            def __enter__(self):
                entered.append("enter")
                return real_lock.__enter__()

            def __exit__(self, *exc):
                entered.append("exit")
                return real_lock.__exit__(*exc)

        monkeypatch.setattr(optimizer_module, "_optimizer_lock", RecordingLock())
        optimizer_module.get_threshold_optimizer()
        assert entered == ["enter", "exit"]

    def test_reset_does_not_replace_the_singleton(self, optimizer_module) -> None:
        """``reset()`` clears state in place; identity must survive."""
        instance = optimizer_module.get_threshold_optimizer()
        instance.reset()
        assert optimizer_module.get_threshold_optimizer() is instance
        assert instance.get_results() == {}
        assert instance.get_history() == []
