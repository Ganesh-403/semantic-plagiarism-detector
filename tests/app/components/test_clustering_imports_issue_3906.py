"""Regression coverage for issue #3906.

``app/components/clustering.py`` is a 1085-line module implementing
hierarchical / k-means document clustering, topic modelling and pattern
evolution tracking (Issue #1984).  Its import block covered only the optional
scientific dependencies inside a ``try:``.  Everything else the module used --
``numpy``, ``pandas``, ``matplotlib``, ``streamlit``, ``os``,
``datetime``/``timedelta``, the ``typing`` aliases, and ``logger`` itself -- had
no binding anywhere in the file.

The module does not use ``from __future__ import annotations``, so annotations
are evaluated while the class body executes.  The failure was therefore at
*import* time, on the first method of the first class::

    File "app/components/clustering.py", line 44, in SemanticClusterer
        def fit_hierarchical(self, similarity_matrix: np.ndarray,
                                                      ^^
    NameError: name 'np' is not defined

None of the 1085 lines was reachable.

There was a second, latent bug in the same block: ``logger`` was referenced on
line 19 inside ``except ImportError``, so a machine without scipy/sklearn would
have traded the ``ImportError`` for a ``NameError`` and lost the diagnostic the
handler was written to emit.  ``logger`` is now defined *before* the ``try:``.

The tests below go past "it imports": the module was effectively untested dead
weight, so ``SemanticClusterer`` is exercised end to end on a small
similarity matrix with a known cluster structure.
"""

from __future__ import annotations

import ast
import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[3] / "app" / "components" / "clustering.py"
)


def _load_module(name: str = "clustering_under_test") -> ModuleType:
    """Import ``clustering.py`` by path.

    ``app/components`` has no package ``__init__`` re-export for this module, and
    loading by path is what makes the import-time ``NameError`` in #3906 visible
    as a test failure rather than a collection error.
    """
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


@pytest.fixture(scope="module")
def clustering() -> ModuleType:
    module = _load_module()
    yield module
    sys.modules.pop("clustering_under_test", None)


@pytest.fixture()
def two_block_similarity(clustering):
    """A 6x6 similarity matrix with two obvious blocks of three documents.

    Documents 0-2 are near-identical to one another, as are 3-5, and the two
    groups barely resemble each other.  Any correct clustering must separate
    them, which makes the assertions below independent of linkage details.
    """
    np = clustering.np
    matrix = np.full((6, 6), 0.05, dtype=float)
    matrix[:3, :3] = 0.95
    matrix[3:, 3:] = 0.92
    np.fill_diagonal(matrix, 1.0)
    return matrix


class TestModuleImports:
    """The headline symptom of #3906."""

    def test_module_imports_without_nameerror(self) -> None:
        module = _load_module("clustering_import_probe")
        try:
            assert module is not None
        finally:
            sys.modules.pop("clustering_import_probe", None)

    @pytest.mark.parametrize(
        "name",
        ["logging", "os", "datetime", "timedelta", "np", "pd", "plt", "st", "logger"],
    )
    def test_required_names_are_bound(self, clustering, name: str) -> None:
        assert hasattr(clustering, name), (
            f"clustering.py uses {name!r} but does not bind it"
        )

    @pytest.mark.parametrize("name", ["Any", "Dict", "List", "Optional", "Tuple"])
    def test_typing_aliases_are_bound(self, clustering, name: str) -> None:
        """These appear in evaluated annotations, so a miss is fatal at import."""
        assert hasattr(clustering, name)

    def test_third_party_aliases_point_at_the_right_libraries(self, clustering) -> None:
        assert clustering.np.__name__ == "numpy"
        assert clustering.pd.__name__ == "pandas"
        assert clustering.st.__name__ == "streamlit"
        assert clustering.plt.__name__ == "matplotlib.pyplot"

    def test_public_surface_is_present(self, clustering) -> None:
        for name in (
            "SemanticClusterer",
            "TopicExtractor",
            "PatternEvolutionTracker",
            "plot_cluster_dendrogram",
            "plot_cluster_scatter",
            "render_clustering_tab",
            "CLUSTERING_AVAILABLE",
        ):
            assert hasattr(clustering, name), f"missing public name {name!r}"


class TestLoggerIsUsableByTheImportGuard:
    """The latent second bug: ``logger`` was used before it existed."""

    def test_logger_is_a_real_logger(self, clustering) -> None:
        assert isinstance(clustering.logger, logging.Logger)

    def test_logger_is_defined_before_the_try_block(self) -> None:
        """Ordering is the whole point -- the ``except`` clause logs through it.

        If ``logger = logging.getLogger(...)`` sat *after* the ``try:``, a
        machine missing scipy/sklearn would raise NameError from inside the
        exception handler instead of degrading gracefully.
        """
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8-sig"))

        logger_line = None
        try_line = None
        for node in tree.body:
            if (
                logger_line is None
                and isinstance(node, ast.Assign)
                and any(
                    isinstance(t, ast.Name) and t.id == "logger" for t in node.targets
                )
            ):
                logger_line = node.lineno
            if try_line is None and isinstance(node, ast.Try):
                try_line = node.lineno

        assert logger_line is not None, "clustering.py never assigns `logger`"
        assert try_line is not None, "expected the optional-dependency try/except"
        assert logger_line < try_line, (
            f"`logger` is assigned at line {logger_line} but the optional-dependency "
            f"block that logs through it starts at line {try_line}"
        )

    def test_missing_optional_dependency_logs_instead_of_raising(self, caplog) -> None:
        """Simulate the degraded install the ``except ImportError`` is for."""
        blocked = ("sklearn", "scipy")
        saved = {
            name: module
            for name, module in sys.modules.items()
            if name.split(".")[0] in blocked
        }
        for name in list(saved):
            del sys.modules[name]

        real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

        def blocking_import(name, *args, **kwargs):
            if name.split(".")[0] in blocked:
                raise ImportError(f"No module named {name!r}")
            return real_import(name, *args, **kwargs)

        import builtins

        builtins.__import__ = blocking_import
        try:
            with caplog.at_level(logging.WARNING):
                module = _load_module("clustering_degraded")
            assert module.CLUSTERING_AVAILABLE is False
            assert any(
                "Clustering dependencies not available" in record.message
                for record in caplog.records
            ), "the ImportError handler did not log its diagnostic"
        finally:
            builtins.__import__ = real_import
            sys.modules.pop("clustering_degraded", None)
            sys.modules.update(saved)


class TestNoUndefinedNames:
    """A narrow F821 check over the module, so this cannot recur."""

    def test_no_undefined_names_anywhere_in_the_module(self) -> None:
        import builtins as _builtins

        source = MODULE_PATH.read_text(encoding="utf-8-sig")
        tree = ast.parse(source)

        defined: set[str] = set(dir(_builtins))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                defined.update((a.asname or a.name).split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                defined.update(a.asname or a.name for a in node.names)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defined.add(node.name)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                defined.add(node.id)
            elif isinstance(node, ast.arg):
                defined.add(node.arg)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                defined.add(node.name)
            elif isinstance(node, ast.Global):
                defined.update(node.names)

        undefined = sorted(
            {
                f"line {node.lineno}: {node.id}"
                for node in ast.walk(tree)
                if isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id not in defined
            }
        )
        assert not undefined, "undefined names in clustering.py:\n" + "\n".join(undefined)


class TestSemanticClusterer:
    """Exercise the engine that #3906 made unreachable."""

    def test_constructs_with_defaults(self, clustering) -> None:
        clusterer = clustering.SemanticClusterer()
        assert clusterer.linkage == "ward"
        assert clusterer.metric == "precomputed"
        assert clusterer.is_fitted is False
        assert clusterer.labels is None

    def test_fit_hierarchical_separates_two_blocks(
        self, clustering, two_block_similarity
    ) -> None:
        clusterer = clustering.SemanticClusterer(linkage="average")
        labels = clusterer.fit_hierarchical(two_block_similarity, n_clusters=2)

        assert clusterer.is_fitted is True
        assert len(labels) == 6
        assert len(set(labels)) == 2
        # Documents 0-2 belong together, and 3-5 belong together.
        assert len(set(labels[:3])) == 1
        assert len(set(labels[3:])) == 1
        assert labels[0] != labels[3]

    def test_fit_hierarchical_leaves_the_input_untouched(
        self, clustering, two_block_similarity
    ) -> None:
        """``fit_hierarchical`` derives a distance matrix; the caller's stays put."""
        original = two_block_similarity.copy()
        clustering.SemanticClusterer(linkage="average").fit_hierarchical(
            two_block_similarity, n_clusters=2
        )
        assert clustering.np.array_equal(two_block_similarity, original)

    def test_cluster_membership_maps_names(self, clustering, two_block_similarity) -> None:
        names = ["a.pdf", "b.pdf", "c.pdf", "x.pdf", "y.pdf", "z.pdf"]
        clusterer = clustering.SemanticClusterer(linkage="average")
        clusterer.fit_hierarchical(two_block_similarity, n_clusters=2)

        membership = clusterer.get_cluster_membership(names)
        assert sum(len(v) for v in membership.values()) == 6
        groups = sorted(sorted(v) for v in membership.values())
        assert groups == [["a.pdf", "b.pdf", "c.pdf"], ["x.pdf", "y.pdf", "z.pdf"]]

    def test_cluster_membership_is_empty_before_fitting(self, clustering) -> None:
        assert clustering.SemanticClusterer().get_cluster_membership(["a.pdf"]) == {}

    def test_suspicious_clusters_flag_the_dense_groups(
        self, clustering, two_block_similarity
    ) -> None:
        """Both blocks sit far above the 0.70 default, so both are suspicious."""
        clusterer = clustering.SemanticClusterer(linkage="average")
        clusterer.fit_hierarchical(two_block_similarity, n_clusters=2)
        assert sorted(clusterer.get_suspicious_clusters()) == [0, 1]

    def test_suspicious_clusters_respect_the_threshold(
        self, clustering, two_block_similarity
    ) -> None:
        clusterer = clustering.SemanticClusterer(linkage="average")
        clusterer.fit_hierarchical(two_block_similarity, n_clusters=2)
        assert clusterer.get_suspicious_clusters(similarity_threshold=0.99) == []

    def test_suspicious_clusters_empty_before_fitting(self, clustering) -> None:
        assert clustering.SemanticClusterer().get_suspicious_clusters() == []

    def test_centroids_have_one_row_per_cluster(
        self, clustering, two_block_similarity
    ) -> None:
        clusterer = clustering.SemanticClusterer(linkage="average")
        clusterer.fit_hierarchical(two_block_similarity, n_clusters=2)
        centroids = clusterer.get_cluster_centroids(two_block_similarity)
        assert centroids.shape == (2, 6)

    def test_centroids_empty_before_fitting(self, clustering) -> None:
        centroids = clustering.SemanticClusterer().get_cluster_centroids(
            clustering.np.zeros((3, 3))
        )
        assert centroids.size == 0

    def test_metrics_are_populated_after_fitting(
        self, clustering, two_block_similarity
    ) -> None:
        clusterer = clustering.SemanticClusterer(linkage="average")
        clusterer.fit_hierarchical(two_block_similarity, n_clusters=2)
        metrics = clusterer.cluster_metrics
        assert metrics["n_clusters"] == 2
        assert sorted(metrics["cluster_sizes"]) == [3, 3]


class TestPatternEvolutionTracker:
    """``add_scan`` needed ``datetime``; ``get_evolution_data`` needed ``pd``."""

    def test_add_scan_defaults_the_timestamp(self, clustering, two_block_similarity) -> None:
        tracker = clustering.PatternEvolutionTracker()
        tracker.add_scan(two_block_similarity, ["a", "b", "c", "x", "y", "z"])
        assert len(tracker.history) == 1
        assert isinstance(tracker.history[0]["timestamp"], clustering.datetime)

    def test_evolution_data_is_an_empty_frame_when_unused(self, clustering) -> None:
        frame = clustering.PatternEvolutionTracker().get_evolution_data()
        assert isinstance(frame, clustering.pd.DataFrame)
        assert frame.empty

    def test_evolution_data_has_one_row_per_scan(
        self, clustering, two_block_similarity
    ) -> None:
        tracker = clustering.PatternEvolutionTracker()
        names = ["a", "b", "c", "x", "y", "z"]
        tracker.add_scan(two_block_similarity, names)
        tracker.add_scan(two_block_similarity, names)

        frame = tracker.get_evolution_data()
        assert len(frame) == 2
        for column in ("timestamp", "n_documents", "avg_similarity", "max_similarity"):
            assert column in frame.columns
        assert set(frame["n_documents"]) == {6}

    def test_emerging_patterns_is_empty_without_history(self, clustering) -> None:
        assert clustering.PatternEvolutionTracker().get_emerging_patterns() == []

    def test_emerging_patterns_uses_the_lookback_window(
        self, clustering, two_block_similarity
    ) -> None:
        """Exercises the ``datetime.now() - timedelta(...)`` cutoff."""
        tracker = clustering.PatternEvolutionTracker()
        tracker.add_scan(two_block_similarity, ["a", "b", "c", "x", "y", "z"])
        assert isinstance(tracker.get_emerging_patterns(lookback_days=7), list)
        assert tracker.get_emerging_patterns(lookback_days=0) == []
