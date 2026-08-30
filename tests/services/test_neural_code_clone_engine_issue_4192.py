"""
tests/services/test_neural_code_clone_engine_issue_4192.py
------------------------------------------------------------
Regression tests for the four defects in ``src/services/neural_code_clone_engine.py``
(issue #4192). The module did not compile, so its existing test
(``tests/test_neural_code_clone_engine.py``) could not even be collected.

Defect 1 -- ``NeuralCodeCloneEngine`` was absent from the module entirely, while
``src/pages/neural_code_clone_dashboard.py`` imports it by name and calls
``NeuralCodeCloneEngine.analyze_code_pair(...)``.

Defect 2 -- that class's body had leaked into ``NeuralCodeCloneDetector.scan_for_code_clones``,
where it referenced ``cls``, ``ast_sim``, ``token_sim`` and ``classify_clone_type``,
none of which exist in an instance method scanning token sets. It landed at the
wrong indentation, which is the reported ``IndentationError``. The loop
``scan_for_code_clones`` actually needed -- iterating the indexed repositories and
computing the Jaccard intersection over union -- was gone, even though the
surviving code below referenced ``clone_matches``, ``file_id``, ``repo``,
``intersection`` and ``union``.

Defect 3 -- ``NeuralCodeCloneDetector`` had no ``__init__``, so
``self.indexed_code_repositories`` and ``self.similarity_threshold`` were never
initialised and ``NeuralCodeCloneDetector(similarity_threshold=0.70)`` -- exactly
what the existing test does -- would have raised ``TypeError``.

Defect 4 -- ``calculate_token_cosine_similarity`` kept only its guard clause and
fell off the end returning ``None`` for every valid pair of vectors.

The scoring thresholds asserted below are the ones the restored code sets. They
are pinned deliberately: the classifier is the part a future change is most
likely to move silently, and the Type-1..4 taxonomy is a documented contract in
``src/models/neural_code_clone_model.py``.
"""

import ast
from datetime import datetime
from pathlib import Path

import pytest

from src.models.neural_code_clone_model import CodeAstEmbedding, CodeCloneMatch
from src.services.neural_code_clone_engine import (
    NeuralCodeCloneDetector,
    NeuralCodeCloneEngine,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ENGINE_PATH = REPO_ROOT / "src" / "services" / "neural_code_clone_engine.py"
DASHBOARD_PATH = REPO_ROOT / "src" / "pages" / "neural_code_clone_dashboard.py"

REFERENCE_CODE = """
def calculate_factorial(n):
    if n <= 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
"""

# Same shape, every identifier renamed - the textbook Type-2 clone.
RENAMED_CODE = """
def compute_factorial_val(num_val):
    if num_val <= 1:
        return 1
    total_res = 1
    for idx in range(2, num_val + 1):
        total_res *= idx
    return total_res
"""

UNRELATED_CODE = """
class HttpRetryPolicy:
    def should_retry(self, status):
        return status in {500, 502, 503}
"""


def embedding(file_id, *, tokens, complexity, vector, language="python"):
    return CodeAstEmbedding(
        file_id=file_id,
        source_language=language,
        ast_token_count=tokens,
        cyclomatic_complexity=complexity,
        vector_embedding=vector,
    )


@pytest.fixture
def detector():
    return NeuralCodeCloneDetector(similarity_threshold=0.70)


@pytest.fixture
def indexed_detector(detector):
    detector.index_repository_file(
        file_id="FILE-001",
        file_path="src/utils/math_utils.py",
        code_content=REFERENCE_CODE,
        language="python",
    )
    return detector


@pytest.fixture(scope="module")
def engine_tree():
    return ast.parse(ENGINE_PATH.read_text(encoding="utf-8"), filename="engine.py")


# ── the module parses and both classes are present ─────────────────────────────


def test_module_compiles():
    """The whole of the IndentationError.

    Before the fix: unexpected indent at line 93.
    """
    compile(ENGINE_PATH.read_text(encoding="utf-8"), "engine.py", "exec")


def test_both_classes_are_defined(engine_tree):
    """Defect 1. The engine class was missing, not merely broken."""
    classes = [
        node.name for node in engine_tree.body if isinstance(node, ast.ClassDef)
    ]
    assert classes == ["NeuralCodeCloneDetector", "NeuralCodeCloneEngine"]


def test_dashboard_imports_resolve():
    """The dashboard's two imports must both bind to something real."""
    tree = ast.parse(DASHBOARD_PATH.read_text(encoding="utf-8"), filename="dash.py")
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "src.services.neural_code_clone_engine"
        for alias in node.names
    }
    assert imported == {"NeuralCodeCloneEngine"}
    assert hasattr(NeuralCodeCloneEngine, "analyze_code_pair")


def test_scan_method_no_longer_references_the_leaked_names(engine_tree):
    """Defect 2. ``cls`` and friends must be gone from the instance method.

    This is the assert that fails if the foreign block is ever pasted back.
    """
    method = next(
        node
        for node in ast.walk(engine_tree)
        if isinstance(node, ast.FunctionDef) and node.name == "scan_for_code_clones"
    )
    names = {n.id for n in ast.walk(method) if isinstance(n, ast.Name)}
    assert not ({"cls", "ast_sim", "token_sim", "semantic_sim"} & names)


# ── Defect 3: the detector initialises its own state ───────────────────────────


def test_detector_accepts_a_similarity_threshold():
    """The constructor call the existing test makes, which used to TypeError."""
    assert NeuralCodeCloneDetector(similarity_threshold=0.85).similarity_threshold == 0.85


def test_detector_threshold_defaults():
    assert NeuralCodeCloneDetector().similarity_threshold == 0.70


def test_detector_starts_with_an_empty_index(detector):
    assert detector.indexed_code_repositories == {}


def test_two_detectors_do_not_share_an_index(indexed_detector):
    """The index must be per-instance state, not a class attribute."""
    assert NeuralCodeCloneDetector().indexed_code_repositories == {}
    assert "FILE-001" in indexed_detector.indexed_code_repositories


# ── Defect 4: cosine similarity actually computes ──────────────────────────────


def test_cosine_of_a_vector_with_itself_is_one():
    """The case that used to return ``None``."""
    assert NeuralCodeCloneDetector.calculate_token_cosine_similarity(
        [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]
    ) == 1.0


def test_cosine_of_orthogonal_vectors_is_zero():
    assert NeuralCodeCloneDetector.calculate_token_cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_ignores_magnitude():
    """It measures angle: a scaled vector is the same direction."""
    assert NeuralCodeCloneDetector.calculate_token_cosine_similarity(
        [1.0, 2.0, 3.0], [10.0, 20.0, 30.0]
    ) == 1.0


def test_cosine_always_returns_a_float():
    """The return annotation says float; ``None`` broke every caller."""
    result = NeuralCodeCloneDetector.calculate_token_cosine_similarity([1.0, 1.0], [1.0, 0.0])
    assert isinstance(result, float)


@pytest.mark.parametrize(
    "vec1,vec2",
    [
        ([], [1.0]),          # empty left
        ([1.0], []),          # empty right
        ([1.0, 2.0], [1.0]),  # length mismatch
    ],
)
def test_cosine_guard_clauses_still_return_zero(vec1, vec2):
    """The one part that survived must keep working."""
    assert NeuralCodeCloneDetector.calculate_token_cosine_similarity(vec1, vec2) == 0.0


def test_cosine_of_a_zero_vector_is_zero():
    """A zero vector has no direction, so no angle - and no division by zero."""
    assert NeuralCodeCloneDetector.calculate_token_cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


# ── Defect 2: the scan loop is back ────────────────────────────────────────────


def test_indexing_records_the_token_set(indexed_detector):
    entry = indexed_detector.indexed_code_repositories["FILE-001"]
    assert entry["filePath"] == "src/utils/math_utils.py"
    assert entry["language"] == "python"
    assert entry["astTokensCount"] > 0
    assert isinstance(entry["astTokenSet"], set)
    assert len(entry["semanticHash"]) == 64  # SHA-256 hex digest


def test_scan_finds_a_renamed_clone(indexed_detector):
    """The end-to-end case the existing test covers, now reachable."""
    matches = indexed_detector.scan_for_code_clones(RENAMED_CODE, language="python")

    assert len(matches) == 1
    match = matches[0]
    assert match["matchedFileId"] == "FILE-001"
    assert match["matchedFilePath"] == "src/utils/math_utils.py"
    assert match["jaccardSimilarityScore"] >= 0.70
    assert match["detectedCloneType"] in {
        "TYPE_1_EXACT",
        "TYPE_2_RENAMED",
        "TYPE_3_MODIFIED",
    }
    assert match["confidenceGrade"] in {"CRITICAL", "HIGH"}


def test_scanning_identical_code_scores_one(indexed_detector):
    """Identical token sets means intersection == union."""
    matches = indexed_detector.scan_for_code_clones(REFERENCE_CODE, language="python")
    assert matches[0]["jaccardSimilarityScore"] == 1.0
    assert matches[0]["detectedCloneType"] == "TYPE_1_EXACT"
    assert matches[0]["confidenceGrade"] == "CRITICAL"


def test_scan_ignores_code_below_the_threshold(indexed_detector):
    """Unrelated code must not be reported at all."""
    assert indexed_detector.scan_for_code_clones(UNRELATED_CODE, language="python") == []


def test_scan_of_an_empty_index_returns_nothing(detector):
    """No division by zero on an empty union, no crash."""
    assert detector.scan_for_code_clones(REFERENCE_CODE, language="python") == []


def test_scan_of_empty_query_code_returns_nothing(indexed_detector):
    assert indexed_detector.scan_for_code_clones("", language="python") == []


def test_scan_filters_by_language(indexed_detector):
    """``language`` is a real filter, not an ignored parameter.

    Token vocabularies are language-specific, so a Python token set says
    nothing about a Java one. Scanning the very code that was indexed, but
    under a different language, must find nothing.
    """
    assert indexed_detector.scan_for_code_clones(REFERENCE_CODE, language="java") == []


def test_scan_ranks_matches_by_descending_similarity(detector):
    """The sort at the end of the method needs more than one match to mean anything."""
    detector.index_repository_file(
        "EXACT", "a.py", REFERENCE_CODE, language="python"
    )
    detector.index_repository_file(
        "RENAMED", "b.py", RENAMED_CODE, language="python"
    )

    matches = detector.scan_for_code_clones(REFERENCE_CODE, language="python")

    assert len(matches) == 2
    assert matches[0]["matchedFileId"] == "EXACT"
    scores = [m["jaccardSimilarityScore"] for m in matches]
    assert scores == sorted(scores, reverse=True)


def test_threshold_is_honoured():
    """A stricter detector rejects what a lenient one accepts."""
    lenient = NeuralCodeCloneDetector(similarity_threshold=0.10)
    strict = NeuralCodeCloneDetector(similarity_threshold=0.999)
    for det in (lenient, strict):
        det.index_repository_file("F", "a.py", REFERENCE_CODE, language="python")

    assert lenient.scan_for_code_clones(RENAMED_CODE, language="python")
    assert strict.scan_for_code_clones(RENAMED_CODE, language="python") == []


# ── Defect 1: the engine classifies pairs ──────────────────────────────────────


def test_analyze_code_pair_returns_the_domain_model():
    """The dashboard reads ``.clone_type`` and ``.overall_clone_score`` off this."""
    vector = [0.12, 0.45, 0.78, 0.90, 0.33]
    match = NeuralCodeCloneEngine.analyze_code_pair(
        "Snippet_A.py",
        "Snippet_B.py",
        embedding("Snippet_A.py", tokens=100, complexity=3, vector=vector),
        embedding("Snippet_B.py", tokens=100, complexity=3, vector=vector),
    )

    assert isinstance(match, CodeCloneMatch)
    assert match.source_file_id == "Snippet_A.py"
    assert match.target_file_id == "Snippet_B.py"
    assert match.clone_id.startswith("CLONE-")
    assert isinstance(match.detected_at, datetime)
    assert isinstance(match.overall_clone_score, float)


def test_clone_ids_are_unique():
    vector = [0.1, 0.2, 0.3]
    args = (
        embedding("a", tokens=10, complexity=1, vector=vector),
        embedding("b", tokens=10, complexity=1, vector=vector),
    )
    ids = {
        NeuralCodeCloneEngine.analyze_code_pair("a", "b", *args).clone_id
        for _ in range(25)
    }
    assert len(ids) == 25


def test_identical_embeddings_classify_as_type_1():
    vector = [0.12, 0.45, 0.78, 0.90, 0.33]
    match = NeuralCodeCloneEngine.analyze_code_pair(
        "a",
        "b",
        embedding("a", tokens=100, complexity=3, vector=vector),
        embedding("b", tokens=100, complexity=3, vector=vector),
    )
    assert match.clone_type == "Type-1 (Exact)"
    assert match.overall_clone_score == 1.0
    assert match.obfuscation_detected is False


def test_near_identical_embeddings_classify_as_type_2():
    """Same structure, slightly different size - renamed identifiers."""
    match = NeuralCodeCloneEngine.analyze_code_pair(
        "a",
        "b",
        embedding("a", tokens=100, complexity=3, vector=[0.10, 0.20, 0.30, 0.40]),
        embedding("b", tokens=92, complexity=3, vector=[0.10, 0.21, 0.29, 0.41]),
    )
    assert match.clone_type == "Type-2 (Renamed)"
    assert match.obfuscation_detected is False


def test_intact_ast_with_diverged_tokens_flags_obfuscation():
    """Type-3, and the disagreement between the two signals is the tell.

    An AST that stays put while the surface tokens fall away is code that has
    been rewritten to hide the match, so obfuscation is set.
    """
    match = NeuralCodeCloneEngine.analyze_code_pair(
        "a",
        "b",
        embedding("a", tokens=100, complexity=3, vector=[0.10, 0.20, 0.30, 0.40]),
        embedding("b", tokens=50, complexity=9, vector=[0.10, 0.22, 0.28, 0.44]),
    )
    assert match.clone_type == "Type-3 (Modified AST)"
    assert match.token_overlap_score < 0.60
    assert match.obfuscation_detected is True


def test_unrelated_embeddings_are_not_a_clone():
    match = NeuralCodeCloneEngine.analyze_code_pair(
        "a",
        "b",
        embedding("a", tokens=100, complexity=3, vector=[1.0, 0.0, 0.0, 0.0]),
        embedding("b", tokens=20, complexity=12, vector=[0.0, 0.0, 0.0, 1.0]),
    )
    assert match.clone_type == "No Clone Detected"
    assert match.overall_clone_score == 0.0
    assert match.obfuscation_detected is False


@pytest.mark.parametrize(
    "ast_sim,token_sim,semantic_sim,expected_type,expected_obfuscation",
    [
        (1.00, 1.00, 1.00, "Type-1 (Exact)", False),
        (0.99, 0.99, 0.99, "Type-1 (Exact)", False),
        (0.96, 0.90, 0.94, "Type-2 (Renamed)", False),
        (0.85, 0.70, 0.79, "Type-3 (Modified AST)", False),
        (0.85, 0.40, 0.67, "Type-3 (Modified AST)", True),
        (0.50, 0.90, 0.72, "Type-4 (Semantic Equivalent)", True),
        (0.20, 0.20, 0.20, "No Clone Detected", False),
    ],
)
def test_classify_clone_type_boundaries(
    ast_sim, token_sim, semantic_sim, expected_type, expected_obfuscation
):
    """The taxonomy is a documented contract; pin every band of it."""
    clone_type, obfuscation = NeuralCodeCloneEngine.classify_clone_type(
        ast_sim, token_sim, semantic_sim
    )
    assert clone_type == expected_type
    assert obfuscation is expected_obfuscation


# ── the token profile comparison ───────────────────────────────────────────────


def test_token_profile_of_matching_files_is_one():
    vector = [0.1, 0.2, 0.3]
    assert NeuralCodeCloneEngine.compare_token_profiles(
        embedding("a", tokens=100, complexity=3, vector=vector),
        embedding("b", tokens=100, complexity=3, vector=vector),
    ) == 1.0


def test_token_profile_scales_with_the_size_ratio():
    """A 50-token file is half a 100-token one, however alike the vectors."""
    vector = [0.1, 0.2, 0.3]
    assert NeuralCodeCloneEngine.compare_token_profiles(
        embedding("a", tokens=100, complexity=3, vector=vector),
        embedding("b", tokens=50, complexity=3, vector=vector),
    ) == 0.5


def test_complexity_divergence_discounts_the_profile():
    """Equal length with different branching is a different algorithm."""
    vector = [0.1, 0.2, 0.3]
    same_complexity = NeuralCodeCloneEngine.compare_token_profiles(
        embedding("a", tokens=100, complexity=3, vector=vector),
        embedding("b", tokens=100, complexity=3, vector=vector),
    )
    diverged = NeuralCodeCloneEngine.compare_token_profiles(
        embedding("a", tokens=100, complexity=3, vector=vector),
        embedding("b", tokens=100, complexity=13, vector=vector),
    )
    assert diverged < same_complexity


def test_complexity_penalty_is_capped():
    """A wild complexity gap must not drive the score negative."""
    vector = [0.1, 0.2, 0.3]
    score = NeuralCodeCloneEngine.compare_token_profiles(
        embedding("a", tokens=100, complexity=1, vector=vector),
        embedding("b", tokens=100, complexity=500, vector=vector),
    )
    assert score == pytest.approx(1.0 - NeuralCodeCloneEngine.MAX_COMPLEXITY_PENALTY)
    assert score >= 0.0


def test_token_profile_of_empty_files_is_zero():
    """No tokens on either side means no division by zero."""
    vector = [0.1, 0.2, 0.3]
    assert NeuralCodeCloneEngine.compare_token_profiles(
        embedding("a", tokens=0, complexity=0, vector=vector),
        embedding("b", tokens=0, complexity=0, vector=vector),
    ) == 0.0


def test_ensemble_weights_sum_to_one():
    """Otherwise ``overall_clone_score`` could not stay inside 0.0-1.0."""
    total = (
        NeuralCodeCloneEngine.AST_WEIGHT
        + NeuralCodeCloneEngine.TOKEN_WEIGHT
        + NeuralCodeCloneEngine.SEMANTIC_WEIGHT
    )
    assert total == pytest.approx(1.0)


@pytest.mark.parametrize("tokens_b,complexity_b", [(100, 3), (60, 5), (20, 11), (5, 40)])
def test_every_score_stays_within_range(tokens_b, complexity_b):
    """All four scores are documented as 0.0-1.0 on the domain model."""
    match = NeuralCodeCloneEngine.analyze_code_pair(
        "a",
        "b",
        embedding("a", tokens=100, complexity=3, vector=[0.3, 0.4, 0.5, 0.6]),
        embedding("b", tokens=tokens_b, complexity=complexity_b, vector=[0.2, 0.5, 0.4, 0.7]),
    )
    for score in (
        match.ast_similarity_score,
        match.token_overlap_score,
        match.neural_semantic_similarity,
        match.overall_clone_score,
    ):
        assert 0.0 <= score <= 1.0


def test_analysis_is_symmetric_in_score():
    """Swapping the two files must not change how alike they are."""
    a = embedding("a", tokens=100, complexity=3, vector=[0.3, 0.4, 0.5])
    b = embedding("b", tokens=70, complexity=6, vector=[0.35, 0.38, 0.55])

    forward = NeuralCodeCloneEngine.analyze_code_pair("a", "b", a, b)
    backward = NeuralCodeCloneEngine.analyze_code_pair("b", "a", b, a)

    assert forward.overall_clone_score == backward.overall_clone_score
    assert forward.clone_type == backward.clone_type
