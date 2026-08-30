"""
tests/api/test_metric_family_schema_issue_4190.py
--------------------------------------------------
Regression tests for the missing ``/metrics/json`` response models (issue #4190).

``src/api/schemas.py`` did not parse. ``MetricSample`` had been reduced to a
mangled one-line string literal with no class body at all, and ``MetricFamily``
was not in the file at any point — even though ``src/api/routers/admin.py``
imports it by name and uses it as the endpoint's ``response_model``. Since every
router imports ``src.api.schemas``, this took down the whole API package, not
just the metrics endpoint.

The models are reconstructed from what ``src.core.metrics.generate_metrics_json()``
actually emits, so the tests below are written against that generator rather
than against a hand-written fixture: whatever the endpoint really returns has to
validate against the models that describe it. The Prometheus text exposition
format is the ground truth on both sides.

``docs/metrics_json.md`` documented the family and sample key as ``namc`` in
four places. Anyone building a dashboard against the documented schema would
have been reading a field the endpoint never sends, so the doc is covered here
too.
"""

import ast
from pathlib import Path

import pytest
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
from prometheus_client.exposition import generate_latest
from prometheus_client.parser import text_string_to_metric_families
from pydantic import BaseModel, ValidationError

from src.api.schemas import MetricFamily, MetricSample

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMAS_PATH = REPO_ROOT / "src" / "api" / "schemas.py"
ADMIN_ROUTER_PATH = REPO_ROOT / "src" / "api" / "routers" / "admin.py"
DOC_PATH = REPO_ROOT / "docs" / "metrics_json.md"


def _payload_from(registry):
    """Rebuild what ``generate_metrics_json()`` produces, against a test registry.

    ``generate_metrics_json()`` reads the global ``REGISTRY``, which carries the
    application's real metrics and anything another test happened to register.
    This mirrors its body over an isolated registry so the assertions below are
    about shape rather than about whatever else is loaded.
    """
    raw = generate_latest(registry).decode("utf-8")
    payload = {}
    for family in text_string_to_metric_families(raw):
        payload[family.name] = {
            "name": family.name,
            "type": family.type,
            "help": family.documentation,
            "metrics": [
                {"name": sample.name, "labels": sample.labels, "value": sample.value}
                for sample in family.samples
            ],
        }
    return payload


@pytest.fixture
def registry():
    return CollectorRegistry()


@pytest.fixture(scope="module")
def schemas_tree():
    return ast.parse(SCHEMAS_PATH.read_text(encoding="utf-8"), filename="schemas.py")


@pytest.fixture(scope="module")
def doc_text():
    return DOC_PATH.read_text(encoding="utf-8")


# ── the module parses and both models exist ────────────────────────────────────


def test_schemas_module_compiles():
    """The whole of the SyntaxError.

    Before the fix: unterminated string literal at line 597.
    """
    compile(SCHEMAS_PATH.read_text(encoding="utf-8"), "schemas.py", "exec")


@pytest.mark.parametrize("name", ["MetricSample", "MetricFamily"])
def test_model_is_defined_as_a_pydantic_model(name):
    model = {"MetricSample": MetricSample, "MetricFamily": MetricFamily}[name]
    assert issubclass(model, BaseModel)


def test_metric_sample_declares_the_generator_s_three_keys():
    assert set(MetricSample.model_fields) == {"name", "labels", "value"}


def test_metric_family_declares_the_generator_s_four_keys():
    assert set(MetricFamily.model_fields) == {"name", "type", "help", "metrics"}


def test_admin_router_imports_metric_family_from_schemas():
    """The import that had no definition behind it.

    ``admin.py`` names ``MetricFamily`` in its ``from src.api.schemas import``
    block; this is the assert that fails if the model is dropped again.
    """
    tree = ast.parse(ADMIN_ROUTER_PATH.read_text(encoding="utf-8"), filename="admin.py")
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "src.api.schemas"
        for alias in node.names
    }
    assert "MetricFamily" in imported


def test_metrics_json_route_is_annotated_with_the_family_model(schemas_tree):
    """``response_model=dict[str, MetricFamily]`` is what drives the OpenAPI doc."""
    source = ADMIN_ROUTER_PATH.read_text(encoding="utf-8")
    assert "response_model=dict[str, MetricFamily]" in source


# ── the models describe what the endpoint really returns ───────────────────────


def test_counter_payload_validates(registry):
    """A counter's accumulated value arrives as a ``_total`` sample.

    Whether a ``_created`` timestamp sample rides along depends on the
    prometheus-client version, so this asserts on ``_total`` and lets every
    other sample simply have to validate.
    """
    counter = Counter(
        "documents", "Documents ingested", ["status"], registry=registry
    )
    counter.labels(status="success").inc(10)

    family = MetricFamily.model_validate(_payload_from(registry)["documents"])

    assert family.name == "documents"
    assert family.type == "counter"
    assert family.help == "Documents ingested"

    total = next(s for s in family.metrics if s.name == "documents_total")
    assert total.value == 10.0
    assert total.labels == {"status": "success"}
    assert all(sample.name.startswith("documents") for sample in family.metrics)


def test_gauge_payload_validates_and_keeps_empty_labels(registry):
    """An unlabelled gauge is one sample carrying the bare family name."""
    Gauge("index_size_bytes", "Index size on disk", registry=registry).set(2048.0)

    family = MetricFamily.model_validate(_payload_from(registry)["index_size_bytes"])

    assert family.type == "gauge"
    assert len(family.metrics) == 1
    assert family.metrics[0].name == "index_size_bytes"
    assert family.metrics[0].labels == {}
    assert family.metrics[0].value == 2048.0


def test_histogram_payload_validates_with_bucket_sum_and_count(registry):
    """Buckets, ``_sum`` and ``_count`` all have to fit one sample model.

    The ``le`` bucket boundary arrives as a label, which is why ``labels`` is
    typed as a plain string map rather than anything narrower.
    """
    histogram = Histogram(
        "pipeline_duration_seconds",
        "Duration of each pipeline stage in seconds",
        ["stage"],
        registry=registry,
    )
    histogram.labels(stage="embed").observe(0.35)

    family = MetricFamily.model_validate(
        _payload_from(registry)["pipeline_duration_seconds"]
    )

    assert family.type == "histogram"
    names = {sample.name for sample in family.metrics}
    assert "pipeline_duration_seconds_bucket" in names
    assert "pipeline_duration_seconds_sum" in names
    assert "pipeline_duration_seconds_count" in names

    bucket = next(
        s for s in family.metrics if s.name == "pipeline_duration_seconds_bucket"
    )
    assert "le" in bucket.labels
    assert bucket.labels["stage"] == "embed"

    total = next(
        s for s in family.metrics if s.name == "pipeline_duration_seconds_count"
    )
    assert total.value == 1.0


def test_a_whole_mixed_payload_validates_as_the_endpoint_response_type(registry):
    """``dict[str, MetricFamily]`` is the declared response model; check it whole."""
    Counter("uploads", "Uploads", ["status"], registry=registry).labels(
        status="success"
    ).inc()
    Gauge("active_threads", "Active threads", registry=registry).set(4)
    Histogram("stage_seconds", "Stage seconds", registry=registry).observe(0.1)

    payload = _payload_from(registry)
    validated = {
        name: MetricFamily.model_validate(family) for name, family in payload.items()
    }

    assert {"uploads", "active_threads", "stage_seconds"} <= set(validated)
    assert all(isinstance(f, MetricFamily) for f in validated.values())
    assert all(
        isinstance(sample, MetricSample)
        for family in validated.values()
        for sample in family.metrics
    )


def test_empty_registry_serialises_to_an_empty_mapping(registry):
    """No metrics registered means ``{}``, not a malformed family."""
    assert _payload_from(registry) == {}


# ── field defaults and validation ──────────────────────────────────────────────


def test_labels_default_to_an_empty_mapping():
    """Unlabelled samples omit the key entirely rather than sending null."""
    assert MetricSample(name="active_threads", value=4.0).labels == {}


def test_metrics_default_to_an_empty_list():
    assert MetricFamily(name="x", type="gauge", help="h").metrics == []


def test_sample_value_is_coerced_to_float():
    """Prometheus values are always floats, including counter timestamps."""
    assert isinstance(MetricSample(name="n", value=3).value, float)


def test_sample_rejects_a_missing_value():
    with pytest.raises(ValidationError):
        MetricSample(name="n")


def test_family_rejects_a_sample_of_the_wrong_shape():
    """A nested sample missing ``value`` must fail at the family boundary."""
    with pytest.raises(ValidationError):
        MetricFamily(
            name="documents",
            type="counter",
            help="h",
            metrics=[{"name": "documents_total", "labels": {}}],
        )


def test_family_round_trips_through_json():
    """Serialising and reparsing must preserve every documented key."""
    family = MetricFamily(
        name="documents",
        type="counter",
        help="Documents ingested",
        metrics=[MetricSample(name="documents_total", labels={}, value=10.0)],
    )
    assert MetricFamily.model_validate_json(family.model_dump_json()) == family


# ── the documentation matches the payload ──────────────────────────────────────


def test_doc_does_not_use_the_misspelled_key(doc_text):
    """``namc`` appeared four times where the endpoint sends ``name``."""
    assert "namc" not in doc_text


def test_doc_example_keys_are_all_real_model_fields(doc_text):
    """Every key in the doc's JSON blocks must exist on one of the two models.

    This is the check that catches a typo like ``namc`` generically, rather
    than pinning the one spelling that happened to be wrong.
    """
    import json
    import re

    known = set(MetricFamily.model_fields) | set(MetricSample.model_fields)
    blocks = re.findall(r"```json\n(.*?)```", doc_text, flags=re.S)
    assert blocks, "the doc should carry at least one JSON example"

    checked = 0
    for block in blocks:
        try:
            parsed = json.loads(block)
        except json.JSONDecodeError:
            continue  # the schema block uses a "a | b" placeholder for the type
        for family in parsed.values():
            assert set(family) <= known, f"unknown keys: {set(family) - known}"
            for sample in family.get("metrics", []):
                assert set(sample) <= known, f"unknown keys: {set(sample) - known}"
            checked += 1
    assert checked, "no example family was parsed out of the doc"


def test_doc_example_validates_against_the_models(doc_text):
    """The published example must actually be a legal response."""
    import json
    import re

    blocks = re.findall(r"```json\n(.*?)```", doc_text, flags=re.S)
    examples = []
    for block in blocks:
        try:
            examples.append(json.loads(block))
        except json.JSONDecodeError:
            continue

    assert examples, "the doc's example output should be valid JSON"
    for example in examples:
        for family in example.values():
            MetricFamily.model_validate(family)
