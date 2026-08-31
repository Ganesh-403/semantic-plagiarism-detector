"""Tests for background-friendly intelligent document categorization."""

import importlib.util
import sys
import types
from pathlib import Path


class _FakeStreamlit(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.cache_calls = 0

    def cache_data(self, **_kwargs):
        def decorator(func):
            cache = {}

            def cached(*args, **kwargs):
                key = (args, tuple(sorted(kwargs.items())))
                if key not in cache:
                    self.cache_calls += 1
                    cache[key] = func(*args, **kwargs)
                return cache[key]

            cached.__wrapped__ = func
            return cached

        return decorator


def _load_module(monkeypatch):
    fake_streamlit = _FakeStreamlit()
    monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)

    for name in ("plotly", "plotly.graph_objects", "plotly.express"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    module_path = (
        Path(__file__).parents[2] / "app" / "components" / "IntelligentTagging_Doc.py"
    )
    spec = importlib.util.spec_from_file_location(
        "intelligent_tagging_doc", module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, fake_streamlit


def test_categorization_analysis_is_cached(monkeypatch):
    module, fake_streamlit = _load_module(monkeypatch)
    manager = module.TagManager()
    categorizer = module.AutoCategorizer(manager, module.IntelligentTagGenerator())
    content = "research study methodology literature review analysis"

    first = categorizer.categorize_document("doc-1", content)
    first_cache_calls = fake_streamlit.cache_calls

    original = module.IntelligentTagGenerator.generate_tags
    monkeypatch.setattr(
        module.IntelligentTagGenerator,
        "generate_tags",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("cache miss")),
    )
    second = categorizer.categorize_document("doc-2", content)

    assert first["category"] == second["category"]
    assert first["assigned_tags"] == second["assigned_tags"]
    assert fake_streamlit.cache_calls == first_cache_calls
    assert original is not None


def test_cached_analysis_does_not_skip_tag_assignment(monkeypatch):
    module, _fake_streamlit = _load_module(monkeypatch)
    manager = module.TagManager()
    categorizer = module.AutoCategorizer(manager, module.IntelligentTagGenerator())
    content = "machine learning neural network model training"

    categorizer.categorize_document("doc-1", content)
    categorizer.categorize_document("doc-2", content)

    assert manager.get_document_tags("doc-1")
    assert manager.get_document_tags("doc-2")


def test_empty_content_short_circuits_before_cached_analysis(monkeypatch):
    module, fake_streamlit = _load_module(monkeypatch)
    manager = module.TagManager()
    categorizer = module.AutoCategorizer(manager, module.IntelligentTagGenerator())

    result = categorizer.categorize_document("empty", "")

    assert result == {"status": "failed", "reason": "No content"}
    assert fake_streamlit.cache_calls == 0
