from unittest.mock import Mock

from src.visualization.analytics import (
    build_visualization_lazily,
)


def test_factory_is_not_called_when_visualization_is_disabled():
    factory = Mock(return_value="figure")

    result = build_visualization_lazily(False, factory)

    assert result is None
    factory.assert_not_called()


def test_factory_is_called_once_when_visualization_is_enabled():
    figure = object()
    factory = Mock(return_value=figure)

    result = build_visualization_lazily(True, factory)

    assert result is figure
    factory.assert_called_once_with()


def test_factory_exception_is_not_hidden():
    factory = Mock(side_effect=RuntimeError("render failed"))

    try:
        build_visualization_lazily(True, factory)
    except RuntimeError as exc:
        assert str(exc) == "render failed"
    else:
        raise AssertionError("Expected RuntimeError")
