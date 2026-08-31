"""
tests/app/components/test_tag_renderer.py
-----------------------------------------
Comprehensive unit tests for the tag rendering UI components (Issue #2812).

Verifies that low-confidence tags receive warning icons, muted colors,
and that the HTML output is safe from XSS vulnerabilities.
"""

from unittest.mock import patch

from app.components.tag_renderer import (
    render_low_confidence_warning,
    render_tag,
    render_tag_collection,
)
from src.core.models.categorization import DocumentTag, TagSource


class TestRenderTag:
    """Test suite for the render_tag HTML generation."""

    def test_high_confidence_tag_no_warning_icon(self):
        """Verify high confidence tags do not include the warning icon."""
        tag = DocumentTag(
            name="Machine Learning", source=TagSource.AI_GENERATED, confidence=0.95
        )
        html_out = render_tag(tag)

        assert "⚠️" not in html_out
        assert "tag-low-confidence" not in html_out
        assert "Machine Learning" in html_out

    def test_low_confidence_tag_has_warning_icon(self):
        """Verify low confidence tags (<0.6) include the warning icon."""
        tag = DocumentTag(
            name="Quantum Physics", source=TagSource.AI_GENERATED, confidence=0.40
        )
        html_out = render_tag(tag)

        assert "⚠️" in html_out
        assert "tag-low-confidence" in html_out
        assert "Quantum Physics" in html_out

    def test_low_confidence_threshold_boundary(self):
        """Verify the 0.6 threshold boundary is respected."""
        # Exactly 0.6 should NOT be low confidence
        tag_60 = DocumentTag(name="Test", confidence=0.60)
        assert "tag-low-confidence" not in render_tag(tag_60)

        # 0.59 should be low confidence
        tag_59 = DocumentTag(name="Test", confidence=0.59)
        assert "tag-low-confidence" in render_tag(tag_59)

    def test_manual_tag_styling(self):
        """Verify manual tags get the manual source class."""
        tag = DocumentTag(name="Manual Tag", source=TagSource.MANUAL, confidence=1.0)
        html_out = render_tag(tag)

        assert "tag-source-manual" in html_out
        assert "⚠️" not in html_out

    def test_confidence_percentage_display(self):
        """Verify AI tags display confidence percentage when requested."""
        tag = DocumentTag(name="AI Tag", source=TagSource.AI_GENERATED, confidence=0.85)

        html_with_conf = render_tag(tag, show_confidence=True)
        assert "(85%)" in html_with_conf

        html_without_conf = render_tag(tag, show_confidence=False)
        assert "(85%)" not in html_without_conf

    def test_xss_prevention_in_tag_name(self):
        """Verify tag names are HTML-escaped to prevent XSS."""
        malicious_name = '<script>alert("xss")</script>'
        tag = DocumentTag(name=malicious_name)
        html_out = render_tag(tag)

        # The raw script tag should NOT be in the output
        assert "<script>" not in html_out
        # It should be escaped
        assert "&lt;script&gt;" in html_out

    def test_css_classes_applied_correctly(self):
        """Verify all expected CSS classes are present in the span."""
        tag = DocumentTag(name="Test", source=TagSource.AI_GENERATED, confidence=0.3)
        html_out = render_tag(tag)

        assert (
            'class="tag-badge tag-source-ai_generated tag-low-confidence"' in html_out
        )

    def test_tag_badge_has_aria_button_attrs(self):
        tag = DocumentTag(name="Machine Learning", confidence=0.9)
        html_out = render_tag(tag)
        assert 'role="button"' in html_out
        assert 'aria-label="Machine Learning"' in html_out


class TestRenderTagCollection:
    """Test suite for rendering collections of tags in Streamlit."""

    @patch("app.components.tag_renderer.st")
    def test_renders_empty_collection_silently(self, mock_st):
        """Verify empty collections do not render anything."""
        render_tag_collection([])
        mock_st.markdown.assert_not_called()

    @patch("app.components.tag_renderer.st")
    def test_injects_css_once(self, mock_st):
        """Verify the CSS style block is injected via st.markdown."""
        tags = [DocumentTag(name="Tag1"), DocumentTag(name="Tag2")]
        render_tag_collection(tags)

        # First call should be the CSS injection
        css_call = mock_st.markdown.call_args_list[0]
        assert "<style>" in css_call[0][0]
        assert ".tag-low-confidence" in css_call[0][0]

    @patch("app.components.tag_renderer.st")
    def test_renders_title_if_provided(self, mock_st):
        """Verify the title is rendered when provided."""
        tags = [DocumentTag(name="Tag1")]
        render_tag_collection(tags, title="Document Tags")

        # Check if title was rendered
        calls = [call[0][0] for call in mock_st.markdown.call_args_list]
        assert any("Document Tags" in c for c in calls)

    @patch("app.components.tag_renderer.st")
    def test_renders_multiple_tags_inline(self, mock_st):
        """Verify multiple tags are rendered in a single HTML block."""
        tags = [
            DocumentTag(name="Tag1", confidence=0.9),
            DocumentTag(name="Tag2", confidence=0.4),
        ]
        render_tag_collection(tags)

        # The last markdown call should contain both tags
        final_html = mock_st.markdown.call_args_list[-1][0][0]
        assert "Tag1" in final_html
        assert "Tag2" in final_html
        assert "⚠️" in final_html  # Tag2 is low confidence


class TestRenderLowConfidenceWarning:
    """Test suite for the low confidence summary warning."""

    @patch("app.components.tag_renderer.st")
    def test_renders_warning_for_multiple_tags(self, mock_st):
        """Verify warning is rendered when count > 0."""
        render_low_confidence_warning(3)
        mock_st.warning.assert_called_once()
        assert "3 AI-generated tags" in mock_st.warning.call_args[0][0]

    @patch("app.components.tag_renderer.st")
    def test_renders_singular_warning(self, mock_st):
        """Verify singular grammar when count == 1."""
        render_low_confidence_warning(1)
        assert "1 AI-generated tag" in mock_st.warning.call_args[0][0]
        assert "tags" not in mock_st.warning.call_args[0][0]

    @patch("app.components.tag_renderer.st")
    def test_no_warning_for_zero_count(self, mock_st):
        """Verify no warning is rendered when count == 0."""
        render_low_confidence_warning(0)
        mock_st.warning.assert_not_called()
