# MIT License
#
# Copyright (c) 2026 Ganesh Kambli
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
tests/utils/test_tii_xml_roundtrip_issue_3635.py
-------------------------------------------------
Regression tests for Issue #3635.

``validate_tii_xml()`` rejected every report ``generate_tii_xml()`` produced, for
two independent reasons:

* the generator declared the schema with a plain ``xmlns``, which ``ElementTree``
  expands into a real namespace on re-parse, so the root tag came back as
  ``{http://www.turnitin.com/...}originalityReport`` and never matched the
  validator's bare-string comparison;
* ``overallSimilarity`` was written inside ``<submission>`` but looked for
  directly under the root, so the check would have failed even with the
  namespace sorted out.

The load-bearing assertion in this file is the round-trip — anything this module
writes, this module must accept. A validator that cannot recognise its own
output is a hard-coded ``False`` with extra steps.
"""

import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import pytest

from src.utils.export_tii_xml import (
    TII_SCHEMA_LOCATION,
    TII_SCHEMA_VERSION,
    XSI_NAMESPACE,
    generate_tii_xml,
    validate_tii_xml,
)

SAMPLE_REPORT = {
    "document_id": "doc_123",
    "title": "Test Essay",
    "author": "Alice",
    "submission_date": "2026-08-25T12:00:00+00:00",
    "similarity_score": 25,
    "matches": [{"source": "Internet", "score": 10, "start": 0, "end": 50}],
}


class TestRoundTrip:
    """Whatever the generator writes, the validator must accept."""

    def test_full_report_validates(self):
        assert validate_tii_xml(generate_tii_xml(SAMPLE_REPORT)) is True

    def test_minimal_report_validates(self):
        assert validate_tii_xml(generate_tii_xml({})) is True

    @pytest.mark.parametrize(
        "report",
        [
            {"document_id": "1"},
            {"similarity_score": 0},
            {"similarity_score": 100},
            {"matches": []},
            {"matches": [{"source": "Wiki", "score": 20}]},
            {"text_content": "The full body of the essay."},
            SAMPLE_REPORT,
        ],
    )
    def test_every_shape_validates(self, report):
        assert validate_tii_xml(generate_tii_xml(report)) is True

    def test_validates_with_text_excluded(self):
        report = dict(SAMPLE_REPORT, text_content="body")

        assert validate_tii_xml(generate_tii_xml(report, include_text=False)) is True


class TestUnqualifiedElementNames:
    """Consumers read plain tags; the schema declaration must not qualify them."""

    def test_root_tag_is_not_namespaced(self):
        root = ET.fromstring(generate_tii_xml(SAMPLE_REPORT))

        assert root.tag == "originalityReport"

    def test_children_are_reachable_without_a_namespace_prefix(self):
        root = ET.fromstring(generate_tii_xml(SAMPLE_REPORT))

        assert root.find("submission") is not None
        assert root.find("overallSimilarity") is not None
        assert root.find("matches") is not None

    def test_submission_fields_are_reachable(self):
        root = ET.fromstring(generate_tii_xml(SAMPLE_REPORT))
        submission = root.find("submission")

        assert submission.find("id").text == "doc_123"
        assert submission.find("title").text == "Test Essay"
        assert submission.find("author").text == "Alice"
        assert submission.find("date").text == "2026-08-25T12:00:00+00:00"

    def test_schema_is_still_declared(self):
        """Dropping the default xmlns must not drop the schema reference."""
        root = ET.fromstring(generate_tii_xml(SAMPLE_REPORT))

        assert (
            root.get(f"{{{XSI_NAMESPACE}}}noNamespaceSchemaLocation")
            == TII_SCHEMA_LOCATION
        )

    def test_version_attribute_is_present(self):
        root = ET.fromstring(generate_tii_xml(SAMPLE_REPORT))

        assert root.get("version") == TII_SCHEMA_VERSION


class TestOverallSimilarityPlacement:
    """The score belongs where the validator and the ingest side look for it."""

    def test_score_is_a_direct_child_of_the_root(self):
        root = ET.fromstring(generate_tii_xml(SAMPLE_REPORT))

        assert root.find("overallSimilarity") is not None

    def test_score_is_not_duplicated_inside_submission(self):
        root = ET.fromstring(generate_tii_xml(SAMPLE_REPORT))

        assert root.find("submission/overallSimilarity") is None

    @pytest.mark.parametrize(
        "score, expected", [(0, "0"), (25, "25"), (99.7, "99"), (100, "100")]
    )
    def test_score_is_rendered_as_a_whole_percentage(self, score, expected):
        root = ET.fromstring(generate_tii_xml({"similarity_score": score}))

        assert root.find("overallSimilarity").text == expected

    def test_score_carries_its_unit(self):
        root = ET.fromstring(generate_tii_xml(SAMPLE_REPORT))

        assert root.find("overallSimilarity").get("unit") == "percent"


class TestMatches:
    """Match highlights survive the round-trip."""

    def test_single_match_is_emitted(self):
        root = ET.fromstring(generate_tii_xml(SAMPLE_REPORT))

        assert len(root.find("matches").findall("match")) == 1

    def test_multiple_matches_are_emitted(self):
        report = {
            "matches": [
                {"source": "Internet", "score": 10, "start": 0, "end": 50},
                {"source": "Journal", "score": 30, "start": 80, "end": 120},
                {"source": "Student Repository", "score": 5},
            ]
        }

        root = ET.fromstring(generate_tii_xml(report))

        assert len(root.find("matches").findall("match")) == 3

    def test_highlight_coordinates_are_preserved(self):
        root = ET.fromstring(generate_tii_xml(SAMPLE_REPORT))
        highlight = root.find("matches/match/highlight")

        assert highlight.get("start") == "0"
        assert highlight.get("end") == "50"

    def test_match_without_coordinates_omits_the_highlight(self):
        root = ET.fromstring(generate_tii_xml({"matches": [{"source": "Wiki"}]}))

        assert root.find("matches/match/highlight") is None

    def test_empty_match_list_still_emits_the_container(self):
        root = ET.fromstring(generate_tii_xml({"matches": []}))

        assert root.find("matches") is not None
        assert root.find("matches").findall("match") == []


class TestValidatorRejections:
    """The validator must still say no to documents that are actually wrong."""

    def test_rejects_malformed_xml(self):
        assert validate_tii_xml("<originalityReport><submission>") is False

    def test_rejects_empty_string(self):
        assert validate_tii_xml("") is False

    def test_rejects_wrong_root_element(self):
        assert validate_tii_xml("<report><submission/></report>") is False

    def test_rejects_missing_submission(self):
        xml = "<originalityReport><overallSimilarity>10</overallSimilarity></originalityReport>"

        assert validate_tii_xml(xml) is False

    def test_rejects_missing_overall_similarity(self):
        assert (
            validate_tii_xml("<originalityReport><submission/></originalityReport>")
            is False
        )

    def test_rejects_a_nested_submission_that_is_not_a_direct_child(self):
        xml = (
            "<originalityReport><wrapper><submission/></wrapper>"
            "<overallSimilarity>10</overallSimilarity></originalityReport>"
        )

        assert validate_tii_xml(xml) is False


class TestLegacyReportsStillValidate:
    """Archived exports carry the old namespace and the old score placement."""

    LEGACY_NS = "http://www.turnitin.com/static/resources/files/turnitin_sdk_v1p0p0.xsd"

    def test_namespaced_report_with_nested_score_validates(self):
        xml = (
            f'<originalityReport xmlns="{self.LEGACY_NS}" version="1.0.0">'
            "<submission><id>1</id>"
            '<overallSimilarity unit="percent">25</overallSimilarity>'
            "</submission><matches/></originalityReport>"
        )

        assert validate_tii_xml(xml) is True

    def test_namespaced_report_with_root_level_score_validates(self):
        xml = (
            f'<originalityReport xmlns="{self.LEGACY_NS}" version="1.0.0">'
            "<submission><id>1</id></submission>"
            '<overallSimilarity unit="percent">25</overallSimilarity>'
            "</originalityReport>"
        )

        assert validate_tii_xml(xml) is True

    def test_namespaced_report_missing_the_score_is_still_rejected(self):
        xml = (
            f'<originalityReport xmlns="{self.LEGACY_NS}">'
            "<submission><id>1</id></submission></originalityReport>"
        )

        assert validate_tii_xml(xml) is False


class TestSubmissionDate:
    """The date field is documented as ISO 8601."""

    def test_supplied_date_is_used_verbatim(self):
        root = ET.fromstring(
            generate_tii_xml({"submission_date": "2020-01-01T00:00:00+00:00"})
        )

        assert root.find("submission/date").text == "2020-01-01T00:00:00+00:00"

    def test_default_date_is_timezone_aware(self):
        """datetime.utcnow() produced a naive stamp with no offset."""
        root = ET.fromstring(generate_tii_xml({}))

        parsed = datetime.fromisoformat(root.find("submission/date").text)

        assert parsed.tzinfo is not None
        assert parsed.utcoffset() == timezone.utc.utcoffset(None)
