"""
src/utils/export_tii_xml.py
---------------------------
Export engine for Turnitin-compatible Originality XML reports.

Generates XML schemas containing document metadata, highlight coordinates,
and similarity scores that can be ingested by Turnitin or compatible
LMS platforms for archival and review.

The schema is advertised with ``xsi:noNamespaceSchemaLocation`` rather than a
default ``xmlns``. The originality schema has no target namespace, so declaring
one would qualify every element name and force each consumer to spell out
``{http://www.turnitin.com/...}submission`` just to reach a child.
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Any, Optional
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

XSI_NAMESPACE = "http://www.w3.org/2001/XMLSchema-instance"
TII_SCHEMA_LOCATION = (
    "http://www.turnitin.com/static/resources/files/turnitin_sdk_v1p0p0.xsd"
)
TII_SCHEMA_VERSION = "1.0.0"


def _local_name(tag: str) -> str:
    """Return an element tag without its ``{namespace}`` prefix, if any.

    Reports written before the schema declaration was corrected carry a default
    ``xmlns``, which ``ElementTree`` expands into every tag it parses. Matching
    on the local name keeps those readable instead of failing validation over a
    prefix that says nothing about the document's contents.
    """
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _find_child(parent: ET.Element, name: str) -> Optional[ET.Element]:
    """Find a direct child by local name, ignoring any namespace prefix."""
    for child in parent:
        if _local_name(child.tag) == name:
            return child
    return None


def generate_tii_xml(report_data: dict[str, Any], include_text: bool = True) -> str:
    """Generate a Turnitin-compatible Originality XML report.

    Args:
        report_data: Dictionary containing:
            - 'document_id': Unique document identifier.
            - 'author': Author name.
            - 'title': Document title.
            - 'submission_date': ISO 8601 timestamp.
            - 'similarity_score': Overall similarity percentage (0-100).
            - 'matches': List of match dictionaries with 'source', 'score', 'start', 'end'.
            - 'text_content': Full text of the document (optional).
        include_text: Whether to include the full text in the XML.

    Returns:
        A formatted XML string.
    """
    root = ET.Element("originalityReport")
    root.set("xmlns:xsi", XSI_NAMESPACE)
    root.set("xsi:noNamespaceSchemaLocation", TII_SCHEMA_LOCATION)
    root.set("version", TII_SCHEMA_VERSION)

    # Submission metadata
    submission = ET.SubElement(root, "submission")
    ET.SubElement(submission, "id").text = str(
        report_data.get("document_id", "unknown")
    )
    ET.SubElement(submission, "title").text = report_data.get("title", "Untitled")
    ET.SubElement(submission, "author").text = report_data.get("author", "Unknown")
    ET.SubElement(submission, "date").text = report_data.get(
        "submission_date", datetime.now(timezone.utc).isoformat()
    )

    # Overall score. This belongs directly under the root: it describes the
    # report rather than the submission's metadata, and it is where both
    # validate_tii_xml() and the ingest side look for it.
    score_elem = ET.SubElement(root, "overallSimilarity")
    score_elem.text = str(int(report_data.get("similarity_score", 0)))
    score_elem.set("unit", "percent")

    # Matches/Sources
    matches_elem = ET.SubElement(root, "matches")
    matches = report_data.get("matches", [])

    for match in matches:
        match_elem = ET.SubElement(matches_elem, "match")
        ET.SubElement(match_elem, "source").text = match.get("source", "Internet")
        ET.SubElement(match_elem, "score").text = str(int(match.get("score", 0)))

        # Highlight coordinates
        if "start" in match and "end" in match:
            highlight = ET.SubElement(match_elem, "highlight")
            highlight.set("start", str(match["start"]))
            highlight.set("end", str(match["end"]))

    # Full text (optional)
    if include_text and "text_content" in report_data:
        text_elem = ET.SubElement(root, "text")
        text_elem.text = report_data["text_content"]

    # Pretty print the XML
    rough_string = ET.tostring(root, encoding="unicode")
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def validate_tii_xml(xml_string: str) -> bool:
    """Basic validation to ensure the XML is well-formed and contains required tags.

    Tags are matched on their local name, so a report carrying a default
    ``xmlns`` — anything written before the schema declaration was corrected —
    validates on the same terms as one written today.
    """
    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError:
        return False

    if _local_name(root.tag) != "originalityReport":
        return False

    submission = _find_child(root, "submission")
    if submission is None:
        return False

    # Older reports nested the score inside <submission>; accept it in either
    # place so previously archived exports still validate. Compared with `is
    # None` rather than `or`: an Element with no children is falsy, and
    # <overallSimilarity> never has any.
    score = _find_child(root, "overallSimilarity")
    if score is None:
        score = _find_child(submission, "overallSimilarity")
    if score is None:
        return False

    return True
