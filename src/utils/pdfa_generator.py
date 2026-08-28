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
src/utils/pdfa_generator.py
---------------------------
Export engine for PDF/A-1b compliant metadata and archival preparation.

Generates the XMP (Extensible Metadata Platform) metadata packets required
for PDF/A compliance, including document provenance, conformance level,
and archival timestamps.

Note: Full PDF/A binary compliance (font embedding, color profiles) requires
a specialized PDF library (e.g., pikepdf). This module focuses on generating
the compliant metadata structure and providing a wrapper for injection.
"""

import logging
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any, Dict
from xml.dom import minidom

logger = logging.getLogger(__name__)


def generate_xmp_metadata(
    title: str,
    author: str,
    creation_date: datetime,
    modification_date: datetime,
    producer: str = "Semantic Plagiarism Detector",
) -> str:
    """Generate an XMP metadata packet for PDF/A-1b compliance.

    The XMP packet is an RDF/XML structure that must be embedded in the
    PDF file header for it to be recognized as PDF/A compliant.

    Args:
        title: Document title.
        author: Document author.
        creation_date: Original creation timestamp.
        modification_date: Last modification timestamp.
        producer: Software that produced the PDF.

    Returns:
        A string containing the XMP RDF/XML packet.
    """
    # XMP namespaces
    NS_RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    NS_DC = "http://purl.org/dc/elements/1.1/"
    NS_XMP = "http://ns.adobe.com/xap/1.0/"
    NS_PDFA_ID = "http://www.aiim.org/pdfa/ns/id/"

    # Build RDF structure
    rdf = ET.Element("rdf:RDF")
    rdf.set("xmlns:rdf", NS_RDF)
    rdf.set("xmlns:dc", NS_DC)
    rdf.set("xmlns:xmp", NS_XMP)
    rdf.set("xmlns:pdfaid", NS_PDFA_ID)

    # Description block
    desc = ET.SubElement(rdf, "rdf:Description")
    desc.set("rdf:about", "")

    # Dublin Core metadata
    dc_title = ET.SubElement(desc, "dc:title")
    dc_title_alt = ET.SubElement(dc_title, "rdf:Alt")
    li = ET.SubElement(dc_title_alt, "rdf:li")
    li.set("xml:lang", "x-default")
    li.text = title

    dc_creator = ET.SubElement(desc, "dc:creator")
    dc_creator_seq = ET.SubElement(dc_creator, "rdf:Seq")
    li = ET.SubElement(dc_creator_seq, "rdf:li")
    li.text = author

    # XMP metadata
    ET.SubElement(desc, "xmp:CreateDate").text = creation_date.isoformat()
    ET.SubElement(desc, "xmp:ModifyDate").text = modification_date.isoformat()
    ET.SubElement(desc, "xmp:CreatorTool").text = producer

    # PDF/A Identification (Critical for compliance)
    ET.SubElement(desc, "pdfaid:part").text = "1"
    ET.SubElement(desc, "pdfaid:conformance").text = "B"

    # Serialize to string
    rough_string = ET.tostring(rdf, encoding="unicode")

    # Wrap in XMP packet header/footer
    xmp_packet = f"""<?xpacket begin='\ufeff' id='W5M0MpXN3iX3'?>
{rough_string}
<?xpacket end='w'?>"""

    return xmp_packet


def prepare_pdfa_metadata(report_data: dict[str, Any]) -> str:
    """High-level wrapper to generate XMP metadata from a report dictionary."""
    now = datetime.utcnow()
    return generate_xmp_metadata(
        title=report_data.get("title", "Plagiarism Report"),
        author=report_data.get("author", "System"),
        creation_date=now,
        modification_date=now,
    )
