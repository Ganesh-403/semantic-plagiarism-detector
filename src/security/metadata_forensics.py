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
src/security/metadata_forensics.py
----------------------------------
Document Provenance and Metadata Forensics Engine.

Extracts and analyzes deep metadata from PDF and DOCX files to detect
suspicious document provenance, such as purchased essays or copied templates.
Analyzes core properties, revision histories, and creation timestamps.
"""

import logging
import re
import struct
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Standard DOCX namespaces for XML parsing
DOCX_CORE_NS = {
    "cp": "http://schemas.openxmlformats.org/package/2006/metadata/core-properties",
    "dc": "http://purl.org/dc/elements/1.1/",
    "dcterms": "http://purl.org/dc/terms/",
    "dcmitype": "http://purl.org/dc/dcmitype/",
    "xsi": "http://www.w3.org/2001/XMLSchema-instance",
}


def extract_docx_metadata(file_bytes: bytes) -> Dict[str, Any]:
    """Extract core and extended properties from a DOCX file.

    DOCX files are ZIP archives containing XML metadata in docProps/core.xml
    and docProps/app.xml. This function extracts author, creation date,
    modification date, revision count, and template information.

    Args:
        file_bytes: Raw bytes of the DOCX file.

    Returns:
        Dictionary containing extracted metadata properties.
    """
    metadata = {
        "creator": None,
        "last_modified_by": None,
        "created": None,
        "modified": None,
        "revision": None,
        "template": None,
        "total_editing_time": None,
        "application": None,
    }

    try:
        with zipfile.ZipFile(file_bytes) as zf:
            # Extract core properties
            if "docProps/core.xml" in zf.namelist():
                core_xml = zf.read("docProps/core.xml")
                root = ET.fromstring(core_xml)  # nosec

                metadata["creator"] = _get_xml_text(root, "dc:creator", DOCX_CORE_NS)
                metadata["last_modified_by"] = _get_xml_text(
                    root, "cp:lastModifiedBy", DOCX_CORE_NS
                )
                metadata["created"] = _get_xml_text(
                    root, "dcterms:created", DOCX_CORE_NS
                )
                metadata["modified"] = _get_xml_text(
                    root, "dcterms:modified", DOCX_CORE_NS
                )
                metadata["revision"] = _get_xml_text(root, "cp:revision", DOCX_CORE_NS)

            # Extract extended properties (app.xml)
            if "docProps/app.xml" in zf.namelist():
                app_xml = zf.read("docProps/app.xml")
                app_root = ET.fromstring(app_xml)  # nosec
                app_ns = {
                    "ep": "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
                }

                metadata["template"] = _get_xml_text(app_root, "ep:Template", app_ns)
                metadata["total_editing_time"] = _get_xml_text(
                    app_root, "ep:TotalTime", app_ns
                )
                metadata["application"] = _get_xml_text(
                    app_root, "ep:Application", app_ns
                )

    except zipfile.BadZipFile:
        logger.error("Invalid DOCX file: Not a valid ZIP archive.")
    except ET.ParseError as e:
        logger.error("Failed to parse DOCX XML metadata: %s", e)
    except Exception as e:
        logger.error("Unexpected error extracting DOCX metadata: %s", e)

    return metadata


def extract_pdf_metadata(file_bytes: bytes) -> Dict[str, Any]:
    """Extract basic metadata from a PDF file using byte-level parsing.

    This is a lightweight extraction method that searches for standard PDF
    Info dictionary keys (/Author, /CreationDate, /ModDate, /Creator, /Producer)
    without requiring heavy external PDF libraries.

    Args:
        file_bytes: Raw bytes of the PDF file.

    Returns:
        Dictionary containing extracted metadata properties.
    """
    metadata = {
        "author": None,
        "creator": None,
        "producer": None,
        "creation_date": None,
        "mod_date": None,
    }

    try:
        # Convert bytes to string for regex matching, ignoring decode errors
        text = file_bytes.decode("latin-1", errors="ignore")

        # Regex patterns for PDF Info dictionary
        patterns = {
            "author": r"/Author\s*\((.*?)\)",
            "creator": r"/Creator\s*\((.*?)\)",
            "producer": r"/Producer\s*\((.*?)\)",
            "creation_date": r"/CreationDate\s*\((.*?)\)",
            "mod_date": r"/ModDate\s*\((.*?)\)",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                metadata[key] = match.group(1).strip()

    except Exception as e:
        logger.error("Unexpected error extracting PDF metadata: %s", e)

    return metadata


def _get_xml_text(
    root: ET.Element, tag: str, namespaces: Dict[str, str]
) -> Optional[str]:
    """Helper to safely extract text from an XML element."""
    elem = root.find(tag, namespaces)
    return elem.text if elem is not None and elem.text else None


def parse_pdf_date(date_str: Optional[str]) -> Optional[datetime]:
    """Parse a PDF date string (e.g., D:20231025120000Z) into a datetime object."""
    if not date_str:
        return None

    # Remove 'D:' prefix if present
    if date_str.startswith("D:"):
        date_str = date_str[2:]

    # Basic parsing for YYYYMMDDHHMMSS
    try:
        # Clean up timezone offsets for simple parsing
        clean_str = re.sub(r"[Z\+\-\'].*$", "", date_str)
        if len(clean_str) >= 14:
            return datetime.strptime(clean_str[:14], "%Y%m%d%H%M%S")
        elif len(clean_str) >= 8:
            return datetime.strptime(clean_str[:8], "%Y%m%d")
    except ValueError:
        logger.debug("Failed to parse PDF date: %s", date_str)

    return None
