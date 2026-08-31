"""
src/core/formatting_entropy_extractor.py
----------------------------------------
Document Formatting Entropy Extractor.

Parses DOCX/LaTeX styling hierarchies and computes formatting entropy
to detect template plagiarism and document cloning, even when the
textual content is entirely rewritten.
"""

import re
import math
import zipfile
import xml.etree.ElementTree as ET
import logging
from typing import List, Dict, Any
from collections import Counter

logger = logging.getLogger(__name__)


def extract_docx_styles(file_bytes: bytes) -> List[str]:
    """Extract style definitions and formatting tags from a DOCX file.

    Parses the styles.xml and document.xml inside the DOCX zip to extract
    the hierarchy of styles applied to the document.

    Args:
        file_bytes: Raw bytes of the DOCX file.

    Returns:
        List of style IDs and formatting tags used in the document.
    """
    styles = []
    try:
        with zipfile.ZipFile(file_bytes) as zf:
            # Extract styles from word/styles.xml
            if "word/styles.xml" in zf.namelist():
                styles_xml = zf.read("word/styles.xml")
                root = ET.fromstring(styles_xml)  # nosec
                ns = {
                    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
                }

                for style in root.findall(".//w:style", ns):
                    style_id = style.get(
                        "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}styleId"
                    )
                    if style_id:
                        styles.append(f"style:{style_id}")

            # Extract inline formatting from word/document.xml
            if "word/document.xml" in zf.namelist():
                doc_xml = zf.read("word/document.xml")
                root = ET.fromstring(doc_xml)  # nosec
                ns = {
                    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
                }

                for rPr in root.findall(".//w:rPr", ns):
                    for child in rPr:
                        tag = child.tag.split("}")[-1]
                        styles.append(f"rPr:{tag}")

    except Exception as e:
        logger.error("Failed to parse DOCX styles: %s", e)

    return styles


def extract_latex_macros(source: str) -> List[str]:
    """Extract macro definitions and environment usage from LaTeX source.

    Args:
        source: LaTeX source code string.

    Returns:
        List of macros and environments used.
    """
    macros = []

    # Extract \newcommand and \def
    new_cmd_pattern = re.compile(
        r"\\(?:newcommand|def|renewcommand)\{?\\([a-zA-Z]+)\}?", re.IGNORECASE
    )
    for match in new_cmd_pattern.finditer(source):
        macros.append(f"macro:{match.group(1)}")

    # Extract \begin{environment}
    env_pattern = re.compile(r"\\begin\{([a-zA-Z\*]+)\}", re.IGNORECASE)
    for match in env_pattern.finditer(source):
        macros.append(f"env:{match.group(1)}")

    # Extract document class and packages
    class_pattern = re.compile(
        r"\\documentclass(?:\[.*?\])?\{([a-zA-Z]+)\}", re.IGNORECASE
    )
    for match in class_pattern.finditer(source):
        macros.append(f"class:{match.group(1)}")

    pkg_pattern = re.compile(
        r"\\usepackage(?:\[.*?\])?\{([a-zA-Z0-9,]+)\}", re.IGNORECASE
    )
    for match in pkg_pattern.finditer(source):
        pkgs = match.group(1).split(",")
        for pkg in pkgs:
            macros.append(f"pkg:{pkg.strip()}")

    return macros


def compute_formatting_entropy(styles: List[str]) -> float:
    """Compute Shannon entropy of the formatting styles.

    High entropy indicates a diverse set of styles (custom formatting).
    Low entropy indicates uniform styling (likely a standard template).

    Args:
        styles: List of style tags/macros.

    Returns:
        Shannon entropy value.
    """
    if not styles:
        return 0.0

    counts = Counter(styles)
    total = len(styles)
    entropy = 0.0

    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)

    return round(entropy, 4)
