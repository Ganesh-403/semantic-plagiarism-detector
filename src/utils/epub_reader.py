"""Bounded EPUB text extraction in manifest reading order, without extraction to disk."""

from __future__ import annotations

import io
import posixpath
import zipfile
from urllib.parse import unquote

from bs4 import BeautifulSoup
from defusedxml import ElementTree

MAX_EXPANDED_BYTES = 25_000_000
MAX_MEMBERS = 2000


def extract_epub_text(source) -> str:
    if isinstance(source, (bytes, bytearray, memoryview)):
        source = io.BytesIO(source)
    with zipfile.ZipFile(source) as archive:
        members = archive.infolist()
        if len(members) > MAX_MEMBERS or sum(m.file_size for m in members) > MAX_EXPANDED_BYTES:
            raise ValueError("EPUB exceeds expanded size or member limit")
        names = {m.filename for m in members}
        if len(names) != len(members):
            raise ValueError("EPUB contains ambiguous duplicate members")
        container = ElementTree.fromstring(archive.read("META-INF/container.xml"))
        rootfile = container.find(".//{*}rootfile")
        if rootfile is None or not rootfile.get("full-path"):
            raise ValueError("EPUB has no package document")
        package_path = rootfile.get("full-path")
        package = ElementTree.fromstring(archive.read(package_path))
        manifest = {item.get("id"): item for item in package.findall("./{*}manifest/{*}item")}
        spine = package.findall("./{*}spine/{*}itemref")
        if not spine:
            raise ValueError("EPUB has no reading order")
        parts = []
        for itemref in spine:
            item = manifest.get(itemref.get("idref"))
            if item is None:
                raise ValueError("EPUB spine references a missing manifest item")
            if item.get("media-type") not in ("application/xhtml+xml", "text/html"):
                continue
            href = unquote(item.get("href", "").split("#", 1)[0])
            path = posixpath.normpath(posixpath.join(posixpath.dirname(package_path), href))
            if path.startswith(("../", "/")) or ":" in path or "\\" in path:
                raise ValueError("Unsafe EPUB content path")
            soup = BeautifulSoup(archive.read(path), "html.parser")
            for element in soup(["script", "style"]):
                element.decompose()
            body = soup.body or soup
            text = body.get_text(" ", strip=True)
            if text:
                parts.append(text)
        return "\n\n".join(parts)
