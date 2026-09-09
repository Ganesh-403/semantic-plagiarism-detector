"""Project PDF operations backed by BSD/Apache-licensed pypdf and PDFium.

Coordinates exposed to callers use a top-left origin, matching the application's
existing annotation contract. PDF serialization converts them to PDF coordinates.
PDFium calls are serialized because its C API does not support concurrent calls.
"""

from __future__ import annotations

import io
import threading
from contextlib import closing
from pathlib import Path
from dataclasses import dataclass

import pypdfium2 as pdfium
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ArrayObject, FloatObject, NameObject, TextStringObject
from pypdf.annotations import Highlight
from reportlab.pdfgen import canvas

_LOCK = threading.RLock()
PDF_ENCRYPT_AES_256 = 5
PDF_ENCRYPT_NONE = 1
FileDataError = ValueError


@dataclass
class Point:
    x: float
    y: float

    def __iter__(self):
        return iter((self.x, self.y))


class Matrix:
    def __init__(self, *values):
        if len(values) == 2:
            self.values = (values[0], 0, 0, values[1], 0, 0)
        elif len(values) == 6:
            self.values = values
        else:
            raise ValueError("Matrix requires two scale factors or six affine values")


class Rect:
    def __init__(self, *values):
        values = tuple(values[0]) if len(values) == 1 else values
        if len(values) != 4:
            raise ValueError("Rect requires four coordinates")
        self.x0, self.y0, self.x1, self.y1 = map(float, values)

    def __iter__(self):
        return iter((self.x0, self.y0, self.x1, self.y1))

    def __getitem__(self, key):
        return tuple(self)[key]

    def __eq__(self, other):
        try:
            return tuple(self) == tuple(other)
        except TypeError:
            return False

    @property
    def width(self):
        return self.x1 - self.x0

    @property
    def height(self):
        return self.y1 - self.y0

    def contains(self, other):
        r = Rect(other)
        return self.x0 <= r.x0 <= r.x1 <= self.x1 and self.y0 <= r.y0 <= r.y1 <= self.y1

    def intersects(self, other):
        r = Rect(other)
        return max(self.x0, r.x0) < min(self.x1, r.x1) and max(self.y0, r.y0) < min(
            self.y1, r.y1
        )

    def __mul__(self, matrix):
        a, b, c, d, e, f = matrix.values
        pts = [
            (a * x + c * y + e, b * x + d * y + f)
            for x in (self.x0, self.x1)
            for y in (self.y0, self.y1)
        ]
        return Rect(
            min(x for x, y in pts),
            min(y for x, y in pts),
            max(x for x, y in pts),
            max(y for x, y in pts),
        )


class Annotation:
    def __init__(self, page, data):
        self.page, self.data = page, data

    def set_info(self, *, content=None, title=None):
        self.page.parent._invalidate()
        if content is not None:
            self.data[NameObject("/Contents")] = TextStringObject(content)
        if title is not None:
            self.data[NameObject("/T")] = TextStringObject(title)

    @property
    def info(self):
        return {
            "content": str(self.data.get("/Contents", "")),
            "title": str(self.data.get("/T", "")),
        }

    def set_colors(self, *, stroke):
        self.page.parent._invalidate()
        self.data[NameObject("/C")] = ArrayObject([FloatObject(v) for v in stroke])

    @property
    def colors(self):
        return {"stroke": tuple(float(v) for v in self.data.get("/C", []))}

    @property
    def type(self):
        return (
            (8, "Highlight")
            if self.data.get("/Subtype") == "/Highlight"
            else (0, "Text")
        )

    @property
    def rect(self):
        x0, y0, x1, y1 = map(float, self.data["/Rect"])
        return Rect(x0, self.page._height - y1, x1, self.page._height - y0)

    @property
    def vertices(self):
        points = self.data.get("/QuadPoints", [])
        return [(float(points[i]), self.page._height - float(points[i + 1]))
                for i in range(0, len(points), 2)]

    def update(self):
        pass


class Pixmap:
    def __init__(self, image):
        self.image = image
        self.width, self.height = image.size
        self.n = len(image.getbands())
        self.samples = image.tobytes()

    def tobytes(self, output="png"):
        buf = io.BytesIO()
        self.image.save(buf, format=output)
        return buf.getvalue()

    def save(self, path):
        self.image.save(path)


class Page:
    def __init__(self, document, number):
        self.parent, self.number = document, number

    @property
    def _page(self):
        return self.parent._get_writer().pages[self.number]

    @property
    def _height(self):
        return float(self._page.mediabox.height)

    @property
    def rotation(self):
        return self._page.rotation % 360

    def set_rotation(self, rotation):
        self.parent._invalidate()
        self._page.rotation = rotation % 360

    @property
    def rect(self):
        w, h = float(self._page.mediabox.width), self._height
        return Rect(0, 0, h, w) if self.rotation in (90, 270) else Rect(0, 0, w, h)

    @property
    def rotation_matrix(self):
        w, h = float(self._page.mediabox.width), self._height
        return {
            0: Matrix(1, 1),
            90: Matrix(0, 1, -1, 0, h, 0),
            180: Matrix(-1, 0, 0, -1, w, h),
            270: Matrix(0, -1, 1, 0, 0, w),
        }[self.rotation]

    @property
    def derotation_matrix(self):
        a, b, c, d, e, f = self.rotation_matrix.values
        return Matrix(d, -b, -c, a, c * f - d * e, b * e - a * f)

    def get_text(self, option="text", **kwargs):
        with _LOCK, pdfium.PdfDocument(self.parent._render_bytes()) as doc:
            with closing(doc[self.number]) as page, closing(page.get_textpage()) as text:
                value = text.get_text_range()
        if option == "text":
            return value
        if option == "blocks":
            return (
                [(0, 0, self.rect.width, self.rect.height, value, 0, 0)]
                if value
                else []
            )
        raise ValueError(f"Unsupported text representation: {option}")

    def search_for(self, needle, **kwargs):
        rects = []
        with _LOCK, pdfium.PdfDocument(self.parent._render_bytes()) as doc:
            with (
                closing(doc[self.number]) as page,
                closing(page.get_textpage()) as text,
                closing(text.search(needle)) as search,
            ):
                while (match := search.get_next()) is not None:
                    for i in range(text.count_rects(*match)):
                        left, bottom, right, top = text.get_rect(i)
                        rects.append(
                            Rect(left, self._height - top, right, self._height - bottom)
                        )
        return rects

    def add_highlight_annot(self, rect):
        r = Rect(rect)
        y0, y1 = self._height - r.y1, self._height - r.y0
        quad = ArrayObject(
            [FloatObject(v) for v in (r.x0, y1, r.x1, y1, r.x0, y0, r.x1, y0)]
        )
        data = Highlight(rect=(r.x0, y0, r.x1, y1), quad_points=quad)
        self.parent._get_writer().add_annotation(self.number, data)
        self.parent._invalidate()
        return Annotation(self, data)

    def annots(self):
        for ref in self._page.get("/Annots", []):
            yield Annotation(self, ref.get_object())

    def get_pixmap(self, matrix=None, alpha=False, **kwargs):
        scale = matrix.values[0] if matrix else 1
        with _LOCK, pdfium.PdfDocument(self.parent._render_bytes()) as doc:
            with closing(doc[self.number]) as page, closing(page.render(scale=scale)) as bitmap:
                image = bitmap.to_pil().convert("RGBA" if alpha else "RGB").copy()
        return Pixmap(image)

    def insert_text(self, point, text, fontsize=11, fontname="Helvetica", **kwargs):
        x, y = tuple(point)
        buf = io.BytesIO()
        pdf = canvas.Canvas(
            buf, pagesize=(float(self._page.mediabox.width), self._height)
        )
        pdf.setFont(
            "Helvetica" if fontname in ("helv", "Helvetica") else fontname, fontsize
        )
        for i, line in enumerate(str(text).splitlines()):
            pdf.drawString(x, self._height - y - i * fontsize * 1.2, line)
        pdf.save()
        self._page.merge_page(PdfReader(buf).pages[0])
        self.parent._invalidate()

    def insert_image(self, rect, filename=None, stream=None, **kwargs):
        from reportlab.lib.utils import ImageReader

        r = Rect(rect)
        buf = io.BytesIO()
        pdf = canvas.Canvas(
            buf, pagesize=(float(self._page.mediabox.width), self._height)
        )
        pdf.drawImage(
            ImageReader(io.BytesIO(stream) if stream else str(filename)),
            r.x0,
            self._height - r.y1,
            r.width,
            r.height,
        )
        pdf.save()
        self._page.merge_page(PdfReader(buf).pages[0])
        self.parent._invalidate()

    def get_images(self, **kwargs):
        resources = self._page["/Resources"]
        objects = resources.get("/XObject", {})
        if hasattr(objects, "get_object"):
            objects = objects.get_object()
        return [
            (ref.idnum,)
            for ref in objects.values()
            if ref.get_object().get("/Subtype") == "/Image"
        ]

    def get_fonts(self, full=False):
        fonts = self._page["/Resources"].get("/Font", {})
        if hasattr(fonts, "get_object"):
            fonts = fonts.get_object()
        result = []
        for ref in fonts.values():
            font = ref.get_object()
            xref = ref.idnum
            self.parent._fonts[xref] = font
            result.append((xref, "", "", str(font.get("/BaseFont", ""))))
        return result


class Document:
    def __init__(self, filename=None, *, stream=None, filetype=None):
        self.is_closed = False
        self._fonts = {}
        self._render_cache = None
        self._writer = None
        self._reader = None
        self._authenticated = True
        data = (
            stream
            if stream is not None
            else Path(filename).read_bytes()
            if filename
            else None
        )
        if data is None:
            self._writer = PdfWriter()
        else:
            try:
                self._reader = PdfReader(io.BytesIO(data), strict=False)
                self._authenticated = not self._reader.is_encrypted
            except Exception as exc:
                raise FileDataError("Invalid PDF data") from exc

    def _get_writer(self):
        if self.is_closed:
            raise ValueError("Document is closed")
        if not self._authenticated:
            raise ValueError("PDF requires authentication")
        if self._writer is None:
            self._writer = PdfWriter(clone_from=self._reader)
        return self._writer

    def _invalidate(self):
        self._render_cache = None

    def _render_bytes(self):
        self._get_writer()  # Enforce authentication and closed-document checks.
        if self._render_cache is None:
            self._render_cache = self.tobytes()
        return self._render_cache

    @property
    def is_encrypted(self):
        return not self._authenticated

    @property
    def needs_pass(self):
        return self.is_encrypted

    @property
    def is_pdf(self):
        return True

    def authenticate(self, password):
        if not self._reader or not self._reader.is_encrypted:
            return 1
        try:
            result = self._reader.decrypt(password)
        except Exception:
            return 0
        self._authenticated = bool(result)
        self._invalidate()
        return int(result)

    @property
    def page_count(self):
        return len(self._get_writer().pages)

    def __len__(self):
        return self.page_count

    def __iter__(self):
        return (Page(self, i) for i in range(self.page_count))

    def __getitem__(self, number):
        if number < 0:
            number += self.page_count
        if not 0 <= number < self.page_count:
            raise IndexError(number)
        return Page(self, number)

    load_page = __getitem__

    def new_page(self, width=595, height=842):
        self._get_writer().add_blank_page(width, height)
        self._invalidate()
        return self[self.page_count - 1]

    @property
    def metadata(self):
        meta = self._get_writer().metadata or {}
        names = {"creationDate": "CreationDate", "modDate": "ModDate"}
        return {
            key: str(meta.get("/" + names.get(key, key.title()), ""))
            for key in (
                "title",
                "author",
                "subject",
                "keywords",
                "creator",
                "producer",
                "creationDate",
                "modDate",
            )
        }

    def set_metadata(self, metadata):
        self._invalidate()
        names = {"creationDate": "CreationDate", "modDate": "ModDate"}
        self._get_writer().metadata = None
        self._get_writer().add_metadata(
            {"/" + names.get(k, k.title()): str(v) for k, v in metadata.items()}
        )

    def del_xml_metadata(self):
        self._invalidate()
        self._get_writer()._root_object.pop("/Metadata", None)

    def extract_font(self, xref):
        font = self._fonts[xref]
        descriptors = [font]
        descriptors.extend(ref.get_object() for ref in font.get("/DescendantFonts", []))
        for item in descriptors:
            desc = item.get("/FontDescriptor", {})
            if hasattr(desc, "get_object"):
                desc = desc.get_object()
            for name in ("/FontFile", "/FontFile2", "/FontFile3"):
                if name in desc:
                    return ("", "", "", desc[name].get_data())
        return ("", "", "", b"")

    def tobytes(
        self, *, deflate=False, encryption=None, user_pw="", owner_pw=None, **kwargs
    ):
        writer = self._get_writer()
        if encryption and encryption != PDF_ENCRYPT_NONE:
            writer = PdfWriter(clone_from=PdfReader(io.BytesIO(self.tobytes())))
            writer.encrypt(user_pw, owner_password=owner_pw, algorithm="AES-256")
        if deflate:
            for page in writer.pages:
                page.compress_content_streams()
        if kwargs.get("garbage") or kwargs.get("clean"):
            writer.compress_identical_objects(remove_duplicates=True, remove_unreferenced=True)
        buf = io.BytesIO()
        writer.write(buf)
        return buf.getvalue()

    write = tobytes

    def save(self, filename, **kwargs):
        Path(filename).write_bytes(self.tobytes(**kwargs))

    def close(self):
        self._invalidate()
        if self._writer is not None:
            self._writer.close()
        if self._reader is not None:
            self._reader.close()
        self.is_closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


open = Document
