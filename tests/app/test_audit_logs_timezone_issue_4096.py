"""
tests/app/test_audit_logs_timezone_issue_4096.py
-------------------------------------------------
Regression tests for the missing ``timezone`` import in
``app/pages/3_Audit_Logs.py`` (issue #4096).

The CSV export stamps its filename in UTC:

    file_name=f"audit_logs_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"

but line 12 read ``from datetime import datetime`` and nothing else. ``timezone``
was never bound, so the f-string raised ``NameError``.

The important detail is *when*. The f-string is evaluated while
``st.download_button(...)`` is being constructed, not when a user clicks it, so
the failure took down the whole audit-log view as soon as the export block was
reached. The page only survived when there were no logs to show and the
``else`` branch was taken -- which is why nobody hit it.

Audit-log export is compliance-facing, so the tests below exercise the render
path with rows present rather than only asserting on the import line.
"""

import ast
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PAGE_PATH = Path(__file__).resolve().parents[2] / "app" / "pages" / "3_Audit_Logs.py"

STUBBED_MODULES = (
    "streamlit",
    "src",
    "src.db",
    "src.db.auth",
    "src.db.security_audit",
)

# Top-level statements worth executing. The rest of this page is rendering:
# st.set_page_config() near the top and a `if __name__ == "__main__"` block at
# the bottom.
LOADABLE_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Assign,
    ast.AnnAssign,
)

EXPORT_FILENAME_RE = re.compile(r"^audit_logs_(\d{8}_\d{6})\.csv$")


class SessionState(dict):
    """Stand-in for ``st.session_state``: dict access plus attribute access."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:  # pragma: no cover - mirrors streamlit's error
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


class FakeStreamlit:
    """Minimal streamlit double that records the calls these tests read.

    Only the surface ``render_audit_logs()`` actually touches is implemented;
    anything else falls through to a MagicMock attribute.
    """

    def __init__(self, role="admin", button_returns=False):
        self.session_state = SessionState(role=role)
        self.download_button_calls = []
        self.error_calls = []
        self.info_calls = []
        self._button_returns = button_returns

    def columns(self, spec):
        count = spec if isinstance(spec, int) else len(spec)
        return [MagicMock() for _ in range(count)]

    def selectbox(self, *_args, **_kwargs):
        return "All Event Types"

    def text_input(self, *_args, **_kwargs):
        return ""

    def button(self, *_args, **_kwargs):
        return self._button_returns

    def download_button(self, label, **kwargs):
        self.download_button_calls.append({"label": label, **kwargs})

    def error(self, message):
        self.error_calls.append(message)

    def info(self, message):
        self.info_calls.append(message)

    def __getattr__(self, _name):
        return MagicMock()


@pytest.fixture(scope="module")
def page_source():
    return PAGE_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def page_tree(page_source):
    return ast.parse(page_source, filename=PAGE_PATH.name)


@pytest.fixture(scope="module")
def page_namespace(page_tree):
    """Execute the page's imports and defs, but not its rendering statements."""
    saved = {name: sys.modules.get(name) for name in STUBBED_MODULES}
    for name in STUBBED_MODULES:
        sys.modules[name] = MagicMock()

    try:
        body = [node for node in page_tree.body if isinstance(node, LOADABLE_NODES)]
        namespace = {"__name__": "audit_logs_isolated"}
        exec(  # noqa: S102 - deliberately loading a page module without running it
            compile(ast.Module(body=body, type_ignores=[]), PAGE_PATH.name, "exec"),
            namespace,
        )
        yield namespace
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


AUDIT_ROWS = [
    {
        "id": 1,
        "timestamp": "2026-08-29 10:00:00",
        "event_type": "login",
        "username": "admin",
        "details": "Admin login successful",
    },
    {
        "id": 2,
        "timestamp": "2026-08-29 10:05:00",
        "event_type": "file_upload",
        "username": "teacher1",
        "details": "Uploaded assignment.pdf",
    },
]


def render(page_namespace, rows=AUDIT_ROWS, role="admin"):
    """Run ``render_audit_logs()`` against doubles; return the fake streamlit."""
    fake_st = FakeStreamlit(role=role)
    page_namespace["st"] = fake_st
    page_namespace["get_distinct_audit_event_types"] = lambda: ["login", "file_upload"]
    page_namespace["get_recent_audit_events"] = lambda **_kwargs: rows
    page_namespace["get_audit_events_count"] = lambda **_kwargs: len(rows)
    page_namespace["render_audit_logs"]()
    return fake_st


# ── the import itself ──────────────────────────────────────────────────────────


def test_timezone_is_imported_from_datetime(page_tree):
    """``timezone`` must be pulled in beside ``datetime``.

    Asserted on the AST rather than by substring so an unrelated mention of the
    word elsewhere in the file cannot satisfy it.
    """
    imported = set()
    for node in ast.walk(page_tree):
        if isinstance(node, ast.ImportFrom) and node.module == "datetime":
            imported.update(alias.name for alias in node.names)

    assert "timezone" in imported, "from datetime import datetime, timezone"
    assert "datetime" in imported


def test_every_name_the_page_uses_is_bound(page_namespace):
    """``timezone`` resolves in the page's namespace after import."""
    assert page_namespace["timezone"] is timezone
    assert page_namespace["datetime"] is datetime


# ── the render path that used to raise ─────────────────────────────────────────


def test_render_with_rows_does_not_raise(page_namespace):
    """The export block must be reachable.

    Before the fix this raised ``NameError: name 'timezone' is not defined``
    partway through rendering, after the table had already been drawn.
    """
    fake_st = render(page_namespace)
    assert fake_st.error_calls == []


def test_export_button_is_offered_when_rows_exist(page_namespace):
    fake_st = render(page_namespace)
    assert len(fake_st.download_button_calls) == 1


def test_export_filename_matches_the_expected_shape(page_namespace):
    """``audit_logs_YYYYmmdd_HHMMSS.csv``."""
    fake_st = render(page_namespace)
    filename = fake_st.download_button_calls[0]["file_name"]
    assert EXPORT_FILENAME_RE.match(filename), filename


def test_export_filename_is_stamped_in_utc(page_namespace):
    """The stamp must track UTC, not the host's local clock.

    This is the behaviour the missing import was hiding. Comparing against a
    naive ``datetime.now()`` would pass on a UTC machine and fail everywhere
    else, so the assertion is against ``datetime.now(timezone.utc)`` with a
    generous window.
    """
    before = datetime.now(timezone.utc)
    fake_st = render(page_namespace)
    after = datetime.now(timezone.utc)

    stamp = EXPORT_FILENAME_RE.match(
        fake_st.download_button_calls[0]["file_name"]
    ).group(1)
    parsed = datetime.strptime(stamp, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)

    assert before.replace(microsecond=0) <= parsed <= after + (after - before)


def test_export_carries_csv_metadata(page_namespace):
    """The mime type and payload survived alongside the filename."""
    call = render(page_namespace).download_button_calls[0]
    assert call["mime"] == "text/csv"
    assert isinstance(call["data"], bytes)


def test_export_payload_contains_the_audit_rows(page_namespace):
    """The CSV is built from the fetched rows, not an empty frame."""
    call = render(page_namespace).download_button_calls[0]
    payload = call["data"].decode("utf-8")
    assert "Admin login successful" in payload
    assert "teacher1" in payload


# ── the branches that already worked must keep working ─────────────────────────


def test_no_export_button_when_there_are_no_rows(page_namespace):
    """The empty branch is the one that used to mask the bug; it still holds."""
    fake_st = render(page_namespace, rows=[])
    assert fake_st.download_button_calls == []
    assert fake_st.info_calls


def test_non_admin_is_refused_before_anything_is_fetched(page_namespace):
    """Access control still short-circuits ahead of the export block."""
    fake_st = render(page_namespace, role="teacher")
    assert fake_st.error_calls
    assert "Access Denied" in fake_st.error_calls[0]
    assert fake_st.download_button_calls == []
