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

"""Unit tests for ``app/session_keys.py`` (Issue #2554).

``SessionKeys`` is the single source of truth for every key the Streamlit UI
writes into ``st.session_state``. It had no test module at all, which is how a
bad merge that collapsed the class header into its first member --

    class SessionKeys(str, Enum):    SESSION_ID = "session_id"
    SESSION_ID = "session_id"
        AUTHENTICATED = "authenticated"

-- could sit on ``main`` while making ``app/state_manager.py`` (and therefore
``streamlit run app/streamlit_app.py``) fail to import outright.

These tests cover three things:

* the module parses and imports, and the enum survived the repair intact;
* the ``str`` mixin behaves the way the call sites assume, because that is what
  makes ``st.session_state[SessionKeys.LANG]`` and ``st.session_state["lang"]``
  address the same slot;
* every ``SessionKeys.*`` attribute referenced from ``app/`` actually exists,
  so a member deleted by a future merge fails here instead of at runtime.
"""

from __future__ import annotations

import ast
import pathlib
import re
from enum import Enum

import pytest

from app.session_keys import SessionKeys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "app" / "session_keys.py"


# ---------------------------------------------------------------------------
# Source-level guards: the exact breakage from #2554
# ---------------------------------------------------------------------------


def test_module_source_compiles():
    """The file must parse. This is the assertion that #2554 failed."""
    source = MODULE_PATH.read_text(encoding="utf-8")
    compile(source, "app/session_keys.py", "exec")


def test_module_defines_exactly_one_top_level_class():
    """Guard against a merge leaving a second, shadowing class behind."""
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    classes = [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
    assert classes == [
        "SessionKeys"
    ], f"expected a single SessionKeys class at module level, found {classes}"


def test_no_stray_module_level_key_assignments():
    """The enum members must live in the class, not at module scope.

    The #2554 merge left ``SESSION_ID = "session_id"`` dedented to column 0.
    That assignment is syntactically fine on its own, so only a check like this
    catches it once the indentation damage around it is repaired.
    """
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    stray = [
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id.isupper()
    ]
    assert not stray, f"constants leaked to module scope: {stray}"


def test_module_has_a_docstring():
    import app.session_keys as module

    assert module.__doc__, "app/session_keys.py lost its module docstring"


# ---------------------------------------------------------------------------
# Enum shape
# ---------------------------------------------------------------------------


def test_session_keys_is_a_str_enum():
    assert issubclass(SessionKeys, Enum)
    assert issubclass(SessionKeys, str)


def test_session_id_member_survived():
    """``SESSION_ID`` is the member the bad merge mangled.

    It is also the one member ``init_session_state`` cannot work without: it
    seeds the per-session UUID that every cached-state lookup is keyed on.
    """
    assert SessionKeys.SESSION_ID.value == "session_id"


def test_enum_is_not_empty():
    assert len(list(SessionKeys)) > 30, (
        "SessionKeys lost members - the enum should carry every session-state "
        "key used by the app"
    )


def test_member_values_are_unique():
    """Duplicate values would make two members silently alias each other.

    ``Enum`` folds a duplicate value into an alias rather than raising, so
    ``len(list(SessionKeys))`` alone would not notice.
    """
    values = [member.value for member in SessionKeys]
    duplicates = {value for value in values if values.count(value) > 1}
    assert not duplicates, f"duplicate SessionKeys values: {sorted(duplicates)}"


def test_member_names_are_screaming_snake_case():
    offenders = [
        member.name
        for member in SessionKeys
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", member.name)
    ]
    assert not offenders, f"non-conforming member names: {offenders}"


def test_member_value_matches_lowercased_name():
    """The documented convention, and what makes the mapping predictable."""
    mismatched = {
        member.name: member.value
        for member in SessionKeys
        if member.value != member.name.lower()
    }
    assert not mismatched, f"value does not match name.lower(): {mismatched}"


def test_all_values_are_non_empty_strings():
    for member in SessionKeys:
        assert isinstance(member.value, str)
        assert member.value.strip() == member.value != ""


# ---------------------------------------------------------------------------
# str mixin behaviour relied on by the call sites
# ---------------------------------------------------------------------------


def test_member_equals_its_raw_string():
    """Mixed-in ``str`` equality is what lets old bare-string keys keep working."""
    assert SessionKeys.LANG == "lang"
    assert "lang" == SessionKeys.LANG


def test_member_hashes_like_its_raw_string():
    """Equal-and-same-hash is the property dict lookups depend on.

    ``st.session_state`` is dict-backed, so without this ``state[SessionKeys.LANG]``
    and ``state["lang"]`` would be two different entries.
    """
    state: dict = {}
    state[SessionKeys.LANG] = "en"

    assert state["lang"] == "en"
    assert "lang" in state
    assert SessionKeys.LANG in state
    assert len(state) == 1

    state["lang"] = "fr"
    assert state[SessionKeys.LANG] == "fr"
    assert len(state) == 1, "writing via the raw string created a second entry"


def test_str_returns_the_bare_value():
    """``__str__`` is overridden so f-strings and widget keys stay readable.

    Without the override this would be ``"SessionKeys.SESSION_ID"``, which
    would leak the enum repr into Streamlit widget identifiers and log lines.
    """
    assert str(SessionKeys.SESSION_ID) == "session_id"
    assert f"{SessionKeys.LANG}" == "lang"


def test_members_are_usable_as_widget_keys():
    """Widget ``key=`` values must be distinct plain strings.

    ``app/streamlit_app.py`` passes these members directly as ``key=``.
    """
    widget_keys = [
        SessionKeys.LANG_SELECTOR,
        SessionKeys.THRESHOLD_SLIDER,
        SessionKeys.LEXICAL_THRESHOLD_SLIDER,
        SessionKeys.SEMANTIC_THRESHOLD_SLIDER,
        SessionKeys.CHUNK_MATRIX_CHECKBOX,
        SessionKeys.FAISS_TOP_K_SLIDER,
        SessionKeys.CHUNK_SIZE_SLIDER,
        SessionKeys.CHUNK_OVERLAP_SLIDER,
        SessionKeys.OCR_LANGUAGE_SELECTOR,
        SessionKeys.OCR_DPI_SLIDER,
        SessionKeys.CLASS_FILTER_SELECTBOX,
    ]
    rendered = [str(key) for key in widget_keys]

    assert len(set(rendered)) == len(rendered), "widget keys collide"
    assert all(key.isidentifier() for key in rendered)


def test_lookup_by_value_round_trips():
    for member in SessionKeys:
        assert SessionKeys(member.value) is member


def test_lookup_by_name_round_trips():
    for member in SessionKeys:
        assert SessionKeys[member.name] is member


def test_unknown_value_raises():
    with pytest.raises(ValueError):
        SessionKeys("definitely_not_a_session_key")


# ---------------------------------------------------------------------------
# Members that specific call sites require
# ---------------------------------------------------------------------------


# Keys read or written by ``app/state_manager.py``. If one of these disappears,
# session bootstrap or the inactivity timeout breaks at runtime.
STATE_MANAGER_KEYS = (
    "SESSION_ID",
    "AUTHENTICATED",
    "USERNAME",
    "ROLE",
    "PDF_PASSWORDS",
    "LANG",
    "SESSION_START_TIME",
    "MODEL_LOAD_TIME",
    "LAST_INTERACTION",
)


@pytest.mark.parametrize("name", STATE_MANAGER_KEYS)
def test_state_manager_keys_exist(name: str):
    assert hasattr(
        SessionKeys, name
    ), f"app/state_manager.py references SessionKeys.{name}, which is missing"


def test_every_referenced_member_exists():
    """Scan ``app/`` for ``SessionKeys.<NAME>`` and check each one resolves.

    This is the broad net: it fails when a merge drops a member that is still
    referenced somewhere in the UI, without anyone having to remember to update
    the explicit list above.
    """
    pattern = re.compile(r"\bSessionKeys\.([A-Z][A-Z0-9_]*)\b")
    missing: dict[str, set[str]] = {}

    for path in sorted((REPO_ROOT / "app").rglob("*.py")):
        if "__pycache__" in path.parts or path == MODULE_PATH:
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        unknown = {
            name for name in pattern.findall(source) if not hasattr(SessionKeys, name)
        }
        if unknown:
            missing[path.relative_to(REPO_ROOT).as_posix()] = unknown

    assert not missing, "references to undefined SessionKeys members:\n" + "\n".join(
        f"  - {path}: {sorted(names)}" for path, names in sorted(missing.items())
    )
