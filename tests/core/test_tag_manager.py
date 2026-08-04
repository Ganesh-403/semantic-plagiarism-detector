
from src.core.tag_manager import TagManager


def test_parse_tags_empty():
    assert TagManager.parse_tags("") == ""
    assert TagManager.parse_tags(None) == ""
    assert TagManager.parse_tags("    ") == ""
    assert TagManager.parse_tags(",") == ""
    assert TagManager.parse_tags("#") == ""

def test_parse_tags_normalization():
    # Mixed cases, spaces, commas, missing hashes
    raw = "hw1, Final,   #draft"
    expected = "#draft,#final,#hw1"
    assert TagManager.parse_tags(raw) == expected

def test_parse_tags_removes_invalid_chars():
    # Should strip punctuation that isn't alphanumeric or hash
    raw = "assignment-1!, @hw2, #hw$3"
    expected = "#assignment1,#hw2,#hw3"
    assert TagManager.parse_tags(raw) == expected

def test_parse_tags_deduplication():
    raw = "#hw1, hw1, HW1, #HW1"
    expected = "#hw1"
    assert TagManager.parse_tags(raw) == expected

def test_extract_unique_tags():
    db_column = [
        "#hw1,#draft",
        "",
        None,
        "#final,#hw1",
        "#exam",
        "#draft"
    ]
    expected = ["#draft", "#exam", "#final", "#hw1"]
    assert TagManager.extract_unique_tags(db_column) == expected

def test_extract_unique_tags_empty():
    assert TagManager.extract_unique_tags([]) == []
    assert TagManager.extract_unique_tags([None, "", "  "]) == []

def test_has_matching_tag():
    doc_tags = "#draft,#hw1"

    assert TagManager.has_matching_tag(doc_tags, "#hw1") is True
    assert TagManager.has_matching_tag(doc_tags, "#draft") is True
    assert TagManager.has_matching_tag(doc_tags, "#final") is False

    # "All Tags" or empty should pass
    assert TagManager.has_matching_tag(doc_tags, "All Tags") is True
    assert TagManager.has_matching_tag(doc_tags, "") is True

    # Invalid document tags string
    assert TagManager.has_matching_tag("", "#hw1") is False
    assert TagManager.has_matching_tag(None, "#hw1") is False

def test_has_matching_tag_exact_match():
    # Prevent substring matching issues (e.g. #hw matching #hw1)
    doc_tags = "#hw10,#final"
    assert TagManager.has_matching_tag(doc_tags, "#hw1") is False
    assert TagManager.has_matching_tag(doc_tags, "#hw10") is True
# Additional test cases for extra coverage
def test_extract_unique_tags_with_spaces():
    assert TagManager.extract_unique_tags(['#hw1, #final', ' #hw1 ']) == ['#final', '#hw1']
def test_has_matching_tag_spaces():
    assert TagManager.has_matching_tag(' #hw1 , #final ', '#hw1') is True

from unittest.mock import patch


def test_apply_tag_bulk():
    with patch('src.core.tag_manager.TagManager.parse_tags', return_value="#newtag"), \
         patch('src.db.corpus_db.get_document_tags', side_effect=["#oldtag", "#newtag", ""]), \
         patch('src.db.corpus_db.update_document_tags') as update_mock:
        from src.core.tag_manager import TagManager
        TagManager.apply_tag(["doc1.pdf", "doc2.pdf", "doc3.pdf"], "#newtag")
        assert update_mock.call_count == 2
        update_mock.assert_any_call("doc1.pdf", "#newtag,#oldtag")
        update_mock.assert_any_call("doc3.pdf", "#newtag")


def test_remove_tag_bulk():
    with patch('src.core.tag_manager.TagManager.parse_tags', return_value="#badtag"), \
         patch('src.db.corpus_db.get_document_tags', side_effect=["#badtag,#goodtag", "#goodtag", ""]), \
         patch('src.db.corpus_db.update_document_tags') as update_mock:
        from src.core.tag_manager import TagManager
        TagManager.remove_tag(["doc1.pdf", "doc2.pdf", "doc3.pdf"], "#badtag")
        assert update_mock.call_count == 1
        update_mock.assert_called_once_with("doc1.pdf", "#goodtag")


# ── sanitize_tag_name tests (#936) ───────────────────────────────────────────

import pytest
from src.core.tag_manager import sanitize_tag_name


def test_sanitize_tag_name_removes_html_tags():
    assert sanitize_tag_name("<b>#hw1</b>") == "#hw1"
    assert sanitize_tag_name("<script>alert('xss')</script>#tag") == "alert('xss')#tag"


def test_sanitize_tag_name_removes_slashes_and_whitespace():
    assert sanitize_tag_name(" #class / section \\ test ") == "#classsectiontest"
    assert sanitize_tag_name("tag/name\\1") == "tagname1"


def test_sanitize_tag_name_truncates_to_30_chars():
    long_tag = "a" * 50
    result = sanitize_tag_name(long_tag)
    assert len(result) == 30
    assert result == "a" * 30


def test_sanitize_tag_name_rejects_empty_and_whitespace():
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        sanitize_tag_name("")

    with pytest.raises(ValueError, match="empty or whitespace-only"):
        sanitize_tag_name("   ")

    with pytest.raises(ValueError, match="empty or whitespace-only"):
        sanitize_tag_name("\t\n ")

    with pytest.raises(ValueError, match="empty or whitespace-only"):
        sanitize_tag_name(None)


def test_sanitize_tag_name_rejects_empty_after_sanitization():
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        sanitize_tag_name("<script></script>")

    with pytest.raises(ValueError, match="empty or whitespace-only"):
        sanitize_tag_name(" // \\  ")


def test_tag_manager_static_sanitize_tag_name():
    assert TagManager.sanitize_tag_name("<b>#valid_tag</b>") == "#valid_tag"
