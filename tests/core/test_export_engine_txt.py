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

import builtins
import textwrap

import src.core.export_engine as export_engine
from src.core.export_engine import LMSExportEngine


def test_generate_incident_txt_empty_returns_none():
    assert LMSExportEngine.generate_incident_txt([]) is None


def test_generate_incident_txt_formats_flagged_pairs():
    incidents = [
        {
            "doc_a": "student1.pdf",
            "doc_b": "student2.pdf",
            "similarity": 0.95,
        },
        {
            "doc_a": "essay.docx",
            "doc_b": "reference.txt",
            "similarity": 0.82,
        },
    ]

    report = LMSExportEngine.generate_incident_txt(incidents)

    assert report is not None
    assert report.startswith("SEMANTIC PLAGIARISM INCIDENT REPORT")
    assert "Total flagged pairs: 2" in report
    assert "Incident #1" in report
    assert "Document A: student1.pdf" in report
    assert "Document B: student2.pdf" in report
    assert "Similarity: 95.0% (0.9500)" in report
    assert "Severity: High" in report
    assert "Incident #2" in report
    assert "Similarity: 82.0% (0.8200)" in report
    assert "Severity: Medium" in report
    assert report.endswith("End of report\n")


def test_generate_incident_txt_handles_missing_keys():
    report = LMSExportEngine.generate_incident_txt(
        [
            {
                "doc_a": "known.pdf",
            }
        ]
    )

    assert report is not None
    assert "Document A: known.pdf" in report
    assert "Document B: Unknown" in report
    assert "Similarity: 0.0% (0.0000)" in report
    assert "Severity: Low" in report


def test_generate_incident_txt_includes_optional_details():
    report = LMSExportEngine.generate_incident_txt(
        [
            {
                "doc_a": "a.pdf",
                "doc_b": "b.pdf",
                "similarity": 0.91,
                "matched_length": 42,
                "matched_text": "A matching paragraph.",
            }
        ]
    )

    assert report is not None
    assert "Matched length: 42 words" in report
    assert "Matching text:" in report
    assert "A matching paragraph." in report


def test_generate_incident_txt_preserves_unicode():
    report = LMSExportEngine.generate_incident_txt(
        [
            {
                "doc_a": "निबंध.pdf",
                "doc_b": "résumé.txt",
                "similarity": 0.88,
            }
        ]
    )

    assert report is not None
    assert "निबंध.pdf" in report
    assert "résumé.txt" in report


def test_generate_incident_txt_invalid_similarity_returns_none():
    report = LMSExportEngine.generate_incident_txt(
        [
            {
                "doc_a": "a.pdf",
                "doc_b": "b.pdf",
                "similarity": "not-a-number",
            }
        ]
    )

    assert report is None


def test_generate_incident_txt_includes_date_flagged_and_threshold():
    report = LMSExportEngine.generate_incident_txt(
        [
            {
                "doc_a": "a.pdf",
                "doc_b": "b.pdf",
                "similarity": 0.85,
                "date_flagged": "2024-06-01T12:00:00",
                "threshold_at_time_of_flag": 0.59,
            }
        ]
    )

    assert report is not None
    assert "Date flagged: 2024-06-01T12:00:00" in report
    assert "Threshold at time of flag: 0.59" in report


def test_generate_incident_txt_omits_audit_fields_when_absent():
    report = LMSExportEngine.generate_incident_txt(
        [{"doc_a": "a.pdf", "doc_b": "b.pdf", "similarity": 0.75}]
    )

    assert report is not None
    assert "Date flagged:" not in report
    assert "Threshold at time of flag:" not in report


# ---------------------------------------------------------------------------
# Regression tests for issue #3564
#
# generate_incident_txt() wrote its header and the first half of each incident
# block through an io.StringIO buffer, then appended the rest to a list named
# `lines` that was never created. `lines.append("")` sat inside the loop with
# no guard, so the first incident raised NameError -- and because the report
# was read back with buffer.getvalue(), a naive `lines = []` patch would have
# left matching text, date flagged, threshold and the footer silently missing.
#
# The tests below therefore assert on the *shape* of the finished report, not
# just on substring membership, so a report that stops early is a failure even
# when it does not raise.
# ---------------------------------------------------------------------------

FULLY_POPULATED_INCIDENT = {
    "doc_a": "thesis_chapter_3.pdf",
    "doc_b": "source_article.pdf",
    "similarity": 0.9123,
    "matched_length": 137,
    "matched_text": "Semantic drift complicates naive lexical comparison.",
    "date_flagged": "2026-03-14T09:30:00",
    "threshold_at_time_of_flag": 0.72,
}


def test_populated_incident_does_not_raise():
    """The regression itself: one incident with every field must not raise."""
    report = LMSExportEngine.generate_incident_txt([FULLY_POPULATED_INCIDENT])

    assert report is not None


def test_report_carries_every_supplied_field():
    """No field is dropped between the buffer and the returned string."""
    report = LMSExportEngine.generate_incident_txt([FULLY_POPULATED_INCIDENT])

    assert "Document A: thesis_chapter_3.pdf" in report
    assert "Document B: source_article.pdf" in report
    assert "Similarity: 91.2% (0.9123)" in report
    assert "Severity: High" in report
    assert "Matched length: 137 words" in report
    assert "Matching text:" in report
    assert "Semantic drift complicates naive lexical comparison." in report
    assert "Date flagged: 2026-03-14T09:30:00" in report
    assert "Threshold at time of flag: 0.72" in report


def test_report_fields_appear_in_a_stable_order():
    """The block reads top to bottom in the documented order.

    Asserting on order is what catches a report that was assembled through two
    different sinks: with the `lines` splice in place the trailing fields were
    written to a list nobody read, so they could not follow the buffered ones.
    """
    report = LMSExportEngine.generate_incident_txt([FULLY_POPULATED_INCIDENT])

    positions = [
        report.index("Incident #1"),
        report.index("Document A:"),
        report.index("Document B:"),
        report.index("Similarity:"),
        report.index("Severity:"),
        report.index("Matched length:"),
        report.index("Matching text:"),
        report.index("Date flagged:"),
        report.index("Threshold at time of flag:"),
        report.index("End of report"),
    ]

    assert positions == sorted(positions)


def test_footer_follows_the_last_incident():
    """The closing rule and footer are the last thing in the report."""
    report = LMSExportEngine.generate_incident_txt([FULLY_POPULATED_INCIDENT])

    assert report.endswith(f"{'=' * 38}\nEnd of report\n")
    assert report.index("End of report") > report.index("Threshold at time of flag:")


def test_sparse_incident_omits_optional_labels():
    """Fields the incident does not supply leave no empty label behind."""
    report = LMSExportEngine.generate_incident_txt(
        [{"doc_a": "a.pdf", "doc_b": "b.pdf", "similarity": 0.5}]
    )

    assert report is not None
    assert "Matched length:" not in report
    assert "Matching text:" not in report
    assert "Date flagged:" not in report
    assert "Threshold at time of flag:" not in report
    assert "End of report" in report


def test_matching_text_falls_back_to_the_matching_text_key():
    """Rows from the older schema spell the field 'matching_text'."""
    report = LMSExportEngine.generate_incident_txt(
        [
            {
                "doc_a": "a.pdf",
                "doc_b": "b.pdf",
                "similarity": 0.6,
                "matching_text": "legacy key spelling",
            }
        ]
    )

    assert report is not None
    assert "Matching text:" in report
    assert "legacy key spelling" in report


def test_blank_matching_text_is_not_written():
    """Whitespace-only matched text is treated as absent, not as a match."""
    report = LMSExportEngine.generate_incident_txt(
        [
            {
                "doc_a": "a.pdf",
                "doc_b": "b.pdf",
                "similarity": 0.6,
                "matched_text": "   \n  ",
            }
        ]
    )

    assert report is not None
    assert "Matching text:" not in report


def test_every_incident_gets_its_own_complete_block():
    """Multi-incident reports repeat the full block, not just the header half."""
    second = dict(FULLY_POPULATED_INCIDENT)
    second["doc_a"] = "appendix.pdf"
    second["matched_text"] = "A second matching passage."
    second["date_flagged"] = "2026-03-15T11:00:00"

    report = LMSExportEngine.generate_incident_txt([FULLY_POPULATED_INCIDENT, second])

    assert report is not None
    assert "Total flagged pairs: 2" in report
    assert report.count("Incident #") == 2
    assert report.count("Matching text:") == 2
    assert report.count("Date flagged:") == 2
    assert "A second matching passage." in report
    assert report.count("End of report") == 1


def test_incidents_are_separated_by_a_blank_line():
    """The blank line the loop writes between blocks survives."""
    second = {"doc_a": "c.pdf", "doc_b": "d.pdf", "similarity": 0.4}

    report = LMSExportEngine.generate_incident_txt(
        [{"doc_a": "a.pdf", "doc_b": "b.pdf", "similarity": 0.9}, second]
    )

    assert report is not None
    assert "\n\nIncident #2\n" in report


def test_min_match_length_filter_still_applies_to_a_populated_report():
    """Filtering and the restored writes compose, rather than one masking the other."""
    below = dict(FULLY_POPULATED_INCIDENT)
    below["doc_a"] = "short_match.pdf"
    below["matched_length"] = 3

    report = LMSExportEngine.generate_incident_txt(
        [FULLY_POPULATED_INCIDENT, below], min_match_length=50
    )

    assert report is not None
    assert "thesis_chapter_3.pdf" in report
    assert "short_match.pdf" not in report
    assert "Total flagged pairs: 1" in report


def test_generate_incident_txt_body_references_no_undefined_names():
    """Guard the function against another half-applied refactor.

    The `lines` splice passed every import-time check and only failed at run
    time, on a branch the suite did not reach. Compiling the function's source
    and comparing the names it loads against the names it can actually see
    catches that class of edit directly.
    """
    import ast
    import inspect

    source = inspect.getsource(LMSExportEngine.generate_incident_txt)
    tree = ast.parse(textwrap.dedent(source))
    function = tree.body[0]

    assigned = {
        node.id
        for node in ast.walk(function)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    # `except OSError as exception` binds through ExceptHandler.name rather
    # than a Name node, so collect those separately.
    assigned |= {
        node.name
        for node in ast.walk(function)
        if isinstance(node, ast.ExceptHandler) and node.name
    }
    arguments = {
        argument.arg for argument in function.args.args + function.args.kwonlyargs
    }
    loaded = {
        node.id
        for node in ast.walk(function)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }

    module_scope = set(vars(export_engine)) | set(dir(builtins))
    unresolved = loaded - assigned - arguments - module_scope

    assert not unresolved, f"generate_incident_txt reads undefined names: {unresolved}"
