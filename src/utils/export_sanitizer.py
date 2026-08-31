"""Shared spreadsheet / CSV formula-injection sanitizer for export modules."""

# Characters that make a spreadsheet application treat a cell as a formula
# rather than as text. Excel, LibreOffice Calc and Google Sheets all evaluate a
# cell whose content starts with one of these.
FORMULA_TRIGGER_PREFIXES = ("=", "+", "-", "@", "\t", "\r")

# Control characters stripped from cell values. Embedded newlines and tabs can
# be used to shift the formula trigger past a naive prefix check.
_CONTROL_CHARACTERS = dict.fromkeys(range(0, 32))


def sanitize_spreadsheet_value(value):
    """Neutralise spreadsheet formula injection in a cell value.

    Document labels come from uploaded filenames, which are user-controlled.
    A file named ``=HYPERLINK("https://attacker.example","Open")`` becomes a
    live formula when the exported report is opened, so any string starting
    with a formula trigger is prefixed with a single quote -- the standard
    spreadsheet escape that forces the cell to be read as text.

    Non-string values (the float similarity scores) are returned unchanged so
    numeric cells keep their type and number formatting.

    Args:
        value: The cell value to sanitize.

    Returns:
        The sanitized value. Strings may gain a leading apostrophe; anything
        that is not a string is returned as-is.

    Examples:
        >>> sanitize_spreadsheet_value("=1+1")
        "'=1+1"
        >>> sanitize_spreadsheet_value("essay.docx")
        'essay.docx'
        >>> sanitize_spreadsheet_value(0.85)
        0.85
    """
    if not isinstance(value, str):
        return value

    # Strip control characters first: a value such as "\r\n=cmd()" would
    # otherwise slip past the prefix check while still being parsed as a
    # formula once the spreadsheet normalises the line endings.
    cleaned = value.translate(_CONTROL_CHARACTERS)

    if cleaned.startswith(FORMULA_TRIGGER_PREFIXES):
        return f"'{cleaned}"

    return cleaned
