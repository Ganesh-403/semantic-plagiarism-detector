def get_user_count() -> int:
    """Returns the total number of registered users in the system."""
    with _connect() as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM users")
        row = cursor.fetchone()
        return row[0] if row else 0


def format_user_created_date(iso_str: str) -> str:
    """Format an ISO date string as a human-readable date (e.g. "Jul 28, 2026").

    Acceptance criteria (issue #1049):
        - Parse ISO string and return formatted date string.
        - Handle empty/invalid inputs gracefully.

    Args:
        iso_str: An ISO 8601 date/datetime string (e.g.
            ``"2026-07-28T14:30:00Z"``, ``"2026-07-28"``,
            ``"2026-07-28 14:30:00"``).

    Returns:
        A formatted date string like ``"Jul 28, 2026"`` on success,
        or ``"Unknown"`` if the input is empty, ``None``, or cannot be
        parsed.
    """
    if not iso_str or not isinstance(iso_str, str):
        return "Unknown"

    iso_str = iso_str.strip()
    if not iso_str:
        return "Unknown"

    # Try dateutil.parser first — it handles virtually any ISO format.
    try:
        from dateutil import parser as dateutil_parser

        dt = dateutil_parser.parse(iso_str)
        return dt.strftime("%b %d, %Y")
    except Exception:
        pass

    # Fallback: try Python's datetime.fromisoformat (3.7+).
    # Strip trailing 'Z' which fromisoformat doesn't accept in 3.9–3.10.
    cleaned = iso_str.rstrip("Z")
    for parser_fn in (
        datetime.datetime.fromisoformat,
        lambda s: datetime.datetime.strptime(s, "%Y-%m-%d"),
        lambda s: datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S"),
        lambda s: datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S"),
    ):
        try:
            dt = parser_fn(cleaned)
            return dt.strftime("%b %d, %Y")
        except Exception:
            continue

    return "Unknown"
