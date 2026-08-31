"""
tests/infrastructure/test_ecsoc_automation.py
---------------------------------------------
Unit tests validating GitHub Actions workflow ecsoc-automation.yml (Issue #2819).
"""

import re
from pathlib import Path

WORKFLOW_PATH = Path(".github/workflows/ecsoc-automation.yml")


def test_ecsoc_automation_contains_hidden_claim_comment():
    content = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "<!-- ecsoc-claim-user: ${claimer} -->" in content
    assert r"<!--\s*ecsoc-claim-user:\s*([a-zA-Z0-9-]+)\s*-->" in content


def test_hidden_comment_regex_is_robust_to_greeting_changes():
    pattern = re.compile(r"<!--\s*ecsoc-claim-user:\s*([a-zA-Z0-9-]+)\s*-->")

    # Custom greeting text variations that would break 'Hi @...' matching
    comment_1 = (
        "Hello there contributor! You got it.\n\n<!-- ecsoc-claim-user: alice-dev -->"
    )
    comment_2 = "Greetings! Assigned.\n<!--   ecsoc-claim-user:  bob_123   -->"
    comment_3 = "Welcome @alice-dev to the team!\n<!-- ecsoc-claim-user: charlie99 -->"

    m1 = pattern.search(comment_1)
    assert m1 is not None
    assert m1.group(1) == "alice-dev"

    m3 = pattern.search(comment_3)
    assert m3 is not None
    assert m3.group(1) == "charlie99"


def test_ecsoc_automation_uses_search_api_for_claim_limit():
    """Verify issue-claim limit check uses GitHub Search API (Issue #2793)."""
    content = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "github.rest.search.issuesAndPullRequests" in content
    assert "is:issue is:open assignee:" in content
