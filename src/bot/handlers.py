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

async def handle_claim_force_command(
    comment_body: str, comment_author: str, issue_number: int, repo_client
) -> bool:
    """
    Handles the /claim-force @username maintainer command to bypass limits and assign an issue.
    """
    if not comment_body.strip().startswith("/claim-force"):
        return False

    # Check if the author is a maintainer/collaborator (omitted or mocked based on your auth check)
    if not await repo_client.is_maintainer(comment_author):
        print(f"Unauthorized /claim-force attempt by non-maintainer {comment_author}")
        return False

    # Extract target username from command (e.g. /claim-force @johndoe or /claim-force johndoe)
    parts = comment_body.split()
    if len(parts) < 2:
        print("Missing target username in /claim-force command.")
        return False

    target_user = parts[1].lstrip("@")

    try:
        # Bypass limits and forcefully assign the issue
        await repo_client.assign_issue(issue_number, target_user)
        await repo_client.add_comment(
            issue_number,
            f"✅ Issue successfully assigned to @{target_user} via maintainer manual override (`/claim-force`).",
        )
        print(
            f"Successfully force-assigned issue #{issue_number} to {target_user} by {comment_author}."
        )
        return True
    except Exception as e:
        print(f"Failed to execute /claim-force for issue #{issue_number}: {e}")
        return False
