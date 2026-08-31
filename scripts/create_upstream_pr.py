#!/usr/bin/env python3
"""
scripts/create_upstream_pr.py
Automates creation of an issue and a linked pull request to the main upstream repository
via the GitHub REST API securely using environment variables.
"""

import os
import sys
import httpx
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Configuration
UPSTREAM_OWNER = "Ganesh-403"
REPO_NAME = "semantic-plagiarism-detector"
HEAD_OWNER = "karan-chaos"
BRANCH_NAME = "feature/test-file-streaming"
BASE_BRANCH = "main"

ISSUE_TITLE = "Unit tests for stream_upload_file_to_disk chunked streaming"
ISSUE_BODY = "Implements automated unit tests verifying chunking and exception cleanup for `src/utils/file_streaming.py` as requested in issue #3245."
PR_TITLE = "Add comprehensive unit tests for stream_upload_file_to_disk"

def main():
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        logging.error("GITHUB_TOKEN environment variable is not set. Please set it before running this script.")
        sys.exit(1)

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }

    base_url = f"https://api.github.com/repos/{UPSTREAM_OWNER}/{REPO_NAME}"
    
    with httpx.Client(headers=headers) as client:
        # 1. Create Issue
        logging.info(f"Creating issue in {UPSTREAM_OWNER}/{REPO_NAME}...")
        issue_resp = client.post(
            f"{base_url}/issues",
            json={"title": ISSUE_TITLE, "body": ISSUE_BODY}
        )
        
        if issue_resp.status_code != 201:
            logging.error(f"Failed to create issue: {issue_resp.status_code} {issue_resp.text}")
            sys.exit(1)
            
        issue_data = issue_resp.json()
        issue_number = issue_data["number"]
        issue_url = issue_data["html_url"]
        logging.info(f"Successfully created Issue #{issue_number}: {issue_url}")
        
        # 2. Create Pull Request
        pr_body = f"Resolves #{issue_number}\n\nThis PR thoroughly implements all requirement constraints including 700+ line test limits, disk IO tests, max limits, chunk sizes, and API error simulations."
        head_ref = f"{HEAD_OWNER}:{BRANCH_NAME}"
        
        logging.info(f"Creating pull request from {head_ref} to {BASE_BRANCH}...")
        pr_resp = client.post(
            f"{base_url}/pulls",
            json={
                "title": PR_TITLE,
                "body": pr_body,
                "head": head_ref,
                "base": BASE_BRANCH,
                "maintainer_can_modify": True
            }
        )
        
        if pr_resp.status_code != 201:
            logging.error(f"Failed to create pull request: {pr_resp.status_code} {pr_resp.text}")
            sys.exit(1)
            
        pr_url = pr_resp.json()["html_url"]
        logging.info(f"Successfully created Pull Request: {pr_url}")

if __name__ == "__main__":
    main()
