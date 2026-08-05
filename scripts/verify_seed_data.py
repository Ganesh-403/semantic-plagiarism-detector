#!/usr/bin/env python3
"""
scripts/verify_seed_data.py
----------------------------
Verify that the expected seed data (default admin user and sample corpus)
exists in the database files.

Usage:
    python scripts/verify_seed_data.py

Exit codes:
    0 - All seed data verification checks passed
    1 - One or more verification checks failed
"""

import os
import sys

# Add the project root to the path for imports
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.db.auth import init_db, verify_user, get_user_role
from src.db.corpus_db import init_corpus_db, get_all_documents


def verify_admin_user() -> bool:
    """Verify the default admin user exists with correct credentials."""
    print("  Checking default admin user...")
    
    try:
        # Initialize DB to ensure tables exist
        init_db()
        
        # Check if admin user exists
        if not verify_user("admin", "admin123"):
            print("    ✗ FAIL: Default admin user not found or credentials incorrect")
            return False
        
        # Check admin role
        role = get_user_role("admin")
        if role != "admin":
            print(f"    ✗ FAIL: Admin user has incorrect role: {role}")
            return False
        
        print("    ✓ PASS: Default admin user exists with correct credentials and role")
        return True
        
    except Exception as e:
        print(f"    ✗ FAIL: Error checking admin user: {e}")
        return False


def verify_corpus_documents() -> bool:
    """Verify that sample corpus documents exist in the corpus database."""
    print("  Checking corpus documents...")
    
    try:
        # Initialize corpus DB to ensure tables exist
        init_corpus_db()
        
        # Get all documents
        documents = get_all_documents()
        
        if not documents:
            print("    ✗ FAIL: No documents found in corpus database")
            return False
        
        print(f"    ✓ PASS: Found {len(documents)} document(s) in corpus")
        for doc in documents:
            print(f"      - {doc['filename']} (uploaded: {doc['upload_date']})")
        
        return True
        
    except Exception as e:
        print(f"    ✗ FAIL: Error checking corpus documents: {e}")
        return False


def main() -> int:
    """Run all seed data verification checks and return appropriate exit code."""
    print("\n" + "=" * 60)
    print("Seed Data Verification")
    print("=" * 60 + "\n")
    
    results = []
    
    # Verify admin user
    results.append(verify_admin_user())
    
    # Verify corpus documents
    results.append(verify_corpus_documents())
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    if all(results):
        print("\nAll seed data verification checks PASSED ✓\n")
        return 0
    else:
        failed_count = sum(1 for r in results if not r)
        print(f"\n{failed_count} seed data verification check(s) FAILED ✗\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
