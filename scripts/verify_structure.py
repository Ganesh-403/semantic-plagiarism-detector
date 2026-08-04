#!/usr/bin/env python3
import sys
from pathlib import Path

def main():
    project_root = Path(__file__).resolve().parent.parent

    required_dirs = [
        "src/core",
        "src/db",
        "app",
        "tests",
        "config"
    ]

    required_inits = [
        "src/__init__.py",
        "src/core/__init__.py",
        "src/db/__init__.py",
        # app/ doesn't typically have __init__.py in this repo based on structure
        # but the prompt says: "Verify required __init__.py files exist where appropriate
        # (inside Python packages such as src/, src/core/, src/db/, app/ if applicable)."
    ]

    missing_items = []

    for d in required_dirs:
        dir_path = project_root / d
        if not dir_path.is_dir():
            missing_items.append(f"Directory: {d}")

    for init_file in required_inits:
        file_path = project_root / init_file
        if not file_path.is_file():
            missing_items.append(f"File: {init_file}")

    if missing_items:
        print("Verification failed. The following items are missing:")
        for item in missing_items:
            print(f"  - {item}")
        sys.exit(1)

    print("Project directory structure verified successfully.")
    sys.exit(0)

if __name__ == "__main__":
    main()
