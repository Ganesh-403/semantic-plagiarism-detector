#!/usr/bin/env python3
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

import argparse
import os
import subprocess
import sys
from datetime import datetime

from src.core.logging_setup import setup_logging


def check_dependencies():
    """Ensure pytest and coverage are installed."""
    import importlib.util

    if not (
        importlib.util.find_spec("coverage") and importlib.util.find_spec("pytest")
    ):
        print("Error: Missing dependency.")
        print("Please install requirements: pip install -r requirements.txt")
        sys.exit(1)


def run_tests(args):
    """
    Executes the pytest test suite dynamically based on parsed arguments.
    Enforces coverage thresholds and builds JUnit XML reports.
    """
    cmd = ["pytest"]

    # 1. Scope selection
    if args.unit:
        cmd.extend(["-m", "unit"])
    elif args.integration:
        cmd.extend(["-m", "integration"])

    # 2. Parallel execution
    if getattr(args, "parallel", False):
        cmd.extend(["-n", "auto"])

    # 3. Coverage flags
    if getattr(args, "coverage", False) or args.enforce_coverage:
        cov_cmd = [
            "--cov=src",
            "--cov=app",
            "--cov-report=term-missing",
            "--cov-report=xml",
        ]
        if args.enforce_coverage:
            cov_cmd.extend(
                [
                    f"--cov-fail-under={args.enforce_coverage}",
                    f"--junitxml=test-reports/junit-{datetime.now().strftime('%Y%m%d%H%M%S')}.xml",
                ]
            )
        cmd.extend(cov_cmd)
    else:
        cmd.extend(["--cov=src", "--cov-report=html"])

    # 4. Verbosity
    if args.verbose:
        cmd.append("-vv")

    print(f"Executing Test Runner: {' '.join(cmd)}")

    # 5. Environment isolation
    env = os.environ.copy()
    env["TESTING_MODE"] = "1"

    try:
        result = subprocess.run(cmd, env=env, check=False)
        if result.returncode != 0:
            print(
                f"\n❌ Tests failed or coverage fell below threshold ({args.enforce_coverage}%)."
            )
            sys.exit(result.returncode)
        else:
            print("\n✅ All tests passed successfully.")
    except Exception as e:
        print(f"Failed to execute pytest: {e}")
        sys.exit(1)


def main():
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Automated Test Runner for Semantic Plagiarism Detector"
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--all",
        action="store_true",
        help="Run the entire test suite (unit + integration)",
    )
    group.add_argument(
        "--unit", action="store_true", help="Run only isolated unit tests"
    )
    group.add_argument(
        "--integration", action="store_true", help="Run only integration tests"
    )

    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument(
        "--coverage", action="store_true", help="Enable test coverage reporting"
    )
    parser.add_argument(
        "--enforce-coverage",
        type=int,
        metavar="PERCENT",
        help="Fail the build if code coverage drops below PERCENT",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Increase test output verbosity"
    )

    args = parser.parse_args()

    check_dependencies()
    run_tests(args)


if __name__ == "__main__":
    main()
