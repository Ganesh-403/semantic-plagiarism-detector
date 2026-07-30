#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from datetime import datetime


def check_dependencies():
    """Ensure pytest and coverage are installed."""
    import importlib.util
    if not (importlib.util.find_spec("coverage") and importlib.util.find_spec("pytest")):
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
        
    # 2. Coverage flags
    if args.enforce_coverage:
        cmd.extend([
            "--cov=src",
            "--cov=app",
            "--cov-report=term-missing",
            f"--cov-fail-under={args.enforce_coverage}",
            f"--junitxml=test-reports/junit-{datetime.now().strftime('%Y%m%d%H%M%S')}.xml"
        ])
    else:
        cmd.extend(["--cov=src", "--cov-report=html"])
        
    # 3. Verbosity
    if args.verbose:
        cmd.append("-vv")
        
    print(f"Executing Test Runner: {' '.join(cmd)}")
    
    # 4. Environment isolation
    env = os.environ.copy()
    env["TESTING_MODE"] = "1"
    
    try:
        result = subprocess.run(cmd, env=env, check=False)
        if result.returncode != 0:
            print(f"\\n❌ Tests failed or coverage fell below threshold ({args.enforce_coverage}%).")
            sys.exit(result.returncode)
        else:
            print("\\n✅ All tests passed successfully.")
    except Exception as e:
        print(f"Failed to execute pytest: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Automated Test Runner for Semantic Plagiarism Detector")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Run the entire test suite (unit + integration)")
    group.add_argument("--unit", action="store_true", help="Run only isolated unit tests")
    group.add_argument("--integration", action="store_true", help="Run only integration tests")
    
    parser.add_argument("--enforce-coverage", type=int, metavar="PERCENT", 
                        help="Fail the build if code coverage drops below PERCENT")
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase test output verbosity")
    
    args = parser.parse_args()
    
    check_dependencies()
    run_tests(args)

if __name__ == "__main__":
    main()
