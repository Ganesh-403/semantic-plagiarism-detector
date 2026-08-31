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

"""
src/security/code_sandbox.py
----------------------------
Secure Code Execution Sandboxing.

Executes submitted Python code in an isolated, resource-restricted subprocess
with strict timeouts and memory limits. Captures standard I/O and exception
traces to generate a behavioral fingerprint for code plagiarism detection.
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default resource limits
DEFAULT_TIMEOUT_SECONDS = 5.0
DEFAULT_MAX_MEMORY_MB = 256


def execute_code_sandbox(
    code: str,
    test_cases: Optional[list[dict[str, Any]]] = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_memory_mb: int = DEFAULT_MAX_MEMORY_MB,
) -> dict[str, Any]:
    """Execute Python code in a restricted subprocess and capture traces.

    Args:
        code: The Python source code to execute.
        test_cases: Optional list of test cases to run against the code.
                    Each test case should be a dict with 'input' and 'expected_output'.
        timeout: Maximum execution time in seconds before killing the process.
        max_memory_mb: Maximum memory usage in MB (enforced via ulimit on Unix).

    Returns:
        Dictionary containing stdout, stderr, return_code, execution_time,
        and memory_limit_exceeded flag.
    """
    if not code or not isinstance(code, str):
        return {
            "stdout": "",
            "stderr": "Error: Empty or invalid code submitted.",
            "return_code": -1,
            "execution_time": 0.0,
            "memory_limit_exceeded": False,
            "test_results": [],
        }

    # Create a temporary file for the code to avoid shell injection issues
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        code_file_path = f.name

    # Create a temporary file for the test runner script
    test_runner_code = _generate_test_runner(code_file_path, test_cases)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(test_runner_code)
        runner_file_path = f.name

    result = {
        "stdout": "",
        "stderr": "",
        "return_code": -1,
        "execution_time": 0.0,
        "memory_limit_exceeded": False,
        "test_results": [],
    }

    try:
        # Build the command
        # Use ulimit to restrict memory on Unix-like systems
        # Note: ulimit -v restricts virtual memory in KB
        max_memory_kb = max_memory_mb * 1024

        if sys.platform != "win32":
            # Prepend ulimit to the command
            cmd = f"ulimit -v {max_memory_kb}; {sys.executable} {runner_file_path}"
            shell = True
        else:
            # Windows doesn't support ulimit natively in cmd.
            # We rely on timeout and job objects, but for simplicity in this cross-platform
            # sandbox, we just use timeout and skip strict memory limits on Windows.
            cmd = [sys.executable, runner_file_path]
            shell = False

        start_time = time.perf_counter()

        # Run the subprocess
        proc = subprocess.run(
            cmd,
            shell=shell,  # nosec
            capture_output=True,
            text=True,
            timeout=timeout,
            executable="/bin/bash" if sys.platform != "win32" and shell else None,
        )

        end_time = time.perf_counter()
        result["execution_time"] = round(end_time - start_time, 4)
        result["stdout"] = proc.stdout
        result["stderr"] = proc.stderr
        result["return_code"] = proc.returncode

        # Check for memory limit exceeded (usually returns 137 or specific stderr on Linux)
        if (
            "MemoryError" in proc.stderr
            or "Killed" in proc.stderr
            or proc.returncode == 137
        ):
            result["memory_limit_exceeded"] = True

        # Parse test results from stdout (assuming the runner outputs JSON at the end)
        try:
            # The runner script prints a JSON string prefixed with "TEST_RESULTS:"
            for line in proc.stdout.split("\n"):
                if line.startswith("TEST_RESULTS:"):
                    json_str = line.replace("TEST_RESULTS:", "").strip()
                    result["test_results"] = json.loads(json_str)
                    break
        except Exception as e:
            logger.debug("Failed to parse test results JSON: %s", e)

    except subprocess.TimeoutExpired:
        result["stderr"] = f"Error: Execution timed out after {timeout} seconds."
        result["return_code"] = -1
        result["execution_time"] = timeout
    except Exception as e:
        result["stderr"] = f"Error: Sandbox execution failed: {str(e)}"
        result["return_code"] = -1
    finally:
        # Cleanup temporary files
        try:
            os.unlink(code_file_path)
            os.unlink(runner_file_path)
        except OSError:
            pass

    return result


def _generate_test_runner(
    code_file_path: str, test_cases: Optional[list[dict[str, Any]]]
) -> str:
    """Generate a Python script that imports the submitted code and runs test cases."""
    # Normalize path for Windows/Unix
    safe_path = code_file_path.replace("\\", "\\\\")

    runner_script = f"""
import sys
import json
import traceback
import importlib.util

# Load the submitted code as a module
spec = importlib.util.spec_from_file_location("submitted_code", "{safe_path}")
module = importlib.util.module_from_spec(spec)

test_results = []

try:
    spec.loader.exec_module(module)

    # If test cases are provided, run them
    test_cases = {json.dumps(test_cases) if test_cases else "[]"}

    for i, tc in enumerate(test_cases):
        test_input = tc.get("input")
        expected_output = tc.get("expected_output")

        # Assume the submitted code has a main function called 'solution' or 'main'
        func = getattr(module, "solution", None) or getattr(module, "main", None)

        if func is None:
            test_results.append({{"test_id": i, "status": "error", "message": "No 'solution' or 'main' function found."}})
            continue

        try:
            # Execute the function with the test input
            if isinstance(test_input, list):
                actual_output = func(*test_input)
            elif isinstance(test_input, dict):
                actual_output = func(**test_input)
            else:
                actual_output = func(test_input) if test_input is not None else func()

            if actual_output == expected_output:
                test_results.append({{"test_id": i, "status": "passed"}})
            else:
                test_results.append({{
                    "test_id": i,
                    "status": "failed",
                    "expected": expected_output,
                    "actual": actual_output
                }})
        except Exception as e:
            test_results.append({{
                "test_id": i,
                "status": "error",
                "message": str(e),
                "traceback": traceback.format_exc()
            }})

except Exception as e:
    print(f"Error loading submitted code: {{e}}", file=sys.stderr)
    print(traceback.format_exc(), file=sys.stderr)

# Output the test results as JSON for the sandbox to parse
print("TEST_RESULTS:" + json.dumps(test_results))
"""
    return runner_script
