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

import os
import subprocess
import sys
import tempfile
from pathlib import Path


def test_temp_manager_atexit_cleanup():
    """Integration test using subprocess to verify that `atexit` cleanup removes registered temp files on normal exit."""

    # Create a temporary script that registers a temp file with temp_manager
    script_content = """
import sys
import os
from pathlib import Path

# Add project root to path if needed
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.temp_manager import register_temp_path

# Create an actual temporary file to track
temp_file = Path("test_temp_cleanup_target.tmp")
temp_file.write_text("temporary data for atexit test")

# Register the temp path for cleanup on exit
register_temp_path(temp_file)

# Exit normally to trigger atexit handlers
sys.exit(0)
"""

    script_path = Path("temp_atexit_runner.py")
    script_path.write_text(script_content)

    try:
        # Run the temporary script in a separate subprocess
        result = subprocess.run(
            [sys.executable, str(script_path)], capture_output=True, text=True
        )

        assert (
            result.returncode == 0
        ), f"Child process failed with output: {result.stderr}"

        # Verify that the temp file no longer exists on disk after child termination
        target_file = Path("test_temp_cleanup_target.tmp")
        assert (
            not target_file.exists()
        ), "Temporary file was not cleaned up upon process exit."

    finally:
        # Cleanup temporary runner script if left behind
        if script_path.exists():
            script_path.unlink()

        # Ensure target file is cleaned up if the test failed midway
        target_file = Path("test_temp_cleanup_target.tmp")
        if target_file.exists():
            target_file.unlink()
