import subprocess
import sys
import os
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
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0, f"Child process failed with output: {result.stderr}"
        
        # Verify that the temp file no longer exists on disk after child termination
        target_file = Path("test_temp_cleanup_target.tmp")
        assert not target_file.exists(), "Temporary file was not cleaned up upon process exit."
        
    finally:
        # Cleanup temporary runner script if left behind
        if script_path.exists():
            script_path.unlink()
        
        # Ensure target file is cleaned up if the test failed midway
        target_file = Path("test_temp_cleanup_target.tmp")
        if target_file.exists():
            target_file.unlink()
