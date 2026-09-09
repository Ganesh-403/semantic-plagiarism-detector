import subprocess
import sys
from pathlib import Path


def test_temp_manager_atexit_cleanup(tmp_path):
    target = tmp_path / "cleanup-target.tmp"
    script = """
import sys
from pathlib import Path
from src.utils.temp_manager import register_temp_path
path = Path(sys.argv[1])
path.write_text("temporary data", encoding="utf-8")
register_temp_path(str(path))
assert path.exists()
"""
    result = subprocess.run([sys.executable, "-c", script, str(target)],
                            cwd=Path(__file__).resolve().parents[2],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert not target.exists()
