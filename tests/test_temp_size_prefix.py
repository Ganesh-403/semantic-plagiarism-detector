"""Temporary directory accounting exercises actual filesystem traversal."""

from src.utils import temp_manager


def test_recursive_size_uses_configured_temp_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(temp_manager.tempfile, "gettempdir", lambda: str(tmp_path))
    (tmp_path / "spd_file").write_bytes(b"a" * 10)
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "other_file").write_bytes(b"b" * 7)
    assert temp_manager.get_temp_directory_size_bytes() == 17
    (tmp_path / "spd_file").unlink()
    assert temp_manager.get_temp_directory_size_bytes() == 7


def test_missing_temp_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(temp_manager.tempfile, "gettempdir", lambda: str(tmp_path / "missing"))
    assert temp_manager.get_temp_directory_size_bytes() == 0


def test_inaccessible_file_does_not_hide_other_sizes(tmp_path, monkeypatch):
    monkeypatch.setattr(temp_manager.tempfile, "gettempdir", lambda: str(tmp_path))
    blocked = tmp_path / "blocked"
    blocked.write_bytes(b"secret")
    (tmp_path / "visible").write_bytes(b"abc")
    original_stat = temp_manager.os.stat
    def stat(path, **kwargs):
        if str(path) == str(blocked):
            raise PermissionError("unreadable")
        return original_stat(path, **kwargs)
    monkeypatch.setattr(temp_manager.os, "stat", stat)
    assert temp_manager.get_temp_directory_size_bytes() == 3
