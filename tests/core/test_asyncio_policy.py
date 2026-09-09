from unittest.mock import patch


def test_windows_asyncio_policy_not_called_on_linux(monkeypatch):
    from src.utils import os_compat
    monkeypatch.setattr(os_compat, "_PATCHES_APPLIED", False)
    monkeypatch.setattr(os_compat, "get_os_platform", lambda: "linux")
    with patch("asyncio.WindowsSelectorEventLoopPolicy", create=True) as policy:
        assert os_compat.apply_asyncio_patches() is False
        policy.assert_not_called()
