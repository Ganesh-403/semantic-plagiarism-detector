from unittest.mock import patch


def test_windows_asyncio_policy_not_called_on_linux():
    """Test that WindowsSelectorEventLoopPolicy is explicitly not invoked when running on Linux."""

    with patch("sys.platform", "linux"):
        with patch(
            "asyncio.WindowsSelectorEventLoopPolicy", create=True
        ) as mock_policy:
            # Trigger your application's startup policy setup or re-import the entry module
            import importlib

            import src.main  # type: ignore

            importlib.reload(src.main)

            mock_policy.assert_not_called()
