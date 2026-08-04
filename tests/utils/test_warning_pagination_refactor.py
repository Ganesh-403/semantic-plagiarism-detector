from pathlib import Path


WARNING_LIST_PATH = Path("src/utils/warning_list.py")


def test_warning_list_uses_shared_pagination_helper():
    source = WARNING_LIST_PATH.read_text(encoding="utf-8")

    assert (
        "from src.utils.pagination import "
        "PaginationPage, paginate_items"
    ) in source
    assert "return paginate_items(" in source


def test_warning_list_no_longer_computes_page_bounds():
    source = WARNING_LIST_PATH.read_text(encoding="utf-8")

    assert "math.ceil(total_items / safe_page_size)" not in source
    assert "safe_page = min(max(1, int(page))" not in source


def test_warning_page_is_shared_pagination_type():
    source = WARNING_LIST_PATH.read_text(encoding="utf-8")

    assert (
        "WarningPage = "
        "PaginationPage[dict[str, Any]]"
    ) in source
