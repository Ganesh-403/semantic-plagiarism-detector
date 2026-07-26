from src.utils.warning_list import (filter_warnings, paginate_warnings,
                                    prepare_warning_page, sort_warnings)

WARNINGS = [
    {"doc_a": "Zeta.pdf", "doc_b": "Alpha.pdf", "similarity": 0.91, "severity": "High"},
    {
        "doc_a": "Beta.pdf",
        "doc_b": "Gamma.pdf",
        "similarity": 0.78,
        "severity": "Medium",
    },
    {
        "doc_a": "Alpha.pdf",
        "doc_b": "Delta.pdf",
        "similarity": 0.91,
        "severity": "High",
    },
    {
        "doc_a": "Notes.pdf",
        "doc_b": "Essay.pdf",
        "similarity": 0.81,
        "severity": "Medium",
    },
]


def test_search_matches_either_document_case_insensitively():
    results = filter_warnings(WARNINGS, "ALPHA")
    assert len(results) == 2


def test_empty_search_returns_everything():
    assert len(filter_warnings(WARNINGS, " ")) == 4


def test_fuzzy_search_handles_minor_typos():
    # "Alpaha" is a typo for "Alpha"
    results = filter_warnings(WARNINGS, "Alpaha")
    assert len(results) == 2

    # "Ztaa" is a typo for "Zeta"
    results_zeta = filter_warnings(WARNINGS, "Ztaa")
    assert len(results_zeta) == 1
    assert results_zeta[0]["doc_a"] == "Zeta.pdf"


def test_multi_column_sorting():
    results = sort_warnings(
        WARNINGS,
        primary_field="similarity",
        primary_descending=True,
        secondary_field="doc_a",
        secondary_descending=False,
    )
    assert [item["similarity"] for item in results] == [0.91, 0.91, 0.81, 0.78]
    assert results[0]["doc_a"] == "Alpha.pdf"
    assert results[1]["doc_a"] == "Zeta.pdf"


def test_filename_sorting():
    results = sort_warnings(
        WARNINGS,
        primary_field="doc_a",
        primary_descending=False,
    )
    assert [item["doc_a"] for item in results] == [
        "Alpha.pdf",
        "Beta.pdf",
        "Notes.pdf",
        "Zeta.pdf",
    ]


def test_pagination_and_page_clamping():
    warnings = [
        {
            "doc_a": f"A-{i}.pdf",
            "doc_b": f"B-{i}.pdf",
            "similarity": 0.8,
            "severity": "Medium",
        }
        for i in range(23)
    ]
    page_two = paginate_warnings(warnings, page=2, page_size=10)
    final_page = paginate_warnings(warnings, page=99, page_size=10)

    assert len(page_two.items) == 10
    assert page_two.start_index == 11
    assert page_two.end_index == 20
    assert final_page.page == 3
    assert len(final_page.items) == 3


def test_filtering_occurs_before_pagination():
    warnings = [
        {
            "doc_a": f"target-{i}.pdf" if i < 12 else f"other-{i}.pdf",
            "doc_b": "reference.pdf",
            "similarity": 0.7 + i / 100,
            "severity": "Medium",
        }
        for i in range(20)
    ]

    filtered, page = prepare_warning_page(
        warnings,
        search_query="target",
        page=2,
        page_size=10,
    )
    assert len(filtered) == 12
    assert len(page.items) == 2
    assert page.total_pages == 2


def test_filter_warnings_by_minimum_match_length():
    warnings = [
        {
            "doc_a": "doc1.pdf",
            "doc_b": "doc2.pdf",
            "similarity": 0.8,
            "severity": "Medium",
            "matched_length": 5,
        },
        {
            "doc_a": "doc1.pdf",
            "doc_b": "doc3.pdf",
            "similarity": 0.85,
            "severity": "High",
            "matched_length": 150,
        },
        {
            "doc_a": "doc2.pdf",
            "doc_b": "doc3.pdf",
            "similarity": 0.75,
            "severity": "Medium",
            "matched_length": 50,
        },
    ]

    # Filter with min_match_length = 50 -> should exclude the 5-word match
    filtered = filter_warnings(warnings, min_match_length=50)
    assert len(filtered) == 2
    assert all(item["matched_length"] >= 50 for item in filtered)

    # Filter with min_match_length = 200 -> should exclude all matches
    filtered_none = filter_warnings(warnings, min_match_length=200)
    assert len(filtered_none) == 0

    # Filter routing in prepare_warning_page
    sorted_items, page = prepare_warning_page(warnings, min_match_length=50)
    assert len(sorted_items) == 2
    assert page.total_items == 2
