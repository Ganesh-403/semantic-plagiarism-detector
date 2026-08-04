import src.db.corpus_db as corpus_db


def test_add_document_sanitizes_filename_before_storage(
    monkeypatch,
    tmp_path,
):
    database = tmp_path / "corpus.db"
    monkeypatch.setattr(corpus_db, "_DB_PATH", str(database))

    corpus_db.init_corpus_db()
    inserted = corpus_db.add_document(
        "<script>alert(1)</script>.pdf",
        "security-hash",
    )

    assert inserted is True
    documents = corpus_db.get_all_documents()
    assert documents[0]["filename"] == "alert_1.pdf"
