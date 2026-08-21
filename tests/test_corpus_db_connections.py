import threading

from src.db.corpus_db import _all_connections, close_connections, get_connection


def test_close_all_connections_across_threads(tmp_path):
    db_file = str(tmp_path / "test.db")

    def worker():
        conn = get_connection(db_file)
        conn.execute("SELECT 1")

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Connections were created in sub-threads
    assert len(_all_connections) == 2

    # Global close from main thread
    close_connections(all_threads=True)

    assert len(_all_connections) == 0
